import logging
import numpy as np
import warnings
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.color import label2rgb 
import cv2
import openslide
import os
import traceback
from skimage.util import img_as_ubyte, img_as_float
from skimage.transform import resize, rescale
from skimage.measure import regionprops, label
import sys
import numpy as np
import logging
import visualization
from stain_utils import get_dab_mask

logger = logging.getLogger(__name__)

def _get_mask_centroid(mask: np.ndarray):
    """Helper to find the centroid of the largest component in a binary mask."""
    if mask is None or not np.any(mask):
        return None
    labeled_mask = label(mask.astype(np.uint8), connectivity=2)
    props = regionprops(labeled_mask)
    if not props:
        return None
    total_area = 0
    sum_y = 0
    sum_x = 0
    for prop in props:
        sum_y += prop.centroid[0] * prop.area
        sum_x += prop.centroid[1] * prop.area
        total_area += prop.area
    if total_area == 0:
        return None
    return (sum_y / total_area, sum_x / total_area)


def extract_aligned_mask_patch(
    full_mask: np.ndarray,
    top_left_coords_in_full_mask: tuple, 
    target_patch_shape: tuple            
) -> np.ndarray:
    """
    Extracts a patch from a full mask, ensuring it aligns with a target shape
    defined by top-left coordinates and dimensions. Handles boundary conditions.
    """
    coord_x, coord_y = top_left_coords_in_full_mask
    target_h, target_w = target_patch_shape

    if not isinstance(full_mask, np.ndarray) or full_mask.ndim != 2:
        logger.error("extract_aligned_mask_patch: full_mask must be a 2D numpy array.")
        return np.zeros(target_patch_shape, dtype=np.uint8 if full_mask is None else full_mask.dtype)

    full_mask_h, full_mask_w = full_mask.shape

    extracted_patch = np.zeros((target_h, target_w), dtype=full_mask.dtype)

    src_y_start_in_full_mask = coord_y
    src_x_start_in_full_mask = coord_x

    src_y_end_in_full_mask = coord_y + target_h
    src_x_end_in_full_mask = coord_x + target_w

    intersect_src_y_start = max(0, src_y_start_in_full_mask)
    intersect_src_x_start = max(0, src_x_start_in_full_mask)
    intersect_src_y_end = min(full_mask_h, src_y_end_in_full_mask)
    intersect_src_x_end = min(full_mask_w, src_x_end_in_full_mask)

    if intersect_src_y_start < intersect_src_y_end and intersect_src_x_start < intersect_src_x_end:
        data_to_copy_from_full_mask = full_mask[
            intersect_src_y_start:intersect_src_y_end,
            intersect_src_x_start:intersect_src_x_end
        ]

        dest_y_start_in_patch = intersect_src_y_start - src_y_start_in_full_mask
        dest_x_start_in_patch = intersect_src_x_start - src_x_start_in_full_mask
        
        dest_y_start_in_patch = max(0, dest_y_start_in_patch)
        dest_x_start_in_patch = max(0, dest_x_start_in_patch)

        h_paste, w_paste = data_to_copy_from_full_mask.shape

        if (dest_y_start_in_patch + h_paste <= target_h and
            dest_x_start_in_patch + w_paste <= target_w):
            extracted_patch[
                dest_y_start_in_patch : dest_y_start_in_patch + h_paste,
                dest_x_start_in_patch : dest_x_start_in_patch + w_paste
            ] = data_to_copy_from_full_mask
        else:
            h_paste_clipped = min(h_paste, target_h - dest_y_start_in_patch)
            w_paste_clipped = min(w_paste, target_w - dest_x_start_in_patch)
            if h_paste_clipped > 0 and w_paste_clipped > 0:
                 extracted_patch[
                    dest_y_start_in_patch : dest_y_start_in_patch + h_paste_clipped,
                    dest_x_start_in_patch : dest_x_start_in_patch + w_paste_clipped
                ] = data_to_copy_from_full_mask[:h_paste_clipped, :w_paste_clipped]
                 logger.warning(f"extract_aligned_mask_patch: Paste region was clipped. "
                                f"Full mask shape: {full_mask.shape}, "
                                f"Coords: ({coord_x},{coord_y}), "
                                f"Target shape: ({target_h},{target_w}). "
                                f"Pasted ({w_paste_clipped}x{h_paste_clipped}) at "
                                f"({dest_x_start_in_patch},{dest_y_start_in_patch})")

    return extracted_patch

def extract_patch_from_slide(slide, coords_level, size_level, hotspot_level):
    downsample = slide.level_downsamples[hotspot_level]
    x_l0, y_l0 = int(coords_level[0] * downsample), int(coords_level[1] * downsample)
    hs_patch_pil = slide.read_region((x_l0, y_l0), hotspot_level, size_level).convert("RGB")
    hs_patch_rgb = np.array(hs_patch_pil)
    return hs_patch_rgb


def classify_labels_by_dab(
    labels_filtered: np.ndarray,
    hs_patch_rgb: np.ndarray,
    tumor_cell_mask_patch: np.ndarray, 
    dab_threshold: float = 0.15,
    min_dab_positive_ratio: float = 0.1
) -> tuple:
    """
    Classify StarDist-detected nuclei as DAB- or DAB+ based on staining.
    """

    if labels_filtered is None or hs_patch_rgb is None or tumor_cell_mask_patch is None:
        logger.error("classify_labels_by_dab received None for critical input (labels, RGB patch, or tumor mask).")
        shape = labels_filtered.shape if labels_filtered is not None else (0,0)
        return np.zeros(shape, dtype=np.uint8), 0, 0, [], []

    if labels_filtered.shape != tumor_cell_mask_patch.shape:
        logger.warning(f"Resizing tumor_cell_mask_patch from {tumor_cell_mask_patch.shape} to {labels_filtered.shape} to match labels.")
        tumor_cell_mask_patch = cv2.resize(tumor_cell_mask_patch.astype(np.uint8), (labels_filtered.shape[1], labels_filtered.shape[0]), interpolation=cv2.INTER_NEAREST)

    labels_classified = np.zeros_like(labels_filtered, dtype=np.uint8)
    positive_tumor_nuclei_centroids = []
    all_tumor_nuclei_centroids = []
    ki67_positive_tumor_nuclei_count = 0
    total_tumor_nuclei_count = 0

    labels_filtered_int = labels_filtered.astype(np.uint16)
    props = regionprops(labels_filtered_int)

    for prop in props:
        label = prop.label
        if not np.all(np.isfinite(prop.centroid)):
             logger.warning(f"Non-finite centroid for label {label}. Skipping."); continue
        
        centroid_y_float, centroid_x_float = prop.centroid
        all_tumor_nuclei_centroids.append((centroid_y_float, centroid_x_float))
        total_tumor_nuclei_count += 1

        min_r, min_c, max_r, max_c = prop.bbox
        nucleus_patch_rgb_bbox = hs_patch_rgb[min_r:max_r, min_c:max_c]
        nucleus_mask_in_bbox = (labels_filtered_int[min_r:max_r, min_c:max_c] == label)

        if nucleus_patch_rgb_bbox.size == 0:
            labels_classified[labels_filtered_int == label] = 1
            continue

        try:
            dab_mask_in_bbox = get_dab_mask(nucleus_patch_rgb_bbox, dab_threshold)
            dab_positive_pixels_in_nucleus = (dab_mask_in_bbox.astype(bool) & nucleus_mask_in_bbox).sum()
            total_nucleus_pixels = nucleus_mask_in_bbox.sum()
            dab_positive_ratio = dab_positive_pixels_in_nucleus / total_nucleus_pixels if total_nucleus_pixels > 0 else 0

            nucleus_mask_global = (labels_filtered_int == label)
            if dab_positive_ratio > min_dab_positive_ratio:
                labels_classified[nucleus_mask_global] = 2  # DAB+ tumor nucleus
                ki67_positive_tumor_nuclei_count += 1
                positive_tumor_nuclei_centroids.append((centroid_y_float, centroid_x_float))
            else:
                labels_classified[nucleus_mask_global] = 1  # DAB- tumor nucleus
        except Exception as e:
            logger.error(f"Error processing DAB for tumor nucleus label {label}: {e}", exc_info=True)
            nucleus_mask_global = (labels_filtered_int == label)
            if np.any(nucleus_mask_global): labels_classified[nucleus_mask_global] = 1
            continue

    logger.debug(f"Classified tumor nuclei: {total_tumor_nuclei_count} total, {ki67_positive_tumor_nuclei_count} positive.")
    return (labels_classified, ki67_positive_tumor_nuclei_count, total_tumor_nuclei_count,
            positive_tumor_nuclei_centroids, all_tumor_nuclei_centroids)


def predict_patch_stardist(model, image_patch_rgb, tumor_cell_patch, actual_pixel_size_um):
    """
    Runs StarDist prediction on a single RGB image patch, handling rescaling
    and size filtering.

    Args:
        model (StarDist2D): Loaded StarDist2D model.
        image_patch_rgb (np.ndarray): Input RGB image patch (H, W, 3).
        actual_pixel_size_um (float): Actual pixel size of the input patch (microns).

    Returns:
        tuple: (labels_filtered, details, n_objects_after_filter) containing:
                 - labels_filtered (np.ndarray | None): Filtered label image (uint16).
                 - details (dict | None): Details dictionary from predict_instances.
                 - n_objects_after_filter (int): Count of objects after all filtering.
               Returns (None, None, 0) on error.
    """

    probability_threshold = 0.1
    nms_overlap_threshold = 0.3
    target_pixel_size = 0.5  

    logger.info("Processing patch for StarDist prediction...")
    logger.debug(f"\tOriginal input image_patch_rgb shape: {image_patch_rgb.shape}, dtype: {image_patch_rgb.dtype}")
    if len(image_patch_rgb.shape) == 3 and image_patch_rgb.shape[2] == 3:
        image_gray = cv2.cvtColor(image_patch_rgb, cv2.COLOR_RGB2GRAY)
    elif len(image_patch_rgb.shape) == 2: 
        image_gray = image_patch_rgb
    else:
        logger.error(f"\tUnsupported image_patch_rgb shape: {image_patch_rgb.shape}. Expected 2D or 3D with 3 channels.")
        return None, None, 0

    img_to_process = img_as_float(image_gray)
    img_to_process = 1.0 - img_to_process

    rescaled_tumor_cell_patch = tumor_cell_patch.astype(np.uint8) 
    rescale_factor = 1.0

    if actual_pixel_size_um is None or target_pixel_size is None:
        logger.warning("\tPixel size info missing. Skipping rescaling for image and tumor_cell_patch.")
    elif abs(actual_pixel_size_um - target_pixel_size) < 1e-6:
        logger.debug(f"\tActual pixel size ({actual_pixel_size_um:.3f}) matches target ({target_pixel_size:.3f}). No rescaling needed for image and tumor_cell_patch.")
    else:
        rescale_factor = actual_pixel_size_um / target_pixel_size

        original_height, original_width = img_to_process.shape 
        new_height_img = int(original_height * rescale_factor)
        new_width_img = int(original_width * rescale_factor)
        max_dimension = 15000 

        if new_height_img > max_dimension or new_width_img > max_dimension:
            logger.warning(f"\tImage rescaling would create image of size {new_width_img}x{new_height_img}, which exceeds maximum {max_dimension}x{max_dimension}. Skipping image rescaling.")
            rescale_factor = 1.0 

        if abs(rescale_factor - 1.0) > 1e-6:
            logger.info(f"\tRescaling prediction input image by factor {rescale_factor:.4f} (from {actual_pixel_size_um:.3f} to {target_pixel_size:.3f} um/pix)...")
            try:
                img_to_process_rescaled = rescale( 
                    img_to_process,
                    rescale_factor,
                    anti_aliasing=True,
                    mode='reflect',
                    preserve_range=True
                )
                img_to_process_rescaled = np.clip(img_to_process_rescaled, 0, 1)
                logger.debug(f"\tRescaled image shape: {img_to_process_rescaled.shape}")

                target_mask_shape = img_to_process_rescaled.shape[:2]

                rescaled_tumor_cell_patch_temp = rescale(
                    tumor_cell_patch.astype(float),
                    rescale_factor,
                    order=0,
                    anti_aliasing=False,
                    mode='constant',
                    cval=0,
                    preserve_range=True
                )
                rescaled_tumor_cell_patch_temp = (rescaled_tumor_cell_patch_temp > 0.5).astype(np.uint8)
                logger.debug(f"\tRescaled tumor_cell_patch shape (initial): {rescaled_tumor_cell_patch_temp.shape}")

                if rescaled_tumor_cell_patch_temp.shape != target_mask_shape:
                    logger.warning(
                        f"\tShape mismatch after rescaling tumor_cell_patch: "
                        f"\t{rescaled_tumor_cell_patch_temp.shape} vs rescaled image: {target_mask_shape}. "
                        f"\tAttempting corrective resize."
                    )
                    rescaled_tumor_cell_patch = resize(
                        rescaled_tumor_cell_patch_temp,
                        target_mask_shape, 
                        order=0,
                        anti_aliasing=False,
                        preserve_range=True,
                        mode='constant',
                        cval=0
                    )
                    rescaled_tumor_cell_patch = (rescaled_tumor_cell_patch > 0.5).astype(np.uint8)
                else:
                    rescaled_tumor_cell_patch = rescaled_tumor_cell_patch_temp

                logger.debug(f"\tFinal rescaled tumor_cell_patch shape: {rescaled_tumor_cell_patch.shape}")
                img_to_process = img_to_process_rescaled 

            except Exception as e:
                logger.error(f"\tError during image or tumor_cell_patch rescaling: {e}", exc_info=True)
                return None, None, 0
        else:
            logger.debug(f"\tRescaling skipped due to safety check or factor being too close to 1.0.")
            if tumor_cell_patch.shape != img_to_process.shape[:2]:
                 logger.warning(f"Original tumor_cell_patch shape {tumor_cell_patch.shape} different from image {img_to_process.shape[:2]} and no rescale. Resizing tumor_cell_patch.")
                 rescaled_tumor_cell_patch = resize(
                        tumor_cell_patch.astype(np.uint8),
                        img_to_process.shape[:2],
                        order=0, anti_aliasing=False, preserve_range=True, mode='constant', cval=0
                    )
                 rescaled_tumor_cell_patch = (rescaled_tumor_cell_patch > 0.5).astype(np.uint8)

    try:
        img_norm = normalize(img_to_process, 1.0, 99.0, axis=(0, 1)) 
        logger.debug(f"\tNormalization complete. Normalized image shape: {img_norm.shape}")

        labels_pred_scaled, details = model.predict_instances(
            img_norm,
            prob_thresh=probability_threshold,
            nms_thresh=nms_overlap_threshold,
            scale=None,
            return_predict=False
        )
        n_objects = labels_pred_scaled.max()
        logger.info(f"\tStarDist found {n_objects} raw objects (prob_thresh={probability_threshold}).")

    except Exception as e:
        logger.error(f"\tError during StarDist prediction: {e}", exc_info=True)
        return None, None, 0

    if not isinstance(labels_pred_scaled, np.ndarray):
        logger.error(f"\tPredicted labels are not a numpy array (type: {type(labels_pred_scaled)}). Cannot proceed.")
        return None, None, 0 
    if labels_pred_scaled.ndim != 2:
        logger.error(f"\tPredicted labels have unexpected dimensions ({labels_pred_scaled.ndim}). Expected 2. Cannot proceed.")
        return None, None, 0

    if rescaled_tumor_cell_patch is not None and rescaled_tumor_cell_patch.shape == labels_pred_scaled.shape:
        logger.info("\tApplying tumor_cell_patch filter to StarDist labels...")
        props = regionprops(labels_pred_scaled)
        labels_to_remove = []

        for prop in props:
            centroid_y, centroid_x = int(prop.centroid[0]), int(prop.centroid[1])
            
            if not (0 <= centroid_y < rescaled_tumor_cell_patch.shape[0] and \
                    0 <= centroid_x < rescaled_tumor_cell_patch.shape[1]):
                labels_to_remove.append(prop.label) 
                continue

            if rescaled_tumor_cell_patch[centroid_y, centroid_x] == 0:
                labels_to_remove.append(prop.label)
        
        if labels_to_remove:
            logger.info("\tRemoving {len(labels_to_remove)} StarDist objects outside tumor_cell_patch.")
            mask_to_remove = np.isin(labels_pred_scaled, labels_to_remove)
            labels_pred_scaled[mask_to_remove] = 0
            n_objects = len(np.unique(labels_pred_scaled)) - 1 if 0 in np.unique(labels_pred_scaled) else len(np.unique(labels_pred_scaled))
            if n_objects < 0: n_objects = 0 
            logger.info(f"\tNumber of objects after tumor_cell_patch filtering: {n_objects}")
        else:
            logger.info("\tNo StarDist objects needed removal based on tumor_cell_patch.")
    elif rescaled_tumor_cell_patch is None:
        logger.warning("\trescaled_tumor_cell_patch is None. Skipping filtering based on it.")
    else:
        logger.warning(f"\tShape mismatch: rescaled_tumor_cell_patch {rescaled_tumor_cell_patch.shape} vs labels_pred_scaled {labels_pred_scaled.shape}. Skipping filtering.")

    labels_pred = None
    original_shape = image_patch_rgb.shape[:2]

    if abs(rescale_factor - 1.0) > 1e-6: 
        logger.debug(f"\tResizing predicted labels from {labels_pred_scaled.shape} back to original shape {original_shape}...")
        try:
            labels_pred = resize(
                labels_pred_scaled,
                output_shape=original_shape,
                order=0, 
                preserve_range=True,
                anti_aliasing=False
            )
            labels_pred = labels_pred.astype(np.uint16)
        except Exception as resize_err:
            logger.error(f"\tError resizing labels back to original size: {resize_err}", exc_info=True)
            return None, None, 0 
    else:
        logger.debug("\tNo input rescaling was applied, using predicted labels directly (checking shape).")
        if labels_pred_scaled.shape != original_shape:
            logger.warning(f"\tPredicted label shape {labels_pred_scaled.shape} differs from original image shape {original_shape} even without rescaling. Attempting corrective resize.")
            try:
                labels_pred = resize(labels_pred_scaled, output_shape=original_shape, order=0, preserve_range=True, anti_aliasing=False)
                labels_pred = labels_pred.astype(np.uint16)
            except Exception as resize_err:
                logger.error(f"\tError during corrective resize: {resize_err}", exc_info=True)
                return None, None, 0 
        else:
            labels_pred = labels_pred_scaled
        labels_pred = labels_pred.astype(np.uint16)


    logger.info(f"\tApplying dynamic size filtering to labels at original resolution ({labels_pred.shape})...")
    labels_pred_filtered = labels_pred.copy()

    min_cell_diameter_um = 4.0    
    max_cell_diameter_um = 40.0   

    if actual_pixel_size_um is None or actual_pixel_size_um <= 0:
        logger.warning(f"\tCannot perform dynamic size filtering: actual_pixel_size_um is invalid ({actual_pixel_size_um}). Skipping this filter.")
        n_objects_after_filter = len(np.unique(labels_pred_filtered[labels_pred_filtered > 0]))
    else:
        min_cell_radius_pixels = (min_cell_diameter_um / 2.0) / actual_pixel_size_um
        max_cell_radius_pixels = (max_cell_diameter_um / 2.0) / actual_pixel_size_um

        min_area_pixels = np.pi * (min_cell_radius_pixels ** 2)
        max_area_pixels = np.pi * (max_cell_radius_pixels ** 2)

        logger.info(f"\tDynamic size filtering using actual_pixel_size: {actual_pixel_size_um:.3f} µm/pixel:")
        logger.info(f"\t  - Min physical cell diameter: {min_cell_diameter_um:.1f} µm  -> Min area: {min_area_pixels:.1f} pixels")
        logger.info(f"\t  - Max physical cell diameter: {max_cell_diameter_um:.1f} µm  -> Max area: {max_area_pixels:.1f} pixels")

        num_labels_before_dynamic_filter = labels_pred.max() 
        
        if num_labels_before_dynamic_filter > 0:
            object_labels, object_sizes_pixels = np.unique(labels_pred[labels_pred > 0], return_counts=True)

            if len(object_sizes_pixels) > 0:
                logger.debug(f"\tObject size stats before dynamic filtering (pixels): "
                            f"\tmin={np.min(object_sizes_pixels)}, max={np.max(object_sizes_pixels)}, "
                            f"\tmean={np.mean(object_sizes_pixels):.1f}, median={np.median(object_sizes_pixels):.1f}")

                labels_too_small = object_labels[object_sizes_pixels < min_area_pixels]
                labels_too_large = object_labels[object_sizes_pixels > max_area_pixels]

                objects_to_remove = np.unique(np.concatenate([labels_too_small, labels_too_large]))

                if len(objects_to_remove) > 0:
                    logger.info(f"\tRemoving {len(labels_too_small)} objects smaller than {min_area_pixels:.1f} pixels.")
                    logger.info(f"\tRemoving {len(labels_too_large)} objects larger than {max_area_pixels:.1f} pixels.")
                    logger.info(f"\tTotal unique objects to remove by size: {len(objects_to_remove)} out of {len(object_labels)} objects.")

                    mask_to_remove = np.isin(labels_pred_filtered, objects_to_remove)
                    labels_pred_filtered[mask_to_remove] = 0
                else:
                    logger.info("\tNo objects found outside the dynamic physical size range.")
            else:
                logger.debug("\tNo valid objects (label > 0) found to apply dynamic size statistics.")
        else:
            logger.info("\tNo objects detected in the prediction (before dynamic size filtering), skipping.")

        n_objects_after_filter = len(np.unique(labels_pred_filtered[labels_pred_filtered > 0]))
        logger.info(f"\tNumber of objects after dynamic size filtering: {n_objects_after_filter}")

    logger.info("\tStarDist prediction and all filtering complete.")
    return labels_pred_filtered, details, n_objects_after_filter


def refine_hotspot_with_stardist(
    candidate_hotspot, slide, stardist_model, hotspot_level,
    actual_pixel_size_um, tumor_cell_mask_l2, debug_dir=None, candidate_index=0,
    dab_threshold=0.15, min_dab_positive_ratio=0.1,
    min_cells=500, max_cells=600, max_iterations=25, resize_factor=0.15
):
    func_name = 'refine_hotspot_with_stardist'
    hotspot = candidate_hotspot.copy()
    full_mask_h, full_mask_w = tumor_cell_mask_l2.shape

    initial_w, initial_h = hotspot['size_level']
    initial_x, initial_y = hotspot['coords_level']
    center_x = initial_x + initial_w / 2.0
    center_y = initial_y + initial_h / 2.0
    current_target_size_w = float(initial_w)
    current_target_size_h = float(initial_h)

    final_hs_patch_rgb = None
    final_labels_filtered = None
    final_corresponding_tumor_cell_patch = None
    final_refined_nuclei_count = 0
    final_coords_x, final_coords_y = initial_x, initial_y

    iteration = 0
    while iteration < max_iterations:
        if final_hs_patch_rgb is not None and final_labels_filtered is not None:
            temp_classified_labels, _, _, _, _ = classify_labels_by_dab(
                final_labels_filtered, final_hs_patch_rgb, final_corresponding_tumor_cell_patch,
                dab_threshold, min_dab_positive_ratio
            )
            
            dab_positive_mask = (temp_classified_labels == 2)
            centroid_of_dab_pos = _get_mask_centroid(dab_positive_mask)

            if centroid_of_dab_pos:
                cy, cx = centroid_of_dab_pos
                center_x = final_coords_x + cx
                center_y = final_coords_y + cy
                logger.debug(f"[{func_name}] Iter {iteration+1}: Re-centering FoV to DAB+ nuclei centroid at ({center_x:.1f}, {center_y:.1f})")

        coords_x = int(round(center_x - current_target_size_w / 2.0))
        coords_y = int(round(center_y - current_target_size_h / 2.0))
        coords_x = np.clip(coords_x, 0, full_mask_w - 1)
        coords_y = np.clip(coords_y, 0, full_mask_h - 1)
        target_w_int = int(round(current_target_size_w))
        target_h_int = int(round(current_target_size_h))
        target_w_int = min(target_w_int, full_mask_w - coords_x)
        target_h_int = min(target_h_int, full_mask_h - coords_y)
        target_w_int = max(1, target_w_int)
        target_h_int = max(1, target_h_int)
        
        logger.debug(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: "
                     f"Attempting FoV with target size ({target_w_int}x{target_h_int}) at ({coords_x},{coords_y})")

        hs_patch_rgb = extract_patch_from_slide(
            slide, (coords_x, coords_y), (target_w_int, target_h_int), hotspot_level
        )

        if hs_patch_rgb is None or hs_patch_rgb.size == 0:
            logger.warning(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: Failed to extract RGB patch. Breaking loop.")
            break

        patch_actual_h, patch_actual_w = hs_patch_rgb.shape[:2]

        extracted_tumor_cell_patch = extract_aligned_mask_patch(
            full_mask=tumor_cell_mask_l2,
            top_left_coords_in_full_mask=(coords_x, coords_y),
            target_patch_shape=(patch_actual_h, patch_actual_w)
        )

        current_labels_filtered, _, current_n_objects = predict_patch_stardist(
            stardist_model, hs_patch_rgb, extracted_tumor_cell_patch, actual_pixel_size_um
        )
        
        if current_labels_filtered is None:
            logger.warning(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: predict_patch_stardist failed. Ending refinement for this candidate.")
            break

        final_hs_patch_rgb = hs_patch_rgb
        final_labels_filtered = current_labels_filtered
        final_corresponding_tumor_cell_patch = extracted_tumor_cell_patch
        final_refined_nuclei_count = current_n_objects
        final_coords_x, final_coords_y = coords_x, coords_y

        logger.info(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: "
                    f"Found {final_refined_nuclei_count} nuclei. Target FoV ({current_target_size_w:.0f}x{current_target_size_h:.0f})")

        if min_cells <= final_refined_nuclei_count <= max_cells:
            logger.info(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: Nuclei count in range. Finalizing.")
            break
        
        elif final_refined_nuclei_count < min_cells:
            current_target_size_w *= (1 + resize_factor)
            current_target_size_h *= (1 + resize_factor)
        else: 
            current_target_size_w *= (1 - resize_factor)
            current_target_size_h *= (1 - resize_factor)

        current_target_size_w = np.clip(current_target_size_w, 64, 32768)
        current_target_size_h = np.clip(current_target_size_h, 64, 32768)

        iteration += 1
        if iteration == max_iterations:
            logger.warning(f"[{func_name}] Cand {candidate_index+1}: Max iterations reached. Using results from last valid iteration.")

    if final_hs_patch_rgb is None or final_labels_filtered is None:
        logger.error(f"[{func_name}] Cand {candidate_index+1}: No valid patches/labels after FoV adjustment. Cannot proceed.")
        hotspot.update({
            'stardist_total_count_filtered': 0, 'stardist_ki67_pos_count': 0, 'stardist_proliferation_index': 0.0,
            'positive_centroids': [], 'all_centroids': [], 'stardist_labels': None
        })
        hotspot['size_level'] = (int(round(current_target_size_w)), int(round(current_target_size_h)))
        downsample_hl = slide.level_downsamples[hotspot_level]
        hotspot['coords_l0'] = (int(coords_x * downsample_hl), int(coords_y * downsample_hl))
        hotspot['size_l0'] = (int(hotspot['size_level'][0] * downsample_hl), int(hotspot['size_level'][1] * downsample_hl))
        return hotspot

    patch_actual_h_final, patch_actual_w_final = final_hs_patch_rgb.shape[:2]
    hotspot['coords_level'] = (final_coords_x, final_coords_y)
    hotspot['size_level'] = (patch_actual_w_final, patch_actual_h_final)

    downsample_hl = slide.level_downsamples[hotspot_level]
    hotspot['coords_l0'] = (int(hotspot['coords_level'][0] * downsample_hl), int(hotspot['coords_level'][1] * downsample_hl))
    hotspot['size_l0'] = (int(hotspot['size_level'][0] * downsample_hl), int(hotspot['size_level'][1] * downsample_hl))

    if final_refined_nuclei_count == 0:
        logger.warning(f"[{func_name}] Cand {candidate_index+1}: No nuclei detected in final FoV. Reporting zero counts.")
        hotspot.update({
            'stardist_total_count_filtered': 0, 'stardist_ki67_pos_count': 0, 'stardist_proliferation_index': 0.0,
            'positive_centroids': [], 'all_centroids': [], 'stardist_labels': None
        })
        return hotspot

    classified_labels_dab, dab_pos_count, total_nuclei_for_dab_check, pos_centroids, all_centroids = classify_labels_by_dab(
        final_labels_filtered, final_hs_patch_rgb, final_corresponding_tumor_cell_patch,
        dab_threshold, min_dab_positive_ratio
    )
    
    if total_nuclei_for_dab_check != final_refined_nuclei_count:
        logger.warning(f"[{func_name}] Cand {candidate_index+1}: Mismatch in nuclei counts. "
                       f"predict_patch_stardist refined count: {final_refined_nuclei_count}, "
                       f"classify_labels_by_dab total count: {total_nuclei_for_dab_check}. "
                       f"Using predict_patch_stardist count for PI.")
    
    final_total_tumor_cell_nuclei = final_refined_nuclei_count

    hotspot.update({
        'stardist_total_count_filtered': final_total_tumor_cell_nuclei, 
        'stardist_ki67_pos_count': dab_pos_count,                   
        'stardist_proliferation_index': dab_pos_count / final_total_tumor_cell_nuclei if final_total_tumor_cell_nuclei > 0 else 0.0,
        'positive_centroids': pos_centroids,
        'all_centroids': all_centroids,
        'stardist_labels': final_labels_filtered,
        'classified_labels_dab_hs': classified_labels_dab
    })

    logger.info(
        f"[{func_name}] Candidate {candidate_index+1} (Final Patch L{hotspot_level} {hotspot['size_level']} at {hotspot['coords_level']}): "
        f"Total Tumor Cells={final_total_tumor_cell_nuclei}, "
        f"Ki67+ Tumor Cells={dab_pos_count}, PI={hotspot['stardist_proliferation_index']:.2%}"
    )

    if debug_dir and final_hs_patch_rgb is not None and final_labels_filtered is not None:
        save_path = os.path.join(debug_dir, f"refined_hotspot_{candidate_index+1}_comparison.png")
        visualization.save_stardist_comparison_plot(
            hs_patch_rgb=final_hs_patch_rgb,
            labels_filtered=final_labels_filtered,
            ref_mask=final_corresponding_tumor_cell_patch,
            classified_labels_dab=classified_labels_dab,
            save_path=save_path
        )

    return hotspot