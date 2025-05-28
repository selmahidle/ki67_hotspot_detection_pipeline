import logging
import numpy as np
import warnings
from stardist.models import StarDist2D
from csbdeep.utils import normalize
# import glob # Not used directly here
from skimage.color import label2rgb # Only needed if refine_hotspot generates overlay (which it does)
# import imageio # Not used
import cv2
import openslide
import os
import traceback
# from visualization import save_stardist_comparison_plot, format_for_save # Not used here
from skimage.util import img_as_ubyte, img_as_float
from skimage.transform import resize, rescale
from skimage.measure import regionprops
import sys
import numpy as np
import logging

# Assumed to be in stain_utils.py and imported correctly elsewhere when needed
# from stain_utils import get_dab_mask

logger = logging.getLogger(__name__)

def extract_aligned_mask_patch(
    full_mask: np.ndarray,
    top_left_coords_in_full_mask: tuple, # (x, y) coordinates at the full_mask's level
    target_patch_shape: tuple             # (height, width) of the desired output patch
) -> np.ndarray:
    """
    Extracts a patch from a full mask, ensuring it aligns with a target shape
    defined by top-left coordinates and dimensions. Handles boundary conditions.

    Args:
        full_mask (np.ndarray): The large 2D mask from which to extract.
        top_left_coords_in_full_mask (tuple): (x, y) top-left coordinates
                                              for the patch extraction, relative
                                              to the full_mask.
        target_patch_shape (tuple): Desired (height, width) of the output patch.
                                    This typically matches the shape of an RGB
                                    image patch extracted using the same coordinates
                                    and target dimensions.

    Returns:
        np.ndarray: The extracted mask patch, with the same dtype as full_mask.
                    It will have dimensions specified by target_patch_shape.
                    Areas outside full_mask bounds will be 0.
    """
    coord_x, coord_y = top_left_coords_in_full_mask
    target_h, target_w = target_patch_shape

    if not isinstance(full_mask, np.ndarray) or full_mask.ndim != 2:
        logger.error("extract_aligned_mask_patch: full_mask must be a 2D numpy array.")
        # Return an empty patch of the target shape if full_mask is invalid
        return np.zeros(target_patch_shape, dtype=np.uint8 if full_mask is None else full_mask.dtype)

    full_mask_h, full_mask_w = full_mask.shape

    # Initialize the output patch with zeros (background)
    extracted_patch = np.zeros((target_h, target_w), dtype=full_mask.dtype)

    # Define the RoI in the full_mask's coordinate system
    # These are the coordinates from where we want to START reading in the full_mask
    # and how much we want to read (up to target_h, target_w)
    src_y_start_in_full_mask = coord_y
    src_x_start_in_full_mask = coord_x
    # These define the END of the region we intend to read in full_mask
    src_y_end_in_full_mask = coord_y + target_h
    src_x_end_in_full_mask = coord_x + target_w

    # Determine the actual valid region to read from full_mask (intersection)
    # This handles cases where the target patch is partially or fully outside full_mask
    intersect_src_y_start = max(0, src_y_start_in_full_mask)
    intersect_src_x_start = max(0, src_x_start_in_full_mask)
    intersect_src_y_end = min(full_mask_h, src_y_end_in_full_mask)
    intersect_src_x_end = min(full_mask_w, src_x_end_in_full_mask)

    # If there's a valid intersection (overlap)
    if intersect_src_y_start < intersect_src_y_end and intersect_src_x_start < intersect_src_x_end:
        # The actual data we can copy from the full_mask
        data_to_copy_from_full_mask = full_mask[
            intersect_src_y_start:intersect_src_y_end,
            intersect_src_x_start:intersect_src_x_end
        ]

        # Determine where to paste this data_to_copy into our `extracted_patch`
        # This requires calculating the offset relative to the `extracted_patch`'s (0,0)
        # which corresponds to (src_x_start_in_full_mask, src_y_start_in_full_mask)
        dest_y_start_in_patch = intersect_src_y_start - src_y_start_in_full_mask
        dest_x_start_in_patch = intersect_src_x_start - src_x_start_in_full_mask
        
        # Ensure paste coordinates are non-negative (can happen if top-left is negative, though typically not the case here)
        dest_y_start_in_patch = max(0, dest_y_start_in_patch)
        dest_x_start_in_patch = max(0, dest_x_start_in_patch)

        # Calculate the dimensions of the data to paste
        h_paste, w_paste = data_to_copy_from_full_mask.shape

        # Ensure the paste operation does not go out of bounds of the extracted_patch
        # (This should generally hold if dest_..._start_in_patch are correct and non-negative)
        if (dest_y_start_in_patch + h_paste <= target_h and
            dest_x_start_in_patch + w_paste <= target_w):
            extracted_patch[
                dest_y_start_in_patch : dest_y_start_in_patch + h_paste,
                dest_x_start_in_patch : dest_x_start_in_patch + w_paste
            ] = data_to_copy_from_full_mask
        else:
            # This case implies an issue with coordinate calculation or target_patch_shape
            # For robustness, try to paste the intersection that fits
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
    Classify StarDist-detected nuclei as DAB- or DAB+ based on staining,
    considering ONLY nuclei within the provided tumor_cell_mask_patch.

    Args:
        labels_filtered (ndarray): 2D label image from StarDist (H, W, uint16).
        hs_patch_rgb (ndarray): Original RGB image patch (H, W, 3).
        tumor_cell_mask_patch (ndarray): Binary tumor cell mask patch (H, W, uint8, values 0 or 1).
                                    Must be the same shape as labels_filtered.
        dab_threshold (float): OD threshold for DAB mask generation.
        min_dab_positive_ratio (float): Min ratio of DAB+ pixels for nucleus positivity.

    Returns:
        tuple: Contains:
            - labels_classified (ndarray): 2D array (H,W,uint8) with:
                                           0 (bg),
                                           1 (DAB- tumor nucleus),
                                           2 (DAB+ tumor nucleus),
                                           3 (non-tumor nucleus - optional, currently not used, effectively bg)
            - ki67_positive_tumor_nuclei_count (int): Count of DAB+ nuclei WITHIN tumor.
            - total_tumor_nuclei_count (int): Total nuclei WITHIN tumor.
            - positive_tumor_nuclei_centroids (list): (y,x) for DAB+ tumor nuclei.
            - all_tumor_nuclei_centroids (list): (y,x) for ALL tumor nuclei.
    """
    try:
        from stain_utils import get_dab_mask # Local import
    except ImportError:
        logger.error("Could not import get_dab_mask from stain_utils.")
        shape = labels_filtered.shape if labels_filtered is not None else (0,0)
        return np.zeros(shape, dtype=np.uint8), 0, 0, [], []

    if labels_filtered is None or hs_patch_rgb is None or tumor_cell_mask_patch is None:
        logger.error("classify_labels_by_dab received None for critical input (labels, RGB patch, or tumor mask).")
        shape = labels_filtered.shape if labels_filtered is not None else \
                (hs_patch_rgb.shape[:2] if hs_patch_rgb is not None else \
                 (tumor_cell_mask_patch.shape[:2] if tumor_cell_mask_patch is not None else (0,0)))
        return np.zeros(shape, dtype=np.uint8), 0, 0, [], []

    if labels_filtered.shape != tumor_cell_mask_patch.shape:
        logger.error(f"Shape mismatch: labels_filtered {labels_filtered.shape} vs tumor_cell_mask_patch {tumor_cell_mask_patch.shape}")
        # Attempt to resize tumor_cell_mask_patch if it's the one with different shape, as labels_filtered dictates processing
        if tumor_cell_mask_patch.shape != hs_patch_rgb.shape[:2]:
             logger.warning(f"Resizing tumor_cell_mask_patch from {tumor_cell_mask_patch.shape} to {hs_patch_rgb.shape[:2]} to match RGB patch")
             tumor_cell_mask_patch = cv2.resize(tumor_cell_mask_patch.astype(np.uint8), (hs_patch_rgb.shape[1], hs_patch_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        if labels_filtered.shape != tumor_cell_mask_patch.shape: # Check again after potential resize
             logger.error("Shape mismatch persists after attempting to resize tumor_cell_mask_patch. Cannot proceed.")
             return np.zeros_like(labels_filtered, dtype=np.uint8), 0, 0, [], []


    labels_classified = np.zeros_like(labels_filtered, dtype=np.uint8)
    positive_tumor_nuclei_centroids = []
    all_tumor_nuclei_centroids = []
    ki67_positive_tumor_nuclei_count = 0
    total_tumor_nuclei_count = 0 # Counts only nuclei within the tumor mask

    if not np.issubdtype(labels_filtered.dtype, np.integer):
        logger.warning("labels_filtered dtype not integer, attempting conversion.")
        try:
            labels_filtered_int = labels_filtered.astype(np.uint16)
        except ValueError:
            logger.error("Could not convert labels_filtered to integer type."); return np.zeros_like(labels_filtered, dtype=np.uint8), 0, 0, [], []
    else:
        labels_filtered_int = labels_filtered

    try:
        props = regionprops(labels_filtered_int)
    except TypeError as e_props:
         logger.error(f"TypeError during regionprops: {e_props}"); return np.zeros_like(labels_filtered, dtype=np.uint8), 0, 0, [], []

    for prop in props:
        label = prop.label
        if not np.all(np.isfinite(prop.centroid)):
             logger.warning(f"Non-finite centroid for label {label}. Skipping."); continue

        # Centroid coordinates (row, col) which correspond to (y, x)
        centroid_y_float, centroid_x_float = prop.centroid
        # Convert to int for indexing masks
        centroid_y, centroid_x = int(round(centroid_y_float)), int(round(centroid_x_float))

        # Ensure centroid is within bounds of the tumor_cell_mask_patch
        if not (0 <= centroid_y < tumor_cell_mask_patch.shape[0] and \
                0 <= centroid_x < tumor_cell_mask_patch.shape[1]):
            logger.debug(f"Centroid ({centroid_y},{centroid_x}) for label {label} is outside tumor_cell_mask_patch bounds. Skipping.")
            continue

        # --- TUMOR MASK CHECK ---
        if tumor_cell_mask_patch[centroid_y, centroid_x] == 0: # Assuming tumor mask is 0 (stroma/bg) or 1 (tumor)
            # logger.debug(f"Nucleus label {label} centroid ({centroid_y},{centroid_x}) is NOT in tumor mask. Skipping.")
            # Optionally mark these in labels_classified with a different value if needed later, e.g., 3 for non-tumor nucleus
            # For now, they remain 0 (background) in labels_classified.
            continue
        # -------------------------

        # If we reach here, the nucleus is within the tumor mask
        all_tumor_nuclei_centroids.append((centroid_y_float, centroid_x_float)) # Store float centroid
        total_tumor_nuclei_count += 1

        min_r, min_c, max_r, max_c = prop.bbox
        min_r, min_c = max(0, min_r), max(0, min_c)
        max_r, max_c = min(labels_filtered_int.shape[0], max_r), min(labels_filtered_int.shape[1], max_c)
        if min_r >= max_r or min_c >= max_c:
             logger.warning(f"Invalid bbox for tumor nucleus label {label}. Marking as DAB-.");
             labels_classified[labels_filtered_int == label] = 1; continue

        nucleus_patch_rgb_bbox = hs_patch_rgb[min_r:max_r, min_c:max_c]
        nucleus_mask_in_bbox = (labels_filtered_int[min_r:max_r, min_c:max_c] == label)

        if nucleus_patch_rgb_bbox.size == 0 or nucleus_mask_in_bbox.size == 0 or np.sum(nucleus_mask_in_bbox) == 0:
            logger.warning(f"Empty patch/mask for tumor nucleus label {label}. Marking as DAB-.");
            labels_classified[labels_filtered_int == label] = 1; continue

        try:
            if nucleus_patch_rgb_bbox.ndim == 2: nucleus_patch_rgb_bbox = np.stack([nucleus_patch_rgb_bbox]*3, axis=-1)
            elif nucleus_patch_rgb_bbox.shape[-1] != 3:
                 logger.warning(f"Nucleus patch for label {label} has {nucleus_patch_rgb_bbox.shape[-1]} channels. Marking as DAB-.");
                 labels_classified[labels_filtered_int == label] = 1; continue

            dab_mask_in_bbox = get_dab_mask(nucleus_patch_rgb_bbox, dab_threshold)
            if dab_mask_in_bbox is None or dab_mask_in_bbox.shape != nucleus_mask_in_bbox.shape:
                 logger.warning(f"DAB mask invalid/mismatch for label {label}. Marking as DAB-.");
                 labels_classified[labels_filtered_int == label] = 1; continue

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

    logger.debug(f"Classified TUMOR nuclei: {total_tumor_nuclei_count} total, {ki67_positive_tumor_nuclei_count} positive.")
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
        tuple: (labels_filtered, details) containing:
                 - labels_filtered (np.ndarray | None): Filtered label image (uint16).
                 - details (dict | None): Details dictionary from predict_instances.
               Returns (None, None) on error during prediction or critical processing.
    """
    # --- Define Prediction Parameters ---
    probability_threshold = 0.4
    nms_overlap_threshold = 0.3
    target_pixel_size = 0.2 # Target pixel size for rescaling model input

    logger.info("Processing patch for StarDist prediction...")
    logger.debug(f"Input patch shape: {image_patch_rgb.shape}, dtype: {image_patch_rgb.dtype}, actual_pixel_size: {actual_pixel_size_um}")

    # --- Rescaling Logic ---
    img_to_process = img_as_float(image_patch_rgb)
    rescaled_tumor_cell_patch = tumor_cell_patch.astype(np.uint8)
    rescale_factor = 1.0 # Default if no rescaling needed

    if actual_pixel_size_um is None or target_pixel_size is None:
        logger.warning("Pixel size info missing. Skipping rescaling for image and tumor_cell_patch.")
    elif abs(actual_pixel_size_um - target_pixel_size) < 1e-6:
        logger.debug(f"Actual pixel size ({actual_pixel_size_um:.3f}) matches target ({target_pixel_size:.3f}). No rescaling needed for image and tumor_cell_patch.")
    else:
        rescale_factor = actual_pixel_size_um / target_pixel_size
        logger.info(f"Rescaling prediction input image and tumor_cell_patch by factor {rescale_factor:.4f} (from {actual_pixel_size_um:.3f} to {target_pixel_size:.3f} um/pix)...")
        try:
            img_to_process = rescale(
                img_to_process,
                rescale_factor,
                anti_aliasing=True,
                mode='reflect',
                preserve_range=True,
                channel_axis=-1
            )
            img_to_process = np.clip(img_to_process, 0, 1) # Ensure range after rescale
            logger.debug(f"Rescaled image shape: {img_to_process.shape}")

            # Rescale tumor_cell_patch to match the new image dimensions
            # Use order=0 (nearest neighbor) for masks to preserve binary nature
            # Rescale tumor_cell_patch to match the new image dimensions
            # Use order=0 (nearest neighbor) for masks to preserve binary nature
            rescaled_tumor_cell_patch = rescale(
                tumor_cell_patch.astype(float), # Rescale works well with float
                rescale_factor,
                order=0, # Nearest-neighbor for masks
                anti_aliasing=False, # No anti-aliasing for masks
                mode='constant', # Pad with 0 (background) if needed
                cval=0,
                preserve_range=True # Preserve 0-1 range
                # output_shape=img_to_process.shape[:2] # <<< REMOVED THIS LINE
            )
            # Ensure it's binary (0 or 1) after rescaling, e.g., by thresholding
            rescaled_tumor_cell_patch = (rescaled_tumor_cell_patch > 0.5).astype(np.uint8)
            logger.debug(f"Rescaled tumor_cell_patch shape: {rescaled_tumor_cell_patch.shape}")

            # OPTIONAL BUT RECOMMENDED: Add a check and potential corrective resize
            # if shapes don't perfectly match after rescaling (due to floating point of rescale_factor)
            if rescaled_tumor_cell_patch.shape != img_to_process.shape[:2]:
                logger.warning(
                    f"Shape mismatch after rescaling tumor_cell_patch: "
                    f"{rescaled_tumor_cell_patch.shape} vs img_to_process: {img_to_process.shape[:2]}. "
                    f"Attempting corrective resize."
                )
                # Using skimage.transform.resize for consistency, or cv2.resize
                rescaled_tumor_cell_patch = resize(
                    rescaled_tumor_cell_patch,
                    img_to_process.shape[:2], # Target shape (height, width)
                    order=0, # Nearest-neighbor for masks
                    anti_aliasing=False,
                    preserve_range=True,
                    mode='constant',
                    cval=0
                )
                rescaled_tumor_cell_patch = (rescaled_tumor_cell_patch > 0.5).astype(np.uint8) # Ensure binary
                logger.debug(f"Correctively resized tumor_cell_patch shape: {rescaled_tumor_cell_patch.shape}")
            # Ensure it's binary (0 or 1) after rescaling, e.g., by thresholding
            rescaled_tumor_cell_patch = (rescaled_tumor_cell_patch > 0.5).astype(np.uint8)
            logger.debug(f"Rescaled tumor_cell_patch shape: {rescaled_tumor_cell_patch.shape}")

        except Exception as e:
            logger.error(f"Error during image or tumor_cell_patch rescaling: {e}", exc_info=True)
            return None, None, 0 # Return 0 for count as well

            
    # --- Normalization and Prediction ---
    try:
        # Normalize based on the potentially rescaled image
        img_norm = normalize(img_to_process, 1.0, 99.0, axis=(0, 1)) # Assuming 3D input (H, W, C)
        logger.debug("Normalization complete.")

        labels_pred_scaled, details = model.predict_instances(
            img_norm,
            prob_thresh=probability_threshold,
            nms_thresh=nms_overlap_threshold,
            scale=None, # Already handled rescaling manually if needed
            return_predict=False # We don't need the intermediate probability maps here
        )
        n_objects = labels_pred_scaled.max()
        logger.info(f"StarDist found {n_objects} raw objects (prob_thresh={probability_threshold}).")

    except Exception as e:
        logger.error(f"Error during StarDist prediction: {e}", exc_info=True)
        return None, None

    # --- Validation of Predicted Labels ---
    if not isinstance(labels_pred_scaled, np.ndarray):
        logger.error(f"Predicted labels are not a numpy array (type: {type(labels_pred_scaled)}). Cannot proceed.")
        return None, None, 0 # Also return 0 for count
    if labels_pred_scaled.ndim != 2:
        logger.error(f"Predicted labels have unexpected dimensions ({labels_pred_scaled.ndim}). Expected 2. Cannot proceed.")
        return None, None, 0 # Also return 0 for count

    # --- Filter StarDist labels based on rescaled_tumor_cell_patch ---
    if rescaled_tumor_cell_patch is not None and rescaled_tumor_cell_patch.shape == labels_pred_scaled.shape:
        logger.info("Applying tumor_cell_patch filter to StarDist labels...")
        props = regionprops(labels_pred_scaled)
        labels_to_remove = []

        for prop in props:
            # Check if centroid of the nucleus is within the tumor cell mask region
            centroid_y, centroid_x = int(prop.centroid[0]), int(prop.centroid[1])
            
            # Boundary check for centroid
            if not (0 <= centroid_y < rescaled_tumor_cell_patch.shape[0] and \
                    0 <= centroid_x < rescaled_tumor_cell_patch.shape[1]):
                labels_to_remove.append(prop.label) # Remove if centroid is out of bounds
                continue

            if rescaled_tumor_cell_patch[centroid_y, centroid_x] == 0:
                labels_to_remove.append(prop.label)
        
        if labels_to_remove:
            logger.info(f"Removing {len(labels_to_remove)} StarDist objects outside tumor_cell_patch.")
            # Create a boolean mask for efficient removal
            mask_to_remove = np.isin(labels_pred_scaled, labels_to_remove)
            labels_pred_scaled[mask_to_remove] = 0
            # Update n_objects (the count of actual objects)
            n_objects = len(np.unique(labels_pred_scaled)) - 1 if 0 in np.unique(labels_pred_scaled) else len(np.unique(labels_pred_scaled))
            if n_objects < 0: n_objects = 0 # Ensure non-negative
            logger.info(f"Number of objects after tumor_cell_patch filtering: {n_objects}")
        else:
            logger.info("No StarDist objects needed removal based on tumor_cell_patch.")
    elif rescaled_tumor_cell_patch is None:
        logger.warning("rescaled_tumor_cell_patch is None. Skipping filtering based on it.")
    else: # Shape mismatch
        logger.warning(f"Shape mismatch: rescaled_tumor_cell_patch {rescaled_tumor_cell_patch.shape} vs labels_pred_scaled {labels_pred_scaled.shape}. Skipping filtering.")


    # --- Resize Label Mask Back to Original Image Size (if needed) ---
    labels_pred = None
    original_shape = image_patch_rgb.shape[:2] # Target shape

    if abs(rescale_factor - 1.0) > 1e-6: # Check if rescaling *was* applied
        logger.debug(f"Resizing predicted labels from {labels_pred_scaled.shape} back to original shape {original_shape}...")
        try:
            labels_pred = resize(
                labels_pred_scaled,
                output_shape=original_shape,
                order=0, # Nearest neighbor for labels
                preserve_range=True,
                anti_aliasing=False
            )
            labels_pred = labels_pred.astype(np.uint16)
        except Exception as resize_err:
            logger.error(f"Error resizing labels back to original size: {resize_err}", exc_info=True)
            return None, None # Return None if resizing fails, as labels won't match original image
    else:
        logger.debug("No input rescaling was applied, using predicted labels directly (checking shape).")
        if labels_pred_scaled.shape != original_shape:
            logger.warning(f"Predicted label shape {labels_pred_scaled.shape} differs from original image shape {original_shape} even without rescaling. Attempting corrective resize.")
            try:
                labels_pred = resize(labels_pred_scaled, output_shape=original_shape, order=0, preserve_range=True, anti_aliasing=False)
            except Exception as resize_err:
                logger.error(f"Error during corrective resize: {resize_err}", exc_info=True)
                return None, None # Fail if corrective resize fails
        else:
            labels_pred = labels_pred_scaled # Shapes match, use directly
        labels_pred = labels_pred.astype(np.uint16)


    # --- Post-processing: Size Filtering ---
    size_filter_factor = 5.0 # Filter objects smaller than 1/Nth of the largest
    min_absolute_size_threshold = 15 # Minimum pixel count to keep

    logger.info(f"Applying size filtering (factor 1/{size_filter_factor}, min absolute {min_absolute_size_threshold} pixels)...")
    labels_pred_filtered = labels_pred.copy() # Work on a copy

    num_labels_before_filter = labels_pred.max()
    if num_labels_before_filter > 0:
        object_labels, object_sizes = np.unique(labels_pred[labels_pred > 0], return_counts=True)

        if len(object_sizes) > 0:
            max_size = np.max(object_sizes)
            relative_size_threshold = max_size / size_filter_factor
            final_size_threshold = max(relative_size_threshold, min_absolute_size_threshold)

            logger.debug(f"Largest object size: {max_size} pixels.")
            logger.debug(f"Calculated size threshold: {final_size_threshold:.2f} pixels.")

            small_object_labels = object_labels[object_sizes < final_size_threshold]
            num_small_objects = len(small_object_labels)

            if num_small_objects > 0:
                logger.info(f"Removing {num_small_objects} out of {len(object_labels)} objects smaller than threshold.")
                mask_to_remove = np.isin(labels_pred_filtered, small_object_labels)
                labels_pred_filtered[mask_to_remove] = 0
            else:
                logger.debug("No objects found smaller than the threshold.")

            n_objects_after_filter = len(np.unique(labels_pred_filtered[labels_pred_filtered > 0]))
            logger.info(f"Number of objects after size filtering: {n_objects_after_filter}")

        else:
            logger.debug("No valid objects found (label > 0) to calculate size statistics.")
    else:
        logger.info("No objects detected in the initial prediction, skipping size filtering.")

    logger.info("StarDist prediction and filtering complete.")
    # Return filtered labels (original size) and details (from scaled prediction)
    # Note: Centroids in 'details' are relative to the SCALED image used for prediction.
    # The centroids returned by the updated classify_labels_by_dab are relative to the ORIGINAL patch size.
    return labels_pred_filtered, details, n_objects_after_filter


def refine_hotspot_with_stardist(
    candidate_hotspot, slide, stardist_model, hotspot_level,
    actual_pixel_size_um, tumor_cell_mask_l2, debug_dir=None, candidate_index=0,
    dab_threshold=0.15, min_dab_positive_ratio=0.1,
    min_cells=500, max_cells=600, max_iterations=5, resize_factor=0.15
):
    func_name = 'refine_hotspot_with_stardist'
    hotspot = candidate_hotspot.copy()

    size_w, size_h = hotspot['size_level']
    coords_x, coords_y = hotspot['coords_level']
    iteration = 0

        # --- Initialize variables to store results from the "best" or last valid iteration ---
    final_hs_patch_rgb = None
    final_labels_filtered = None # This will be already filtered by tumor_cell_patch
    final_corresponding_tumor_cell_patch = None # The mask patch that aligned with final_hs_patch_rgb
    final_refined_nuclei_count = 0 # Count of nuclei after tumor_cell_patch filtering
    # Keep track of the target FoV sizes for logging/reporting
    current_target_size_w = float(hotspot['size_level'][0])
    current_target_size_h = float(hotspot['size_level'][1])

    iteration = 0
    while iteration < max_iterations:
        # Target size for this iteration's extraction (use current_target_size_w/h)
        target_w_int = int(round(current_target_size_w))
        target_h_int = int(round(current_target_size_h))

        # Prevent zero or negative sizes for extraction
        target_w_int = max(1, target_w_int)
        target_h_int = max(1, target_h_int)

        logger.debug(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: "
                     f"Attempting FoV with target size ({target_w_int}x{target_h_int}) at ({coords_x},{coords_y})")

        hs_patch_rgb = extract_patch_from_slide(
            slide, (coords_x, coords_y), (target_w_int, target_h_int), hotspot_level
        )

        if hs_patch_rgb is None or hs_patch_rgb.size == 0:
            logger.warning(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: Failed to extract RGB patch. Adjusting FoV.")
            # Heuristic: if extraction failed, perhaps FoV is too small/large or problematic
            if current_target_size_w * current_target_size_h < (128*128): # Arbitrary small threshold
                current_target_size_w *= (1 + resize_factor)
                current_target_size_h *= (1 + resize_factor)
            else:
                current_target_size_w *= (1 - resize_factor)
                current_target_size_h *= (1 - resize_factor)
            iteration += 1
            continue # Try next iteration with adjusted FoV

        # Actual dimensions of the extracted RGB patch
        patch_actual_h, patch_actual_w = hs_patch_rgb.shape[:2]
        logger.debug(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: "
                     f"Extracted RGB patch actual size ({patch_actual_w}x{patch_actual_h})")

        extracted_tumor_cell_patch = extract_aligned_mask_patch(
            full_mask=tumor_cell_mask_l2,
            top_left_coords_in_full_mask=(coords_x, coords_y),
            target_patch_shape=(patch_actual_h, patch_actual_w) # Use actual shape of hs_patch_rgb
        )

        # current_labels_filtered is already filtered by extracted_tumor_cell_patch
        # current_n_objects is the count *after* this filtering
        current_labels_filtered, _, current_n_objects = predict_patch_stardist(
            stardist_model, hs_patch_rgb, extracted_tumor_cell_patch, actual_pixel_size_um
        )

        if current_labels_filtered is None: # predict_patch_stardist itself failed
            logger.warning(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: predict_patch_stardist failed. Adjusting FoV.")
            # Heuristic adjustment similar to RGB extraction failure
            if current_target_size_w * current_target_size_h < (128*128):
                current_target_size_w *= (1 + resize_factor)
                current_target_size_h *= (1 + resize_factor)
            else:
                current_target_size_w *= (1 - resize_factor)
                current_target_size_h *= (1 - resize_factor)
            iteration += 1
            continue

        # Store the results of this potentially successful iteration
        final_hs_patch_rgb = hs_patch_rgb
        final_labels_filtered = current_labels_filtered
        final_corresponding_tumor_cell_patch = extracted_tumor_cell_patch
        final_refined_nuclei_count = current_n_objects

        nuclei_count_for_loop_logic = final_refined_nuclei_count # Use the refined count
        logger.info(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: "
                    f"Found {nuclei_count_for_loop_logic} nuclei (within tumor cell mask). Target FoV ({current_target_size_w:.0f}x{current_target_size_h:.0f})")

        if min_cells <= nuclei_count_for_loop_logic <= max_cells:
            logger.info(f"[{func_name}] Cand {candidate_index+1}, Iter {iteration+1}: Refined nuclei count ({nuclei_count_for_loop_logic}) in range. Finalizing FoV.")
            break # Found suitable FoV
        elif nuclei_count_for_loop_logic < min_cells:
            current_target_size_w *= (1 + resize_factor)
            current_target_size_h *= (1 + resize_factor)
        else: # nuclei_count_for_loop_logic > max_cells
            current_target_size_w *= (1 - resize_factor)
            current_target_size_h *= (1 - resize_factor)

        current_target_size_w = np.clip(current_target_size_w, 256, 4096) # Clip target FoV size
        current_target_size_h = np.clip(current_target_size_h, 256, 4096)

        iteration += 1
        if iteration == max_iterations:
            logger.warning(f"[{func_name}] Cand {candidate_index+1}: Max iterations reached. Using results from last valid iteration (Iter {iteration}).")

    if final_hs_patch_rgb is None or final_labels_filtered is None:
        logger.error(f"[{func_name}] Cand {candidate_index+1}: No valid patches or labels after FoV adjustment. Cannot proceed with DAB classification.")
        hotspot.update({
            'stardist_total_count_filtered': 0,
            'stardist_ki67_pos_count': 0,
            'stardist_proliferation_index': 0.0,
            'positive_centroids': [],
            'all_centroids': []
        })
        # Update size_level to the last attempted target FoV for reporting, even if failed
        # and L0 based on that and initial coords.
        # (This assumes current_target_size_w/h reflect the last attempt)
        hotspot['size_level'] = (int(round(current_target_size_w)), int(round(current_target_size_h)))
        downsample_hl = slide.level_downsamples[hotspot_level]
        hotspot['coords_l0'] = (int(coords_x * downsample_hl), int(coords_y * downsample_hl))
        hotspot['size_l0'] = (int(hotspot['size_level'][0] * downsample_hl), int(hotspot['size_level'][1] * downsample_hl))
        return hotspot

    # Update hotspot's size_level to reflect the *actual* dimensions of the patch that was processed
    # and led to final_labels_filtered and final_refined_nuclei_count.
    patch_actual_h_final, patch_actual_w_final = final_hs_patch_rgb.shape[:2]
    hotspot['size_level'] = (patch_actual_w_final, patch_actual_h_final)

    # Recalculate L0 coordinates and size based on initial top-left (coords_x, coords_y)
    # and the *actual* final patch size stored in hotspot['size_level'].
    downsample_hl = slide.level_downsamples[hotspot_level]
    hotspot['coords_l0'] = (int(coords_x * downsample_hl), int(coords_y * downsample_hl))
    hotspot['size_l0'] = (int(hotspot['size_level'][0] * downsample_hl), int(hotspot['size_level'][1] * downsample_hl))


    if final_refined_nuclei_count == 0: # No nuclei found within the tumor cell mask by predict_patch_stardist
        logger.warning(f"[{func_name}] Cand {candidate_index+1}: No nuclei detected within tumor cell mask by predict_patch_stardist. Reporting zero Ki67 counts.")
        hotspot.update({
            'stardist_total_count_filtered': 0, # This is already the refined count
            'stardist_ki67_pos_count': 0,
            'stardist_proliferation_index': 0.0,
            'positive_centroids': [],
            'all_centroids': []
        })
        return hotspot

    # Now, classify these tumor_cell_mask-filtered nuclei for DAB staining.
    # final_labels_filtered already contains only nuclei within tumor_cell_mask regions.
    # final_corresponding_tumor_cell_patch is the aligned mask patch.
    # The `total_nuclei` from this call will be the same as final_refined_nuclei_count.
    classified_labels_dab, dab_pos_count, total_nuclei_for_dab_check, pos_centroids, all_centroids = classify_labels_by_dab(
        final_labels_filtered,                 # Labels already filtered by tumor cell mask
        final_hs_patch_rgb,                    # Corresponding RGB patch
        final_corresponding_tumor_cell_patch, # Aligned tumor cell mask patch (used by classify_labels_by_dab for centroid check)
        dab_threshold,
        min_dab_positive_ratio
    )
    
    # Sanity check: total_nuclei_for_dab_check should ideally match final_refined_nuclei_count
    if total_nuclei_for_dab_check != final_refined_nuclei_count:
        logger.warning(f"[{func_name}] Cand {candidate_index+1}: Mismatch in nuclei counts. "
                       f"predict_patch_stardist refined count: {final_refined_nuclei_count}, "
                       f"classify_labels_by_dab total count: {total_nuclei_for_dab_check}. "
                       f"Using classify_labels_by_dab count for PI.")
    
    # Use the count from classify_labels_by_dab as it's the final authority after its internal checks
    final_total_tumor_cell_nuclei = total_nuclei_for_dab_check


    # Save debug overlay if enabled
    if debug_dir:
        # Use the iteration number from when the loop exited or max_iterations
        debug_iter_num_str = f"iter{iteration}" if iteration <= max_iterations else f"iter{max_iterations}_final"
        hs_debug_dir = os.path.join(debug_dir, f"candidate_{candidate_index+1:02d}_hs_{debug_iter_num_str}")
        os.makedirs(hs_debug_dir, exist_ok=True)
        
        overlay = np.zeros_like(final_hs_patch_rgb)
        if classified_labels_dab is not None and classified_labels_dab.shape == overlay.shape[:2]:
             overlay[classified_labels_dab == 1] = [0, 0, 255]   # Blue for DAB- tumor cells
             overlay[classified_labels_dab == 2] = [0, 255, 0]   # Green for DAB+ tumor cells
        else:
            logger.warning(f"[{func_name}] Cand {candidate_index+1}: classified_labels_dab is None or shape mismatch for debug overlay.")

        vis = cv2.addWeighted(final_hs_patch_rgb, 0.6, overlay, 0.4, 0)
        cv2.imwrite(os.path.join(hs_debug_dir, "classified_overlay_dab.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(hs_debug_dir, "patch_rgb_final.png"), cv2.cvtColor(final_hs_patch_rgb, cv2.COLOR_RGB2BGR))
        if final_corresponding_tumor_cell_patch is not None:
            cv2.imwrite(os.path.join(hs_debug_dir, "tumor_cell_mask_patch_final.png"), final_corresponding_tumor_cell_patch * 255)
        if final_labels_filtered is not None: # Save the mask-filtered labels before DAB
            # For visualization, convert labels to something viewable (e.g., color or scaled grayscale)
            # This is a simple way to visualize the instance segmentation mask
            labels_vis = (final_labels_filtered > 0).astype(np.uint8) * 255 # Binary visualization
            # Or use label2rgb if available and desired:
            # from skimage.color import label2rgb
            # labels_vis = label2rgb(final_labels_filtered, image=final_hs_patch_rgb, bg_label=0, image_alpha=0.3)
            # labels_vis = (labels_vis * 255).astype(np.uint8) # Convert to uint8 for imwrite
            cv2.imwrite(os.path.join(hs_debug_dir, "labels_filtered_by_mask.png"), labels_vis)


    hotspot.update({
        'stardist_total_count_filtered': final_total_tumor_cell_nuclei, # Nuclei within tumor cell mask
        'stardist_ki67_pos_count': dab_pos_count,                   # Of those, how many are DAB+
        'stardist_proliferation_index': dab_pos_count / final_total_tumor_cell_nuclei if final_total_tumor_cell_nuclei > 0 else 0.0,
        'positive_centroids': pos_centroids,
        'all_centroids': all_centroids
    })

    logger.info(
        f"[{func_name}] Candidate {candidate_index+1} (Actual Patch L{hotspot_level} {hotspot['size_level']}): "
        f"Total Tumor Cells={final_total_tumor_cell_nuclei}, "
        f"Ki67+ Tumor Cells={dab_pos_count}, PI={hotspot['stardist_proliferation_index']:.2%}"
    )

    return hotspot