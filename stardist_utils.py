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
from skimage.measure import regionprops # <-- Import needed for centroids
import sys
# import time # Not used

# Assumed to be in stain_utils.py and imported correctly elsewhere when needed
# from stain_utils import get_dab_mask

logger = logging.getLogger(__name__)

def extract_patch_from_slide(slide, coords_level, size_level, hotspot_level):
    downsample = slide.level_downsamples[hotspot_level]
    x_l0, y_l0 = int(coords_level[0] * downsample), int(coords_level[1] * downsample)
    hs_patch_pil = slide.read_region((x_l0, y_l0), hotspot_level, size_level).convert("RGB")
    hs_patch_rgb = np.array(hs_patch_pil)
    return hs_patch_rgb


def classify_labels_by_dab(labels_filtered, hs_patch_rgb, dab_threshold=0.15, min_dab_positive_ratio=0.1):
    """
    Classify nuclei in a label image as DAB- (1) or DAB+ (2) based on staining,
    and return counts and centroid coordinates.

    Args:
        labels_filtered (ndarray): 2D label image after StarDist filtering (uint16).
        hs_patch_rgb (ndarray): Original RGB image patch (H, W, 3).
        dab_threshold (float): OD threshold for DAB mask generation.
        min_dab_positive_ratio (float): Minimum ratio of DAB+ pixels to consider nucleus positive.

    Returns:
        tuple: Contains:
            - labels_classified (ndarray): 2D array with 0 (bg), 1 (DAB-), 2 (DAB+).
            - ki67_positive_count (int): Count of DAB+ nuclei.
            - total_valid_nuclei (int): Total number of filtered nuclei processed.
            - positive_centroids (list): List of (y, x) coordinates for DAB+ nuclei centroids.
            - all_valid_centroids (list): List of (y, x) coordinates for ALL valid nuclei centroids.
    """
    # Import locally if not globally available or causes circular dependencies
    try:
        from stain_utils import get_dab_mask
    except ImportError:
        logger.error("Could not import get_dab_mask from stain_utils in classify_labels_by_dab")
        # Return empty/default values to indicate failure
        shape = labels_filtered.shape
        return np.zeros(shape, dtype=np.uint8), 0, 0, [], []

    labels_classified = np.zeros_like(labels_filtered, dtype=np.uint8)
    positive_centroids = []
    all_valid_centroids = []
    ki67_positive_count = 0
    total_valid_nuclei = 0

    # Use regionprops to get labels and centroids directly from the filtered labels
    # Ensure labels_filtered is integer type for regionprops
    if not np.issubdtype(labels_filtered.dtype, np.integer):
        logger.warning("labels_filtered dtype is not integer, attempting conversion for regionprops.")
        try:
            labels_filtered_int = labels_filtered.astype(np.uint16)
        except ValueError:
            logger.error("Could not convert labels_filtered to integer type.")
            return np.zeros_like(labels_filtered, dtype=np.uint8), 0, 0, [], []
    else:
        labels_filtered_int = labels_filtered

    try:
        props = regionprops(labels_filtered_int) # Ignores label 0 automatically
    except TypeError as e_props:
         logger.error(f"TypeError during regionprops (check labels_filtered dtype?): {e_props}")
         return np.zeros_like(labels_filtered, dtype=np.uint8), 0, 0, [], []


    for prop in props:
        label = prop.label
        # Centroid coordinates (y, x) relative to patch
        # Ensure centroid is finite (can be NaN for empty regions? although props shouldn't return them)
        if not np.all(np.isfinite(prop.centroid)):
             logger.warning(f"Non-finite centroid for label {label}. Skipping.")
             continue
        centroid_y, centroid_x = prop.centroid
        all_valid_centroids.append((centroid_y, centroid_x))
        total_valid_nuclei += 1 # Count every valid property found

        # Get bounding box and mask for this nucleus
        min_r, min_c, max_r, max_c = prop.bbox
        # Ensure bbox indices are valid
        min_r, min_c = max(0, min_r), max(0, min_c)
        max_r, max_c = min(labels_filtered_int.shape[0], max_r), min(labels_filtered_int.shape[1], max_c)
        # Check if bbox has valid size AFTER clipping
        if min_r >= max_r or min_c >= max_c:
             logger.warning(f"Invalid bounding box for label {label} after clipping. Skipping.")
             # Mark as DAB- by default if cannot process
             labels_classified[labels_filtered_int == label] = 1
             continue

        nucleus_patch_rgb_bbox = hs_patch_rgb[min_r:max_r, min_c:max_c]
        # Create mask relative to the patch bbox
        nucleus_mask_in_bbox = (labels_filtered_int[min_r:max_r, min_c:max_c] == label)

        # Additional check: Ensure patch and mask are not empty after slicing
        if nucleus_patch_rgb_bbox.size == 0 or nucleus_mask_in_bbox.size == 0 or np.sum(nucleus_mask_in_bbox) == 0:
            logger.warning(f"Empty patch or mask for label {label} within bbox. Skipping.")
            labels_classified[labels_filtered_int == label] = 1 # Mark as DAB-
            continue

        # Get DAB+ mask and apply it only on the nucleus area
        try:
            # Ensure patch has 3 dimensions for get_dab_mask
            if nucleus_patch_rgb_bbox.ndim == 2:
                # Handle grayscale case if it occurs, although input hs_patch_rgb should be 3D
                 logger.warning(f"Nucleus patch for label {label} is 2D, attempting stack.")
                 nucleus_patch_rgb_bbox = np.stack([nucleus_patch_rgb_bbox]*3, axis=-1)
            elif nucleus_patch_rgb_bbox.shape[-1] != 3:
                 logger.warning(f"Nucleus patch for label {label} has {nucleus_patch_rgb_bbox.shape[-1]} channels, expected 3. Skipping.")
                 labels_classified[labels_filtered_int == label] = 1 # Mark as DAB-
                 continue


            dab_mask_in_bbox = get_dab_mask(nucleus_patch_rgb_bbox, dab_threshold)

            # Ensure dab_mask is 2D and matches nucleus_mask_in_bbox shape
            if dab_mask_in_bbox is None:
                 logger.warning(f"get_dab_mask returned None for label {label}. Skipping positivity check.")
                 labels_classified[labels_filtered_int == label] = 1 # Mark as DAB-
                 continue
            if dab_mask_in_bbox.shape != nucleus_mask_in_bbox.shape:
                 logger.warning(f"DAB mask shape {dab_mask_in_bbox.shape} mismatch with nucleus mask shape {nucleus_mask_in_bbox.shape} for label {label}. Skipping positivity check.")
                 labels_classified[labels_filtered_int == label] = 1 # Mark as DAB-
                 continue

            # Calculate positive ratio within the nucleus mask
            # Make sure masks are boolean for bitwise AND
            dab_positive_pixels_in_nucleus = (dab_mask_in_bbox.astype(bool) & nucleus_mask_in_bbox).sum()
            total_nucleus_pixels = nucleus_mask_in_bbox.sum() # Use sum of boolean mask
            dab_positive_ratio = dab_positive_pixels_in_nucleus / total_nucleus_pixels if total_nucleus_pixels > 0 else 0

            # --- Classification ---
            # Get the mask in the full labels_filtered image to update labels_classified
            nucleus_mask_global = (labels_filtered_int == label)
            if dab_positive_ratio > min_dab_positive_ratio:
                labels_classified[nucleus_mask_global] = 2  # DAB+
                ki67_positive_count += 1
                positive_centroids.append((centroid_y, centroid_x)) # Add centroid if positive
            else:
                labels_classified[nucleus_mask_global] = 1  # DAB-

        except Exception as e:
            # Skip this nucleus if error occurs
            logger.error(f"Error processing label {label} in classify_labels_by_dab: {e}", exc_info=True)
            # Mark as DAB- on error
            nucleus_mask_global = (labels_filtered_int == label)
            if np.any(nucleus_mask_global): # Check if mask exists before assigning
                 labels_classified[nucleus_mask_global] = 1
            continue # Skip to next property

    # Final count should reflect successfully processed nuclei if needed, but total_valid_nuclei counts all props found.
    logger.debug(f"Classified nuclei: {total_valid_nuclei} total, {ki67_positive_count} positive.")
    return labels_classified, ki67_positive_count, total_valid_nuclei, positive_centroids, all_valid_centroids


# --- predict_patch_stardist function remains the same as the refactored version ---
# (Using logging, no image saving)
def predict_patch_stardist(model, image_patch_rgb, actual_pixel_size_um):
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
    rescale_factor = 1.0 # Default if no rescaling needed

    if actual_pixel_size_um is None or target_pixel_size is None:
        logger.warning("Pixel size info missing. Skipping rescaling.")
    elif abs(actual_pixel_size_um - target_pixel_size) < 1e-6:
        logger.debug(f"Actual pixel size ({actual_pixel_size_um:.3f}) matches target ({target_pixel_size:.3f}). No rescaling needed.")
    else:
        rescale_factor = actual_pixel_size_um / target_pixel_size
        logger.info(f"Rescaling prediction input image by factor {rescale_factor:.4f} (from {actual_pixel_size_um:.3f} to {target_pixel_size:.3f} um/pix)...")
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
        except Exception as e:
            logger.error(f"Error during image rescaling: {e}", exc_info=True)
            return None, None # Cannot proceed without correctly scaled image

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
        return None, None
    if labels_pred_scaled.ndim != 2:
        logger.error(f"Predicted labels have unexpected dimensions ({labels_pred_scaled.ndim}). Expected 2. Cannot proceed.")
        return None, None

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
    actual_pixel_size_um, debug_dir=None, candidate_index=0,
    dab_threshold=0.15, min_dab_positive_ratio=0.1,
    min_cells=500, max_cells=600, max_iterations=5, resize_factor=0.15
):
    func_name = 'refine_hotspot_with_stardist'
    hotspot = candidate_hotspot.copy()

    size_w, size_h = hotspot['size_level']
    coords_x, coords_y = hotspot['coords_level']
    iteration = 0

    while iteration < max_iterations:
        hs_patch_rgb = extract_patch_from_slide(
            slide, (coords_x, coords_y), (int(size_w), int(size_h)), hotspot_level
        )

        labels_filtered, _, n_objects_after_filter = predict_patch_stardist(
            stardist_model, hs_patch_rgb, actual_pixel_size_um
        )

        if labels_filtered is None:
            logger.warning("StarDist failed, aborting refinement.")
            return None

        nuclei_count = n_objects_after_filter
        logger.info(f"Iteration {iteration+1}: Detected {nuclei_count} nuclei.")

        if min_cells <= nuclei_count <= max_cells:
            break
        elif nuclei_count < min_cells:
            size_w *= (1 + resize_factor)
            size_h *= (1 + resize_factor)
        else:
            size_w *= (1 - resize_factor)
            size_h *= (1 - resize_factor)

        size_w = np.clip(size_w, 256, 4096)
        size_h = np.clip(size_h, 256, 4096)

        iteration += 1

    hotspot['size_level'] = (int(size_w), int(size_h))

    if labels_filtered is None or n_objects_after_filter == 0:
        logger.warning("No nuclei detected after resizing.")
        hotspot.update({
            'stardist_total_count_filtered': 0,
            'stardist_ki67_pos_count': 0,
            'stardist_proliferation_index': 0.0,
            'positive_centroids': [],
            'all_centroids': []
        })
        return hotspot

    classified_labels, dab_pos_count, total_nuclei, pos_centroids, all_centroids = classify_labels_by_dab(
        labels_filtered, hs_patch_rgb, dab_threshold, min_dab_positive_ratio
    )

    # Save debug overlay if enabled
    if debug_dir:
        hs_debug_dir = os.path.join(debug_dir, f"candidate_{candidate_index+1:02d}_hs")
        os.makedirs(hs_debug_dir, exist_ok=True)
        overlay = np.zeros_like(hs_patch_rgb)
        overlay[classified_labels == 1] = [0, 0, 255]   # Blue for DAB- (Ki67-)
        overlay[classified_labels == 2] = [0, 255, 0]   # Green for DAB+ (Ki67+)
        vis = cv2.addWeighted(hs_patch_rgb, 0.6, overlay, 0.4, 0)
        cv2.imwrite(os.path.join(hs_debug_dir, "classified_overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(hs_debug_dir, "patch_rgb.png"), cv2.cvtColor(hs_patch_rgb, cv2.COLOR_RGB2BGR))

    hotspot.update({
        'stardist_total_count_filtered': total_nuclei,
        'stardist_ki67_pos_count': dab_pos_count,
        'stardist_proliferation_index': dab_pos_count / total_nuclei if total_nuclei else 0.0,
        'positive_centroids': pos_centroids,
        'all_centroids': all_centroids
    })

    logger.info(
        f"[{func_name}] Candidate {candidate_index+1}: Total={total_nuclei}, "
        f"Ki67+={dab_pos_count}, PI={hotspot['stardist_proliferation_index']:.2%}"
    )

    return hotspot