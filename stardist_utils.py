import logging
import numpy as np
import warnings
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import glob
from skimage.color import label2rgb
import imageio
import cv2     
import openslide
import os 
import traceback
from visualization import save_stardist_comparison_plot, format_for_save
from skimage.util import img_as_ubyte, img_as_float
from skimage.transform import resize, rescale
import sys
import time
from stain_utils import get_dab_mask


logger = logging.getLogger(__name__)


def classify_labels_by_dab(labels_filtered, hs_patch_rgb, dab_threshold=0.15, min_dab_positive_ratio=0.1):
    """
    Classify nuclei in a label image as DAB- (1) or DAB+ (2) based on staining.

    Args:
        labels_filtered (ndarray): 2D label image after StarDist filtering.
        hs_patch_rgb (ndarray): Original RGB image patch (H, W, 3).
        dab_threshold (float): OD threshold for DAB mask generation.
        min_dab_positive_ratio (float): Minimum ratio of DAB+ pixels to consider nucleus positive.

    Returns:
        labels_classified (ndarray): 2D array with values 0 (bg), 1 (DAB-), 2 (DAB+).
        ki67_positive_count (int): Count of DAB+ nuclei.
        total_valid_nuclei (int): Total number of filtered nuclei.
    """
    from stain_utils import get_dab_mask  # assumed to be available

    labels_classified = np.zeros_like(labels_filtered, dtype=np.uint8)
    unique_labels = np.unique(labels_filtered)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    ki67_positive_count = 0

    for label in unique_labels:
        mask = labels_filtered == label
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue

        # Get bounding box of this nucleus
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1
        nucleus_patch = hs_patch_rgb[y_min:y_max, x_min:x_max]
        nucleus_mask = mask[y_min:y_max, x_min:x_max]

        # Get DAB+ mask and apply it only on the nucleus area
        try:
            dab_mask = get_dab_mask(nucleus_patch, dab_threshold)
            dab_positive_ratio = (dab_mask & nucleus_mask).sum() / nucleus_mask.sum()

            if dab_positive_ratio > min_dab_positive_ratio:
                labels_classified[mask] = 2  # DAB+
                ki67_positive_count += 1
            else:
                labels_classified[mask] = 1  # DAB-
        except Exception as e:
            # Skip this nucleus if error occurs
            print(f"[classify_labels_by_dab] Error on label {label}: {e}")
            continue

    total_valid_nuclei = len(unique_labels)
    return labels_classified, ki67_positive_count, total_valid_nuclei


import logging
import numpy as np
import warnings
from stardist.models import StarDist2D
from csbdeep.utils import normalize
# Removed: import glob (unused)
# Removed: from skimage.color import label2rgb (unused)
# Removed: import imageio (unused)
import cv2 # Still needed for resize? Check if labels_pred = resize() is used
import openslide # Keep if needed elsewhere, but not directly used here
import os # Keep for path manipulation if needed, or os.makedirs if errors logged to files
import traceback
# Removed: from visualization import save_stardist_comparison_plot, format_for_save (unused)
from skimage.util import img_as_ubyte, img_as_float
from skimage.transform import resize, rescale
import sys
# Removed: import time (unused)
# Removed: from stain_utils import get_dab_mask (unused in this function)

logger = logging.getLogger(__name__)

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
    probability_threshold = 0.5
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
    size_filter_factor = 7.0 # Filter objects smaller than 1/Nth of the largest
    min_absolute_size_threshold = 10 # Minimum pixel count to keep

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
        # labels_pred_filtered is already a copy of labels_pred (likely all zeros)

    # --- Return Results ---
    # The details dictionary contains useful info like centroids ('points') from the *scaled* prediction
    # Ensure centroids are potentially scaled back if needed by the calling function,
    # although often labels_filtered is used directly.
    logger.info("StarDist prediction and filtering complete.")
    return labels_pred_filtered, details


def refine_hotspot_with_stardist(
    candidate_hotspot: dict,
    stardist_model,
    slide: openslide.OpenSlide,
    hotspot_level: int,
    actual_pixel_size_um: float,
    debug_dir: str = None,
    candidate_index: int = 0,
    dab_threshold: float = 0.15
) -> dict | None:
    func_name = 'refine_hotspot_with_stardist'
    hotspot = candidate_hotspot.copy()

    if not all(k in hotspot for k in ['coords_level', 'size_level']):
        logger.error(f"[{func_name}] Candidate {candidate_index+1} missing required keys.")
        return None
    if slide is None:
        logger.error(f"[{func_name}] OpenSlide object is None for candidate {candidate_index+1}.")
        return None

    try:
        # Read image patch
        downsample = slide.level_downsamples[hotspot_level]
        x_l2, y_l2 = hotspot['coords_level']
        w_l2, h_l2 = hotspot['size_level']
        x_l0, y_l0 = int(x_l2 * downsample), int(y_l2 * downsample)

        hs_patch_pil = slide.read_region((x_l0, y_l0), hotspot_level, (w_l2, h_l2)).convert("RGB")
        hs_patch_rgb = np.array(hs_patch_pil)
        if hs_patch_rgb.shape[:2] != (h_l2, w_l2):
            hs_patch_rgb = cv2.resize(hs_patch_rgb, (w_l2, h_l2), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        logger.error(f"[{func_name}] Failed to read patch: {e}", exc_info=True)
        return None

    # Prepare debug folder
    hs_debug_dir = None
    if debug_dir:
        try:
            hs_debug_dir = os.path.join(debug_dir, f"candidate_{candidate_index+1:02d}_hs")
            os.makedirs(hs_debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(hs_debug_dir, "patch_rgb.png"), cv2.cvtColor(hs_patch_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.warning(f"[{func_name}] Could not create debug folder: {e}")
            hs_debug_dir = None

    # Run StarDist
    labels_filtered, details = predict_patch_stardist(
        model=stardist_model,
        image_patch_rgb=hs_patch_rgb,
        actual_pixel_size_um=actual_pixel_size_um
    )

    if labels_filtered is None or labels_filtered.max() == 0:
        logger.warning(f"[{func_name}] No nuclei found after filtering.")
        return None

    # Classify nuclei into DAB- (1) and DAB+ (2)
    classified_labels, dab_pos_count, total_nuclei = classify_labels_by_dab(
        labels_filtered, hs_patch_rgb, dab_threshold=dab_threshold
    )

    # Save classified label overlay
    if hs_debug_dir:
        try:
            overlay = np.zeros_like(hs_patch_rgb)
            overlay[classified_labels == 1] = [0, 0, 255]   # Blue for DAB- 
            overlay[classified_labels == 2] = [0, 255, 0]   # Green for DAB+
            vis = cv2.addWeighted(hs_patch_rgb, 0.6, overlay, 0.4, 0)
            cv2.imwrite(os.path.join(hs_debug_dir, "classified_overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.warning(f"[{func_name}] Failed to save classification overlay: {e}")

    # Update hotspot results
    hotspot['stardist_total_count_filtered'] = total_nuclei
    hotspot['stardist_ki67_pos_count'] = dab_pos_count
    hotspot['stardist_proliferation_index'] = (
        dab_pos_count / total_nuclei if total_nuclei > 0 else 0.0
    )

    logger.info(
        f"[{func_name}] Candidate {candidate_index+1}: Total={total_nuclei}, "
        f"Ki67+={dab_pos_count}, PI={hotspot['stardist_proliferation_index']:.2%}"
    )

    return hotspot
