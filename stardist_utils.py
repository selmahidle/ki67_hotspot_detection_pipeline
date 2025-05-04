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


logger = logging.getLogger(__name__)


def predict_patch_stardist(model, image_patch_rgb, actual_pixel_size_um):
    """
    Runs StarDist prediction on a single RGB image patch using predefined default
    parameters matching the reference script, unless overridden. Requires the
    actual pixel size of the input patch.

    Args:
        model: Loaded StarDist2D model.
        image_patch_rgb: Input RGB image patch (H, W, 3).
        actual_pixel_size_um: Actual pixel size of the input patch (microns).
        target_pixel_size_um: Target pixel size for rescaling. Defaults match script.
        prob_thresh: Probability threshold. Defaults match script.
        nms_thresh: NMS threshold. Defaults match script.
        normalize_perc_low: Normalization percentile. Defaults match script.
        normalize_perc_high: Normalization percentile. Defaults match script.
        apply_size_filter: Apply size filtering post-prediction. Defaults to True.
        size_filter_factor: Relative size filter factor. Defaults match script.
        min_absolute_size: Absolute min object size. Defaults match script.

    Returns:
        tuple: (labels_filtered, details) containing the filtered label image and
               the details dictionary from predict_instances.
               Returns (None, None) on error.
    """

    save_folder = "/cluster/home/selmahi/stardist_fix_for_pipeline_running_main_outputs"
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving outputs to: {save_folder}")


    # --- Define Prediction Parameters (Mapped from QuPath) ---
    probability_threshold = 0.5 # QuPath: Detection probability threshold 0.5 fra før
    nms_overlap_threshold = 0.3 
    target_pixel_size = 0.2      # QuPath: Requested pixel size (µm) 0.3 fra før

    # ================== IMPORTANT ==================
    #  You MUST set this value to the actual pixel size of YOUR images in MICRONS
    #  If your images are already at the target_pixel_size, set this to target_pixel_size
    ACTUAL_PIXEL_SIZE_MICRONS = 1.0 # <<<--- REPLACE THIS VALUE: e.g., 0.25 if original is 0.25 um/pix 0.23 hvis heile slides
    # ===============================================

    print(f"\nProcessing file for stardist")
    print(f"Loaded hotspot shape: {image_patch_rgb.shape}, dtype: {image_patch_rgb.dtype}")
    img_to_process = img_as_float(image_patch_rgb)

    if ACTUAL_PIXEL_SIZE_MICRONS is None or target_pixel_size is None:
        print("  Pixel size info missing. Skipping rescaling.")
    elif abs(ACTUAL_PIXEL_SIZE_MICRONS - target_pixel_size) < 1e-6:
        print(f"  Actual pixel size ({ACTUAL_PIXEL_SIZE_MICRONS:.3f}) matches target ({target_pixel_size:.3f}). No rescaling needed.")
    else:
        rescale_factor = ACTUAL_PIXEL_SIZE_MICRONS / target_pixel_size
        print(f"Rescaling prediction input image by factor {rescale_factor:.4f} (from {ACTUAL_PIXEL_SIZE_MICRONS:.3f} to {target_pixel_size:.3f} um/pix)...")
        start_time = time.time()

        img_to_process = rescale(
            img_to_process,
            rescale_factor,
            anti_aliasing=True,
            mode='reflect',
            preserve_range=True,
            channel_axis=-1  # Explicitly tell rescale the last axis is channels
        )

        img_to_process = np.clip(img_to_process, 0, 1) # Ensure range after rescale
    img_norm = normalize(img_to_process, 1.0, 99.0, axis=None if img_to_process.ndim == 2 else (0, 1))

    labels_pred_scaled, details = model.predict_instances(
        img_norm,
        prob_thresh=probability_threshold,
        nms_thresh=nms_overlap_threshold,
        scale=None, 
        return_predict=False
    )
    n_objects = labels_pred_scaled.max()
    print(f"Found n_objects = {n_objects} with prob_thresh = {probability_threshold}")

    if not isinstance(labels_pred_scaled, np.ndarray):
        print(f"Error: Predicted labels are not a numpy array (type: {type(labels_pred_scaled)}). Skipping file.", file=sys.stderr)
        return
    if labels_pred_scaled.ndim != 2:
        print(f"Error: Predicted labels have unexpected dimensions ({labels_pred_scaled.ndim}). Skipping file.", file=sys.stderr)
        return

    n_objects = labels_pred_scaled.max()

    # --- Resize Label Mask Back to Original Image Size ---
    labels_pred = None # Initialize
    # Target shape should match the original image dimensions before any rescaling
    original_shape = image_patch_rgb.shape[:2] # Use the shape of the initial RGB image


    if abs(rescale_factor - 1.0) > 1e-6:
        labels_pred = resize(
            labels_pred_scaled,
            output_shape=original_shape, # Target shape
            order=0,                     # Nearest neighbor interpolation for labels
            preserve_range=True,         # Keep original label values (important!)
            anti_aliasing=False          # No anti-aliasing for labels
        )
        labels_pred = labels_pred.astype(np.uint16)
    else:
        print("No rescaling was applied, using predicted labels directly.")
        # Ensure shape matches original image shape even if no rescale factor was applied
        # (e.g., if normalization somehow changed shape subtly, though unlikely)
        if labels_pred_scaled.shape != original_shape:
            print(f"Warning: Predicted label shape {labels_pred_scaled.shape} differs from original image shape {original_shape} even without rescaling. Attempting resize.", file=sys.stderr)
            try:
                labels_pred = resize(labels_pred_scaled, output_shape=original_shape, order=0, preserve_range=True, anti_aliasing=False)
            except Exception as resize_err:
                print(f"Error during corrective resize: {resize_err}. Skipping visualization/save.", file=sys.stderr)
            return
        else:
            labels_pred = labels_pred_scaled
        labels_pred = labels_pred.astype(np.uint16)


    # Post processing to remove noisy small
    size_filter_factor = 7.0 # Filter objects smaller than 1/7th of the largest. Adjust as needed (e.g., 5.0 for 1/5th)
    min_absolute_size_threshold = 10 # Optional: Set a minimum pixel count to keep, regardless of relative size

    print(f"Applying size filtering (factor 1/{size_filter_factor}, min absolute {min_absolute_size_threshold} pixels)...")
    labels_pred_filtered = labels_pred.copy() # Work on a copy

    if labels_pred.max() > 0: # Proceed only if objects were detected
        # Calculate size (pixel count) for each labeled object (excluding background label 0)
        object_labels, object_sizes = np.unique(labels_pred[labels_pred > 0], return_counts=True)

        if len(object_sizes) > 0:
            max_size = np.max(object_sizes)
            # Calculate the size threshold based on the largest object
            relative_size_threshold = max_size / size_filter_factor
            # Use the larger of the relative threshold and the absolute minimum threshold
            final_size_threshold = max(relative_size_threshold, min_absolute_size_threshold)

            print(f"    Largest object size: {max_size} pixels.")
            print(f"    Calculated size threshold: {final_size_threshold:.2f} pixels.")

            # Identify labels of objects smaller than the threshold
            small_object_labels = object_labels[object_sizes < final_size_threshold]
            num_objects_before = len(object_labels)
            num_small_objects = len(small_object_labels)

            if num_small_objects > 0:
                print(f"    Removing {num_small_objects} out of {num_objects_before} objects smaller than the threshold.")
                # Set pixels belonging to small objects to background (0)
                mask_to_remove = np.isin(labels_pred_filtered, small_object_labels)
                labels_pred_filtered[mask_to_remove] = 0
                # Optional: Relabel sequentially if needed for downstream tasks, but often not necessary for visualization
                # labels_pred_filtered, _, _ = segmentation.relabel_sequential(labels_pred_filtered)
            else:
                print("    No objects found smaller than the threshold.")

            # Update object count after filtering for logging
            # Update object count after filtering for logging
            remaining_labels = np.unique(labels_pred_filtered[labels_pred_filtered > 0])
            n_objects_after_filter = len(remaining_labels) # Count unique labels > 0
            print(f"    Number of objects after size filtering: {n_objects_after_filter}") # This will now report the correct count (likely 63 in your example)

        else:
            print("    No objects found to calculate size statistics.")
            # Keep n_objects as calculated before filtering if no positive labels were found initially
    else:
        print("    No objects detected in the initial prediction, skipping size filtering.")
        # labels_pred_filtered is already a copy of labels_pred (which is likely all zeros)

    # IMPORTANT: Use the filtered labels for visualization from now on
    labels_for_visualization = labels_pred_filtered

    # --- Prepare Original RGB Image for Display ---
    original_rgb_display_ubyte = format_for_save(image_patch_rgb)

    # --- Prepare Predicted Overlay for Display ---
    if labels_for_visualization is not None and isinstance(labels_for_visualization, np.ndarray) and labels_for_visualization.ndim == 2:
        # Ensure labels_pred shape matches original image shape
        if labels_for_visualization.shape != original_shape:
            print(f"Error: Final predicted label shape {labels_for_visualization.shape} unexpectedly differs from original image shape {original_shape} before overlay. Skipping overlay.", file=sys.stderr)
            labels_overlay_ubyte = np.zeros_like(original_rgb_display_ubyte) # Use shape from original RGB display
        else:
            print(f"  Visualizing predicted mask overlay (max label: {labels_for_visualization.max()})...")
            labels_rgb_overlay = label2rgb(
                labels_for_visualization,
                image=image_patch_rgb, 
                bg_label=0,
                bg_color=None, # Use image background
                kind='overlay',
                image_alpha=0.5, # Transparency of original image
                alpha=0.5 # Transparency of label color
            )
            labels_overlay_ubyte = format_for_save(labels_rgb_overlay)
    else:
        print("Error: Predicted labels ('labels_for_visualization') are invalid or None. Creating black placeholder.", file=sys.stderr)
        labels_overlay_ubyte = np.zeros_like(original_rgb_display_ubyte)

    valid_components = True
    components = [original_rgb_display_ubyte, labels_overlay_ubyte]
    component_names = ["Original RGB", "Prediction (Overlay)"]

    h = components[0].shape[0]
    w = components[0].shape[1]
    shapes_match = True
    for i, comp in enumerate(components): # Check all components now
        if comp.shape[0] != h or comp.shape[1] != w:
            print(f"Warning: Shape mismatch before stacking! Target is {h}x{w}, "
                f"but '{component_names[i]}' is {comp.shape[0]}x{comp.shape[1]}. Skipping save.", file=sys.stderr)
            shapes_match = False
            break
        if comp.shape[2] != 3: # Should be caught above, but double-check channels
            print(f"Warning: Channel mismatch for '{component_names[i]}' ({comp.shape[2]} channels). Expected 3. Skipping save.", file=sys.stderr)
            shapes_match = False
            break

    try:
        row_img = np.hstack(components)
    except ValueError as stack_err:
            print(f"Error during np.hstack: {stack_err}. Shapes were: {[c.shape for c in components]}. Skipping save.", file=sys.stderr)
            return

    basename = f"patch_{int(time.time() * 1000)}"
    save_path = os.path.join(save_folder, f"{basename}_stardist_fix.jpg") 

    try:
        imageio.imwrite(save_path, row_img, quality=95)
    except Exception as save_err:
        print(f"Error saving image {save_path}: {save_err}", file=sys.stderr)
    except FileNotFoundError as fnf_e:
        print(f"Error: File not found during processing loop: {fnf_e}", file=sys.stderr)
        return 
    except Exception as e:
        print(f"Error: An unexpected error occurred during save: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    
    print(f"--- File fisished processing ---")

    return labels_pred_filtered, details, labels_pred_filtered.shape


def refine_hotspot_with_stardist(
    candidate_hotspot: dict,
    stardist_model,
    slide: openslide.OpenSlide,
    dab_plus_mask_l2: np.ndarray,
    cell_mask_binary_l2: np.ndarray,
    hotspot_level: int,
    actual_pixel_size_um: float = 1.0, # Use constant as default
    debug_dir: str = None,
    candidate_index: int = 0 # For debug filenames
) -> dict | None:
    """
    Performs StarDist prediction on a candidate hotspot patch and calculates
    refined cell counts based on DAB+ and SMP Cell masks.

    Args:
        candidate_hotspot: Dictionary describing the candidate hotspot (coords, size).
                           Must contain 'coords_level', 'size_level'.
        stardist_model: The loaded StarDist model.
        slide: The OpenSlide object for reading the patch.
        dab_plus_mask_l2: The full DAB+ mask at hotspot_level resolution.
        cell_mask_binary_l2: The full SMP Cell mask at hotspot_level resolution.
        hotspot_level: The WSI level the masks and analysis correspond to.
        actual_pixel_size_um: Pixel size (microns per pixel) for StarDist prediction scaling.
                               Crucial for correct results.
        debug_dir: Optional path to save debug outputs for this specific candidate.
                   If provided, a subdirectory named 'candidate_XX_hs' will be created.
        candidate_index: Optional index for naming debug files (0-based).

    Returns:
        The updated candidate_hotspot dictionary with added StarDist count keys:
        - 'stardist_total_count_filtered': Total nuclei after size filter.
        - 'stardist_smp_cell_count': Nuclei centroids within SMP mask.
        - 'stardist_dab_smp_cell_count': Nuclei centroids within BOTH DAB+ and SMP masks.
        Returns None if refinement fails for this candidate.
    """
    func_name = 'refine_hotspot_with_stardist'
    hotspot = candidate_hotspot.copy() # Work on a copy to avoid modifying original dict

    # --- Input Validation ---
    if not all(k in hotspot for k in ['coords_level', 'size_level']):
         logger.error(f"[{func_name}] Candidate {candidate_index+1} dictionary missing required keys ('coords_level', 'size_level').")
         return None
    if  slide is None or dab_plus_mask_l2 is None or cell_mask_binary_l2 is None:
         logger.error(f"[{func_name}] Missing required input model or mask for candidate {candidate_index+1}.")
         return None
    if actual_pixel_size_um <= 0:
        logger.error(f"[{func_name}] Invalid actual_pixel_size_um: {actual_pixel_size_um} for candidate {candidate_index+1}.")
        return None

    try:
        # --- Get Coordinates and Read Patch ---
        level_h_full, level_w_full = cell_mask_binary_l2.shape[:2] # Get dimensions from a full L2 mask
        downsample_l2 = slide.level_downsamples[hotspot_level]

        x_l2, y_l2 = hotspot['coords_level']
        w_l2, h_l2 = hotspot['size_level']

        # Clip coordinates and dimensions to ensure they are within the bounds of the level
        x_l2 = max(0, x_l2); y_l2 = max(0, y_l2) # Ensure non-negative start
        w_l2 = min(w_l2, level_w_full - x_l2) # Adjust width based on start and level width
        h_l2 = min(h_l2, level_h_full - y_l2) # Adjust height based on start and level height

        if w_l2 <= 0 or h_l2 <= 0:
            logger.warning(f"[{func_name}] Candidate {candidate_index+1} has invalid dimensions ({w_l2}x{h_l2}) after clipping to level bounds ({level_w_full}x{level_h_full}). Skipping.")
            return None

        logger.debug(f"[{func_name}] Candidate {candidate_index+1}: L{hotspot_level} coords=({x_l2},{y_l2}), size=({w_l2},{h_l2})")

        hs_patch_rgb = None
        try:
            # Calculate Level 0 coordinates for reading
            x_l0 = int(x_l2 * downsample_l2); y_l0 = int(y_l2 * downsample_l2)
            # Read region expects (location_l0_top_left, level_to_read_from, size_at_that_level)
            hs_patch_pil = slide.read_region((x_l0, y_l0), hotspot_level, (w_l2, h_l2)).convert('RGB')
            hs_patch_rgb = np.array(hs_patch_pil)

            # Ensure read patch matches expected size (can sometimes differ slightly due to rounding)
            if hs_patch_rgb.shape[0] != h_l2 or hs_patch_rgb.shape[1] != w_l2:
                logger.warning(f"[{func_name}] Read patch size {hs_patch_rgb.shape[:2]} != expected ({h_l2},{w_l2}). Resizing.")
                hs_patch_rgb = cv2.resize(hs_patch_rgb, (w_l2, h_l2), interpolation=cv2.INTER_LINEAR)
        except openslide.OpenSlideError as ose:
             logger.error(f"[{func_name}] OpenSlideError reading patch for candidate {candidate_index+1} at L{hotspot_level}({x_l2},{y_l2}, {w_l2}x{h_l2}), L0=({x_l0},{y_l0}): {ose}")
             return None
        except Exception as e:
            logger.error(f"[{func_name}] Failed reading patch for candidate {candidate_index+1}: {e}", exc_info=True)
            return None

        # --- Prepare Debug Directory ---
        hs_debug_dir = None
        if debug_dir:
            try:
                # Create a specific subdirectory for this candidate's debug files
                hs_debug_dir = os.path.join(debug_dir, f"candidate_{candidate_index+1:02d}_hs")
                os.makedirs(hs_debug_dir, exist_ok=True)
                # Save the raw patch if successfully read
                if hs_patch_rgb is not None:
                     cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{candidate_index+1:02d}_patch_L{hotspot_level}.png"), cv2.cvtColor(hs_patch_rgb, cv2.COLOR_RGB2BGR))
            except Exception as e_mkdir:
                 logger.error(f"[{func_name}] Failed to create debug directory {hs_debug_dir}: {e_mkdir}. Debugging disabled for this candidate.")
                 hs_debug_dir = None # Disable further debug saving

        # --- Run StarDist Prediction ---
        labels_filtered, details, label_shape = predict_patch_stardist(stardist_model, hs_patch_rgb, actual_pixel_size_um)

        # --- Initialize Counts ---
        hotspot['stardist_total_count'] = 0
        hotspot['stardist_total_count_filtered'] = 0
        hotspot['stardist_smp_cell_count'] = 0         # Denominator for Ki67%
        hotspot['stardist_dab_smp_cell_count'] = 0     # Numerator & Ranking Score


        # --- Get Centroids and Corresponding Masks ---
        centroids = details.get('points') # StarDist 'points' are centroids (y, x)
        valid_centroids = isinstance(centroids, np.ndarray) and centroids.ndim == 2 and centroids.shape[1] == 2

        if valid_centroids and len(centroids) > 0:
            hotspot['stardist_total_count'] = len(centroids) # Total predicted nuclei (centroids from StarDist output)

            # Extract corresponding mask patches from the *full* L2 masks
            # Use the clipped coordinates (x_l2, y_l2, w_l2, h_l2)
            dab_patch_orig = dab_plus_mask_l2[y_l2 : y_l2 + h_l2, x_l2 : x_l2 + w_l2]
            cell_patch_smp_orig = cell_mask_binary_l2[y_l2 : y_l2 + h_l2, x_l2 : x_l2 + w_l2]
            target_shape_yx = labels_filtered.shape[:2]
            target_shape_wh = (target_shape_yx[1], target_shape_yx[0]) # cv2 uses (w, h) for resize dsize

            dab_patch_resized = None
            cell_patch_smp_resized = None

            # Check if resizing is needed and possible
            if target_shape_yx[0] > 0 and target_shape_yx[1] > 0:
                # Resize original mask patches to match the StarDist label output dimensions
                if dab_patch_orig.shape[:2] != target_shape_yx:
                        dab_patch_resized = cv2.resize(dab_patch_orig.astype(np.uint8), target_shape_wh, interpolation=cv2.INTER_NEAREST)
                else: dab_patch_resized = dab_patch_orig.astype(np.uint8) # Ensure uint8 type

                if cell_patch_smp_orig.shape[:2] != target_shape_yx:
                        cell_patch_smp_resized = cv2.resize(cell_patch_smp_orig.astype(np.uint8), target_shape_wh, interpolation=cv2.INTER_NEAREST)
                else: cell_patch_smp_resized = cell_patch_smp_orig.astype(np.uint8) # Ensure uint8 type

            else:
                    logger.warning(f"[{func_name}] Invalid target shape {target_shape_yx} from StarDist labels for candidate {candidate_index+1}. Cannot perform counting.")
                    # Return hotspot with counts=0 but potentially non-zero unfiltered count
                    return hotspot

            # --- Save Debug Resized Masks (if dir exists) ---
            if hs_debug_dir:
                try:
                    if dab_patch_resized is not None: cv2.imwrite(os.path.join(hs_debug_dir,f"hs_{candidate_index+1:02d}_dab_patch_resized.png"), dab_patch_resized * 255)
                    if cell_patch_smp_resized is not None: cv2.imwrite(os.path.join(hs_debug_dir,f"hs_{candidate_index+1:02d}_smpcell_patch_resized.png"), cell_patch_smp_resized * 255)
                except Exception as e_save_masks:
                    logger.error(f"[{func_name}] Error saving debug resized masks for candidate {candidate_index+1}: {e_save_masks}")


            # --- Generate Comparison Plot (Debug, if dir exists) ---
            # Check all components needed for the plot are valid
            if hs_debug_dir and hs_patch_rgb is not None and labels_filtered is not None and dab_patch_resized is not None:
                try:
                    save_path_comparison = os.path.join(hs_debug_dir, f"hs_{candidate_index+1:02d}_stardist_comparison.jpg")
                    # Call the function imported from visualization module
                    save_stardist_comparison_plot(
                        hs_patch_rgb=hs_patch_rgb,
                        labels_filtered=labels_filtered, # Show overlay of filtered labels
                        ref_mask=dab_patch_resized, # Use resized DAB mask as reference panel
                        save_path=save_path_comparison
                    )
                except NameError: # Handles case where import failed
                        pass # Warning already logged during import attempt
                except Exception as e_comp_plot:
                    logger.error(f"[{func_name}] Error calling save_stardist_comparison_plot for candidate {candidate_index+1}: {e_comp_plot}", exc_info=True)

            # --- Count Centroids based on Resized Masks ---
            positive_count = 0 # Count for DAB+ & SMP+ (Numerator & Score)
            smp_cell_area_count = 0 # Count for SMP+ (Denominator)
            # Create overlay image only if debugging is enabled
            counted_overlay = hs_patch_rgb.copy() if hs_debug_dir and hs_patch_rgb is not None else None

            # Check if masks needed for counting are valid
            if dab_patch_resized is not None and cell_patch_smp_resized is not None:
                try:
                    # Convert float centroids to int for indexing
                    centroids_int = np.rint(centroids).astype(int)
                    centroids_int[:, 0] = np.clip(centroids_int[:, 0], 0, target_shape_yx[0] - 1)
                    centroids_int[:, 1] = np.clip(centroids_int[:, 1], 0, target_shape_yx[1] - 1)

                    # Get bounds of the *resized* masks for checking
                    max_y, max_x = target_shape_yx[0] - 1, target_shape_yx[1] - 1
                    cy, cx = -1, -1 # Initialize for potential error logging

                    for idx in range(centroids_int.shape[0]):
                        # Centroid coordinates (y, x) are relative to the patch
                        cy, cx = centroids_int[idx, 0], centroids_int[idx, 1]

                        # Check if centroid is within the bounds of the resized masks
                        if not (0 <= cy <= max_y and 0 <= cx <= max_x):
                            logger.warning(f"[{func_name}] Centroid {idx} ({cy},{cx}) outside bounds ({max_y},{max_x}) for candidate {candidate_index+1}. Skipping.")
                            continue

                        # Check masks at the centroid location
                        # Masks are uint8, > 0 means positive
                        is_in_smp_cell = (cell_patch_smp_resized[cy, cx] > 0)
                        is_dab_positive = (dab_patch_resized[cy, cx] > 0)

                        # Increment counts based on mask values
                        if is_in_smp_cell:
                            smp_cell_area_count += 1 # Count if in SMP+ region (denominator)
                            if is_dab_positive:
                                positive_count += 1 # Count if also in DAB+ region (numerator/score)
                                # Draw green circle on debug overlay if available
                                if counted_overlay is not None: cv2.circle(counted_overlay, (cx, cy), 5, (0, 255, 0), -1) # Green: DAB+ & SMP+
                            else:
                                    # Draw blue circle for SMP+ only
                                    if counted_overlay is not None: cv2.circle(counted_overlay, (cx, cy), 5, (255, 0, 0), -1) # Blue: SMP+, DAB-
                        elif is_dab_positive:
                                # Draw yellow circle for DAB+ only (outside SMP region)
                                if counted_overlay is not None: cv2.circle(counted_overlay, (cx, cy), 3, (0, 255, 255), -1) # Yellow: DAB+, SMP-
                        else:
                                # Draw red circle for background (neither DAB+ nor SMP+)
                                if counted_overlay is not None: cv2.circle(counted_overlay, (cx, cy), 3, (0, 0, 255), -1) # Red: Neither

                except IndexError as e_idx:
                        # This error is serious, indicates problem with coordinate logic or resizing
                        logger.error(f"[{func_name}] IndexError during counting for candidate {candidate_index+1} (cy={cy}, cx={cx}, mask_shape={target_shape_yx}): {e_idx}", exc_info=True)
                        # Reset counts to be safe
                        positive_count = 0; smp_cell_area_count = 0
                except Exception as e_count:
                    logger.error(f"[{func_name}] Unexpected error during counting loop for candidate {candidate_index+1}: {e_count}", exc_info=True)
                    # Might continue, but counts could be incomplete

            else:
                    logger.warning(f"[{func_name}] One or both resized masks are None for candidate {candidate_index+1}. Cannot perform counting.")


            # --- Save Debug Counting Overlay (if created) ---
            if counted_overlay is not None and hs_debug_dir:
                try:
                    cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{candidate_index+1:02d}_counted_centroids_vis.png"), cv2.cvtColor(counted_overlay, cv2.COLOR_RGB2BGR))
                except Exception as e_save_overlay:
                        logger.error(f"[{func_name}] Error saving debug counted overlay for candidate {candidate_index+1}: {e_save_overlay}")

            # --- Update Hotspot Dictionary with Final Counts ---
            # Get count from filtered labels if available
            if labels_filtered is not None:
                hotspot['stardist_total_count_filtered'] = len(np.unique(labels_filtered[labels_filtered > 0]))
            else: hotspot['stardist_total_count_filtered'] = 0 # If filtered labels are None

            hotspot['stardist_smp_cell_count'] = smp_cell_area_count
            hotspot['stardist_dab_smp_cell_count'] = positive_count # This is the primary ranking score
            logger.debug(f"[{func_name}] Candidate {candidate_index+1}: Counts - Total(Unfilt)={hotspot['stardist_total_count']}, Total(Filt)={hotspot['stardist_total_count_filtered']}, In SMP Area={smp_cell_area_count}, DAB+&SMP+={positive_count}")

        else:
            # Case where StarDist found no valid centroids
            logger.debug(f"[{func_name}] Candidate {candidate_index+1}: No valid centroids found by StarDist. All counts remain 0.")
            # All count keys were already initialized to 0

        # Return the updated dictionary containing the counts
        return hotspot

    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during refinement for candidate {candidate_index+1}: {e}")
        logger.error(traceback.format_exc()) # Log full traceback
        return None # Indicate failure
