import os
import traceback
import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import openslide
from tqdm import tqdm

from utils import create_weight_map

logger = logging.getLogger(__name__)


def collect_patches_for_level(slide, level, transforms, patch_size=512, overlap=256, condition_mask=None):
    """
    Collects patches and metadata directly from a WSI level for processing.

    Iterates through the specified level of an OpenSlide object, extracts
    patches, applies transformations, and gathers necessary information
    for inference and stitching. Can optionally use a condition_mask to
    process only relevant regions.

    Note: This function reads directly from the WSI. The pipeline in the
          refactored script uses JPGs for segmentation stages, so this
          function might not be directly called by the main pipeline anymore,
          but it's kept here as it represents the original WSI patching logic.

    Args:
        slide (openslide.OpenSlide): The WSI object.
        level (int): The WSI level to extract patches from.
        transforms (callable): Albumentations transforms to apply to each patch.
        patch_size (int): The desired size (width and height) of the patches.
        overlap (int): The number of overlapping pixels between adjacent patches.
        condition_mask (np.ndarray, optional): A binary mask at the target `level`.
                                               If provided, only patches overlapping
                                               significantly (mean > 0.05) with
                                               this mask are collected.

    Returns:
        tuple: (patches_to_process, patch_locations, patch_weights, level_dims)
               - patches_to_process (list): List of transformed patch tensors (C, H, W).
               - patch_locations (list): List of tuples (y_start, y_end, x_start, x_end)
                                         indicating patch coordinates at the `level`.
               - patch_weights (list): List of 2D weight map arrays (H, W) for stitching.
               - level_dims (tuple): Dimensions (width, height) of the specified `level`.
               Returns (None, None, None, None) if collection fails or no valid
               patches are found.
    """
    logger.warning("collect_patches_for_level reads directly from WSI. The current pipeline uses JPGs. Ensure this function is needed.")
    if not isinstance(slide, openslide.OpenSlide):
        logger.error("Invalid 'slide' object provided.")
        return None, None, None, None
    if level < 0 or level >= slide.level_count:
        logger.error(f"Invalid level specified: {level}. Slide has {slide.level_count} levels.")
        return None, None, None, None

    try:
        level_dims = slide.level_dimensions[level]
        level_w, level_h = level_dims[0], level_dims[1]
        logger.info(f"Collecting patches from WSI Level {level} ({level_w}x{level_h}) | Patch Size: {patch_size}, Overlap: {overlap}")

        stride = patch_size - overlap
        if stride <= 0:
             logger.error(f"Overlap ({overlap}) must be less than patch size ({patch_size}). Stride is {stride}.")
             return None, None, None, level_dims # Return dims even on error

        patches_to_process = []
        patch_locations = []
        patch_weights = []

        for y in tqdm(range(0, level_h, stride), desc=f"Collecting Patches L{level}"):
            for x in range(0, level_w, stride):
                # Calculate patch coordinates and size for this level
                y_start = y
                x_start = x
                # Read region slightly larger if possible, then crop? No, read exact size based on level.
                # Size calculation must be careful at edges
                current_patch_w = min(patch_size, level_w - x_start)
                current_patch_h = min(patch_size, level_h - y_start)

                # Skip tiny edge fragments smaller than half a stride
                if current_patch_w < stride // 2 or current_patch_h < stride // 2:
                    continue

                y_end = y_start + current_patch_h
                x_end = x_start + current_patch_w

                # --- Condition Mask Check ---
                if condition_mask is not None:
                    if y_end > condition_mask.shape[0] or x_end > condition_mask.shape[1]:
                        logger.warning(f"Patch L{level} ({x_start},{y_start}) coords exceed condition_mask shape {condition_mask.shape}. Skipping.")
                        continue
                    patch_condition_mask = condition_mask[y_start:y_end, x_start:x_end]
                    # Check if the mean value of the mask in this patch is above a threshold (e.g., 5%)
                    if patch_condition_mask.size == 0 or np.mean(patch_condition_mask) < 0.05:
                        continue # Skip if no significant overlap

                # --- Read Patch from WSI ---
                # Calculate corresponding Level 0 coordinates for reading
                level0_x = int(x_start * slide.level_downsamples[level])
                level0_y = int(y_start * slide.level_downsamples[level])
                try:
                    # Read region expects (location, level, size)
                    # Size is (width, height)
                    patch_pil = slide.read_region((level0_x, level0_y), level, (current_patch_w, current_patch_h)).convert('RGB')
                    patch_np = np.array(patch_pil)
                except openslide.OpenSlideError as e:
                    logger.warning(f"OpenSlideError reading patch L{level} ({x_start},{y_start}) Size({current_patch_w},{current_patch_h}): {e}. Skipping.")
                    continue
                except Exception as e:
                     logger.error(f"Unexpected error reading patch L{level} ({x_start},{y_start}): {e}. Skipping.")
                     logger.error(traceback.format_exc())
                     continue

                if patch_np is None or patch_np.size == 0:
                    logger.warning(f"Empty patch read L{level} ({x_start},{y_start}). Skipping.")
                    continue

                # Verify read patch dimensions (should match current_patch_w/h)
                actual_h, actual_w = patch_np.shape[:2]
                if actual_h != current_patch_h or actual_w != current_patch_w:
                    logger.warning(f"Read patch L{level} ({x_start},{y_start}) size mismatch: Expected ({current_patch_w}x{current_patch_h}), Got ({actual_w}x{actual_h}). Resizing.")
                    patch_np = cv2.resize(patch_np, (current_patch_w, current_patch_h), interpolation=cv2.INTER_LINEAR)
                    # Update actual dimensions after resize
                    actual_h, actual_w = current_patch_h, current_patch_w

                # --- Preprocessing (Ensure 3 channels) ---
                if len(patch_np.shape) == 3 and patch_np.shape[2] == 4:
                    patch_np = cv2.cvtColor(patch_np, cv2.COLOR_RGBA2RGB)
                elif len(patch_np.shape) == 2:
                    patch_np = cv2.cvtColor(patch_np, cv2.COLOR_GRAY2RGB)
                elif len(patch_np.shape) != 3 or patch_np.shape[2] != 3:
                     logger.warning(f"Patch L{level} ({x_start},{y_start}) has unexpected shape {patch_np.shape}. Attempting to use first 3 channels.")
                     if len(patch_np.shape) == 3 and patch_np.shape[2] > 3:
                         patch_np = patch_np[:, :, :3]
                     else:
                         logger.error("Cannot convert patch to 3 channels. Skipping.")
                         continue

                # --- Apply Transforms ---
                try:
                    # Pad if patch is smaller than target patch_size before transforming
                    # This is needed if transforms expect a fixed size (e.g., some CNNs)
                    # Calculate padding needed
                    pad_h = max(0, patch_size - actual_h)
                    pad_w = max(0, patch_size - actual_w)

                    if pad_h > 0 or pad_w > 0:
                         # Pad with zeros (black)
                         padded_patch_np = cv2.copyMakeBorder(patch_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                         logger.log(logging.DEBUG-1, f"Padded patch ({actual_w}x{actual_h}) to ({patch_size}x{patch_size}) for transform.")
                    else:
                         padded_patch_np = patch_np

                    transformed = transforms(image=padded_patch_np)
                    patch_tensor = transformed['image'] # Shape: (C, H, W) - should be (C, patch_size, patch_size)
                except Exception as e:
                     logger.error(f"Error transforming patch L{level} ({x_start},{y_start}): {e}. Skipping.")
                     logger.error(traceback.format_exc())
                     continue

                # --- Store patch and metadata ---
                patches_to_process.append(patch_tensor)
                patch_locations.append((y_start, y_start + actual_h, x_start, x_start + actual_w)) # Use actual height/width for location
                patch_weights.append(create_weight_map((actual_h, actual_w))) # Weight map for the *original* (non-padded) patch size

        total_patches_collected = len(patches_to_process)
        logger.info(f"Collected {total_patches_collected} patches from WSI Level {level}.")
        if total_patches_collected == 0:
            logger.warning(f"No processable patches found for WSI Level {level}.")
            # Return level_dims even if no patches
            return None, None, None, level_dims

        return patches_to_process, patch_locations, patch_weights, level_dims

    except Exception as e:
        logger.error(f"Error during WSI patch collection for level {level}: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None


def run_inference_on_patches(model, device, output_channels, batch_size, level_or_id, level_dims,
                             patches_to_process, patch_locations, patch_weights, model_name="Model"):
    """
    Runs inference on pre-collected patches and stitches the PROBABILITY results.

    This function takes patches (already transformed tensors), runs them through
    the model in batches, and combines the overlapping probability outputs using
    provided weight maps.

    Args:
        model (torch.nn.Module): The segmentation model (already on `device`).
        device (torch.device): CUDA or CPU device.
        output_channels (int): Expected number of output channels from the model.
        batch_size (int): Number of patches to process in parallel.
        level_or_id (int or str): Identifier for logging (e.g., WSI level or image ID).
        level_dims (tuple): Dimensions (width, height) of the full area being processed.
        patches_to_process (list): List of patch tensors (C, H, W).
        patch_locations (list): List of tuples (y_start, y_end, x_start, x_end)
                                corresponding to `patches_to_process`.
        patch_weights (list): List of 2D weight map arrays (H, W) corresponding
                              to `patches_to_process` (matching original patch dims).
        model_name (str): Optional name for the model for logging messages.

    Returns:
        np.ndarray or None: The stitched probability map for the entire area
                            (shape: H, W, num_classes) or None if processing fails.
    """
    if not patches_to_process:
        logger.warning(f"Received 0 patches for inference ({level_or_id}). Returning empty result map if dimensions known.")
        if level_dims and len(level_dims) == 2:
             level_w, level_h = level_dims
             return np.zeros((level_h, level_w, output_channels), dtype=np.float32)
        else:
             logger.error("Cannot create empty result map: level_dims invalid.")
             return None

    try:
        # Ensure model is on the correct device (redundant if already done, but safe)
        model.to(device)
        model.eval() # Set model to evaluation mode

        level_w, level_h = level_dims[0], level_dims[1]
        total_patches_collected = len(patches_to_process)

        if total_patches_collected == 0:
            logger.warning(f"Received 0 patches for inference on {level_or_id}. Returning empty result.")
            return np.zeros((level_h, level_w, output_channels), dtype=np.float32)
        if len(patch_locations) != total_patches_collected or len(patch_weights) != total_patches_collected:
             logger.error(f"Mismatch in lengths: patches ({total_patches_collected}), locations ({len(patch_locations)}), weights ({len(patch_weights)}). Cannot proceed.")
             return None

        logger.info(f"Running inference on {total_patches_collected} patches for {level_or_id}...")

        # Initialize accumulators for weighted probabilities and total weights
        # Shape: (H, W, C) where C is number of classes
        predictions_sum = np.zeros((level_h, level_w, output_channels), dtype=np.float64) # Use float64 for accumulator precision
        weight_accumulator = np.zeros((level_h, level_w, output_channels), dtype=np.float64)
        epsilon = 1e-9 # Small value to prevent division by zero

        processed_count = 0
        # --- Process patches in batches ---
        with torch.no_grad():
            for i in tqdm(range(0, total_patches_collected, batch_size), desc=f"Processing Batches {level_or_id} ({model_name})"):
                batch_indices = range(i, min(i + batch_size, total_patches_collected))
                batch_tensors = []
                valid_batch_indices = [] # Keep track of indices successfully added to batch

                # --- Prepare Batch ---
                for idx in batch_indices:
                    patch_tensor = patches_to_process[idx]
                    if patch_tensor is None:
                        logger.warning(f"Patch tensor at index {idx} is None. Skipping.")
                        continue
                    # Basic check for tensor type and shape (optional but good)
                    if not isinstance(patch_tensor, torch.Tensor):
                         logger.warning(f"Item at index {idx} is not a tensor ({type(patch_tensor)}). Skipping.")
                         continue
                    if patch_tensor.ndim != 3: # Should be C, H, W
                         logger.warning(f"Patch tensor at index {idx} has incorrect ndim {patch_tensor.ndim} (expected 3). Skipping.")
                         continue

                    batch_tensors.append(patch_tensor)
                    valid_batch_indices.append(idx) # Store original index

                if not batch_tensors:
                    logger.warning(f"Batch starting at index {i} is empty after validation. Skipping.")
                    continue

                # Stack tensors to create batch and move to device
                try:
                    batch_input = torch.stack(batch_tensors).to(device)
                except RuntimeError as stack_err:
                    logger.error(f"Error stacking batch starting at index {i}: {stack_err}")
                    logger.error("Individual tensor shapes in failed batch:")
                    for bt in batch_tensors: logger.error(f"  Shape: {bt.shape}")
                    continue # Skip this batch
                except Exception as e_stack:
                    logger.error(f"Unexpected error stacking batch {i}: {e_stack}")
                    continue

                # --- Inference ---
                try:
                    batch_output_logits = model(batch_input) # Shape (B, C, H, W) - Raw logits
                    # Apply Softmax or Sigmoid (if needed, depends on model's activation)
                    # Assuming model outputs logits and we need probabilities:
                    # If model has sigmoid/softmax activation layer, this step is redundant.
                    # If model has activation=None, we need it here.
                    # The original script seemed to assume logits and applied softmax here.
                    batch_probs_tensor = F.softmax(batch_output_logits, dim=1) # Get probabilities
                    # Move results back to CPU and convert to NumPy
                    batch_probs = batch_probs_tensor.cpu().numpy() # Shape (B, C, H, W)
                except Exception as e_inf:
                    logger.error(f"Error during model inference for batch {i // batch_size} ({model_name}): {e_inf}")
                    logger.error(f"Input batch shape: {batch_input.shape}")
                    continue # Skip processing this batch's results

                # --- Stitching ---
                # Iterate through the *results* in the batch
                for j, original_idx in enumerate(valid_batch_indices):
                    try:
                        # --- Get data for this patch result ---
                        patch_prob_all_classes = batch_probs[j] # Shape (C, H, W)
                        # Location corresponds to the original patch size (before padding)
                        y_start, y_end, x_start, x_end = patch_locations[original_idx]
                        # Weight map also corresponds to the original patch size
                        weight = patch_weights[original_idx] # Shape (H_orig, W_orig)

                        # --- Validate shapes ---
                        if patch_prob_all_classes.shape[0] != output_channels:
                            logger.error(f"Output channel mismatch for patch {original_idx} ({model_name}). Expected {output_channels}, Got {patch_prob_all_classes.shape[0]}. Skipping patch.")
                            continue

                        # Model output dimensions (potentially padded)
                        pred_c, pred_h, pred_w = patch_prob_all_classes.shape
                        # Original patch dimensions (from location info)
                        orig_h = y_end - y_start
                        orig_w = x_end - x_start

                        # --- Crop prediction if it was padded ---
                        # The model output (pred_h, pred_w) might be larger than the original patch size
                        # if padding was applied before inference. Crop it back.
                        if pred_h > orig_h or pred_w > orig_w:
                            logger.log(logging.DEBUG-1, f"Cropping prediction ({pred_w}x{pred_h}) back to original patch size ({orig_w}x{orig_h}) for patch {original_idx}.")
                            patch_prob_cropped = patch_prob_all_classes[:, :orig_h, :orig_w]
                        else:
                            patch_prob_cropped = patch_prob_all_classes

                        # Verify weight map shape matches the *original* (now cropped) prediction dimensions
                        if weight.shape != (orig_h, orig_w):
                            logger.warning(f"Weight map shape {weight.shape} mismatch with cropped prediction shape {(orig_h, orig_w)} for patch {original_idx}. Resizing weight map.")
                            weight = cv2.resize(weight, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                        # --- Prepare for accumulation ---
                        # Transpose probabilities to (H, W, C) for easier broadcasting with weights
                        patch_prob_transposed = patch_prob_cropped.transpose(1, 2, 0) # Shape (orig_h, orig_w, C)

                        # Expand weight map dims: (H, W) -> (H, W, 1) -> (H, W, C)
                        weight_expanded = np.repeat(weight[:, :, np.newaxis], output_channels, axis=2)

                        # --- Accumulate weighted predictions and weights ---
                        # Get slice of the global accumulators
                        target_slice_preds = predictions_sum[y_start:y_end, x_start:x_end, :]
                        target_slice_weights = weight_accumulator[y_start:y_end, x_start:x_end, :]

                        # Check shapes before adding (should match due to cropping/resizing)
                        if target_slice_preds.shape != patch_prob_transposed.shape:
                             logger.error(f"Shape mismatch during accumulation for patch {original_idx}. Target: {target_slice_preds.shape}, Patch Probs: {patch_prob_transposed.shape}. Skipping.")
                             continue
                        if target_slice_weights.shape != weight_expanded.shape:
                             logger.error(f"Shape mismatch during accumulation for patch {original_idx}. Target: {target_slice_weights.shape}, Patch Weights: {weight_expanded.shape}. Skipping.")
                             continue

                        # Add weighted probabilities and the weights themselves
                        target_slice_preds += patch_prob_transposed * weight_expanded
                        target_slice_weights += weight_expanded

                        processed_count += 1

                    except IndexError as e_idx:
                        # This might happen if patch_locations are outside level_dims
                        logger.error(f"IndexError during stitching patch {original_idx} ({model_name}): {e_idx}. Location: {(y_start, y_end, x_start, x_end)}, Map Size: {(level_h, level_w)}. Skipping patch.")
                        continue
                    except ValueError as e_val:
                        logger.error(f"ValueError during stitching patch {original_idx} ({model_name}): {e_val}. Skipping patch.")
                        continue
                    except Exception as e_stitch:
                        logger.error(f"Unexpected error stitching patch {original_idx} ({model_name}): {e_stitch}")
                        logger.error(traceback.format_exc())
                        continue

        logger.info(f"Finished processing {processed_count}/{total_patches_collected} patches for {level_or_id} ({model_name}).")

        if processed_count == 0:
             logger.error(f"No patches were successfully processed for {level_or_id}. Returning empty result.")
             return np.zeros((level_h, level_w, output_channels), dtype=np.float32)

        # --- Normalize predictions by the accumulated weight map ---
        # Avoid division by zero where weight is zero
        final_probabilities = np.zeros_like(predictions_sum, dtype=np.float32) # Output as float32

        # Check where weight is sufficiently large to avoid division by tiny numbers
        valid_weights_mask = weight_accumulator > epsilon

        np.divide(predictions_sum, weight_accumulator, out=final_probabilities, where=valid_weights_mask)

        # Optional: Clip final probabilities to [0, 1] range to handle potential numerical issues
        final_probabilities = np.clip(final_probabilities, 0.0, 1.0)

        logger.info(f"Probability map stitching complete for {level_or_id} ({model_name}). Final shape: {final_probabilities.shape}")
        return final_probabilities.astype(np.float32) # Ensure float32 return

    except Exception as e:
        logger.error(f"Error during inference/stitching for {level_or_id}: {e}")
        logger.error(traceback.format_exc())
        return None


def segment_level(slide, level, model, device, transforms, output_channels, batch_size, patch_size=512, overlap=256, condition_mask=None):
    """
    (Wrapper) Performs segmentation on a specific WSI level using WSI patch collection
    followed by inference and stitching.

    Note: This is a convenience wrapper around collect_patches_for_level and
          run_inference_on_patches for direct WSI processing. The refactored
          pipeline uses JPGs and calls run_inference_on_patches directly after
          manual JPG patching.

    Args:
        slide (openslide.OpenSlide): The WSI object.
        level (int): The WSI level to process.
        model (torch.nn.Module): The segmentation model.
        device (torch.device): CUDA or CPU device.
        transforms (callable): Albumentations transforms.
        output_channels (int): Expected number of output channels.
        batch_size (int): Batch size for inference.
        patch_size (int): Size of patches to extract.
        overlap (int): Overlap between patches.
        condition_mask (np.ndarray, optional): Mask to constrain processing region.

    Returns:
        np.ndarray or None: The stitched probability map for the level, or None on failure.
    """
    logger.warning("Using segment_level wrapper for direct WSI processing. Ensure this is intended behavior.")
    try:
        # 1. Collect patches directly from WSI level
        patches, locations, weights, lvl_dims = collect_patches_for_level(
            slide, level, transforms, patch_size, overlap, condition_mask
        )

        if patches is None or not patches:
            logger.error(f"Patch collection failed or returned no patches for Level {level}.")
            # Try to return empty map if dimensions are known
            if lvl_dims and len(lvl_dims) == 2:
                level_w, level_h = lvl_dims
                return np.zeros((level_h, level_w, output_channels), dtype=np.float32)
            else:
                return None # Cannot determine size

        # 2. Run inference and stitching on collected patches
        final_probabilities = run_inference_on_patches(
            model, device, output_channels, batch_size, f"Level {level}", lvl_dims,
            patches, locations, weights, model_name=f"Model_L{level}"
        )

        # Clear memory explicitly (optional, Python GC should handle it)
        del patches, locations, weights

        return final_probabilities

    except Exception as e:
        logger.error(f"Error in segment_level wrapper for level {level}: {e}")
        logger.error(traceback.format_exc())
        return None