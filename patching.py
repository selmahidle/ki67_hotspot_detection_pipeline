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
             return None, None, None, level_dims

        patches_to_process = []
        patch_locations = []
        patch_weights = []

        for y in tqdm(range(0, level_h, stride), desc=f"Collecting Patches L{level}"):
            for x in range(0, level_w, stride):
                y_start = y
                x_start = x
                current_patch_w = min(patch_size, level_w - x_start)
                current_patch_h = min(patch_size, level_h - y_start)

                if current_patch_w < stride // 2 or current_patch_h < stride // 2:
                    continue

                y_end = y_start + current_patch_h
                x_end = x_start + current_patch_w

                if condition_mask is not None:
                    if y_end > condition_mask.shape[0] or x_end > condition_mask.shape[1]:
                        logger.warning(f"Patch L{level} ({x_start},{y_start}) coords exceed condition_mask shape {condition_mask.shape}. Skipping.")
                        continue
                    patch_condition_mask = condition_mask[y_start:y_end, x_start:x_end]
                    if patch_condition_mask.size == 0 or np.mean(patch_condition_mask) < 0.05:
                        continue

                level0_x = int(x_start * slide.level_downsamples[level])
                level0_y = int(y_start * slide.level_downsamples[level])
                try:
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

                actual_h, actual_w = patch_np.shape[:2]
                if actual_h != current_patch_h or actual_w != current_patch_w:
                    logger.warning(f"Read patch L{level} ({x_start},{y_start}) size mismatch: Expected ({current_patch_w}x{current_patch_h}), Got ({actual_w}x{actual_h}). Resizing.")
                    patch_np = cv2.resize(patch_np, (current_patch_w, current_patch_h), interpolation=cv2.INTER_LINEAR)
                    actual_h, actual_w = current_patch_h, current_patch_w

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
                try:
                    pad_h = max(0, patch_size - actual_h)
                    pad_w = max(0, patch_size - actual_w)

                    if pad_h > 0 or pad_w > 0:
                         padded_patch_np = cv2.copyMakeBorder(patch_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                         logger.log(logging.DEBUG-1, f"Padded patch ({actual_w}x{actual_h}) to ({patch_size}x{patch_size}) for transform.")
                    else:
                         padded_patch_np = patch_np

                    transformed = transforms(image=padded_patch_np)
                    patch_tensor = transformed['image']
                except Exception as e:
                     logger.error(f"Error transforming patch L{level} ({x_start},{y_start}): {e}. Skipping.")
                     logger.error(traceback.format_exc())
                     continue

                patches_to_process.append(patch_tensor)
                patch_locations.append((y_start, y_start + actual_h, x_start, x_start + actual_w))
                patch_weights.append(create_weight_map((actual_h, actual_w)))

        total_patches_collected = len(patches_to_process)
        logger.info(f"Collected {total_patches_collected} patches from WSI Level {level}.")
        if total_patches_collected == 0:
            logger.warning(f"No processable patches found for WSI Level {level}.")
            return None, None, None, level_dims

        return patches_to_process, patch_locations, patch_weights, level_dims

    except Exception as e:
        logger.error(f"Error during WSI patch collection for level {level}: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None


def run_inference_on_patches(model, device, output_channels, batch_size, level_or_id, level_dims,
                             patches_to_process, patch_locations, patch_weights, model_name="Model"):
    if not patches_to_process:
        logger.warning(f"Received 0 patches for inference ({level_or_id}). Returning empty result map if dimensions known.")
        if level_dims and len(level_dims) == 2:
             level_w, level_h = level_dims
             if output_channels == 1:
                 return np.zeros((level_h, level_w), dtype=np.float32)
             else:
                 return np.zeros((level_h, level_w, output_channels), dtype=np.float32)
        else:
             logger.error("Cannot create empty result map: level_dims invalid.")
             return None

    try:
        model.to(device)
        model.eval()

        level_w, level_h = level_dims[0], level_dims[1]
        if output_channels == 1:
            predictions_sum = np.zeros((level_h, level_w), dtype=np.float64)
            weight_accumulator = np.zeros((level_h, level_w), dtype=np.float64)
        else:
            predictions_sum = np.zeros((level_h, level_w, output_channels), dtype=np.float64)
            weight_accumulator = np.zeros((level_h, level_w, output_channels), dtype=np.float64)
        epsilon = 1e-9

        total_patches_collected = len(patches_to_process)

        if total_patches_collected == 0: # This is somewhat redundant due to the first check.
            logger.warning(f"Received 0 patches for inference on {level_or_id}. Returning empty result.")
            if output_channels == 1:
                 return np.zeros((level_h, level_w), dtype=np.float32)
            else:
                 return np.zeros((level_h, level_w, output_channels), dtype=np.float32)

        if len(patch_locations) != total_patches_collected or len(patch_weights) != total_patches_collected:
             logger.error(f"Mismatch in lengths: patches ({total_patches_collected}), locations ({len(patch_locations)}), weights ({len(patch_weights)}). Cannot proceed.")
             return None

        logger.info(f"Running inference on {total_patches_collected} patches for {level_or_id} ({model_name})...")

        processed_count = 0
        with torch.no_grad():
            for i in tqdm(range(0, total_patches_collected, batch_size), desc=f"Processing Batches {level_or_id} ({model_name})"):
                batch_indices = range(i, min(i + batch_size, total_patches_collected))
                batch_tensors = []
                valid_batch_indices = []

                for idx in batch_indices:
                    patch_tensor = patches_to_process[idx]
                    if patch_tensor is None:
                        logger.warning(f"Patch tensor at index {idx} is None. Skipping.")
                        continue
                    if not isinstance(patch_tensor, torch.Tensor):
                         logger.warning(f"Item at index {idx} is not a tensor ({type(patch_tensor)}). Skipping.")
                         continue
                    if patch_tensor.ndim != 3:
                         logger.warning(f"Patch tensor at index {idx} has incorrect ndim {patch_tensor.ndim} (expected 3). Skipping.")
                         continue
                    batch_tensors.append(patch_tensor)
                    valid_batch_indices.append(idx)

                if not batch_tensors:
                    logger.warning(f"Batch starting at index {i} is empty after validation. Skipping.")
                    continue
                try:
                    batch_input = torch.stack(batch_tensors).to(device)
                except RuntimeError as stack_err:
                    logger.error(f"Error stacking batch starting at index {i}: {stack_err}")
                    logger.error("Individual tensor shapes in failed batch:")
                    for bt in batch_tensors: logger.error(f"  Shape: {bt.shape}")
                    continue
                except Exception as e_stack:
                    logger.error(f"Unexpected error stacking batch {i}: {e_stack}")
                    continue
                try:
                    batch_output_logits = model(batch_input)
                    if output_channels == 1:
                        batch_probs_tensor = torch.sigmoid(batch_output_logits)
                    else:
                        batch_probs_tensor = F.softmax(batch_output_logits, dim=1)
                    batch_probs = batch_probs_tensor.cpu().numpy()
                except Exception as e_inf:
                    logger.error(f"Error during model inference for batch {i // batch_size} ({model_name}): {e_inf}")
                    logger.error(f"Input batch shape: {batch_input.shape}")
                    continue

                for j, original_idx in enumerate(valid_batch_indices):
                    try:
                        patch_prob_output = batch_probs[j]
                        y_start, y_end, x_start, x_end = patch_locations[original_idx]
                        weight = patch_weights[original_idx]

                        if patch_prob_output.shape[0] != output_channels:
                            logger.error(f"Output channel mismatch for patch {original_idx} ({model_name}). Expected {output_channels}, Got {patch_prob_output.shape[0]}. Skipping patch.")
                            continue

                        pred_c, pred_h, pred_w = patch_prob_output.shape
                        orig_h = y_end - y_start
                        orig_w = x_end - x_start

                        if pred_h > orig_h or pred_w > orig_w:
                            logger.log(logging.DEBUG-1, f"Cropping prediction ({pred_w}x{pred_h}) back to original patch size ({orig_w}x{orig_h}) for patch {original_idx}.")
                            patch_prob_cropped = patch_prob_output[:, :orig_h, :orig_w]
                        else:
                            patch_prob_cropped = patch_prob_output

                        if weight.shape != (orig_h, orig_w):
                            logger.warning(f"Weight map shape {weight.shape} mismatch with cropped prediction's spatial shape {(orig_h, orig_w)} for patch {original_idx}. Resizing weight map.")
                            weight = cv2.resize(weight, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                        if output_channels == 1:
                            current_patch_prob_2d = np.squeeze(patch_prob_cropped, axis=0)
                            target_slice_preds = predictions_sum[y_start:y_end, x_start:x_end]
                            target_slice_weights = weight_accumulator[y_start:y_end, x_start:x_end]
                            if target_slice_preds.shape != current_patch_prob_2d.shape:
                                 logger.error(f"Shape mismatch for 2D accumulation (patch {original_idx}). Target: {target_slice_preds.shape}, Patch Probs: {current_patch_prob_2d.shape}. Skipping.")
                                 continue
                            if target_slice_weights.shape != weight.shape:
                                 logger.error(f"Shape mismatch for 2D accumulation (patch {original_idx}). Target Weights: {target_slice_weights.shape}, Patch Weights: {weight.shape}. Skipping.")
                                 continue
                            target_slice_preds += current_patch_prob_2d * weight
                            target_slice_weights += weight
                        else:
                            patch_prob_transposed = patch_prob_cropped.transpose(1, 2, 0)
                            weight_expanded = np.repeat(weight[:, :, np.newaxis], output_channels, axis=2)
                            target_slice_preds = predictions_sum[y_start:y_end, x_start:x_end, :]
                            target_slice_weights = weight_accumulator[y_start:y_end, x_start:x_end, :]
                            if target_slice_preds.shape != patch_prob_transposed.shape:
                                 logger.error(f"Shape mismatch for 3D accumulation (patch {original_idx}). Target: {target_slice_preds.shape}, Patch Probs: {patch_prob_transposed.shape}. Skipping.")
                                 continue
                            if target_slice_weights.shape != weight_expanded.shape:
                                 logger.error(f"Shape mismatch for 3D accumulation (patch {original_idx}). Target Weights: {target_slice_weights.shape}, Patch Weights: {weight_expanded.shape}. Skipping.")
                                 continue
                            target_slice_preds += patch_prob_transposed * weight_expanded
                            target_slice_weights += weight_expanded
                        processed_count += 1
                    except IndexError as e_idx:
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
             logger.error(f"No patches were successfully processed for {level_or_id}. Returning empty result based on output_channels.")
             if output_channels == 1:
                 return np.zeros((level_h, level_w), dtype=np.float32)
             else:
                 return np.zeros((level_h, level_w, output_channels), dtype=np.float32)

        final_probabilities = np.zeros_like(predictions_sum, dtype=np.float32)
        valid_weights_mask = weight_accumulator > epsilon
        np.divide(predictions_sum, weight_accumulator, out=final_probabilities, where=valid_weights_mask)
        final_probabilities = np.clip(final_probabilities, 0.0, 1.0)
        logger.info(f"Probability map stitching complete for {level_or_id} ({model_name}). Final shape: {final_probabilities.shape}")
        return final_probabilities.astype(np.float32)
    except Exception as e:
        logger.error(f"Error during inference/stitching for {level_or_id}: {e}")
        logger.error(traceback.format_exc())
        return None

def segment_level(slide, level, model, device, transforms, output_channels, batch_size, patch_size=512, overlap=256, condition_mask=None):
    logger.warning("Using segment_level wrapper for direct WSI processing. Ensure this is intended behavior.")
    try:
        patches, locations, weights, lvl_dims = collect_patches_for_level(
            slide, level, transforms, patch_size, overlap, condition_mask
        )
        if patches is None or not patches:
            logger.error(f"Patch collection failed or returned no patches for Level {level}.")
            if lvl_dims and len(lvl_dims) == 2:
                level_w, level_h = lvl_dims
                if output_channels == 1: # Added conditional return shape
                    return np.zeros((level_h, level_w), dtype=np.float32)
                else:
                    return np.zeros((level_h, level_w, output_channels), dtype=np.float32)
            else:
                return None
        final_probabilities = run_inference_on_patches(
            model, device, output_channels, batch_size, f"Level {level}", lvl_dims,
            patches, locations, weights, model_name=f"Model_L{level}"
        )
        del patches, locations, weights
        return final_probabilities
    except Exception as e:
        logger.error(f"Error in segment_level wrapper for level {level}: {e}")
        logger.error(traceback.format_exc())
        return None