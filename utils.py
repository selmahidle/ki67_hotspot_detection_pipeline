import numpy as np
import torch
import torch.nn as nn
import logging
import openslide


logger = logging.getLogger(__name__)


def get_actual_pixel_size_um(slide: openslide.OpenSlide, level: int, fallback_value: float = 0.25) -> float | None:
    """
    Retrieves the average pixel size (microns per pixel) for a specific level
    from OpenSlide slide metadata.

    Args:
        slide: The OpenSlide object.
        level (int): The desired WSI level for which to calculate the pixel size.
        fallback_value (float): Value to use if Level 0 MPP metadata is missing or invalid.

    Returns:
        float: Actual pixel size in microns for the specified level, or None if level is invalid.
               Returns the fallback value * adjusted for the level * if L0 MPP is invalid.
               Logs warnings if metadata is missing/invalid or fallback is used.
    """
    if not isinstance(slide, openslide.OpenSlide):
        logger.error("Invalid slide object passed.")
        return None
    if not (0 <= level < slide.level_count):
        logger.error(f"Invalid level ({level}) requested. Slide has {slide.level_count} levels.")
        return None

    mpp_x = None
    mpp_y = None
    mpp_l0 = None 

    try:
        mpp_x_str = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y_str = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)

        if mpp_x_str and mpp_y_str:
            mpp_x = float(mpp_x_str)
            mpp_y = float(mpp_y_str)

            if 0.1 <= mpp_x <= 1.0 and 0.1 <= mpp_y <= 1.0:
                 mpp_l0 = (mpp_x + mpp_y) / 2.0
                 logger.info(f"MPP L0 from slide: X={mpp_x:.4f}, Y={mpp_y:.4f} -> Avg={mpp_l0:.4f} µm/px")
            else:
                 logger.warning(f"MPP L0 values ({mpp_x}, {mpp_y}) seem outside expected range (0.1-1.0).")
        else:
            logger.warning("MPP L0 properties not found in slide metadata.")

    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse MPP L0 values: {e}")
    except Exception as e:
         logger.error(f"Unexpected error reading MPP L0 properties: {e}", exc_info=True)

    if mpp_l0 is None:
        mpp_l0 = fallback_value
        logger.warning(f"Falling back to default L0 MPP: {mpp_l0:.4f} µm/px.")

    try:
        downsample_at_level = slide.level_downsamples[level]
        mpp_at_level = mpp_l0 * downsample_at_level
        logger.info(f"Calculated MPP for Level {level}: {mpp_at_level:.4f} µm/px (L0 MPP: {mpp_l0:.4f}, Downsample: {downsample_at_level:.2f})")
        return mpp_at_level
    except IndexError:
        logger.error(f"Level {level} downsample factor not found unexpectedly.")
        return None
    except Exception as e:
         logger.error(f"Error calculating MPP at level {level}: {e}", exc_info=True)
         return None


def convert_batchnorm_to_groupnorm(module):
    """
    Recursively convert all BatchNorm2d layers in a module to GroupNorm layers.

    Args:
        module (torch.nn.Module): The PyTorch module to convert.

    Returns:
        torch.nn.Module: The modified module with GroupNorm layers.
    """
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        num_channels = module.num_features
        num_groups = max(1, num_channels // 8)

        logger.debug(f"Converting BatchNorm2d(num_features={num_channels}) to GroupNorm(num_groups={num_groups}, num_channels={num_channels})")

        module_output = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=module.eps,
            affine=module.affine 
        )

        if module.affine:
            with torch.no_grad():
                if hasattr(module, 'weight') and module.weight is not None:
                     module_output.weight.copy_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                     module_output.bias.copy_(module.bias)

        params = list(module.parameters())
        if params:
            module_output = module_output.to(device=params[0].device, dtype=params[0].dtype)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_batchnorm_to_groupnorm(child)
        )

    del module
    return module_output


def create_weight_map(shape):
    """
    Creates a 2D weight map for blending overlapping patches during stitching.

    The map has weights of 1.0 in the center, fading linearly to 0.0 at the
    edges (over 10% of the width/height). This helps reduce stitching artifacts.

    Args:
        shape (tuple): A tuple representing the (height, width) of the patch.

    Returns:
        np.ndarray: A 2D numpy array of dtype float32 with the calculated weights,
                    or None if the input shape is invalid.
    """
    if not isinstance(shape, (tuple, list)) or len(shape) < 2:
        logger.error(f"Cannot create weight map for shape {shape}. Requires tuple/list of length >= 2.")
        return None

    height, width = shape[:2]
    if height <= 0 or width <= 0:
        logger.error(f"Invalid shape dimensions for weight map: {height}x{width}")
        return None

    weight = np.ones((height, width), dtype=np.float32)

    edge_size_y = max(1, int(height * 0.1)) 
    edge_size_x = max(1, int(width * 0.1))

    ramp_y = np.linspace(0, 1, edge_size_y, dtype=np.float32)
    ramp_x = np.linspace(0, 1, edge_size_x, dtype=np.float32)

    if height > 1:
        weight[:edge_size_y, :] *= ramp_y[:, np.newaxis]
        weight[-edge_size_y:, :] *= ramp_y[::-1][:, np.newaxis]

    if width > 1: 
        weight[:, :edge_size_x] *= ramp_x
        weight[:, -edge_size_x:] *= ramp_x[::-1]

    weight = np.clip(weight, 0.0, 1.0)

    return weight


def combine_model_predictions(prob_maps, strategy='average'):
    """
    Combines probability maps from multiple models using a specified strategy.

    Args:
        prob_maps (list): A list of numpy arrays, where each array is a
                          probability map (H, W, C) from a model.
        strategy (str): The combination strategy to use. Options:
                        'average': Computes the mean probability across models.
                        'majority': Performs majority voting on thresholded predictions
                                    (assumes channel 1 is foreground).
                        Default is 'average'.

    Returns:
        np.ndarray or None: The combined probability map (H, W, C) or None if
                            input is invalid or combination fails.
    """
    if not prob_maps:
        logger.error("Received empty list of probability maps for combination.")
        return None

    if not isinstance(prob_maps, list) or len(prob_maps) == 0:
        logger.error("Input 'prob_maps' must be a non-empty list.")
        return None

    if len(prob_maps) == 1:
        logger.info("Only one probability map provided, returning it directly.")
        return prob_maps[0].astype(np.float32) if prob_maps[0] is not None else None

    first_map = prob_maps[0]
    if first_map is None:
        logger.error("First probability map in the list is None.")
        return None
    first_shape = first_map.shape
    if not all(p is not None and p.shape == first_shape for p in prob_maps):
         logger.error("Probability maps have inconsistent shapes or contain None values.")
         for i, p in enumerate(prob_maps):
             shape_str = "None" if p is None else str(p.shape)
             logger.error(f" Map {i} shape: {shape_str}")
         return None

    logger.info(f"Combining {len(prob_maps)} model predictions using strategy: {strategy}")

    try:
        if strategy == 'average':
            combined = np.mean(np.stack(prob_maps, axis=0), axis=0)
            return combined.astype(np.float32)

        elif strategy == 'majority':
            threshold = 0.5
            num_classes = first_shape[-1]

            if num_classes < 2:
                logger.error("Majority voting requires at least 2 output channels (bg/fg).")
                return None

            predictions = [(p[:, :, 1] > threshold).astype(np.uint8) for p in prob_maps]
            stacked_preds = np.stack(predictions, axis=0)
            combined_majority = (np.sum(stacked_preds, axis=0) > len(prob_maps) / 2).astype(np.uint8)

            h, w = combined_majority.shape
            output_prob_map = np.zeros(first_shape, dtype=np.float32)
            output_prob_map[:, :, 1] = combined_majority.astype(np.float32)
            output_prob_map[:, :, 0] = 1.0 - output_prob_map[:, :, 1]
    
            if num_classes > 2:
                logger.warning("Majority voting applied only to channel 1 vs 0 for >2 classes.")

            return output_prob_map

        else:
            logger.warning(f"Unsupported combination strategy: {strategy}. Falling back to 'average'.")
            combined = np.mean(np.stack(prob_maps, axis=0), axis=0)
            return combined.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during model prediction combination (strategy: {strategy}): {e}")
        logger.error(traceback.format_exc()) 
        return None