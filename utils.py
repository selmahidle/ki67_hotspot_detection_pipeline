import numpy as np
import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)

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
        # Determine the number of groups. Using num_channels // 8 is a common heuristic.
        # Ensure num_groups is at least 1 and divides num_channels if possible,
        # but prioritize having at least one group.
        # A common strategy is to use 32 groups if possible, or find a divisor.
        # Here, we stick to the num_channels // 8 approach as requested previously,
        # ensuring it's at least 1.
        num_groups = max(1, num_channels // 8)

        # Ensure num_groups evenly divides num_channels if > 1.
        # If not, find the largest divisor <= num_groups.
        # This adds robustness but might deviate slightly if the original heuristic
        # didn't guarantee divisibility. Let's stick to the simpler max(1, ...)
        # based on the original code's apparent intent.
        # if num_channels % num_groups != 0:
        #     # Find largest divisor <= num_groups
        #     found = False
        #     for ng in range(num_groups, 0, -1):
        #         if num_channels % ng == 0:
        #             num_groups = ng
        #             found = True
        #             break
        #     if not found: # Should not happen if num_groups starts at 1
        #         logger.warning(f"Could not find suitable divisor for GroupNorm with {num_channels} channels. Using num_groups=1.")
        #         num_groups = 1

        logger.debug(f"Converting BatchNorm2d(num_features={num_channels}) to GroupNorm(num_groups={num_groups}, num_channels={num_channels})")

        module_output = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=module.eps,
            affine=module.affine # Keep affine status from original BatchNorm
        )

        # Copy weights and biases if affine transformation was enabled in BatchNorm
        if module.affine:
            with torch.no_grad():
                # Check if parameters exist before copying
                if hasattr(module, 'weight') and module.weight is not None:
                     module_output.weight.copy_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                     module_output.bias.copy_(module.bias)

        # Ensure the new layer is on the same device and dtype as the original (if possible)
        # Check if the original module had parameters to determine device/dtype
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

    height, width = shape[:2] # Take only the first two dimensions
    if height <= 0 or width <= 0:
        logger.error(f"Invalid shape dimensions for weight map: {height}x{width}")
        return None

    weight = np.ones((height, width), dtype=np.float32)

    # Define edge size for falloff (e.g., 10% of each dimension)
    edge_size_y = max(1, int(height * 0.1)) # Ensure at least 1 pixel falloff if possible
    edge_size_x = max(1, int(width * 0.1))

    # Create linear ramps for edges
    ramp_y = np.linspace(0, 1, edge_size_y, dtype=np.float32)
    ramp_x = np.linspace(0, 1, edge_size_x, dtype=np.float32)

    # Apply vertical ramps
    if height > 1: # Avoid modifying if height is 1
        weight[:edge_size_y, :] *= ramp_y[:, np.newaxis]
        weight[-edge_size_y:, :] *= ramp_y[::-1][:, np.newaxis]

    # Apply horizontal ramps
    if width > 1: # Avoid modifying if width is 1
        weight[:, :edge_size_x] *= ramp_x
        weight[:, -edge_size_x:] *= ramp_x[::-1]

    # Clip values to ensure they are strictly between 0 and 1 after multiplication
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
        # Ensure correct type even for single map
        return prob_maps[0].astype(np.float32) if prob_maps[0] is not None else None

    # Check shapes are consistent
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
            # Stack along a new axis (axis=0) and compute the mean
            combined = np.mean(np.stack(prob_maps, axis=0), axis=0)
            return combined.astype(np.float32)

        elif strategy == 'majority':
            # This assumes binary segmentation where channel 1 is the target class
            threshold = 0.5
            num_classes = first_shape[-1]

            if num_classes < 2:
                logger.error("Majority voting requires at least 2 output channels (bg/fg).")
                return None

            # Threshold each probability map's foreground channel (channel 1)
            predictions = [(p[:, :, 1] > threshold).astype(np.uint8) for p in prob_maps]
            # Stack thresholded predictions
            stacked_preds = np.stack(predictions, axis=0)
            # Sum along the model axis and check if sum > half the models
            combined_majority = (np.sum(stacked_preds, axis=0) > len(prob_maps) / 2).astype(np.uint8)

            # Reconstruct a probability-like map (0.0 or 1.0)
            h, w = combined_majority.shape
            output_prob_map = np.zeros(first_shape, dtype=np.float32) # Use original shape
            output_prob_map[:, :, 1] = combined_majority.astype(np.float32)
            output_prob_map[:, :, 0] = 1.0 - output_prob_map[:, :, 1]
            # If more than 2 classes, other channels remain 0, which might need adjustment
            # depending on the specific multi-class majority logic desired.
            if num_classes > 2:
                logger.warning("Majority voting applied only to channel 1 vs 0 for >2 classes.")

            return output_prob_map

        else:
            logger.warning(f"Unsupported combination strategy: {strategy}. Falling back to 'average'.")
            combined = np.mean(np.stack(prob_maps, axis=0), axis=0)
            return combined.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during model prediction combination (strategy: {strategy}): {e}")
        logger.error(traceback.format_exc()) # Include traceback for debugging
        return None