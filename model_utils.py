import os
import glob
import logging
import traceback
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from monai.networks.nets import AttentionUnet
from utils import convert_batchnorm_to_groupnorm


logger = logging.getLogger(__name__)

def load_models_from_subdirs(base_dir, model_type, encoder, device, apply_groupnorm=False):
    model_list = []
    subdirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    for subdir in subdirs:
        model = create_model(model_type, encoder, None, num_classes=2, activation='sigmoid')
        if apply_groupnorm:
            model = convert_batchnorm_to_groupnorm(model)
        model = load_latest_checkpoint(model, subdir, device)
        if model is not None:
            model.to(device) 
            model_list.append(model)
    return model_list


import segmentation_models_pytorch as smp
import logging

# It's good practice to define the logger at the module level
logger = logging.getLogger(__name__)

def create_model(model_type, encoder, encoder_weights, num_classes, activation):
    """
    Creates a segmentation model using the segmentation-models-pytorch library.

    Args:
        model_type (str): The type of the model (e.g., "Unet", "DeepLabV3Plus").
        encoder (str): The name of the encoder (e.g., "resnet18").
        encoder_weights (str or None): Pre-trained weights for the encoder (e.g., "imagenet", None).
        num_classes (int): The number of output classes.
        activation (str or callable or None): Activation function for the output layer.
                                            For binary segmentation with logits output (e.g. for BCEWithLogitsLoss),
                                            this should be None. For multi-class, could be "softmax".

    Returns:
        torch.nn.Module: The created segmentation model.

    Raises:
        ValueError: If an invalid model_type is specified.
        Exception: If model creation fails for other reasons.
    """
    model_class = getattr(smp, model_type, None)
    if model_class is None:
        logger.error(f"Invalid model type specified: {model_type}. Not found in segmentation_models_pytorch.")
        raise ValueError(f"Invalid model type specified: {model_type}. Not found in segmentation_models_pytorch.")

    logger.info(f"Attempting to create model: {model_type} with encoder: {encoder}, "
                f"encoder_weights: {encoder_weights}, classes: {num_classes}, activation: {activation}")

    try:
        model_params = {
            'encoder_name': encoder,
            'encoder_weights': encoder_weights,
            'in_channels': 3, 
            'classes': num_classes,
            'activation': activation,
        }

        if model_type == "DeepLabV3Plus":
            model_params['encoder_output_stride'] = 16
            model_params['decoder_atrous_rates'] = (12, 24, 36)
            logger.info(f"Setting DeepLabV3Plus specific parameters: "
                        f"encoder_output_stride={model_params['encoder_output_stride']}, "
                        f"decoder_atrous_rates={model_params['decoder_atrous_rates']}")

        model = model_class(**model_params)
        logger.info(f"Successfully created model: {model_type} with encoder: {encoder}, "
                    f"classes: {num_classes}, activation: {activation}.")
        return model
    except Exception as e:
        logger.error(f"Error creating model {model_type} with encoder {encoder}: {e}", exc_info=True)
        raise


def create_and_load_attention_unet(checkpoint_path, device, weights_only_load=True):
    """
    Creates an AttentionUNet model with predefined architecture, loads weights
    from a checkpoint, and sets it to evaluation mode.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (torch.device): Device to load the model and checkpoint onto.
        weights_only_load (bool): Passed to torch.load's weights_only argument for load_model_checkpoint.

    Returns:
        torch.nn.Module: The loaded AttentionUNet model, or None on failure.
    """
    logger.info(f"Creating MONAI AttentionUNet for cell segmentation.")
    try:
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3
        ).to(device)

        logger.info(f"Loading AttentionUNet checkpoint from: {checkpoint_path}")

        model = load_model_checkpoint(model, checkpoint_path, device, weights_only=weights_only_load)
        
        if model is not None:
            model.eval()
            logger.info("Successfully created and loaded AttentionUNet model.")
            return model
        else:
            logger.error("Failed to load checkpoint into AttentionUNet model.")
            return None

    except Exception as e:
        logger.error(f"Error creating or loading AttentionUNet model: {e}", exc_info=True)
        return None


def load_model_checkpoint(model, checkpoint_path, device, *, weights_only=True):
    """
    Loads a model state dictionary from a .pth checkpoint file.

    Args:
        model (torch.nn.Module): Model instance.
        checkpoint_path (str): Path to the .pth file.
        device (torch.device): Device to load onto.
        weights_only (bool): Passed to torch.load. Set to False for older checkpoints
                             that include non-tensor data, IF THE SOURCE IS TRUSTED.
                             Defaults to True for security.
    Returns:
        torch.nn.Module: Model with loaded weights.
    Raises:
        FileNotFoundError, ValueError, Exception
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path} onto device: {device}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)

        state_dict = None
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            if state_dict is checkpoint:
                 logger.warning("Checkpoint dict lacks 'model_state_dict' or 'state_dict'. Assuming dict IS the state_dict.")
            elif state_dict is not None:
                 logger.debug(f"Loaded state_dict from key '{'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'}'.")
        elif isinstance(checkpoint, nn.Module):
             logger.warning("Checkpoint is full nn.Module object. Extracting state_dict.")
             state_dict = checkpoint.state_dict()
        else:
            logger.warning("Checkpoint not dict or nn.Module. Assuming object IS state_dict.")
            state_dict = checkpoint

        if state_dict is None or not isinstance(state_dict, dict):
             logger.error(f"Could not extract a valid state dictionary from checkpoint: {checkpoint_path}")
             raise ValueError("Invalid checkpoint format: Cannot extract state_dict.")

        cleaned_state_dict = {}
        is_data_parallel = isinstance(model, nn.DataParallel)
        logger.debug(f"Model is DataParallel: {is_data_parallel}")

        for k, v in state_dict.items():
            new_key = k
            if k.startswith('module.') and not is_data_parallel:
                new_key = k[len('module.'):]
            elif not k.startswith('module.') and is_data_parallel:
                new_key = 'module.' + k

            if new_key.startswith('encoder.model.'):
                new_key = 'encoder.' + new_key[len('encoder.model.'):]

            cleaned_state_dict[new_key] = v.float() if isinstance(v, torch.Tensor) else v
        
        model.float()
        try:
            load_info = model.load_state_dict(cleaned_state_dict, strict=True)
            strict_success = True
            if hasattr(load_info, 'missing_keys') and load_info.missing_keys:
                 logger.warning(f"Strict loading reported missing keys: {load_info.missing_keys}")
                 strict_success = False
            if hasattr(load_info, 'unexpected_keys') and load_info.unexpected_keys:
                 logger.warning(f"Strict loading reported unexpected keys: {load_info.unexpected_keys}")
                 strict_success = False

            if strict_success:
                 logger.info(f"Loaded checkpoint '{Path(checkpoint_path).name}' successfully (strict).")
            else:
                 logger.warning("Strict loading had issues. Attempting non-strict loading for more info...")
                 load_info_nonstrict = model.load_state_dict(cleaned_state_dict, strict=False)
                 logger.info(f"Loaded checkpoint '{Path(checkpoint_path).name}' successfully (non-strict).")
                 if load_info_nonstrict.missing_keys: logger.warning(f"  Non-strict - Missing: {load_info_nonstrict.missing_keys}")
                 if load_info_nonstrict.unexpected_keys: logger.warning(f"  Non-strict - Unexpected: {load_info_nonstrict.unexpected_keys}")

        except RuntimeError as e:
             logger.warning(f"Strict loading failed for {Path(checkpoint_path).name}: {e}. Attempting non-strict loading.")
             try:
                 load_info_nonstrict = model.load_state_dict(cleaned_state_dict, strict=False)
                 logger.info(f"Loaded checkpoint '{Path(checkpoint_path).name}' successfully (non-strict).")
                 if load_info_nonstrict.missing_keys: logger.warning(f"  Non-strict - Missing: {load_info_nonstrict.missing_keys}")
                 if load_info_nonstrict.unexpected_keys: logger.warning(f"  Non-strict - Unexpected: {load_info_nonstrict.unexpected_keys}")
             except Exception as e_nonstrict:
                  logger.error(f"Non-strict loading also failed for {Path(checkpoint_path).name}: {e_nonstrict}", exc_info=True)
                  raise e_nonstrict

        model.to(device)
        logger.debug(f"Model placed on device {device} after loading checkpoint.")
        return model

    except FileNotFoundError: raise
    except pickle.UnpicklingError as pe:
         logger.error(f"UnpicklingError loading checkpoint {checkpoint_path} (likely weights_only issue or corrupted file): {pe}", exc_info=True)
         logger.info("If this error persists and the checkpoint source is trusted, try loading with weights_only=False.")
         raise pe
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}", exc_info=True)
        raise


def load_latest_checkpoint(model, checkpoint_dir, device):
    """
    Finds the latest modified .pth file in a directory and loads it into the model.
    Determines if weights_only=False should be used based on directory name.
    """
    logger.info(f"Searching for latest checkpoint in: {checkpoint_dir}")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        logger.error(f"No checkpoints (.pth files) found in {checkpoint_dir}")
        return None

    try:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    except Exception as e:
        logger.error(f"Error finding latest checkpoint in {checkpoint_dir}: {e}")
        return None

    norm_dir = os.path.normpath(checkpoint_dir)
    base_dir_name = os.path.basename(norm_dir)
    use_weights_only_false = True
    final_weights_only_flag = not use_weights_only_false
    logger.debug(f"Checkpoint Dir: {checkpoint_dir}, Base: {base_dir_name}, Use weights_only=False Flag: {use_weights_only_false}, Final weights_only value: {final_weights_only_flag}")

    model = load_model_checkpoint(model, latest_checkpoint, device, weights_only=final_weights_only_flag)
    return model