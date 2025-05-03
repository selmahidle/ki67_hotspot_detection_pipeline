import os
import glob
import logging
import traceback
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils import convert_batchnorm_to_groupnorm 


logger = logging.getLogger(__name__)

def load_models_from_subdirs(base_dir, model_type, encoder, device, apply_groupnorm=False):
    model_list = []
    subdirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    for subdir in subdirs:
        model = create_model(model_type, encoder, None, num_classes=2, activation='sigmoid')
        if apply_groupnorm:
            model = convert_batchnorm_to_groupnorm(model)
        model.to(device)
        model = load_latest_checkpoint(model, subdir, device)
        if model is not None:
            model.to(device)
            model_list.append(model)
    return model_list
    

def create_model(model_type, encoder, encoder_weights, num_classes, activation):
    """
    Creates a segmentation model using the segmentation-models-pytorch library.
    """
    model_class = getattr(smp, model_type, None)
    if model_class is None:
        raise ValueError(f"Invalid model type specified: {model_type}. Not found in segmentation_models_pytorch.")

    logger.info(f"Creating model: {model_type} with encoder: {encoder}, classes: {num_classes}, activation: {activation}")

    try:
        model_params = {
            'encoder_name': encoder, 'encoder_weights': encoder_weights,
            'in_channels': 3, 'classes': num_classes, 'activation': activation,
        }
        if model_type == "DeepLabV3Plus":
             model_params['encoder_output_stride'] = 16
             model_params['decoder_atrous_rates'] = (12, 24, 36)
             logger.info("Applying DeepLabV3Plus specific parameters: encoder_output_stride=16, decoder_atrous_rates=(12, 24, 36)")
        model = model_class(**model_params)
        logger.info(f"Successfully created model: {model_type} with encoder {encoder}")
        return model
    except Exception as e:
        logger.error(f"Error creating model {model_type} with encoder {encoder}: {e}", exc_info=True)
        raise


def load_model_checkpoint(model, checkpoint_path, device, *, weights_only=True): # Added weights_only arg with default True
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

        # --- Extract state_dict ---
        state_dict = None
        if isinstance(checkpoint, dict):
            # --- Simplified state_dict extraction ---
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            if state_dict is checkpoint: # Check if get() returned the checkpoint itself (i.e., keys not found)
                 logger.warning("Checkpoint dict lacks 'model_state_dict' or 'state_dict'. Assuming dict IS the state_dict.")
            elif state_dict is not None:
                 logger.debug(f"Loaded state_dict from key '{'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'}'.")
            # --------------------------------------
        elif isinstance(checkpoint, nn.Module):
             logger.warning("Checkpoint is full nn.Module object. Extracting state_dict.")
             state_dict = checkpoint.state_dict()
        else:
            logger.warning("Checkpoint not dict or nn.Module. Assuming object IS state_dict.")
            state_dict = checkpoint

        if state_dict is None or not isinstance(state_dict, dict):
             logger.error(f"Could not extract a valid state dictionary from checkpoint: {checkpoint_path}")
             raise ValueError("Invalid checkpoint format: Cannot extract state_dict.")

        # --- Clean state_dict keys ---
        cleaned_state_dict = {}
        is_data_parallel = isinstance(model, nn.DataParallel)
        logger.debug(f"Model is DataParallel: {is_data_parallel}")

        for k, v in state_dict.items():
            new_key = k
            if k.startswith('module.') and not is_data_parallel:
                new_key = k[len('module.'):]
                # logger.log(logging.DEBUG - 1, f"Removed 'module.' prefix: {k} -> {new_key}")
            elif not k.startswith('module.') and is_data_parallel:
                new_key = 'module.' + k
                # logger.log(logging.DEBUG - 1, f"Added 'module.' prefix: {k} -> {new_key}")

            if new_key.startswith('encoder.model.'):
                # old_key_debug = new_key
                new_key = 'encoder.' + new_key[len('encoder.model.'):]
                # logger.log(logging.DEBUG - 1, f"Adjusted 'encoder.model.' prefix: {old_key_debug} -> {new_key}")

            cleaned_state_dict[new_key] = v.float() if isinstance(v, torch.Tensor) else v

        # --- Load state_dict into model ---
        model.float()
        try:
            # Use new assignment API for load_state_dict in recent PyTorch versions
            load_info = model.load_state_dict(cleaned_state_dict, strict=True)
            strict_success = True
            # Check returned info object for issues
            if hasattr(load_info, 'missing_keys') and load_info.missing_keys:
                 logger.warning(f"Strict loading reported missing keys: {load_info.missing_keys}")
                 strict_success = False
            if hasattr(load_info, 'unexpected_keys') and load_info.unexpected_keys:
                 logger.warning(f"Strict loading reported unexpected keys: {load_info.unexpected_keys}")
                 strict_success = False

            if strict_success:
                 logger.info(f"Loaded checkpoint '{Path(checkpoint_path).name}' successfully (strict).")
            else:
                 # If strict loading finished but reported issues, still try non-strict for better logging maybe
                 logger.warning("Strict loading had issues. Attempting non-strict loading for more info...")
                 load_info_nonstrict = model.load_state_dict(cleaned_state_dict, strict=False)
                 logger.info(f"Loaded checkpoint '{Path(checkpoint_path).name}' successfully (non-strict).")
                 if load_info_nonstrict.missing_keys: logger.warning(f"  Non-strict - Missing: {load_info_nonstrict.missing_keys}")
                 if load_info_nonstrict.unexpected_keys: logger.warning(f"  Non-strict - Unexpected: {load_info_nonstrict.unexpected_keys}")

        except RuntimeError as e: # Catch errors during strict loading attempt
             logger.warning(f"Strict loading failed for {Path(checkpoint_path).name}: {e}. Attempting non-strict loading.")
             try:
                 load_info_nonstrict = model.load_state_dict(cleaned_state_dict, strict=False)
                 logger.info(f"Loaded checkpoint '{Path(checkpoint_path).name}' successfully (non-strict).")
                 if load_info_nonstrict.missing_keys: logger.warning(f"  Non-strict - Missing: {load_info_nonstrict.missing_keys}")
                 if load_info_nonstrict.unexpected_keys: logger.warning(f"  Non-strict - Unexpected: {load_info_nonstrict.unexpected_keys}")
             except Exception as e_nonstrict:
                  logger.error(f"Non-strict loading also failed for {Path(checkpoint_path).name}: {e_nonstrict}", exc_info=True)
                  raise e_nonstrict # Re-raise the non-strict error

        model.to(device)
        logger.debug(f"Model placed on device {device} after loading checkpoint.")
        return model

    except FileNotFoundError: raise
    except pickle.UnpicklingError as pe: # Catch the specific error # IMPORT pickle needed
         logger.error(f"UnpicklingError loading checkpoint {checkpoint_path} (likely weights_only issue): {pe}", exc_info=True)
         raise pe # Re-raise after logging
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

    # --- Determine if weights_only=False should be used --- # MODIFIED SECTION
    # Use os.path.normpath to handle potential extra slashes
    # Use os.path.basename to get the final directory component
    # Use lower() for case-insensitive comparison (safer)
    norm_dir = os.path.normpath(checkpoint_dir)
    base_dir_name = os.path.basename(norm_dir)
    # --- More robust check ---
    use_weights_only_false = "model1_fra_henrik" in base_dir_name.lower() # Check lowercase

    # --- Determine final flag and add Debug Prints ---
    final_weights_only_flag = not use_weights_only_false
    # print(f"\n--- DEBUG: load_latest_checkpoint ---") # Optional: Use logger.debug instead of print
    # print(f"Input checkpoint_dir: {checkpoint_dir}")
    # print(f"Normalized base_dir_name: {base_dir_name}")
    # print(f"Checking for 'model1_fra_henrik' in lowercase: {use_weights_only_false}")
    # print(f"Flag use_weights_only_false: {use_weights_only_false}")
    # print(f"Calling load_model_checkpoint with weights_only = {final_weights_only_flag}")
    # print(f"-------------------------------------\n")
    logger.debug(f"Checkpoint Dir: {checkpoint_dir}, Base: {base_dir_name}, Use weights_only=False Flag: {use_weights_only_false}, Final weights_only value: {final_weights_only_flag}") # Use logger instead
    # ------------------------------------------------------

    try:
        # --- Pass the correctly determined weights_only flag ---
        model = load_model_checkpoint(model, latest_checkpoint, device, weights_only=final_weights_only_flag) # PASS ARGUMENT
        # -------------------------------------------------------
        return model
    except Exception as e:
        logger.error(f"Failed to load latest checkpoint from {latest_checkpoint}.")
        # The specific error was already logged in load_model_checkpoint
        return None