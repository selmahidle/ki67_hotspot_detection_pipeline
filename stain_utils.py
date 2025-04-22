import logging
import traceback
import numpy as np
from skimage.color import rgb2hed, separate_stains, combine_stains 
from skimage.exposure import rescale_intensity


logger = logging.getLogger(__name__)


# Define the H&E-DAB stain matrix based on user values or common estimates.
# Values represent the optical density for [Red, Green, Blue] channels for each stain.
# Row 0: Hematoxylin
# Row 1: DAB (Diaminobenzidine)
# Row 2: Residual (often representing Eosin or background counterstain)
# These values might need tuning based on the specific staining protocol.
custom_hed_from_rgb = np.array([
    [0.651, 0.701, 0.29],  # Hematoxylin OD vector
    [0.269, 0.568, 0.778],  # DAB OD vector
    [0.709, 0.423, 0.561]   # Residual/Eosin OD vector (example, adjust as needed)
    # Reference: Ruifrok AC, Johnston DA. Quantification of histochemical
    # staining by color deconvolution. Anal Quant Cytol Histol. 2001 Aug;23(4):291-9.
    # Default skimage Hematoxylin=[0.65, 0.70, 0.29], Eosin=[0.07, 0.99, 0.11], DAB=[0.27, 0.57, 0.78]
    # Our custom matrix uses H, DAB, Residual. Skimage default uses H, E, D.
])

# Calculate the inverse matrix for stain separation (RGB from HED)
# This matrix is used by skimage.color.separate_stains
try:
    custom_rgb_from_hed = np.linalg.inv(custom_hed_from_rgb)
    logger.info("Successfully calculated inverse matrix for custom H&E-DAB stains.")
except np.linalg.LinAlgError:
    logger.error("Custom H&E-DAB stain matrix is singular (cannot invert). Falling back to default skimage rgb2hed.")
    custom_rgb_from_hed = None # Signal to use default rgb2hed


def get_dab_mask(rgb_image, dab_threshold=0.15):
    """
    Performs color deconvolution on an RGB image patch and returns a binary
    mask identifying DAB-positive pixels based on a threshold applied to the
    raw DAB optical density.

    Uses a custom H-DAB-Residual stain matrix if available and valid, otherwise
    falls back to scikit-image's standard rgb2hed function.

    Args:
        rgb_image (np.ndarray): Input image patch in RGB format. Can be uint8 (0-255)
                               or float (0-1).
        dab_threshold (float): Optical density threshold for classifying a pixel
                               as DAB positive. Pixels with DAB OD > threshold
                               are marked. Default is 0.15.

    Returns:
        np.ndarray: A binary mask (uint8, values 0 or 1) of the same height
                    and width as the input image, where 1 indicates DAB positive.
                    Returns an empty mask (all zeros) on error or invalid input.
    """
    if rgb_image is None:
        logger.error("Input rgb_image to get_dab_mask is None.")
        return None # Indicate error

    logger.debug(f"Getting DAB mask with threshold {dab_threshold}...")

    try:
        # --- Input Validation and Conversion ---
        # Ensure input is numpy array
        if not isinstance(rgb_image, np.ndarray):
             try:
                  rgb_image = np.array(rgb_image)
             except Exception as e:
                  logger.error(f"Could not convert input image to numpy array: {e}")
                  return np.zeros(rgb_image.shape[:2], dtype=np.uint8) # Try to return empty mask

        # Ensure input is uint8, 0-255 range for deconvolution functions
        if rgb_image.dtype != np.uint8:
            logger.debug(f"Input image dtype is {rgb_image.dtype}, converting to uint8 for deconvolution.")
            if np.max(rgb_image) <= 1.0 and rgb_image.dtype.kind == 'f':
                # Float 0-1 range
                rgb_uint8 = (rgb_image * 255).astype(np.uint8)
            else:
                # Try direct conversion, clip values to be safe
                try:
                    rgb_uint8 = np.clip(rgb_image, 0, 255).astype(np.uint8)
                except ValueError as e:
                    logger.error(f"Cannot convert image (dtype: {rgb_image.dtype}, range: [{np.min(rgb_image)}, {np.max(rgb_image)}]) to uint8 for deconvolution: {e}. Returning empty mask.")
                    return np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        else:
            rgb_uint8 = rgb_image

        # Handle blank/constant image patches (cannot deconvolve)
        if np.max(rgb_uint8) == np.min(rgb_uint8):
             logger.debug("Input image is constant (blank). Returning empty DAB mask.")
             return np.zeros(rgb_uint8.shape[:2], dtype=np.uint8)

        # Check shape
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
             logger.error(f"Input image must be 3-channel RGB. Got shape {rgb_uint8.shape}. Returning empty mask.")
             return np.zeros(rgb_uint8.shape[:2], dtype=np.uint8)


        # --- Perform Color Deconvolution ---
        dab_channel_od = None
        if custom_rgb_from_hed is not None:
            # --- Use Custom Stain Separation ---
            logger.debug("Attempting stain separation using custom H-DAB-Residual matrix.")
            try:
                 # separate_stains expects the *inverse* matrix (RGB from HED)
                 stains_custom_od = separate_stains(rgb_uint8, custom_rgb_from_hed)
                 # In our custom matrix: H=[0], DAB=[1], Residual=[2]
                 dab_channel_od = stains_custom_od[:, :, 1]
                 logger.debug("Successfully separated stains using custom matrix.")
            except Exception as sep_err:
                 logger.warning(f"Custom stain separation failed: {sep_err}. Falling back to default skimage rgb2hed.")
                 # Fall through to use default rgb2hed below

        if dab_channel_od is None:
            # --- Use Default Skimage rgb2hed ---
            logger.debug("Using default skimage rgb2hed for stain separation.")
            # rgb2hed returns optical density for H, E, D stains
            hed_od = rgb2hed(rgb_uint8)
            # DAB is the third channel (index 2) in the standard HED matrix
            dab_channel_od = hed_od[:, :, 2]
            logger.debug("Successfully separated stains using default rgb2hed.")

        # Log some stats about the raw DAB OD values
        if dab_channel_od is not None:
            logger.debug(f"Raw DAB OD stats - Min: {np.min(dab_channel_od):.4f}, Max: {np.max(dab_channel_od):.4f}, Mean: {np.mean(dab_channel_od):.4f}")
        else:
             # This shouldn't happen if logic above is correct, but handle defensively
             logger.error("Failed to obtain DAB channel OD after attempting both methods.")
             return np.zeros(rgb_uint8.shape[:2], dtype=np.uint8)


        # --- Threshold Raw Optical Density ---
        # Apply threshold directly to the raw DAB optical density channel
        # Pixels with OD *greater* than the threshold are considered positive
        dab_mask = (dab_channel_od > dab_threshold).astype(np.uint8)

        logger.debug(f"Applied threshold {dab_threshold} to raw DAB OD. Positive pixels found: {np.sum(dab_mask)}")
        return dab_mask

    except ValueError as e:
        # Specific errors often raised by skimage deconvolution on problematic images
        logger.warning(f"Color deconvolution failed likely due to problematic pixel values (e.g., pure white/black): {e}. Returning empty DAB mask.")
        # Return empty mask matching input shape if possible
        if hasattr(rgb_image, 'shape') and len(rgb_image.shape) >= 2:
            return np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        else:
             return None # Cannot determine shape
    except Exception as e:
        logger.error(f"Unexpected error in get_dab_mask: {e}")
        logger.error(traceback.format_exc())
        # Return empty mask matching input shape if possible
        if hasattr(rgb_image, 'shape') and len(rgb_image.shape) >= 2:
            return np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        else:
            return None