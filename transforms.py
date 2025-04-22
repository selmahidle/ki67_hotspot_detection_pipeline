# transforms.py

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class NormalizeImage(A.ImageOnlyTransform):
    """
    Albumentations transform to normalize image pixel values to the 0-1 range.

    Handles both integer (e.g., 0-255) and float inputs. If the image range
    is zero (constant image), it attempts a standard float conversion.
    Ensures the output is float32 for PyTorch compatibility.
    """
    def __init__(self, p=1.0): 
        super(NormalizeImage, self).__init__(always_apply=False, p=p)

    def apply(self, img, **params):
        if img is None:
            logger.error("Input image to NormalizeImage is None.")
            return None 

        min_value = np.min(img)
        max_value = np.max(img)

        if max_value > min_value:
            # Normalize to 0-1 range
            normalized = (img.astype(np.float32) - min_value) / (max_value - min_value)
        elif max_value == min_value:
            logger.warning("Image has constant value. Normalizing by dividing by 255 if max > 1, else assuming already 0-1 float.")
            # Handle constant images: if it looks like uint8, divide by 255, else assume float
            if max_value > 1:
                normalized = img.astype(np.float32) / 255.0
            else:
                normalized = img.astype(np.float32) # Assume already in 0-1 range
        else: # Should not happen if min/max work correctly
             normalized = img.astype(np.float32)

        # Ensure output is float32 for PyTorch
        return normalized.astype(np.float32)


def apply_clahe(image, clip_limit=4.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Converts image to LAB color space, applies CLAHE to the L channel,
    and converts back to RGB. Handles grayscale and RGBA inputs by converting
    them to RGB first. Ensures input is uint8 for OpenCV CLAHE function.

    Args:
        image (np.ndarray or PIL Image): Input image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.

    Returns:
        np.ndarray: The CLAHE-enhanced image in RGB format (uint8).
    """
    if image is None:
        logger.error("Input image to apply_clahe is None.")
        return None

    if not isinstance(image, np.ndarray):
        try:
            image = np.array(image)
        except Exception as e:
            logger.error(f"Could not convert input to numpy array for CLAHE: {e}")
            return image # Return original input

    # --- Convert to uint8 ---
    # Check if float (0-1 range) and scale
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0 and np.min(image) >= 0.0 and image.dtype.kind == 'f':
            logger.debug("CLAHE input is float 0-1, converting to uint8.")
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            try:
                # Try direct conversion, might work for int types or fail for others
                logger.debug(f"CLAHE input is {image.dtype}, attempting direct conversion to uint8.")
                image_uint8 = image.astype(np.uint8)
            except ValueError:
                 logger.error(f"Cannot convert image (dtype: {image.dtype}, range: [{np.min(image)}, {np.max(image)}]) to uint8 for CLAHE. Returning original.")
                 return image # Return original if conversion fails
    else:
        image_uint8 = image

    # --- Ensure RGB format ---
    if len(image_uint8.shape) == 2: # Grayscale
        logger.debug("CLAHE input is grayscale, converting to RGB.")
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    elif len(image_uint8.shape) == 3 and image_uint8.shape[2] == 4: # RGBA
        logger.debug("CLAHE input is RGBA, converting to RGB.")
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_RGBA2RGB)
    elif len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3: # RGB
        image_rgb = image_uint8
    else:
         logger.error(f"CLAHE expects Grayscale, RGB, or RGBA image, got shape {image_uint8.shape}. Returning original.")
         return image # Return original non-uint8 image

    # --- Apply CLAHE ---
    try:
        # Convert RGB to LAB
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)

        # Merge channels and convert back to RGB
        limg = cv2.merge((cl, a, b))
        final_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        logger.debug("CLAHE applied successfully.")
        return final_rgb
    except cv2.error as e:
         logger.error(f"OpenCV error during CLAHE processing: {e}")
         return image_rgb # Return RGB image if CLAHE fails
    except Exception as e:
         logger.error(f"Unexpected error during CLAHE: {e}")
         return image_rgb # Return RGB image


class ApplyCLAHE(A.ImageOnlyTransform):
    """
    Albumentations-compatible transform to apply CLAHE using the apply_clahe function.
    """
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=1.0): # Removed always_apply default from signature
        # Pass always_apply=False to the superclass, let p handle probability
        super(ApplyCLAHE, self).__init__(always_apply=False, p=p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        logger.debug(f"ApplyCLAHE transform initialized with clip_limit={clip_limit}, tile_grid_size={tile_grid_size}, p={p}")

    def apply(self, img, **params):
        return apply_clahe(img, self.clip_limit, self.tile_grid_size)


def get_transforms():
    """
    Returns the standard Albumentations transform pipeline including CLAHE,
    Normalization, and conversion to a PyTorch tensor.

    Used for Tumor Segmentation stage in the original script.
    """
    return A.Compose([
        ApplyCLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), # Use p=1.0
        NormalizeImage(p=1.0), # Use p=1.0
        ToTensorV2()
    ])

def get_transforms_no_clahe():
    """
    Returns an Albumentations transform pipeline including only Normalization
    and conversion to a PyTorch tensor.

    Used for Cell Segmentation stage in the original script.
    """
    return A.Compose([
        NormalizeImage(p=1.0), # Use p=1.0
        ToTensorV2()
    ])