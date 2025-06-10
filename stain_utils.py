import logging
import numpy as np
from skimage.color import rgb2hed, separate_stains

logger = logging.getLogger(__name__)

custom_hed_from_rgb = np.array([
    [0.651, 0.701, 0.29],
    [0.269, 0.568, 0.778],
    [0.709, 0.423, 0.561]
])

try:
    custom_rgb_from_hed = np.linalg.inv(custom_hed_from_rgb)
except np.linalg.LinAlgError:
    logger.error("Custom stain matrix inversion failed. Using default.")
    custom_rgb_from_hed = None

def get_dab_mask(rgb_image, dab_threshold=0.15):
    if rgb_image is None:
        logger.error("Input rgb_image is None.")
        return None

    try:
        if not isinstance(rgb_image, np.ndarray):
            rgb_image = np.array(rgb_image)

        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image * 255 if np.max(rgb_image) <= 1 else rgb_image, 0, 255).astype(np.uint8)

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            logger.error("Input must be a 3-channel RGB image.")
            return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

        if custom_rgb_from_hed is not None:
            try:
                stains = separate_stains(rgb_image, custom_rgb_from_hed)
                dab_od = stains[:, :, 1]
            except:
                dab_od = rgb2hed(rgb_image)[:, :, 2]
        else:
            dab_od = rgb2hed(rgb_image)[:, :, 2]

        return (dab_od > dab_threshold).astype(np.uint8)

    except Exception as e:
        logger.error(f"Error generating DAB mask: {e}")
        return np.zeros(rgb_image.shape[:2], dtype=np.uint8)
