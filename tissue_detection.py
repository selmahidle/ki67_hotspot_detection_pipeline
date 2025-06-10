import logging
import numpy as np
import cv2
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu

logger = logging.getLogger(__name__)

def calculate_entropy_mask(gray_image, disk_radius=5):
    if gray_image is None or gray_image.ndim != 2:
        logger.error("Input must be a 2D grayscale image.")
        return None

    gray_image_uint8 = np.clip(gray_image * 255 if gray_image.max() <= 1 else gray_image, 0, 255).astype(np.uint8)

    try:
        entropy_img = entropy(gray_image_uint8, disk(disk_radius))
        entropy_scaled = ((entropy_img - entropy_img.min()) / (entropy_img.ptp()) * 255).astype(np.uint8)
        threshold = threshold_otsu(entropy_scaled)
        return (entropy_scaled > threshold).astype(np.uint8) * 255
    except Exception as e:
        logger.error(f"Entropy calculation failed: {e}")
        return np.zeros_like(gray_image_uint8)


def detect_tissue(image_bgr, threshold_tissue_ratio=0.05,
                  threshold_intensity_low=70, threshold_intensity_high=230,
                  entropy_disk_radius=5,
                  morph_kernel_small_size=3,
                  morph_kernel_medium_size=7,
                  morph_kernel_large_size=21,
                  min_contour_area_small=500,
                  min_hole_area=2000,
                  min_background_island_area=1000):

    if image_bgr is None or image_bgr.ndim < 2:
        logger.error("Invalid input image.")
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr

    tissue_mask = calculate_entropy_mask(gray, entropy_disk_radius)
    if tissue_mask is None:
        return np.zeros(gray.shape, dtype=np.uint8)

    background_mask = ((gray < threshold_intensity_low) | (gray > threshold_intensity_high)).astype(np.uint8) * 255
    tissue_mask[background_mask > 0] = 0

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_small_size, morph_kernel_small_size))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_medium_size, morph_kernel_medium_size))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_large_size, morph_kernel_large_size))

    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_medium)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_contour_area_small:
            tissue_mask[labels == i] = 0

    inv_mask = cv2.bitwise_not(tissue_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_hole_area:
            tissue_mask[labels == i] = 255

    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_large)

    inv_mask_final = cv2.bitwise_not(tissue_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask_final, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_background_island_area:
            tissue_mask[labels == i] = 255

    if threshold_tissue_ratio > 0 and np.mean(tissue_mask > 0) < threshold_tissue_ratio:
        return np.zeros_like(tissue_mask)

    return tissue_mask
