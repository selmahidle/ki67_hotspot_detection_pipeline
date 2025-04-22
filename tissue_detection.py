import logging
import numpy as np
import cv2
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu


logger = logging.getLogger(__name__)


def calculate_entropy_mask(gray_image, disk_radius=5):
    """
    Calculates an entropy mask to highlight textured regions (likely tissue).

    Applies entropy filtering and then thresholds the result using Otsu's method
    to separate high-entropy (textured) regions from low-entropy (background) regions.

    Args:
        gray_image (np.ndarray): A 2D grayscale numpy array (uint8).
        disk_radius (int): The radius of the disk structuring element for
                           entropy calculation.

    Returns:
        np.ndarray: A binary mask (uint8, values 0 or 255) where high-entropy
                    regions are marked as 255. Returns an empty mask on error.
    """
    if gray_image is None or gray_image.ndim != 2:
        logger.error("Invalid input: calculate_entropy_mask requires a 2D grayscale image.")
        return None # Return None to indicate error

    # Ensure input is uint8 for entropy filter compatibility if needed
    if gray_image.dtype != np.uint8:
        logger.warning(f"Input image dtype is {gray_image.dtype}, converting to uint8 for entropy filter.")
        # Check range before converting
        if np.max(gray_image) <= 1.0 and gray_image.dtype.kind == 'f':
            gray_image_uint8 = (gray_image * 255).astype(np.uint8)
        else:
            try:
                # Attempt direct conversion, clip if necessary
                gray_image_uint8 = np.clip(gray_image, 0, 255).astype(np.uint8)
            except ValueError:
                 logger.error("Cannot safely convert input image to uint8. Returning empty mask.")
                 return np.zeros_like(gray_image, dtype=np.uint8)
    else:
        gray_image_uint8 = gray_image

    logger.debug(f"Calculating entropy with disk radius {disk_radius}...")
    try:
        # Apply entropy filter
        entropy_image = entropy(gray_image_uint8, disk(disk_radius))

        # Check for constant entropy image (can happen with blank inputs)
        if np.max(entropy_image) == np.min(entropy_image):
             logger.warning("Entropy image is constant. Thresholding might be meaningless. Returning empty mask.")
             return np.zeros_like(gray_image_uint8, dtype=np.uint8)

        # --- Thresholding using Otsu ---
        # Scale entropy image to 0-255 range for Otsu if it's not already
        if np.max(entropy_image) > 0: # Avoid division by zero
             entropy_scaled = ( (entropy_image - np.min(entropy_image)) /
                               (np.max(entropy_image) - np.min(entropy_image)) * 255 ).astype(np.uint8)
        else: # If max is 0, it's already constant (handled above, but safe check)
             entropy_scaled = entropy_image.astype(np.uint8)


        # Calculate Otsu threshold on the scaled entropy image
        otsu_thresh_value = threshold_otsu(entropy_scaled)
        logger.debug(f"Otsu threshold for entropy mask: {otsu_thresh_value}")

        # Create binary mask: regions *above* the threshold are considered tissue
        mask = (entropy_scaled > otsu_thresh_value).astype(np.uint8) * 255
        logger.debug(f"Entropy mask created. Positive pixels: {np.sum(mask > 0)}")
        return mask

    except Exception as e:
        logger.error(f"Error calculating entropy mask: {e}")
        # Return an empty mask of the same size as input
        return np.zeros_like(gray_image_uint8, dtype=np.uint8)


def detect_tissue(image_bgr, threshold_tissue_ratio=0.05,
                  threshold_intensity_low=70, threshold_intensity_high=230,
                  entropy_disk_radius=5,
                  morph_kernel_small_size=3,
                  morph_kernel_medium_size=7,
                  morph_kernel_large_size=21,
                  min_contour_area_small=500,
                  min_hole_area=2000,
                  min_background_island_area=1000):
    """
    Detects tissue regions in an image using intensity and entropy filtering,
    followed by morphological operations for cleaning.

    Args:
        image_bgr (np.ndarray): Input image in BGR format (as read by OpenCV).
        threshold_tissue_ratio (float): Minimum required tissue area ratio relative
                                        to the total image area. If below this, an
                                        empty mask is returned. Set to 0 to disable.
        threshold_intensity_low (int): Pixels below this intensity in grayscale are
                                      considered background.
        threshold_intensity_high (int): Pixels above this intensity in grayscale are
                                       considered background.
        entropy_disk_radius (int): Radius for the entropy filter disk.
        morph_kernel_small_size (int): Size for the smallest morphological kernel.
        morph_kernel_medium_size (int): Size for the medium morphological kernel.
        morph_kernel_large_size (int): Size for the large morphological kernel.
        min_contour_area_small (int): Minimum area (pixels) for small tissue
                                      contours to be kept.
        min_hole_area (int): Minimum area (pixels) for holes within tissue
                             to be filled.
        min_background_island_area (int): Minimum area (pixels) for small background
                                          islands (within tissue after closing)
                                          to be filled (removed).

    Returns:
        np.ndarray or None: Binary mask (uint8, values 0 or 255) where tissue is 255.
                           Returns None if input is invalid.
                           Returns an empty mask if tissue ratio is below threshold.
    """
    if image_bgr is None or image_bgr.ndim < 2:
        logger.error("Invalid input image provided to detect_tissue.")
        return None

    logger.info("Starting classical tissue detection...")
    h, w = image_bgr.shape[:2]
    logger.debug(f"Input image dimensions: {w}x{h}")

    # --- 1. Convert to Grayscale ---
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
         logger.warning("Input image is BGRA, converting to grayscale.")
         gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
    elif image_bgr.ndim == 2:
        logger.info("Input image is already grayscale.")
        gray = image_bgr
    else:
        logger.error(f"Unsupported image format/shape for grayscale conversion: {image_bgr.shape}")
        return None

    # --- 2. Initial Tissue Estimation using Entropy ---
    # Calculate entropy mask (imported helper function)
    tissue_mask_entropy = calculate_entropy_mask(gray, disk_radius=entropy_disk_radius)
    if tissue_mask_entropy is None:
        logger.error("Failed to calculate entropy mask.")
        return np.zeros((h, w), dtype=np.uint8) # Return empty mask

    # --- 3. Identify Background using Intensity Thresholds ---
    logger.debug(f"Identifying background using intensity thresholds: <{threshold_intensity_low} or >{threshold_intensity_high}")
    below_thresh_mask = (gray < threshold_intensity_low)
    above_thresh_mask = (gray > threshold_intensity_high)
    background_mask = np.logical_or(below_thresh_mask, above_thresh_mask).astype(np.uint8) * 255
    logger.debug(f"Background mask created. Positive pixels: {np.sum(background_mask > 0)}")

    # --- 4. Remove Background from Entropy Mask ---
    # Start with the entropy mask and subtract the definite background
    tissue_mask_combined = tissue_mask_entropy.copy()
    tissue_mask_combined[background_mask > 0] = 0
    logger.debug("Removed intensity-based background from entropy mask.")

    # --- 5. Morphological Cleaning ---
    # Define kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_small_size, morph_kernel_small_size))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_medium_size, morph_kernel_medium_size))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_large_size, morph_kernel_large_size))
    logger.debug("Defined morphological kernels.")

    # Initial closing to connect nearby regions (use medium kernel)
    tissue_mask_cleaned = cv2.morphologyEx(tissue_mask_combined, cv2.MORPH_CLOSE, kernel_medium)
    logger.debug(f"Applied initial closing (kernel size: {morph_kernel_medium_size}).")

    # Remove small isolated tissue fragments
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask_cleaned, connectivity=8)
    filtered_contours_small = 0
    if num_labels > 1: # If there are components beyond the background (label 0)
        for i in range(1, num_labels): # Iterate through components (skip background label 0)
            if stats[i, cv2.CC_STAT_AREA] < min_contour_area_small:
                tissue_mask_cleaned[labels == i] = 0
                filtered_contours_small += 1
    logger.debug(f"Removed {filtered_contours_small} small tissue components (<{min_contour_area_small} pixels).")

    # Fill small holes within tissue regions
    # Invert mask to find holes
    mask_inv = cv2.bitwise_not(tissue_mask_cleaned)
    num_labels_holes, labels_holes, stats_holes, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    filled_holes = 0
    if num_labels_holes > 1:
        # Identify the background component in the inverted mask (usually the largest)
        # background_label_inv = np.argmax(stats_holes[1:, cv2.CC_STAT_AREA]) + 1 # Find largest component excluding label 0
        # Assume label 0 in the inverted image corresponds to the actual tissue from the original
        for i in range(1, num_labels_holes): # Iterate through components in inverted mask (holes)
             # if i == background_label_inv: continue # Skip the main background component
             if stats_holes[i, cv2.CC_STAT_AREA] < min_hole_area:
                tissue_mask_cleaned[labels_holes == i] = 255 # Fill the hole
                filled_holes += 1
    logger.debug(f"Filled {filled_holes} small holes (<{min_hole_area} pixels).")


    # Additional closing to further merge regions and smooth contours (use large kernel)
    tissue_mask_cleaned = cv2.morphologyEx(tissue_mask_cleaned, cv2.MORPH_CLOSE, kernel_large)
    logger.debug(f"Applied large closing (kernel size: {morph_kernel_large_size}).")

    # Remove small background "islands" that might have been created by closing
    # This is similar to filling holes, but applied again after the large closing
    mask_inv_final = cv2.bitwise_not(tissue_mask_cleaned)
    num_labels_bg, labels_bg, stats_bg, _ = cv2.connectedComponentsWithStats(mask_inv_final, connectivity=8)
    filled_bg_islands = 0
    if num_labels_bg > 1:
        for i in range(1, num_labels_bg):
            if stats_bg[i, cv2.CC_STAT_AREA] < min_background_island_area:
                tissue_mask_cleaned[labels_bg == i] = 255 # Fill background island
                filled_bg_islands += 1
    logger.debug(f"Filled {filled_bg_islands} small background islands (<{min_background_island_area} pixels).")


    # --- 6. Verify Minimum Tissue Ratio ---
    if threshold_tissue_ratio > 0:
        tissue_pixels = np.sum(tissue_mask_cleaned > 0)
        total_pixels = h * w
        tissue_ratio = tissue_pixels / total_pixels if total_pixels > 0 else 0
        logger.info(f"Final tissue ratio: {tissue_ratio:.4f} (Threshold: {threshold_tissue_ratio})")
        if tissue_ratio < threshold_tissue_ratio:
            logger.warning(f"Tissue ratio is below threshold. Returning empty mask.")
            return np.zeros((h, w), dtype=np.uint8)

    logger.info("Classical tissue detection finished successfully.")
    return tissue_mask_cleaned