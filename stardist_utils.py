import logging
import numpy as np
import warnings
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.transform import resize, rescale
from skimage.util import img_as_float


logger = logging.getLogger(__name__)

STARDIST_MODEL_NAME = '2D_versatile_he'
PROB_THRESH = 0.5   # QuPath: Detection probability threshold 0.5 fra før
NMS_THRESH = 0.3
TARGET_PIXEL_SIZE_UM = 0.2  # QuPath: Requested pixel size (µm) 0.3 fra før
NORM_PERC_LOW = 1.0
NORM_PERC_HIGH = 99.0
SIZE_FILTER_FACTOR = 7.0
MIN_ABSOLUTE_SIZE = 10
ACTUAL_PIXEL_SIZE_MICRONS = 1.0 # <- use 0.23 if running on a full slide


def load_stardist_model(model_name=STARDIST_MODEL_NAME):
    logger.info(f"Loading StarDist model: {model_name}")
    try:
        model = StarDist2D.from_pretrained(model_name)
        logger.info(f"StarDist model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading StarDist model: {e}", exc_info=True)
        return None


def predict_patch_stardist(model: StarDist2D,
                           image_patch_rgb: np.ndarray,
                           actual_pixel_size_um: float = ACTUAL_PIXEL_SIZE_MICRONS, 
                           target_pixel_size_um: float = TARGET_PIXEL_SIZE_UM, 
                           prob_thresh: float = PROB_THRESH, 
                           nms_thresh: float = NMS_THRESH,   
                           normalize_perc_low: float = NORM_PERC_LOW,
                           normalize_perc_high: float = NORM_PERC_HIGH,
                           apply_size_filter: bool = True, 
                           size_filter_factor: float = SIZE_FILTER_FACTOR,
                           min_absolute_size: int = MIN_ABSOLUTE_SIZE):
    """
    Runs StarDist prediction on a single RGB image patch using predefined default
    parameters matching the reference script, unless overridden. Requires the
    actual pixel size of the input patch.

    Args:
        model: Loaded StarDist2D model.
        image_patch_rgb: Input RGB image patch (H, W, 3).
        actual_pixel_size_um: Actual pixel size of the input patch (microns).
        target_pixel_size_um: Target pixel size for rescaling. Defaults match script.
        prob_thresh: Probability threshold. Defaults match script.
        nms_thresh: NMS threshold. Defaults match script.
        normalize_perc_low: Normalization percentile. Defaults match script.
        normalize_perc_high: Normalization percentile. Defaults match script.
        apply_size_filter: Apply size filtering post-prediction. Defaults to True.
        size_filter_factor: Relative size filter factor. Defaults match script.
        min_absolute_size: Absolute min object size. Defaults match script.

    Returns:
        tuple: (labels_filtered, details) as described previously.
               Returns (None, None) on error.
    """

    if model is None or image_patch_rgb is None: return None, None
    if not isinstance(image_patch_rgb, np.ndarray) or image_patch_rgb.ndim != 3 or image_patch_rgb.shape[2] != 3: return None, None
    if actual_pixel_size_um is None:
        logger.error("actual_pixel_size_um must be provided for predict_patch_stardist.")
        return None, None

    original_shape = image_patch_rgb.shape[:2]
    logger.debug(f"Processing patch {original_shape}, actual um/pix: {actual_pixel_size_um}")

    try:
        # 1. Float conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            img_float = img_as_float(image_patch_rgb)

        # 2. Rescaling
        img_to_process = img_float
        rescale_factor = 1.0
        if abs(actual_pixel_size_um - target_pixel_size_um) > 1e-6:
            rescale_factor = actual_pixel_size_um / target_pixel_size_um
            logger.debug(f" Rescaling by {rescale_factor:.4f}")
            img_to_process = rescale(img_float, rescale_factor, anti_aliasing=True, mode='reflect', preserve_range=True, channel_axis=-1)
            img_to_process = np.clip(img_to_process, 0, 1)

        # 3. Normalization
        img_norm = normalize(img_to_process, normalize_perc_low, normalize_perc_high, axis=(0, 1))

        # 4. Prediction
        labels_pred_scaled, details = model.predict_instances(img_norm,  prob_thresh=prob_thresh, nms_thresh=nms_thresh)

        # 5. Inverse Rescaling
        if labels_pred_scaled is None: return None, None # Check prediction result
        labels_pred_orig_size = None
        if abs(rescale_factor - 1.0) > 1e-6:
            labels_pred_orig_size = resize(labels_pred_scaled, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
        else:
            labels_pred_orig_size = labels_pred_scaled.astype(np.uint16)
            if labels_pred_orig_size.shape != original_shape: # Correct shape just in case
                 labels_pred_orig_size = resize(labels_pred_orig_size, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)

        # 6. Size Filtering
        labels_filtered = labels_pred_orig_size
        if apply_size_filter and labels_pred_orig_size.max() > 0:
            labels_filtered = labels_pred_orig_size.copy()
            object_labels, object_sizes = np.unique(labels_filtered[labels_filtered > 0], return_counts=True)
            if len(object_sizes) > 0:
                max_size = np.max(object_sizes)
                relative_size_threshold = max_size / size_filter_factor
                final_size_threshold = max(relative_size_threshold, min_absolute_size)
                small_object_labels = object_labels[object_sizes < final_size_threshold]
                if len(small_object_labels) > 0:
                    mask_to_remove = np.isin(labels_filtered, small_object_labels)
                    labels_filtered[mask_to_remove] = 0
        # Note: Returning unfiltered 'details' dictionary

        logger.debug(f" StarDist patch prediction finished. Found {len(details['coord'])} objects before filtering, {len(np.unique(labels_filtered[labels_filtered>0]))} after.")
        return labels_filtered, details

    except Exception as e:
        logger.error(f"Error during StarDist prediction pipeline for patch: {e}", exc_info=True)
        return None, None