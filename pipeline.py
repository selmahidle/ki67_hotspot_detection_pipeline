import os
import traceback
import logging
from pathlib import Path
import numpy as np
import cv2
import openslide
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm
import sys
from transforms import get_transforms, get_transforms_no_clahe
from tissue_detection import detect_tissue
from patching import run_inference_on_patches
from utils import create_weight_map, get_actual_pixel_size_um
from stain_utils import get_dab_mask
from hotspot_detection import identify_hotspots
from visualization import generate_overlay
from stardist_utils import refine_hotspot_with_stardist

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None


# Potentially in utils.py or directly in pipeline.py if not used elsewhere

def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are (x, y, w, h).
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) 
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) 

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) 
    return iou


def apply_nms_to_candidates(candidates, iou_threshold=0.55):
    """
    Apply Non-Maximum Suppression to hotspot candidates.
    Assumes candidates are sorted by density_score in descending order.
    """
    if not candidates:
        return []

    boxes = np.array([[c['coords_level'][0], c['coords_level'][1], c['size_level'][0], c['size_level'][1]] for c in candidates])
    scores = np.array([c['density_score'] for c in candidates]) 
    pick_indices = []

    original_indices = list(range(len(candidates)))

    while len(original_indices) > 0:
        current_original_idx = original_indices[0]
        pick_indices.append(current_original_idx)
        current_box = boxes[current_original_idx]
        remaining_original_indices_next_iter = []
        
        for i in range(1, len(original_indices)):
            other_original_idx = original_indices[i]
            other_box = boxes[other_original_idx]
            
            iou = calculate_iou(current_box, other_box)
            if iou < iou_threshold:
                remaining_original_indices_next_iter.append(other_original_idx)
        
        original_indices = remaining_original_indices_next_iter

    final_candidates = [candidates[i] for i in pick_indices]
    return final_candidates


def process_slide_ki67(slide_path, output_dir, tumor_models, cell_models, stardist_model, device):
    """
    Process a slide for Ki67 analysis using tumor segmentation, cell segmentation (SMP),
    and StarDist-based hotspot refinement including TWO-PASS sub-region analysis for dense hotspots.

    Pipeline:
    1. Load slide, save overviews (ds8, ds4).
    2. Detect tissue (L5).
    3. Segment tumor (SMP using ds8).
    4. Segment cells (SMP using ds4), conditioned on tumor mask -> cell_mask_binary_l2.
    5. Calculate DAB+ mask within tumor regions at L2 -> dab_plus_mask_l2.
    6. Identify *candidate* hotspots based on density of (SMP Cells AND DAB+) at L2.
    7. *Refine* candidates using stardist_utils.refine_hotspot_with_stardist (returns counts & centroids).
    8. Process refined candidates:
        - If cells > max_cells, attempt Pass 1 sub-sampling (e.g., 384x384).
        - If Pass 1 still > max_cells, attempt Pass 2 sub-sampling (e.g., 256x256) within Pass 1 region.
        - Filter final list to keep only hotspots (original or sub-sampled) with cells between min_cells-max_cells.
    9. Re-rank final hotspots based on refined StarDist Ki67+ counts.
    10. Generate final overlay (L2).
    """
    logger.info(f"Processing slide: {os.path.basename(slide_path)}")
    slide = None
    try:
        # --- Setup ---
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        image_output_dir = os.path.join(output_dir, slide_name)
        os.makedirs(image_output_dir, exist_ok=True)
        debug_mask_dir = os.path.join(image_output_dir, "debug_masks")
        os.makedirs(debug_mask_dir, exist_ok=True)
        hotspot_debug_dir = os.path.join(image_output_dir, f"{slide_name}_hotspot_debug")
        os.makedirs(hotspot_debug_dir, exist_ok=True)
        overlay_debug_dir = os.path.join(image_output_dir, f"{slide_name}_overlay_debug")
        os.makedirs(overlay_debug_dir, exist_ok=True)

        # --- Load slide ---
        slide = openslide.open_slide(slide_path)

        # --- Define levels and Parameters ---
        # Levels
        tissue_level = 5; tumor_level = 3; cell_level = 2; hotspot_level = 2
        overlay_level = 2; ds8_level = 3; ds4_level = 2

        # Tumor stage parameters
        tumor_patch_size = 4096; tumor_overlap = 1024; tumor_output_channels = 2; tumor_batch_size = 2; tumor_prob_threshold = 0.3

        # Cell stage parameters (SMP)
        cell_patch_size = 1024; cell_overlap = 256; cell_output_channels = 2; cell_batch_size = 8; cell_prob_threshold = 0.3

        # Hotspot parameters
        hotspot_patch_size_l0 = 2048 # Original candidate FoV size at L0
        hotspot_top_n = 5            # Final number of hotspots to return AFTER filtering
        hotspot_dab_threshold = 0.15 # Threshold for get_dab_mask
        min_cells = 500              # Minimum total cells for a FINAL hotspot
        max_cells = 600              # Maximum total cells for a FINAL hotspot
        # Parameters for sub-region analysis
        sub_patch_size_l2_pass1 = 384 # Target size for Pass 1 sub-sampling
        sub_patch_size_l2_pass2 = 256 # Target size for Pass 2 sub-sampling
        sub_stride = 64               # Stride for sliding window during sub-sampling

        # --- Check Levels Exist ---
        required_levels_actual = {'tissue': tissue_level, 'ds8': ds8_level, 'ds4': ds4_level}
        max_req_level = max(required_levels_actual.values())
        if max_req_level >= slide.level_count:
            logger.error(f"Slide {slide_name} level count ({slide.level_count}) insufficient. Max required level index: {max_req_level}.")
            if slide: slide.close(); return None

        # --- Get Transforms ---
        transforms_clahe = get_transforms(); transforms_no_clahe = get_transforms_no_clahe()

        # === Pre-computation: Save/Load ds8 and ds4 JPGs === #
        # --- ds8 ---
        ds8_path = os.path.join(image_output_dir, f"{slide_name}_ds8.jpg")
        ds8_img = None
        # ... (ds8 loading/generation logic - unchanged) ...
        if os.path.exists(ds8_path):
            logger.info(f"Found existing ds8 image: {ds8_path}")
            ds8_img_bgr = cv2.imread(ds8_path)
            if ds8_img_bgr is None: logger.error(f"Failed to load {ds8_path}.")
            else: ds8_img = cv2.cvtColor(ds8_img_bgr, cv2.COLOR_BGR2RGB)
        if ds8_img is None:
            logger.info(f"Generating ds8 image (Level {ds8_level})...")
            if ds8_level >= slide.level_count: logger.error(f"L{ds8_level} not available."); return None
            try:
                 ds8_dims = slide.level_dimensions[ds8_level]
                 ds8_img_pil = slide.read_region((0,0), ds8_level, ds8_dims).convert('RGB')
                 ds8_img = np.array(ds8_img_pil)
                 cv2.imwrite(ds8_path, cv2.cvtColor(ds8_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                 logger.info(f"Saved ds8 image to {ds8_path}")
            except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError generating ds8: {ose}", exc_info=True); return None
            except Exception as e: logger.error(f"Failed to generate/save ds8: {e}", exc_info=True); return None
        if ds8_img is None: logger.error("ds8 image could not be loaded or generated. Cannot proceed."); return None

        # --- DS4 ---
        ds4_path = os.path.join(image_output_dir, f"{slide_name}_ds4.jpg")
        ds4_img = None
        # ... (ds4 loading/generation logic - unchanged) ...
        logger.info(f"--- DS4 Start: Checking path {ds4_path} ---")
        if os.path.exists(ds4_path):
            logger.info(f"Attempting to load existing ds4 image: {ds4_path}")
            try:
                ds4_img_bgr = cv2.imread(ds4_path)
                if ds4_img_bgr is None: logger.error(f"cv2.imread returned None for existing ds4 file: {ds4_path}")
                else:
                    logger.info(f"Successfully loaded existing ds4 image via cv2, shape: {ds4_img_bgr.shape}")
                    ds4_img = cv2.cvtColor(ds4_img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as load_err: logger.error(f"Exception during cv2.imread for ds4: {load_err}", exc_info=True); ds4_img = None
        if ds4_img is None:
            logger.info(f"Attempting to generate ds4 image (Level {ds4_level})...")
            if ds4_level >= slide.level_count: logger.error(f"Required ds4 Level {ds4_level} not available."); return None
            try:
                ds4_dims = slide.level_dimensions[ds4_level]
                logger.info(f"Reading region for ds4: level={ds4_level}, dims={ds4_dims}")
                ds4_img_pil = slide.read_region((0,0), ds4_level, ds4_dims).convert('RGB')
                logger.info("slide.read_region for ds4 succeeded.")
                ds4_img = np.array(ds4_img_pil)
                logger.info(f"Converted ds4 PIL to numpy, shape: {ds4_img.shape}, dtype: {ds4_img.dtype}")
                if ds4_img is None or ds4_img.size == 0: logger.error("Numpy array for ds4 empty!"); ds4_img = None
                else:
                    logger.info(f"Attempting to write ds4 image to {ds4_path}")
                    success = cv2.imwrite(ds4_path, cv2.cvtColor(ds4_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if success: logger.info(f"Successfully saved generated ds4 image.")
                    else: logger.error(f"cv2.imwrite failed for ds4!"); ds4_img = None
            except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError during ds4 generation: {ose}", exc_info=True); ds4_img = None
            except MemoryError as me: logger.error(f"MemoryError during ds4 generation: {me}", exc_info=True); raise me
            except Exception as gen_err: logger.error(f"Unhandled Exception during ds4 generation: {gen_err}", exc_info=True); ds4_img = None
        if ds4_img is None: logger.error("Failed to load or generate ds4 image. Cannot proceed."); return None
        else: logger.info(f"--- DS4 End: Successfully obtained ds4_img, shape {ds4_img.shape} ---")


        # === Stages 1-7 (Tissue, Tumor, Cells, DAB, Initial Hotspots) ===
        # ... (Code for stages 1-7 is assumed to be unchanged and correct) ...
        logger.info(f"--- Stage 1: Tissue Detection (L{tissue_level}) ---")
        tissue_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tissue_mask_L{tissue_level}.jpg")
        tissue_mask_l5 = None
        if os.path.exists(tissue_mask_filename):
            tissue_mask_l5_255 = cv2.imread(tissue_mask_filename, cv2.IMREAD_GRAYSCALE)
            if tissue_mask_l5_255 is not None: tissue_mask_l5 = (tissue_mask_l5_255 > 128).astype(np.uint8); logger.info(f"Loaded tissue mask L{tissue_level}.")
            else: logger.error(f"Failed to load {tissue_mask_filename}.")
        if tissue_mask_l5 is None:
            logger.info(f"Generating tissue mask L{tissue_level}...")
            if tissue_level >= slide.level_count: logger.error(f"L{tissue_level} not available."); return None
            try:
                 tissue_level_dims = slide.level_dimensions[tissue_level]
                 tissue_img_pil = slide.read_region((0,0), tissue_level, tissue_level_dims).convert('RGB')
                 tissue_img_bgr = cv2.cvtColor(np.array(tissue_img_pil), cv2.COLOR_RGB2BGR)
                 tissue_mask_l5_255 = detect_tissue(tissue_img_bgr, threshold_tissue_ratio=0.05)
                 if tissue_mask_l5_255 is not None and np.sum(tissue_mask_l5_255) > 0:
                      tissue_mask_l5 = (tissue_mask_l5_255 > 0).astype(np.uint8)
                      cv2.imwrite(tissue_mask_filename, tissue_mask_l5 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                      logger.info(f"Saved tissue mask L{tissue_level}.")
                 else: logger.error("Tissue detection failed or result was empty.")
            except Exception as e: logger.error(f"Failed to generate tissue mask: {e}", exc_info=True)
        if tissue_mask_l5 is None or np.sum(tissue_mask_l5) == 0: logger.error("Tissue mask is empty. Cannot proceed."); return None
        tumor_level_h, tumor_level_w = ds8_img.shape[:2]
        tissue_mask_l3 = cv2.resize(tissue_mask_l5, (tumor_level_w, tumor_level_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Upsampled tissue mask to L{tumor_level} ({tissue_mask_l3.shape})")

        logger.info(f"--- Stage 2: Tumor Segmentation (L{tumor_level}) ---")
        processed_tumor_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_L{tumor_level}.jpg")
        consensus_tumor_mask_l3_raw = None
        tumor_mask_l3_processed = None
        skip_tumor_stage = False
        if os.path.exists(processed_tumor_mask_filename):
            logger.info(f"Found existing final tumor mask: {processed_tumor_mask_filename}. Loading.")
            loaded_mask = cv2.imread(processed_tumor_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None: logger.error(f"Failed to load existing tumor mask. Will run inference.")
            else:
                 consensus_tumor_mask_l3_raw = (loaded_mask > 128).astype(np.uint8)
                 tumor_mask_l3_processed = consensus_tumor_mask_l3_raw
                 logger.info(f"Loaded tumor mask L{tumor_level} ({consensus_tumor_mask_l3_raw.shape}).")
                 skip_tumor_stage = True
        else: logger.info(f"Final tumor mask not found. Running inference.")
        if not skip_tumor_stage:
            logger.info(f"Running tumor segmentation using ds8 JPG...")
            tumor_patches = []; tumor_locations = []; tumor_weights = []
            h, w = ds8_img.shape[:2]
            stride_x = tumor_patch_size - tumor_overlap; stride_y = tumor_patch_size - tumor_overlap
            if stride_x <= 0 or stride_y <= 0: logger.error(f"Invalid Tumor patch stride."); return None
            logger.info(f"Collecting tumor patches from ds8 ({w}x{h})...")
            for y in tqdm(range(0, h, stride_y), desc="Collecting Tumor Patches"):
                for x in range(0, w, stride_x):
                    y_start, x_start = y, x
                    end_y, end_x = min(y_start + tumor_patch_size, h), min(x_start + tumor_patch_size, w)
                    window_h, window_w = end_y - y_start, end_x - x_start
                    if window_w < stride_x // 2 or window_h < stride_y // 2: continue
                    patch = ds8_img[y_start:end_y, x_start:end_x];
                    if patch.shape[0] <= 0 or patch.shape[1] <= 0: continue
                    mask_patch = tissue_mask_l3[y_start:end_y, x_start:end_x]
                    if np.mean(mask_patch) < 0.05: continue
                    pad_h = max(0, tumor_patch_size - window_h); pad_w = max(0, tumor_patch_size - window_w)
                    padded_patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                    transformed = transforms_clahe(image=padded_patch); tumor_patches.append(transformed["image"])
                    tumor_locations.append((y_start, end_y, x_start, end_x))
                    weight_map = create_weight_map((window_h, window_w))
                    if weight_map is None: logger.error(f"Failed to create weight map for tumor patch. Skipping."); continue
                    if pad_h > 0 or pad_w > 0: padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    else: padded_weight_map = weight_map
                    tumor_weights.append(padded_weight_map)
            if not tumor_patches: logger.error(f"Tumor patch collection failed."); return None
            if not (len(tumor_patches) == len(tumor_locations) == len(tumor_weights)): logger.error(f"Tumor data mismatch!"); return None
            logger.info(f"Collected {len(tumor_patches)} tumor patches.")
            tumor_prob_maps_l3 = []
            for i, model in enumerate(tumor_models):
                model_name = f"TumorModel_{i+1}"; logger.info(f"Running tumor inference with {model_name}...")
                probs = run_inference_on_patches(model, device, tumor_output_channels, tumor_batch_size, f"Tumor L{tumor_level}", (w, h), tumor_patches, tumor_locations, tumor_weights, model_name)
                if probs is not None: tumor_prob_maps_l3.append(probs)
                else: logger.warning(f"{model_name} failed.")
            del tumor_patches, tumor_locations, tumor_weights
            if not tumor_prob_maps_l3: logger.error("All tumor models failed."); return None
            individual_model_masks = []
            num_successful_models = len(tumor_prob_maps_l3)
            logger.info(f"Processing {num_successful_models} tumor model outputs...")
            for i, prob_map in enumerate(tumor_prob_maps_l3):
                model_mask_bool = (prob_map[:, :, 1] > tumor_prob_threshold)
                individual_model_masks.append(model_mask_bool)
            if len(individual_model_masks) != num_successful_models or num_successful_models == 0: logger.error("Consensus failed."); return None
            consensus_mask_bool = np.logical_and.reduce(individual_model_masks)
            consensus_tumor_mask_l3_raw = consensus_mask_bool.astype(np.uint8) * tissue_mask_l3
            del tumor_prob_maps_l3, individual_model_masks, consensus_mask_bool
            logger.info(f"Raw tumor consensus mask generated. Pixels: {np.sum(consensus_tumor_mask_l3_raw)}")
            smooth_kernel_size = 5; kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
            tumor_mask_l3_processed = cv2.morphologyEx(consensus_tumor_mask_l3_raw, cv2.MORPH_CLOSE, kernel)
            tumor_mask_l3_processed = cv2.morphologyEx(tumor_mask_l3_processed, cv2.MORPH_OPEN, kernel)
            logger.info(f"Applied smoothing. Pixels after: {np.sum(tumor_mask_l3_processed)}")
            try: cv2.imwrite(processed_tumor_mask_filename, tumor_mask_l3_processed * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            except Exception as e_save: logger.error(f"Failed to save final processed tumor mask: {e_save}")
        if tumor_mask_l3_processed is None or np.sum(tumor_mask_l3_processed) == 0: logger.error("Tumor mask empty/invalid."); return None
        cell_level_h, cell_level_w = ds4_img.shape[:2]
        tumor_mask_l2 = cv2.resize(tumor_mask_l3_processed, (cell_level_w, cell_level_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Upsampled processed tumor mask to L{hotspot_level} ({tumor_mask_l2.shape})")

        logger.info(f"--- Stage 3: Cell Segmentation (SMP L{cell_level}) ---")
        cell_mask_filename = os.path.join(image_output_dir, f"{slide_name}_cell_mask_binary_L{cell_level}.jpg")
        cell_mask_binary_l2 = None
        skip_cell_stage = False
        if os.path.exists(cell_mask_filename):
            loaded_cell_mask = cv2.imread(cell_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_cell_mask is not None:
                 cell_mask_binary_l2 = (loaded_cell_mask > 128).astype(np.uint8)
                 skip_cell_stage = True; logger.info(f"Loaded existing SMP cell mask L{cell_level}.")
            else: logger.error(f"Failed to load {cell_mask_filename}. Will run inference.")
        if not skip_cell_stage:
            logger.info(f"Running SMP cell segmentation inference L{cell_level}...")
            cell_patches = []; cell_locations = []; cell_weights = []
            h, w = ds4_img.shape[:2]
            stride_x = cell_patch_size - cell_overlap; stride_y = cell_patch_size - cell_overlap
            if stride_x <= 0 or stride_y <= 0: logger.error(f"Invalid SMP cell patch stride."); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
            else:
                 logger.info(f"Collecting SMP cell patches from ds4 ({w}x{h})...")
                 for y in tqdm(range(0, h, stride_y), desc="Collecting Cell Patches"):
                     for x in range(0, w, stride_x):
                         y_start, x_start = y, x
                         end_y, end_x = min(y_start + cell_patch_size, h), min(x_start + cell_patch_size, w)
                         window_h, window_w = end_y - y_start, end_x - x_start
                         if window_w < stride_x//2 or window_h < stride_y//2: continue
                         patch = ds4_img[y_start:end_y, x_start:end_x]
                         if patch.shape[0] <= 0 or patch.shape[1] <= 0: continue
                         mask_patch = tumor_mask_l2[y_start:end_y, x_start:end_x]
                         if np.mean(mask_patch) < 0.05: continue
                         pad_h=max(0, cell_patch_size-window_h); pad_w=max(0, cell_patch_size-window_w)
                         padded_patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                         transformed = transforms_no_clahe(image=padded_patch); cell_patches.append(transformed["image"])
                         cell_locations.append((y_start, end_y, x_start, end_x))
                         weight_map = create_weight_map((window_h, window_w))
                         if weight_map is None: logger.error(f"Failed create weight map for cell patch. Skipping."); continue
                         if pad_h > 0 or pad_w > 0: padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                         else: padded_weight_map = weight_map
                         cell_weights.append(padded_weight_map)
                 if not cell_patches: logger.warning("SMP Cell patch collection failed."); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
                 elif not (len(cell_patches) == len(cell_locations) == len(cell_weights)): logger.error(f"Cell data mismatch!"); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
                 else:
                      logger.info(f"Collected {len(cell_patches)} SMP cell patches.")
                      cell_prob_maps_l2 = []
                      for i, model in enumerate(cell_models):
                           model_name = f"CellModel_{i+1}"; logger.info(f"Running SMP cell inference with {model_name}...")
                           probs = run_inference_on_patches(model, device, cell_output_channels, cell_batch_size, f"Cell L{cell_level}", (w, h), cell_patches, cell_locations, cell_weights, model_name)
                           if probs is not None: cell_prob_maps_l2.append(probs)
                           else: logger.warning(f"{model_name} failed.")
                      del cell_patches, cell_locations, cell_weights
                      if not cell_prob_maps_l2: logger.error("All SMP cell models failed."); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
                      else:
                           logger.info(f"Averaging {len(cell_prob_maps_l2)} SMP cell probability maps...")
                           combined_cell_prob = np.mean(cell_prob_maps_l2, axis=0).astype(np.float32)
                           del cell_prob_maps_l2
                           cell_mask_binary_l2_raw = (combined_cell_prob[:,:,1] > cell_prob_threshold).astype(np.uint8)
                           cell_mask_binary_l2 = cell_mask_binary_l2_raw * tumor_mask_l2
                           logger.info(f"Final SMP cell mask created. Pixels: {np.sum(cell_mask_binary_l2)}")
                           try: cv2.imwrite(cell_mask_filename, cell_mask_binary_l2 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                           except Exception as e_save: logger.error(f"Failed to save final cell mask: {e_save}")
        if cell_mask_binary_l2 is None: logger.warning("Cell mask None. Creating empty."); cell_mask_binary_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
        if np.sum(cell_mask_binary_l2) == 0: logger.warning("SMP Cell mask is empty.")

        logger.info(f"--- Stage 4: DAB+ Mask Calculation (L{hotspot_level}) ---")
        dab_plus_mask_l2 = None
        if np.sum(tumor_mask_l2) > 0:
            try:
                tumor_labels = label(tumor_mask_l2); tumor_props = regionprops(tumor_labels)
                if tumor_props:
                    min_r = min(prop.bbox[0] for prop in tumor_props); min_c = min(prop.bbox[1] for prop in tumor_props)
                    max_r = max(prop.bbox[2] for prop in tumor_props); max_c = max(prop.bbox[3] for prop in tumor_props)
                    bbox_h, bbox_w = max_r - min_r, max_c - min_c
                    logger.info(f"Tumor bounding box at L{hotspot_level}: ({min_c}, {min_r}, {bbox_w}, {bbox_h})")
                    if bbox_h > 0 and bbox_w > 0:
                        level0_x_read = int(min_c * slide.level_downsamples[hotspot_level]); level0_y_read = int(min_r * slide.level_downsamples[hotspot_level])
                        try:
                            logger.debug(f"Reading region for DAB: L0=({level0_x_read},{level0_y_read}), L={hotspot_level}, Size=({bbox_w},{bbox_h})")
                            rgb_patch_l2_pil = slide.read_region((level0_x_read, level0_y_read), hotspot_level, (bbox_w, bbox_h)).convert('RGB')
                            rgb_patch_l2 = np.array(rgb_patch_l2_pil)
                            if rgb_patch_l2.shape[0]!=bbox_h or rgb_patch_l2.shape[1]!=bbox_w: logger.warning(f"Read patch size mismatch for DAB. Resizing."); rgb_patch_l2=cv2.resize(rgb_patch_l2, (bbox_w,bbox_h), interpolation=cv2.INTER_LINEAR)
                            dab_plus_mask_patch = get_dab_mask(rgb_patch_l2, hotspot_dab_threshold)
                            if dab_plus_mask_patch is not None:
                                dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
                                dab_plus_mask_l2[min_r:max_r, min_c:max_c] = dab_plus_mask_patch
                                logger.info(f"Calculated DAB+ mask L2. Pixels: {np.sum(dab_plus_mask_l2)}")
                            else: logger.error("get_dab_mask returned None.")
                        except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError reading patch for DAB: {ose}", exc_info=True)
                        except Exception as e_read_dab: logger.error(f"Error reading/processing patch for DAB: {e_read_dab}", exc_info=True)
                    else: logger.warning(f"Invalid tumor bbox for DAB mask.")
                else: logger.warning("No tumor regions found for DAB.")
            except Exception as e_dab: logger.error(f"Error during DAB preparation: {e_dab}", exc_info=True)
        if dab_plus_mask_l2 is None: logger.warning("DAB+ mask failed/skipped. Using empty."); dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)

        logger.info(f"--- Stage 5: Initial Hotspot Candidate ID (L{hotspot_level}) ---")
        hotspot_target_mask_coarse = (cell_mask_binary_l2 > 0) & (dab_plus_mask_l2 > 0); hotspot_target_mask_coarse = hotspot_target_mask_coarse.astype(np.uint8)
        target_pixels = np.sum(hotspot_target_mask_coarse); logger.info(f"Coarse target mask (DAB+ & SMP+) positive pixels: {target_pixels}")
        candidate_hotspots = []
        if target_pixels == 0: logger.warning("Coarse target mask empty. No candidates found.")
        else:
             logger.info(f"Identifying initial hotspot candidates...")
             num_initial_candidates = 10 # Or adjust as needed
             candidate_hotspots = identify_hotspots(slide=slide, level=hotspot_level, hotspot_target_mask=hotspot_target_mask_coarse, hotspot_patch_size_l0=hotspot_patch_size_l0, top_n_hotspots=num_initial_candidates, debug_dir=hotspot_debug_dir)
             logger.info(f"Found {len(candidate_hotspots)} initial candidates.")

        # --- Apply NMS to initial candidates ---
        if candidate_hotspots:
            logger.info(f"Applying Non-Maximum Suppression to {len(candidate_hotspots)} initial candidates with IoU threshold 0.5...")
            candidates_after_nms = apply_nms_to_candidates(candidate_hotspots, iou_threshold=0.5)
            logger.info(f"Number of candidates after NMS: {len(candidates_after_nms)}")
            candidate_hotspots = candidates_after_nms 
        else:
            logger.warning("No initial candidates to apply NMS to.")

        # === 8. StarDist Refinement, Sub-sampling, Filtering, and Re-ranking ===
        hotspots = [] # Final ranked hotspots list
        if not candidate_hotspots:
            logger.warning("No initial candidates found for StarDist refinement.")
        else:
            logger.info(f"--- Stage 6: StarDist Refinement & Re-ranking (L{hotspot_level}) ---")
            refined_hotspots_results = [] # Holds the raw refined results

            refinement_debug_base_dir = os.path.join(hotspot_debug_dir, "refinement_patches")
            os.makedirs(refinement_debug_base_dir, exist_ok=True)
            actual_pixel_size_um = get_actual_pixel_size_um(slide, level=hotspot_level, fallback_value=0.25)
            if actual_pixel_size_um is None:
                logger.error("Failed to determine actual pixel size. Skipping StarDist refinement."); return None

            # --- Refine each candidate ---
            logger.info(f"Refining {len(candidate_hotspots)} initial candidates...")
            for i, candidate in enumerate(tqdm(candidate_hotspots, desc="Refining Hotspots")):

                updated_hotspot = refine_hotspot_with_stardist(
                    candidate_hotspot=candidate, stardist_model=stardist_model, slide=slide,
                    hotspot_level=hotspot_level, actual_pixel_size_um=actual_pixel_size_um,
                    dab_threshold=hotspot_dab_threshold, debug_dir=refinement_debug_base_dir,
                    candidate_index=i
                )

                if updated_hotspot is not None:
                    if all(k in updated_hotspot for k in ['stardist_ki67_pos_count', 'stardist_total_count_filtered', 'positive_centroids', 'all_centroids']):
                         refined_hotspots_results.append(updated_hotspot)
                         logger.debug(f"Cand {i+1} refined: Ki67+={updated_hotspot['stardist_ki67_pos_count']}, Total={updated_hotspot['stardist_total_count_filtered']}")
                    else: logger.warning(f"Refinement missing keys for cand {i+1}. Skipping.")
                else: logger.warning(f"Refinement failed for cand index {i+1}. Skipping.")

            # --- Process refined hotspots (sub-sampling & checks) ---
            processed_hotspots = [] # Holds hotspots ready for final filtering/ranking
            if refined_hotspots_results:
                logger.info(f"Processing {len(refined_hotspots_results)} refined hotspots...")

                for i, hs in enumerate(refined_hotspots_results):
                    total_cells_pass0 = hs.get('stardist_total_count_filtered', 0)
                    ki67_count_pass0 = hs.get('stardist_ki67_pos_count', 0)
                    candidate_id = i + 1 # For logging

                    # --- Apply Cell Count Logic ---
                    if total_cells_pass0 < min_cells:
                        logger.info(f"  Cand {candidate_id} (Ki67+ {ki67_count_pass0}): Discarding - Too few cells ({total_cells_pass0} < {min_cells})")
                        continue # DISCARD

                    elif min_cells <= total_cells_pass0 <= max_cells:
                        logger.debug(f"  Cand {candidate_id} (Ki67+ {ki67_count_pass0}): Keeping original - Cell count ({total_cells_pass0}) is within range [{min_cells}-{max_cells}]")
                        processed_hotspots.append(hs) # KEEP original

                    else: # total_cells_pass0 > max_cells
                        logger.info(f"  Cand {candidate_id} (Ki67+ {ki67_count_pass0}): Above cell range ({total_cells_pass0} > {max_cells}), attempting Pass 1 sub-sampling ({sub_patch_size_l2_pass1}x{sub_patch_size_l2_pass1})...")

                        # Retrieve centroids, check validity
                        positive_centroids = hs.get('positive_centroids')
                        all_centroids = hs.get('all_centroids')
                        if not isinstance(positive_centroids, list) or not isinstance(all_centroids, list) or not all_centroids:
                             logger.warning(f"  Cand {candidate_id}: Invalid/missing centroid data for Pass 1 sub-sampling. Discarding.")
                             continue # DISCARD

                        patch_h, patch_w = hs.get('size_level', (0,0))[::-1] # Original patch H, W
                        if patch_h < sub_patch_size_l2_pass1 or patch_w < sub_patch_size_l2_pass1:
                             logger.warning(f"  Cand {candidate_id}: Orig patch size ({patch_w}x{patch_h}) too small for Pass 1 ({sub_patch_size_l2_pass1}). Discarding.")
                             continue # DISCARD

                        # --- Pass 1 Sub-sampling ---
                        best_sub_region_pass1 = {'pos_count': -1, 'total_count': -1, 'y_rel': -1, 'x_rel': -1}
                        pos_centroids_np = np.array(positive_centroids) if positive_centroids else np.empty((0, 2))
                        all_centroids_np = np.array(all_centroids)

                        for y_rel1 in range(0, patch_h - sub_patch_size_l2_pass1 + 1, sub_stride):
                            for x_rel1 in range(0, patch_w - sub_patch_size_l2_pass1 + 1, sub_stride):
                                win_y_min1, win_y_max1 = y_rel1, y_rel1 + sub_patch_size_l2_pass1
                                win_x_min1, win_x_max1 = x_rel1, x_rel1 + sub_patch_size_l2_pass1
                                pos_count1 = 0
                                if pos_centroids_np.shape[0] > 0:
                                    pos_mask1 = (pos_centroids_np[:, 0] >= win_y_min1) & (pos_centroids_np[:, 0] < win_y_max1) & \
                                                (pos_centroids_np[:, 1] >= win_x_min1) & (pos_centroids_np[:, 1] < win_x_max1)
                                    pos_count1 = np.sum(pos_mask1)
                                if pos_count1 > best_sub_region_pass1['pos_count']:
                                    all_mask1 = (all_centroids_np[:, 0] >= win_y_min1) & (all_centroids_np[:, 0] < win_y_max1) & \
                                                (all_centroids_np[:, 1] >= win_x_min1) & (all_centroids_np[:, 1] < win_x_max1)
                                    total_count1 = np.sum(all_mask1)
                                    best_sub_region_pass1 = {'pos_count': pos_count1, 'total_count': total_count1, 'y_rel': y_rel1, 'x_rel': x_rel1}

                        # --- Check Pass 1 Results ---
                        N1 = best_sub_region_pass1['total_count']
                        Ki67_N1 = best_sub_region_pass1['pos_count']
                        if Ki67_N1 < 0 : # Should only happen if loop didn't run or find any window
                             logger.warning(f"  Cand {candidate_id}: Pass 1 sub-sampling failed to find any window. Discarding.")
                             continue # DISCARD
                        logger.info(f"    Pass 1 best sub-region ({sub_patch_size_l2_pass1}x{sub_patch_size_l2_pass1}): Ki67+={Ki67_N1}, Total={N1}")

                        if N1 < min_cells:
                            logger.info(f"    Pass 1 result discarded: Too few cells ({N1} < {min_cells})")
                            continue # DISCARD

                        elif min_cells <= N1 <= max_cells:
                            logger.info(f"    Pass 1 result accepted: Cell count ({N1}) is within range.")
                            # Create hotspot dict from Pass 1 results
                            sub_hotspot = {}
                            for key in ['slide_path', 'level', 'initial_density_score']:
                                 if key in hs: sub_hotspot[key] = hs[key]
                            orig_coords = hs.get('coords_level', (0,0))
                            sub_hotspot['coords_level'] = (orig_coords[0] + best_sub_region_pass1['x_rel'], orig_coords[1] + best_sub_region_pass1['y_rel'])
                            sub_hotspot['size_level'] = (sub_patch_size_l2_pass1, sub_patch_size_l2_pass1)
                            downsample_l2 = slide.level_downsamples[hotspot_level]
                            sub_hotspot['coords_l0'] = (int(sub_hotspot['coords_level'][0] * downsample_l2), int(sub_hotspot['coords_level'][1] * downsample_l2))
                            sub_hotspot['size_l0'] = (int(sub_hotspot['size_level'][0] * downsample_l2), int(sub_hotspot['size_level'][1] * downsample_l2))
                            sub_hotspot['stardist_ki67_pos_count'] = Ki67_N1
                            sub_hotspot['stardist_total_count_filtered'] = N1
                            sub_hotspot['stardist_proliferation_index'] = (Ki67_N1 / N1 if N1 > 0 else 0.0)
                            sub_hotspot['is_subsampled'] = True
                            sub_hotspot['subsample_pass'] = 1
                            processed_hotspots.append(sub_hotspot) # KEEP Pass 1 Result
                            continue # Go to next candidate

                        else: # N1 > max_cells
                             logger.info(f"    Pass 1 result ({N1} cells) still > {max_cells}. Attempting Pass 2 sub-sampling ({sub_patch_size_l2_pass2}x{sub_patch_size_l2_pass2})...")

                             # --- Pass 2 Sub-sampling (within Pass 1 best region) ---
                             best_sub_region_pass2 = {'pos_count': -1, 'total_count': -1, 'y_rel': -1, 'x_rel': -1}
                             # Define bounds for Pass 2 search (relative to original 512 patch)
                             pass1_y_start = best_sub_region_pass1['y_rel']
                             pass1_x_start = best_sub_region_pass1['x_rel']
                             pass1_h = sub_patch_size_l2_pass1
                             pass1_w = sub_patch_size_l2_pass1

                             if pass1_h < sub_patch_size_l2_pass2 or pass1_w < sub_patch_size_l2_pass2:
                                  logger.warning(f"  Cand {candidate_id}: Pass 1 region ({pass1_w}x{pass1_h}) too small for Pass 2 ({sub_patch_size_l2_pass2}). Discarding.")
                                  continue # DISCARD

                             for y_rel2 in range(0, pass1_h - sub_patch_size_l2_pass2 + 1, sub_stride):
                                 for x_rel2 in range(0, pass1_w - sub_patch_size_l2_pass2 + 1, sub_stride):
                                     # Window coordinates relative to ORIGINAL 512 patch
                                     win_y_min2 = pass1_y_start + y_rel2
                                     win_y_max2 = win_y_min2 + sub_patch_size_l2_pass2
                                     win_x_min2 = pass1_x_start + x_rel2
                                     win_x_max2 = win_x_min2 + sub_patch_size_l2_pass2

                                     pos_count2 = 0
                                     if pos_centroids_np.shape[0] > 0:
                                          pos_mask2 = (pos_centroids_np[:, 0] >= win_y_min2) & (pos_centroids_np[:, 0] < win_y_max2) & \
                                                      (pos_centroids_np[:, 1] >= win_x_min2) & (pos_centroids_np[:, 1] < win_x_max2)
                                          pos_count2 = np.sum(pos_mask2)

                                     if pos_count2 > best_sub_region_pass2['pos_count']:
                                          all_mask2 = (all_centroids_np[:, 0] >= win_y_min2) & (all_centroids_np[:, 0] < win_y_max2) & \
                                                      (all_centroids_np[:, 1] >= win_x_min2) & (all_centroids_np[:, 1] < win_x_max2)
                                          total_count2 = np.sum(all_mask2)
                                          best_sub_region_pass2 = {'pos_count': pos_count2, 'total_count': total_count2,
                                                                   'y_rel': y_rel2, # Relative to Pass 1 window start
                                                                   'x_rel': x_rel2} # Relative to Pass 1 window start

                             # --- Check Pass 2 Results ---
                             N2 = best_sub_region_pass2['total_count']
                             Ki67_N2 = best_sub_region_pass2['pos_count']
                             if Ki67_N2 < 0:
                                  logger.warning(f"  Cand {candidate_id}: Pass 2 sub-sampling failed to find any window. Discarding.")
                                  continue # DISCARD
                             logger.info(f"    Pass 2 best sub-region ({sub_patch_size_l2_pass2}x{sub_patch_size_l2_pass2}): Ki67+={Ki67_N2}, Total={N2}")

                             if N2 < min_cells or N2 > max_cells:
                                  logger.info(f"    Pass 2 result discarded: Cell count ({N2}) outside range [{min_cells}-{max_cells}]")
                                  continue # DISCARD
                             else: # min_cells <= N2 <= max_cells
                                  logger.info(f"    Pass 2 result accepted: Cell count ({N2}) is within range.")
                                  # Create hotspot dict from Pass 2 results
                                  sub_hotspot = {}
                                  for key in ['slide_path', 'level', 'initial_density_score']:
                                      if key in hs: sub_hotspot[key] = hs[key]
                                  orig_coords = hs.get('coords_level', (0,0))
                                  # Final coords = original + pass1_offset + pass2_offset
                                  final_x = orig_coords[0] + best_sub_region_pass1['x_rel'] + best_sub_region_pass2['x_rel']
                                  final_y = orig_coords[1] + best_sub_region_pass1['y_rel'] + best_sub_region_pass2['y_rel']
                                  sub_hotspot['coords_level'] = (final_x, final_y)
                                  sub_hotspot['size_level'] = (sub_patch_size_l2_pass2, sub_patch_size_l2_pass2) # Use Pass 2 size
                                  downsample_l2 = slide.level_downsamples[hotspot_level]
                                  sub_hotspot['coords_l0'] = (int(sub_hotspot['coords_level'][0] * downsample_l2), int(sub_hotspot['coords_level'][1] * downsample_l2))
                                  sub_hotspot['size_l0'] = (int(sub_hotspot['size_level'][0] * downsample_l2), int(sub_hotspot['size_level'][1] * downsample_l2))
                                  sub_hotspot['stardist_ki67_pos_count'] = Ki67_N2
                                  sub_hotspot['stardist_total_count_filtered'] = N2
                                  sub_hotspot['stardist_proliferation_index'] = (Ki67_N2 / N2 if N2 > 0 else 0.0)
                                  sub_hotspot['is_subsampled'] = True
                                  sub_hotspot['subsample_pass'] = 2
                                  processed_hotspots.append(sub_hotspot) # KEEP Pass 2 Result
                                  continue # Go to next candidate

            # --- Final Ranking of processed_hotspots ---
            if processed_hotspots:
                # Rank all potentially kept hotspots by Ki67+ count
                processed_hotspots.sort(key=lambda item: item.get('stardist_ki67_pos_count', 0), reverse=True)
                # Select the top N from the processed list
                hotspots = processed_hotspots[:hotspot_top_n]

                logger.info(f"Selected final {len(hotspots)} hotspots (ranked by Ki67+ count, counts are within [{min_cells}-{max_cells}]):")
                for rank, hs in enumerate(hotspots):
                    subsample_info = ""
                    if hs.get('is_subsampled', False):
                        subsample_info = f"* (Subsampled Pass {hs.get('subsample_pass', '?')})"
                    density_val = hs.get('density_score', 'N/A') # Use initial density score if available
                    density_str = f"{density_val:.4f}" if isinstance(density_val, (int, float)) else str(density_val)
                    logger.info(f"  Final Rank {rank+1}{subsample_info}: Ki67+={hs.get('stardist_ki67_pos_count','N/A')}, "
                                f"Total Cells={hs.get('stardist_total_count_filtered', 'N/A')}, "
                                f"PI={hs.get('stardist_proliferation_index', 0.0):.2%}, "
                                f"L0 Coords={hs.get('coords_l0')}, "
                                f"L{hs.get('level', hotspot_level)} Coords={hs.get('coords_level')}, "
                                f"Size L{hs.get('level', hotspot_level)}={hs.get('size_level')}, "
                                f"Initial Density={density_str}")
                    hs['final_score'] = hs.get('stardist_ki67_pos_count', 0) # Ensure final score is set
            else:
                hotspots = [] # Ensure empty list if none passed
                logger.warning("No hotspots available after processing/sub-sampling.")

        # === Stage 9 & 10 (Overlay Generation and Saving) ===
        # ... (Overlay generation and saving logic - unchanged) ...
        logger.info(f"--- Stage 7: Overlay Generation (L{overlay_level}) ---")
        overlay_h, overlay_w = ds4_img.shape[:2]
        target_overlay_shape_wh = (overlay_w, overlay_h)
        tissue_mask_overlay = cv2.resize(tissue_mask_l5, target_overlay_shape_wh, interpolation=cv2.INTER_NEAREST)
        tumor_mask_overlay = tumor_mask_l2
        cell_mask_for_overlay = cell_mask_binary_l2
        dab_mask_for_overlay = dab_plus_mask_l2
        final_overlay = generate_overlay(slide=slide, overlay_level=overlay_level, hotspot_level=hotspot_level, tissue_mask_overlay=tissue_mask_overlay, tumor_mask_overlay=tumor_mask_overlay, cell_mask_binary_l2=cell_mask_for_overlay, hotspots=hotspots, dab_mask_l2=dab_mask_for_overlay, debug_dir=overlay_debug_dir)
        if final_overlay is not None:
            overlay_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}.jpg")
            png_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}_temp.png")
            try:
                pil_image = Image.fromarray(final_overlay); pil_image.save(png_filename)
                with Image.open(png_filename) as img_png: img_png.convert('RGB').save(overlay_filename, quality=95)
                logger.info(f"Saved final overlay to {overlay_filename}")
                try: os.remove(png_filename)
                except OSError: pass
            except Exception as e:
                logger.error(f"Error saving overlay using PIL: {e}. Trying fallback...")
                try: cv2.imwrite(overlay_filename, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95]); logger.info(f"Saved overlay via cv2 fallback.")
                except Exception as e2: logger.error(f"Fallback overlay save failed: {e2}")
        else: logger.error("Failed to generate final overlay.")

        logger.info(f"--- Processing Finished for slide: {slide_name} ---")
        return hotspots

    # --- Error Handling & Cleanup ---
    # ... (Error handling and finally block - unchanged) ...
    except openslide.OpenSlideError as e: logger.error(f"OpenSlide error: {e}", exc_info=True); return None
    except FileNotFoundError as e: logger.error(f"File not found: {e}", exc_info=True); return None
    except MemoryError as e: logger.error(f"MemoryError: {e}", exc_info=True); raise e; return None
    except Exception as e: logger.error(f"Unexpected error in pipeline: {e}", exc_info=True); return None
    finally:
        if slide is not None:
            try: slide.close(); logger.debug("OpenSlide object closed.")
            except Exception as e_close: logger.warning(f"Error closing slide object: {e_close}")