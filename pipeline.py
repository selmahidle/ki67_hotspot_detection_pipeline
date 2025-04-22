# pipeline.py
# Full updated code incorporating StarDist refinement AND restored SMP Tumor/Cell logic
# Includes detailed logging for DS4 image handling & StarDist refinement debugging

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
import sys # Import sys for printing detailed exceptions if logger fails

# --- Local Application Imports ---
from transforms import get_transforms, get_transforms_no_clahe
from tissue_detection import detect_tissue
from patching import run_inference_on_patches
from utils import create_weight_map
from stain_utils import get_dab_mask
from hotspot_detection import identify_hotspots
from visualization import generate_overlay
# --- Add StarDist import ---
from stardist_utils import predict_patch_stardist


logger = logging.getLogger(__name__)

# --- Modify function signature ---
# Add stardist_model, keep cell_models
def process_slide_ki67(slide_path, output_dir, tumor_models, cell_models, stardist_model, device):
    """
    Process a slide for Ki67 analysis using tumor segmentation, cell segmentation (SMP),
    and StarDist-based hotspot refinement.

    Pipeline:
    1. Load slide, save overviews.
    2. Detect tissue.
    3. Segment tumor (SMP).
    4. Segment cells (SMP), conditioned on tumor mask -> cell_mask_binary_l2.
    5. Calculate DAB+ mask within tumor regions at L2 -> dab_plus_mask_l2.
    6. Identify *candidate* hotspots based on density of (SMP Cells AND DAB+) at L2.
    7. *Refine* candidates by running StarDist on candidate patches and counting cells
       whose centroids are within BOTH DAB+ areas AND SMP cell mask areas.
    8. Re-rank hotspots based on refined StarDist counts.
    9. Generate overlay (showing SMP cell mask, StarDist-ranked hotspots).
    """
    logger.info(f"Processing slide: {os.path.basename(slide_path)}")
    slide = None
    try:
        # --- Setup ---
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        image_output_dir = os.path.join(output_dir, slide_name)
        os.makedirs(image_output_dir, exist_ok=True)
        # Define debug directories early
        debug_mask_dir = os.path.join(image_output_dir, "debug_masks")
        os.makedirs(debug_mask_dir, exist_ok=True)
        hotspot_debug_dir = os.path.join(image_output_dir, f"{slide_name}_hotspot_debug")
        os.makedirs(hotspot_debug_dir, exist_ok=True)
        overlay_debug_dir = os.path.join(image_output_dir, f"{slide_name}_overlay_debug")
        os.makedirs(overlay_debug_dir, exist_ok=True)


        # --- Load slide ---
        slide = openslide.open_slide(slide_path)

        # --- Define levels and Parameters ---
        tissue_level = 5
        tumor_level = 3   # Effective level for tumor mask (ds8)
        cell_level = 2    # Effective level for SMP cell mask (ds4)
        hotspot_level = 2 # Level for hotspot candidates and StarDist refinement (L2)
        overlay_level = 2 # Final visualization at L2
        ds8_level = 3
        ds4_level = 2

        # Tumor stage parameters
        tumor_patch_size = 4096
        tumor_overlap = 1024
        tumor_output_channels = 2
        tumor_batch_size = 2
        tumor_prob_threshold = 0.3

        # Cell stage parameters (SMP)
        cell_patch_size = 1024
        cell_overlap = 256
        cell_output_channels = 2
        cell_batch_size = 8 # Reduced default
        cell_prob_threshold = 0.3

        # Hotspot parameters
        hotspot_patch_size_l0 = 2048
        hotspot_top_n = 5 # Final number of hotspots
        hotspot_dab_threshold = 0.15

        # --- Check Levels Exist ---
        required_levels_actual = {'tissue': tissue_level, 'ds8': ds8_level, 'ds4': ds4_level}
        max_req_level = max(required_levels_actual.values())
        if max_req_level >= slide.level_count:
            logger.error(f"Slide {slide_name} level count ({slide.level_count}) insufficient. Max required: {max_req_level}.")
            if slide: slide.close() # Close slide if opened
            return None

        # --- Get Transforms ---
        transforms_clahe = get_transforms()
        transforms_no_clahe = get_transforms_no_clahe()

        # === Pre-computation: Save ds8 and ds4 JPGs === #
        # --- ds8 ---
        ds8_path = os.path.join(image_output_dir, f"{slide_name}_ds8.jpg")
        ds8_img = None
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
                 logger.info(f"Saved ds8 image.")
            except Exception as e: logger.error(f"Failed to generate ds8: {e}", exc_info=True); return None
        if ds8_img is None: # Final check after trying to load/generate
            logger.error("ds8 image could not be loaded or generated. Cannot proceed.")
            return None

        # --- DS4 Handling with DETAILED LOGGING --- #
        ds4_path = os.path.join(image_output_dir, f"{slide_name}_ds4.jpg")
        ds4_img = None
        logger.info(f"--- DS4 Start: Checking path {ds4_path} ---")

        if os.path.exists(ds4_path):
            logger.info(f"Attempting to load existing ds4 image: {ds4_path}")
            try:
                ds4_img_bgr = cv2.imread(ds4_path)
                if ds4_img_bgr is None:
                    logger.error(f"cv2.imread returned None for existing ds4 file: {ds4_path}")
                else:
                    logger.info(f"Successfully loaded existing ds4 image via cv2, shape: {ds4_img_bgr.shape}")
                    ds4_img = cv2.cvtColor(ds4_img_bgr, cv2.COLOR_BGR2RGB)
                    logger.info("Converted loaded ds4 BGR to RGB.")
            except Exception as load_err:
                 logger.error(f"Exception during cv2.imread for ds4: {load_err}", exc_info=True)
                 ds4_img = None

        if ds4_img is None:
            logger.info(f"Attempting to generate ds4 image (Level {ds4_level})...")
            if ds4_level >= slide.level_count:
                 logger.error(f"Required ds4 Level {ds4_level} not available. Cannot generate.")
                 return None
            else:
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
                except Exception as gen_err: logger.error(f"Unhandled Exception during ds4 generation: {gen_err}", exc_info=True); print(f"Unhandled Exception: {gen_err}", file=sys.stderr); traceback.print_exc(file=sys.stderr); ds4_img = None

        if ds4_img is None: logger.error("Failed to load or generate ds4 image. Cannot proceed."); return None
        else: logger.info(f"--- DS4 End: Successfully obtained ds4_img, shape {ds4_img.shape} ---")

        # === 1. Tissue Detection (Level 5) ===
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
                 tissue_mask_l5_255 = detect_tissue(tissue_img_bgr, threshold_tissue_ratio=0.05) # Corrected argument name
                 if tissue_mask_l5_255 is not None and np.sum(tissue_mask_l5_255) > 0:
                      tissue_mask_l5 = (tissue_mask_l5_255 > 0).astype(np.uint8)
                      cv2.imwrite(tissue_mask_filename, tissue_mask_l5 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                      logger.info(f"Saved tissue mask L{tissue_level}.")
                 else: logger.error("Tissue detection failed or result was empty.")
            except Exception as e: logger.error(f"Failed to generate tissue mask: {e}", exc_info=True)

        if tissue_mask_l5 is None or np.sum(tissue_mask_l5) == 0: logger.error("Tissue mask is empty. Cannot proceed."); return None

        # === 2. Upsample Tissue Mask to Tumor Level (L3 equivalent) ===
        tumor_level_h, tumor_level_w = ds8_img.shape[:2]
        tissue_mask_l3 = cv2.resize(tissue_mask_l5, (tumor_level_w, tumor_level_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Upsampled tissue mask to L{tumor_level} ({tissue_mask_l3.shape})")

        # === 3. Tumor Segmentation (using ds8 JPG) === # <-- RESTORED FULL LOGIC
        logger.info(f"--- Stage 2: Tumor Segmentation (L{tumor_level}) ---")
        processed_tumor_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_L{tumor_level}.jpg")
        consensus_tumor_mask_l3_raw = None
        tumor_mask_l3_processed = None
        skip_tumor_stage = False

        if os.path.exists(processed_tumor_mask_filename):
            logger.info(f"Found existing final tumor mask: {processed_tumor_mask_filename}. Loading.")
            loaded_mask = cv2.imread(processed_tumor_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None:
                 logger.error(f"Failed to load existing tumor mask from {processed_tumor_mask_filename}. Will run inference.")
            else:
                 consensus_tumor_mask_l3_raw = (loaded_mask > 128).astype(np.uint8)
                 tumor_mask_l3_processed = consensus_tumor_mask_l3_raw # Use loaded mask as processed one
                 logger.info(f"Loaded tumor mask L{tumor_level} ({consensus_tumor_mask_l3_raw.shape}).")
                 skip_tumor_stage = True
        else:
            logger.info(f"Final tumor mask {processed_tumor_mask_filename} not found. Running inference.")

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
                    padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    tumor_weights.append(padded_weight_map)

            if not tumor_patches: logger.error(f"Tumor patch collection failed."); return None
            logger.info(f"Collected {len(tumor_patches)} tumor patches.")

            tumor_prob_maps_l3 = []
            for i, model in enumerate(tumor_models):
                model_name = f"TumorModel_{i+1}"
                logger.info(f"Running tumor inference with {model_name}...")
                probs = run_inference_on_patches(model, device, tumor_output_channels, tumor_batch_size, tumor_level, (w, h), tumor_patches, tumor_locations, tumor_weights, model_name)
                if probs is not None: tumor_prob_maps_l3.append(probs)
                else: logger.warning(f"{model_name} failed.")
            del tumor_patches, tumor_locations, tumor_weights

            if not tumor_prob_maps_l3: logger.error("All tumor models failed inference."); return None

            individual_model_masks = []
            num_successful_models = len(tumor_prob_maps_l3)
            logger.info(f"Processing {num_successful_models} tumor model outputs...")
            for i, prob_map in enumerate(tumor_prob_maps_l3):
                model_mask_bool = (prob_map[:, :, 1] > tumor_prob_threshold)
                individual_model_masks.append(model_mask_bool)
                model_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_model{i+1}_th{tumor_prob_threshold}_L{tumor_level}.jpg")
                cv2.imwrite(model_mask_filename, model_mask_bool.astype(np.uint8) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if len(individual_model_masks) != num_successful_models or num_successful_models == 0: logger.error("Consensus failed."); return None
            consensus_mask_bool = np.logical_and.reduce(individual_model_masks)
            consensus_tumor_mask_l3_raw = consensus_mask_bool.astype(np.uint8) * tissue_mask_l3 # Apply tissue constraint
            del tumor_prob_maps_l3, individual_model_masks, consensus_mask_bool
            logger.info(f"Raw tumor consensus mask generated. Pixels: {np.sum(consensus_tumor_mask_l3_raw)}")
            raw_consensus_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_raw_consensus_all{num_successful_models}_th{tumor_prob_threshold}_L{tumor_level}.jpg")
            cv2.imwrite(raw_consensus_filename, consensus_tumor_mask_l3_raw * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            smooth_kernel_size = 5; kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
            tumor_mask_l3_processed = cv2.morphologyEx(consensus_tumor_mask_l3_raw, cv2.MORPH_CLOSE, kernel)
            tumor_mask_l3_processed = cv2.morphologyEx(tumor_mask_l3_processed, cv2.MORPH_OPEN, kernel)
            logger.info(f"Applied smoothing to tumor mask.")
            cv2.imwrite(processed_tumor_mask_filename, tumor_mask_l3_processed * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            logger.info(f"Processed tumor mask saved.")
        # --- END OF RESTORED TUMOR BLOCK ---

        if tumor_mask_l3_processed is None or np.sum(tumor_mask_l3_processed) == 0:
            logger.error("Tumor mask empty or invalid after load/inference. Cannot proceed.")
            return None

        # === 4. Upsample Processed Tumor Mask to Analysis Level (L2) ===
        cell_level_h, cell_level_w = ds4_img.shape[:2]
        tumor_mask_l2 = cv2.resize(tumor_mask_l3_processed, (cell_level_w, cell_level_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Upsampled tumor mask to L{hotspot_level} ({tumor_mask_l2.shape})")

        # === 5. Tumor Cell Segmentation (SMP - using ds4 JPG) === # <-- RESTORED FULL LOGIC
        logger.info(f"--- Stage 3: Cell Segmentation (SMP L{cell_level}) ---")
        cell_mask_filename = os.path.join(image_output_dir, f"{slide_name}_cell_mask_binary_L{cell_level}.jpg")
        cell_mask_binary_l2 = None
        skip_cell_stage = False
        if os.path.exists(cell_mask_filename):
            loaded_cell_mask = cv2.imread(cell_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_cell_mask is not None:
                 cell_mask_binary_l2 = (loaded_cell_mask > 128).astype(np.uint8)
                 skip_cell_stage = True; logger.info(f"Loaded existing SMP cell mask L{cell_level}.")
            else: logger.error(f"Failed to load {cell_mask_filename}")
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
                         padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                         cell_weights.append(padded_weight_map)

                 if not cell_patches: logger.warning("SMP Cell patch collection failed."); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
                 else:
                      logger.info(f"Collected {len(cell_patches)} SMP cell patches.")
                      cell_prob_maps_l2 = []
                      for i, model in enumerate(cell_models):
                           model_name = f"CellModel_{i+1}"
                           logger.info(f"Running SMP cell inference with {model_name}...")
                           probs = run_inference_on_patches(model, device, cell_output_channels, cell_batch_size, cell_level, (w, h), cell_patches, cell_locations, cell_weights, model_name)
                           if probs is not None: cell_prob_maps_l2.append(probs)
                           else: logger.warning(f"{model_name} failed.")
                      del cell_patches, cell_locations, cell_weights

                      if not cell_prob_maps_l2: logger.error("All SMP cell models failed."); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
                      else:
                           logger.info(f"Averaging {len(cell_prob_maps_l2)} SMP cell maps...")
                           combined_cell_prob = np.mean(cell_prob_maps_l2, axis=0).astype(np.float32); del cell_prob_maps_l2
                           cell_mask_binary_l2_raw = (combined_cell_prob[:,:,1] > cell_prob_threshold).astype(np.uint8)
                           cell_mask_binary_l2 = cell_mask_binary_l2_raw * tumor_mask_l2
                           logger.info(f"Final SMP cell mask created. Pixels: {np.sum(cell_mask_binary_l2)}")
                           cv2.imwrite(cell_mask_filename, cell_mask_binary_l2 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                           logger.info(f"Saved SMP cell mask.")
        # --- END OF RESTORED CELL BLOCK ---

        if cell_mask_binary_l2 is None: logger.warning("Cell mask is None. Creating empty mask."); cell_mask_binary_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
        if np.sum(cell_mask_binary_l2) == 0: logger.warning("SMP Cell mask is empty after processing.") # Add warning if empty


        # === 6. Pre-calculate DAB+ Mask (L2) ===
        logger.info(f"--- Stage 4: DAB+ Mask Calculation (L{hotspot_level}) ---")
        dab_plus_mask_l2 = None
        if np.sum(tumor_mask_l2) > 0:
            try:
                tumor_labels = label(tumor_mask_l2); tumor_props = regionprops(tumor_labels)
                if tumor_props:
                    min_r, min_c, max_r, max_c = tumor_props[0].bbox
                    for prop in tumor_props[1:]: min_r=min(min_r,prop.bbox[0]);min_c=min(min_c,prop.bbox[1]);max_r=max(max_r,prop.bbox[2]);max_c=max(max_c,prop.bbox[3])
                    bbox_h, bbox_w = max_r - min_r, max_c - min_c
                    if bbox_h > 0 and bbox_w > 0:
                        level0_x_read=int(min_c*slide.level_downsamples[hotspot_level]);level0_y_read=int(min_r*slide.level_downsamples[hotspot_level])
                        try:
                            rgb_patch_l2_pil = slide.read_region((level0_x_read, level0_y_read), hotspot_level, (bbox_w, bbox_h)).convert('RGB')
                            rgb_patch_l2 = np.array(rgb_patch_l2_pil)
                            if rgb_patch_l2.shape[0]!=bbox_h or rgb_patch_l2.shape[1]!=bbox_w: rgb_patch_l2=cv2.resize(rgb_patch_l2,(bbox_w,bbox_h),interpolation=cv2.INTER_LINEAR)
                            dab_plus_mask_patch = get_dab_mask(rgb_patch_l2, hotspot_dab_threshold)
                            if dab_plus_mask_patch is not None:
                                dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
                                dab_plus_mask_l2[min_r:max_r, min_c:max_c] = dab_plus_mask_patch
                                logger.info(f"Calculated DAB+ mask L2. Pixels: {np.sum(dab_plus_mask_l2)}")
                        except Exception as e_read_dab: logger.error(f"Error reading/processing patch for DAB: {e_read_dab}", exc_info=True)
                    else: logger.warning(f"Invalid tumor bbox for DAB: {bbox_w}x{bbox_h}")
                else: logger.warning("No tumor regions found for DAB bbox.")
            except Exception as e_dab: logger.error(f"Error during DAB calculation: {e_dab}", exc_info=True)
        if dab_plus_mask_l2 is None: dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)


        # === 7. Initial Hotspot Candidate ID (based on SMP Cell & DAB density) ===
        logger.info(f"--- Stage 5: Initial Hotspot Candidate ID (L{hotspot_level}) ---")
        hotspot_target_mask_coarse = (cell_mask_binary_l2 > 0) * (dab_plus_mask_l2 > 0)
        hotspot_target_mask_coarse = hotspot_target_mask_coarse.astype(np.uint8)
        candidate_hotspots = []
        if np.sum(hotspot_target_mask_coarse) == 0:
             logger.warning("Coarse target mask (DAB+ SMP Cells) empty.")
        else:
             logger.info(f"Identifying candidates based on DAB+ SMP-Cell density. Target pixels: {np.sum(hotspot_target_mask_coarse)}")
             num_initial_candidates = max(hotspot_top_n * 3, 10)
             candidate_hotspots = identify_hotspots(slide, hotspot_level, hotspot_target_mask_coarse, hotspot_patch_size_l0, num_initial_candidates, hotspot_debug_dir)

        # === 8. StarDist Refinement and Re-ranking ===
        hotspots = []
        if not candidate_hotspots:
            logger.warning("No candidates for StarDist refinement.")
        else:
            logger.info(f"--- Stage 6: StarDist Refinement & Re-ranking (L{hotspot_level}) ---")
            refined_hotspots = []
            actual_pixel_size_um = None
            try: # Get pixel size
                level0_mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.0))
                if level0_mpp_x > 0 and hotspot_level < slide.level_count:
                    actual_pixel_size_um = level0_mpp_x * slide.level_downsamples[hotspot_level]
                    logger.info(f"Actual pixel size L{hotspot_level} = {actual_pixel_size_um:.4f} um/pix")
                else: logger.error("Cannot determine pixel size.")
            except Exception as mpp_err: logger.error(f"Error getting pixel size: {mpp_err}")

            if actual_pixel_size_um is None: # Fallback if pixel size unknown
                 logger.warning("Skipping StarDist refinement (unknown pixel size). Using density ranking.")
                 candidate_hotspots.sort(key=lambda item: item.get('density_score', 0), reverse=True)
                 hotspots = candidate_hotspots[:hotspot_top_n]
            else: # Proceed with refinement
                downsample_l2 = slide.level_downsamples[hotspot_level]
                # --- Create a specific debug directory for this refinement stage --- # MODIFIED
                refinement_debug_base_dir = os.path.join(hotspot_debug_dir, "refinement_patches")
                os.makedirs(refinement_debug_base_dir, exist_ok=True)
                # --------------------------------------------------------------------
                for i, hotspot in enumerate(tqdm(candidate_hotspots, desc="Refining Hotspots")):
                    # --- Create debug dir for THIS hotspot --- # MOVED INSIDE LOOP
                    hs_debug_dir = os.path.join(refinement_debug_base_dir, f"candidate_{i+1:02d}_hs") # More descriptive name
                    os.makedirs(hs_debug_dir, exist_ok=True)
                    # ------------------------------------------
                    x_l2, y_l2 = hotspot['coords_level']; w_l2, h_l2 = hotspot['size_level']
                    x_l2=max(0,x_l2); y_l2=max(0,y_l2) # Clip coords
                    if x_l2+w_l2>cell_level_w: w_l2=cell_level_w-x_l2
                    if y_l2+h_l2>cell_level_h: h_l2=cell_level_h-y_l2
                    if w_l2<=0 or h_l2<=0: continue

                    hs_patch_rgb = None # Read patch
                    try:
                        x_l0=int(x_l2*downsample_l2); y_l0=int(y_l2*downsample_l2)
                        hs_patch_pil = slide.read_region((x_l0,y_l0), hotspot_level, (w_l2,h_l2)).convert('RGB')
                        hs_patch_rgb = np.array(hs_patch_pil)
                        if hs_patch_rgb.shape[0]!=h_l2 or hs_patch_rgb.shape[1]!=w_l2: hs_patch_rgb = cv2.resize(hs_patch_rgb,(w_l2,h_l2),interpolation=cv2.INTER_LINEAR)
                        # --- DEBUG: Save hotspot patch --- # ADDED
                        cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{i+1:02d}_patch_L{hotspot_level}.png"), cv2.cvtColor(hs_patch_rgb, cv2.COLOR_RGB2BGR))
                        # ---------------------------------
                    except Exception as e: logger.error(f"Failed reading patch {i+1}: {e}"); continue

                    # --- Predict WITH and WITHOUT size filter for comparison and debugging --- # MODIFIED
                    labels_unfiltered, details = predict_patch_stardist(stardist_model, hs_patch_rgb, actual_pixel_size_um, apply_size_filter=False)
                    labels_filtered, _ = predict_patch_stardist(stardist_model, hs_patch_rgb, actual_pixel_size_um, apply_size_filter=True) # Need details only once
                    # ------------------------------------------------------------------------------------

                    # --- DEBUG: Save StarDist outputs --- # ADDED
                    if labels_unfiltered is not None:
                         labels_unfiltered_vis = (labels_unfiltered > 0).astype(np.uint8) * 255
                         cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{i+1:02d}_stardist_labels_unfiltered.png"), labels_unfiltered_vis)
                    if labels_filtered is not None:
                         labels_filtered_vis = (labels_filtered > 0).astype(np.uint8) * 255
                         cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{i+1:02d}_stardist_labels_filtered.png"), labels_filtered_vis)
                    # ------------------------------------

                    # --- Initialize counts --- # <-- MODIFIED
                    hotspot['stardist_total_count_unfiltered'] = 0 # Before size filter
                    hotspot['stardist_total_count_filtered'] = 0   # After size filter
                    hotspot['stardist_smp_cell_count'] = 0         # Denominator for Ki67%
                    hotspot['stardist_dab_smp_cell_count'] = 0     # Numerator & Ranking Score

                    if labels_unfiltered is None or details is None: # Check result from unfiltered prediction
                        logger.warning(f"StarDist prediction failed for candidate hotspot {i+1}. Counts set to 0.")
                    else:
                        centroids = details.get('points') # Use 'points' for centroids
                        valid_centroids = isinstance(centroids, np.ndarray) and centroids.ndim == 2 and centroids.shape[1] == 2

                        if valid_centroids and len(centroids) > 0:
                            hotspot['stardist_total_count_unfiltered'] = len(centroids) # Total before any filtering

                            dab_patch = dab_plus_mask_l2[y_l2 : y_l2 + h_l2, x_l2 : x_l2 + w_l2]
                            cell_patch_smp = cell_mask_binary_l2[y_l2 : y_l2 + h_l2, x_l2 : x_l2 + w_l2]

                            target_shape_yx = (labels_unfiltered.shape[0], labels_unfiltered.shape[1]) # Use unfiltered label shape
                            target_shape_wh=(target_shape_yx[1],target_shape_yx[0])

                            if target_shape_yx[0] > 0 and target_shape_yx[1] > 0:
                                if dab_patch.shape!=target_shape_yx: dab_patch=cv2.resize(dab_patch.astype(np.uint8),target_shape_wh,interpolation=cv2.INTER_NEAREST)
                                if cell_patch_smp.shape!=target_shape_yx: cell_patch_smp=cv2.resize(cell_patch_smp.astype(np.uint8),target_shape_wh,interpolation=cv2.INTER_NEAREST)
                            else: dab_patch = None; cell_patch_smp = None

                            # --- DEBUG: Save mask patches --- # ADDED
                            if dab_patch is not None: cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{i+1:02d}_dab_patch.png"), dab_patch * 255)
                            if cell_patch_smp is not None: cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{i+1:02d}_smpcell_patch.png"), cell_patch_smp * 255)
                            # --------------------------------

                            positive_count = 0; smp_cell_area_count = 0
                            counted_overlay = hs_patch_rgb.copy() # --- DEBUG: Prepare overlay for counted cells --- # ADDED
                            if dab_patch is not None and cell_patch_smp is not None:
                                try:
                                    centroids_int = centroids.astype(int)
                                    for idx in range(centroids_int.shape[0]):
                                        cy, cx = centroids_int[idx, 0], centroids_int[idx, 1]
                                        if not (0<=cy<dab_patch.shape[0] and 0<=cx<dab_patch.shape[1]): continue

                                        is_in_smp_cell = (cell_patch_smp[cy, cx] > 0)
                                        is_dab_positive = (dab_patch[cy, cx] > 0)

                                        if is_in_smp_cell:
                                            smp_cell_area_count += 1
                                            if is_dab_positive:
                                                positive_count += 1
                                                cv2.circle(counted_overlay, (cx, cy), 5, (0, 255, 0), -1) # Green: Triple+
                                            else:
                                                cv2.circle(counted_overlay, (cx, cy), 5, (255, 0, 0), -1) # Blue: SMP+, DAB-
                                        elif is_dab_positive:
                                             cv2.circle(counted_overlay, (cx, cy), 3, (0, 255, 255), -1) # Yellow: DAB+, SMP-
                                        else:
                                             cv2.circle(counted_overlay, (cx, cy), 3, (0, 0, 255), -1) # Red: Neither
                                except Exception as e_count: logger.error(f"Error counting in hotspot {i+1}: {e_count}", exc_info=True)

                            # --- DEBUG: Save the counted overlay --- # ADDED
                            cv2.imwrite(os.path.join(hs_debug_dir, f"hs_{i+1:02d}_counted_centroids_vis.png"), cv2.cvtColor(counted_overlay, cv2.COLOR_RGB2BGR))
                            # ---------------------------------------

                            if labels_filtered is not None: # Update count based on filtered labels
                                 hotspot['stardist_total_count_filtered'] = len(np.unique(labels_filtered[labels_filtered > 0]))
                            else: hotspot['stardist_total_count_filtered'] = 0

                            hotspot['stardist_smp_cell_count'] = smp_cell_area_count
                            hotspot['stardist_dab_smp_cell_count'] = positive_count
                            logger.debug(f" Hotspot {i+1}: StarDist Total(Unfilt)={hotspot['stardist_total_count_unfiltered']}, Total(Filt)={hotspot['stardist_total_count_filtered']}, In SMP Area={smp_cell_area_count}, DAB+&SMP+={positive_count}")

                        else: logger.debug(f" Hotspot {i+1}: No valid centroids found. Counts set to 0.")

                    refined_hotspots.append(hotspot)

                if refined_hotspots: # Re-rank
                    refined_hotspots.sort(key=lambda item: item.get('stardist_dab_smp_cell_count', 0), reverse=True)
                    hotspots = refined_hotspots[:hotspot_top_n]
                    logger.info(f"Re-ranked hotspots based on StarDist count. Top {len(hotspots)}:")
                    for i, hs in enumerate(hotspots): logger.info(f"  {i+1}: Count={hs.get('stardist_dab_smp_cell_count','N/A')}")
                    for hs in hotspots: hs['final_score'] = hs.get('stardist_dab_smp_cell_count', 0)
                else: logger.warning("No hotspots remained after refinement.")


        # === 9. Generate Final Overlay (Level 2) ===
        logger.info(f"--- Stage 7: Overlay Generation (L{overlay_level}) ---")
        overlay_h, overlay_w = ds4_img.shape[:2]
        tissue_mask_overlay = cv2.resize(tissue_mask_l5, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
        tumor_mask_overlay = tumor_mask_l2
        cell_mask_for_overlay = cell_mask_binary_l2 # Use SMP mask

        final_overlay = generate_overlay(slide, overlay_level, hotspot_level, tissue_mask_overlay, tumor_mask_overlay, cell_mask_for_overlay, hotspots, dab_mask_l2=dab_plus_mask_l2, debug_dir=overlay_debug_dir)

        if final_overlay is not None: # Save overlay
            overlay_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}.jpg")
            png_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}_temp.png")
            try:
                pil_image=Image.fromarray(final_overlay); pil_image.save(png_filename)
                with Image.open(png_filename) as img_png: img_png.convert('RGB').save(overlay_filename,quality=95)
                logger.info(f"Saved final overlay to {overlay_filename}"); os.remove(png_filename)
            except Exception as e:
                logger.error(f"Error saving overlay: {e}")
                try: cv2.imwrite(overlay_filename, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY),95]); logger.info(f"Saved overlay via cv2 fallback.")
                except Exception as e2: logger.error(f"Fallback overlay save failed: {e2}")
        else: logger.error("Failed to generate final overlay.")

        logger.info(f"--- Processing Finished for slide: {slide_name} ---")
        return hotspots # Return StarDist-ranked results

    except openslide.OpenSlideError as e: logger.error(f"OpenSlide error: {e}", exc_info=True); return None
    except FileNotFoundError as e: logger.error(f"File not found: {e}", exc_info=True); return None
    except MemoryError as e: logger.error(f"MemoryError: {e}", exc_info=True); return None
    except Exception as e: logger.error(f"Unexpected error in pipeline: {e}", exc_info=True); return None
    finally:
        if slide is not None:
            try: slide.close()
            except Exception as e: logger.warning(f"Error closing slide: {e}")