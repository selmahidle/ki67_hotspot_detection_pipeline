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
import torch
from transforms import get_transforms, get_transforms_no_clahe
from tissue_detection import detect_tissue
from patching import run_inference_on_patches
from utils import create_weight_map, get_actual_pixel_size_um
from stain_utils import get_dab_mask
from hotspot_detection import identify_hotspots
import visualization
from stardist_utils import refine_hotspot_with_stardist

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None


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


def process_slide_ki67(slide_path, output_dir, tumor_model, cell_model, stardist_model, device):
    """
    Process a slide for Ki67 analysis using a single tumor segmentation model,
    a single cell segmentation model (AttentionUNet), and StarDist-based hotspot refinement.
    """
    logger.info(f"Processing slide: {os.path.basename(slide_path)} with single tumor model and AttentionUNet for cells.")
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
        tissue_level = 5; tumor_level = 3; cell_level = 2; hotspot_level = 2
        overlay_level = 2; ds8_level = 3; ds4_level = 2

        # Tumor stage parameters (DeepLabV3+ ResNet18 model)
        tumor_patch_size = 1024  
        tumor_overlap = 256     
        tumor_output_channels = 1
        tumor_batch_size = 4      
        tumor_prob_threshold = 0.5

        # Cell stage parameters (AttentionUNet)
        cell_patch_size = 1024; cell_overlap = 256
        cell_output_channels = 1 
        cell_batch_size = 8; cell_prob_threshold = 0.3

        hotspot_patch_size_l0 = 2048
        hotspot_top_n = 5
        hotspot_dab_threshold = 0.15

        # --- Check Levels Exist ---
        required_levels_actual = {'tissue': tissue_level, 'ds8': ds8_level, 'ds4': ds4_level}
        max_req_level = max(required_levels_actual.values())
        if max_req_level >= slide.level_count:
            logger.error(f"Slide {slide_name} level count ({slide.level_count}) insufficient. Max required level index: {max_req_level}.")
            if slide: slide.close(); return None

        transforms_for_segmentation = get_transforms_no_clahe() 

        # === Pre-computation: Save/Load ds8 and ds4 JPGs ===
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
                 logger.info(f"Saved ds8 image to {ds8_path}")
            except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError generating ds8: {ose}", exc_info=True); return None
            except Exception as e: logger.error(f"Failed to generate/save ds8: {e}", exc_info=True); return None
        if ds8_img is None: logger.error("ds8 image could not be loaded or generated. Cannot proceed."); return None

        ds4_path = os.path.join(image_output_dir, f"{slide_name}_ds4.jpg")
        ds4_img = None
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

        logger.info(f"--- Stage 2: Tumor Segmentation (L{tumor_level}) with Single Model ---")
        processed_tumor_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_L{tumor_level}.jpg")
        tumor_mask_l3_processed = None # Final processed mask
        skip_tumor_stage = False

        if os.path.exists(processed_tumor_mask_filename):
            logger.info(f"Found existing final tumor mask: {processed_tumor_mask_filename}. Loading.")
            loaded_mask = cv2.imread(processed_tumor_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None:
                logger.error(f"Failed to load existing tumor mask. Will run inference.")
            else:
                 tumor_mask_l3_processed = (loaded_mask > 128).astype(np.uint8) 
                 logger.info(f"Loaded tumor mask L{tumor_level} ({tumor_mask_l3_processed.shape}).")
                 skip_tumor_stage = True
        else:
            logger.info(f"Final tumor mask not found. Running inference with single tumor model.")

        if not skip_tumor_stage:
            logger.info(f"Running tumor segmentation using ds8 JPG (patch size: {tumor_patch_size})...")
            tumor_patches = []; tumor_locations = []; tumor_weights = []
            h, w = ds8_img.shape[:2]
            stride_x = tumor_patch_size - tumor_overlap; stride_y = tumor_patch_size - tumor_overlap
            if stride_x <= 0 or stride_y <= 0: logger.error(f"Invalid Tumor patch stride."); return None

            logger.info(f"Collecting tumor patches from ds8 ({w}x{h})...")
            for y_coord in tqdm(range(0, h, stride_y), desc="Collecting Tumor Patches"):
                for x_coord in range(0, w, stride_x):
                    y_start, x_start = y_coord, x_coord
                    end_y, end_x = min(y_start + tumor_patch_size, h), min(x_start + tumor_patch_size, w)
                    window_h, window_w = end_y - y_start, end_x - x_start
                    if window_w < stride_x // 2 or window_h < stride_y // 2: continue # Skip very small edge patches

                    patch = ds8_img[y_start:end_y, x_start:end_x];
                    if patch.shape[0] <= 0 or patch.shape[1] <= 0: continue

                    mask_patch_tissue = tissue_mask_l3[y_start:end_y, x_start:end_x] # Use tissue mask at L3
                    if np.mean(mask_patch_tissue) < 0.05: continue # Skip patches with low tissue content

                    pad_h = max(0, tumor_patch_size - window_h); pad_w = max(0, tumor_patch_size - window_w)
                    padded_patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])

                    # Use transforms_for_segmentation (no CLAHE, 0-1 scaling)
                    transformed = transforms_for_segmentation(image=padded_patch)
                    tumor_patches.append(transformed["image"])
                    tumor_locations.append((y_start, end_y, x_start, end_x))

                    weight_map = create_weight_map((window_h, window_w))
                    if weight_map is None: logger.error(f"Failed to create weight map for tumor patch. Skipping."); continue
                    if pad_h > 0 or pad_w > 0:
                        padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    else:
                        padded_weight_map = weight_map
                    tumor_weights.append(padded_weight_map)

            if not tumor_patches: logger.error(f"Tumor patch collection failed."); return None
            if not (len(tumor_patches) == len(tumor_locations) == len(tumor_weights)):
                logger.error(f"Tumor data collection mismatch!"); return None
            logger.info(f"Collected {len(tumor_patches)} tumor patches.")

            model_name = f"TumorModel_Single_{tumor_model.__class__.__name__}"
            logger.info(f"Running tumor inference with {model_name}...")
            # run_inference_on_patches should handle sigmoid if model outputs logits and output_channels=1
            # and return a 2D probability map (H, W)
            tumor_prob_map_l3 = run_inference_on_patches(
                model=tumor_model,
                device=device,
                output_channels=tumor_output_channels, # Should be 1
                batch_size=tumor_batch_size,
                level_or_id=f"Tumor L{tumor_level}",
                level_dims=(w, h),
                patches_to_process=tumor_patches,
                patch_locations=tumor_locations,
                patch_weights=tumor_weights,
                model_name=model_name
            )
            del tumor_patches, tumor_locations, tumor_weights # Free memory

            if tumor_prob_map_l3 is None:
                logger.error(f"{model_name} inference failed. Cannot generate tumor mask."); return None

            # Convert probability map (0-1 range) to binary mask
            # Ensure tumor_prob_map_l3 is 2D (H,W)
            if tumor_prob_map_l3.ndim == 3 and tumor_prob_map_l3.shape[-1] == 1:
                tumor_prob_map_l3 = tumor_prob_map_l3.squeeze(-1)
            elif tumor_prob_map_l3.ndim != 2:
                logger.error(f"Tumor probability map has unexpected dimensions: {tumor_prob_map_l3.shape}. Expected 2D or 3D with last dim 1.")
                return None

            raw_tumor_mask_l3 = (tumor_prob_map_l3 > tumor_prob_threshold).astype(np.uint8)
            raw_tumor_mask_l3_on_tissue = raw_tumor_mask_l3 * tissue_mask_l3 # Apply tissue mask

            logger.info(f"Raw tumor mask (on tissue) generated. Positive pixels: {np.sum(raw_tumor_mask_l3_on_tissue)}")

            # Apply morphological operations for smoothing
            smooth_kernel_size = 5 # Can be adjusted
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
            tumor_mask_l3_processed = cv2.morphologyEx(raw_tumor_mask_l3_on_tissue, cv2.MORPH_CLOSE, kernel)
            tumor_mask_l3_processed = cv2.morphologyEx(tumor_mask_l3_processed, cv2.MORPH_OPEN, kernel)
            logger.info(f"Applied smoothing to tumor mask. Pixels after: {np.sum(tumor_mask_l3_processed)}")

            try:
                cv2.imwrite(processed_tumor_mask_filename, tumor_mask_l3_processed * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                logger.info(f"Saved final processed tumor mask to {processed_tumor_mask_filename}")
            except Exception as e_save:
                logger.error(f"Failed to save final processed tumor mask: {e_save}")

        if tumor_mask_l3_processed is None or np.sum(tumor_mask_l3_processed) == 0:
            logger.error("Tumor mask is empty or invalid after processing. Cannot proceed."); return None

        cell_level_h, cell_level_w = ds4_img.shape[:2]
        tumor_mask_l2 = cv2.resize(tumor_mask_l3_processed, (cell_level_w, cell_level_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Upsampled processed tumor mask to L{hotspot_level} ({tumor_mask_l2.shape})")


        logger.info(f"--- Stage 3: Cell Segmentation (AttentionUNet L{cell_level}) ---")
        cell_mask_filename = os.path.join(image_output_dir, f"{slide_name}_cell_mask_binary_L{cell_level}_AttnUNet.jpg")
        cell_mask_binary_l2 = None
        skip_cell_stage = False
        if os.path.exists(cell_mask_filename):
            loaded_cell_mask = cv2.imread(cell_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_cell_mask is not None:
                 cell_mask_binary_l2 = (loaded_cell_mask > 128).astype(np.uint8)
                 skip_cell_stage = True; logger.info(f"Loaded existing AttentionUNet cell mask L{cell_level}.")
            else: logger.error(f"Failed to load {cell_mask_filename}. Will run inference.")

        if not skip_cell_stage:
            logger.info(f"Running AttentionUNet cell segmentation inference L{cell_level}...")
            cell_patches = []; cell_locations = []; cell_weights = []
            h_cell, w_cell = ds4_img.shape[:2]
            stride_x_cell = cell_patch_size - cell_overlap; stride_y_cell = cell_patch_size - cell_overlap
            if stride_x_cell <= 0 or stride_y_cell <= 0:
                logger.error(f"Invalid AttentionUNet cell patch stride.")
                cell_mask_binary_l2 = np.zeros((h_cell, w_cell), dtype=np.uint8)
            else:
                 logger.info(f"Collecting AttentionUNet cell patches from ds4 ({w_cell}x{h_cell})...")
                 for y_coord_cell in tqdm(range(0, h_cell, stride_y_cell), desc="Collecting Cell Patches (AttnUNet)"):
                     for x_coord_cell in range(0, w_cell, stride_x_cell):
                         y_start_c, x_start_c = y_coord_cell, x_coord_cell
                         end_y_c, end_x_c = min(y_start_c + cell_patch_size, h_cell), min(x_start_c + cell_patch_size, w_cell)
                         window_h_c, window_w_c = end_y_c - y_start_c, end_x_c - x_start_c
                         if window_w_c < stride_x_cell//2 or window_h_c < stride_y_cell//2: continue

                         patch_c = ds4_img[y_start_c:end_y_c, x_start_c:end_x_c]
                         if patch_c.shape[0] <= 0 or patch_c.shape[1] <= 0: continue

                         mask_patch_tumor_for_cell = tumor_mask_l2[y_start_c:end_y_c, x_start_c:end_x_c]
                         if np.mean(mask_patch_tumor_for_cell) < 0.05: continue # Cells only in tumor regions

                         pad_h_c = max(0, cell_patch_size-window_h_c); pad_w_c = max(0, cell_patch_size-window_w_c)
                         padded_patch_c = cv2.copyMakeBorder(patch_c, 0, pad_h_c, 0, pad_w_c, cv2.BORDER_CONSTANT, value=[0,0,0])
                         # Use transforms_for_segmentation (no CLAHE, 0-1 scaling)
                         transformed_c = transforms_for_segmentation(image=padded_patch_c); cell_patches.append(transformed_c["image"])
                         cell_locations.append((y_start_c, end_y_c, x_start_c, end_x_c))

                         weight_map_c = create_weight_map((window_h_c, window_w_c))
                         if weight_map_c is None: logger.error(f"Failed create weight map for cell patch. Skipping."); continue
                         if pad_h_c > 0 or pad_w_c > 0: padded_weight_map_c = cv2.copyMakeBorder(weight_map_c, 0, pad_h_c, 0, pad_w_c, cv2.BORDER_CONSTANT, value=0)
                         else: padded_weight_map_c = weight_map_c
                         cell_weights.append(padded_weight_map_c)

                 if not cell_patches:
                     logger.warning("AttentionUNet Cell patch collection failed.")
                     cell_mask_binary_l2 = np.zeros((h_cell, w_cell), dtype=np.uint8)
                 elif not (len(cell_patches) == len(cell_locations) == len(cell_weights)):
                     logger.error(f"AttentionUNet Cell data mismatch.")
                     cell_mask_binary_l2 = np.zeros((h_cell, w_cell), dtype=np.uint8)
                 else:
                      logger.info(f"Collected {len(cell_patches)} AttentionUNet cell patches.")
                      attn_unet_model_name = "AttentionUNet_CellModel"
                      logger.info(f"Running AttentionUNet cell inference with {attn_unet_model_name}...")
                      # run_inference_on_patches should handle sigmoid if model outputs logits and output_channels=1
                      # and return a 2D probability map (H, W)
                      cell_prob_map_l2 = run_inference_on_patches(
                          model=cell_model, # This is the AttentionUNet model instance
                          device=device,
                          output_channels=cell_output_channels, # Should be 1
                          batch_size=cell_batch_size,
                          level_or_id=f"Cell L{cell_level} (AttnUNet)",
                          level_dims=(w_cell, h_cell),
                          patches_to_process=cell_patches,
                          patch_locations=cell_locations,
                          patch_weights=cell_weights,
                          model_name=attn_unet_model_name
                      )
                      del cell_patches, cell_locations, cell_weights # Free memory

                      if cell_prob_map_l2 is None:
                          logger.error("AttentionUNet cell model inference failed.")
                          cell_mask_binary_l2 = np.zeros((h_cell, w_cell), dtype=np.uint8)
                      else:
                           # Ensure cell_prob_map_l2 is 2D (H,W)
                           if cell_prob_map_l2.ndim == 3 and cell_prob_map_l2.shape[-1] == 1:
                               cell_prob_map_l2 = cell_prob_map_l2.squeeze(-1)
                           elif cell_prob_map_l2.ndim != 2:
                                logger.error(f"Cell probability map has unexpected dimensions: {cell_prob_map_l2.shape}. Expected 2D or 3D with last dim 1.")
                                cell_mask_binary_l2 = np.zeros((h_cell, w_cell), dtype=np.uint8) # Error case
                           else: # Proceed if 2D
                               raw_cell_mask_l2 = (cell_prob_map_l2 > cell_prob_threshold).astype(np.uint8)
                               cell_mask_binary_l2 = raw_cell_mask_l2 * tumor_mask_l2 # Filter cells by tumor mask
                               logger.info(f"Final AttentionUNet cell mask created. Pixels: {np.sum(cell_mask_binary_l2)}")
                               try:
                                   cv2.imwrite(cell_mask_filename, cell_mask_binary_l2 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                                   logger.info(f"Saved AttentionUNet cell mask to {cell_mask_filename}")
                               except Exception as e_save:
                                   logger.error(f"Failed to save final AttentionUNet cell mask: {e_save}")

        if cell_mask_binary_l2 is None: # Should be initialized to zeros if something failed earlier
            logger.warning("AttentionUNet Cell mask is None after processing. Creating empty mask.")
            cell_mask_binary_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
        if np.sum(cell_mask_binary_l2) == 0:
            logger.warning("AttentionUNet Cell mask is empty (no cells detected or all filtered by tumor mask).")


        # --- Stage 4: DAB+ Mask Calculation ---
        logger.info(f"--- Stage 4: DAB+ Mask Calculation (L{hotspot_level}) ---")
        dab_plus_mask_l2 = None
        if np.sum(tumor_mask_l2) > 0: # Only calculate DAB if there's tumor
            try:
                # Find bounding box of the tumor_mask_l2 to read only relevant region
                tumor_labels = label(tumor_mask_l2); tumor_props = regionprops(tumor_labels)
                if tumor_props: # If tumor regions exist
                    min_r = min(prop.bbox[0] for prop in tumor_props); min_c = min(prop.bbox[1] for prop in tumor_props)
                    max_r = max(prop.bbox[2] for prop in tumor_props); max_c = max(prop.bbox[3] for prop in tumor_props)
                    bbox_h_dab, bbox_w_dab = max_r - min_r, max_c - min_c
                    logger.info(f"Tumor bounding box at L{hotspot_level} for DAB: (x,y,w,h) = ({min_c}, {min_r}, {bbox_w_dab}, {bbox_h_dab})")

                    if bbox_h_dab > 0 and bbox_w_dab > 0:
                        level0_x_read = int(min_c * slide.level_downsamples[hotspot_level])
                        level0_y_read = int(min_r * slide.level_downsamples[hotspot_level])
                        try:
                            logger.debug(f"Reading region for DAB: L0 coords=({level0_x_read},{level0_y_read}), "
                                         f"Level={hotspot_level}, Size=({bbox_w_dab},{bbox_h_dab})")
                            rgb_patch_l2_pil = slide.read_region((level0_x_read, level0_y_read), hotspot_level, (bbox_w_dab, bbox_h_dab)).convert('RGB')
                            rgb_patch_l2_np = np.array(rgb_patch_l2_pil)

                            # Ensure read patch matches expected dimensions
                            if rgb_patch_l2_np.shape[0] != bbox_h_dab or rgb_patch_l2_np.shape[1] != bbox_w_dab:
                                logger.warning(f"Read patch size mismatch for DAB. Expected ({bbox_h_dab},{bbox_w_dab}), Got ({rgb_patch_l2_np.shape[0]},{rgb_patch_l2_np.shape[1]}). Resizing.")
                                rgb_patch_l2_np = cv2.resize(rgb_patch_l2_np, (bbox_w_dab, bbox_h_dab), interpolation=cv2.INTER_LINEAR)

                            dab_plus_mask_patch = get_dab_mask(rgb_patch_l2_np, hotspot_dab_threshold)
                            if dab_plus_mask_patch is not None:
                                dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
                                dab_plus_mask_l2[min_r:max_r, min_c:max_c] = dab_plus_mask_patch
                                logger.info(f"Calculated DAB+ mask L2. Positive pixels: {np.sum(dab_plus_mask_l2)}")
                            else: logger.error("get_dab_mask returned None.")
                        except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError reading patch for DAB: {ose}", exc_info=True)
                        except Exception as e_read_dab: logger.error(f"Error reading/processing patch for DAB: {e_read_dab}", exc_info=True)
                    else: logger.warning(f"Invalid tumor bounding box for DAB mask (width or height is zero).")
                else: logger.warning("No tumor regions found by regionprops for DAB mask calculation.")
            except Exception as e_dab: logger.error(f"Error during DAB mask preparation: {e_dab}", exc_info=True)

        if dab_plus_mask_l2 is None:
            logger.warning("DAB+ mask calculation failed or was skipped. Using an empty mask for DAB+.")
            dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)


        logger.info(f"--- Stage 5: Initial Hotspot Candidate ID (L{hotspot_level}) ---")
        # Target mask: Intersection of (AttentionUNet Cells) AND (DAB+) AND (Tumor Mask)
        hotspot_target_mask_coarse = (cell_mask_binary_l2 > 0) & \
                                     (dab_plus_mask_l2 > 0) & \
                                     (tumor_mask_l2 > 0) # Ensure candidates are within tumor
        hotspot_target_mask_coarse = hotspot_target_mask_coarse.astype(np.uint8)
        target_pixels = np.sum(hotspot_target_mask_coarse)
        logger.info(f"Coarse target mask (AttnUNet Cells+ & DAB+ & Tumor+) positive pixels: {target_pixels}")
        candidate_hotspots = []
        if target_pixels == 0:
            logger.warning("Coarse target mask for hotspot ID is empty. No candidates will be found.")
        else:
             logger.info(f"Identifying initial hotspot candidates...")
             num_initial_candidates = 10 # Number of candidates to find based on density
             candidate_hotspots = identify_hotspots(
                 slide=slide,
                 level=hotspot_level,
                 hotspot_target_mask=hotspot_target_mask_coarse,
                 hotspot_patch_size_l0=hotspot_patch_size_l0,
                 top_n_hotspots=num_initial_candidates,
                 debug_dir=hotspot_debug_dir
             )
             logger.info(f"Found {len(candidate_hotspots)} initial candidates from density.")

        # --- Apply NMS to initial candidates ---
        if candidate_hotspots:
            logger.info(f"Applying Non-Maximum Suppression to {len(candidate_hotspots)} initial candidates with IoU threshold 0.5...")
            candidates_after_nms = apply_nms_to_candidates(candidate_hotspots, iou_threshold=0.5)
            logger.info(f"Number of candidates after NMS: {len(candidates_after_nms)}")
            candidate_hotspots = candidates_after_nms
        else:
            logger.warning("No initial candidates to apply NMS to.")


        # === StarDist Refinement & Re-ranking ===
        hotspots = [] # Final list of ranked hotspots
        if not candidate_hotspots:
            logger.warning("No initial candidates found for StarDist refinement.")
        else:
            logger.info(f"--- Stage 6: StarDist Refinement & Re-ranking (L{hotspot_level}) ---")
            refined_hotspots_results = []

            refinement_debug_base_dir = os.path.join(hotspot_debug_dir, "refinement_patches")
            os.makedirs(refinement_debug_base_dir, exist_ok=True)
            actual_pixel_size_um = get_actual_pixel_size_um(slide, level=hotspot_level, fallback_value=0.25)
            if actual_pixel_size_um is None:
                logger.error("Failed to determine actual pixel size. Skipping StarDist refinement."); return None

            logger.info(f"Refining {len(candidate_hotspots)} initial candidates with StarDist...")
            for i, candidate in enumerate(tqdm(candidate_hotspots, desc="Refining Hotspots with StarDist")):
                # Pass cell_mask_binary_l2 as tumor_cell_mask_l2 for StarDist refinement context
                updated_hotspot = refine_hotspot_with_stardist(
                    candidate_hotspot=candidate,
                    stardist_model=stardist_model,
                    slide=slide,
                    hotspot_level=hotspot_level,
                    actual_pixel_size_um=actual_pixel_size_um,
                    dab_threshold=hotspot_dab_threshold,
                    debug_dir=refinement_debug_base_dir,
                    candidate_index=i,
                    tumor_cell_mask_l2=cell_mask_binary_l2 # Context for StarDist (e.g. nuclei within cell regions)
                )
                if updated_hotspot is not None:
                    # Check for essential keys from refinement
                    if all(k in updated_hotspot for k in ['stardist_ki67_pos_count', 'stardist_total_count_filtered', 'positive_centroids', 'all_centroids']):
                         refined_hotspots_results.append(updated_hotspot)
                         logger.debug(f"Candidate {i+1} refined: Ki67+={updated_hotspot['stardist_ki67_pos_count']}, Total={updated_hotspot['stardist_total_count_filtered']}")
                    else:
                        logger.warning(f"Refinement output for candidate {i+1} missing essential keys. Skipping. Keys found: {list(updated_hotspot.keys())}")
                else:
                    logger.warning(f"StarDist refinement failed for candidate index {i+1}. Skipping.")


            if refined_hotspots_results:
                # Sort by StarDist Ki67 positive count
                refined_hotspots_results.sort(key=lambda item: item.get('stardist_ki67_pos_count', 0), reverse=True)
                hotspots = refined_hotspots_results[:hotspot_top_n] # Select top N

                logger.info(f"Selected final {len(hotspots)} hotspots (ranked by StarDist Ki67+ count):")
                for rank, hs in enumerate(hotspots):
                    density_val = hs.get('density_score', 'N/A') # This is the initial density score
                    density_str = f"{density_val:.4f}" if isinstance(density_val, (float, int)) else str(density_val)
                    logger.info(f"  Final Rank {rank+1}: Ki67+ (StarDist)={hs.get('stardist_ki67_pos_count','N/A')}, "
                                f"Total Cells (StarDist)={hs.get('stardist_total_count_filtered', 'N/A')}, "
                                f"PI (StarDist)={hs.get('stardist_proliferation_index', 0.0):.2%}, "
                                f"L0 Coords={hs.get('coords_l0')}, "
                                f"L{hs.get('level', hotspot_level)} Coords={hs.get('coords_level')}, "
                                f"Size L{hs.get('level', hotspot_level)}={hs.get('size_level')}, "
                                f"Initial Candidate Density={density_str}")
                    hs['final_score'] = hs.get('stardist_ki67_pos_count', 0) # Ensure final score is set
            else:
                hotspots = [] # Ensure empty list if no valid hotspots
                logger.warning("No hotspots available after StarDist refinement and ranking.")


        # === Stage 7: Overlay Generation ===
        logger.info(f"--- Stage 7: Overlay Generation (L{overlay_level}) ---")
        overlay_h, overlay_w = ds4_img.shape[:2] # ds4_img is at overlay_level (L2)
        target_overlay_shape_wh = (overlay_w, overlay_h)

        # Resize tissue_mask_l5 to overlay_level (L2)
        tissue_mask_overlay_final = cv2.resize(tissue_mask_l5, target_overlay_shape_wh, interpolation=cv2.INTER_NEAREST)

        # tumor_mask_l2 is already at overlay_level
        # cell_mask_binary_l2 is already at overlay_level
        final_overlay = visualization.generate_overlay(
            slide=slide,
            overlay_level=overlay_level,
            hotspot_level=hotspot_level, # Level at which hotspot coords are defined for drawing
            tissue_mask_overlay=tissue_mask_overlay_final,
            tumor_mask_overlay=tumor_mask_l2,
            cell_mask_binary_l2=cell_mask_binary_l2,
            hotspots=hotspots, # Final ranked hotspots
            debug_dir=overlay_debug_dir
        )
        if final_overlay is not None:
            overlay_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}.jpg")
            png_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}_temp.png") # Temp for lossless intermediate
            try:
                pil_image = Image.fromarray(final_overlay)
                pil_image.save(png_filename) # Save as PNG first
                with Image.open(png_filename) as img_png:
                    # Convert to RGB and save as JPEG with high quality, no subsampling
                    img_png.convert('RGB').save(overlay_filename, quality=95, subsampling=0)
                logger.info(f"Saved final overlay to {overlay_filename}")
                try:
                    os.remove(png_filename) # Clean up temp PNG
                except OSError: pass # Ignore if removal fails
            except Exception as e_pil_save:
                logger.error(f"Error saving overlay using PIL: {e_pil_save}. Trying fallback with OpenCV...")
                try:
                    cv2.imwrite(overlay_filename, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    logger.info(f"Saved overlay via OpenCV fallback to {overlay_filename}.")
                except Exception as e_cv2_save:
                    logger.error(f"Fallback overlay save with OpenCV also failed: {e_cv2_save}")
        else:
            logger.error("Failed to generate final overlay.")


        logger.info(f"--- Processing Finished for slide: {slide_name} ---")
        return hotspots


    except openslide.OpenSlideError as e:
        logger.error(f"OpenSlide error during processing of {slide_path}: {e}", exc_info=True)
        return None
    except FileNotFoundError as e:
        logger.error(f"File not found during processing of {slide_path}: {e}", exc_info=True)
        return None
    except MemoryError as e:
        logger.error(f"MemoryError during processing of {slide_path}: {e}", exc_info=True)
        # Re-raise MemoryError so it can be caught by a higher-level handler if needed,
        # or to stop the script if memory is exhausted.
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in pipeline for slide {slide_path}: {e}", exc_info=True)
        # Optionally log the full traceback string for more detailed debugging info
        # logger.error(traceback.format_exc())
        return None
    finally:
        if slide is not None:
            try:
                slide.close()
                logger.debug(f"OpenSlide object for {slide_path} closed.")
            except Exception as e_close:
                logger.warning(f"Error closing slide object for {slide_path}: {e_close}")