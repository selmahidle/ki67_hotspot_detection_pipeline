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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    return iou


def apply_nms_to_candidates(candidates, iou_threshold=0.5, force_level_coords=False):
    """
    Applies NMS to a list of candidates.
    Assumes the 'candidates' list is already sorted from highest to lowest score.
    """
    if not candidates:
        return []

    coord_key = 'coords_level' if force_level_coords else 'coords_l0'
    size_key = 'size_level' if force_level_coords else 'size_l0'
    keep = []

    boxes = np.array([
        [c[coord_key][0], c[coord_key][1], c[size_key][0], c[size_key][1]]
        for c in candidates
    ])

    while len(candidates) > 0:
        best_candidate = candidates.pop(0)
        best_box = boxes[0]
        boxes = boxes[1:]
        keep.append(best_candidate)

        if len(candidates) == 0:
            break

        ious = np.array([calculate_iou(best_box, other_box) for other_box in boxes])
        remaining_indices = np.where(ious <= iou_threshold)[0]
        candidates = [candidates[i] for i in remaining_indices]
        boxes = boxes[remaining_indices]

    return keep


def process_slide_ki67(slide_path, output_dir, tumor_models, cell_model, stardist_model, device):
    """
    Process a slide for Ki67 analysis. Supports single-model or multi-model consensus
    for tumor segmentation, a single cell model (AttentionUNet), and StarDist refinement.
    """
    slide = None
    try:
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        logger.info(f"Processing slide: {slide_name} with {len(tumor_models)} tumor model(s).")
        
        image_output_dir = os.path.join(output_dir, slide_name)
        os.makedirs(image_output_dir, exist_ok=True)
        debug_mask_dir = os.path.join(image_output_dir, "debug_masks")
        os.makedirs(debug_mask_dir, exist_ok=True)
        hotspot_debug_dir = os.path.join(image_output_dir, f"{slide_name}_hotspot_debug")
        os.makedirs(hotspot_debug_dir, exist_ok=True)
        overlay_debug_dir = os.path.join(image_output_dir, f"{slide_name}_overlay_debug")
        os.makedirs(overlay_debug_dir, exist_ok=True)

        slide = openslide.open_slide(slide_path)

        # --- Define levels and general parameters ---
        tissue_level = 5; tumor_level = 3; cell_level = 2; hotspot_level = 2
        overlay_level = 2; ds8_level = 3; ds4_level = 2

        cell_patch_size = 1024; cell_overlap = 256
        cell_output_channels = 1 
        cell_batch_size = 8; cell_prob_threshold = 0.3

        hotspot_patch_size_l0 = 2048
        hotspot_top_n = 5
        hotspot_dab_threshold = 0.1

        required_levels_actual = {'tissue': tissue_level, 'ds8': ds8_level, 'ds4': ds4_level}
        max_req_level = max(required_levels_actual.values())
        if max_req_level >= slide.level_count:
            logger.error(f"Slide {slide_name} level count ({slide.level_count}) insufficient. Max required level index: {max_req_level}.")
            if slide: slide.close(); return None

        # --- Load and/or generate ds8 and ds4 images ---
        ds8_path = os.path.join(image_output_dir, f"{slide_name}_ds8.jpg")
        ds8_img = None
        if os.path.exists(ds8_path):
            ds8_img_bgr = cv2.imread(ds8_path)
            if ds8_img_bgr is not None: ds8_img = cv2.cvtColor(ds8_img_bgr, cv2.COLOR_BGR2RGB)
        if ds8_img is None:
            logger.info(f"Generating ds8 image (Level {ds8_level})...")
            ds8_dims = slide.level_dimensions[ds8_level]
            ds8_img_pil = slide.read_region((0,0), ds8_level, ds8_dims).convert('RGB')
            ds8_img = np.array(ds8_img_pil)
            cv2.imwrite(ds8_path, cv2.cvtColor(ds8_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ds8_img is None: logger.error("ds8 image could not be loaded or generated. Cannot proceed."); return None

        ds4_path = os.path.join(image_output_dir, f"{slide_name}_ds4.jpg")
        ds4_img = None
        if os.path.exists(ds4_path):
            ds4_img_bgr = cv2.imread(ds4_path)
            if ds4_img_bgr is not None: ds4_img = cv2.cvtColor(ds4_img_bgr, cv2.COLOR_BGR2RGB) # BUG FIX: was ds8_img_bgr
        if ds4_img is None:
            logger.info(f"Generating ds4 image (Level {ds4_level})...")
            ds4_dims = slide.level_dimensions[ds4_level]
            ds4_img_pil = slide.read_region((0,0), ds4_level, ds4_dims).convert('RGB')
            ds4_img = np.array(ds4_img_pil)
            cv2.imwrite(ds4_path, cv2.cvtColor(ds4_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ds4_img is None: logger.error("ds4 image could not be loaded or generated. Cannot proceed."); return None

        # --- Stage 1: Tissue Detection ---
        logger.info(f"--- Stage 1: Tissue Detection (L{tissue_level}) ---")
        tissue_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tissue_mask_L{tissue_level}.jpg")
        tissue_mask_l5 = None
        if os.path.exists(tissue_mask_filename):
            tissue_mask_l5_255 = cv2.imread(tissue_mask_filename, cv2.IMREAD_GRAYSCALE)
            if tissue_mask_l5_255 is not None: tissue_mask_l5 = (tissue_mask_l5_255 > 128).astype(np.uint8)
        if tissue_mask_l5 is None:
            logger.info(f"Generating tissue mask L{tissue_level}...")
            tissue_level_dims = slide.level_dimensions[tissue_level]
            tissue_img_pil = slide.read_region((0,0), tissue_level, tissue_level_dims).convert('RGB')
            tissue_img_bgr = cv2.cvtColor(np.array(tissue_img_pil), cv2.COLOR_RGB2BGR)
            tissue_mask_l5_255 = detect_tissue(tissue_img_bgr, threshold_tissue_ratio=0.05)
            if tissue_mask_l5_255 is not None and np.sum(tissue_mask_l5_255) > 0:
                tissue_mask_l5 = (tissue_mask_l5_255 > 0).astype(np.uint8)
                cv2.imwrite(tissue_mask_filename, tissue_mask_l5 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if tissue_mask_l5 is None or np.sum(tissue_mask_l5) == 0: logger.error("Tissue mask is empty. Cannot proceed."); return None
        
        tumor_level_h, tumor_level_w = ds8_img.shape[:2]
        tissue_mask_l3 = cv2.resize(tissue_mask_l5, (tumor_level_w, tumor_level_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Upsampled tissue mask to L{tumor_level} ({tissue_mask_l3.shape})")

        # --- Stage 2: Tumor Segmentation (Single vs. Consensus) ---
        logger.info(f"--- Stage 2: Tumor Segmentation (L{tumor_level}) ---")
        is_consensus_mode = len(tumor_models) > 1

        if is_consensus_mode:
            logger.info(f"Running in Consensus Mode with {len(tumor_models)} models.")
            tumor_patch_size = 4096; tumor_overlap = 1024
            tumor_output_channels = 2; tumor_batch_size = 2
            tumor_prob_threshold = 0.3
            transforms_for_tumor_seg = get_transforms() # With CLAHE
            processed_tumor_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_consensus_L{tumor_level}.jpg")
        else:
            logger.info("Running in Single Model Mode.")
            tumor_patch_size = 1024; tumor_overlap = 256
            tumor_output_channels = 1; tumor_batch_size = 4
            tumor_prob_threshold = 0.5
            transforms_for_tumor_seg = get_transforms_no_clahe()
            processed_tumor_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_single_L{tumor_level}.jpg")

        tumor_mask_l3_processed = None 
        if os.path.exists(processed_tumor_mask_filename):
            loaded_mask = cv2.imread(processed_tumor_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is not None:
                tumor_mask_l3_processed = (loaded_mask > 128).astype(np.uint8) 
                logger.info(f"Loaded existing final tumor mask: {processed_tumor_mask_filename}")
        
        if tumor_mask_l3_processed is None:
            logger.info(f"Final tumor mask not found. Running inference...")
            tumor_patches, tumor_locations, tumor_weights = [], [], []
            h, w = ds8_img.shape[:2]
            stride_x, stride_y = tumor_patch_size - tumor_overlap, tumor_patch_size - tumor_overlap
            
            for y_coord in tqdm(range(0, h, stride_y), desc="Collecting Tumor Patches"):
                for x_coord in range(0, w, stride_x):
                    y_start, x_start = y_coord, x_coord
                    end_y, end_x = min(y_start + tumor_patch_size, h), min(x_start + tumor_patch_size, w)
                    window_h, window_w = end_y - y_start, end_x - x_start
                    if window_w < stride_x // 2 or window_h < stride_y // 2: continue
                    if np.mean(tissue_mask_l3[y_start:end_y, x_start:end_x]) < 0.05: continue 
                    
                    patch = ds8_img[y_start:end_y, x_start:end_x]
                    pad_h, pad_w = max(0, tumor_patch_size - window_h), max(0, tumor_patch_size - window_w)
                    padded_patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                    transformed = transforms_for_tumor_seg(image=padded_patch)
                    tumor_patches.append(transformed["image"])
                    tumor_locations.append((y_start, end_y, x_start, end_x))
                    weight_map = create_weight_map((window_h, window_w))
                    padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    tumor_weights.append(padded_weight_map)

            if not tumor_patches: logger.error(f"Tumor patch collection failed."); return None
            
            raw_tumor_mask_l3 = None
            if is_consensus_mode:
                individual_model_masks = []
                for i, model in enumerate(tumor_models):
                    model_name = f"TumorModel_Consensus_{i+1}"
                    prob_map = run_inference_on_patches(model, device, tumor_output_channels, tumor_batch_size, f"Tumor L{tumor_level}", (w, h), tumor_patches, tumor_locations, tumor_weights, model_name)
                    if prob_map is None: logger.error(f"{model_name} failed. Consensus cannot be reached."); return None
                    binary_mask = (prob_map[:, :, 1] > tumor_prob_threshold) # Channel 1 is foreground
                    individual_model_masks.append(binary_mask)
                logger.info("Combining masks using strict consensus (logical AND)...")
                raw_tumor_mask_l3 = np.logical_and.reduce(individual_model_masks).astype(np.uint8)
            else:
                tumor_model = tumor_models[0]
                model_name = f"TumorModel_Single_{tumor_model.__class__.__name__}"
                tumor_prob_map_l3 = run_inference_on_patches(tumor_model, device, tumor_output_channels, tumor_batch_size, f"Tumor L{tumor_level}", (w, h), tumor_patches, tumor_locations, tumor_weights, model_name)
                if tumor_prob_map_l3 is None: logger.error(f"{model_name} inference failed."); return None
                if tumor_prob_map_l3.ndim == 3: tumor_prob_map_l3 = tumor_prob_map_l3.squeeze(-1)
                raw_tumor_mask_l3 = (tumor_prob_map_l3 > tumor_prob_threshold).astype(np.uint8)
            
            del tumor_patches, tumor_locations, tumor_weights
            raw_tumor_mask_l3_on_tissue = raw_tumor_mask_l3 * tissue_mask_l3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tumor_mask_l3_processed = cv2.morphologyEx(raw_tumor_mask_l3_on_tissue, cv2.MORPH_CLOSE, kernel)
            tumor_mask_l3_processed = cv2.morphologyEx(tumor_mask_l3_processed, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(processed_tumor_mask_filename, tumor_mask_l3_processed * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        if tumor_mask_l3_processed is None or np.sum(tumor_mask_l3_processed) == 0:
            logger.error("Tumor mask is empty. Cannot proceed."); return None

        cell_level_h, cell_level_w = ds4_img.shape[:2]
        tumor_mask_l2 = cv2.resize(tumor_mask_l3_processed, (cell_level_w, cell_level_h), interpolation=cv2.INTER_NEAREST)

        # --- Stage 3: Cell Segmentation ---
        logger.info(f"--- Stage 3: Cell Segmentation (AttentionUNet L{cell_level}) ---")
        cell_mask_filename = os.path.join(image_output_dir, f"{slide_name}_cell_mask_binary_L{cell_level}_AttnUNet.jpg")
        cell_mask_binary_l2 = None
        if os.path.exists(cell_mask_filename):
            loaded_cell_mask = cv2.imread(cell_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_cell_mask is not None:
                 cell_mask_binary_l2 = (loaded_cell_mask > 128).astype(np.uint8)
        
        if cell_mask_binary_l2 is None:
            logger.info(f"Running AttentionUNet cell segmentation inference L{cell_level}...")
            cell_patches, cell_locations, cell_weights = [], [], []
            h_cell, w_cell = ds4_img.shape[:2]
            stride_x_cell, stride_y_cell = cell_patch_size - cell_overlap, cell_patch_size - cell_overlap
            for y_coord_cell in tqdm(range(0, h_cell, stride_y_cell), desc="Collecting Cell Patches (AttnUNet)"):
                for x_coord_cell in range(0, w_cell, stride_x_cell):
                    y_start_c, x_start_c = y_coord_cell, x_coord_cell
                    end_y_c, end_x_c = min(y_start_c + cell_patch_size, h_cell), min(x_start_c + cell_patch_size, w_cell)
                    window_h_c, window_w_c = end_y_c - y_start_c, end_x_c - x_start_c
                    if window_w_c < stride_x_cell//2 or window_h_c < stride_y_cell//2: continue
                    if np.mean(tumor_mask_l2[y_start_c:end_y_c, x_start_c:end_x_c]) < 0.05: continue
                    patch_c = ds4_img[y_start_c:end_y_c, x_start_c:end_x_c]
                    pad_h_c, pad_w_c = max(0, cell_patch_size-window_h_c), max(0, cell_patch_size-window_w_c)
                    padded_patch_c = cv2.copyMakeBorder(patch_c, 0, pad_h_c, 0, pad_w_c, cv2.BORDER_CONSTANT, value=[0,0,0])
                    transformed_c = get_transforms_no_clahe()(image=padded_patch_c)
                    cell_patches.append(transformed_c["image"])
                    cell_locations.append((y_start_c, end_y_c, x_start_c, end_x_c))
                    weight_map_c = create_weight_map((window_h_c, window_w_c))
                    padded_weight_map_c = cv2.copyMakeBorder(weight_map_c, 0, pad_h_c, 0, pad_w_c, cv2.BORDER_CONSTANT, value=0)
                    cell_weights.append(padded_weight_map_c)

            if cell_patches:
                cell_prob_map_l2 = run_inference_on_patches(cell_model, device, cell_output_channels, cell_batch_size, f"Cell L{cell_level}", (w_cell, h_cell), cell_patches, cell_locations, cell_weights, "AttentionUNet_CellModel")
                del cell_patches, cell_locations, cell_weights
                if cell_prob_map_l2 is not None:
                    if cell_prob_map_l2.ndim == 3: cell_prob_map_l2 = cell_prob_map_l2.squeeze(-1)
                    raw_cell_mask_l2 = (cell_prob_map_l2 > cell_prob_threshold).astype(np.uint8)
                    cell_mask_binary_l2 = raw_cell_mask_l2 * tumor_mask_l2
                    cv2.imwrite(cell_mask_filename, cell_mask_binary_l2 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        if cell_mask_binary_l2 is None:
            logger.warning("Cell mask is None. Creating empty mask for subsequent steps.")
            cell_mask_binary_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)

        # --- Stage 4: DAB+ Mask Calculation ---
        logger.info(f"--- Stage 4: DAB+ Mask Calculation (L{hotspot_level}) ---")
        dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
        if np.sum(tumor_mask_l2) > 0:
            bbox = [p.bbox for p in regionprops(label(tumor_mask_l2))]
            if bbox:
                min_r, min_c, max_r, max_c = np.min([b[0] for b in bbox]), np.min([b[1] for b in bbox]), np.max([b[2] for b in bbox]), np.max([b[3] for b in bbox])
                bbox_h_dab, bbox_w_dab = max_r - min_r, max_c - min_c
                if bbox_h_dab > 0 and bbox_w_dab > 0:
                    level0_x, level0_y = int(min_c * slide.level_downsamples[hotspot_level]), int(min_r * slide.level_downsamples[hotspot_level])
                    rgb_patch_pil = slide.read_region((level0_x, level0_y), hotspot_level, (bbox_w_dab, bbox_h_dab)).convert('RGB')
                    dab_patch = get_dab_mask(np.array(rgb_patch_pil), hotspot_dab_threshold)
                    if dab_patch is not None: dab_plus_mask_l2[min_r:max_r, min_c:max_c] = dab_patch

        # --- Stage 5: Initial Hotspot Candidate ID ---
        logger.info(f"--- Stage 5: Initial Hotspot Candidate ID (L{hotspot_level}) ---")
        hotspot_target_mask_coarse = (cell_mask_binary_l2 > 0) & (dab_plus_mask_l2 > 0) & (tumor_mask_l2 > 0)
        candidate_hotspots = []
        if np.sum(hotspot_target_mask_coarse) > 0:
            candidate_hotspots = identify_hotspots(slide, hotspot_level, hotspot_target_mask_coarse.astype(np.uint8), hotspot_patch_size_l0, 10, hotspot_debug_dir)
            if candidate_hotspots:
                candidate_hotspots = apply_nms_to_candidates(candidate_hotspots, iou_threshold=0.5, force_level_coords=False)

        # --- Stage 6: StarDist Refinement & Re-ranking ---
        hotspots = []
        if not candidate_hotspots:
            logger.warning("No initial candidates available for StarDist refinement.")
        else:
            logger.info(f"--- Stage 6: StarDist Refinement & Re-ranking (L{hotspot_level}) ---")
            refined_hotspots_results = []
            refinement_debug_base_dir = os.path.join(hotspot_debug_dir, "refinement_patches")
            os.makedirs(refinement_debug_base_dir, exist_ok=True)
            actual_pixel_size_um = get_actual_pixel_size_um(slide, level=hotspot_level)

            for i, candidate in enumerate(tqdm(candidate_hotspots, desc="Refining Hotspots with StarDist")):
                updated_hotspot = refine_hotspot_with_stardist(
                    candidate_hotspot=candidate,
                    stardist_model=stardist_model,
                    slide=slide,
                    hotspot_level=hotspot_level,
                    actual_pixel_size_um=actual_pixel_size_um,
                    dab_threshold=hotspot_dab_threshold,
                    debug_dir=refinement_debug_base_dir,
                    candidate_index=i,
                    tumor_cell_mask_l2=cell_mask_binary_l2 
                )
                if updated_hotspot and all(k in updated_hotspot for k in ['stardist_ki67_pos_count', 'stardist_total_count_filtered']):
                    refined_hotspots_results.append(updated_hotspot)
            
            if refined_hotspots_results:
                refined_hotspots_results.sort(key=lambda hs: hs.get('stardist_proliferation_index', 0), reverse=True)
                final_hotspots_after_nms = apply_nms_to_candidates(refined_hotspots_results, iou_threshold=0.5, force_level_coords=True)
                hotspots = final_hotspots_after_nms[:hotspot_top_n]
                for rank, hs in enumerate(hotspots):
                    hs['final_score'] = hs.get('stardist_proliferation_index', 0.0)
                    logger.info(f"  Final Rank {rank+1}: PI={hs['final_score']:.2%}, Coords=L{hotspot_level} {hs.get('coords_level')}")

        # --- Stage 7: Overlay Generation ---
        logger.info(f"--- Stage 7: Overlay Generation (L{overlay_level}) ---")
        overlay_h, overlay_w = ds4_img.shape[:2] 
        tissue_mask_overlay_final = cv2.resize(tissue_mask_l5, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)

        final_overlay = visualization.generate_overlay(
            slide=slide,
            overlay_level=overlay_level,
            hotspot_level=hotspot_level,
            tissue_mask_overlay=tissue_mask_overlay_final,
            tumor_mask_overlay=tumor_mask_l2,
            cell_mask_binary_l2=cell_mask_binary_l2,
            dab_plus_mask_l2_overlay=dab_plus_mask_l2, # Ensure correct parameter name
            hotspots=hotspots,
            debug_dir=overlay_debug_dir
        )
        if final_overlay is not None:
            overlay_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}.jpg")
            Image.fromarray(final_overlay).convert('RGB').save(overlay_filename, quality=95, subsampling=0)
            logger.info(f"Saved final overlay to {overlay_filename}")
        else:
            logger.error("Failed to generate final overlay.")

        logger.info(f"--- Processing Finished for slide: {slide_name} ---")
        return hotspots

    except Exception as e:
        logger.error(f"Unexpected error in pipeline for slide {slide_path}: {e}", exc_info=True)
        if isinstance(e, MemoryError): raise e 
        return None
    finally:
        if slide is not None:
            try: slide.close()
            except Exception as e_close: logger.warning(f"Error closing slide: {e_close}")