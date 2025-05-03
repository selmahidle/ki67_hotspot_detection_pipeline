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

# --- Local Application Imports ---
from transforms import get_transforms, get_transforms_no_clahe
from tissue_detection import detect_tissue
from patching import run_inference_on_patches
from utils import create_weight_map
from stain_utils import get_dab_mask
from hotspot_detection import identify_hotspots
from visualization import generate_overlay 
from stardist_utils import refine_hotspot_with_stardist

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None # Allow loading large images if needed by PIL indirectly

# --- Modify function signature ---
def process_slide_ki67(slide_path, output_dir, tumor_models, cell_models, device):
    """
    Process a slide for Ki67 analysis using tumor segmentation, cell segmentation (SMP),
    and StarDist-based hotspot refinement.

    Pipeline:
    1. Load slide, save overviews (ds8, ds4).
    2. Detect tissue (L5).
    3. Segment tumor (SMP using ds8).
    4. Segment cells (SMP using ds4), conditioned on tumor mask -> cell_mask_binary_l2.
    5. Calculate DAB+ mask within tumor regions at L2 -> dab_plus_mask_l2.
    6. Identify *candidate* hotspots based on density of (SMP Cells AND DAB+) at L2.
    7. *Refine* candidates using stardist_utils.refine_hotspot_with_stardist.
    8. Re-rank hotspots based on refined StarDist counts.
    9. Generate final overlay (L2).
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
        # Levels
        tissue_level = 5
        tumor_level = 3   # Effective level for tumor mask (ds8)
        cell_level = 2    # Effective level for SMP cell mask (ds4)
        hotspot_level = 2 # Level for hotspot candidates and StarDist refinement (L2)
        overlay_level = 2 # Final visualization at L2
        ds8_level = 3     # WSI level corresponding to ds8
        ds4_level = 2     # WSI level corresponding to ds4

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
        hotspot_patch_size_l0 = 2048 # Field of view size at L0
        hotspot_top_n = 5 # Final number of hotspots to return
        hotspot_dab_threshold = 0.15 # Threshold for get_dab_mask

        # --- Check Levels Exist ---
        required_levels_actual = {'tissue': tissue_level, 'ds8': ds8_level, 'ds4': ds4_level}
        max_req_level = max(required_levels_actual.values())
        if max_req_level >= slide.level_count:
            logger.error(f"Slide {slide_name} level count ({slide.level_count}) insufficient. Max required level index: {max_req_level}.")
            if slide: slide.close()
            return None

        # --- Get Transforms ---
        transforms_clahe = get_transforms()
        transforms_no_clahe = get_transforms_no_clahe()

        # === Pre-computation: Save/Load ds8 and ds4 JPGs === #
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
                 # Save as JPEG
                 cv2.imwrite(ds8_path, cv2.cvtColor(ds8_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                 logger.info(f"Saved ds8 image to {ds8_path}")
            except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError generating ds8: {ose}", exc_info=True); return None
            except Exception as e: logger.error(f"Failed to generate/save ds8: {e}", exc_info=True); return None
        if ds8_img is None: # Final check
            logger.error("ds8 image could not be loaded or generated. Cannot proceed.")
            if slide: slide.close()
            return None

        # --- DS4 ---
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
            else:
                try:
                    ds4_dims = slide.level_dimensions[ds4_level]
                    logger.info(f"Reading region for ds4: level={ds4_level}, dims={ds4_dims}")
                    # Handle potential large image read errors
                    try:
                         ds4_img_pil = slide.read_region((0,0), ds4_level, ds4_dims).convert('RGB')
                    except openslide.OpenSlideError as ose:
                         logger.error(f"OpenSlideError reading ds4 region: {ose}. Try checking slide format/integrity.")
                         return None
                    except MemoryError as me: # Catch memory errors explicitly
                         logger.error(f"MemoryError reading ds4 region ({ds4_dims}). Slide level might be too large.")
                         raise me # Re-raise MemoryError as it might indicate system limitation

                    logger.info("slide.read_region for ds4 succeeded.")
                    ds4_img = np.array(ds4_img_pil)
                    logger.info(f"Converted ds4 PIL to numpy, shape: {ds4_img.shape}, dtype: {ds4_img.dtype}")
                    if ds4_img is None or ds4_img.size == 0: logger.error("Numpy array for ds4 empty!"); ds4_img = None
                    else:
                        logger.info(f"Attempting to write ds4 image to {ds4_path}")
                        success = cv2.imwrite(ds4_path, cv2.cvtColor(ds4_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if success: logger.info(f"Successfully saved generated ds4 image.")
                        else: logger.error(f"cv2.imwrite failed for ds4!"); ds4_img = None # Reset ds4_img if save fails
                except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError during ds4 generation: {ose}", exc_info=True); ds4_img = None
                except MemoryError as me: logger.error(f"MemoryError during ds4 generation: {me}", exc_info=True); raise me # Re-raise
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
                 tissue_mask_l5_255 = detect_tissue(tissue_img_bgr, threshold_tissue_ratio=0.05)
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

        # === 3. Tumor Segmentation (using ds8 JPG) === #
        logger.info(f"--- Stage 2: Tumor Segmentation (L{tumor_level}) ---")
        processed_tumor_mask_filename = os.path.join(image_output_dir, f"{slide_name}_tumor_mask_L{tumor_level}.jpg")
        consensus_tumor_mask_l3_raw = None
        tumor_mask_l3_processed = None
        skip_tumor_stage = False

        if os.path.exists(processed_tumor_mask_filename):
            logger.info(f"Found existing final tumor mask: {processed_tumor_mask_filename}. Loading.")
            loaded_mask = cv2.imread(processed_tumor_mask_filename, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is None: logger.error(f"Failed to load existing tumor mask from {processed_tumor_mask_filename}. Will run inference.")
            else:
                 consensus_tumor_mask_l3_raw = (loaded_mask > 128).astype(np.uint8)
                 tumor_mask_l3_processed = consensus_tumor_mask_l3_raw # Use loaded mask
                 logger.info(f"Loaded tumor mask L{tumor_level} ({consensus_tumor_mask_l3_raw.shape}).")
                 skip_tumor_stage = True
        else: logger.info(f"Final tumor mask {processed_tumor_mask_filename} not found. Running inference.")

        if not skip_tumor_stage:
            # --- Tumor Inference Logic ---
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
                    tumor_locations.append((y_start, end_y, x_start, end_x)) # Store original location

                    # Create weight map for the ORIGINAL patch dimensions
                    weight_map = create_weight_map((window_h, window_w))
                    if weight_map is None:
                        logger.error(f"Failed to create weight map for tumor patch {window_w}x{window_h}. Skipping patch.")
                        if tumor_patches: tumor_patches.pop() # Remove corresponding patch/location
                        if tumor_locations: tumor_locations.pop()
                        continue

                    # *** REVERTED: Pad the weight map and append the PADDED version ***
                    if pad_h > 0 or pad_w > 0:
                        padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    else:
                        padded_weight_map = weight_map # No padding needed
                    tumor_weights.append(padded_weight_map)
                    # *****************************************************************

            # Check collected data consistency
            if not tumor_patches: logger.error(f"Tumor patch collection failed (no patches gathered)."); return None
            if not (len(tumor_patches) == len(tumor_locations) == len(tumor_weights)):
                 logger.error(f"Mismatch between collected tumor patches ({len(tumor_patches)}), locations ({len(tumor_locations)}), and weights ({len(tumor_weights)})!")
                 return None
            logger.info(f"Collected {len(tumor_patches)} tumor patches.")

            # Run inference
            tumor_prob_maps_l3 = []
            for i, model in enumerate(tumor_models):
                model_name = f"TumorModel_{i+1}"
                logger.info(f"Running tumor inference with {model_name}...")
                # Call run_inference_on_patches with PADDED weights
                probs = run_inference_on_patches(
                    model, device, tumor_output_channels, tumor_batch_size,
                    f"Tumor L{tumor_level}", (w, h), tumor_patches,
                    tumor_locations, tumor_weights, model_name # Pass padded weights
                )
                if probs is not None: tumor_prob_maps_l3.append(probs)
                else: logger.warning(f"{model_name} failed during inference.")
            del tumor_patches, tumor_locations, tumor_weights # Free memory

            if not tumor_prob_maps_l3: logger.error("All tumor models failed inference."); return None

            # Process model outputs
            individual_model_masks = []
            num_successful_models = len(tumor_prob_maps_l3)
            logger.info(f"Processing {num_successful_models} tumor model outputs...")
            for i, prob_map in enumerate(tumor_prob_maps_l3):
                model_mask_bool = (prob_map[:, :, 1] > tumor_prob_threshold)
                individual_model_masks.append(model_mask_bool)
                model_mask_filename = os.path.join(debug_mask_dir, f"{slide_name}_tumor_mask_model{i+1}_th{tumor_prob_threshold}_L{tumor_level}.png")
                try: cv2.imwrite(model_mask_filename, model_mask_bool.astype(np.uint8) * 255)
                except Exception as e_save: logger.warning(f"Could not save debug tumor mask {model_mask_filename}: {e_save}")

            # Consensus and Post-processing
            if len(individual_model_masks) != num_successful_models or num_successful_models == 0: logger.error("Consensus failed due to mismatch or no successful models."); return None
            consensus_mask_bool = np.logical_and.reduce(individual_model_masks)
            consensus_tumor_mask_l3_raw = consensus_mask_bool.astype(np.uint8) * tissue_mask_l3
            del tumor_prob_maps_l3, individual_model_masks, consensus_mask_bool
            logger.info(f"Raw tumor consensus mask generated. Pixels: {np.sum(consensus_tumor_mask_l3_raw)}")
            raw_consensus_filename = os.path.join(debug_mask_dir, f"{slide_name}_tumor_mask_raw_consensus_L{tumor_level}.png")
            try: cv2.imwrite(raw_consensus_filename, consensus_tumor_mask_l3_raw * 255)
            except Exception as e_save: logger.warning(f"Could not save debug raw consensus mask: {e_save}")

            # Morphological smoothing
            smooth_kernel_size = 5; kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
            tumor_mask_l3_processed = cv2.morphologyEx(consensus_tumor_mask_l3_raw, cv2.MORPH_CLOSE, kernel)
            tumor_mask_l3_processed = cv2.morphologyEx(tumor_mask_l3_processed, cv2.MORPH_OPEN, kernel)
            logger.info(f"Applied smoothing to tumor mask. Pixels after smoothing: {np.sum(tumor_mask_l3_processed)}")
            try: cv2.imwrite(processed_tumor_mask_filename, tumor_mask_l3_processed * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            except Exception as e_save: logger.error(f"Failed to save final processed tumor mask {processed_tumor_mask_filename}: {e_save}")
            debug_processed_filename = os.path.join(debug_mask_dir, f"{slide_name}_tumor_mask_processed_L{tumor_level}.png")
            try: cv2.imwrite(debug_processed_filename, tumor_mask_l3_processed * 255)
            except Exception as e_save: logger.warning(f"Could not save debug processed tumor mask: {e_save}")
        # --- END OF TUMOR INFERENCE BLOCK ---

        if tumor_mask_l3_processed is None or np.sum(tumor_mask_l3_processed) == 0:
            logger.error("Tumor mask empty or invalid after load/inference. Cannot proceed.")
            return None

        # === 4. Upsample Processed Tumor Mask to Analysis Level (L2) ===
        cell_level_h, cell_level_w = ds4_img.shape[:2]
        tumor_mask_l2 = cv2.resize(tumor_mask_l3_processed, (cell_level_w, cell_level_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Upsampled processed tumor mask to L{hotspot_level} ({tumor_mask_l2.shape})")
        debug_tumor_l2_fn = os.path.join(debug_mask_dir, f"{slide_name}_tumor_mask_processed_L{hotspot_level}.png")
        try: cv2.imwrite(debug_tumor_l2_fn, tumor_mask_l2*255)
        except Exception as e_save: logger.warning(f"Could not save L2 tumor mask {debug_tumor_l2_fn}: {e_save}")


        # === 5. Tumor Cell Segmentation (SMP - using ds4 JPG) === #
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
            # --- Cell Inference Logic ---
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

                         # Create weight map for the ORIGINAL patch dimensions
                         weight_map = create_weight_map((window_h, window_w))
                         if weight_map is None:
                             logger.error(f"Failed to create weight map for cell patch {window_w}x{window_h}. Skipping.")
                             if cell_patches: cell_patches.pop()
                             if cell_locations: cell_locations.pop()
                             continue

                         # *** REVERTED: Pad the weight map and append the PADDED version ***
                         if pad_h > 0 or pad_w > 0:
                             padded_weight_map = cv2.copyMakeBorder(weight_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                         else:
                              padded_weight_map = weight_map
                         cell_weights.append(padded_weight_map)
                         # *******************************************************************

                 # Check collected data consistency
                 if not cell_patches: logger.warning("SMP Cell patch collection failed (no patches gathered matching tumor)."); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
                 if not (len(cell_patches) == len(cell_locations) == len(cell_weights)):
                     logger.error(f"Mismatch between collected cell patches ({len(cell_patches)}), locations ({len(cell_locations)}), and weights ({len(cell_weights)})!")
                     cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8) # Set empty mask on error

                 # Only run inference if patches were collected
                 if cell_patches:
                      logger.info(f"Collected {len(cell_patches)} SMP cell patches.")
                      cell_prob_maps_l2 = []
                      for i, model in enumerate(cell_models):
                           model_name = f"CellModel_{i+1}"
                           logger.info(f"Running SMP cell inference with {model_name}...")
                           # Call run_inference_on_patches with PADDED weights
                           probs = run_inference_on_patches(
                               model, device, cell_output_channels, cell_batch_size,
                               f"Cell L{cell_level}", (w, h), cell_patches,
                               cell_locations, cell_weights, model_name # Pass padded weights
                           )
                           if probs is not None: cell_prob_maps_l2.append(probs)
                           else: logger.warning(f"{model_name} failed during inference.")
                      del cell_patches, cell_locations, cell_weights # Free memory

                      if not cell_prob_maps_l2: logger.error("All SMP cell models failed."); cell_mask_binary_l2 = np.zeros((h, w), dtype=np.uint8)
                      else:
                           logger.info(f"Averaging {len(cell_prob_maps_l2)} SMP cell probability maps...")
                           combined_cell_prob = np.mean(cell_prob_maps_l2, axis=0).astype(np.float32)
                           del cell_prob_maps_l2 # Free memory
                           cell_mask_binary_l2_raw = (combined_cell_prob[:,:,1] > cell_prob_threshold).astype(np.uint8)
                           cell_mask_binary_l2 = cell_mask_binary_l2_raw * tumor_mask_l2
                           logger.info(f"Final SMP cell mask created (thresholded & tumor masked). Pixels: {np.sum(cell_mask_binary_l2)}")
                           try: cv2.imwrite(cell_mask_filename, cell_mask_binary_l2 * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                           except Exception as e_save: logger.error(f"Failed to save final cell mask {cell_mask_filename}: {e_save}")
                           debug_cell_fn = os.path.join(debug_mask_dir, f"{slide_name}_cell_mask_binary_L{cell_level}.png")
                           try: cv2.imwrite(debug_cell_fn, cell_mask_binary_l2 * 255)
                           except Exception as e_save: logger.warning(f"Could not save debug cell mask {debug_cell_fn}: {e_save}")
        # --- END OF CELL INFERENCE BLOCK ---

        if cell_mask_binary_l2 is None: logger.warning("Cell mask is None after load/inference. Creating empty mask."); cell_mask_binary_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
        if np.sum(cell_mask_binary_l2) == 0: logger.warning("SMP Cell mask is empty after processing (or no cells found in tumor).")


        # === 6. Pre-calculate DAB+ Mask (L2) ===
        logger.info(f"--- Stage 4: DAB+ Mask Calculation (L{hotspot_level}) ---")
        dab_plus_mask_l2 = None
        # Only calculate DAB if tumor mask is not empty
        if np.sum(tumor_mask_l2) > 0:
            try:
                tumor_labels = label(tumor_mask_l2); tumor_props = regionprops(tumor_labels)
                if tumor_props:
                    min_r = min(prop.bbox[0] for prop in tumor_props)
                    min_c = min(prop.bbox[1] for prop in tumor_props)
                    max_r = max(prop.bbox[2] for prop in tumor_props)
                    max_c = max(prop.bbox[3] for prop in tumor_props)
                    bbox_h, bbox_w = max_r - min_r, max_c - min_c
                    logger.info(f"Tumor bounding box at L{hotspot_level}: ({min_c}, {min_r}, {bbox_w}, {bbox_h})")

                    if bbox_h > 0 and bbox_w > 0:
                        level0_x_read = int(min_c * slide.level_downsamples[hotspot_level])
                        level0_y_read = int(min_r * slide.level_downsamples[hotspot_level])
                        try:
                            logger.debug(f"Reading region for DAB: L0=({level0_x_read},{level0_y_read}), Level={hotspot_level}, Size=({bbox_w},{bbox_h})")
                            rgb_patch_l2_pil = slide.read_region((level0_x_read, level0_y_read), hotspot_level, (bbox_w, bbox_h)).convert('RGB')
                            rgb_patch_l2 = np.array(rgb_patch_l2_pil)
                            if rgb_patch_l2.shape[0]!=bbox_h or rgb_patch_l2.shape[1]!=bbox_w:
                                 logger.warning(f"Read patch size {rgb_patch_l2.shape[:2]} for DAB != expected ({bbox_h},{bbox_w}). Resizing.")
                                 rgb_patch_l2=cv2.resize(rgb_patch_l2, (bbox_w,bbox_h), interpolation=cv2.INTER_LINEAR)

                            dab_plus_mask_patch = get_dab_mask(rgb_patch_l2, hotspot_dab_threshold)

                            if dab_plus_mask_patch is not None:
                                dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)
                                dab_plus_mask_l2[min_r:max_r, min_c:max_c] = dab_plus_mask_patch
                                logger.info(f"Calculated DAB+ mask L2. Pixels: {np.sum(dab_plus_mask_l2)}")
                                debug_dab_fn = os.path.join(debug_mask_dir, f"{slide_name}_dab_plus_mask_L{hotspot_level}.png")
                                try: cv2.imwrite(debug_dab_fn, dab_plus_mask_l2 * 255)
                                except Exception as e_save: logger.warning(f"Could not save debug DAB mask {debug_dab_fn}: {e_save}")
                            else: logger.error("get_dab_mask returned None.")

                        except openslide.OpenSlideError as ose: logger.error(f"OpenSlideError reading patch for DAB: {ose}", exc_info=True)
                        except Exception as e_read_dab: logger.error(f"Error reading/processing patch for DAB: {e_read_dab}", exc_info=True)
                    else: logger.warning(f"Invalid tumor bbox calculated for DAB mask: W={bbox_w}, H={bbox_h}")
                else: logger.warning("No tumor regions found by regionprops for DAB mask calculation.")
            except Exception as e_dab: logger.error(f"Error during DAB calculation preparation: {e_dab}", exc_info=True)

        if dab_plus_mask_l2 is None:
            logger.warning("DAB+ mask calculation failed or skipped. Using empty mask.")
            dab_plus_mask_l2 = np.zeros((cell_level_h, cell_level_w), dtype=np.uint8)


        # === 7. Initial Hotspot Candidate ID (based on SMP Cell & DAB density) ===
        logger.info(f"--- Stage 5: Initial Hotspot Candidate ID (L{hotspot_level}) ---")
        hotspot_target_mask_coarse = (cell_mask_binary_l2 > 0) & (dab_plus_mask_l2 > 0)
        hotspot_target_mask_coarse = hotspot_target_mask_coarse.astype(np.uint8)
        target_pixels = np.sum(hotspot_target_mask_coarse)
        logger.info(f"Coarse target mask (DAB+ & SMP+) positive pixels: {target_pixels}")
        debug_target_fn = os.path.join(debug_mask_dir, f"{slide_name}_hotspot_target_mask_coarse_L{hotspot_level}.png")
        try: cv2.imwrite(debug_target_fn, hotspot_target_mask_coarse * 255)
        except Exception as e_save: logger.warning(f"Could not save debug target mask: {e_save}")

        candidate_hotspots = []
        if target_pixels == 0:
             logger.warning("Coarse target mask for hotspot candidates is empty. No candidates will be found.")
        else:
             logger.info(f"Identifying initial hotspot candidates based on DAB+ & SMP+ density...")
             num_initial_candidates = max(hotspot_top_n * 4, 15)
             candidate_hotspots = identify_hotspots(
                 slide=slide, level=hotspot_level,
                 hotspot_target_mask=hotspot_target_mask_coarse,
                 hotspot_patch_size_l0=hotspot_patch_size_l0,
                 top_n_hotspots=num_initial_candidates,
                 debug_dir=hotspot_debug_dir
             )
             logger.info(f"Found {len(candidate_hotspots)} initial candidates.")


        # === 8. StarDist Refinement and Re-ranking ===
        hotspots = [] # Final ranked hotspots
        if not candidate_hotspots:
            logger.warning("No initial candidates found for StarDist refinement.")
        else:
            logger.info(f"--- Stage 6: StarDist Refinement & Re-ranking (L{hotspot_level}) ---")
            refined_hotspots_results = []

            refinement_debug_base_dir = os.path.join(hotspot_debug_dir, "refinement_patches")
            os.makedirs(refinement_debug_base_dir, exist_ok=True)

            # --- Determine Actual Pixel Size (MPP) ---
            mpp_x_str = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            mpp_y_str = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
            actual_pixel_size_um = None
            default_mpp = 0.25 # Example fallback

            if mpp_x_str and mpp_y_str:
                try:
                    mpp_x = float(mpp_x_str); mpp_y = float(mpp_y_str)
                    if 0.1 < mpp_x < 1.0 and 0.1 < mpp_y < 1.0:
                         actual_pixel_size_um = (mpp_x + mpp_y) / 2.0
                         logger.info(f"Using MPP from slide properties: X={mpp_x:.4f}, Y={mpp_y:.4f}. Average used: {actual_pixel_size_um:.4f} um/pixel")
                    else: logger.warning(f"MPP values ({mpp_x}, {mpp_y}) seem outside expected range.")
                except ValueError: logger.warning(f"Could not convert MPP properties to float.")

            if actual_pixel_size_um is None:
                actual_pixel_size_um = default_mpp
                logger.warning(f"Could not determine reliable MPP. Falling back to default: {actual_pixel_size_um:.4f} um/pixel. ACCURACY MAY BE AFFECTED.")

            # --- Refine each candidate ---
            for i, candidate in enumerate(tqdm(candidate_hotspots, desc="Refining Hotspots")):
                updated_hotspot = refine_hotspot_with_stardist(
                    candidate_hotspot=candidate,
                    slide=slide,
                    dab_plus_mask_l2=dab_plus_mask_l2,
                    cell_mask_binary_l2=cell_mask_binary_l2,
                    hotspot_level=hotspot_level,
                    actual_pixel_size_um=actual_pixel_size_um,
                    debug_dir=refinement_debug_base_dir,
                    candidate_index=i
                )
                if updated_hotspot is not None:
                    if 'stardist_dab_smp_cell_count' in updated_hotspot:
                         refined_hotspots_results.append(updated_hotspot)
                    else: logger.warning(f"Refinement for candidate {i} missing count results. Skipping.")
                else: logger.warning(f"Refinement failed for candidate index {i}. Skipping.")

            # --- Re-ranking based on refined counts ---
            if refined_hotspots_results:
                refined_hotspots_results.sort(key=lambda item: item.get('stardist_dab_smp_cell_count', 0), reverse=True)
                hotspots = refined_hotspots_results[:hotspot_top_n]
                logger.info(f"Re-ranked hotspots based on StarDist count (DAB+ & SMP+). Top {len(hotspots)}:")
                for rank, hs in enumerate(hotspots):
                    logger.info(f"  Rank {rank+1}: Count={hs.get('stardist_dab_smp_cell_count','N/A')}, "
                                f"L0 Coords={hs.get('coords_l0')}, "
                                f"L{hs.get('level','?')} Coords={hs.get('coords_level')}, "
                                f"Initial Density={hs.get('density_score', 'N/A'):.4f}, "
                                f"SMP Cells in Area={hs.get('stardist_smp_cell_count', 'N/A')}")
                    hs['final_score'] = hs.get('stardist_dab_smp_cell_count', 0)
            else: logger.warning("No hotspots remained after refinement process.")


        # === 9. Generate Final Overlay (Level 2) ===
        logger.info(f"--- Stage 7: Overlay Generation (L{overlay_level}) ---")
        overlay_h, overlay_w = ds4_img.shape[:2]
        target_overlay_shape_wh = (overlay_w, overlay_h)

        tissue_mask_overlay = cv2.resize(tissue_mask_l5, target_overlay_shape_wh, interpolation=cv2.INTER_NEAREST)
        tumor_mask_overlay = tumor_mask_l2 # Already L2
        cell_mask_for_overlay = cell_mask_binary_l2 # Already L2
        dab_mask_for_overlay = dab_plus_mask_l2 # Already L2

        final_overlay = generate_overlay(
            slide=slide, overlay_level=overlay_level, hotspot_level=hotspot_level,
            tissue_mask_overlay=tissue_mask_overlay, tumor_mask_overlay=tumor_mask_overlay,
            cell_mask_binary_l2=cell_mask_for_overlay, hotspots=hotspots,
            dab_mask_l2=dab_mask_for_overlay, debug_dir=overlay_debug_dir
        )

        # --- Save Final Overlay ---
        if final_overlay is not None:
            overlay_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}.jpg")
            png_filename = os.path.join(image_output_dir, f"{slide_name}_final_overlay_L{overlay_level}_temp.png")
            try:
                pil_image = Image.fromarray(final_overlay)
                pil_image.save(png_filename)
                with Image.open(png_filename) as img_png:
                     img_png.convert('RGB').save(overlay_filename, quality=95)
                logger.info(f"Saved final overlay to {overlay_filename}")
                try: os.remove(png_filename)
                except OSError: pass
            except Exception as e:
                logger.error(f"Error saving overlay using PIL: {e}. Trying fallback with OpenCV...")
                try:
                    cv2.imwrite(overlay_filename, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    logger.info(f"Saved overlay via cv2 fallback to {overlay_filename}.")
                except Exception as e2: logger.error(f"Fallback overlay save failed: {e2}")
        else: logger.error("Failed to generate final overlay image.")

        logger.info(f"--- Processing Finished for slide: {slide_name} ---")
        return hotspots

    # --- Error Handling ---
    except openslide.OpenSlideError as e: logger.error(f"OpenSlide error: {e}", exc_info=True); return None
    except FileNotFoundError as e: logger.error(f"File not found: {e}", exc_info=True); return None
    except MemoryError as e: logger.error(f"MemoryError: {e}", exc_info=True); raise e; return None # Re-raise might be better
    except Exception as e: logger.error(f"Unexpected error in pipeline: {e}", exc_info=True); return None
    finally:
        # --- Cleanup ---
        if slide is not None:
            try: slide.close(); logger.debug("OpenSlide object closed.")
            except Exception as e_close: logger.warning(f"Error closing slide object: {e_close}")