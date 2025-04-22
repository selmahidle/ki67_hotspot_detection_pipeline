import os
import logging
import glob
import argparse
from pathlib import Path
import torch
from PIL import Image

from model_utils import create_model, load_latest_checkpoint
from utils import convert_batchnorm_to_groupnorm 
from pipeline import process_slide_ki67 
from stardist_utils import load_stardist_model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ki67 Hotspot Analysis on a WSI")
    parser.add_argument("--slide_path", type=str,
                        default="/cluster/home/selmahi/mib_pipeline/datasets/ndpi_tumor_segmentation/ndpi_files/wz8shzxklk6w.ndpi",
                        help="Path to the input WSI file.")
    parser.add_argument("--output_dir", type=str, default="/cluster/home/selmahi/mib_pipeline/ki67_output_stardist_test_DEBUG",
                        help="Directory to save output images and results.")
    parser.add_argument("--tumor_model_base_dir", type=str,
                        default="/cluster/home/selmahi/mib_pipeline/mib_pipeline_scripts/checkpoints/tumor_segmentation_checkpoints",
                        help="Base directory containing subfolders for each tumor segmentation model checkpoint.")
    parser.add_argument("--cell_model_base_dir", type=str,
                        default="/cluster/home/selmahi/mib_pipeline/mib_pipeline_scripts/checkpoints/cell_segmentation_checkpoints",
                        help="Base directory containing subfolders for each cell segmentation model checkpoint.")
    parser.add_argument("--model_type", type=str, default="DeepLabV3Plus",
                        help="Segmentation model type (e.g., DeepLabV3Plus, Unet).")
    parser.add_argument("--encoder", type=str, default="resnet18",
                        help="Encoder backbone for the segmentation models.")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Models ---
    # 1. Tumor Models (Load latest from each subfolder in the base directory)
    logger.info("Loading Tumor Models...")
    tumor_models = []
    tumor_subdirs = [d for d in glob.glob(os.path.join(args.tumor_model_base_dir, '*')) if os.path.isdir(d)]
    if not tumor_subdirs:
         logger.error(f"No subdirectories found in tumor model base directory: {args.tumor_model_base_dir}")
         exit()
    for i, subdir in enumerate(tumor_subdirs):
        logger.info(f"Loading tumor model from: {subdir}")
        # Create model with sigmoid activation
        model = create_model(args.model_type, args.encoder, None, num_classes=2, activation='sigmoid')

        # !!! RESTORED GroupNorm CONVERSION FOR TUMOR MODELS !!!
        # This is needed because checkpoints likely not saved with GroupNorm layers.
        logger.info("Converting tumor model to GroupNorm before loading checkpoint...")
        model = convert_batchnorm_to_groupnorm(model)

        # Load checkpoint onto the correct device
        model.to(device) # Move model structure to device first
        model = load_latest_checkpoint(model, subdir, device) # load_latest handles map_location

        if model is not None:
            # Ensure model is definitely on device after loading
            model.to(device)
            tumor_models.append(model)
        else:
             logger.warning(f"Could not load tumor model from {subdir}. Skipping.")

    if not tumor_models:
        logger.error("Failed to load any tumor models. Exiting.")
        exit()
    logger.info(f"Loaded {len(tumor_models)} tumor models.")

    # 2. Cell Models (Load latest from each subfolder in the base directory)
    logger.info("Loading Cell Models...")
    cell_models = []
    cell_subdirs = [d for d in glob.glob(os.path.join(args.cell_model_base_dir, '*')) if os.path.isdir(d)]
    if not cell_subdirs:
         logger.error(f"No subdirectories found in cell model base directory: {args.cell_model_base_dir}")
         exit()
    for i, subdir in enumerate(cell_subdirs):
         logger.info(f"Loading cell model from: {subdir}")
         # Reverted: num_classes=2
         model = create_model(args.model_type, args.encoder, None, num_classes=2, activation='sigmoid') # Assuming semantic output
         # Not converting cell models to GroupNorm as per user instruction
         model = load_latest_checkpoint(model, subdir, device)
         if model is not None:
             model.to(device)
             cell_models.append(model)
         else:
             logger.warning(f"Could not load cell model from {subdir}. Skipping.")
    if not cell_models:
        logger.error("Failed to load any cell models. Exiting.")
        exit()
    logger.info(f"Loaded {len(cell_models)} cell models.")

    # Load StarDist model
    logger.info("Loading StarDist Model...")
    stardist_model = load_stardist_model() 
    if stardist_model is None:
        logger.error(f"Failed to load StarDist model. Exiting.")
        exit()


    # --- Process Slide ---
    logger.info(f"Starting Ki67 hotspot analysis for: {args.slide_path}")
    hotspot_results = process_slide_ki67(
        slide_path=args.slide_path,
        output_dir=args.output_dir,
        tumor_models=tumor_models,
        cell_models=cell_models,
        stardist_model=stardist_model,
        device=device
    )

# main.py (Results Handling with NEW Proliferation Index)

    if hotspot_results is not None:
        logger.info("Analysis complete. Hotspot results (StarDist based):")
        hotspot_level = 2
        results_file = os.path.join(args.output_dir, Path(args.slide_path).stem + "_hotspots_stardist.txt")
        with open(results_file, 'w') as f:
            f.write(f"Slide: {Path(args.slide_path).name}\n")
            f.write(f"Identified {len(hotspot_results)} final hotspots (ranked by StarDist counts).\n")
            f.write("-" * 30 + "\n")
            for i, hs in enumerate(hotspot_results):
                f.write(f"Hotspot {i+1}:\n")
                # ... (Write coordinates and sizes) ...
                f.write(f"  Level 0 Coords (x,y): {hs.get('coords_l0', 'N/A')}\n")
                f.write(f"  Level 0 Size (w,h): {hs.get('size_l0', 'N/A')}\n")
                f.write(f"  Level {hs.get('level', 'N/A')} Coords (x,y): {hs.get('coords_level', 'N/A')}\n")
                f.write(f"  Level {hs.get('level', 'N/A')} Size (w,h): {hs.get('size_level', 'N/A')}\n")

                # --- Get Counts ---
                stardist_dab_smp_count = hs.get('stardist_dab_smp_cell_count', 0) # Numerator
                stardist_smp_count = hs.get('stardist_smp_cell_count', 0) # <-- NEW Denominator
                stardist_total = hs.get('stardist_total_count', 0) # Total predicted

                # --- Calculate Proliferation Index using NEW denominator ---
                ki67_index = 0.0
                if stardist_smp_count > 0: # Use the count within SMP cells as denominator
                    try:
                        ki67_index = (float(stardist_dab_smp_count) / float(stardist_smp_count)) * 100.0
                    except (ValueError, TypeError): ki67_index = 'N/A'
                else: ki67_index = 0.0 # Or 'N/A' if no cells counted in SMP area

                # --- Write Results ---
                f.write(f"  Score (StarDist DAB+ & SMP+ Count): {stardist_dab_smp_count}\n")
                f.write(f"  (StarDist Nuclei within SMP Mask): {stardist_smp_count}\n") # Write new denominator
                f.write(f"  (StarDist Total Predicted in Patch): {stardist_total}\n")
                # Format Ki67 index nicely
                if isinstance(ki67_index, float): f.write(f"  Ki67 Index (of cells in SMP Mask %): {ki67_index:.2f}%\n") # Clarify denominator
                else: f.write(f"  Ki67 Index (of cells in SMP Mask %): {ki67_index}\n")
                # --------------------------------------

                # --- Optionally keep the initial density score ---
                initial_density = hs.get('density_score', 'N/A')
                if initial_density != 'N/A' and isinstance(initial_density, (float, int)): f.write(f"  (Initial Candidate Density Score): {initial_density:.4f}\n")
                # ----------------------------------------------------------

                f.write("-" * 10 + "\n")
        logger.info(f"Hotspot details saved to: {results_file}")
    else:
        logger.error(f"Analysis failed for slide: {args.slide_path}")

    logger.info("Script finished.")
    