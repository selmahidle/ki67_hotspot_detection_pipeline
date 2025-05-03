import os
import logging
import glob
import argparse
from pathlib import Path
import torch
from PIL import Image
from model_utils import create_model, load_latest_checkpoint, load_models_from_subdirs
from pipeline import process_slide_ki67 
from datetime import datetime



Image.MAX_IMAGE_PIXELS = None


def write_hotspot_results(hotspot_results, output_dir, slide_path, hotspot_level=2):
    logger = logging.getLogger(__name__)

    results_file = os.path.join(output_dir, Path(slide_path).stem + "_hotspots_stardist.txt")
    with open(results_file, 'w') as f:
        f.write(f"Slide: {Path(slide_path).name}\n")
        f.write(f"Identified {len(hotspot_results)} final hotspots (ranked by StarDist counts).\n")
        f.write("-" * 30 + "\n")

        for i, hs in enumerate(hotspot_results):
            f.write(f"Hotspot {i+1}:\n")
            f.write(f"  Level 0 Coords (x,y): {hs.get('coords_l0', 'N/A')}\n")
            f.write(f"  Level 0 Size (w,h): {hs.get('size_l0', 'N/A')}\n")
            f.write(f"  Level {hs.get('level', hotspot_level)} Coords (x,y): {hs.get('coords_level', 'N/A')}\n")
            f.write(f"  Level {hs.get('level', hotspot_level)} Size (w,h): {hs.get('size_level', 'N/A')}\n")

            stardist_dab_smp_count = hs.get('stardist_dab_smp_cell_count', 0)
            stardist_smp_count = hs.get('stardist_smp_cell_count', 0)
            stardist_total = hs.get('stardist_total_count', 0)

            try:
                ki67_index = (float(stardist_dab_smp_count) / float(stardist_smp_count)) * 100.0 if stardist_smp_count > 0 else 0.0
            except (ValueError, TypeError):
                ki67_index = 'N/A'

            f.write(f"  Score (StarDist DAB+ & SMP+ Count): {stardist_dab_smp_count}\n")
            f.write(f"  (StarDist Nuclei within SMP Mask): {stardist_smp_count}\n")
            f.write(f"  (StarDist Total Predicted in Patch): {stardist_total}\n")

            if isinstance(ki67_index, float):
                f.write(f"  Ki67 Index (of cells in SMP Mask %): {ki67_index:.2f}%\n")
            else:
                f.write(f"  Ki67 Index (of cells in SMP Mask %): {ki67_index}\n")

            initial_density = hs.get('density_score', 'N/A')
            if initial_density != 'N/A' and isinstance(initial_density, (float, int)):
                f.write(f"  (Initial Candidate Density Score): {initial_density:.4f}\n")

            f.write("-" * 10 + "\n")

    logger.info(f"Hotspot details saved to: {results_file}")



if __name__ == "__main__":

    default_output_base = "/cluster/home/selmahi/outputs_from_fixing_stardist"
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    default_output_dir = os.path.join(default_output_base, timestamp)

    parser = argparse.ArgumentParser(description="Run Ki67 Hotspot Analysis on a WSI")
    parser.add_argument("--slide_path", type=str,
                        default="/cluster/home/selmahi/mib_pipeline/datasets/ndpi_tumor_segmentation/ndpi_files/wz8shzxklk6w.ndpi",
                        help="Path to the input WSI file.")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
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

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "pipeline.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Models ---
    logger.info("Loading Models...")
    tumor_models = load_models_from_subdirs(args.tumor_model_base_dir, args.model_type, args.encoder, device, apply_groupnorm=True)
    cell_models = load_models_from_subdirs(args.cell_model_base_dir, args.model_type, args.encoder, device, apply_groupnorm=False)

    if not tumor_models:
        logger.error("No tumor models loaded. Exiting.")
        exit()

    if not cell_models:
        logger.error("No cell models loaded. Exiting.")
        exit()

    logger.info(f"Loaded {len(tumor_models)} tumor models and {len(cell_models)} cell models.")

    # --- Process Slide ---
    logger.info(f"Starting Ki67 hotspot analysis for: {args.slide_path}")
    hotspot_results = process_slide_ki67(
        slide_path=args.slide_path,
        output_dir=args.output_dir,
        tumor_models=tumor_models,
        cell_models=cell_models,
        device=device
    )

    if hotspot_results is not None:
        logger.info("Analysis complete.")
        write_hotspot_results(hotspot_results, args.output_dir, args.slide_path)
    else:
        logger.error(f"Analysis failed for slide: {args.slide_path}")


    logger.info("Script finished.")