import os
import logging
import glob
import argparse
from pathlib import Path
import torch
from PIL import Image
from model_utils import (
    create_model,
    create_and_load_attention_unet,
    load_model_checkpoint,
    load_models_from_subdirs
)
from pipeline import process_slide_ki67
from datetime import datetime
from stardist.models import StarDist2D


Image.MAX_IMAGE_PIXELS = None


def write_hotspot_results(hotspot_results, output_dir, slide_path, hotspot_level=2):
    """
    Writes the final hotspot analysis results to a text file.
    The results are presented in the same order as they were selected by the pipeline,
    which is ranked by the Proliferation Index.
    """
    logger = logging.getLogger(__name__)

    if not hotspot_results:
        logger.warning("No hotspot results provided to write_hotspot_results. Skipping file creation.")
        return

    results_file = os.path.join(output_dir, Path(slide_path).stem + "_hotspots_final_results.txt")
    try:
        with open(results_file, 'w') as f:
            f.write(f"Slide: {Path(slide_path).name}\n")
            valid_results = [hs for hs in hotspot_results if isinstance(hs, dict)]
            
            f.write(f"Final {len(valid_results)} hotspots (ranked by Proliferation Index).\n")
            f.write("-" * 30 + "\n")

            for i, hs in enumerate(valid_results):
                f.write(f"Hotspot Rank {i+1}:\n")
                proliferation_index_fraction = hs.get('stardist_proliferation_index', 0.0)
                ki67_pos_count = hs.get('stardist_ki67_pos_count', 0)
                total_filtered_count = hs.get('stardist_total_count_filtered', 0)
                initial_density = hs.get('density_score', 'N/A')

                try:
                    proliferation_index_percent = float(proliferation_index_fraction) * 100.0
                    f.write(f"  Ki67 Proliferation Index (%): {proliferation_index_percent:.2f}%\n")
                except (ValueError, TypeError):
                     f.write(f"  Ki67 Proliferation Index (%): N/A\n")

                f.write(f"  Ki67-Positive Nuclei: {ki67_pos_count}\n")
                f.write(f"  Total Tumor Nuclei: {total_filtered_count}\n")
                f.write(f"  Level {hs.get('level', hotspot_level)} Coords (x,y): {hs.get('coords_level', 'N/A')}\n")
                f.write(f"  Level {hs.get('level', hotspot_level)} Size (w,h): {hs.get('size_level', 'N/A')}\n")
                f.write(f"  Level 0 Coords (x,y): {hs.get('coords_l0', 'N/A')}\n")
                f.write(f"  Level 0 Size (w,h): {hs.get('size_l0', 'N/A')}\n")

                if initial_density != 'N/A' and isinstance(initial_density, (float, int)):
                    f.write(f"  (Initial Candidate Density Score): {initial_density:.4f}\n")

                f.write("-" * 10 + "\n")

        logger.info(f"Final hotspot results saved to: {results_file}")

    except IOError as e:
        logger.error(f"Error writing hotspot results file {results_file}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in write_hotspot_results: {e}", exc_info=True)



if __name__ == "__main__":

    default_output_base = "/cluster/home/selmahi/pipeline_outputs"
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    default_output_dir = os.path.join(default_output_base, timestamp)

    parser = argparse.ArgumentParser(description="Run Ki67 Hotspot Analysis on a WSI")
    parser.add_argument("--slide_path", type=str, required=True, help="Path to the input WSI file.")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save output images and results.")
    
    # --- Updated Arguments for Optional Workflows ---
    parser.add_argument("--tumor_model_path", type=str, default=None, help="Path to a single trained tumor segmentation model (.pth file).")
    parser.add_argument("--tumor_model_base_dir", type=str, default=None, help="Base directory of tumor models for consensus mode. Each subfolder should contain one model.")
    
    parser.add_argument("--tumor_model_type", type=str, default="DeepLabV3Plus", help="Tumor segmentation model type (e.g., DeepLabV3Plus, Unet).")
    parser.add_argument("--tumor_encoder", type=str, default="resnet18", help="Encoder backbone for the tumor segmentation model.")
    parser.add_argument("--attention_unet_path", type=str, required=True, help="Path to the trained AttentionUNet .pth file for cell segmentation.")

    args = parser.parse_args()
    
    # --- Argument Validation ---
    if not (args.tumor_model_path or args.tumor_model_base_dir) or \
       (args.tumor_model_path and args.tumor_model_base_dir):
        parser.error("Please provide exactly one of --tumor_model_path (for single model) or --tumor_model_base_dir (for consensus mode).")

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

    logger.info("--- Loading Tumor Model(s) ---")
    tumor_models = []

    # --- Logic to load single model or multiple models for consensus ---
    if args.tumor_model_path:
        logger.info(f"--- Single Tumor Model Workflow ---")
        logger.info(f"Creating tumor model ({args.tumor_model_type} with {args.tumor_encoder} encoder)...")
        # For single model workflow, num_classes is 1 (binary mask output) and activation is None for BCEWithLogitsLoss
        tumor_model = create_model(
            model_type=args.tumor_model_type,
            encoder=args.tumor_encoder,
            encoder_weights=None, 
            num_classes=1,        
            activation=None     
        )
        if tumor_model is None:
            logger.error("Failed to create tumor model structure. Exiting.")
            exit()

        logger.info(f"Loading tumor model checkpoint from: {args.tumor_model_path}")
        try:
            tumor_model = load_model_checkpoint(tumor_model, args.tumor_model_path, device, weights_only=True)
            tumor_model.eval()
            tumor_model.to(device)
            tumor_models.append(tumor_model)
            logger.info("Single tumor model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load single tumor model checkpoint: {e}", exc_info=True)
            exit()

    elif args.tumor_model_base_dir:
        logger.info(f"--- Consensus Tumor Model Workflow ---")
        # Consensus workflow uses models with GroupNorm, num_classes=2 and sigmoid activation
        tumor_models = load_models_from_subdirs(
            base_dir=args.tumor_model_base_dir,
            model_type=args.tumor_model_type,
            encoder=args.tumor_encoder,
            device=device,
            apply_groupnorm=True
        )
        if not tumor_models:
            logger.error(f"Failed to load any tumor models from base directory: {args.tumor_model_base_dir}. Exiting.")
            exit()
        logger.info(f"Successfully loaded {len(tumor_models)} tumor models for consensus.")

    # --- Load Cell Model (No changes needed here) ---
    logger.info(f"Loading AttentionUNet cell model using checkpoint: {args.attention_unet_path}")
    cell_model_attention_unet = create_and_load_attention_unet(
        checkpoint_path=args.attention_unet_path,
        device=device
    )

    if cell_model_attention_unet is None:
        logger.error("AttentionUNet cell model could not be loaded. Exiting.")
        exit()

    logger.info(f"Loaded {len(tumor_models)} tumor model(s) and 1 AttentionUNet cell model.")

    stardist_model_instance = StarDist2D.from_pretrained('2D_versatile_fluo')

    # --- Call Pipeline ---
    logger.info(f"Starting Ki67 hotspot analysis for: {args.slide_path}")
    hotspot_results = process_slide_ki67(
        slide_path=args.slide_path,
        output_dir=args.output_dir,
        tumor_models=tumor_models, # Pass the list of models
        cell_model=cell_model_attention_unet,
        stardist_model=stardist_model_instance,
        device=device
    )

    if hotspot_results is not None:
        logger.info("Analysis complete.")
        write_hotspot_results(hotspot_results, args.output_dir, args.slide_path)
    else:
        logger.error(f"Analysis failed for slide: {args.slide_path}")

    logger.info("Script finished.")