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
from stardist.models import StarDist2D


Image.MAX_IMAGE_PIXELS = None


def write_hotspot_results(hotspot_results, output_dir, slide_path, hotspot_level=2):
    """
    Writes the hotspot analysis results to a text file, using counts derived
    from local DAB classification per nucleus.
    """
    logger = logging.getLogger(__name__) # Get logger inside function if not passed

    # Check if hotspot_results is empty or None
    if not hotspot_results:
        logger.warning("No hotspot results provided to write_hotspot_results. Skipping file creation.")
        return

    results_file = os.path.join(output_dir, Path(slide_path).stem + "_hotspots_stardist_localDAB.txt") # Added suffix
    try:
        with open(results_file, 'w') as f:
            f.write(f"Slide: {Path(slide_path).name}\n")
            # Filter results to only include those that potentially have the new keys
            valid_results = [hs for hs in hotspot_results if isinstance(hs, dict)]
            f.write(f"Processed {len(valid_results)} candidate hotspots (ranked by local DAB Ki67+ count).\n")
            f.write("-" * 30 + "\n")

            # Sort results by the new Ki67+ count before writing (optional, but good practice)
            # Use .get with a default of 0 to handle cases where a candidate might have failed refinement
            valid_results.sort(key=lambda hs: hs.get('stardist_ki67_pos_count', 0), reverse=True)

            for i, hs in enumerate(valid_results):
                f.write(f"Hotspot {i+1}:\n")
                f.write(f"  Level 0 Coords (x,y): {hs.get('coords_l0', 'N/A')}\n")
                f.write(f"  Level 0 Size (w,h): {hs.get('size_l0', 'N/A')}\n")
                f.write(f"  Level {hs.get('level', hotspot_level)} Coords (x,y): {hs.get('coords_level', 'N/A')}\n")
                f.write(f"  Level {hs.get('level', hotspot_level)} Size (w,h): {hs.get('size_level', 'N/A')}\n")

                # --- Fetch results using the NEW keys ---
                ki67_pos_count = hs.get('stardist_ki67_pos_count', 0)
                total_filtered_count = hs.get('stardist_total_count_filtered', 0)
                # Get the pre-calculated index (assuming it's stored 0-1)
                proliferation_index_fraction = hs.get('stardist_proliferation_index', 0.0)

                # --- Write results using the NEW keys ---
                f.write(f"  Score (Local DAB Ki67+ Count): {ki67_pos_count}\n")
                f.write(f"  Total Filtered Nuclei: {total_filtered_count}\n")

                # Display the proliferation index as a percentage
                try:
                    # Ensure the index is float before formatting
                    proliferation_index_percent = float(proliferation_index_fraction) * 100.0
                    f.write(f"  Ki67 Proliferation Index (%): {proliferation_index_percent:.2f}%\n")
                except (ValueError, TypeError):
                     f.write(f"  Ki67 Proliferation Index (%): N/A\n")


                # Keep optional initial density score if present
                initial_density = hs.get('density_score', 'N/A')
                if initial_density != 'N/A' and isinstance(initial_density, (float, int)):
                    f.write(f"  (Initial Candidate Density Score): {initial_density:.4f}\n")

                f.write("-" * 10 + "\n")

        logger.info(f"Hotspot details (using local DAB counts) saved to: {results_file}")

    except IOError as e:
        logger.error(f"Error writing hotspot results file {results_file}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in write_hotspot_results: {e}", exc_info=True)



if __name__ == "__main__":

    default_output_base = "/cluster/home/selmahi/pipeline_outputs"
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    default_output_dir = os.path.join(default_output_base, timestamp)

    parser = argparse.ArgumentParser(description="Run Ki67 Hotspot Analysis on a WSI")
    parser.add_argument("--slide_path", type=str, help="Path to the input WSI file.")
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

    model = StarDist2D.from_pretrained('2D_versatile_he')

    # --- Process Slide ---
    logger.info(f"Starting Ki67 hotspot analysis for: {args.slide_path}")
    hotspot_results = process_slide_ki67(
        slide_path=args.slide_path,
        output_dir=args.output_dir,
        tumor_models=tumor_models,
        cell_models=cell_models,
        stardist_model=model,
        device=device
    )

    if hotspot_results is not None:
        logger.info("Analysis complete.")
        write_hotspot_results(hotspot_results, args.output_dir, args.slide_path)
    else:
        logger.error(f"Analysis failed for slide: {args.slide_path}")


    logger.info("Script finished.")