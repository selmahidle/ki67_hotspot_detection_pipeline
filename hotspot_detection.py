import logging
import os
import numpy as np
import cv2 
from tqdm import tqdm
import openslide

logger = logging.getLogger(__name__)


def identify_hotspots(slide, level, hotspot_target_mask,
                      hotspot_patch_size_l0=2048, top_n_hotspots=7, debug_dir=None):
    """
    Identifies hotspots based on the density of positive pixels within a
    pre-calculated binary target mask (e.g., DAB+ cells within tumor).

    Uses an efficient sliding window approach directly on the mask.

    Args:
        slide (openslide.OpenSlide): WSI object, used primarily to get
                                     level downsamples for coordinate mapping.
                                     Consider passing downsample factor directly
                                     to reduce dependency.
        level (int): The WSI level at which `hotspot_target_mask` exists and
                     analysis is performed.
        hotspot_target_mask (np.ndarray): Binary mask (uint8, values 0 or 1)
                                          where 1 indicates pixels of interest
                                          (e.g., positive cells) for density calculation.
                                          Must be at the specified `level`.
        hotspot_patch_size_l0 (int): Desired hotspot field-of-view size at Level 0.
                                     This is converted to the patch size at `level`.
        top_n_hotspots (int): The number of top-ranked hotspots to return.
        debug_dir (str, optional): Path to a directory where debug visualization
                                   images (target mask, top hotspots, heatmap)
                                   should be saved. If None, no debug images are saved.

    Returns:
        list: A list of dictionaries, each describing a hotspot. Returns empty list
              if no hotspots are found or on error. Each dictionary contains:
              - 'coords_l0': (x, y) tuple of top-left corner at Level 0.
              - 'size_l0': (width, height) tuple at Level 0.
              - 'coords_level': (x, y) tuple of top-left corner at analysis `level`.
              - 'size_level': (width, height) tuple at analysis `level`.
              - 'density_score': Float score (mean of target mask in patch, 0-1).
              - 'level': The analysis level (int).
    """
    logger.info(f"Identifying top {top_n_hotspots} hotspots at Level {level} based on target mask density...")

    # --- Input Validation ---
    if hotspot_target_mask is None:
        logger.error("Hotspot target mask is None. Cannot identify hotspots.")
        return []
    if not isinstance(hotspot_target_mask, np.ndarray) or hotspot_target_mask.ndim != 2:
        logger.error(f"Invalid hotspot_target_mask provided. Expected 2D numpy array, got {type(hotspot_target_mask)} with ndim={hotspot_target_mask.ndim}.")
        return []
    if level < 0 or level >= slide.level_count:
         logger.error(f"Invalid level {level} specified for hotspot analysis. Slide levels: {slide.level_count}.")
         return []
    if np.sum(hotspot_target_mask) == 0:
        logger.warning("Hotspot target mask is empty (all zeros). No hotspots will be found.")
        return []

    try:
        # --- Calculate Parameters for Analysis Level ---
        level_downsample = slide.level_downsamples[level]
        level_h, level_w = hotspot_target_mask.shape # Mask dimensions at analysis level

        # Calculate patch size at the analysis `level` based on L0 size
        hotspot_patch_size_level = int(hotspot_patch_size_l0 / level_downsample)
        if hotspot_patch_size_level <= 0:
             logger.error(f"Calculated hotspot patch size at Level {level} is non-positive ({hotspot_patch_size_level}). Check L0 size ({hotspot_patch_size_l0}) and downsample ({level_downsample}).")
             return []

        # Use a stride for the sliding window (e.g., quarter patch size for overlap)
        stride_level = max(1, hotspot_patch_size_level // 4)
        logger.info(f"Analysis Level {level} (Downsample: {level_downsample:.2f}): Mask Size {level_w}x{level_h}, Patch Size {hotspot_patch_size_level}x{hotspot_patch_size_level}, Stride {stride_level}")

        # --- Sliding Window Scan ---
        hotspot_candidates = []
        max_density = 0.0 # Track max density found

        # Iterate using stride, ensuring patch doesn't go out of bounds
        for y in tqdm(range(0, level_h - stride_level + 1, stride_level), desc=f"Scanning L{level} for Hotspots"):
            for x in range(0, level_w - stride_level + 1, stride_level):
                # Define patch boundaries at analysis level
                y_start = y
                x_start = x
                # Ensure patch end doesn't exceed mask dimensions
                y_end = min(y_start + hotspot_patch_size_level, level_h)
                x_end = min(x_start + hotspot_patch_size_level, level_w)
                patch_h = y_end - y_start
                patch_w = x_end - x_start

                # Skip if patch dimensions are zero (can happen with stride logic at edge)
                if patch_h <= 0 or patch_w <= 0:
                    continue

                # Extract the patch from the target mask
                patch_target_mask = hotspot_target_mask[y_start:y_end, x_start:x_end]

                # Calculate density score = mean value of the binary mask within the patch
                # This represents the fraction of positive pixels in the patch.
                density_score = np.mean(patch_target_mask) if patch_target_mask.size > 0 else 0.0

                # Store candidate if density is positive
                if density_score > 1e-6: # Use small epsilon to avoid floating point issues
                    hotspot_candidates.append({
                        'coords_level': (x_start, y_start), # Store top-left corner (x, y)
                        'size_level': (patch_w, patch_h),   # Store actual patch size (w, h)
                        'density_score': density_score,
                        'level': level
                    })
                    if density_score > max_density:
                        max_density = density_score

        if not hotspot_candidates:
            logger.warning("No hotspot candidates found with density > 0.")
            return []

        logger.info(f"Found {len(hotspot_candidates)} candidate regions. Max density score: {max_density:.4f}")

        # --- Rank and Select Top Hotspots ---
        # Sort candidates by density score in descending order
        hotspot_candidates.sort(key=lambda item: item['density_score'], reverse=True)

        # Select the top N non-overlapping (or minimally overlapping) hotspots?
        # Current implementation takes the absolute top N based on score, regardless of overlap.
        # Consider adding Non-Maximum Suppression (NMS) here if overlap is an issue.
        top_hotspots_raw = hotspot_candidates[:top_n_hotspots]

        # --- Calculate Level 0 Coordinates ---
        final_hotspots = []
        for hotspot in top_hotspots_raw:
             x_level, y_level = hotspot['coords_level']
             w_level, h_level = hotspot['size_level']
             # Calculate corresponding L0 coordinates and size
             hotspot['coords_l0'] = (int(x_level * level_downsample), int(y_level * level_downsample))
             # L0 size should ideally correspond to the target L0 patch size,
             # but edge patches might be smaller. We store the actual L0 size mapped from level size.
             hotspot['size_l0'] = (int(w_level * level_downsample), int(h_level * level_downsample))
             final_hotspots.append(hotspot)
             logger.debug(f" Selected Hotspot: L{level} ({x_level},{y_level}) {w_level}x{h_level}, "
                          f"L0 ({hotspot['coords_l0'][0]},{hotspot['coords_l0'][1]}) {hotspot['size_l0'][0]}x{hotspot['size_l0'][1]}, "
                          f"Density: {hotspot['density_score']:.4f}")

        # --- Optional Debug Visualization ---
        if debug_dir and final_hotspots:
            os.makedirs(debug_dir, exist_ok=True) # Ensure directory exists
            logger.info(f"Saving hotspot debug visualizations to: {debug_dir}")

            # 1. Visualize Target Mask (as grayscale)
            target_mask_vis_path = os.path.join(debug_dir, f"hotspot_target_mask_l{level}.png")
            try:
                 # Save mask directly (assuming values are 0 or 1, multiply by 255 for visibility)
                 cv2.imwrite(target_mask_vis_path, hotspot_target_mask * 255)
            except Exception as e_vis:
                 logger.error(f"Failed to save target mask visualization: {e_vis}")


            # 2. Visualize Top N Hotspots overlaid on the target mask
            hotspot_top_vis = cv2.cvtColor(hotspot_target_mask * 150, cv2.COLOR_GRAY2BGR) # Dim background
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)] # BGR
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, hotspot in enumerate(final_hotspots):
                x, y = hotspot['coords_level']
                w, h = hotspot['size_level']
                color = colors[i % len(colors)]
                # Draw rectangle boundary
                cv2.rectangle(hotspot_top_vis, (x, y), (x+w, y+h), color, max(1, int(level_w * 0.0015))) # Thickness relative to image size
                # Add text label (Hotspot number and score)
                score_text = f"#{i+1}: {hotspot['density_score']:.3f}"
                cv2.putText(hotspot_top_vis, score_text, (x + 5, y + 20), font, 0.6, color, 2)
            top_vis_path = os.path.join(debug_dir, f"top_{top_n_hotspots}_hotspots_l{level}.png")
            try:
                cv2.imwrite(top_vis_path, hotspot_top_vis)
            except Exception as e_vis:
                 logger.error(f"Failed to save top hotspots visualization: {e_vis}")


            # 3. Visualize Density Heatmap (optional, can be slow for large images)
            logger.info("Generating density heatmap (this might take a moment)...")
            try:
                density_heatmap = np.zeros((level_h, level_w), dtype=np.float32)
                # Fill heatmap based on candidate scores (more accurate than just drawing rectangles)
                # This can be done more efficiently using integral images or convolution if needed.
                # Simple accumulation for now:
                weight_map = np.zeros((level_h, level_w), dtype=np.float32)
                epsilon = 1e-6
                for candidate in hotspot_candidates:
                    x, y = candidate['coords_level']
                    w, h = candidate['size_level']
                    score = candidate['density_score']
                    density_heatmap[y:y+h, x:x+w] += score
                    weight_map[y:y+h, x:x+w] += 1.0
                # Normalize heatmap where weights are non-zero
                valid_weights = weight_map > epsilon
                density_heatmap[valid_weights] /= weight_map[valid_weights]

                # Normalize to 0-1 for colormap
                if max_density > 0:
                     density_heatmap /= max_density # Normalize by max observed density
                density_heatmap = np.clip(density_heatmap, 0, 1)

                heatmap_vis = cv2.applyColorMap(np.uint8(density_heatmap * 255), cv2.COLORMAP_JET)
                heatmap_path = os.path.join(debug_dir, f"hotspot_density_heatmap_l{level}.png")
                cv2.imwrite(heatmap_path, heatmap_vis)
                logger.info("Density heatmap saved.")
            except Exception as e_heatmap:
                 logger.error(f"Failed to generate or save density heatmap: {e_heatmap}")


        logger.info(f"Successfully identified {len(final_hotspots)} top hotspots.")
        return final_hotspots

    except Exception as e:
        logger.error(f"Error during hotspot identification: {e}")
        logger.error(traceback.format_exc())
        return []