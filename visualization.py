import logging
import os
import numpy as np
import cv2
from PIL import Image 
import openslide
import traceback


logger = logging.getLogger(__name__)


def generate_overlay(slide, overlay_level, hotspot_level,
                     tissue_mask_overlay, tumor_mask_overlay, cell_mask_binary_l2,
                     hotspots, dab_mask_l2=None, debug_dir=None):
    """
    Generates an overlay image showing WSI, tissue/tumor outlines, cell regions,
    DAB+ areas (if provided), and identified hotspots.

    Args:
        slide (openslide.OpenSlide): The OpenSlide object for the WSI.
        overlay_level (int): The WSI level to use for the base overlay image.
        hotspot_level (int): The WSI level at which hotspot analysis was performed
                             (used for scaling hotspot coordinates).
        tissue_mask_overlay (np.ndarray): Binary tissue mask (0/1 or 0/255) already
                                          resized to the `overlay_level`.
        tumor_mask_overlay (np.ndarray): Binary tumor mask (0/1 or 0/255) already
                                         resized to the `overlay_level`.
        cell_mask_binary_l2 (np.ndarray or None): Binary cell mask (0/1 or 0/255) at Level 2.
                                          This function will handle resizing it to
                                          `overlay_level` if needed. Can be None.
        hotspots (list): List of hotspot dictionaries as returned by
                         `identify_hotspots`. Should contain coordinates at
                         `hotspot_level`.
        dab_mask_l2 (np.ndarray, optional): Binary DAB+ mask (0/1 or 0/255) at Level 2.
                                            If provided, contours are drawn.
                                            This function will handle resizing it.
                                            If None, DAB contours are skipped.
        debug_dir (str, optional): Directory to save intermediate debug images
                                   (e.g., mask sanity checks, hotspot drawing).
                                   If None, no debug images are saved.

    Returns:
        np.ndarray or None: The overlay image as an RGB NumPy array (H, W, 3),
                           or None if overlay generation fails.
    """
    logger.info(f"Generating overlay for Level {overlay_level}...")

    # --- Input Validation ---
    if not isinstance(slide, openslide.OpenSlide):
         logger.error("Invalid 'slide' object provided for overlay generation.")
         return None
    if overlay_level < 0 or overlay_level >= slide.level_count:
         logger.error(f"Invalid overlay_level {overlay_level}. Slide levels: {slide.level_count}.")
         return None
    if hotspot_level < 0 or hotspot_level >= slide.level_count:
         logger.error(f"Invalid hotspot_level {hotspot_level}. Slide levels: {slide.level_count}.")
         return None

    try:
        # --- Read Base Image for Overlay ---
        overlay_dims = slide.level_dimensions[overlay_level]
        overlay_w, overlay_h = overlay_dims[0], overlay_dims[1]
        logger.info(f"Reading base image for overlay: Level {overlay_level} ({overlay_w}x{overlay_h})")
        try:
            base_image_pil = slide.read_region((0, 0), overlay_level, (overlay_w, overlay_h)).convert('RGB')
            overlay_rgb = np.array(base_image_pil)
        except openslide.OpenSlideError as e:
             logger.error(f"OpenSlide error reading overlay base image L{overlay_level}: {e}")
             return None
        except Exception as e_read:
             logger.error(f"Error reading overlay base image L{overlay_level}: {e_read}", exc_info=True) # Add exc_info
             return None

        # Work in BGR for OpenCV drawing functions
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        logger.debug("Base overlay image read and converted to BGR.")

        target_overlay_shape = (overlay_h, overlay_w) # Define target shape tuple (H, W)

        # --- Prepare Masks for Overlay Level ---
        # Helper function to prepare and validate mask
        def prepare_mask(mask, mask_name, target_shape=None): # Default target_shape to None
            if mask is None:
                # logger.warning(f"{mask_name} mask is None.") # Reduce noise if None is expected
                return None
            if not isinstance(mask, np.ndarray):
                 logger.warning(f"{mask_name} mask is not a numpy array ({type(mask)}), cannot process.")
                 return None

            # Ensure mask is binary (0 or 255) for findContours/drawing
            mask_max = np.max(mask) if mask.size > 0 else 0 # Handle empty mask case
            if mask.dtype != np.uint8 or (mask_max > 1 and mask_max != 255):
                logger.debug(f"Converting {mask_name} mask (dtype: {mask.dtype}, max: {mask_max}) to uint8 0/255.")
                mask_255 = ((mask > 0) * 255).astype(np.uint8)
            else:
                 mask_255 = mask # Assume already uint8 0/255 or 0/1

            if target_shape is not None and mask_255.shape[:2] != target_shape:
                 logger.info(f"Resizing {mask_name} mask from {mask_255.shape[:2]} to {target_shape}...")
                 # Ensure target_shape has valid dimensions before accessing indices
                 if len(target_shape) >= 2 and target_shape[0] > 0 and target_shape[1] > 0:
                     try:
                          # Target size for cv2.resize is (width, height)
                          resized_mask = cv2.resize(mask_255, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                          return resized_mask
                     except Exception as resize_err:
                          logger.error(f"Error resizing {mask_name} mask: {resize_err}", exc_info=True)
                          return None # Return None if resize fails
                 else:
                     logger.error(f"Invalid target_shape {target_shape} provided for resizing {mask_name}.")
                     return None # Return None if target shape invalid
            else:
                 # Return the mask without resizing if target_shape is None or already matches
                 return mask_255


        # Prepare masks, passing the target shape where needed
        tissue_mask_overlay_ready = prepare_mask(tissue_mask_overlay, "Tissue", target_overlay_shape)
        tumor_mask_overlay_ready = prepare_mask(tumor_mask_overlay, "Tumor", target_overlay_shape)
        cell_mask_l2_ready = prepare_mask(cell_mask_binary_l2, "Cell (L2)") # Validate L2 mask (no target shape)
        dab_mask_l2_ready = prepare_mask(dab_mask_l2, "DAB (L2)")         # Validate L2 DAB mask (no target shape)

        # Resize L2 masks to overlay level if they exist and levels differ
        cell_mask_overlay_ready = None
        if cell_mask_l2_ready is not None:
             if cell_mask_l2_ready.shape[:2] != target_overlay_shape:
                 logger.info(f"Resizing validated Cell mask from L2 ({cell_mask_l2_ready.shape[:2]}) to overlay level {target_overlay_shape}...")
                 cell_mask_overlay_ready = cv2.resize(cell_mask_l2_ready, (target_overlay_shape[1], target_overlay_shape[0]), interpolation=cv2.INTER_NEAREST)
             else:
                 cell_mask_overlay_ready = cell_mask_l2_ready # Already at correct level

        dab_mask_overlay_ready = None
        if dab_mask_l2_ready is not None:
             if dab_mask_l2_ready.shape[:2] != target_overlay_shape:
                  logger.info(f"Resizing validated DAB mask from L2 ({dab_mask_l2_ready.shape[:2]}) to overlay level {target_overlay_shape}...")
                  dab_mask_overlay_ready = cv2.resize(dab_mask_l2_ready, (target_overlay_shape[1], target_overlay_shape[0]), interpolation=cv2.INTER_NEAREST)
             else:
                  dab_mask_overlay_ready = dab_mask_l2_ready


        # --- Sanity Check Mask Saving ---
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"Saving mask sanity checks to: {debug_dir}")
            if tissue_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"tissue_mask_overlay_L{overlay_level}.png"), tissue_mask_overlay_ready)
            if tumor_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"tumor_mask_overlay_L{overlay_level}.png"), tumor_mask_overlay_ready)
            # Save the L2 cell mask *before* potential resizing for clarity
            if cell_mask_l2_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"cell_mask_L2_input.png"), cell_mask_l2_ready)
            # Save the overlay-ready cell mask (after potential resizing)
            if cell_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"cell_mask_overlay_L{overlay_level}.png"), cell_mask_overlay_ready)
            if dab_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"dab_mask_overlay_L{overlay_level}.png"), dab_mask_overlay_ready)


        # --- Define Colors and Drawing Parameters ---
        tissue_color = (0, 255, 0); tumor_color = (255, 0, 0)
        cell_region_color = (0, 200, 255); cell_boundary_color = (0, 255, 255)
        dab_contour_color = (0, 0, 255)
        hotspot_colors = [(0,0,255),(0,255,255),(255,0,255),(255,255,0),(0,255,0)]
        contour_thickness_tissue = 8; contour_thickness_tumor = 4
        hotspot_thickness = 20; dab_contour_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 3.0; label_thickness = 3


        # --- Draw Tissue Contours ---
        if tissue_mask_overlay_ready is not None and np.sum(tissue_mask_overlay_ready) > 0:
            logger.debug("Drawing tissue contours...")
            contours, _ = cv2.findContours(tissue_mask_overlay_ready, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, tissue_color, contour_thickness_tissue)

        # --- Draw Tumor Contours ---
        if tumor_mask_overlay_ready is not None and np.sum(tumor_mask_overlay_ready) > 0:
            logger.debug("Drawing tumor contours...")
            contours, _ = cv2.findContours(tumor_mask_overlay_ready, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, tumor_color, contour_thickness_tumor)

        # --- Draw Cell Regions and Boundaries --- # <-- USES cell_mask_overlay_ready
        if cell_mask_overlay_ready is not None and np.sum(cell_mask_overlay_ready) > 0:
            logger.info("Drawing cell regions and boundaries...")
            cell_overlay_fill = overlay_bgr.copy(); cell_overlay_fill[cell_mask_overlay_ready > 0] = cell_region_color
            alpha_cell = 0.3; cv2.addWeighted(cell_overlay_fill, alpha_cell, overlay_bgr, 1 - alpha_cell, 0, overlay_bgr)
            kernel_cell = np.ones((3,3), np.uint8); dilated = cv2.dilate(cell_mask_overlay_ready, kernel_cell, 1); eroded = cv2.erode(cell_mask_overlay_ready, kernel_cell, 1)
            boundary = dilated - eroded; boundary_thinned = np.zeros_like(boundary); boundary_thinned[::2, ::2] = boundary[::2, ::2]
            overlay_bgr[boundary_thinned > 0] = cell_boundary_color
            logger.debug("Applied cell boundary and fill.")
            # Save debug visualization
            if debug_dir:
                cell_mask_debug = np.ones_like(overlay_bgr) * 200; cell_mask_debug[cell_mask_overlay_ready > 0] = cell_region_color
                cell_mask_debug[boundary_thinned > 0] = cell_boundary_color
                cv2.imwrite(os.path.join(debug_dir, f"debug_cell_mask_vis_L{overlay_level}.png"), cell_mask_debug)
        else:
             logger.info("Skipping cell region drawing (mask empty or None).")

        # --- Draw DAB+ Contours --- # <-- USES dab_mask_overlay_ready
        if dab_mask_overlay_ready is not None and np.sum(dab_mask_overlay_ready) > 0:
            logger.info("Drawing DAB+ contours...")
            contours_dab, _ = cv2.findContours(dab_mask_overlay_ready, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours_dab, -1, dab_contour_color, dab_contour_thickness)
        else:
            logger.info("Skipping DAB+ contours (mask empty or None).")

        # --- Draw Hotspots --- #
        if hotspots:
            logger.info(f"Drawing {len(hotspots)} hotspots...")
            try: scale_factor = slide.level_downsamples[overlay_level] / slide.level_downsamples[hotspot_level]
            except ZeroDivisionError: logger.error("Cannot calculate hotspot scale factor."); scale_factor = 1.0

            hotspot_debug_img = None
            if debug_dir: hotspot_debug_img = overlay_rgb.copy()

            for i, hotspot in enumerate(hotspots):
                hs_x_hl, hs_y_hl = hotspot['coords_level']; hs_w_hl, hs_h_hl = hotspot['size_level']
                # --- Use final_score if available, else density_score ---
                score_val = hotspot.get('final_score', hotspot.get('density_score', 0.0)) # Get score
                score_label = "Cnt" if 'final_score' in hotspot else "Dens" # Label depends on which score was found

                hs_x_ol=int(hs_x_hl*scale_factor); hs_y_ol=int(hs_y_hl*scale_factor)
                hs_w_ol=int(hs_w_hl*scale_factor); hs_h_ol=int(hs_h_hl*scale_factor)
                hs_x_ol=max(0,min(hs_x_ol, overlay_w-hotspot_thickness)); hs_y_ol=max(0,min(hs_y_ol, overlay_h-hotspot_thickness))
                hs_w_ol=max(max(1,hs_w_ol),hotspot_thickness*2); hs_h_ol=max(max(1,hs_h_ol),hotspot_thickness*2)
                if hs_x_ol+hs_w_ol > overlay_w: hs_w_ol=overlay_w-hs_x_ol
                if hs_y_ol+hs_h_ol > overlay_h: hs_h_ol=overlay_h-hs_y_ol

                hotspot_color = hotspot_colors[i % len(hotspot_colors)]
                cv2.rectangle(overlay_bgr, (hs_x_ol, hs_y_ol), (hs_x_ol+hs_w_ol, hs_y_ol+hs_h_ol), hotspot_color, hotspot_thickness)

                label_text = f"HS {i+1}:{score_label}={score_val:.1f}" # Format score
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, label_thickness)
                label_x = max(10, hs_x_ol + (hs_w_ol - text_w)//2); label_y = max(text_h + 20, hs_y_ol - 15)
                # Draw text with shadow
                for off_x in [-1,1]:
                    for off_y in [-1,1]: cv2.putText(overlay_bgr,label_text,(label_x+off_x*2,label_y+off_y*2),font,font_scale,(0,0,0),label_thickness+1,cv2.LINE_AA)
                cv2.putText(overlay_bgr, label_text, (label_x, label_y), font, font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)

                # Draw on debug image
                if hotspot_debug_img is not None:
                     hotspot_color_rgb = (hotspot_color[2],hotspot_color[1],hotspot_color[0])
                     cv2.rectangle(hotspot_debug_img,(hs_x_ol,hs_y_ol),(hs_x_ol+hs_w_ol,hs_y_ol+hs_h_ol),hotspot_color_rgb,hotspot_thickness//2)
                     cv2.putText(hotspot_debug_img,label_text,(label_x,label_y),font,font_scale*0.7,hotspot_color_rgb,label_thickness-1)

            # Save hotspot debug image
            if hotspot_debug_img is not None and debug_dir:
                hotspot_debug_path = os.path.join(debug_dir, f"debug_hotspots_only_L{overlay_level}.png")
                cv2.imwrite(hotspot_debug_path, cv2.cvtColor(hotspot_debug_img, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved hotspot-only debug image to {hotspot_debug_path}")
        else: logger.info("No hotspots provided to draw.")

        # --- Add Legend --- #
        logger.debug("Adding legend...")
        legend_x = 30; legend_y_start = 30; box_size = int(20*font_scale); legend_spacing = int(30*font_scale)
        text_offset_x = box_size+15; text_offset_y = box_size//2+int(font_scale*2)
        items = [("Tissue Outline",tissue_color), ("Tumor Outline",tumor_color)]
        # Conditionally add Cell Region to legend if it was drawn
        if cell_mask_overlay_ready is not None and np.sum(cell_mask_overlay_ready) > 0:
             items.append(("Cell Region", cell_region_color))
        items.extend([("DAB+ Contour",dab_contour_color), ("Hotspot Region",hotspot_colors[0])])

        legend_y = legend_y_start
        for i, (text, color) in enumerate(items):
             cv2.rectangle(overlay_bgr,(legend_x,legend_y),(legend_x+box_size,legend_y+box_size),color,-1)
             text_pos=(legend_x+text_offset_x, legend_y+text_offset_y)
             for off_x in [-1,1]:
                 for off_y in [-1,1]: cv2.putText(overlay_bgr,text,(text_pos[0]+off_x,text_pos[1]+off_y),font,font_scale*0.7,(0,0,0),label_thickness,cv2.LINE_AA)
             cv2.putText(overlay_bgr,text,text_pos,font,font_scale*0.7,(255,255,255),label_thickness-1,cv2.LINE_AA)
             legend_y += legend_spacing

        # --- Convert Final Overlay back to RGB ---
        final_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        logger.info("Overlay generation complete.")
        return final_overlay_rgb

    except Exception as e:
        logger.error(f"Error generating overlay: {e}")
        logger.error(traceback.format_exc())
        return None