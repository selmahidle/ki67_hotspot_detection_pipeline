import logging
import os
import numpy as np
import cv2
from PIL import Image
import openslide
import traceback
import imageio
from skimage.color import label2rgb
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize
import warnings

logger = logging.getLogger(__name__)

def format_for_save(img_array):
    """Converts image array to 3-channel uint8 for saving."""
    func_name = "format_for_save"
    if not isinstance(img_array, np.ndarray):
         logger.warning(f"[{func_name}]: Received non-ndarray input type {type(img_array)}. Returning small black image.")
         return np.zeros((100, 100, 3), dtype=np.uint8)

    if img_array.size == 0:
         logger.warning(f"[{func_name}]: Received empty ndarray. Returning small black image.")
         return np.zeros((100, 100, 3), dtype=np.uint8)

    if img_array.dtype == np.uint8:
        img_ubyte = img_array
    else:
        # Normalize float arrays to 0-1 if they exceed 1.0 or have negative values
        if img_array.dtype in (np.float32, np.float64, float):
            max_val = np.max(img_array)
            min_val = np.min(img_array)
            if min_val < 0 or max_val > 1.0:
                if np.isclose(min_val, max_val): # Handle constant image
                    img_array = np.zeros_like(img_array) if min_val < 0.5 else np.ones_like(img_array)
                else:
                    img_array = (img_array - min_val) / (max_val - min_val) # Scale to 0-1
            # Clamp slightly to avoid potential floating point issues at boundaries
            img_array = np.clip(img_array, 0, 1)
        # Convert to uint8
        try:
            # Use img_as_ubyte which scales 0-1 float to 0-255 uint8
            with warnings.catch_warnings(): # Suppress potential loss of precision warning
                warnings.simplefilter("ignore")
                img_ubyte = img_as_ubyte(img_array)
        except ValueError as e:
            logger.warning(f"[{func_name}]: img_as_ubyte failed ({e}). Returning black image. Input stats: min={np.min(img_array)}, max={np.max(img_array)}, dtype={img_array.dtype}")
            h, w = img_array.shape[:2] if img_array.ndim >= 2 else (100, 100)
            return np.zeros((h, w, 3), dtype=np.uint8)

    # Ensure 3 channels
    if img_ubyte.ndim == 2:
        return np.stack([img_ubyte] * 3, axis=-1)
    elif img_ubyte.ndim == 3:
        if img_ubyte.shape[2] == 1:
            return np.concatenate([img_ubyte] * 3, axis=-1)
        elif img_ubyte.shape[2] == 4: # Handle RGBA, drop alpha
            return img_ubyte[..., :3]
        elif img_ubyte.shape[2] == 3:
            return img_ubyte 
        else:
            logger.warning(f"[{func_name}]: Unexpected channel count ({img_ubyte.shape[2]}) in image with shape {img_ubyte.shape}. Returning first 3 channels if possible, else black.")
            h, w = img_ubyte.shape[:2]
            if img_ubyte.shape[2] > 3:
                return img_ubyte[..., :3]
            else:
                return np.zeros((h, w, 3), dtype=np.uint8)
    else:
        logger.warning(f"[{func_name}]: Unexpected dimensions ({img_ubyte.ndim}) in image with shape {img_ubyte.shape}. Returning black image.")
        try:
             h, w = img_ubyte.shape[:2]
             return np.zeros((h, w, 3), dtype=np.uint8)
        except:
             return np.zeros((100, 100, 3), dtype=np.uint8)


def save_stardist_comparison_plot(hs_patch_rgb, labels_filtered, ref_mask, save_path):
    """
    Generates and saves a side-by-side comparison image similar to the
    original standalone StarDist script: [Original Patch | Ref Mask | Prediction Overlay].

    Args:
        hs_patch_rgb (np.ndarray): The original RGB patch (uint8, H, W, 3).
        labels_filtered (np.ndarray): The final StarDist label mask (uint16/int, H, W),
                                      potentially after size filtering.
        ref_mask (np.ndarray): A reference mask to display (e.g., DAB mask patch,
                                 SMP Cell mask patch). Should be binary (0/1 or 0/255, H, W).
        save_path (str): Full path including filename where the comparison JPG/PNG
                         should be saved.
    """
    logger.debug(f"Generating StarDist comparison plot for: {save_path}")
    try:
        # 1. Prepare original patch image
        original_display = format_for_save(hs_patch_rgb) # Ensure uint8, 3ch
        if original_display is None or original_display.ndim != 3 or original_display.shape[2] != 3:
             logger.error("Original patch could not be formatted correctly for comparison plot.")
             return
        original_shape = original_display.shape[:2] # H, W

        # 2. Prepare Predicted Overlay (using FILTERED labels)
        pred_overlay_display = None
        if labels_filtered is not None and isinstance(labels_filtered, np.ndarray) and labels_filtered.ndim == 2:
             labels_viz = labels_filtered
             if labels_viz.shape != original_shape: # Resize labels if needed
                 logger.warning(f"Resizing filtered labels {labels_viz.shape} to match original {original_shape} for plot.")
                 labels_viz = resize(labels_viz, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(labels_filtered.dtype)

             with warnings.catch_warnings(): # Convert original patch to float 0-1 for label2rgb
                 warnings.simplefilter("ignore"); original_patch_float = img_as_float(hs_patch_rgb)

             # Create overlay
             pred_overlay_rgb = label2rgb(labels_viz, image=original_patch_float, bg_label=0, bg_color=None, kind='overlay', image_alpha=0.5, alpha=0.5)
             pred_overlay_display = format_for_save(pred_overlay_rgb) # Format for saving
             if pred_overlay_display is None: raise ValueError("format_for_save failed for prediction overlay")
        else:
            logger.warning("Invalid filtered labels for comparison plot. Using black placeholder.")
            pred_overlay_display = np.zeros_like(original_display)

        # 3. Prepare Reference Mask display (e.g., DAB mask)
        gt_display = None
        if ref_mask is not None and isinstance(ref_mask, np.ndarray) and ref_mask.ndim == 2:
             ref_mask_viz = ref_mask
             if ref_mask_viz.shape != original_shape: # Resize if needed
                  logger.warning(f"Resizing reference mask {ref_mask_viz.shape} to {original_shape} for plot.")
                  ref_mask_viz = resize(ref_mask_viz, original_shape, order=0, preserve_range=True, anti_aliasing=False)
             # Convert to 0/255 uint8 RGB
             # --- Fix applied here too for consistency ---
             ref_mask_uint8 = ((ref_mask_viz > 0) * 255).astype(np.uint8) # Explicitly cast here
             gt_display = format_for_save(ref_mask_uint8) # Pass the uint8 version
             # --- End fix ---
             if gt_display is None: raise ValueError("format_for_save failed for reference mask")
        else:
            logger.warning("Invalid reference mask for comparison plot. Using black placeholder.")
            gt_display = np.zeros_like(original_display)

        # 4. Stack and Save
        target_h = original_display.shape[0]
        components = []
        valid = True
        for img, name in [(original_display, "Original"), (gt_display, "Reference"), (pred_overlay_display, "Prediction")]:
             if img is not None and img.shape[0] == target_h and img.ndim == 3 and img.shape[2] == 3:
                 components.append(img)
             else:
                 logger.error(f"Component '{name}' has invalid shape/type for stacking: {img.shape if hasattr(img, 'shape') else type(img)}")
                 valid = False; break

        if valid and len(components) == 3:
            combined_img = np.hstack(components)
            try:
                imageio.imwrite(save_path, combined_img) # Use imageio which handles formats well
                logger.debug(f" Saved comparison plot to {save_path}")
            except Exception as e_save:
                logger.error(f"Failed to save comparison plot using imageio to {save_path}: {e_save}")

        else:
             logger.error(f"Could not stack images for comparison plot for {save_path}.")

    except Exception as e_plot:
        logger.error(f"Error generating comparison plot saved to {save_path}: {e_plot}", exc_info=True)


def generate_overlay(slide, overlay_level, hotspot_level,
                     tissue_mask_overlay, tumor_mask_overlay, cell_mask_binary_l2,
                     hotspots, dab_mask_l2=None, debug_dir=None):
    """
    Generates an overlay image showing WSI, tissue/tumor outlines, cell regions,
    DAB+ areas (if provided), and identified hotspots.
    """
    logger.info(f"Generating overlay for Level {overlay_level}...")
    # --- Input Validation ---
    if not isinstance(slide, openslide.OpenSlide): logger.error("Invalid 'slide' object."); return None
    if overlay_level < 0 or overlay_level >= slide.level_count: logger.error(f"Invalid overlay_level {overlay_level}."); return None
    if hotspot_level < 0 or hotspot_level >= slide.level_count: logger.error(f"Invalid hotspot_level {hotspot_level}."); return None
    try:
        # --- Read Base Image ---
        overlay_dims = slide.level_dimensions[overlay_level]; overlay_w, overlay_h = overlay_dims[0], overlay_dims[1]
        logger.info(f"Reading base image L{overlay_level} ({overlay_w}x{overlay_h})")
        try: base_image_pil = slide.read_region((0,0),overlay_level,(overlay_w,overlay_h)).convert('RGB'); overlay_rgb=np.array(base_image_pil)
        except Exception as e: logger.error(f"Error reading overlay base: {e}", exc_info=True); return None
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        target_overlay_shape = (overlay_h, overlay_w)

        def prepare_mask(mask, mask_name, target_shape=None):
            """Helper to prepare mask: ensure uint8, 0/255, and optionally resize."""
            if mask is None: return None
            if not isinstance(mask, np.ndarray): logger.warning(f"{mask_name} not np array."); return None
            # Ensure mask is binary 0/255 uint8
            if mask.dtype != np.uint8 or np.any((mask != 0) & (mask != 1) & (mask != 255)):
                 mask_binary = ((mask > 0) * 255).astype(np.uint8)
            else:
                 mask_binary = mask # Assume already 0/255 uint8

            # Resize if needed
            if target_shape is not None and mask_binary.shape[:2] != target_shape:
                 if len(target_shape)>=2 and target_shape[0]>0 and target_shape[1]>0:
                      try:
                           logger.debug(f"Resizing {mask_name} from {mask_binary.shape[:2]} to {target_shape}")
                           return cv2.resize(mask_binary,(target_shape[1],target_shape[0]),interpolation=cv2.INTER_NEAREST)
                      except Exception as e: logger.error(f"Resize fail {mask_name}: {e}"); return None
                 else: logger.error(f"Invalid target shape {target_shape} for {mask_name}."); return None
            else:
                return mask_binary # Return original binary mask if shape matches or no target

        # Prepare masks using the helper
        tissue_mask_overlay_ready = prepare_mask(tissue_mask_overlay, "Tissue", target_overlay_shape)
        tumor_mask_overlay_ready = prepare_mask(tumor_mask_overlay, "Tumor", target_overlay_shape)
        # Cell and DAB masks are expected at L2 (hotspot_level), resizing handled within generate_overlay if needed
        cell_mask_l2_ready = prepare_mask(cell_mask_binary_l2, "Cell (L2)")
        dab_mask_l2_ready = prepare_mask(dab_mask_l2, "DAB (L2)")

        # Resize cell/DAB masks specifically for overlay drawing if overlay_level != hotspot_level (L2)
        cell_mask_overlay_ready = cell_mask_l2_ready # Assume same level initially
        dab_mask_overlay_ready = dab_mask_l2_ready
        if overlay_level != hotspot_level:
            if cell_mask_l2_ready is not None:
                cell_mask_overlay_ready = prepare_mask(cell_mask_l2_ready, "Cell (for Overlay)", target_overlay_shape)
            if dab_mask_l2_ready is not None:
                dab_mask_overlay_ready = prepare_mask(dab_mask_l2_ready, "DAB (for Overlay)", target_overlay_shape)

        # --- Save Mask Sanity Checks ---
        if debug_dir:
             os.makedirs(debug_dir, exist_ok=True)
             # Save the masks *after* potential resizing for the overlay
             if tissue_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"tissue_mask_overlay_L{overlay_level}.png"), tissue_mask_overlay_ready)
             if tumor_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"tumor_mask_overlay_L{overlay_level}.png"), tumor_mask_overlay_ready)
             if cell_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"cell_mask_overlay_L{overlay_level}.png"), cell_mask_overlay_ready)
             if dab_mask_overlay_ready is not None: cv2.imwrite(os.path.join(debug_dir, f"dab_mask_overlay_L{overlay_level}.png"), dab_mask_overlay_ready)


        # --- Define Colors and Drawing Parameters ---
        tissue_color=(0,255,0) # Green
        tumor_color=(255,0,0) # Blue
        cell_region_color=(0,200,255) # Yellowish-orange
        cell_boundary_color=(0,255,255) # Yellow
        dab_contour_color=(0,0,255) # Red
        hotspot_colors=[(0,0,255),(0,255,255),(255,0,255),(255,255,0),(0,255,0)] # BGR: Red, Yellow, Magenta, Cyan, Green
        contour_thickness_tissue=max(1, int(overlay_w * 0.0015)) # Relative thickness
        contour_thickness_tumor=max(1, int(overlay_w * 0.001))
        hotspot_thickness=max(4, int(overlay_w * 0.004)) # Make hotspot border thicker
        dab_contour_thickness=max(1, int(overlay_w * 0.0005))
        font=cv2.FONT_HERSHEY_SIMPLEX
        font_scale=max(0.8, overlay_w / 10000.0) # Scale font with image width
        label_thickness=max(1, int(font_scale * 1.5))


        # --- Draw Elements ---
        # Tissue Contours
        if tissue_mask_overlay_ready is not None and np.sum(tissue_mask_overlay_ready)>0:
            contours,_ = cv2.findContours(tissue_mask_overlay_ready,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr,contours,-1,tissue_color,contour_thickness_tissue)
            logger.debug(f"Drew {len(contours)} tissue contours.")

        # Tumor Contours
        if tumor_mask_overlay_ready is not None and np.sum(tumor_mask_overlay_ready)>0:
            contours,_=cv2.findContours(tumor_mask_overlay_ready,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr,contours,-1,tumor_color,contour_thickness_tumor)
            logger.debug(f"Drew {len(contours)} tumor contours.")

        # Cell Regions/Boundaries
        if cell_mask_overlay_ready is not None and np.sum(cell_mask_overlay_ready)>0:
            logger.info("Drawing cell regions/boundaries...")
            # Fill
            cell_overlay_fill=overlay_bgr.copy()
            cell_overlay_fill[cell_mask_overlay_ready>0]=cell_region_color
            alpha_cell=0.2 # Make fill slightly more transparent
            cv2.addWeighted(cell_overlay_fill,alpha_cell,overlay_bgr,1-alpha_cell,0,overlay_bgr);
            # Boundary (Thinned)
            kernel=np.ones((3,3),np.uint8);
            dilated=cv2.dilate(cell_mask_overlay_ready,kernel,1);
            eroded=cv2.erode(cell_mask_overlay_ready,kernel,1);
            boundary=dilated-eroded;
            boundary_thinned=np.zeros_like(boundary);
            boundary_thinned[::2,::2]=boundary[::2,::2]; # Thinning step
            overlay_bgr[boundary_thinned>0]=cell_boundary_color;
            logger.debug("Applied cell boundary/fill.")
        else: logger.info("Skipping cell region drawing (mask empty or None).")

        # DAB Contours
        if dab_mask_overlay_ready is not None and np.sum(dab_mask_overlay_ready)>0:
            logger.info("Drawing DAB+ contours...")
            contours_dab,_=cv2.findContours(dab_mask_overlay_ready,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr,contours_dab,-1,dab_contour_color,dab_contour_thickness)
            logger.debug(f"Drew {len(contours_dab)} DAB+ contours.")
        else: logger.info("Skipping DAB+ contours (mask empty or None).")

        # Hotspots
        if hotspots:
            logger.info(f"Drawing {len(hotspots)} hotspots...")
            scale_factor = slide.level_downsamples[overlay_level] / slide.level_downsamples[hotspot_level]
            logger.debug(f"Hotspot scaling factor L{hotspot_level}->L{overlay_level}: {scale_factor:.3f}")

            for i, hotspot in enumerate(hotspots):
                hs_x_hl, hs_y_hl = hotspot['coords_level']
                hs_w_hl, hs_h_hl = hotspot['size_level']

                # Scale coordinates and size to overlay_level
                hs_x = int(hs_x_hl * scale_factor)
                hs_y = int(hs_y_hl * scale_factor)
                hs_w = int(hs_w_hl * scale_factor)
                hs_h = int(hs_h_hl * scale_factor)

                # Ensure dimensions are at least 1 pixel
                hs_w = max(1, hs_w); hs_h = max(1, hs_h)
                # Ensure coordinates are within bounds
                hs_x = min(max(0, hs_x), overlay_w - hs_w)
                hs_y = min(max(0, hs_y), overlay_h - hs_h)

                logger.debug(f" Hotspot {i+1}: L{overlay_level} Rect = ({hs_x},{hs_y}, {hs_w}x{hs_h})")
                color = hotspot_colors[i % len(hotspot_colors)]
                # Draw rectangle
                cv2.rectangle(overlay_bgr, (hs_x, hs_y), (hs_x + hs_w, hs_y + hs_h), color, hotspot_thickness)

                # Add label (Hotspot number and Final Score/Count)
                score = hotspot.get('final_score', hotspot.get('density_score', -1)) # Use final score if avail
                label_text = f"HS {i+1}: {score}" if isinstance(score, int) else f"HS {i+1}: {score:.3f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale*1.5, label_thickness+1) # Slightly larger font for HS label

                # Position label inside top-left corner
                label_x = hs_x + hotspot_thickness + 5
                label_y = hs_y + hotspot_thickness + text_h + 5

                # Ensure label stays within image bounds
                label_x = min(label_x, overlay_w - text_w - 5)
                label_y = min(label_y, overlay_h - 5)
                label_x = max(label_x, 5)
                label_y = max(label_y, text_h + 5)

                # Draw text with black outline for better visibility
                cv2.putText(overlay_bgr, label_text, (label_x + 1, label_y + 1), font, font_scale*1.5, (0,0,0), label_thickness+1, cv2.LINE_AA)
                cv2.putText(overlay_bgr, label_text, (label_x, label_y), font, font_scale*1.5, color, label_thickness, cv2.LINE_AA)
        else: logger.info("No hotspots provided to draw.")

        # --- Add Legend --- #
        # (Legend drawing code remains largely the same, ensure colors match definitions above)
        legend_x = 30
        legend_y = 30
        box_size = int(25 * font_scale)
        legend_spacing = int(35 * font_scale)
        text_offset_y = box_size // 2 + int(5 * font_scale) # Adjust based on font scale

        def draw_legend_item(img, y, color, text):
            cv2.rectangle(img, (legend_x, y), (legend_x+box_size, y+box_size), color, -1)
            # Draw text with shadow
            cv2.putText(img, text, (legend_x + box_size + 10 + 1, y + text_offset_y + 1), font, font_scale, (0,0,0), label_thickness+1, cv2.LINE_AA)
            cv2.putText(img, text, (legend_x + box_size + 10, y + text_offset_y), font, font_scale, color, label_thickness, cv2.LINE_AA)
            return y + legend_spacing

        legend_y = draw_legend_item(overlay_bgr, legend_y, tissue_color, "Tissue Outline")
        legend_y = draw_legend_item(overlay_bgr, legend_y, tumor_color, "Tumor Outline")
        legend_y = draw_legend_item(overlay_bgr, legend_y, cell_boundary_color, "Cell Boundary")
        legend_y = draw_legend_item(overlay_bgr, legend_y, dab_contour_color, "DAB+ Contour")
        legend_y = draw_legend_item(overlay_bgr, legend_y, hotspot_colors[0], "Hotspot Region (Ranked)")


        # --- Convert Final Overlay back to RGB ---
        final_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        logger.info("Overlay generation complete.")
        return final_overlay_rgb

    except Exception as e:
        logger.error(f"Error generating overlay: {e}")
        logger.error(traceback.format_exc()) # Use imported traceback
        return None