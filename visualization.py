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
import matplotlib.pyplot as plt
from stain_utils import get_dab_mask


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
        if img_array.dtype in (np.float32, np.float64, float):
            max_val = np.max(img_array)
            min_val = np.min(img_array)
            if min_val < 0 or max_val > 1.0: # Normalize if not in [0,1] range
                if np.isclose(min_val, max_val): # Handle constant image
                    img_array = np.zeros_like(img_array) if min_val < 0.5 else np.ones_like(img_array)
                else:
                    img_array = (img_array - min_val) / (max_val - min_val + 1e-9) # Normalize to [0,1]
            img_array = np.clip(img_array, 0, 1) # Clip to ensure [0,1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Suppress precision loss warnings
                img_ubyte = img_as_ubyte(img_array)
        except ValueError as e:
            logger.warning(f"[{func_name}]: img_as_ubyte failed ({e}). Returning black image. Input stats: min={np.min(img_array)}, max={np.max(img_array)}, dtype={img_array.dtype}")
            h, w = img_array.shape[:2] if img_array.ndim >= 2 else (100, 100)
            return np.zeros((h, w, 3), dtype=np.uint8)

    if img_ubyte.ndim == 2: # Grayscale
        return np.stack([img_ubyte] * 3, axis=-1) # Convert to 3-channel
    elif img_ubyte.ndim == 3:
        if img_ubyte.shape[2] == 1: # Single channel
            return np.concatenate([img_ubyte] * 3, axis=-1) # Convert to 3-channel
        elif img_ubyte.shape[2] == 4: # RGBA
            return img_ubyte[..., :3] # Return RGB
        elif img_ubyte.shape[2] == 3: # RGB
            return img_ubyte
        else:
            logger.warning(f"[{func_name}]: Unexpected channel count ({img_ubyte.shape[2]}) in image with shape {img_ubyte.shape}. Returning first 3 channels if possible, else black.")
            h, w = img_ubyte.shape[:2]
            if img_ubyte.shape[2] > 3: return img_ubyte[..., :3]
            else: return np.zeros((h, w, 3), dtype=np.uint8)
    else:
        logger.warning(f"[{func_name}]: Unexpected dimensions ({img_ubyte.ndim}) in image with shape {img_ubyte.shape}. Returning black image.")
        try:
            h, w = img_ubyte.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        except: return np.zeros((100, 100, 3), dtype=np.uint8)


def save_stardist_comparison_plot(hs_patch_rgb, labels_filtered, ref_mask, save_path):
    """
    Generates and saves a Matplotlib comparison plot: original patch, reference mask, StarDist overlay.
    """
    logger.debug(f"Generating Matplotlib StarDist comparison plot for: {save_path}")

    try:
        original_display = format_for_save(hs_patch_rgb.copy())
        if original_display is None:
            logger.error("Original patch (hs_patch_rgb) could not be formatted. Cannot create comparison plot.")
            original_display = np.zeros((100, 100, 3), dtype=np.uint8)
            original_shape = (100,100)
        elif original_display.ndim != 3 or original_display.shape[2] != 3:
             logger.warning(f"Formatted original_display has unexpected shape {original_display.shape}. Attempting to use, but might fail.")
             original_shape = original_display.shape[:2]
        else:
            original_shape = original_display.shape[:2]

        pred_overlay_display = np.zeros_like(original_display, dtype=np.uint8)
        if labels_filtered is not None and isinstance(labels_filtered, np.ndarray) and labels_filtered.ndim == 2 and labels_filtered.size > 0:
            labels_viz = labels_filtered
            if labels_viz.shape != original_shape:
                logger.debug(f"Resizing labels_filtered from {labels_viz.shape} to {original_shape}")
                labels_viz = resize(labels_viz, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(labels_filtered.dtype)

            base_for_overlay = hs_patch_rgb.copy()
            if base_for_overlay.shape[:2] != labels_viz.shape:
                 base_for_overlay_resized = cv2.resize(base_for_overlay, (labels_viz.shape[1], labels_viz.shape[0]))
            else:
                 base_for_overlay_resized = base_for_overlay

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hs_patch_rgb_float_for_overlay = img_as_float(base_for_overlay_resized)

            pred_overlay_rgb_float = label2rgb(labels_viz, image=hs_patch_rgb_float_for_overlay, bg_label=0, bg_color=None, kind='overlay', image_alpha=0.5, alpha=0.5)
            formatted_pred_overlay = format_for_save(pred_overlay_rgb_float)
            if formatted_pred_overlay is not None and formatted_pred_overlay.shape == pred_overlay_display.shape :
                pred_overlay_display = formatted_pred_overlay
            elif formatted_pred_overlay is not None:
                logger.warning(f"Prediction overlay shape {formatted_pred_overlay.shape} mismatch with placeholder {pred_overlay_display.shape}. Using placeholder.")
            else:
                logger.warning("format_for_save failed for prediction overlay. Using placeholder.")
        else:
            logger.debug("labels_filtered is None, empty, or not a 2D array. Using placeholder for prediction overlay.")

        gt_display = np.zeros_like(original_display, dtype=np.uint8)
        if ref_mask is not None and isinstance(ref_mask, np.ndarray) and ref_mask.ndim == 2 and ref_mask.size > 0:
            ref_mask_viz = ref_mask
            if ref_mask_viz.shape != original_shape:
                logger.debug(f"Resizing ref_mask from {ref_mask_viz.shape} to {original_shape}")
                ref_mask_viz = resize(ref_mask_viz, original_shape, order=0, preserve_range=True, anti_aliasing=False)

            ref_mask_binary_uint8 = ((ref_mask_viz > 0) * 255).astype(np.uint8)
            formatted_gt_display = format_for_save(ref_mask_binary_uint8)
            if formatted_gt_display is not None and formatted_gt_display.shape == gt_display.shape:
                gt_display = formatted_gt_display
            elif formatted_gt_display is not None:
                logger.warning(f"GT display shape {formatted_gt_display.shape} mismatch with placeholder {gt_display.shape}. Using placeholder.")
            else:
                logger.warning("format_for_save failed for reference mask. Using placeholder.")
        else:
            logger.debug("ref_mask is None, empty, or not a 2D array. Using placeholder for GT display.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = axes.ravel()

        titles = ['Original Patch', 'Reference Mask', 'StarDist Prediction']
        images_to_plot = [original_display, gt_display, pred_overlay_display]

        for i, (ax, img_data, title_text) in enumerate(zip(axes, images_to_plot, titles)):
            if img_data is not None:
                if img_data.dtype != np.uint8:
                    logger.warning(f"Image data for '{title_text}' is not uint8 (dtype: {img_data.dtype}). Attempting to display.")
                ax.imshow(img_data)
            else:
                ax.text(0.5, 0.5, 'Image\nNot Available', ha='center', va='center', fontsize=10, transform=ax.transAxes)
                logger.warning(f"Image data for subplot '{title_text}' was None.")
            ax.set_title(title_text, fontsize=12)
            ax.axis('off')

        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully saved Matplotlib comparison plot to {save_path}")
    except Exception as e_plot:
        logger.error(f"Error generating Matplotlib comparison plot for {save_path}: {e_plot}", exc_info=True)


def draw_label_contours_on_bgr(bgr_image_roi, labels, line_color=(0, 255, 0), line_thickness=1):
    """Draws contours from a label mask directly onto a BGR image ROI (in place)."""
    if labels is None or labels.size == 0 or bgr_image_roi is None:
        return

    labels_to_process = labels
    if labels.shape[:2] != bgr_image_roi.shape[:2]:
        logger.warning(f"draw_label_contours_on_bgr: Shape mismatch labels {labels.shape} vs ROI {bgr_image_roi.shape}. Resizing labels.")
        try:
            labels_resized = cv2.resize(labels.astype(np.uint16),
                                     (bgr_image_roi.shape[1], bgr_image_roi.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            labels_to_process = labels_resized
        except Exception as e_resize:
            logger.error(f"Failed to resize labels for contour drawing: {e_resize}")
            return

    unique_labels = np.unique(labels_to_process[labels_to_process > 0])
    if len(unique_labels) == 0:
        return

    for label_id in unique_labels:
        mask = (labels_to_process == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr_image_roi, contours, -1, line_color, line_thickness)


def generate_overlay(slide, overlay_level, hotspot_level,
                     tissue_mask_overlay, tumor_mask_overlay, cell_mask_binary_l2,
                     hotspots,
                     debug_dir=None):
    """
    Generates an overlay image with tissue/tumor outlines, cell masks, DAB+ regions, and hotspots.
    """
    logger.info(f"Generating overlay for Level {overlay_level}...")
    if not isinstance(slide, openslide.OpenSlide):
        logger.error("Invalid 'slide' object provided to generate_overlay.")
        return None

    try:
        overlay_dims = slide.level_dimensions[overlay_level]
        overlay_w, overlay_h = overlay_dims[0], overlay_dims[1]

        try:
            base_image_pil = slide.read_region((0, 0), overlay_level, (overlay_w, overlay_h)).convert('RGB')
            overlay_rgb_original = np.array(base_image_pil) # Keep original RGB for get_dab_mask
            overlay_bgr = cv2.cvtColor(overlay_rgb_original, cv2.COLOR_RGB2BGR) # Work in BGR for OpenCV
        except Exception as e_read:
            logger.error(f"Failed to read base image for overlay L{overlay_level}: {e_read}")
            return None

        # Colors and Thicknesses (BGR for OpenCV)
        tissue_color = (0, 255, 0)      # Green
        tumor_color = (255, 0, 0)       # Blue
        cell_region_fill_color = (150, 255, 255) # Light Yellow for fill (general cell mask)
        dab_intersect_cell_fill_color = (0, 0, 200) # Darker Red for fill (DAB+ within cells)
        dab_intersect_cell_border_color = (0, 0, 255) # Bright Red for border (DAB+ within cells)
        hotspot_box_colors = [(0,165,255),(0,255,255),(255,0,255),(255,255,0),(0,255,0)] # Orange, Yellow, Magenta, Cyan, Green

        ct_tissue = max(1, int(overlay_w * 0.0015))
        ct_tumor = max(1, int(overlay_w * 0.0015))
        hs_box_thick = max(2, int(overlay_w * 0.002))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_legend = max(0.7, overlay_w / 12000.0)
        font_scale_hotspot_label = max(1.0, overlay_w / 8000.0)
        lbl_thick = max(1, int(font_scale_legend * 1.5))
        hs_lbl_thick = max(2, int(font_scale_hotspot_label * 1.5))

        def _draw_stardist_in_roi(roi_bgr_to_draw_on, stardist_labels_hs, classified_dab_hs, fill_alpha=0.35, border_thick=1):
            """Helper to draw StarDist objects within a hotspot ROI, colored by DAB status."""
            if roi_bgr_to_draw_on is None or stardist_labels_hs is None or classified_dab_hs is None:
                return

            target_shape_hw = roi_bgr_to_draw_on.shape[:2]
            try:
                if stardist_labels_hs.shape != target_shape_hw:
                    stardist_labels_hs = cv2.resize(stardist_labels_hs.astype(np.uint16), (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
                if classified_dab_hs.shape != target_shape_hw:
                    classified_dab_hs = cv2.resize(classified_dab_hs.astype(np.uint8), (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
            except Exception as e_resize_hs:
                logger.error(f"Error resizing masks for hotspot internal drawing: {e_resize_hs}")
                return

            sd_color_dab_pos_bgr = (255, 0, 0)  # Blue for DAB+ StarDist cells
            sd_color_dab_neg_bgr = (0, 255, 0)  # Green for DAB- StarDist cells

            unique_sd_ids = np.unique(stardist_labels_hs[stardist_labels_hs != 0])
            for sd_id in unique_sd_ids:
                obj_mask_bool = (stardist_labels_hs == sd_id)
                obj_mask_u8 = obj_mask_bool.astype(np.uint8)
                dab_px_in_obj = classified_dab_hs[obj_mask_bool]
                if dab_px_in_obj.size == 0: continue

                counts = np.bincount(dab_px_in_obj.astype(int), minlength=3) # 0: no class, 1: pos, 2: neg
                n_pos, n_neg = counts[1], counts[2]

                fill_col, border_col = None, None
                if n_pos > 0 and n_pos >= n_neg: # Classified as DAB positive
                    fill_col, border_col = sd_color_dab_pos_bgr, sd_color_dab_pos_bgr
                elif n_neg > 0: # Classified as DAB negative
                    fill_col, border_col = sd_color_dab_neg_bgr, sd_color_dab_neg_bgr

                if fill_col:
                    contours, _ = cv2.findContours(obj_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    temp_fill_layer = roi_bgr_to_draw_on.copy()
                    cv2.drawContours(temp_fill_layer, contours, -1, fill_col, cv2.FILLED)
                    cv2.addWeighted(temp_fill_layer, fill_alpha, roi_bgr_to_draw_on, 1 - fill_alpha, 0, roi_bgr_to_draw_on)
                    cv2.drawContours(roi_bgr_to_draw_on, contours, -1, border_col, border_thick)

        if tissue_mask_overlay is not None and np.any(tissue_mask_overlay):
            tissue_mask_final = tissue_mask_overlay.astype(np.uint8)
            if tissue_mask_final.shape[:2] != (overlay_h, overlay_w):
                tissue_mask_final = cv2.resize(tissue_mask_final, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(tissue_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, tissue_color, ct_tissue)

        if tumor_mask_overlay is not None and np.any(tumor_mask_overlay):
            tumor_mask_final = tumor_mask_overlay.astype(np.uint8)
            if tumor_mask_final.shape[:2] != (overlay_h, overlay_w):
                tumor_mask_final = cv2.resize(tumor_mask_final, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(tumor_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, tumor_color, ct_tumor)

        # General Cell Mask (from cell_mask_binary_l2)
        cell_mask_scaled_overlay = None 
        if cell_mask_binary_l2 is not None and np.any(cell_mask_binary_l2):
            cell_mask_u8 = cell_mask_binary_l2.astype(np.uint8)
            if cell_mask_u8.shape[:2] != (overlay_h, overlay_w):
                cell_mask_scaled_overlay = cv2.resize(cell_mask_u8, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            else:
                cell_mask_scaled_overlay = cell_mask_u8

            if cell_mask_scaled_overlay is not None and np.any(cell_mask_scaled_overlay):
                fill_layer_yellow_cells = overlay_bgr.copy()
                fill_layer_yellow_cells[cell_mask_scaled_overlay > 0] = cell_region_fill_color # cell_region_fill_color is (224, 255, 255) Light Yellow
                
                alpha_yellow_fill = 0.4
                cv2.addWeighted(fill_layer_yellow_cells, alpha_yellow_fill, overlay_bgr, 1 - alpha_yellow_fill, 0, overlay_bgr)


        # DAB+ regions (on-the-fly from overlay, intersected with cell mask)
        dab_plus_on_overlay_img = get_dab_mask(overlay_rgb_original, dab_threshold=0.15) # Use original RGB
        if dab_plus_on_overlay_img.size > 0 and cell_mask_scaled_overlay is not None and np.any(cell_mask_scaled_overlay):
            if dab_plus_on_overlay_img.shape != cell_mask_scaled_overlay.shape:
                 logger.warning(f"Shape mismatch: on-the-fly DAB mask {dab_plus_on_overlay_img.shape} vs scaled cell mask {cell_mask_scaled_overlay.shape}. Resizing DAB mask.")
                 dab_plus_on_overlay_img = cv2.resize(dab_plus_on_overlay_img.astype(np.uint8), (cell_mask_scaled_overlay.shape[1], cell_mask_scaled_overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

            final_dab_regions_to_draw = (dab_plus_on_overlay_img > 0) & (cell_mask_scaled_overlay > 0)
            final_dab_regions_to_draw = final_dab_regions_to_draw.astype(np.uint8)

            if np.any(final_dab_regions_to_draw):
                fill_layer_dab = overlay_bgr.copy()
                fill_layer_dab[final_dab_regions_to_draw > 0] = dab_intersect_cell_fill_color
                cv2.addWeighted(fill_layer_dab, 0.35, overlay_bgr, 0.65, 0, overlay_bgr) # Slightly more opaque fill
                contours_dab, _ = cv2.findContours(final_dab_regions_to_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_bgr, contours_dab, -1, dab_intersect_cell_border_color, 1)

        # Hotspots with internal StarDist details
        if hotspots:
            logger.info(f"Drawing {len(hotspots)} hotspots with internal StarDist details...")
            sf_coords_hs_drawing = slide.level_downsamples[overlay_level] / slide.level_downsamples[hotspot_level]

            for i, hotspot_data in enumerate(hotspots):
                x_hl, y_hl = hotspot_data['coords_level']
                w_hl, h_hl = hotspot_data['size_level']
                x_ol = int(x_hl * sf_coords_hs_drawing)
                y_ol = int(y_hl * sf_coords_hs_drawing)
                w_ol = int(w_hl * sf_coords_hs_drawing)
                h_ol = int(h_hl * sf_coords_hs_drawing)

                x_ol_clamped = min(max(0, x_ol), overlay_bgr.shape[1] - 1)
                y_ol_clamped = min(max(0, y_ol), overlay_bgr.shape[0] - 1)
                w_ol_clamped = min(w_ol, overlay_bgr.shape[1] - x_ol_clamped)
                h_ol_clamped = min(h_ol, overlay_bgr.shape[0] - y_ol_clamped)

                if w_ol_clamped <= 0 or h_ol_clamped <= 0:
                    logger.warning(f"Skipping hotspot {i+1} due to zero/negative clamped dimensions for drawing.")
                    continue

                hotspot_roi_on_overlay_bgr = overlay_bgr[y_ol_clamped : y_ol_clamped + h_ol_clamped, x_ol_clamped : x_ol_clamped + w_ol_clamped]
                stardist_labels_for_hs = hotspot_data.get('stardist_labels')
                classified_labels_dab_for_hs = hotspot_data.get('classified_labels_dab_hs') # This key needs to be in hotspot_data

                if hotspot_roi_on_overlay_bgr.size > 0 and stardist_labels_for_hs is not None and classified_labels_dab_for_hs is not None:
                    _draw_stardist_in_roi(
                        hotspot_roi_on_overlay_bgr, # Pass the actual slice to be modified
                        stardist_labels_for_hs,
                        classified_labels_dab_for_hs,
                        fill_alpha=0.35,
                        border_thick=1
                    )
                else:
                    if stardist_labels_for_hs is None: logger.debug(f"Hotspot {i+1} missing 'stardist_labels'.")
                    if classified_labels_dab_for_hs is None: logger.debug(f"Hotspot {i+1} missing 'classified_labels_dab_hs'.")
                    if hotspot_roi_on_overlay_bgr.size == 0 : logger.warning(f"Hotspot {i+1} ROI is empty.")

                hs_col = hotspot_box_colors[i % len(hotspot_box_colors)]
                cv2.rectangle(overlay_bgr, (x_ol_clamped, y_ol_clamped), (x_ol_clamped + w_ol_clamped, y_ol_clamped + h_ol_clamped), hs_col, hs_box_thick)

                pi = hotspot_data.get('stardist_proliferation_index')
                lbl_text_hs = f"HS{i+1}"
                if pi is not None: lbl_text_hs += f": {pi*100:.1f}%"
                else: lbl_text_hs += ": N/A"

                (tw_hs,th_hs),bl_hs = cv2.getTextSize(lbl_text_hs, font, font_scale_hotspot_label, hs_lbl_thick)
                lby1_hs = max(th_hs + bl_hs + 5, y_ol_clamped - th_hs - bl_hs - 10) # Position label above box
                lbx1_hs = max(5, x_ol_clamped + (w_ol_clamped - tw_hs) // 2) # Centered if possible
                tx_hs = lbx1_hs
                ty_hs = lby1_hs + th_hs
                cv2.putText(overlay_bgr, lbl_text_hs, (tx_hs+1, ty_hs+1), font, font_scale_hotspot_label, (0,0,0), hs_lbl_thick+1, cv2.LINE_AA) # Shadow
                cv2.putText(overlay_bgr, lbl_text_hs, (tx_hs, ty_hs), font, font_scale_hotspot_label, (255,255,255), hs_lbl_thick, cv2.LINE_AA) # Text
        else:
            logger.info("No hotspots provided to draw internal StarDist details for.")

        # Legend
        lx, ly = 30, 30
        bs = int(25 * font_scale_legend)
        ls = int(35 * font_scale_legend)
        toy = bs // 2 + int(5 * font_scale_legend)

        def draw_legend_item(img, y_pos, color, txt):
            cv2.rectangle(img, (lx, y_pos), (lx + bs, y_pos + bs), color, -1)
            # Text with shadow
            cv2.putText(img, txt, (lx + bs + 11, y_pos + toy + 1), font, font_scale_legend, (0,0,0), lbl_thick + 1, cv2.LINE_AA)
            cv2.putText(img, txt, (lx + bs + 10, y_pos + toy), font, font_scale_legend, color, lbl_thick, cv2.LINE_AA)
            return y_pos + ls

        legend_items = [
            (tissue_color, "Tissue Outline"),
            (tumor_color, "Tumor Outline"),
            (cell_region_fill_color, "Tumor Cell Region"), 
            (dab_intersect_cell_border_color, "DAB+ in Tumor Cell Mask"),
            (hotspot_box_colors[0], "Hotspot Region (Example Color)"),
            ((255,0,0), "DAB+ StarDist Cells (in HS)"), # Blue (BGR)
            ((0,255,0), "DAB- StarDist Cells (in HS)")  # Green (BGR)
        ]
        for color, text in legend_items:
            ly = draw_legend_item(overlay_bgr, ly, color, text)

        final_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        logger.info("Overlay generation complete.")
        return final_overlay_rgb

    except Exception as e:
        logger.error(f"Error generating overlay: {e}", exc_info=True)
        return None


def format_image_for_save_rgba_or_bgr(
    image_input: np.ndarray,
    output_format: str = "BGR",
    target_alpha_value: float = 1.0
) -> np.ndarray | None:
    """Converts input image to uint8 and specified BGR or BGRA format."""
    if image_input is None:
        logger.warning("format_image_for_save_rgba_or_bgr: None input.")
        return None
    img_u8 = image_input
    if not (np.issubdtype(img_u8.dtype,np.uint8) and img_u8.dtype==np.uint8):
        if np.issubdtype(img_u8.dtype,np.floating):
            m,M=np.min(img_u8),np.max(img_u8)
            if m < -0.001 or M > 1.001: # If not approx in [0,1]
                img_u8 = (img_u8-m)/(M-m+1e-9) if not np.isclose(M,m) else (np.zeros_like(img_u8) if m<0.5 else np.ones_like(img_u8))
            img_u8 = np.clip(img_u8*255.,0,255).astype(np.uint8)
        elif np.issubdtype(img_u8.dtype,np.integer):
            m_val = img_u8.max()
            if m_val > 255 or img_u8.min() < 0: # If not in [0,255] for integer types
                img_u8 = np.clip((img_u8/(m_val/255. if m_val > 0 else 1.)),0,255).astype(np.uint8)
            else:
                img_u8 = img_u8.astype(np.uint8)
        else:
            logger.error(f"format_image_for_save_rgba_or_bgr: Unsupported dtype {img_u8.dtype}")
            return None

    alpha_byte = np.clip(int(target_alpha_value*255),0,255)

    if img_u8.ndim == 2 or (img_u8.ndim == 3 and img_u8.shape[2] == 1): # Grayscale or single-channel
        img_gray = img_u8[:,:,0] if img_u8.ndim == 3 else img_u8
        if output_format.upper() == "BGR":
            return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        elif output_format.upper() == "BGRA":
            bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            alpha_channel = np.full((img_gray.shape[0], img_gray.shape[1]), alpha_byte, dtype=np.uint8)
            return cv2.merge((bgr[:,:,0],bgr[:,:,1],bgr[:,:,2],alpha_channel))
    elif img_u8.ndim == 3 and img_u8.shape[2] == 3: # Assume RGB input
        if output_format.upper() == "BGR":
            return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        elif output_format.upper() == "BGRA":
            bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            alpha_channel = np.full((img_u8.shape[0], img_u8.shape[1]), alpha_byte, dtype=np.uint8)
            return cv2.merge((bgr[:,:,0],bgr[:,:,1],bgr[:,:,2],alpha_channel))
    elif img_u8.ndim == 3 and img_u8.shape[2] == 4: # Assume RGBA input
        if output_format.upper() == "BGR":
            return cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGR)
        elif output_format.upper() == "BGRA":
            # Convert RGBA to BGRA, then set the alpha channel
            bgra_converted = cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGRA)
            bgra_converted[:,:,3] = alpha_byte # Set new alpha
            return bgra_converted

    logger.error(f"format_image_for_save_rgba_or_bgr: Unsupported input shape {img_u8.shape} or output format {output_format}")
    return None


def create_line_overlay(image_rgb, labels, alpha=0.7, line_color=(0, 255, 0), line_thickness=1):
    """Creates a BGRA image with contours from 'labels' drawn over 'image_rgb'."""
    img_for_overlay = image_rgb.copy()
    # Ensure uint8 RGB
    if not (np.issubdtype(img_for_overlay.dtype, np.unsignedinteger) and img_for_overlay.dtype == np.uint8):
        if np.issubdtype(img_for_overlay.dtype, np.floating):
            img_for_overlay = np.clip(img_for_overlay * 255, 0, 255).astype(np.uint8)
        else:
            img_for_overlay = img_for_overlay.astype(np.uint8)

    if img_for_overlay.ndim == 2: # Grayscale to RGB
        img_for_overlay = cv2.cvtColor(img_for_overlay, cv2.COLOR_GRAY2RGB)
    elif img_for_overlay.ndim == 3 and img_for_overlay.shape[2] == 4: # RGBA to RGB
        img_for_overlay = img_for_overlay[:,:,:3]

    overlay_draw_rgb = np.ascontiguousarray(img_for_overlay) # Ensure C-contiguous for OpenCV

    if labels is None or labels.size == 0:
        logger.warning("create_line_overlay: Labels array is None or empty.")
        return format_image_for_save_rgba_or_bgr(overlay_draw_rgb, "BGRA", target_alpha_value=alpha)

    unique_labels = np.unique(labels[labels > 0])
    if len(unique_labels) == 0:
        logger.debug("create_line_overlay: No objects found in labels.")
        return format_image_for_save_rgba_or_bgr(overlay_draw_rgb, "BGRA", target_alpha_value=alpha)

    # Draw contours on the RGB image (OpenCV expects BGR for color, but will draw on RGB fine)
    # The line_color is (0,255,0) which is Green in RGB, Green in BGR.
    for label_id in unique_labels:
        mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_draw_rgb, contours, -1, line_color, line_thickness)

    # Convert the final RGB image (with lines) to BGRA
    final_bgra_output = format_image_for_save_rgba_or_bgr(overlay_draw_rgb, "BGRA", target_alpha_value=alpha)
    return final_bgra_output