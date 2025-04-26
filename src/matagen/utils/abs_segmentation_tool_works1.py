import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr
import shutil 
import torch
from scipy.interpolate import interp1d
from pyod.models.knn import KNN
from scipy.cluster import vq
from PIL import Image
from statsmodels.nonparametric.smoothers_lowess import lowess
import PIL
import os
import pandas as pd
from scipy.signal import savgol_filter
from matagen.utils.plot_data_extraction.plot_digitizer import PlotDigitizer
from matagen.utils.axis_alignment.utils import AxisAlignment
from matagen.utils.plot_data_extraction.SpatialEmbeddings.src.utils import transforms as my_transforms

# Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
checkpoints_base = os.path.join(project_root, 'checkpoints', 'axis_alignment')
checkpoints_plot_extract_base = os.path.join(project_root, 'checkpoints', 'plot_data_extraction')

# Configurations
# Axis Alignment Configuration
axis_align_opt = {
    # region detection
    "config_file": os.path.join(checkpoints_base, "fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py"),
    "checkpoint_file": os.path.join(checkpoints_base, "epoch_200.pth"),
    "refinement": True,
    # tick detection
    "cuda": False, # Or True if GPU available and desired
    "canvas_size": 1280,
    "mag_ratio": 1.5,
    "poly": False,
    "text_threshold": 0.1,
    "low_text": 0.5,
    "link_threshold": 0.7,
    "show_time": False,
    "refine": True,
    "trained_model": os.path.join(checkpoints_base, "craft_mlt_25k.pth"),
    "refiner_model": os.path.join(checkpoints_base, "craft_refiner_CTW1500.pth"),
    # tick recognition
    "workers": 0,
    "saved_model": os.path.join(checkpoints_base, "TPS-ResNet-BiLSTM-Attn.pth"),
    "batch_max_length": 25, "imgH": 32, "imgW": 100, "rgb": False,
    "character": "0123456789abcdefghijklmnopqrstuvwxyz", "sensitive": False, "PAD": True,
    "Transformation": "TPS", "FeatureExtraction": "ResNet", "SequenceModeling": "BiLSTM",
    "Prediction": "Attn", "num_fiducial": 20, "input_channel": 1,
    "output_channel": 512, "hidden_size": 256,
}

# Configuration for PlotDigitizer's Segmentation model (Spatial Embedding)
plot_extract_opt = {
    "cuda": False, 
    "display": False,
    "save": False, 
    "num_workers": 0,
    "checkpoint_path": os.path.join(checkpoints_plot_extract_base, "checkpoint_0999.pth"),
     "dataset": {
        'name': 'cityscapes',
        'kwargs': {
            'img_height': 256,     
            'img_width': 512,    
            'norm_mean': [0.485, 0.456, 0.406], 
            'norm_std': [0.229, 0.224, 0.225]  
        }
    },
    "model": {
        'name': 'branched_erfnet', 
        'kwargs': {
            'num_classes': [3, 1], 
        }
    }
}

####################################### Main functions #########################################

def recognize_text(img_path, use_gpu=False):
    """Recognize text using EasyOCR."""
    try:
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        return reader.readtext(img_path)
    except Exception as e:
        print(f"Error initializing EasyOCR or reading text: {e}")
        return []


def filter_legend_text(ocr_result):
    """
    Selective legend text filter that focuses on chemical/material names
    while ignoring wavelength information and numeric entries.
    
    Args:
        ocr_result: Results from EasyOCR containing (bbox, text, prob) tuples
        
    Returns:
        Tuple of (labels, bboxes) containing filtered legend labels and their bounding boxes
    """
    labels = []
    bboxes = []
    
    # Print all OCR results for debugging
    print(f"Total OCR results: {len(ocr_result)}")
    for i, (bbox, text, prob) in enumerate(ocr_result):
        print(f"  [{i}] '{text}' (confidence: {prob:.2f})")
    
    # Set minimum confidence threshold
    min_confidence = 0.3
    
    # Patterns to explicitly INCLUDE in legend labels
    import re
    include_patterns = [
        r'^p\d+',               
        r'^pentamer\s+[a-z]',    
        r'^polymer\s+[a-z]',    
        r'^oligomer\s+[a-z0-9]',
        r'^p[a-z]',             
        r'^prodot',          
        r'^pedot',               
        r'^compound\s+\d+',    
        r'^p\(.*\)',           
        r'^p\([^)]+\-[^)]+\)',  
    ]
    
    # Patterns to EXCLUDE
    exclude_patterns = [
        r'^[0-9]+$',            
        r'^[0-9\.]+$',           
        r'^[0-9]+\s*nm$',       
        r'^nm$',                
        r'^[a-z][\)\]}]$',      
        r'wavelength',         
        r'intensity',          
        r'absorbance',          
        r'transmittance',      
        r'^[0-9]',              
        r'00'                    
    ]
    
    # Process each OCR result
    for bbox, text, prob in ocr_result:
        text = text.strip()
        
        # Skip empty text or low confidence results
        if not text or prob < min_confidence:
            continue
            
        # First check if it matches any include patterns - these take precedence
        should_include = False
        for pattern in include_patterns:
            if re.search(pattern, text.lower()):
                should_include = True
                print(f"  Including based on pattern: '{text}'")
                break
                
        if should_include:
            labels.append(text)
            bboxes.append(bbox)
            continue
        
        should_exclude = False
        for pattern in exclude_patterns:
            if re.search(pattern, text.lower()):
                print(f"  Excluding based on pattern: '{text}'")
                should_exclude = True
                break
                
        if should_exclude:
            continue        
       
        if len(text) > 1:
            has_letters = any(c.isalpha() for c in text)
            not_just_unit = not (text.lower().endswith('nm') and len(text) <= 5)
            
            has_polymer_pattern = '(' in text and ')' in text and (
                text.startswith('P(') or 
                'DOT' in text or 
                '-' in text
            )
            
            if (has_letters and not_just_unit) or has_polymer_pattern:
                should_include = True
                print(f"  Including as possible chemical name: '{text}'")
        
        if should_include:
            labels.append(text)
            bboxes.append(bbox)
    
    print(f"Filtered legend labels: {labels}")
    return labels, bboxes

def get_label_colors(image_path, label_bbox):
    """Extract the color associated with a detected label bounding box."""
    sample_width = 20
    sample_height_factor = 0.5

    try:
        top_left, top_right, bottom_right, bottom_left = label_bbox
        text_height = bottom_left[1] - top_left[1]
        text_mid_y = top_left[1] + text_height / 2

        sample_y_delta = (text_height * sample_height_factor) / 2
        crop_left = max(0, top_left[0] - sample_width)
        crop_right = top_left[0]
        crop_top = int(text_mid_y - sample_y_delta)
        crop_bottom = int(text_mid_y + sample_y_delta)

        if crop_left >= crop_right or crop_top >= crop_bottom:
            print(f"Warning: Invalid crop box calculated for label color near {top_left}.")
            return None

        with Image.open(image_path) as im:
            im_rgb = im.convert("RGB")
            cropped_image = im_rgb.crop((crop_left, crop_top, crop_right, crop_bottom))

        cropped_image_rgb = np.array(cropped_image)

        mask_non_white = np.any(cropped_image_rgb < [250, 250, 250], axis=2)
        mask_non_black = np.any(cropped_image_rgb > [5, 5, 5], axis=2)
        mask = mask_non_white & mask_non_black

        pixels = cropped_image_rgb[mask]

        if pixels.size > 0:
            mean_color = np.mean(pixels, axis=0)
            return mean_color
        else:
            return None
    except Exception as e:
        print(f"Error getting label color: {e}")
        return None

def extract_legend_data(image_path):
    """Extracts legend labels and their associated colors."""
    print("Step 1: Performing OCR to find legend text...")
    ocr_results = recognize_text(image_path)
    if not ocr_results:
        print("Warning: OCR found no text.")
        return {}

    print(f"Step 2: Filtering OCR results for potential legend labels (found {len(ocr_results)} text boxes)...")
    labels, bboxes = filter_legend_text(ocr_results)
    if not labels:
        print("Warning: No potential legend labels found after filtering.")
        return {}
    print(f"  Found {len(labels)} potential legend labels: {labels}")

    print("Step 3: Extracting colors associated with labels...")
    legend_dict = {}
    for label, bbox in zip(labels, bboxes):
        color = get_label_colors(image_path, bbox)
        if color is not None:
            legend_dict[label] = color
            print(f"  Color for '{label}': {color.astype(int)}")
        else:
            print(f"  Could not determine color for '{label}'.")

    return legend_dict


def adaptive_color_tolerance(img_array, legend_dict, initial_tolerance=30, max_tolerance=70):
    """
    Adaptively determine color tolerance based on image characteristics.
    
    Args:
        img_array: RGB numpy array (0-255)
        legend_dict: Dictionary of {label: color}
        initial_tolerance: Initial tolerance value
        max_tolerance: Maximum tolerance value
    
    Returns:
        Dict mapping labels to tolerance values
    """
    # Analyze image contrast and color distribution
    
    if isinstance(img_array, PIL.Image.Image):
        img_array_np = np.array(img_array)
        gray = cv2.cvtColor(img_array_np, cv2.COLOR_RGB2BGR if img_array.mode == 'RGB' else cv2.COLOR_RGBA2BGR)
    else:
        # Original code for when img_array is already a numpy array
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    contrast = np.std(gray)
    
    # Base tolerance on contrast - lower contrast means higher tolerance needed
    base_tolerance = initial_tolerance + max(0, (50 - contrast)) * 0.5
    base_tolerance = min(base_tolerance, max_tolerance)
    
    print(f"Image contrast: {contrast:.2f}, Base tolerance: {base_tolerance:.2f}")
    
    tolerance_dict = {}
    for label, color in legend_dict.items():
        # Check how distinct this color is from others
        other_colors = [c for l, c in legend_dict.items() if l != label]
        if not other_colors:
            tolerance_dict[label] = base_tolerance
            continue
            
        min_distance = min(np.linalg.norm(color - other_color) for other_color in other_colors)
        
        # Adjust tolerance based on color distinctness
        if min_distance > 100:  # Very distinct color
            tolerance_dict[label] = base_tolerance * 0.8
        elif min_distance < 50:  # Similar to another color
            tolerance_dict[label] = base_tolerance * 0.6
        else:
            tolerance_dict[label] = base_tolerance
            
        print(f"Color tolerance for '{label}': {tolerance_dict[label]:.2f}")
        
    return tolerance_dict

def read_img(path):
  img = cv2.imread(path)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.

def write_img(img, path):
  img = (img * 255).astype(np.uint8)
  print(img.shape, img.dtype)
  Image.fromarray(img).save(path)

def dilate_image(image_path):
  img = read_img(image_path)
  img = np.abs(img - 1)
  print(img.mean())
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  img = cv2.dilate(img, kernel, iterations=1)
  img = np.abs(img - 1)
  write_img(img, image_path)
  return img

def visualize_step(img, title, cmap=None):
    plt.figure(figsize=(10, 6))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Increase window size and adjust polynomial order
def smooth_curve(coords, window_size=21, polyorder=3):
    """
    Apply Savitzky–Golay smoothing with larger window for smoother results.
    
    :param coords: List of (x, y) tuples.
    :param window_size: Savitzky–Golay window length (must be odd and larger for smoother curves).
    :param polyorder: Polynomial order for Savitzky–Golay (must be < window_size).
    :return: List of (x, smoothed_y) tuples.
    """
    if len(coords) <= window_size:
        return coords

    # Split input into x and y arrays
    x_vals, y_vals = zip(*coords)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Apply Savitzky–Golay filter with larger window
    smoothed_y = savgol_filter(y_vals, window_size, polyorder)
    
    # Return coordinated points directly without trimming
    return list(zip(x_vals, smoothed_y))


def sample_curve(coords, num_samples=200):
    """Sample the curve at regular x intervals to reduce noise and data size."""
    if len(coords) <= num_samples:
        return coords
        
    x_vals, y_vals = zip(*sorted(coords, key=lambda p: p[0]))
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # Create interpolation function
    interp_func = interp1d(x_vals, y_vals, kind='linear', bounds_error=False, 
                         fill_value=(y_vals[0], y_vals[-1]))
    
    # Create regular sampling of x values
    x_samples = np.linspace(min(x_vals), max(x_vals), num_samples)
    
    # Get corresponding y values
    y_samples = interp_func(x_samples)
    
    return list(zip(x_samples, y_samples))


def visualize_curve_deduplication(img_array, all_curves, deduplicated_curves):
    """
    Visualize before and after deduplication to verify results.
    """
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title(f"All Detected Curves ({len(all_curves)})")
    for color, points in all_curves.items():
        x, y = zip(*points)
        plt.scatter(x, y, s=1, label=f"RGB={color}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_array)
    plt.title(f"After Deduplication ({len(deduplicated_curves)})")
    for color, points in deduplicated_curves.items():
        x, y = zip(*points)
        plt.scatter(x, y, s=1, label=f"RGB={color}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def transform_coordinates(pixel_coords, axis_info, transform_y=True):
    """
    Transforms pixel coordinates to data coordinates using axis info.
    
    Args:
        pixel_coords: List of (x_pixel, y_pixel) tuples
        axis_info: Dict with axis info
        transform_y: Whether to transform Y coordinates or leave as pixels
        
    Returns:
        List of (x_data, y_data) tuples, or (x_data, y_pixel) if transform_y=False
    """
    if not pixel_coords:
        return []
    
    # Check X-axis info
    if not axis_info or 'x' not in axis_info or \
       not axis_info['x'].get('pixels') or not axis_info['x'].get('values') or \
       len(axis_info['x']['pixels']) < 2:
        print("Warning: X-axis information is missing or incomplete.")
        return None
        
    # Check Y-axis info if transforming Y
    if transform_y and ('y' not in axis_info or \
       not axis_info['y'].get('pixels') or not axis_info['y'].get('values') or \
       len(axis_info['y']['pixels']) < 2):
        print("Warning: Y-axis information is missing or incomplete. Will keep Y in pixel coordinates.")
        transform_y = False

    try:
        x_pixels_curve, y_pixels_curve = zip(*pixel_coords)
        x_pixels_curve = np.array(x_pixels_curve)
        y_pixels_curve = np.array(y_pixels_curve)

        # X-Axis Transformation
        x_tick_pixels = np.array(axis_info['x']['pixels'])
        x_tick_values = np.array(axis_info['x']['values'])

        # Sort X ticks by pixel position
        sort_idx_x = np.argsort(x_tick_pixels)
        x_tick_pixels_sorted = x_tick_pixels[sort_idx_x]
        x_tick_values_sorted = x_tick_values[sort_idx_x]

        # Create interpolation function for X
        interp_x = interp1d(x_tick_pixels_sorted, x_tick_values_sorted, kind='linear', 
                           fill_value="extrapolate", bounds_error=False)
        x_data = interp_x(x_pixels_curve)
        
        # Y-Axis Transformation (if requested)
        if transform_y:
            y_tick_pixels = np.array(axis_info['y']['pixels'])
            y_tick_values = np.array(axis_info['y']['values'])
            
            # Sort Y ticks by pixel position
            sort_idx_y = np.argsort(y_tick_pixels)
            y_tick_pixels_sorted = y_tick_pixels[sort_idx_y]
            y_tick_values_sorted = y_tick_values[sort_idx_y]
            
            # Create interpolation function for Y
            interp_y = interp1d(y_tick_pixels_sorted, y_tick_values_sorted, kind='linear',
                              fill_value="extrapolate", bounds_error=False)
            y_data = interp_y(y_pixels_curve)
            
            # Combine transformed coordinates
            transformed_coords = list(zip(x_data, y_data))
        else:
            # Keep Y as pixel coordinates
            transformed_coords = list(zip(x_data, y_pixels_curve))
            
        return transformed_coords

    except Exception as e:
        print(f"Error during coordinate transformation: {e}")
        return None


def process_and_enhance_image(
    img_rgb,                      # Input RGB image (float 0-1 or uint8 0-255)
    seg_map,                      # Segmentation mask (boolean or 0/255, same HxW as img_rgb)
    # Thickening parameters
    dilation_kernel_size=(2, 2),
    dilation_iterations=5,
    # Enhancement parameters
    clahe_clip_limit=3.0,         # Adjust contrast (try 2.0-5.0)
    clahe_tile_grid_size=(8, 8),
    saturation_boost=2.8,         # Adjust color intensity (try 1.5-3.0)
    brightness_boost=1.2          # Adjust brightness (try 1.0-1.5)
):
    """
    Applies mask, thickens lines using invert-dilate-invert, enhances
    contrast (CLAHE), and boosts brightness/saturation (HSV).

    Args:
        img_rgb: Input RGB image.
        seg_map: Segmentation map.
        dilation_kernel_size: Kernel size for thickening.
        dilation_iterations: Iterations for thickening.
        clahe_clip_limit: Clip limit for CLAHE contrast.
        clahe_tile_grid_size: Tile grid size for CLAHE.
        saturation_boost: Multiplicative factor for saturation boost.
        brightness_boost: Multiplicative factor for brightness/value boost.

    Returns:
        Fully processed and enhanced image (numpy array, uint8 0-255).
        Returns None if seg_map is None.
    """
    # 1. Ensure input is uint8 [0-255] RGB
    if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
        if img_rgb.max() <= 1.0:
             img_255 = (img_rgb * 255).astype(np.uint8)
        else: # Assume it's already 0-255 float
             img_255 = img_rgb.astype(np.uint8)
    elif img_rgb.dtype == np.uint8:
        img_255 = img_rgb
    else:
        raise ValueError("Unsupported image dtype. Use float (0-1) or uint8 (0-255).")

    if img_255.ndim == 2: # Grayscale input
        img_255 = cv2.cvtColor(img_255, cv2.COLOR_GRAY2RGB)
    elif img_255.shape[2] != 3:
         raise ValueError("Input image must be RGB (3 channels).")

    visualize_step(img_255, "1. Original Image")

    # --- Mask Application ---
    if seg_map is None:
        print("Warning: No segmentation map provided.")
        return None

    # Ensure seg_map is boolean and correct shape
    mask_boolean = seg_map.astype(bool)
    if mask_boolean.shape[0] != img_255.shape[0] or mask_boolean.shape[1] != img_255.shape[1]:
         raise ValueError("Segmentation map dimensions must match image dimensions.")
    if mask_boolean.ndim == 3 and mask_boolean.shape[2] == 1:
        mask_boolean = np.squeeze(mask_boolean, axis=2) # Make HxW if HxWx1
    elif mask_boolean.ndim != 2:
         raise ValueError("Mask must be HxW or HxWx1.")

    # Apply the mask: Create a white background and copy foreground pixels
    masked_img = np.full_like(img_255, (255, 255, 255), dtype=np.uint8)
    masked_img[mask_boolean] = img_255[mask_boolean]
    visualize_step(masked_img, "2. Image with Mask Applied")
    current_img = masked_img

    # --- Thickening Lines (Invert-Dilate-Invert) ---
    if dilation_iterations > 0 and dilation_kernel_size[0] > 0 and dilation_kernel_size[1] > 0:
        print(f"Applying Invert-Dilate-Invert...")
        img_inverted = 255 - current_img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        img_dilated_inverted = cv2.dilate(img_inverted, kernel, iterations=dilation_iterations)
        thickened_img = 255 - img_dilated_inverted
        visualize_step(thickened_img, "3. Thickened Image")
        current_img = thickened_img
    # --- End Thickening ---

    # --- Contrast Enhancement (CLAHE on L channel) ---
    print("Applying CLAHE...")
    lab = cv2.cvtColor(current_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_contrast_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    visualize_step(enhanced_contrast_img, "4. After CLAHE Enhancement")
    current_img = enhanced_contrast_img

    # --- Color Intensity & Brightness Boost (HSV) ---
    print("Applying Brightness/Saturation Boost...")
    hsv = cv2.cvtColor(current_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Boost Saturation (S channel)
    if saturation_boost != 1.0:
        s_boosted = np.clip(s * saturation_boost, 0, 255).astype(np.uint8)
    else:
        s_boosted = s

    # Boost Brightness/Value (V channel)
    if brightness_boost != 1.0:
         v_boosted = np.clip(v * brightness_boost, 0, 255).astype(np.uint8)
    else:
         v_boosted = v

    # Merge boosted channels
    final_hsv = cv2.merge((h, s_boosted, v_boosted))
    final_img_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    visualize_step(final_img_rgb, "5. After HSV Boost")

    # --- Re-apply Mask (Recommended) ---
    # Ensures background stays perfectly white after enhancements
    print("Re-applying mask...")
    background_mask = ~mask_boolean # Use the original mask
    final_img_rgb[background_mask] = [255, 255, 255]
    visualize_step(final_img_rgb, "6. Final Image (Mask Re-Applied)")

    return final_img_rgb
#############################################################################################
# --- testing ------

def get_curve_average_colors(img_array, curve_points_dict):
    """
    Calculate the average color of each curve by sampling the original image at curve point locations.
    
    Args:
        img_array: Original RGB image (numpy array, 0-255 values)
        curve_points_dict: Dictionary mapping curve identifiers to lists of (x,y) points
        
    Returns:
        Dictionary mapping curve identifiers to their average RGB colors
    """
    curve_colors = {}
    
    # Ensure image array is in the correct format
    if isinstance(img_array, PIL.Image.Image):
        img_np = np.array(img_array)
    else:
        img_np = img_array.copy()
    
    # Convert to uint8 if needed
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    # Get height and width of image
    height, width = img_np.shape[:2]
    
    for curve_id, points in curve_points_dict.items():
        # Sample pixels at curve points
        valid_colors = []
        for x, y in points:
            # Ensure coordinates are integers and within image bounds
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < width and 0 <= y_int < height:
                pixel_color = img_np[y_int, x_int]
                
                # Skip white pixels (likely background)
                if not np.all(pixel_color > 240):
                    valid_colors.append(pixel_color)
        
        # Calculate average color if we have valid samples
        if valid_colors:
            avg_color = np.mean(valid_colors, axis=0).astype(int)
            curve_colors[curve_id] = tuple(avg_color)
            print(f"Curve {curve_id}: Average color = {tuple(avg_color)}")
        else:
            print(f"Warning: No valid color samples for curve {curve_id}")
    
    return curve_colors

def improved_match_legend_to_curves(legend_dict, valid_curves, img_array):
    """
    Match legend entries to curves using the actual average colors of curve points.
    
    Args:
        legend_dict: Dictionary mapping legend labels to detected legend colors
        valid_curves: Dictionary mapping curve IDs to curve points
        img_array: Original RGB image array
        
    Returns:
        Dictionary mapping legend labels to matched curve points
    """
    # Calculate the average color of each curve
    curve_avg_colors = get_curve_average_colors(img_array, valid_curves)
    
    # Create visualization of the average colors
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title("Legend Colors")
    for i, (label, color) in enumerate(legend_dict.items()):
        plt.bar(i, 1, color=[c/255 for c in color], label=label)
    plt.xticks(range(len(legend_dict)), legend_dict.keys(), rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.title("Curve Average Colors")
    for i, (curve_id, color) in enumerate(curve_avg_colors.items()):
        plt.bar(i, 1, color=[c/255 for c in color], label=str(curve_id))
    plt.xticks(range(len(curve_avg_colors)), curve_avg_colors.keys(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Score all possible matches
    all_matches = []
    for label, legend_color in legend_dict.items():
        for curve_id, curve_color in curve_avg_colors.items():
            # Calculate color similarity score
            color_distance = np.sqrt(np.sum((np.array(legend_color) - np.array(curve_color))**2))
            max_distance = np.sqrt(3 * 255**2)
            similarity = 1.0 - (color_distance / max_distance)
            
            # Add curve quality factor (more points = higher quality)
            point_count = len(valid_curves[curve_id])
            quality_factor = min(1.0, point_count / 1000)
            
            # Weight: 90% color, 10% quality
            match_score = 0.9 * similarity + 0.1 * quality_factor
            
            all_matches.append((label, curve_id, match_score, similarity, curve_color))
    
    # Sort by score (highest first)
    all_matches.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching
    matched_curves = {}
    used_curves = set()
    
    for label, curve_id, score, color_similarity, curve_color in all_matches:
        # Skip if this label or curve is already matched
        if label in matched_curves or curve_id in used_curves:
            continue
        
        # Require a minimum score
        if score > 0.3:  # Adjustable threshold
            matched_curves[label] = valid_curves[curve_id]
            used_curves.add(curve_id)
            print(f"Matched '{label}' (legend color {legend_dict[label]}) to curve {curve_id}")
            print(f"  (avg color {curve_color}, score: {score:.3f}, color similarity: {color_similarity:.3f})")
    
    # Handle unmatched items if needed (force matching)
    unmatched_labels = set(legend_dict.keys()) - set(matched_curves.keys())
    unmatched_curves = set(curve_avg_colors.keys()) - used_curves
    
    if unmatched_labels and unmatched_curves:
        print(f"Attempting to match {len(unmatched_labels)} remaining labels to {len(unmatched_curves)} unused curves...")
        
        # Create all possible remaining matches
        remaining_matches = []
        for label in unmatched_labels:
            legend_color = legend_dict[label]
            for curve_id in unmatched_curves:
                curve_color = curve_avg_colors[curve_id]
                color_distance = np.sqrt(np.sum((np.array(legend_color) - np.array(curve_color))**2))
                max_distance = np.sqrt(3 * 255**2)
                similarity = 1.0 - (color_distance / max_distance)
                remaining_matches.append((label, curve_id, similarity))
        
        # Sort and assign
        remaining_matches.sort(key=lambda x: x[2], reverse=True)
        for label, curve_id, similarity in remaining_matches:
            if label in unmatched_labels and curve_id in unmatched_curves:
                matched_curves[label] = valid_curves[curve_id]
                unmatched_labels.remove(label)
                unmatched_curves.remove(curve_id)
                print(f"Force-matched '{label}' to curve {curve_id} (similarity: {similarity:.3f})")
    
    return matched_curves

def remove_outliers_robust(coords, window_size=5, std_threshold=2.0):
    """More robust outlier removal with two-pass approach."""
    if len(coords) <= window_size:
        return coords
        
    # Sort by x-coordinate
    sorted_coords = sorted(coords, key=lambda p: p[0])
    x_vals, y_vals = zip(*sorted_coords)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # First pass - detect outliers using moving median
    filtered_indices = []
    for i in range(len(y_vals)):
        start = max(0, i - window_size // 2)
        end = min(len(y_vals), i + window_size // 2 + 1)
        window = y_vals[start:end]
        median = np.median(window)
        mad = np.median(np.abs(window - median))  # Median Absolute Deviation
        
        # Use MAD as a robust measure of deviation
        if mad == 0:
            filtered_indices.append(i)
        elif abs(y_vals[i] - median) <= std_threshold * mad * 1.4826:  # Approximate relationship to std dev
            filtered_indices.append(i)
    
    # Keep only non-outlier points
    filtered_coords = [(x_vals[i], y_vals[i]) for i in filtered_indices]
    
    # Second pass - smoothing with Savitzky-Golay
    if len(filtered_coords) > window_size:
        x_filtered, y_filtered = zip(*filtered_coords)
        x_filtered = np.array(x_filtered)
        y_filtered = np.array(y_filtered)
        
        # Apply Savitzky-Golay filter
        y_smoothed = savgol_filter(y_filtered, min(window_size, len(y_filtered)-2), 3)
        
        # Return smoothed result
        return list(zip(x_filtered, y_smoothed))
    else:
        return filtered_coords

def match_legend_to_curves(legend_dict, valid_curves, enhanced_img):
    """
    Advanced color-based matching of legend entries to detected curves,
    specifically optimized for spectral plots.
    
    Args:
        legend_dict: Dictionary of {label: color}
        valid_curves: Dictionary of {color: points}
        enhanced_img: Enhanced RGB image
        
    Returns:
        Dictionary mapping labels to curve points
    """
    matched_curves = {}
    used_curves = set()
    
    # 1. Calculate detailed color metrics for each curve
    curve_metrics = {}
    for curve_color, points in valid_curves.items():
        # Get curve shape characteristics
        x_vals, y_vals = zip(*sorted(points, key=lambda p: p[0]))
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        # Calculate peak positions
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(y_vals, height=y_min + 0.5*(y_max-y_min), distance=len(y_vals)//10)
        peak_positions = [x_vals[p] for p in peaks] if peaks.size > 0 else []
        
        # Store metrics
        curve_metrics[curve_color] = {
            'points': points,
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'peak_positions': peak_positions,
            'point_count': len(points)
        }
    
    # 2. Extract wavelength info from legend labels if available
    label_wavelengths = {}
    import re
    for label, color in legend_dict.items():
        # Look for wavelength patterns like "420 nm" or ranges like "410-450 nm"
        wavelengths = []
        matches = re.findall(r'(\d+)(?:\s*(?:nm|nmi))', label)
        if matches:
            wavelengths = [int(w) for w in matches]
        
        label_wavelengths[label] = wavelengths
    
    # 3. Define advanced scoring function
    def score_match(target_color, curve_color, curve_data, label=None, wavelengths=None):
        # Base color similarity (highest priority)
        color_distance = np.sqrt(np.sum((np.array(curve_color) - np.array(target_color))**2))
        max_distance = np.sqrt(3 * 255**2)
        color_similarity = 1 - (color_distance / max_distance)
        
        # Start with pure color match
        match_score = color_similarity
        
        # If we have wavelength information, boost score for curves that include those wavelengths
        if wavelengths and len(wavelengths) > 0:
            x_min, x_max = curve_data['x_range']
            wavelength_bonus = 0
            
            # Check if the curve's x-range contains the wavelengths mentioned in the label
            for wl in wavelengths:
                if x_min <= wl <= x_max:
                    # Higher bonus for curves that have peaks near the mentioned wavelengths
                    if curve_data['peak_positions']:
                        closest_peak = min(curve_data['peak_positions'], key=lambda p: abs(p - wl))
                        proximity = 1 - min(abs(closest_peak - wl) / 50, 1.0)  # Within 50 nm is good
                        wavelength_bonus += 0.05 * proximity
                    else:
                        wavelength_bonus += 0.02  # Smaller bonus if no peaks detected
            
            # Apply wavelength bonus (small effect, 5% max)
            match_score *= (1 + wavelength_bonus)
        
        # Small bonus for curves with more points (indication of quality)
        point_count = curve_data['point_count']
        quality_factor = min(1.0, point_count / 1000)
        
        # Final score: 95% color, 5% quality
        final_score = 0.95 * match_score + 0.05 * quality_factor
        
        return final_score
    
    # 4. First pass: Score all possible matches
    all_scores = []
    for label, target_color in legend_dict.items():
        wavelengths = label_wavelengths.get(label, [])
        
        for curve_color, curve_data in curve_metrics.items():
            score = score_match(target_color, curve_color, curve_data, label, wavelengths)
            all_scores.append((label, curve_color, score))
    
    # 5. Sort by score (highest first)
    all_scores.sort(key=lambda x: x[2], reverse=True)
    
    # 6. Greedy matching: assign the highest-scoring valid matches first
    for label, curve_color, score in all_scores:
        # Skip if this label already has a match or this curve is already used
        if label in matched_curves or curve_color in used_curves:
            continue
        
        # Only match if the score is reasonable
        if score > 0.25:  # Lower threshold for challenging matches
            matched_curves[label] = curve_metrics[curve_color]['points']
            used_curves.add(curve_color)
            print(f"Matched '{label}' to curve {curve_color} (score: {score:.3f})")
    
    # 7. Final check to ensure all legend items have matches
    unmatched_labels = set(legend_dict.keys()) - set(matched_curves.keys())
    unmatched_curves = set(curve_metrics.keys()) - used_curves
    
    if unmatched_labels and unmatched_curves:
        print(f"Attempting to match {len(unmatched_labels)} remaining labels to {len(unmatched_curves)} unused curves...")
        
        # Force-match remaining labels to remaining curves
        remaining_scores = []
        for label in unmatched_labels:
            target_color = legend_dict[label]
            wavelengths = label_wavelengths.get(label, [])
            
            for curve_color in unmatched_curves:
                curve_data = curve_metrics[curve_color]
                score = score_match(target_color, curve_color, curve_data, label, wavelengths)
                remaining_scores.append((label, curve_color, score))
        
        # Sort and assign remaining matches
        remaining_scores.sort(key=lambda x: x[2], reverse=True)
        while unmatched_labels and unmatched_curves and remaining_scores:
            label, curve_color, score = remaining_scores.pop(0)
            
            if label in unmatched_labels and curve_color in unmatched_curves:
                matched_curves[label] = curve_metrics[curve_color]['points']
                unmatched_labels.remove(label)
                unmatched_curves.remove(curve_color)
                print(f"Force-matched '{label}' to curve {curve_color} (score: {score:.3f})")
    
    # Final results summary
    if len(matched_curves) < len(legend_dict):
        print(f"WARNING: Only matched {len(matched_curves)}/{len(legend_dict)} legend items!")
    else:
        print(f"SUCCESS: Matched all {len(legend_dict)} legend items!")
    
    return matched_curves


def derivative(x_data, y_data):
    """
    Calculate derivatives with error handling for zero differences and duplicate points
    
    Args:
        x_data: List or array of x coordinates
        y_data: List or array of y coordinates
        
    Returns:
        (x_prim, y_prim) - arrays of derivative coordinates
    """
    N = len(x_data)
    if N < 2:
        return [], []  # Return empty lists if not enough points
    
    delta_x = []
    x_prim = []
    y_prim = []
    
    for i in range(N - 1):
        dx = x_data[i+1] - x_data[i]
        
        # Skip points with same x-coordinate (would cause division by zero)
        if abs(dx) < 1e-10:
            continue
            
        delta_x.append(dx)
        x_prim.append((x_data[i+1] + x_data[i]) / 2.0)
        y_prim.append((y_data[i+1] - y_data[i]) / dx)
    
    return x_prim, y_prim


# def deduplicate_curves(valid_curves, similarity_threshold=0.8):
#     """
#     Remove redundant curves that are likely representing the same data series.
    
#     Args:
#         valid_curves: Dictionary mapping colors to curve points
#         similarity_threshold: Threshold for considering curves as duplicates
        
#     Returns:
#         Filtered dictionary with redundant curves removed
#     """
#     # Sort curves by number of points (typically want to keep the one with more points)
#     curve_items = sorted(valid_curves.items(), key=lambda x: len(x[1]), reverse=True)
    
#     # Keep track of which curves to retain
#     curves_to_keep = []
#     redundant_curves = []
    
#     for i, (color1, points1) in enumerate(curve_items):
#         # If we've already marked this curve as redundant, skip
#         if (color1, points1) in redundant_curves:
#             continue
            
#         curves_to_keep.append((color1, points1))
        
#         # Compare with remaining curves
#         for color2, points2 in curve_items[i+1:]:
#             if (color2, points2) in redundant_curves:
#                 continue
                
#             # Calculate similarity between curves
#             similarity = calculate_curve_similarity(points1, points2)
            
#             if similarity > similarity_threshold:
#                 print(f"Found redundant curve: {color2} (similarity: {similarity:.2f})")
#                 redundant_curves.append((color2, points2))
    
#     # Convert back to dictionary
#     filtered_curves = {color: points for color, points in curves_to_keep}
#     print(f"Reduced from {len(valid_curves)} to {len(filtered_curves)} unique curves")
    
#     return filtered_curves

def deduplicate_curves(valid_curves, similarity_threshold=0.8):
    """
    Remove redundant curves that likely represent the same data series using
    spectral characteristics including peak positions, peak heights, and half-width.
    
    Args:
        valid_curves: Dictionary mapping colors to curve points
        similarity_threshold: Threshold for considering curves as duplicates
        
    Returns:
        Filtered dictionary with redundant curves removed
    """
    # Sort curves by number of points (typically want to keep the one with more points)
    curve_items = sorted(valid_curves.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Extract spectral characteristics for each curve
    curve_characteristics = {}
    
    for color, points in curve_items:
        # Sort points by x-coordinate
        sorted_points = sorted(points, key=lambda p: p[0])
        x_values, y_values = zip(*sorted_points)
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        
        # Find peaks
        try:
            from scipy.signal import find_peaks
            peaks, peak_props = find_peaks(y_values, height=0.25, distance=len(y_values)//10)
            
            if len(peaks) == 0:
                # Try again with more relaxed parameters if no peaks found
                peaks, peak_props = find_peaks(y_values, height=0, distance=len(y_values)//20)
            
            peak_positions = [x_values[p] for p in peaks]
            peak_heights = [y_values[p] for p in peaks]
            
            # Calculate half-width for each peak
            half_widths = []
            for peak_idx in peaks:
                peak_height = y_values[peak_idx]
                half_height = peak_height / 2
                
                # Find left half-width point
                left_idx = peak_idx
                while left_idx > 0 and y_values[left_idx] > half_height:
                    left_idx -= 1
                
                # Find right half-width point
                right_idx = peak_idx
                while right_idx < len(y_values) - 1 and y_values[right_idx] > half_height:
                    right_idx += 1
                
                # Calculate half-width
                if left_idx < peak_idx < right_idx:
                    half_width = x_values[right_idx] - x_values[left_idx]
                    half_widths.append(half_width)
                else:
                    half_widths.append(None)  # Unable to calculate
            
            # Store characteristics
            curve_characteristics[color] = {
                'points': points,
                'peak_positions': peak_positions,
                'peak_heights': peak_heights,
                'half_widths': half_widths,
                'x_range': (min(x_values), max(x_values)),
                'y_range': (min(y_values), max(y_values))
            }
            
            print(f"Curve {color}: Found {len(peaks)} peaks at x={peak_positions}")
            if half_widths:
                print(f"  Half-widths: {[round(hw, 2) if hw else None for hw in half_widths]}")
                
        except Exception as e:
            print(f"Error analyzing peaks for curve {color}: {e}")
            # Store basic info if peak analysis fails
            curve_characteristics[color] = {
                'points': points,
                'peak_positions': [],
                'peak_heights': [],
                'half_widths': [],
                'x_range': (min(x_values), max(x_values)),
                'y_range': (min(y_values), max(y_values))
            }
    
    # Function to calculate comprehensive similarity between curves
    def calculate_comprehensive_similarity(char1, char2):
        # 1. Calculate point-based similarity (RMSE of interpolated points)
        try:
            from scipy.interpolate import interp1d
            
            # Sort points by x-coordinate
            points1 = sorted(char1['points'], key=lambda p: p[0]) 
            points2 = sorted(char2['points'], key=lambda p: p[0])
            
            # Extract x and y values
            x1, y1 = zip(*points1)
            x2, y2 = zip(*points2)
            
            # Find overlap region
            x_min = max(min(x1), min(x2))
            x_max = min(max(x1), max(x2))
            
            if x_max <= x_min:
                return 0.0  # No overlap
            
            # Create interpolation functions
            f1 = interp1d(x1, y1, bounds_error=False, fill_value="extrapolate")
            f2 = interp1d(x2, y2, bounds_error=False, fill_value="extrapolate")
            
            # Sample at regular intervals in overlap region
            num_samples = 50
            x_samples = np.linspace(x_min, x_max, num_samples)
            y1_samples = f1(x_samples)
            y2_samples = f2(x_samples)
            
            # Calculate normalized RMSE
            rmse = np.sqrt(np.mean((y1_samples - y2_samples)**2))
            y_range = max(max(y1) - min(y1), max(y2) - min(y2))
            if y_range == 0:
                y_range = 1.0
            
            point_similarity = 1.0 - min(rmse / y_range, 1.0)
        except Exception as e:
            print(f"Error in point similarity calculation: {e}")
            point_similarity = 0.0
        
        # 2. Calculate peak similarity
        peak_similarity = 0.0
        if char1['peak_positions'] and char2['peak_positions']:
            try:
                # Match peaks between the two curves
                matched_peaks = 0
                total_peaks = max(len(char1['peak_positions']), len(char2['peak_positions']))
                
                for pos1, height1 in zip(char1['peak_positions'], char1['peak_heights']):
                    # Find closest peak in the second curve
                    if char2['peak_positions']:
                        closest_idx = np.argmin([abs(pos1 - pos2) for pos2 in char2['peak_positions']])
                        pos2 = char2['peak_positions'][closest_idx]
                        height2 = char2['peak_heights'][closest_idx]
                        
                        # Check if peaks are close enough in position and height
                        pos_diff = abs(pos1 - pos2)
                        height_diff = abs(height1 - height2) / max(height1, height2)
                        
                        # Consider peaks matched if they're within 5% of x-range and 20% of height
                        x_range = max(char1['x_range'][1] - char1['x_range'][0], 
                                     char2['x_range'][1] - char2['x_range'][0])
                        if pos_diff < 0.05 * x_range and height_diff < 0.2:
                            matched_peaks += 1
                
                peak_similarity = matched_peaks / total_peaks if total_peaks > 0 else 0.0
            except Exception as e:
                print(f"Error in peak similarity calculation: {e}")
                peak_similarity = 0.0
        
        # 3. Calculate half-width similarity
        width_similarity = 0.0
        if char1['half_widths'] and char2['half_widths']:
            try:
                # Compare the average half-widths
                avg_width1 = np.mean([w for w in char1['half_widths'] if w is not None])
                avg_width2 = np.mean([w for w in char2['half_widths'] if w is not None])
                
                if avg_width1 > 0 and avg_width2 > 0:
                    width_ratio = min(avg_width1, avg_width2) / max(avg_width1, avg_width2)
                    width_similarity = width_ratio
                else:
                    width_similarity = 0.0
            except Exception as e:
                print(f"Error in width similarity calculation: {e}")
                width_similarity = 0.0
        
            peak_dissimilarity = calculate_peak_dissimilarity(char1, char2)

            # Original weighted combination
            initial_similarity = 0.6 * point_similarity + 0.3 * peak_similarity + 0.1 * width_similarity

            # Apply a penalty for curves with significantly different peak positions
            peak_position_factor = 1.0 - (0.7 * peak_dissimilarity)  # Higher dissimilarity = lower factor
            combined_similarity = initial_similarity * peak_position_factor
        
        return combined_similarity
    
    # Keep track of which curves to retain
    curves_to_keep = []
    redundant_curves = []
    
    for i, (color1, points1) in enumerate(curve_items):
        # If we've already marked this curve as redundant, skip
        if (color1, points1) in redundant_curves:
            continue
            
        curves_to_keep.append((color1, points1))
        
        # Compare with remaining curves
        for color2, points2 in curve_items[i+1:]:
            if (color2, points2) in redundant_curves:
                continue
                
            # Calculate comprehensive similarity
            char1 = curve_characteristics[color1]
            char2 = curve_characteristics[color2]
            similarity = calculate_comprehensive_similarity(char1, char2)
            
            print(f"Similarity between {color1} and {color2}: {similarity:.3f}")
            
            if similarity > similarity_threshold:
                print(f"Found redundant curve: {color2} (similarity: {similarity:.3f})")
                print(f"  Primary curve: peaks at {char1['peak_positions']}")
                print(f"  Redundant curve: peaks at {char2['peak_positions']}")
                redundant_curves.append((color2, points2))
    
    # Convert back to dictionary
    filtered_curves = {color: points for color, points in curves_to_keep}
    print(f"Reduced from {len(valid_curves)} to {len(filtered_curves)} unique curves")
    
    return filtered_curves


def calculate_peak_dissimilarity(char1, char2):
    """
    Calculate how dissimilar two curves are based on peak positions.
    Returns high value if peaks are in different positions.
    """
    # If either curve has no peaks, they can't be compared effectively
    if not char1['peak_positions'] or not char2['peak_positions']:
        return 0.5  # Neutral value
        
    # Find the minimum distance between any pair of peaks
    min_peak_distance = float('inf')
    for peak1 in char1['peak_positions']:
        for peak2 in char2['peak_positions']:
            distance = abs(peak1 - peak2)
            min_peak_distance = min(min_peak_distance, distance)
    
    # Calculate dissimilarity - high value means very different peak positions
    # Scale the distance: 0 = same position, 1 = peaks far apart
    x_range = max(
        char1['x_range'][1] - char1['x_range'][0],
        char2['x_range'][1] - char2['x_range'][0]
    )
    
    # Normalize by 10% of x-range
    normalized_distance = min(min_peak_distance / (0.1 * x_range), 1.0)
    
    return normalized_distance


def enhanced_match_legend_to_curves(legend_dict, valid_curves, enhanced_img):
    """
    Match legend entries to detected curves using multiple criteria.
    """
    matched_curves = {}
    used_curves = set()
    
    # 1. Analyze curves
    curve_properties = {}
    for color, points in valid_curves.items():
        # Sort by x-coordinate
        points_sorted = sorted(points, key=lambda p: p[0])
        x_vals, y_vals = zip(*points_sorted)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(y_vals, height=0.5*max(y_vals), distance=len(y_vals)//10)
        
        # Store key properties
        curve_properties[color] = {
            'points': points,
            'num_peaks': len(peaks),
            'peak_positions': [x_vals[p] for p in peaks],
            'peak_heights': [y_vals[p] for p in peaks],
            'x_range': (min(x_vals), max(x_vals)),
            'y_range': (min(y_vals), max(y_vals)),
        }
    
    # 2. Define scoring function that uses both color and curve properties
    def calculate_match_score(legend_color, curve_color, curve_data):
        # Color similarity (primary criterion)
        color_distance = np.sqrt(np.sum((np.array(curve_color) - np.array(legend_color))**2))
        max_distance = np.sqrt(3 * 255**2)
        color_similarity = 1.0 - (color_distance / max_distance)
        
        # Number of points (quality factor)
        point_count = len(curve_data['points'])
        quality_factor = min(1.0, point_count / 1000)
        
        # Final score: 90% color, 10% quality
        match_score = 0.9 * color_similarity + 0.1 * quality_factor
        
        return match_score
    
    # 3. First pass: Calculate scores for all potential matches
    all_matches = []
    for label, legend_color in legend_dict.items():
        for curve_color, curve_data in curve_properties.items():
            score = calculate_match_score(legend_color, curve_color, curve_data)
            all_matches.append((label, curve_color, score))
    
    # 4. Sort by score and apply greedy matching
    all_matches.sort(key=lambda x: x[2], reverse=True)
    
    # Process matches in order of confidence
    for label, curve_color, score in all_matches:
        # Skip if this label already has a match or this curve is already used
        if label in matched_curves or curve_color in used_curves:
            continue
        
        # Only match if the score is reasonable
        if score > 0.3:  # Lower threshold for challenging matches
            matched_curves[label] = curve_properties[curve_color]['points']
            used_curves.add(curve_color)
            print(f"Matched '{label}' to curve {curve_color} (score: {score:.3f})")
    
    # 5. Handle unmatched items (force matches if needed)
    unmatched_labels = set(legend_dict.keys()) - set(matched_curves.keys())
    unmatched_curves = set(curve_properties.keys()) - used_curves
    
    if unmatched_labels and unmatched_curves:
        print(f"Force-matching {len(unmatched_labels)} remaining labels to {len(unmatched_curves)} unused curves...")
        
        # Create all possible remaining matches
        remaining_matches = []
        for label in unmatched_labels:
            legend_color = legend_dict[label]
            for curve_color in unmatched_curves:
                curve_data = curve_properties[curve_color]
                score = calculate_match_score(legend_color, curve_color, curve_data)
                remaining_matches.append((label, curve_color, score))
        
        # Sort and assign
        remaining_matches.sort(key=lambda x: x[2], reverse=True)
        for label, curve_color, score in remaining_matches:
            if label in unmatched_labels and curve_color in unmatched_curves:
                matched_curves[label] = curve_properties[curve_color]['points']
                unmatched_labels.remove(label)
                unmatched_curves.remove(curve_color)
                print(f"Force-matched '{label}' to curve {curve_color} (score: {score:.3f})")
    
    return matched_curves

def calculate_curve_similarity(points1, points2, num_samples=50):
    """
    Calculate similarity between two curves using sampled points.
    
    Args:
        points1, points2: Lists of (x, y) coordinates
        num_samples: Number of x positions to compare
        
    Returns:
        Similarity score (0-1, higher means more similar)
    """
    # Sort by x coordinate
    points1 = sorted(points1, key=lambda p: p[0])
    points2 = sorted(points2, key=lambda p: p[0])
    
    # Get x range overlap
    x1_min, x1_max = points1[0][0], points1[-1][0]
    x2_min, x2_max = points2[0][0], points2[-1][0]
    
    overlap_min = max(x1_min, x2_min)
    overlap_max = min(x1_max, x2_max)
    
    if overlap_max <= overlap_min:
        return 0.0  # No overlap
    
    # Create interpolation functions
    x1, y1 = zip(*points1)
    x2, y2 = zip(*points2)
    
    from scipy.interpolate import interp1d
    interp1 = interp1d(x1, y1, bounds_error=False, fill_value="extrapolate")
    interp2 = interp1d(x2, y2, bounds_error=False, fill_value="extrapolate")
    
    # Sample at regular intervals in overlap region
    x_samples = np.linspace(overlap_min, overlap_max, num_samples)
    y1_samples = interp1(x_samples)
    y2_samples = interp2(x_samples)
    
    # Calculate normalized root mean square error
    rmse = np.sqrt(np.mean((y1_samples - y2_samples)**2))
    
    # Normalize by y range
    y_range = max(max(y1) - min(y1), max(y2) - min(y2))
    if y_range == 0:
        y_range = 1.0  # Avoid division by zero
        
    normalized_rmse = rmse / y_range
    
    # Convert to similarity score (1 - normalized error)
    similarity = max(0, 1 - normalized_rmse)
    
    return similarity

# This is the main function for screening the detected curves based on color decomposition
# Several criteria are applied to filter out invalid curves
def screen_for_valid_curves(img_array, num_colors=10, min_points=500):
    """
    Screen image for valid curve candidates using palette quantization,
    including derivative-based curvature analysis.
    
    Args:
        img_array: RGB numpy array (0-255)
        num_colors: Number of colors to use in palette quantization
        min_points: Minimum points to consider a valid curve
        
    Returns:
        Dictionary mapping curve colors to point lists
    """
    import PIL
    import numpy as np
    from PIL import Image
    from pyod.models.knn import KNN
    import cv2
    import matplotlib.pyplot as plt
    
    # Convert to PIL Image for color quantization
    # pil_img = Image.fromarray(img_array.astype(np.uint8))
        # Check if img_array is already a PIL Image
    if isinstance(img_array, PIL.Image.Image):
        pil_img = img_array
        # Get image dimensions from PIL Image
        img_width, img_height = pil_img.size
    else:
        # If not PIL Image, convert from numpy array
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        # Get dimensions from numpy array
        img_height, img_width = img_array.shape[:2]

    # pil_img = PIL.Image.open(image_path) 
    pil_img.convert('RGB')
    # Apply palette conversion to quantize colors
    paletted = pil_img.convert('P', palette=PIL.Image.ADAPTIVE, colors=num_colors)
    
    # Get palette information
    palette = paletted.getpalette()
    paletted_array = np.asarray(paletted)
    img_shape = paletted_array.shape
    height, width = paletted_array.shape
    
    # Dictionary to store valid curves detected from palette
    valid_curves = {}
    
    # Screen each palette color
    print(f"Screening {num_colors} palette colors for valid curves...")
    
    for i in range(num_colors):
        # Get pixels for this color index
        y_coords, x_coords = np.where(paletted_array == i)
        # plt.figure(figsize=(12, 8))
        # plt.scatter(x_coords, y_coords, s=1, c='blue')
        # plt.title(f"Original curve (color index {i})")
        # plt.tight_layout()
        # plt.show()
            
     
        if len(x_coords) < min_points:
            print(f"Color index {i}: Too few points ({len(x_coords)}), skipping.")
            continue
            
        # Skip if too many points (likely background or axes)
        if len(x_coords) > 0.2 * height * width: # img_array.shape[0] * img_array.shape[1]:
            print(f"Color index {i}: Too many points ({len(x_coords)}), likely background.")
            continue
        
        # Get the RGB color
        color_rgb = tuple(palette[i*3:i*3+3])
        
        # SCREENING CRITERIA #1: Check point distribution along x-axis
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        x_range = x_max - x_min
        
        # # Skip if x-range is too small
        # if x_range < 50:
        #     print(f"Color index {i}: X-range too small ({x_range}), skipping.")
        #     continue
        
        # Create histogram of x-coordinates
        num_bins = 20
        hist, _ = np.histogram(x_coords, bins=num_bins)
        
        # Calculate percentage of bins that have points
        bins_with_points = np.sum(hist > 0)
        bin_coverage = bins_with_points / num_bins

        # plt.figure(figsize=(12, 8))
        # plt.subplot(2, 1, 1)
        # plt.scatter(x_coords, y_coords, s=1, c='blue')
        
        # Good curves should have points distributed across many bins
        if bin_coverage < 0.5:  # Less than 50% of bins have points
            print(f"Color index {i}: Poor x-distribution (coverage: {bin_coverage:.2f}), skipping.")
            continue
        
        # SCREENING CRITERIA #2: Check for multiple y-values per x-value
        # Group by x-coordinate and check y-variance
        x_to_ys = {}
        for x, y in zip(x_coords, y_coords):
            if x not in x_to_ys:
                x_to_ys[x] = []
            x_to_ys[x].append(y)
        
        # Calculate average y-variance per x
        y_variances = [np.var(ys) for ys in x_to_ys.values() if len(ys) > 1]
        if y_variances:
            avg_y_variance = np.mean(y_variances)
            # Good curves should have low y-variance per x
            if avg_y_variance > 2000:  # Threshold may need tuning
                print(f"Color index {i}: High y-variance ({avg_y_variance:.2f}), likely multiple curves.")
                continue
        
        # SCREENING CRITERIA #3: Derivative-based curvature analysis
        try:
            # Create sorted points
            points = list(zip(x_coords, y_coords))
            points.sort(key=lambda p: p[0])  # Sort by x-coordinate
            
            # Apply KNN outlier detection first to clean points
            points_array = np.array(points)
            clf = KNN(n_neighbors=min(5, len(points)//20), contamination=0.05)
            clf.fit(points_array)
            outliers = clf.labels_
            inlier_indices = np.where(outliers == 0)[0]
            cleaned_points = [points[i] for i in inlier_indices]
            
            # Ensure enough points remain
            if len(cleaned_points) < min_points:
                print(f"Color index {i}: Too few points after outlier removal, skipping.")                
                continue
                
            # Extract x and y arrays from cleaned points
            x_clean, y_clean = zip(*cleaned_points)
            x_bis, y_bis = derivative(*derivative(x_clean, y_clean))
            print('curvature',  np.mean(x_bis) )
            # if np.mean(x_bis)<200:
            #     continue
           
            # Calculate derivatives
            x_deriv, y_deriv = derivative(x_clean, y_clean)
            
            # Calculate second derivatives if needed
            x_deriv2, y_deriv2 = derivative(x_deriv, y_deriv)
            
            # Check derivative statistics for anomalies
            # High variance in derivatives often indicates noise or multiple curves
            deriv_mean = np.mean(y_deriv)
            deriv_std = np.std(y_deriv)
            deriv_range = max(y_deriv) - min(y_deriv)
            
            # Calculate coefficient of variation to assess noisiness
            # (normalized measure of dispersion)
            if abs(deriv_mean) > 1e-6:  # Avoid division by zero
                deriv_cv = deriv_std / abs(deriv_mean)
            else:
                deriv_cv = deriv_std

            # if np.mean(x_deriv)>200:
            #     continue
            
            print(f"Color index {i}: Derivative stats - mean: {deriv_mean:.2f}, std: {deriv_std:.2f}, CV: {deriv_cv:.2f}")
            
            # # Filter based on derivative metrics
            # if deriv_cv > 5.0:  # Very high variation relative to mean
            #     print(f"Color index {i}: High derivative variation (CV: {deriv_cv:.2f}), likely noisy or multiple curves.")
            #     continue

            # Calculate mean curvature (using second derivative approximation)
            if len(y_deriv2) > 0:
                mean_curvature = np.mean(np.abs(y_deriv2))
                print(f"Color index {i}: Mean curvature: {mean_curvature:.4f}")

                if mean_curvature > 5:  # Threshold depends on image scale
                    print(f"Color index {i}: Excessive curvature, likely noise.")
                    continue
            
            print(f"Color index {i}: PASSED derivative screening")            

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.scatter(x_clean, y_clean, s=1, c='blue')
            plt.title(f"Original curve (color index {i})")
            plt.subplot(2, 1, 2)
            plt.scatter(x_deriv, y_deriv, s=1, c='red')
            plt.title("First derivative")
            plt.tight_layout()
            plt.show()
            
            # SCREENING CRITERIA #4: Final smoothness check
            try:
                # Split into chunks for piecewise analysis
                chunk_size = len(cleaned_points) // 5
                if chunk_size > 20:
                    smoothness_scores = []
                    
                    for j in range(0, len(cleaned_points), chunk_size):
                        chunk = cleaned_points[j:j+chunk_size]
                        if len(chunk) < 10:
                            continue
                            
                        chunk_x, chunk_y = zip(*chunk)
                        # Use degree 2 polynomial for better curve fitting
                        coeffs = np.polyfit(chunk_x, chunk_y, 2)
                        poly = np.poly1d(coeffs)
                        predicted_y = poly(chunk_x)
                        
                        # Calculate mean squared error
                        mse = np.mean((np.array(chunk_y) - predicted_y)**2)
                        # Normalize by y-range in this chunk
                        y_range = max(chunk_y) - min(chunk_y)
                        if y_range > 0:
                            normalized_mse = mse / (y_range**2)
                            smoothness_scores.append(normalized_mse)
                    
                    if smoothness_scores:
                        avg_smoothness = np.mean(smoothness_scores)
                        # Threshold for "good" smoothness
                        if avg_smoothness > 0.1:  # Lower is smoother
                            print(f"Color index {i}: Poor smoothness score ({avg_smoothness:.4f}), skipping.")
                            continue
            except Exception as e:
                print(f"Error in smoothness check for color index {i}: {e}")
                # Continue since we've already passed derivative checks
            
            # If we get here, this color passed all screening criteria
            print(f"Color index {i}: PASSED ALL screening criteria")
            
            # Apply morphological operations for smoother curve
            clean_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
            for x, y in cleaned_points:
                clean_mask[y, x] = 255
                
            kernel = np.ones((2, 2), np.uint8)
            mask_processed = cv2.dilate(clean_mask, kernel, iterations=1)
            mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)
            
            y_final, x_final = np.where(mask_processed > 0)
            final_points = list(zip(x_final, y_final))
            
            # Store color and points in valid curves dict
            valid_curves[color_rgb] = final_points
            
        except Exception as e:
            print(f"Error analyzing color index {i}: {e}, skipping.")
            continue
    
    return valid_curves

def visualize_valid_curves(img_array, valid_curves, show_derivatives=True):
    """
    Visualize curves identified as valid in the screening process,
    optionally including derivative visualization.
    
    Args:
        img_array: Original RGB image array (0-255)
        valid_curves: Dictionary mapping colors to curve points
        show_derivatives: Whether to show derivative plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a copy of the original image for visualization
    vis_img = np.copy(img_array)
    
    # Create a separate visualization for each curve
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_array)
    plt.title(f"All Valid Curves ({len(valid_curves)} found)")
    
    # Use different colors for different curves
    viz_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (255, 0, 255), (0, 255, 255),
                 (128, 0, 0), (0, 128, 0), (0, 0, 128)]
    
    # Plot all curves on the combined visualization
    for i, (color, points) in enumerate(valid_curves.items()):
        # Get a distinct color for visualization
        viz_color = viz_colors[i % len(viz_colors)]
        
        # Draw points on the combined visualization
        x_coords, y_coords = zip(*points)
        plt.scatter(x_coords, y_coords, s=1, color=[v/255 for v in viz_color], 
                   label=f"Curve {i+1}: RGB={color}")
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.0))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Also create individual visualizations for each curve
    for i, (color, points) in enumerate(valid_curves.items()):
        # Sort points by x coordinate
        points.sort(key=lambda p: p[0])
        x_coords, y_coords = zip(*points)
        
        # Create a figure with multiple subplots if showing derivatives
        if show_derivatives:
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            
            # Original curve
            axes[0].imshow(img_array)
            axes[0].scatter(x_coords, y_coords, s=1, color=[v/255 for v in viz_colors[i % len(viz_colors)]])
            axes[0].set_title(f"Valid Curve {i+1}: RGB={color} ({len(points)} points)")
            axes[0].axis('off')
            
            # Calculate derivatives
            if len(points) >= 3:  # Need at least 3 points for first derivative
                try:
                    # First derivative
                    x_deriv, y_deriv = derivative(x_coords, y_coords)
                    axes[1].scatter(x_deriv, y_deriv, s=1, color='red')
                    axes[1].set_title("First Derivative (dy/dx)")
                    axes[1].grid(True)
                    
                    # Second derivative (if enough points)
                    if len(x_deriv) >= 3:
                        x_deriv2, y_deriv2 = derivative(x_deriv, y_deriv)
                        axes[2].scatter(x_deriv2, y_deriv2, s=1, color='blue')
                        axes[2].set_title("Second Derivative (d²y/dx²)")
                        axes[2].grid(True)
                    else:
                        axes[2].set_title("Not enough points for second derivative")
                
                except Exception as e:
                    print(f"Error calculating derivatives: {e}")
                    axes[1].set_title("Error in derivative calculation")
                    axes[2].set_title("Error in derivative calculation")
            else:
                axes[1].set_title("Not enough points for derivatives")
                axes[2].set_title("Not enough points for derivatives")
            
            plt.tight_layout()
            plt.show()
        else:
            # Just show the curve on the original image
            plt.figure(figsize=(10, 6))
            plt.imshow(img_array)
            plt.scatter(x_coords, y_coords, s=1, color=[v/255 for v in viz_colors[i % len(viz_colors)]])
            plt.title(f"Valid Curve {i+1}: RGB={color} ({len(points)} points)")
            plt.axis('off')
            plt.show()


def improved_color_matching(legend_dict, valid_curves, img_array):
    """
    Enhanced color matching that handles brightness differences and prioritizes 
    curves with more points while avoiding duplicate matches.
    
    Args:
        legend_dict: Dictionary of {label: RGB color} from legend
        valid_curves: Dictionary of {curve_id: points list}
        img_array: Original RGB image array
        
    Returns:
        Dictionary mapping legend labels to matched curve points
    """
    # Step 1: Calculate the average color of each curve with better sampling
    curve_info = {}
    
    # Ensure image array is in the correct format
    if isinstance(img_array, PIL.Image.Image):
        img_np = np.array(img_array)
    else:
        img_np = img_array.copy()
    
    # Convert to uint8 if needed
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    # Ensure we're working with RGB (3 channels)
    if img_np.ndim == 3 and img_np.shape[2] == 4:  # RGBA
        img_np = img_np[:, :, :3]  # Take only RGB channels
    
    height, width = img_np.shape[:2]
    
    # Sample colors and gather curve information
    for curve_id, points in valid_curves.items():
        # Use uniform sampling to get representative colors
        num_samples = min(200, len(points))
        sample_indices = np.linspace(0, len(points)-1, num_samples, dtype=int)
        sample_points = [points[i] for i in sample_indices]
        
        # Sample pixels at curve points
        valid_colors = []
        for x, y in sample_points:
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < width and 0 <= y_int < height:
                try:
                    pixel_color = img_np[y_int, x_int]
                    
                    # Ensure we have only RGB (3 channels)
                    if len(pixel_color) == 4:  # RGBA
                        pixel_color = pixel_color[:3]
                    
                    # Skip white/very light pixels (likely background)
                    if not np.all(pixel_color > 240):
                        valid_colors.append(pixel_color)
                except IndexError:
                    continue
        
        # Calculate average color if we have valid samples
        if valid_colors:
            # Convert all colors to same shape (3,) for RGB
            valid_colors_rgb = [color[:3] if len(color) > 3 else color for color in valid_colors]
            avg_color = np.mean(valid_colors_rgb, axis=0).astype(int)
            
            # Ensure avg_color is RGB (3 values)
            if len(avg_color) > 3:
                avg_color = avg_color[:3]
            
            # Store curve info including point count for weighting
            curve_info[curve_id] = {
                'avg_color': tuple(avg_color),
                'point_count': len(points),
                'points': points
            }
            print(f"Curve {curve_id}: Average color = {tuple(avg_color)}, {len(points)} points")
    
    # Step 2: Normalize legend colors to ensure they're RGB (3 channels)
    normalized_legend_colors = {}
    for label, color in legend_dict.items():
        # Ensure color is RGB (3 values)
        if len(color) > 3:  # RGBA
            color = color[:3]
        normalized_legend_colors[label] = color
    
    # Step 3: Adjust legend colors to better match curve colors
    adjusted_legend_colors = {}
    
    # Calculate average brightness of both sets
    curve_brightnesses = [np.mean(info['avg_color']) for info in curve_info.values()]
    legend_brightnesses = [np.mean(color) for color in normalized_legend_colors.values()]
    
    avg_curve_brightness = np.mean(curve_brightnesses) if curve_brightnesses else 128
    avg_legend_brightness = np.mean(legend_brightnesses) if legend_brightnesses else 200
    
    # Calculate adjustment factor to darken/lighten legend colors
    brightness_ratio = avg_curve_brightness / avg_legend_brightness if avg_legend_brightness > 0 else 0.7
    print(f"Brightness adjustment ratio: {brightness_ratio:.2f} (legend: {avg_legend_brightness:.1f}, curve: {avg_curve_brightness:.1f})")
    
    for label, color in normalized_legend_colors.items():
        # Apply brightness adjustment to legend colors
        adjusted_color = tuple(max(0, min(255, int(c * brightness_ratio))) for c in color)
        adjusted_legend_colors[label] = adjusted_color
        print(f"Adjusted '{label}' color from {color} to {adjusted_color}")
    
    # Step 4: Create visualization of the adjusted colors
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Legend Colors")
    for i, (label, color) in enumerate(normalized_legend_colors.items()):
        plt.bar(i, 1, color=[c/255 for c in color], label=label)
    plt.xticks(range(len(normalized_legend_colors)), list(normalized_legend_colors.keys()), rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.title("Adjusted Legend Colors")
    for i, (label, color) in enumerate(adjusted_legend_colors.items()):
        plt.bar(i, 1, color=[c/255 for c in color], label=label)
    plt.xticks(range(len(adjusted_legend_colors)), list(adjusted_legend_colors.keys()), rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.title("Curve Average Colors")
    curve_ids = list(curve_info.keys())
    for i, curve_id in enumerate(curve_ids):
        color = curve_info[curve_id]['avg_color']
        plt.bar(i, 1, color=[c/255 for c in color], label=str(curve_id))
    plt.xticks(range(len(curve_ids)), curve_ids, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Step 5: Compute color similarity using multiple metrics
    matching_scores = []
    
    for label, legend_color in adjusted_legend_colors.items():
        for curve_id, info in curve_info.items():
            curve_color = info['avg_color']
            
            # 1. RGB Euclidean distance
            rgb_distance = np.sqrt(np.sum(((np.array(legend_color) - np.array(curve_color))**2)))
            max_rgb_distance = np.sqrt(3 * 255**2)
            rgb_similarity = 1.0 - (rgb_distance / max_rgb_distance)
            
            # 2. HSV similarity (better for perceptual differences)
            try:
                from matplotlib.colors import rgb_to_hsv
                
                # Convert RGB to HSV (0-1 range)
                legend_rgb_norm = np.array([c/255.0 for c in legend_color]).reshape(1, 1, 3)
                curve_rgb_norm = np.array([c/255.0 for c in curve_color]).reshape(1, 1, 3)
                
                legend_hsv = rgb_to_hsv(legend_rgb_norm)[0][0]
                curve_hsv = rgb_to_hsv(curve_rgb_norm)[0][0]
                
                # Calculate distance in HSV space (with hue as circular)
                hue_diff = min(abs(legend_hsv[0] - curve_hsv[0]), 1 - abs(legend_hsv[0] - curve_hsv[0]))
                sat_diff = abs(legend_hsv[1] - curve_hsv[1])
                val_diff = abs(legend_hsv[2] - curve_hsv[2])
                
                # Weight the components (hue most important, then saturation, then value)
                hsv_distance = 0.6 * hue_diff + 0.3 * sat_diff + 0.1 * val_diff
                hsv_similarity = 1.0 - hsv_distance
            except Exception as e:
                print(f"Warning: HSV calculation failed: {e}. Using RGB similarity only.")
                hsv_similarity = rgb_similarity
            
            # Include point count as a quality factor (more points is better)
            point_count = info['point_count']
            quality_factor = min(1.0, point_count / 1000)
            
            # Final score: 40% RGB, 40% HSV, 20% quality
            final_score = 0.4 * rgb_similarity + 0.4 * hsv_similarity + 0.2 * quality_factor
            
            matching_scores.append({
                'label': label,
                'curve_id': curve_id, 
                'score': final_score,
                'rgb_similarity': rgb_similarity,
                'hsv_similarity': hsv_similarity,
                'point_count': point_count,
                'legend_color': legend_color,
                'curve_color': curve_color
            })
    
    # Step 6: Sort scores and apply optimized matching
    matching_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Use a more sophisticated matching algorithm to avoid duplicates
    matched_curves = {}
    used_curves = set()
    
    # First pass: assign curves with highest confidence
    for match in matching_scores:
        label = match['label']
        curve_id = match['curve_id']
        
        # Skip if this label or curve is already matched
        if label in matched_curves or curve_id in used_curves:
            continue
        
        # Only match with a reasonable score
        if match['score'] > 0.3:  # Adjustable threshold
            matched_curves[label] = curve_info[curve_id]['points']
            used_curves.add(curve_id)
            print(f"Matched '{label}' to curve {curve_id}")
            print(f"  Score: {match['score']:.3f}, RGB: {match['rgb_similarity']:.3f}, HSV: {match['hsv_similarity']:.3f}")
            print(f"  Legend color: {match['legend_color']}, Curve color: {match['curve_color']}")
            print(f"  Point count: {match['point_count']}")
    
    # Second pass: assign any remaining labels to best available curves
    unmatched_labels = set(normalized_legend_colors.keys()) - set(matched_curves.keys())
    
    if unmatched_labels:
        print(f"\nForce-matching {len(unmatched_labels)} remaining labels...")
        
        # For each unmatched label, find the best available curve
        for label in unmatched_labels:
            best_match = None
            best_score = -1
            
            for match in matching_scores:
                if match['label'] == label and match['curve_id'] not in used_curves:
                    if match['score'] > best_score:
                        best_score = match['score']
                        best_match = match
            
            if best_match:
                curve_id = best_match['curve_id']
                matched_curves[label] = curve_info[curve_id]['points']
                used_curves.add(curve_id)
                print(f"Force-matched '{label}' to curve {curve_id} (score: {best_match['score']:.3f})")
    
    return matched_curves
##############################################################################################

# This is the main function to extract the data and save to a dictionary

def get_plot_data_dict_hybrid(image_path, axis_align_config, plot_extract_config):
    """
    Hybrid approach to extract plot data using PlotDigitizer first 
    for background removal, then color decomposition for curve extraction.
    
    Args:
        image_path: Path to the plot image
        axis_align_config: Configuration for AxisAlignment
        plot_extract_config: Configuration for PlotDigitizer
        
    Returns:
        Dict mapping legend labels to data series
    """
    print(f"\n--- Starting Hybrid Data Extraction for: {image_path} ---")
    final_data_dict = {}
    temp_dir = None

    # === Step 1: Extract Legend Info ===
    print("Step 1: Extracting legend information...")
    legend_dict = extract_legend_data(image_path)
    if not legend_dict:
        print("Error: Could not extract legend information. Stopping.")
        return {}
    print(f"Step 1: Extracted {len(legend_dict)} legend entries with colors.")
    
    # === Step 2: Axis Alignment ===
    print("\nStep 2: Performing Axis Alignment...")
    axis_info_for_transform = {'x': {'pixels': [], 'values': []}, 'y': {'pixels': [], 'values': []}}
    results_ticks = None
    resize_ratio_x = 1.0
    img_rgb_shape = None
    plot_bbox = None
    
    try:
        axis_alignment = AxisAlignment(axis_align_config)
        # Create temporary directory for AxisAlignment
        img_dir = os.path.dirname(image_path)
        img_filename = os.path.basename(image_path)
        temp_dir_name = f"temp_axis_align_{os.path.splitext(img_filename)[0]}"
        temp_dir = os.path.join(img_dir, temp_dir_name)
        os.makedirs(temp_dir, exist_ok=True)
        shutil.copy(image_path, os.path.join(temp_dir, img_filename))
        
        axis_alignment.load_data(temp_dir)
        img_axis, plot_bbox, results_ticks, results_all = axis_alignment.run(0)
        print("Step 2: Axis alignment successful.")
        
        # Parse Axis Ticks
        print("Step 2a: Parsing axis ticks from results_ticks structure...")
        if results_ticks:
            num_parsed_x = 0
            num_parsed_y = 0
            for tick_info in results_ticks:
                if not isinstance(tick_info, (list, tuple)) or len(tick_info) != 3: continue
                text_value_str, confidence, bbox_array = tick_info
                if not isinstance(bbox_array, np.ndarray) or bbox_array.size != 4: continue
                
                cleaned_text_value = str(text_value_str)
                if (cleaned_text_value.endswith('1') or cleaned_text_value.endswith('l')) \
                   and len(cleaned_text_value) > 1 and confidence < 0.7:
                    potential_num = cleaned_text_value[:-1]
                    try:
                        float(potential_num); cleaned_text_value = potential_num
                    except ValueError: pass
                
                try: value_float = float(cleaned_text_value)
                except (ValueError, TypeError): continue
                
                try: x_min, y_min, x_max, y_max = bbox_array
                except (ValueError, TypeError): continue
                
                width = x_max - x_min; height = y_max - y_min
                if width <= 0 and height <= 0: continue
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                
                # Assume X-axis tick if text is horizontal
                if width > height * 1.5:
                    axis_info_for_transform['x']['pixels'].append(x_center)
                    axis_info_for_transform['x']['values'].append(value_float)
                    num_parsed_x += 1
                # Assume Y-axis tick if text is vertical
                elif height > width * 1.5:
                    axis_info_for_transform['y']['pixels'].append(y_center)
                    axis_info_for_transform['y']['values'].append(value_float)
                    num_parsed_y += 1
            
            print(f"Step 2a: Parsed {num_parsed_x} X-axis ticks and {num_parsed_y} Y-axis ticks.")
            
            # Check if we have enough X-axis ticks
            if num_parsed_x < 2:
                print("Error: Fewer than 2 X-axis ticks parsed. Cannot proceed.")
                if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
                return {}
                
            # For Y-axis, use default 0-1 range if no ticks parsed
            if num_parsed_y < 2 and plot_bbox is not None:
                print("Note: Using default Y-axis range (0-1) since fewer than 2 Y-axis ticks were detected.")
                # Estimate Y-axis pixel positions from plot bounding box
                y_min, y_max = plot_bbox[1], plot_bbox[3]
                axis_info_for_transform['y']['pixels'] = [y_max, y_min]  # Bottom to top
                axis_info_for_transform['y']['values'] = [0.0, 1.0]      # 0 at bottom, 1 at top
                print(f"  Y-axis pixel range set to {y_min:.1f}-{y_max:.1f}")
        else:
            print("Error: Axis alignment returned no ticks. Cannot proceed.")
            if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            return {}
            
    except Exception as e:
        print(f"Error during Axis Alignment or Tick Parsing: {e}")
        if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return {}
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")
    
    # === Step 3: PlotDigitizer for Background Removal ===
    print("\nStep 3: Using PlotDigitizer for background removal...")
    plot_digitizer = PlotDigitizer()
    try:
        plot_digitizer.load_seg("spatialembedding", plot_extract_config)
        success = plot_digitizer.predict_and_process(image_path, denoise=False)
        if not success:
            print("Error: Plot digitization failed during prediction/processing.")
            return {}
            
        res_map = plot_digitizer.result_dict.get('visual', {})
        img_rgb = res_map.get('img_rgb')      # Resized RGB image (0-1 float)
        seg_map = res_map.get('seg_map')      # Resized binary mask
        ins_map = res_map.get('ins_map')      # Resized instance mask (int IDs)
        
        if img_rgb is None:
            print("Error: Missing img_rgb from PlotDigitizer results.")
            return {}
            
        # Get image shape for scaling calculations
        img_rgb_shape = img_rgb.shape
        print(f"Step 3: Plot digitization successful. Resized shape: {img_rgb_shape[:2]}")
        
        # Calculate resize ratio
        orig_img = Image.open(image_path)
        orig_w, orig_h = orig_img.size
        resized_h, resized_w = img_rgb_shape[:2]
        resize_ratio_x = orig_w / resized_w if resized_w > 0 else 1.0
        print(f"  Resize ratio for X-axis (Original/Resized): {resize_ratio_x:.2f}")
        masked_img = seg_map[..., None] * img_rgb
        masked_img[np.where((masked_img==[0,0,0]).all(axis=2))] = [1,1,1]
        im = Image.fromarray((255*masked_img).astype(np.uint8)) # Convert NumPy RGB to PIL Image
        im = im.resize((orig_w, orig_h), Image.BILINEAR) # Resize to original dimensions
        im.save('spec.jpeg')
        new = dilate_image('spec.jpeg')
        plt.imshow(new, interpolation='none')
        plt.savefig('dilated_im.png') 
        enhanced_img = PIL.Image.open('dilated_im.png').resize((orig_w, orig_h), PIL.Image.BILINEAR)
        plt.figure(figsize=(10, 8))
        plt.imshow(enhanced_img)
        plt.title('Enhanced Image after Preprocessing')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error during PlotDigitizer processing: {e}")
        return {}
    
    # === Step 4: Scale Axis Info & Transform Coordinates ===
    print("\nStep 4: Scaling axis info to match resized image...")
    # Adjust X-axis tick pixel positions to match the resized image space
    scaled_axis_info = {'x': {'pixels': [], 'values': []}}
    try:
        if axis_info_for_transform['x']['pixels']:
            safe_resize_ratio_x = resize_ratio_x if resize_ratio_x != 0 else 1.0
            scaled_axis_info['x']['pixels'] = (np.array(axis_info_for_transform['x']['pixels']) / safe_resize_ratio_x).tolist()
            scaled_axis_info['x']['values'] = axis_info_for_transform['x']['values']
            print("  X-axis tick pixel positions scaled to resized image coordinates.")
        else:
            print("Error: No X-axis tick pixel positions available for scaling.")
            return {}
    except Exception as e:
        print(f"Warning: Failed to scale X-axis tick positions: {e}. Using unscaled values.")
        scaled_axis_info['x']['pixels'] = axis_info_for_transform['x']['pixels']
        scaled_axis_info['x']['values'] = axis_info_for_transform['x']['values']
    
    # === Step 5: Extract Curves by Color from Enhanced Image ===
    print("\nStep 5: Extracting curves using color decomposition...")
    # Get adaptive color tolerance
    tolerance_dict = adaptive_color_tolerance(enhanced_img, legend_dict)
    
    # Always transform Y coordinates to 0-1 range
    transform_y = True

    # Make sure Y-axis info exists with proper 0-1 mapping
    if plot_bbox is not None:
        # Create 'y' key in scaled_axis_info if it doesn't exist
        if 'y' not in scaled_axis_info:
            scaled_axis_info['y'] = {'pixels': [], 'values': []}
        
        # If Y-axis info is missing or incomplete, establish Y-axis mapping from plot bounding box
        if len(scaled_axis_info['y'].get('pixels', [])) < 2:
            y_min, y_max = plot_bbox[1], plot_bbox[3]  # Top and bottom of plot area
            scaled_axis_info['y']['pixels'] = [y_max, y_min]  # Bottom to top
            scaled_axis_info['y']['values'] = [0.0, 1.0]      # 0 at bottom, 1 at top
            print("  Using 0-1 scale for Y-axis based on plot bounding box")
    
    print("Step 5a: Prescreening for good curve candidates...")
    valid_curves = screen_for_valid_curves(enhanced_img, min_points=10)

    print(f"Found {len(valid_curves)} valid curve candidates after screening.")

    # Visualize the valid curves before deduplication
    visualize_valid_curves(enhanced_img, valid_curves, show_derivatives=False)

    # Now deduplicate curves that are too similar (optional but helpful)
    deduplicated_curves = deduplicate_curves(valid_curves, similarity_threshold=0.8)
    valid_curves = deduplicate_curves(valid_curves, similarity_threshold=0.8)

    print("Step 5b: Matching legend colors to valid curve candidates...")
    matched_curves = improved_color_matching(legend_dict, valid_curves, enhanced_img)

    for label, pixel_coords in matched_curves.items():
        print(f"Processing matched curve for '{label}'... ({len(pixel_coords)} points)")

        filtered_coords = remove_outliers_robust(pixel_coords, window_size=15, std_threshold=2.0)
        print(f"  After robust outlier removal: {len(filtered_coords)} points")
        
        # Transform coordinates (X and possibly Y)
        data_coords = transform_coordinates(filtered_coords, scaled_axis_info, transform_y=transform_y)
        
        if data_coords:
            sampled_coords = data_coords #sample_curve(data_coords, num_samples=2000)
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x_vals, y_vals = zip(*sorted(sampled_coords, key=lambda p: p[0]))
                smoothed = lowess(y_vals, x_vals, frac=0.1, it=3)
                smoothed_coords = list(zip(smoothed[:, 0], smoothed[:, 1]))
            except (ImportError, ValueError):
                # Fall back to Savitzky-Golay if LOWESS fails
                smoothed_coords = smooth_curve(sampled_coords, window_size=50, polyorder=3)
            
            # Sort by x value
            smoothed_coords.sort(key=lambda p: p[0])
            
            final_data_dict[label] = smoothed_coords
            y_label = "data values" if transform_y else "pixel coordinates"
            print(f"  Successfully extracted data for '{label}' ({len(smoothed_coords)} points, Y in {y_label})")
        else:
            print(f"  Failed to transform coordinates for '{label}'. Skipping.")   
  
    print("\n--- Data Extraction Complete ---")
    print("\nRescaling Y values to full 0-1 range...")
    all_y_values = []
    for label in final_data_dict.keys():
        if final_data_dict[label]:
            _, y_values = zip(*final_data_dict[label])
            all_y_values.extend(y_values)

    if all_y_values:
        global_y_min = min(all_y_values)
        global_y_max = max(all_y_values)
        
        # Only rescale if the range isn't already 0-1
        if abs(global_y_min) > 0.01 or abs(global_y_max - 1) > 0.01:
            print(f"  Current Y range: {global_y_min:.3f} to {global_y_max:.3f}")
            
            # Rescale all curves
            for label in final_data_dict.keys():
                if final_data_dict[label]:
                    points = final_data_dict[label]
                    x_values, y_values = zip(*points)
                    plt.plot(x_values, y_values, marker='.', markersize=4, linestyle='-', label=label)
                    
                    # Apply linear rescaling from [min, max] to [0, 1]
                    y_range = global_y_max - global_y_min
                    if y_range > 0:  # Prevent division by zero
                        y_rescaled = [(y - global_y_min) / y_range for y in y_values]
                        final_data_dict[label] = list(zip(x_values, y_rescaled))
                        print(f"  Rescaled Y values for '{label}'")    

    return final_data_dict

def plot_extracted_data(data_dictionary, y_in_data_units=True):
    """
    Plots the extracted data series.
    
    Args:
        data_dictionary: Dict mapping labels to data points
        y_in_data_units: Whether Y values are in data units (vs pixel coordinates)
    """
    plt.figure(figsize=(10, 6))
    has_data_to_plot = False
    for label, data in data_dictionary.items():
        if data:
            try:
                x_data, y_values = zip(*data)
                plt.plot(x_data, y_values, marker='.', markersize=4, linestyle='-', label=label)
                has_data_to_plot = True
            except ValueError as e:
                print(f"Warning: Could not plot data for label '{label}': {e}")
        else:
            print(f"Warning: No data points to plot for label '{label}'.")

    plt.xlabel("Wavelength (nm)")
    
    if y_in_data_units:
        plt.ylabel("Absorbance")
        plt.title("Extracted Spectral Data")
    else:
        plt.ylabel("Y-Axis (Pixel Coordinate)")
        plt.title("Extracted Plot Data (Y-axis in Pixels)")
        # Invert Y-axis if using pixel coordinates (0 at top)
        plt.gca().invert_yaxis()
    
    if has_data_to_plot:
        plt.legend()
    else:
        print("No data was successfully plotted.")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("extracted_spectra.png")
    plt.show()
    return plt.gcf()

if __name__ == "__main__":
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma200800280\adma200800280\images_folder\page_3_img_3_a.jpg" 
    #image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\nmat2272\nmat2272\images_folder\nmat2272_fig2_b.jpg"  
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\nmat2272\nmat2272\images_folder\nmat2272_fig1_b.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma200902917\adma200902917\images_folder\page_2_img_1_b.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma.201002234\adma.201002234\images_folder\page_2_img_4_0.jpg"#
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\cm202117d\cm202117d\images_folder\page_3_img_2_a.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\cm202117d\cm202117d\images_folder\page_3_img_2_c.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\ma501080u\ma501080u\images_folder\page_4_img_1_a.jpg"

    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\ma501080u\ma501080u\images_folder\page_4_img_1_b.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acsmacrolett.6b00250\acsmacrolett.6b00250\images_folder\page_2_img_1_a.jpg"
    
    # C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acs.macromol.6b01114\acs.macromol.6b01114\images_folder\page_4_img_1_a.jpg
    # C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acs.macromol.6b01114\acs.macromol.6b01114\images_folder\page_4_img_1_b.jpg
    #C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acs.macromol.6b01763\acs.macromol.6b01763\images_folder\page_5_img_1_a.jpg

    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acs.macromol.8b01789\acs.macromol.8b01789\images_folder\page_3_img_1_b.jpg"    

    #C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adom.201800594\adom.201800594\images_folder\page_4_img_1_a.jpg"
    #C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adom.201800594\adom.201800594\images_folder\page_4_img_1_b.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adom.201800594\adom.201800594\images_folder\page_4_img_1_c.jpg"

    #C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acs.chemmater.9b01293\acs.chemmater.9b01293\images_folder\page_6_img_2_a.jpg"
    #C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acs.chemmater.9b01293\acs.chemmater.9b01293\images_folder\page_6_img_2_b.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acs.chemmater.9b01293\acs.chemmater.9b01293\images_folder\page_6_img_2_c.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\c9py01720h\c9py01720h\images_folder\page_3_img_1_b.jpg"
    #image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\c8py01105b\c8py01105b\images_folder\page_3_img_1_a.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\c8py01105b\c8py01105b\images_folder\page_3_img_1_b.jpg"

    final_dict = get_plot_data_dict_hybrid(image_path, axis_align_opt, plot_extract_opt)
       
    # Plot the results
    if final_dict:
        print("Plotting extracted data...")
        plot_extracted_data(final_dict, y_in_data_units=True)
    else:
        print("No data extracted. Check error messages above.")