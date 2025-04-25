import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr
import shutil 
import torch
from scipy.interpolate import interp1d # Not needed after removing check_img
from pyod.models.knn import KNN # Not needed after removing check_img
from scipy.cluster import vq # Still used for kmeans on colors (though maybe not essential)
from PIL import Image
from statsmodels.nonparametric.smoothers_lowess import lowess

import os
import pandas as pd # Keep if used elsewhere, maybe not needed now
from scipy.interpolate import interp1d
import os
import shutil
import pandas as pd

# Import your PlotDigitizer and AxisAlignment classes
from matagen.utils.plot_data_extraction.plot_digitizer import PlotDigitizer
from matagen.utils.axis_alignment.utils import AxisAlignment


# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust the number of '..' based on script location relative to project root!
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..')) # Example: 3 levels down
checkpoints_base = os.path.join(project_root, 'checkpoints', 'axis_alignment')
checkpoints_plot_extract_base = os.path.join(project_root, 'checkpoints', 'plot_data_extraction')

print(f"Project Root (estimated): {project_root}")
print(f"Axis Checkpoints Base: {checkpoints_base}")
print(f"Plot Checkpoints Base: {checkpoints_plot_extract_base}")
print("-" * 20)

# --- Configuration ---
# Ensure file paths exist after calculating project_root
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
    "checkpoint_path": os.path.join(checkpoints_plot_extract_base, "checkpoint_0999.pth"), # CRITICAL: Path to spatial embedding weights
    "dataset": {
        'name': 'cityscapes', # This might be needed by get_model internally, check requirements
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

def smooth_curve_lowess(coords, frac=0.1):
    """
    Apply LOWESS smoothing for better handling of noisy data.
    
    :param coords: List of (x, y) tuples
    :param frac: The fraction of the data used to compute each y-value
    :return: Smoothed coordinates
    """
    if len(coords) < 5:
        return coords
        
    x_vals, y_vals = zip(*sorted(coords, key=lambda p: p[0]))
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # Apply LOWESS smoothing
    smoothed = lowess(y_vals, x_vals, frac=frac, it=3, return_sorted=True)
    
    return list(zip(smoothed[:, 0], smoothed[:, 1]))

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
        r'^p\d+',                # Matches "P1", "P2", etc. (with or without wavelength info)
        r'^pentamer\s+[a-z]',    # Matches "Pentamer A", "Pentamer B", etc.
        r'^polymer\s+[a-z]',     # Matches "Polymer A", "Polymer B", etc.
        r'^oligomer\s+[a-z0-9]', # Matches "Oligomer 1", "Oligomer A", etc.
        r'^p[a-z]',              # Matches "PDot", "PPDot", etc.
        r'^prodot',              # Matches "ProDOT" variations
        r'^pedot',               # Matches "PEDOT" variations
        r'^compound\s+\d+',      # Matches "Compound 1", "Compound 2", etc.
        r'^p\(.*\)',             # Matches "P(LAcDOT)", "P(BAcDOT-DMP)", etc.
        r'^p\([^)]+\-[^)]+\)',   # Specifically match polymer names with hyphens inside parentheses
    ]
    
    # Patterns to EXCLUDE - but only if they don't match the include patterns
    exclude_patterns = [
        r'^[0-9]+$',             # Just numbers like "600", "800"
        r'^[0-9\.]+$',           # Decimal numbers like "0.2", "1.0"
        r'^[0-9]+\s*nm$',        # Just wavelengths like "121nm"
        r'^nm$',                 # Just "nm"
        r'^[a-z][\)\]}]$',       # Subplot labels like "a)", "b)"
        r'wavelength',           # possible axis labels
        r'intensity',            # possible axis labels
        r'absorbance',           # possible axis labels
        r'transmittance',        # possible axis labels
        r'^[0-9]',               # NEVER start with a number
        r'00'                    # NEVER include '00' as this was a common issue
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
                
        # If already marked for inclusion, add it and continue to next item
        if should_include:
            labels.append(text)
            bboxes.append(bbox)
            continue
        
        # Next, check if it should be excluded
        should_exclude = False
        for pattern in exclude_patterns:
            if re.search(pattern, text.lower()):
                print(f"  Excluding based on pattern: '{text}'")
                should_exclude = True
                break
                
        if should_exclude:
            continue
        
        # If not explicitly included or excluded yet, apply additional heuristics
        
        # Check if it looks like a chemical name (contains letters and not just numbers/units)
        if len(text) > 1:
            # Contains letters and isn't just a unit
            has_letters = any(c.isalpha() for c in text)
            not_just_unit = not (text.lower().endswith('nm') and len(text) <= 5)
            
            # Special case for polymer names with parentheses
            has_polymer_pattern = '(' in text and ')' in text and (
                text.startswith('P(') or 
                'DOT' in text or 
                '-' in text
            )
            
            if (has_letters and not_just_unit) or has_polymer_pattern:
                should_include = True
                print(f"  Including as possible chemical name: '{text}'")
        
        # Add to results if we should include this text
        if should_include:
            labels.append(text)
            bboxes.append(bbox)
    
    print(f"Filtered legend labels: {labels}")
    return labels, bboxes

def get_label_colors(image_path, label_bbox):
    """Extract the color associated with a detected label bounding box."""
    # Sample color from a region LEFT of the text bbox
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

def create_color_mask(img, target_color, tolerance=30):
    """
    Creates a binary mask where pixels close to target_color are 1.
    
    Args:
        img: RGB numpy array (H,W,3) with values 0-255
        target_color: RGB color to match (0-255)
        tolerance: Color distance threshold
    """
    # Convert to proper shape and type
    img_array = np.array(img)
    target_color = np.array(target_color).reshape(1, 1, 3)
    
    # Calculate Euclidean distance between each pixel and target color
    color_distance = np.sqrt(np.sum((img_array - target_color)**2, axis=2))
    
    # Create mask where distance is less than tolerance
    mask = color_distance < tolerance
    return mask

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
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
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


def extract_curve_by_color(img_array, target_color, tolerance=30, min_points=10):
    """
    Extracts curve pixels based on color matching.
    
    Args:
        img_array: RGB numpy array (0-255)
        target_color: RGB color to match (0-255)
        tolerance: Color distance threshold
        min_points: Minimum points to consider a valid curve
    """
    # Create color mask
    mask = create_color_mask(img_array, target_color, tolerance)
    
    # Get coordinates of matching pixels
    y_coords, x_coords = np.where(mask)
    
    if len(x_coords) < min_points:
        print(f"Warning: Found only {len(x_coords)} points matching color {target_color}")
        return []
    
    # Combine coordinates
    points = list(zip(x_coords, y_coords))
    return points

def remove_outliers(coords, window_size=5, std_threshold=2.5):
    """Remove outlier points based on local standard deviation."""
    if len(coords) <= window_size:
        return coords
        
    x_vals, y_vals = zip(*sorted(coords, key=lambda p: p[0]))
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # Calculate rolling standard deviation
    filtered_indices = []
    for i in range(len(y_vals)):
        start = max(0, i - window_size // 2)
        end = min(len(y_vals), i + window_size // 2 + 1)
        window = y_vals[start:end]
        mean = np.mean(window)
        std = np.std(window)
        
        if std == 0:
            filtered_indices.append(i)
        elif abs(y_vals[i] - mean) <= std_threshold * std:
            filtered_indices.append(i)
    
    filtered_coords = [(x_vals[i], y_vals[i]) for i in filtered_indices]
    return filtered_coords

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
# def smooth_curve(coords, window_size=5):
#     """Apply simple moving average smoothing to the y-coordinates."""
#     if len(coords) <= window_size:
#         return coords
        
#     x_vals, y_vals = zip(*coords)
#     x_vals = np.array(x_vals)
#     y_vals = np.array(y_vals)
    
#     # Simple moving average
#     smoothed_y = np.convolve(y_vals, np.ones(window_size)/window_size, mode='valid')
    
#     # Trim x values to match smoothed y length
#     offset = window_size // 2
#     trimmed_x = x_vals[offset:offset+len(smoothed_y)]
    
#     if len(trimmed_x) != len(smoothed_y):
#         # If lengths don't match (edge case), use alternative approach
#         trimmed_x = x_vals[:len(smoothed_y)]
    
#     return list(zip(trimmed_x, smoothed_y))

from scipy.signal import savgol_filter

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

def preprocess_segmented_image(img_rgb, seg_map=None):
    """
    Preprocess the segmented image to enhance curve visibility.
    
    Args:
        img_rgb: RGB image from PlotDigitizer (0-1 float)
        seg_map: Binary segmentation mask (optional)
    
    Returns:
        Enhanced RGB image as numpy array (0-255)
    """
    # Convert to 0-255 range
    img_255 = (img_rgb * 255).astype(np.uint8)
    
    # Apply segmentation mask if provided
    if seg_map is not None:
        # Apply mask
        masked_img = np.copy(img_255)
        for i in range(3):
            masked_img[:, :, i] = np.where(seg_map, img_255[:, :, i], 255)
    else:
        masked_img = img_255
    
    # Convert to LAB color space for enhancement
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to RGB
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return enhanced_img

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
        success = plot_digitizer.predict_and_process(image_path, denoise=True)
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
        
        # Preprocess the segmented image
        enhanced_img = preprocess_segmented_image(img_rgb, seg_map)
        
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
    
    # Extract curves for each legend color
    for label, color in legend_dict.items():
        print(f"Processing curve for '{label}'...")
        tolerance = tolerance_dict.get(label, 30)
        
        # Extract pixel coordinates by color
        pixel_coords = extract_curve_by_color(enhanced_img, color, tolerance=tolerance)
        
        if not pixel_coords:
            print(f"  Warning: No points found for '{label}' with tolerance {tolerance}")
            # Try with higher tolerance
            pixel_coords = extract_curve_by_color(enhanced_img, color, tolerance=tolerance*1.5)
            if not pixel_coords:
                print(f"  Still no points found with increased tolerance. Skipping '{label}'.")
                continue
        
        print(f"  Found {len(pixel_coords)} initial points for '{label}'")
        
        # First pass of outlier removal - use the more robust version
        filtered_coords = remove_outliers(pixel_coords, window_size=15, std_threshold=2.0) 
        print(f"  After outlier removal: {len(filtered_coords)} points")
        
        # Transform coordinates (X and possibly Y)
        data_coords = transform_coordinates(filtered_coords, scaled_axis_info, transform_y=transform_y)
        
        if data_coords:
            # Sample the curve to reduce point count
            sampled_coords = sample_curve(data_coords, num_samples=1000)
            
            # Apply LOWESS smoothing for smoother results
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x_vals, y_vals = zip(*sorted(sampled_coords, key=lambda p: p[0]))
                smoothed = lowess(y_vals, x_vals, frac=0.1, it=3)
                smoothed_coords = list(zip(smoothed[:, 0], smoothed[:, 1]))
            except (ImportError, ValueError):
                # Fall back to Savitzky-Golay if LOWESS fails
                smoothed_coords = smooth_curve(sampled_coords, window_size=21, polyorder=3)
            
            # Sort by x value
            smoothed_coords.sort(key=lambda p: p[0])
            
            final_data_dict[label] = smoothed_coords
            y_label = "data values" if transform_y else "pixel coordinates"
            print(f"  Successfully extracted data for '{label}' ({len(smoothed_coords)} points, Y in {y_label})")
        else:
            print(f"  Failed to transform coordinates for '{label}'. Skipping.")
    
    # # Extract curves for each legend color
    # for label, color in legend_dict.items():
    #     print(f"Processing curve for '{label}'...")
    #     tolerance = tolerance_dict.get(label, 30)
        
    #     # Extract pixel coordinates by color
    #     pixel_coords = extract_curve_by_color(enhanced_img, color, tolerance=tolerance)
        
    #     if not pixel_coords:
    #         print(f"  Warning: No points found for '{label}' with tolerance {tolerance}")
    #         # Try with higher tolerance
    #         pixel_coords = extract_curve_by_color(enhanced_img, color, tolerance=tolerance*1.5)
    #         if not pixel_coords:
    #             print(f"  Still no points found with increased tolerance. Skipping '{label}'.")
    #             continue
        
    #     print(f"  Found {len(pixel_coords)} initial points for '{label}'")
        
    #     # Post-process the curve
    #     filtered_coords = remove_outliers(pixel_coords, window_size=5, std_threshold=2.5)
    #     print(f"  After outlier removal: {len(filtered_coords)} points")
        
    #     # Transform coordinates (X and possibly Y)
    #     data_coords = transform_coordinates(filtered_coords, scaled_axis_info, transform_y=transform_y)
        
    #     if data_coords:
    #         # Sample and smooth the curve
    #         sampled_coords = sample_curve(data_coords, num_samples=200)
    #         smoothed_coords = smooth_curve(sampled_coords, window_size=5)
            
    #         # Sort by x value
    #         smoothed_coords.sort(key=lambda p: p[0])
            
    #         final_data_dict[label] = smoothed_coords
    #         y_label = "data values" if transform_y else "pixel coordinates"
    #         print(f"  Successfully extracted data for '{label}' ({len(smoothed_coords)} points, Y in {y_label})")
    #     else:
    #         print(f"  Failed to transform coordinates for '{label}'. Skipping.")
    
    print("\n--- Hybrid Data Extraction Complete ---")
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
    image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acsmacrolett.6b00250\acsmacrolett.6b00250\images_folder\page_2_img_1_a.jpg"
    
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acsmacrolett.6b00250\acsmacrolett.6b00250\images_folder\page_2_img_1_a.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma200800280\adma200800280\images_folder\page_3_img_3_a.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma.201002234\adma.201002234\images_folder\page_2_img_4_0.jpg"#
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\nmat2272\nmat2272\images_folder\nmat2272_fig1_b.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\c9py01720h\c9py01720h\images_folder\page_4_img_1_b.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\nmat2272\nmat2272\images_folder\nmat2272_fig2_b.jpg"
    final_dict = get_plot_data_dict_hybrid(image_path, axis_align_opt, plot_extract_opt)
    
    # Check if Y coordinates were transformed to data units
    y_in_data_units = False
    if final_dict and len(final_dict) > 0:
        # Get the first curve's data points
        first_curve_data = list(final_dict.values())[0]
        if first_curve_data:
            _, y_values = zip(*first_curve_data)
            y_min, y_max = min(y_values), max(y_values)
            y_in_data_units = y_min >= -0.1 and y_max <= 1.1 
            
            print(f"Y values range: {y_min:.3f} to {y_max:.3f}")
            if y_in_data_units:
                print("Y values appear to be in data units (0-1 range for absorbance/transmittance)")
            else:
                print("Y values appear to be in pixel coordinates")
    
    # Plot the results
    if final_dict:
        print("Plotting extracted data...")
        plot_extracted_data(final_dict, y_in_data_units=y_in_data_units)
    else:
        print("No data extracted. Check error messages above.")