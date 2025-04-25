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

import os
import pandas as pd # Keep if used elsewhere, maybe not needed now
from scipy.interpolate import interp1d
import os
import shutil
import pandas as pd

# Import your PlotDigitizer and AxisAlignment classes
from matagen.utils.plot_data_extraction.plot_digitizer import PlotDigitizer
from matagen.utils.axis_alignment.utils import AxisAlignment
from matagen.utils.plot_data_extraction.SpatialEmbeddings.src.utils import transforms as my_transforms

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
    "checkpoint_path": os.path.join(checkpoints_plot_extract_base, "checkpoint_0999.pth"),
    "dataset": {
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': "Plot2Spec_materials_eyes/data/input_plot_extraction",
            'type': 'test',
            'transform': my_transforms.get_transform([
                {
                    "name": "CustomResizePad",
                    "opts": {
                        'keys': ('image', 'instance','label'),
                        "is_test": True,
                    },
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
                {
                    'name': 'Normalize',
                    'opts': {
                        'keys': ('image'),
                        'p': -1,
                    }
                },
            ]),
        }
    },
    # "dataset": {
    #     'name': 'cityscapes',
    #     'kwargs': {
    #         'img_height': 256,     
    #         'img_width': 512,    
    #         'norm_mean': [0.485, 0.456, 0.406], 
    #         'norm_std': [0.229, 0.224, 0.225]  
    #     }
    # },
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

# def filter_legend_text(ocr_result):
#     """Filter OCR results to likely legend labels."""
#     labels = []
#     points = []
#     # Heuristics for filtering
#     prob_threshold = 0.4
#     min_len = 1
#     exclude_keywords = ['wavelength', 'nm', 'absorbance', 'intensity', 'transmittance', 'a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)']
#     require_alpha_start = True

#     for (bbox, text, prob) in ocr_result:
#         text_lower = text.lower()
#         # Apply filters
#         if prob < prob_threshold: continue
#         if len(text) < min_len: continue
#         if require_alpha_start and not text[0].isalpha(): continue
#         if text.replace('.','',1).isdigit(): continue
#         if any(keyword in text_lower for keyword in exclude_keywords): continue

#         labels.append(text)
#         points.append(bbox)

#     return labels, points
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

def improve_segmentation_mask(mask):
    """
    Apply morphological operations to improve connectivity in the segmentation mask
    """
    # Convert to uint8 if it's boolean
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.copy()
    
    # Visualize original mask
    visualize_step(mask_uint8, "Original Segmentation Mask", cmap='gray')
    
    # Apply closing to connect nearby components
    kernel_close = np.ones((3, 3), np.uint8)
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    visualize_step(mask_closed, "After Morphological Closing", cmap='gray')
    
    # Apply dilation to expand components
    kernel_dilate = np.ones((2, 2), np.uint8)
    mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=1)
    visualize_step(mask_dilated, "After Dilation", cmap='gray')
    
    # Find connected components and remove small noise
    num_labels, labels = cv2.connectedComponents(mask_dilated)
    min_size = 50  # Minimum component size to keep
    
    # Keep only components larger than min_size
    mask_filtered = np.zeros_like(mask_dilated)
    for i in range(1, num_labels):
        component_mask = (labels == i)
        if np.sum(component_mask) > min_size:
            mask_filtered[component_mask] = 255
    
    visualize_step(mask_filtered, "After Filtering Small Components", cmap='gray')
    
    return mask_filtered > 0  # Return boolean mask


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

# def preprocess_segmented_image(img_rgb, seg_map=None):
#     # Convert to 0-255 range
#     img_255 = (img_rgb * 255).astype(np.uint8)
#     visualize_step(img_255, "Original Image (0-255)")
    
#     # Apply segmentation mask if provided
#     if seg_map is not None:
#         visualize_step(seg_map, "Segmentation Mask", cmap='gray')
#         # Apply mask
#         masked_img = np.copy(img_255)
#         for i in range(3):
#             masked_img[:, :, i] = np.where(seg_map, img_255[:, :, i], 255)
#             # Now proceed with dilation and connecting points
#         kernel = np.ones((2, 2), np.uint8)
#         mask_dilated = cv2.dilate(masked_img, kernel, iterations=3)
#         # visualize_step(mask_dilated, "After Applying Segmentation Mask")
#     else:
#         masked_img = img_255    
#     # Convert to LAB color space for enhancement
#     lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    
#     # # Split channels
#     l, a, b = cv2.split(lab)
#     visualize_step(l, "L Channel Before CLAHE", cmap='gray')
    
#     # Apply CLAHE to L channel
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     visualize_step(cl, "L Channel After CLAHE", cmap='gray')
    
#     # # Merge channels
#     limg = cv2.merge((cl, a, b))
    
#     # Convert back to RGB
#     enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
#     visualize_step(enhanced_img, "Enhanced Image (Final)")    
#     # kernel = np.ones((2, 2), np.uint8)
#     # enhanced_img = cv2.dilate(enhanced_img, kernel, iterations=3)
#     return enhanced_img #masked_img # mask_dilated 


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

# def preprocess_segmented_image_v2(
#     img_rgb,                      # Input RGB image (float 0-1 or uint8 0-255)
#     seg_map=None,                 # Segmentation mask (boolean or 0/255, same HxW as img_rgb)
#     dilation_kernel_size=(7, 7),  # Kernel for thickening features (try adjusting)
#     dilation_iterations=3,        # Iterations for thickening (try adjusting)
#     clahe_clip_limit=3.0,         # Increased contrast limit
#     clahe_tile_grid_size=(8, 8),  # Standard tile grid size
#     saturation_boost=1.8,         # Factor to boost saturation (1.0 = no change)
#     brightness_boost=1.1          # Factor to boost brightness/value (1.0 = no change)
# ):
#     """
#     Preprocesses an image, optionally applying a segmentation mask,
#     enhancing contrast, boosting color intensity, and thickening features.

#     Args:
#         img_rgb: Input RGB image (numpy array, float 0-1 or uint8 0-255).
#         seg_map: Optional boolean or uint8 segmentation map (True/non-zero means foreground).
#         dilation_kernel_size: Tuple indicating the kernel size for dilation.
#         dilation_iterations: Number of times dilation is applied.
#         clahe_clip_limit: Clip limit for CLAHE contrast enhancement.
#         clahe_tile_grid_size: Tile grid size for CLAHE.
#         saturation_boost: Multiplicative factor for saturation boost in HSV space.
#         brightness_boost: Multiplicative factor for brightness/value boost in HSV space.

#     Returns:
#         Preprocessed/enhanced RGB image (numpy array, uint8 0-255).
#     """
#     # 1. Ensure input is uint8 [0-255]
#     if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
#         if img_rgb.max() <= 1.0:
#              img_255 = (img_rgb * 255).astype(np.uint8)
#         else: # Assume it's already 0-255 float
#              img_255 = img_rgb.astype(np.uint8)
#     elif img_rgb.dtype == np.uint8:
#         img_255 = img_rgb
#     else:
#         raise ValueError("Unsupported image dtype. Use float (0-1) or uint8 (0-255).")

#     if img_255.ndim == 2: # Grayscale input
#         print("Warning: Input image is grayscale. Converting to RGB.")
#         img_255 = cv2.cvtColor(img_255, cv2.COLOR_GRAY2RGB)
#     elif img_255.shape[2] != 3:
#          raise ValueError("Input image must be RGB (3 channels).")


    # visualize_step(img_255, "Original Image (0-255)")

    # # --- Mask Application ---
    # if seg_map is not None:
    #     visualize_step(seg_map, "Original Segmentation Mask", cmap='gray')

    #     # Ensure seg_map is boolean
    #     mask_boolean = seg_map.astype(bool)
    #     if mask_boolean.shape[0] != img_255.shape[0] or mask_boolean.shape[1] != img_255.shape[1]:
    #          raise ValueError("Segmentation map dimensions must match image dimensions.")

    #     # --- Thickening Lines/Features (Option 1: Dilate the Mask) ---
    #     # This approach makes the overall segmented region slightly larger,
    #     # effectively thickening features near the edges of the mask.
    #     if dilation_iterations > 0 and dilation_kernel_size[0] > 0 and dilation_kernel_size[1] > 0 :
    #         print(f"Dilating mask with kernel {dilation_kernel_size} and {dilation_iterations} iterations.")
    #         kernel = np.ones(dilation_kernel_size, np.uint8)
    #         # Ensure mask is uint8 for dilation
    #         mask_uint8 = mask_boolean.astype(np.uint8) * 255
    #         dilated_mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=dilation_iterations)
    #         # Convert back to boolean for application
    #         effective_mask_boolean = dilated_mask_uint8.astype(bool)
    #         visualize_step(dilated_mask_uint8, "Dilated Mask", cmap='gray')
    #     else:
    #         effective_mask_boolean = mask_boolean # Use original mask if no dilation

    #     # Apply the effective mask (original or dilated)
    #     # Create a white background and copy foreground pixels
    #     masked_img = np.full_like(img_255, (255, 255, 255), dtype=np.uint8) # White background

    #     # Handle broadcasting if mask is HxW and image is HxWxC
    #     if effective_mask_boolean.ndim == 2:
    #         masked_img[effective_mask_boolean] = img_255[effective_mask_boolean]
    #     elif effective_mask_boolean.ndim == 3 and effective_mask_boolean.shape[2] == 1:
    #          mask_squeezed = np.squeeze(effective_mask_boolean, axis=2)
    #          masked_img[mask_squeezed] = img_255[mask_squeezed]
    #     else: # Assuming mask is HxWxC boolean already
    #          masked_img[effective_mask_boolean] = img_255[effective_mask_boolean]

    #     visualize_step(masked_img, "Image with Mask Applied (White Background)")
    #     current_img = masked_img
    # else:
    #     # No mask provided, proceed with the original image
    #     current_img = img_255
    #     effective_mask_boolean = None # No mask to reapply later


    # # --- Contrast Enhancement (CLAHE on L channel) ---
    # lab = cv2.cvtColor(current_img, cv2.COLOR_RGB2LAB)
    # l, a, b = cv2.split(lab)
    # visualize_step(l, "L Channel Before CLAHE", cmap='gray')

    # clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    # cl = clahe.apply(l)
    # visualize_step(cl, "L Channel After CLAHE", cmap='gray')

    # limg = cv2.merge((cl, a, b))
    # enhanced_contrast_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    # visualize_step(enhanced_contrast_img, "After CLAHE Enhancement")
    # current_img = enhanced_contrast_img

    # # --- Color Intensity & Brightness Boost (HSV) ---
    # hsv = cv2.cvtColor(current_img, cv2.COLOR_RGB2HSV)
    # h, s, v = cv2.split(hsv)

    # # Boost Saturation (S channel)
    # if saturation_boost != 1.0:
    #     visualize_step(s, "S Channel Before Boost", cmap='gray')
    #     # Multiply and clip to prevent overflow/wrap-around, convert back to uint8
    #     s_boosted = np.clip(s * saturation_boost, 0, 255).astype(np.uint8)
    #     visualize_step(s_boosted, f"S Channel After x{saturation_boost} Boost", cmap='gray')
    # else:
    #     s_boosted = s

    # # Boost Brightness/Value (V channel) - Can also enhance intensity perception
    # if brightness_boost != 1.0:
    #      visualize_step(v, "V Channel Before Boost", cmap='gray')
    #      v_boosted = np.clip(v * brightness_boost, 0, 255).astype(np.uint8)
    #      visualize_step(v_boosted, f"V Channel After x{brightness_boost} Boost", cmap='gray')
    # else:
    #      v_boosted = v


    # # Merge boosted channels
    # final_hsv = cv2.merge((h, s_boosted, v_boosted))
    # final_img_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)


    # # --- Re-apply Mask (Optional but Recommended) ---
    # # Sometimes color/contrast adjustments can slightly affect pure white/black backgrounds.
    # # Re-applying the (potentially dilated) mask ensures the background stays perfectly white.
    # if effective_mask_boolean is not None:
    #     background_mask = ~effective_mask_boolean
    #     if background_mask.ndim == 2:
    #         final_img_rgb[background_mask] = [255, 255, 255]
    #     elif background_mask.ndim == 3 and background_mask.shape[2] == 1:
    #          background_mask_squeezed = np.squeeze(background_mask, axis=2)
    #          final_img_rgb[background_mask_squeezed] = [255, 255, 255]
    #     else: # Assuming HxWxC
    #          final_img_rgb[background_mask] = [255, 255, 255]
    #     visualize_step(final_img_rgb, "Final Image (Mask Re-Applied)")
    # else:
    #      visualize_step(final_img_rgb, "Final Image (No Mask)")


    # --- Thickening Lines/Features (Option 2: Dilate Final Image within Mask) ---
    # If Option 1 (dilating the mask) wasn't sufficient, you could try
    # dilating the *final* processed image, but *only* within the masked area.
    # This is more complex as it requires selectively applying dilation.
    # Example (Conceptual - might need refinement):
    # if seg_map is not None and dilation_iterations > 0:
    #     kernel = np.ones(dilation_kernel_size, np.uint8)
    #     dilated_final = cv2.dilate(final_img_rgb, kernel, iterations=dilation_iterations)
    #     # Only keep dilated pixels where the original mask was true
    #     final_img_rgb[effective_mask_boolean] = dilated_final[effective_mask_boolean]
    #     visualize_step(final_img_rgb, "Final Image (Post-Dilation within Mask)")
    # --- End Option 2 ---


    return final_img_rgb

def dilate_image_from_file(image_filename):
    """Reads an image file, applies invert-dilate-invert."""
    img_read = cv2.imread(image_filename)

    img = np.abs(img_read - 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.dilate(img, kernel, iterations=5)
    img = np.abs(img - 1)
    return img

def trace_spectral_curve(mask, start_point=None):
    """
    Trace a spectral curve through a binary mask with intelligent gap filling
    
    Args:
        mask: Binary mask of the curve
        start_point: Optional (x,y) starting point
        
    Returns:
        List of (x,y) coordinates following the curve
    """
    # Get initial points
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return []
    
    curve_points = []
    if start_point is None:
        # Find leftmost point as starting point
        leftmost_idx = np.argmin(x_coords)
        start_point = (x_coords[leftmost_idx], y_coords[leftmost_idx])
    
    # Used to mark visited points
    visited = np.zeros_like(mask, dtype=bool)
    
    # Add starting point
    curve_points.append(start_point)
    x, y = start_point
    visited[y, x] = True
    
    # Define neighborhood search pattern (8-connectivity)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Maximum allowed gap to bridge (in pixels)
    max_gap = 10
    
    # Continue until we can't find more points
    while True:
        # First try immediate neighbors
        found_next = False
        
        # Get current point
        curr_x, curr_y = curve_points[-1]
        
        # Try immediate neighbors first
        for dx, dy in neighbors:
            nx, ny = curr_x + dx, curr_y + dy
            if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and 
                mask[ny, nx] and not visited[ny, nx]):
                curve_points.append((nx, ny))
                visited[ny, nx] = True
                found_next = True
                break
        
        if found_next:
            continue
            
        # If no immediate neighbor, look for closest unvisited point within max_gap
        best_dist = float('inf')
        best_point = None
        
        # Search in progressively larger neighborhoods
        for gap in range(2, max_gap + 1):
            for dy in range(-gap, gap + 1):
                for dx in range(-gap, gap + 1):
                    if dx == 0 and dy == 0:
                        continue
                        
                    nx, ny = curr_x + dx, curr_y + dy
                    if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and 
                        mask[ny, nx] and not visited[ny, nx]):
                        dist = dx*dx + dy*dy  # squared distance
                        if dist < best_dist:
                            best_dist = dist
                            best_point = (nx, ny)
            
            # If we found a point at this gap size, use it
            if best_point is not None:
                break
        
        # If we found a point within max_gap, add it and continue
        if best_point is not None:
            curve_points.append(best_point)
            visited[best_point[1], best_point[0]] = True
        else:
            # We couldn't find any more points, so we're done
            break
    
    return curve_points


def trace_dense_curve(mask, min_point_distance=1, max_gap=5):
    """
    Trace a spectral curve with high point density
    
    Args:
        mask: Binary mask of the curve
        min_point_distance: Minimum distance between points (smaller = more points)
        max_gap: Maximum gap to bridge
        
    Returns:
        Dense list of (x,y) coordinates following the curve
    """
    # Get initial points
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return []
    
    # Sort points by x-coordinate for consistent left-to-right tracing
    idx = np.argsort(x_coords)
    x_coords = x_coords[idx]
    y_coords = y_coords[idx]
    
    # Group by x-coordinate (to handle multiple y values for same x)
    x_to_ys = {}
    for x, y in zip(x_coords, y_coords):
        if x not in x_to_ys:
            x_to_ys[x] = []
        x_to_ys[x].append(y)
    
    # Create a densely sampled path through the curve
    tracing_points = []
    
    # Start from leftmost point
    min_x = min(x_to_ys.keys())
    current_x = min_x
    if len(x_to_ys[current_x]) > 0:
        current_y = int(np.median(x_to_ys[current_x]))  # Take median y if multiple
        tracing_points.append((current_x, current_y))
    else:
        return []  # Empty curve
    
    # Trace points continuously from left to right
    visited_x = set([current_x])
    remaining_x = sorted(list(set(x_to_ys.keys()) - visited_x))
    
    while remaining_x:
        # Find the closest next x-coordinate
        closest_x = None
        min_distance = float('inf')
        
        for x in remaining_x:
            dist = abs(x - current_x)
            if dist < min_distance:
                min_distance = dist
                closest_x = x
        
        # If gap is too large, stop tracing
        if min_distance > max_gap:
            break
            
        # If we have a valid next x
        if closest_x is not None:
            # Get median y for this x
            median_y = int(np.median(x_to_ys[closest_x]))
            
            # If the gap between points is larger than min_point_distance,
            # interpolate additional points
            if min_distance > min_point_distance:
                # Linear interpolation between current point and next point
                for step in range(1, int(min_distance)):
                    interp_x = int(current_x + step)
                    interp_y = int(current_y + (median_y - current_y) * step / min_distance)
                    tracing_points.append((interp_x, interp_y))
            
            # Add the actual point
            tracing_points.append((closest_x, median_y))
            
            # Update current position
            current_x = closest_x
            current_y = median_y
            
            # Mark as visited
            visited_x.add(current_x)
            remaining_x = sorted(list(set(x_to_ys.keys()) - visited_x))
        else:
            break
    
    return tracing_points


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


def screen_for_valid_curves(img_array, image_path, num_colors=20, min_points=500):
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
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    # pil_img = PIL.Image.open(image_path) 
    pil_img.convert('RGB')
    # Apply palette conversion to quantize colors
    paletted = pil_img.convert('P', palette=PIL.Image.ADAPTIVE, colors=num_colors)
    
    # Get palette information
    palette = paletted.getpalette()
    paletted_array = np.asarray(paletted)
    img_shape = paletted_array.shape
    
    # Dictionary to store valid curves detected from palette
    valid_curves = {}
    
    # Screen each palette color
    print(f"Screening {num_colors} palette colors for valid curves...")
    
    for i in range(num_colors):
        # Get pixels for this color index
        y_coords, x_coords = np.where(paletted_array == i)
        plt.figure(figsize=(12, 8))
        plt.scatter(x_coords, y_coords, s=1, c='blue')
        plt.title(f"Original curve (color index {i})")
        plt.tight_layout()
        plt.show()
            
     
        if len(x_coords) < min_points:
            print(f"Color index {i}: Too few points ({len(x_coords)}), skipping.")
            continue
            
        # Skip if too many points (likely background or axes)
        if len(x_coords) > 0.2 * img_array.shape[0] * img_array.shape[1]:
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
        # if bin_coverage < 0.5:  # Less than 50% of bins have points
        #     print(f"Color index {i}: Poor x-distribution (coverage: {bin_coverage:.2f}), skipping.")
        #     continue
        
        # SCREENING CRITERIA #2: Check for multiple y-values per x-value
        # Group by x-coordinate and check y-variance
        x_to_ys = {}
        for x, y in zip(x_coords, y_coords):
            if x not in x_to_ys:
                x_to_ys[x] = []
            x_to_ys[x].append(y)
        
        # # Calculate average y-variance per x
        # y_variances = [np.var(ys) for ys in x_to_ys.values() if len(ys) > 1]
        # if y_variances:
        #     avg_y_variance = np.mean(y_variances)
        #     # Good curves should have low y-variance per x
        #     if avg_y_variance > 2000:  # Threshold may need tuning
        #         print(f"Color index {i}: High y-variance ({avg_y_variance:.2f}), likely multiple curves.")
        #         continue
        
        # NEW SCREENING CRITERIA #3: Derivative-based curvature analysis
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
                
            # Check for large derivative spikes (indicates discontinuities)
            has_spikes = any(abs(y_val) > 5 * deriv_std for y_val in y_deriv)

            # if np.mean(x_deriv)>200:
            #     continue
            
            print(f"Color index {i}: Derivative stats - mean: {deriv_mean:.2f}, std: {deriv_std:.2f}, CV: {deriv_cv:.2f}")
            
            # # Filter based on derivative metrics
            if deriv_cv > 5.0:  # Very high variation relative to mean
                print(f"Color index {i}: High derivative variation (CV: {deriv_cv:.2f}), likely noisy or multiple curves.")
                continue
                
            # if has_spikes:
            #     print(f"Color index {i}: Detected derivative spikes, likely discontinuous curve.")
            #     continue
            
            # Calculate mean curvature (using second derivative approximation)
            if len(y_deriv2) > 0:
                mean_curvature = np.mean(np.abs(y_deriv2))
                print(f"Color index {i}: Mean curvature: {mean_curvature:.4f}")
                
                # # Very high mean curvature often indicates noise
                if mean_curvature > 5:  # Threshold depends on image scale
                    print(f"Color index {i}: Excessive curvature, likely noise.")
                    continue
            
            # If we get here, the curve passed derivative screening
            print(f"Color index {i}: PASSED derivative screening")
            
            # Optional: Create diagnostic plots for derivatives
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
            # This is still valuable alongside derivative analysis
            # try:
            #     # Split into chunks for piecewise analysis
            #     chunk_size = len(cleaned_points) // 5
            #     if chunk_size > 20:
            #         smoothness_scores = []
                    
            #         for j in range(0, len(cleaned_points), chunk_size):
            #             chunk = cleaned_points[j:j+chunk_size]
            #             if len(chunk) < 10:
            #                 continue
                            
            #             chunk_x, chunk_y = zip(*chunk)
            #             # Use degree 2 polynomial for better curve fitting
            #             coeffs = np.polyfit(chunk_x, chunk_y, 2)
            #             poly = np.poly1d(coeffs)
            #             predicted_y = poly(chunk_x)
                        
            #             # Calculate mean squared error
            #             mse = np.mean((np.array(chunk_y) - predicted_y)**2)
            #             # Normalize by y-range in this chunk
            #             y_range = max(chunk_y) - min(chunk_y)
            #             if y_range > 0:
            #                 normalized_mse = mse / (y_range**2)
            #                 smoothness_scores.append(normalized_mse)
                    
            #         if smoothness_scores:
            #             avg_smoothness = np.mean(smoothness_scores)
            #             # Threshold for "good" smoothness
            #             if avg_smoothness > 0.1:  # Lower is smoother
            #                 print(f"Color index {i}: Poor smoothness score ({avg_smoothness:.4f}), skipping.")
            #                 continue
            # except Exception as e:
            #     print(f"Error in smoothness check for color index {i}: {e}")
            #     # Continue since we've already passed derivative checks
            
            # # If we get here, this color passed all screening criteria
            # print(f"Color index {i}: PASSED ALL screening criteria")
            
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


def apply_mask_and_thicken(
    img_rgb,                      # Input RGB image (float 0-1 or uint8 0-255)
    seg_map,                      # Segmentation mask (boolean or 0/255, same HxW as img_rgb)
    dilation_kernel_size=(2, 2),  # Kernel for thickening (matches your function)
    dilation_iterations=3         # Iterations for thickening (matches your function)
):
    """
    Applies a segmentation mask to an image and thickens the resulting
    foreground features using the invert-dilate-invert method.

    Args:
        img_rgb: Input RGB image (numpy array, float 0-1 or uint8 0-255).
        seg_map: Boolean or uint8 segmentation map (True/non-zero means foreground).
        dilation_kernel_size: Tuple indicating the kernel size for dilation.
        dilation_iterations: Number of times dilation is applied.

    Returns:
        Image with mask applied and foreground thickened (numpy array, uint8 0-255).
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
        print("Warning: Input image is grayscale. Converting to RGB.")
        img_255 = cv2.cvtColor(img_255, cv2.COLOR_GRAY2RGB)
    elif img_255.shape[2] != 3:
         raise ValueError("Input image must be RGB (3 channels).")

    visualize_step(img_255, "Original Image (0-255)")

    # --- Mask Application ---
    if seg_map is None:
        print("Warning: No segmentation map provided. Returning original image.")
        return img_255 # Or return None, depending on desired behavior

    visualize_step(seg_map, "Segmentation Mask", cmap='gray')

    # Ensure seg_map is boolean
    mask_boolean = seg_map.astype(bool)
    if mask_boolean.shape[0] != img_255.shape[0] or mask_boolean.shape[1] != img_255.shape[1]:
         raise ValueError("Segmentation map dimensions must match image dimensions.")

    # Apply the mask: Create a white background and copy foreground pixels
    masked_img = np.full_like(img_255, (255, 255, 255), dtype=np.uint8) # White background

    # Handle broadcasting if mask is HxW and image is HxWxC
    if mask_boolean.ndim == 2:
        masked_img[mask_boolean] = img_255[mask_boolean]
    elif mask_boolean.ndim == 3 and mask_boolean.shape[2] == 1:
         mask_squeezed = np.squeeze(mask_boolean, axis=2)
         masked_img[mask_squeezed] = img_255[mask_squeezed]
    else: # Assuming mask is HxWxC boolean already
         masked_img[mask_boolean] = img_255[mask_boolean]

    visualize_step(masked_img, "Image with Mask Applied")
    current_img = masked_img # This is the image to be thickened

    # --- Thickening Lines/Features (Using Your Invert-Dilate-Invert Method) ---
    print(f"Applying Invert-Dilate-Invert with kernel {dilation_kernel_size} and {dilation_iterations} iterations.")

    # 1. Invert (Use standard uint8 inversion: 255 - value)
    # White background (255) becomes black (0), dark curves become bright.
    img_inverted = 255 - current_img
    visualize_step(img_inverted, "Inverted Image for Dilation")

    # 2. Dilate the bright curves
    # Use MORPH_RECT as in your example
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
    img_dilated_inverted = cv2.dilate(img_inverted, kernel, iterations=dilation_iterations)
    visualize_step(img_dilated_inverted, "Dilated Inverted Image")

    # 3. Invert Back
    # Thick bright curves become thick dark curves, black background becomes white.
    final_thickened_img = 255 - img_dilated_inverted
    visualize_step(final_thickened_img, "Final Thickened Image")
    # --- End Thickening ---

    return final_thickened_img
##############################################################################################



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
    # valid_curves = {}
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

        # 3b.1 Prepare image (curves on white background, uint8)
        print("   Preparing image for saving...")
        if img_rgb.max() <= 1.0: # Check if img_rgb is float 0-1
             img_255 = (img_rgb * 255).astype(np.uint8)
        else: # Assume already uint8 or needs conversion differently
             img_255 = img_rgb.astype(np.uint8)

        img_to_save_uint8 = np.full_like(img_255, (255, 255, 255), dtype=np.uint8) # White background
        mask_boolean = seg_map.astype(bool) # Ensure mask is boolean

        # Apply mask (handle different mask dimensions)
        if mask_boolean.ndim == 2:
            img_to_save_uint8[mask_boolean] = img_255[mask_boolean]
        elif mask_boolean.ndim == 3 and mask_boolean.shape[2] == 1:
             mask_squeezed = np.squeeze(mask_boolean, axis=2)
             img_to_save_uint8[mask_squeezed] = img_255[mask_squeezed]
        else: # Assuming mask is HxWxC boolean already (less common)
             img_to_save_uint8[mask_boolean] = img_255[mask_boolean]

        # 3b.2 Save as JPEG
        img_rgb, seg_map, ins_map = res_map['img_rgb'], res_map['seg_map'], res_map['ins_map']
        masked_img = seg_map[..., None] * img_rgb
        masked_img[np.where((masked_img==[0,0,0]).all(axis=2))] = [1,1,1]
        im = Image.fromarray((255*masked_img).astype(np.uint8)) # Convert NumPy RGB to PIL Image
        im.save('spec.jpeg')
        new = dilate_image('spec.jpeg')
        plt.figure(figsize=(10, 8))
        plt.imshow(new, interpolation='none')
        plt.title('test from colab')
        plt.axis('off')
        plt.show()
        print("   Saving as JPEG (spec.jpeg)...")
        # Make filename unique or place in temp dir if running concurrently
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        jpeg_filename = f"{base_filename}_spec.jpeg" # Use unique name
        try:
            im_to_save = Image.fromarray((255*masked_img).astype(np.uint8)) # Convert NumPy RGB to PIL Image
            im_to_save.save(jpeg_filename)
            print(f"      Saved {jpeg_filename}.")
            print("      NOTE: JPEG saving introduces lossy compression artifacts!")
        except Exception as e:
            print(f"Error saving JPEG: {e}")
            return {} # Or handle error appropriately

        # 3b.3 Call dilation function (defined elsewhere) on saved JPEG
        print("   Calling dilation function on saved JPEG...")
         #new #dilate_image_from_file(jpeg_filename) # Use the function defined outside

        # plt.imshow(enhanced_img, interpolation='none')#, extent=[320,1000,400,0])#
        # plt.savefig('dilated_im.png')
        # plt.show()
        import PIL
        # img_rgb = PIL.Image.open('dilated_im.png')
        # img_rgb.convert('RGB')
        # enhanced_img = img_rgb
        enhanced_img = PIL.Image.open('dilated_im.png').resize((orig_w, orig_h), PIL.Image.BILINEAR)



        # if enhanced_img is None:
        #     print("Error: Dilation function failed.")
        #     # Clean up JPEG before returning
        #     if jpeg_filename and os.path.exists(jpeg_filename):
        #          try: os.remove(jpeg_filename); print(f"      Cleaned up {jpeg_filename}")
        #          except OSError: pass
        #     return {}

        # # (Optional but recommended) Clean up the temporary JPEG file now
        # if jpeg_filename and os.path.exists(jpeg_filename):
        #     try: os.remove(jpeg_filename); print(f"      Cleaned up {jpeg_filename}")
        #     except OSError as e: print(f"Warning: Failed to remove temp file {jpeg_filename}: {e}")



        # if seg_map is not None:
        #     print("Applying morphological closing to seg_map...")
        #     closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Adjust kernel size if needed
        #     seg_map = cv2.morphologyEx(seg_map.astype(np.uint8) * 255, cv2.MORPH_CLOSE, closing_kernel)
        #     seg_map = seg_map.astype(bool) # Convert back to boolean if needed by next function
        #     visualize_step(seg_map, "Closed Seg Map", cmap='gray') 
        
        # Preprocess the segmented image
        # improved_seg_map = improve_segmentation_mask(seg_map)
        # enhanced_img = preprocess_segmented_image_v2(img_rgb, seg_map)
        # enhanced_img = process_and_enhance_image(img_rgb, seg_map)
        # enhanced_img = dilate_image(enhanced_img)
        # enhanced_img = preprocess_segmented_image(img_rgb, seg_map)
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
    valid_curves = screen_for_valid_curves(enhanced_img,'dilated_im.png', min_points=10) ##################maybe change here with my method
    print(f"Found {len(valid_curves)} valid curve candidates after screening.")
    
    visualize_valid_curves(enhanced_img, valid_curves)
    print("Step 5b: Matching legend colors to valid curve candidates...")

    for label, target_color in legend_dict.items():
        print(f"Processing curve for '{label}'...")
        tolerance = tolerance_dict.get(label, 30)
        
        # Find the best matching valid curve
        best_match = None
        best_distance = float('inf')
        best_color = None
        
        for curve_color, curve_points in valid_curves.items():
            # Calculate color distance
            color_distance = np.sqrt(np.sum((np.array(curve_color) - np.array(target_color))**2))
            
            if color_distance < best_distance:
                best_distance = color_distance
                best_match = curve_points
                best_color = curve_color

        pixel_coords = best_match
        
        print(f"  Found {len(pixel_coords)} initial points for '{label}'")
        
        # First pass of outlier removal - use the more robust version
        filtered_coords = remove_outliers_robust(pixel_coords, window_size=15, std_threshold=2.0) ## could we try pyod here??
        print(f"  After robust outlier removal: {len(filtered_coords)} points")
        
        # Transform coordinates (X and possibly Y)
        data_coords = transform_coordinates(filtered_coords, scaled_axis_info, transform_y=transform_y)
        
        if data_coords:
            sampled_coords = sample_curve(data_coords, num_samples=1500)
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
    #good example 3/3 
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acsmacrolett.6b00250\acsmacrolett.6b00250\images_folder\page_2_img_1_a.jpg"
    
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma200800280\adma200800280\images_folder\page_3_img_3_a.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma.201002234\adma.201002234\images_folder\page_2_img_4_0.jpg"#
    image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\nmat2272\nmat2272\images_folder\nmat2272_fig1_b.jpg"
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