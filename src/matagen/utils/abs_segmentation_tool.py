import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import easyocr
import shutil 
import torch
from scipy.interpolate import interp1d
from PIL import Image
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
# Importing PlotDigitizer and AxisAlignment classes
from matagen.utils.plot_data_extraction.plot_digitizer import PlotDigitizer
from matagen.utils.axis_alignment.utils import AxisAlignment


# Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
checkpoints_base = os.path.join(project_root, 'checkpoints', 'axis_alignment')
checkpoints_plot_extract_base = os.path.join(project_root, 'checkpoints', 'plot_data_extraction')

print(f"Project Root (estimated): {project_root}")
print(f"Axis Checkpoints Base: {checkpoints_base}")
print(f"Plot Checkpoints Base: {checkpoints_plot_extract_base}")
print("-" * 20)

# Configuration of the AxisAlignment model (FCOS)
axis_align_opt = {
    # region detection
    "config_file": os.path.join(checkpoints_base, "fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py"),
    "checkpoint_file": os.path.join(checkpoints_base, "epoch_200.pth"),
    "refinement": True,
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
    "cuda": False, # Or True if GPU available and desired
    "display": False,
    "save": False, 
    "num_workers": 0,
    "checkpoint_path": os.path.join(checkpoints_plot_extract_base, "checkpoint_0999.pth"), # CRITICAL: Path to spatial embedding weights
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
            'num_classes': [3, 1], #output channels for model
        }
    }
}

def recognize_text(img_path, use_gpu=False):
    """Recognize text using EasyOCR."""
    try:
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        return reader.readtext(img_path)
    except Exception as e:
        print(f"Error initializing EasyOCR or reading text: {e}")
        return []

def filter_legend_text_selective(ocr_result):
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
    
    # Patterns to exclude from legend labels
    import re
    include_patterns = [
        r'^p\d+',                # Matches "P1", "P2", etc. (with or without wavelength info)
        r'^pentamer\s+[a-z]',    # Matches "Pentamer A", "Pentamer B", etc.
        r'^polymer\s+[a-z]',     # Matches "Polymer A", "Polymer B", etc.
        r'^oligomer\s+[a-z0-9]', # Matches "Oligomer 1", "Oligomer A", etc.
        r'^p[a-z]dot',           # Matches "PDot", "PPDot", etc.
        r'^prodot',              # Matches "ProDOT" variations
        r'^pedot',               # Matches "PEDOT" variations
        r'^compound\s+\d+',      # Matches "Compound 1", "Compound 2", etc.
    ]
    
    # Patterns to EXCLUDE - but only if they don't match the include patterns
    exclude_patterns = [
        r'^[0-9]+$',             # Just numbers like "600", "800"
        r'^[0-9\.]+$',           # Decimal numbers like "0.2", "1.0"
        r'^[0-9]+\s*nm$',        # Just wavelengths like "121nm"
        r'^nm$',                 # Just "nm"
        r'^[a-z][\)\]}]$',       # Subplot labels like "a)", "b)"
        r'wavelength',           # Axis labels
        r'intensity',            # Axis labels
        r'absorbance',           # Axis labels
        r'transmittance',        # Axis labels
        r'^[0-9]',              # NEVER start with a number
        r'00'                  # NEVER include '00' (catches G00)
    ]
    
    # Process each OCR result
    for bbox, text, prob in ocr_result:
        text = text.strip()
        
        # Skip empty text or low confidence results
        if not text or prob < min_confidence:
            continue
        
        # Skip if matches any exclude pattern
        should_exclude = False
        for pattern in exclude_patterns:
            if re.search(pattern, text.lower()):
                print(f"  Excluding based on pattern: '{text}'")
                should_exclude = True
                break
                
        if should_exclude:
            continue
        
        # Include if matches include patterns or meets other criteria
        should_include = False
        
        # Check include patterns
        for pattern in include_patterns:
            if re.search(pattern, text.lower()):
                should_include = True
                print(f"  Including based on pattern: '{text}'")
                break
                
        # Check if it looks like a chemical name (contains letters and not just numbers/units)
        if not should_include and len(text) > 1:
            # Contains letters and isn't just a unit
            has_letters = any(c.isalpha() for c in text)
            not_just_unit = not (text.lower().endswith('nm') and len(text) <= 5)
            
            if has_letters and not_just_unit and '(' not in text and ')' not in text:
                should_include = True
                print(f"  Including as possible chemical name: '{text}'")
        
        # Add to results if we should include this text
        if should_include:
            labels.append(text)
            bboxes.append(bbox)
    
    print(f"Filtered legend labels: {labels}")
    return labels, bboxes

def post_process_legend_items(labels, bboxes):
    """
    Post-process and clean up legend items to handle special cases
    and group related items.
    
    Args:
        labels: List of detected label texts
        bboxes: List of corresponding bounding boxes
    
    Returns:
        Tuple of (processed_labels, processed_bboxes)
    """
    if not labels:
        return [], []
        
    # Group labels that might belong together
    import numpy as np
    
    processed_labels = []
    processed_bboxes = []
    skip_indices = set()
    
    # Sort labels by vertical position (y-coordinate)
    label_positions = []
    for i, bbox in enumerate(bboxes):
        # Calculate center point of bounding box
        center_y = (bbox[0][1] + bbox[2][1]) / 2
        label_positions.append((i, center_y))
    
    # Sort by y-position
    label_positions.sort(key=lambda x: x[1])
    
    # Look for labels that appear on the same line
    for i in range(len(label_positions)):
        if i in skip_indices:
            continue
            
        curr_idx, curr_y = label_positions[i]
        curr_label = labels[curr_idx]
        curr_bbox = bboxes[curr_idx]
        
        # Look ahead for potential matches on same line
        found_match = False
        for j in range(i + 1, len(label_positions)):
            if j in skip_indices:
                continue
                
            next_idx, next_y = label_positions[j]
            next_label = labels[next_idx]
            next_bbox = bboxes[next_idx]
            
            # Check if they're on the same line (y-position within threshold)
            if abs(next_y - curr_y) < 15:  # Adjust threshold as needed
                # Check if next label is a wavelength or modifier for current label
                if next_label.startswith('(') or curr_label.endswith(':'):
                    # Combine them
                    combined_label = curr_label + ' ' + next_label
                    
                    # Create a combined bounding box
                    min_x = min(curr_bbox[0][0], next_bbox[0][0])
                    min_y = min(curr_bbox[0][1], next_bbox[0][1])
                    max_x = max(curr_bbox[2][0], next_bbox[2][0])
                    max_y = max(curr_bbox[2][1], next_bbox[2][1])
                    
                    combined_bbox = [
                        [min_x, min_y],  # top-left
                        [max_x, min_y],  # top-right
                        [max_x, max_y],  # bottom-right
                        [min_x, max_y]   # bottom-left
                    ]
                    
                    processed_labels.append(combined_label)
                    processed_bboxes.append(combined_bbox)
                    
                    skip_indices.add(i)
                    skip_indices.add(j)
                    found_match = True
                    break
        
        # If no match found, add the current label as is
        if not found_match and i not in skip_indices:
            processed_labels.append(curr_label)
            processed_bboxes.append(curr_bbox)
    
    return processed_labels, processed_bboxes


def get_label_colors_improved(image_path, label_bbox, sampling_methods=['left', 'below', 'right']):
    """
    Extract the color associated with a detected label bounding box using multiple sampling methods.
    
    Args:
        image_path: Path to the image
        label_bbox: Bounding box of the label
        sampling_methods: List of sampling methods to try ('left', 'below', 'right', 'above')
        
    Returns:
        Tuple of (color, method) where color is the RGB color and method is the sampling method used
    """
    import numpy as np
    from PIL import Image
    
    # Sampling parameters
    sample_width = 20  # Width of sampling area
    sample_height_factor = 0.5  # Height relative to text height
    distance = 10  # Distance from text bounding box
    
    # Extract bbox coordinates
    top_left, top_right, bottom_right, bottom_left = label_bbox
    
    # Calculate text dimensions
    text_width = top_right[0] - top_left[0]
    text_height = bottom_left[1] - top_left[1]
    text_center_x = (top_left[0] + top_right[0]) / 2
    text_center_y = (top_left[1] + bottom_left[1]) / 2
    
    # Load image
    try:
        with Image.open(image_path) as im:
            im_rgb = im.convert("RGB")
            img_width, img_height = im_rgb.size
            
            # Try each sampling method in order
            for method in sampling_methods:
                crop_box = None
                
                if method == 'left':
                    # Sample to the left of the text
                    crop_left = max(0, top_left[0] - sample_width - distance)
                    crop_right = max(0, top_left[0] - distance)
                    crop_top = max(0, text_center_y - text_height * sample_height_factor / 2)
                    crop_bottom = min(img_height, text_center_y + text_height * sample_height_factor / 2)
                
                elif method == 'right':
                    # Sample to the right of the text
                    crop_left = min(img_width, top_right[0] + distance)
                    crop_right = min(img_width, top_right[0] + distance + sample_width)
                    crop_top = max(0, text_center_y - text_height * sample_height_factor / 2)
                    crop_bottom = min(img_height, text_center_y + text_height * sample_height_factor / 2)
                
                elif method == 'below':
                    # Sample below the text
                    crop_left = max(0, text_center_x - sample_width / 2)
                    crop_right = min(img_width, text_center_x + sample_width / 2)
                    crop_top = min(img_height, bottom_left[1] + distance)
                    crop_bottom = min(img_height, bottom_left[1] + distance + text_height * sample_height_factor)
                
                elif method == 'above':
                    # Sample above the text
                    crop_left = max(0, text_center_x - sample_width / 2)
                    crop_right = min(img_width, text_center_x + sample_width / 2)
                    crop_top = max(0, top_left[1] - distance - text_height * sample_height_factor)
                    crop_bottom = max(0, top_left[1] - distance)
                
                # Skip if invalid crop box
                if crop_left >= crop_right or crop_top >= crop_bottom:
                    continue
                
                # Crop the image to the sampling area
                crop_box = (int(crop_left), int(crop_top), int(crop_right), int(crop_bottom))
                cropped_image = im_rgb.crop(crop_box)
                
                # Convert to numpy array
                cropped_image_rgb = np.array(cropped_image)
                
                # Create masks for non-white and non-black pixels
                mask_non_white = np.any(cropped_image_rgb < [250, 250, 250], axis=2)
                mask_non_black = np.any(cropped_image_rgb > [5, 5, 5], axis=2)
                mask = mask_non_white & mask_non_black
                
                # Get colored pixels
                pixels = cropped_image_rgb[mask]
                
                # If we found colored pixels, return their mean color
                if pixels.size > 0:
                    mean_color = np.mean(pixels, axis=0)
                    print(f"  Found color using '{method}' method: {mean_color.astype(int)}")
                    return mean_color, method
            
            # If no method worked, return None
            print(f"  Could not find color for bbox {label_bbox}")
            return None, None
                
    except Exception as e:
        print(f"Error getting label color: {e}")
        return None, None

def extract_legend_data_selective(image_path):
    """
    Improved function to extract legend labels and their associated colors.
    Focuses on chemical/material names and ignores wavelength information.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Dictionary mapping legend labels to their colors
    """
    import os
    from PIL import Image, ImageDraw
    import numpy as np
    
    print(f"Extracting legend data from {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return {}
    
    # Perform OCR
    print("Step 1: Performing OCR to find legend text...")
    ocr_results = recognize_text(image_path)
    if not ocr_results:
        print("Warning: OCR found no text.")
        return {}
    
    # Filter for potential legend labels
    print(f"Step 2: Filtering OCR results for potential legend labels (found {len(ocr_results)} text boxes)...")
    labels, bboxes = filter_legend_text_selective(ocr_results)
    if not labels:
        print("Warning: No potential legend labels found after filtering.")
        return {}
    
    # Post-process and clean up labels
    labels, bboxes = post_process_legend_items(labels, bboxes)
    print(f"  Found {len(labels)} potential legend labels after processing: {labels}")
    
    # Create a debug image to visualize label detection
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding boxes around detected labels
        for label, bbox in zip(labels, bboxes):
            # Convert points to integers
            box_points = [(int(x), int(y)) for x, y in bbox]
            
            # Draw rectangle
            draw.rectangle([box_points[0], box_points[2]], outline="red", width=2)
            
            # Draw label text
            draw.text((box_points[0][0], box_points[0][1] - 15), label, fill="red")
        
        # Save debug image
        debug_path = os.path.splitext(image_path)[0] + "_legend_debug.png"
        img.save(debug_path)
        print(f"Saved debug image to {debug_path}")
    except Exception as e:
        print(f"Could not create debug image: {e}")
    
    # Extract colors for each label
    print("Step 3: Extracting colors associated with labels...")
    legend_dict = {}
    
    # Try different sampling methods for different plot types
    sampling_methods_presets = [
        ['left', 'below', 'right'],  # Most common placement
        ['below', 'right', 'left'],  # For legends below the plot
        ['right', 'below', 'left'],  # For legends to the right
    ]
    
    for label, bbox in zip(labels, bboxes):
        # Try each preset of sampling methods
        for methods in sampling_methods_presets:
            color, method = get_label_colors_improved(image_path, bbox, sampling_methods=methods)
            if color is not None:
                legend_dict[label] = color
                print(f"  Found color for '{label}' using {method} method: {color.astype(int)}")
                break
                
        if label not in legend_dict:
            print(f"  Could not determine color for '{label}' after trying all methods.")
    
    return legend_dict


##################################################################################################################
#### ----Do not change-----------------

def enforce_curve_continuity(coords, max_gap=5):
    """Remove points that create large jumps in the curve."""
    if len(coords) < 3:
        return coords
        
    sorted_coords = sorted(coords, key=lambda p: p[0])
    result = [sorted_coords[0]]
    
    for i in range(1, len(sorted_coords)):
        prev = result[-1]
        curr = sorted_coords[i]
        
        # Calculate y-distance between consecutive points
        y_gap = abs(curr[1] - prev[1])
        
        # Only keep points that don't create large jumps
        if y_gap <= max_gap:
            result.append(curr)
            
    return result

def create_color_mask(img, target_color, tolerance=30):
    """
    Creates a binary mask where pixels close to target_color are 1.
    Properly handles color space conversion.
    
    Args:
        img: RGB numpy array (H,W,3) with values 0-255
        target_color: RGB color to match (0-255)
        tolerance: Color distance threshold
    """
    # Convert to proper shape and data type
    img_array = np.array(img)
    target_color = np.array(target_color, dtype=np.uint8).reshape(1, 1, 3)
    
    # OPTION 1: Match in RGB space using a weighted Euclidean distance
    r_weight, g_weight, b_weight = 0.30, 0.59, 0.11  # Perceptual weights
    
    r_diff = np.abs(img_array[:, :, 0] - target_color[0, 0, 0])
    g_diff = np.abs(img_array[:, :, 1] - target_color[0, 0, 1])
    b_diff = np.abs(img_array[:, :, 2] - target_color[0, 0, 2])
    
    weighted_distance = np.sqrt(
        (r_weight * r_diff)**2 + 
        (g_weight * g_diff)**2 + 
        (b_weight * b_diff)**2
    )
    
    # Create mask where distance is less than tolerance
    mask = weighted_distance < tolerance
    
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


def remove_outliers_multi_stage(pixel_coords, img_shape=None):
    """
    Comprehensive multi-stage outlier removal tailored for spectral plots.
    
    Args:
        pixel_coords: List of (x, y) coordinate tuples in pixel space
        img_shape: Optional tuple (height, width) of the image for boundary checking
        
    Returns:
        Filtered coordinate list
    """
    if len(pixel_coords) < 10:
        return pixel_coords
    
    print(f"Starting multi-stage outlier removal with {len(pixel_coords)} points")
    
    # Stage 1: Remove points outside image boundaries if img_shape is provided
    if img_shape is not None:
        height, width = img_shape[:2]
        in_bounds = [(x, y) for x, y in pixel_coords if 0 <= x < width and 0 <= y < height]
        print(f"Boundary check: removed {len(pixel_coords) - len(in_bounds)} points")
        pixel_coords = in_bounds
    
    # Stage 2: Apply PyOD-based outlier detection if available
    try:
        # Import PyOD only if available
        from pyod.models.knn import KNN
        
        # Convert to numpy array for PyOD
        coords_array = np.array(pixel_coords)
        
        # Configure KNN detector
        n_neighbors = min(5, len(pixel_coords) - 1)
        detector = KNN(n_neighbors=n_neighbors, method='largest', contamination=0.1)
        
        # Fit and predict
        detector.fit(coords_array)
        outlier_predictions = detector.predict(coords_array)
        
        # Keep only non-outlier points
        inlier_indices = np.where(outlier_predictions == 0)[0]
        filtered_coords = [pixel_coords[i] for i in inlier_indices]
        
        print(f"PyOD KNN detection: removed {len(pixel_coords) - len(filtered_coords)} outliers")
        pixel_coords = filtered_coords
    except ImportError:
        print("PyOD not available, skipping KNN outlier detection")
    except Exception as e:
        print(f"Error during PyOD outlier detection: {e}")
    
    # Stage 3: Apply curve-specific outlier detection
    try:
        filtered_coords = remove_curve_outliers(pixel_coords)
        print(f"Curve-aware detection: removed {len(pixel_coords) - len(filtered_coords)} outliers")
        pixel_coords = filtered_coords
    except Exception as e:
        print(f"Error during curve-aware outlier detection: {e}")
    
    # Stage 4: Final pass with robust statistical method
    try:
        filtered_coords = remove_outliers_robust(pixel_coords, window_size=11, std_threshold=3.0)
        print(f"Robust statistical detection: removed {len(pixel_coords) - len(filtered_coords)} outliers")
        pixel_coords = filtered_coords
    except Exception as e:
        print(f"Error during robust statistical outlier detection: {e}")
    
    print(f"Multi-stage outlier removal complete: {len(pixel_coords)} points remaining")
    return pixel_coords


def remove_curve_outliers(coords, window_size=7, distance_factor=3.0):
    """
    Detect and remove outliers by analyzing curve continuity in sorted x-direction.
    This is particularly effective for plot lines that should be continuous.
    
    Args:
        coords: List of (x, y) coordinate tuples
        window_size: Window size for local outlier detection
        distance_factor: Factor determining the maximum allowed distance from local regression
        
    Returns:
        Filtered coordinates
    """
    import numpy as np
    from scipy.signal import savgol_filter
    
    if len(coords) <= window_size:
        return coords
    
    # Sort by x-coordinate
    sorted_coords = sorted(coords, key=lambda p: p[0])
    x_vals = np.array([p[0] for p in sorted_coords])
    y_vals = np.array([p[1] for p in sorted_coords])
    
    # Apply Savitzky-Golay filter to get expected y values
    half_window = min(window_size // 2, 3)
    poly_order = min(half_window, 2)  # polynomial order must be less than window_size
    y_expected = savgol_filter(y_vals, window_size, poly_order)
    
    # Calculate residuals (distance from expected curve)
    residuals = np.abs(y_vals - y_expected)
    
    # Compute local threshold using rolling window
    thresholds = np.ones_like(residuals) * np.inf
    for i in range(len(residuals)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(residuals), i + window_size // 2 + 1)
        local_residuals = residuals[start_idx:end_idx]
        
        # Use median absolute deviation for robust threshold
        local_median = np.median(local_residuals)
        local_mad = np.median(np.abs(local_residuals - local_median))
        
        # Set threshold as median + factor * MAD
        thresholds[i] = local_median + distance_factor * local_mad * 1.4826  # Scale factor to approximate standard deviation
    
    # Find indices of inliers
    inlier_indices = residuals <= thresholds
    
    # Output results
    filtered_coords = [sorted_coords[i] for i in range(len(sorted_coords)) if inlier_indices[i]]
    outliers_removed = len(sorted_coords) - len(filtered_coords)
    print(f"Curve-aware outlier removal: removed {outliers_removed} outliers from {len(sorted_coords)} points")
    
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

from scipy.signal import savgol_filter


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


def remove_spikes(coords, window_size=5, threshold=3.0):
    """
    Remove spike artifacts from spectral data.
    
    Args:
        coords: List of (x, y) coordinate tuples
        window_size: Size of the window to use for detecting spikes
        threshold: Threshold for spike detection (standard deviations)
        
    Returns:
        Coordinates with spikes removed
    """
    if len(coords) < window_size + 2:
        return coords
        
    import numpy as np
    
    # Sort by x-coordinate
    sorted_coords = sorted(coords, key=lambda p: p[0])
    x_vals = np.array([p[0] for p in sorted_coords])
    y_vals = np.array([p[1] for p in sorted_coords])
    
    # Calculate first and second derivatives
    dy = np.diff(y_vals)
    d2y = np.diff(dy)
    
    # Identify spikes using second derivative
    spike_indices = []
    for i in range(len(d2y)):
        # Calculate local statistics in window
        start = max(0, i - window_size // 2)
        end = min(len(d2y), i + window_size // 2 + 1)
        window = d2y[start:end]
        
        # Calculate z-score
        mean = np.mean(window)
        std = np.std(window) + 1e-10  # Avoid division by zero
        z_score = abs(d2y[i] - mean) / std
        
        # Mark as spike if z-score exceeds threshold
        if z_score > threshold:
            spike_indices.append(i + 1)  # +1 because d2y is shift by 2
    
    # Remove spikes
    filtered_coords = [p for i, p in enumerate(sorted_coords) if i not in spike_indices]
    
    print(f"Spike removal: removed {len(spike_indices)} spike points")
    return filtered_coords


def remove_duplicate_x(coords, strategy='mean'):
    """
    Remove data points with duplicate x values using various strategies.
    
    Args:
        coords: List of (x, y) coordinate tuples
        strategy: How to handle duplicates ('mean', 'first', 'last', 'max', 'min')
        
    Returns:
        Coordinates with unique x values
    """
    if not coords:
        return coords
        
    import numpy as np
    
    # Sort by x-coordinate
    sorted_coords = sorted(coords, key=lambda p: p[0])
    
    # Group by x value
    x_groups = {}
    for x, y in sorted_coords:
        if x not in x_groups:
            x_groups[x] = []
        x_groups[x].append(y)
    
    # Apply strategy to each group
    unique_coords = []
    for x, y_values in x_groups.items():
        if len(y_values) == 1:
            # No duplicates
            unique_coords.append((x, y_values[0]))
        else:
            # Handle duplicates based on strategy
            if strategy == 'mean':
                y = np.mean(y_values)
            elif strategy == 'first':
                y = y_values[0]
            elif strategy == 'last':
                y = y_values[-1]
            elif strategy == 'max':
                y = max(y_values)
            elif strategy == 'min':
                y = min(y_values)
            else:
                # Default to mean
                y = np.mean(y_values)
            
            unique_coords.append((x, y))
    
    # Sort by x again
    unique_coords.sort(key=lambda p: p[0])
    
    num_removed = len(sorted_coords) - len(unique_coords)
    if num_removed > 0:
        print(f"Removed {num_removed} points with duplicate x values using '{strategy}' strategy")
    
    return unique_coords

def process_spectral_curve(data_coords, smoothing_factor=0.3, min_samples=300):
    """
    Comprehensive processing for spectral curves while preserving details.
    
    Args:
        data_coords: List of (x, y) data coordinates
        smoothing_factor: Smoothing factor (0.0-1.0)
        min_samples: Minimum number of points to sample (more may be used if needed)
        
    Returns:
        Processed curve coordinates
    """
    if not data_coords or len(data_coords) < 10:
        return data_coords
    
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Remove spikes
    cleaned_coords = remove_spikes(data_coords, window_size=7, threshold=4.0)
    
    # Apply strong LOWESS smoothing
    smoothed_coords = smooth_spectral_curve(cleaned_coords, smoothing_factor=smoothing_factor)
    
    # Sort by x value
    smoothed_coords.sort(key=lambda p: p[0])
    
    # IMPORTANT: Remove any duplicate x values
    unique_coords = remove_duplicate_x(smoothed_coords, strategy='mean')
    
    # Determine appropriate sampling density
    # Use more points than the minimum if the original data had higher density
    num_samples = max(min_samples, len(data_coords))
    
    # Regular sampling
    x_vals, y_vals = zip(*unique_coords)
    x_min, x_max = min(x_vals), max(x_vals)
    
    try:
        # Try cubic interpolation first
        interp_func = interp1d(x_vals, y_vals, kind='cubic', bounds_error=False, 
                           fill_value=(y_vals[0], y_vals[-1]))
    except ValueError as e:
        # Fall back to linear interpolation if cubic fails
        print(f"Warning: Cubic interpolation failed ({str(e)}), falling back to linear")
        interp_func = interp1d(x_vals, y_vals, kind='linear', bounds_error=False,
                           fill_value=(y_vals[0], y_vals[-1]))
    
    # Create regular sampling of x values
    x_samples = np.linspace(x_min, x_max, num_samples)
    
    # Get corresponding y values
    y_samples = interp_func(x_samples)
    
    # Create final coordinates
    final_coords = list(zip(x_samples, y_samples))
    
    print(f"Processed spectral curve: {len(data_coords)} input points → {len(final_coords)} output points")
    
    return final_coords

def remove_spikes(coords, window_size=5, threshold=3.0):
    """
    Remove spike artifacts from spectral data.
    
    Args:
        coords: List of (x, y) coordinate tuples
        window_size: Size of the window to use for detecting spikes
        threshold: Threshold for spike detection (standard deviations)
        
    Returns:
        Coordinates with spikes removed
    """
    if len(coords) < window_size + 2:
        return coords
        
    import numpy as np
    
    # Sort by x-coordinate
    sorted_coords = sorted(coords, key=lambda p: p[0])
    x_vals = np.array([p[0] for p in sorted_coords])
    y_vals = np.array([p[1] for p in sorted_coords])
    
    try:
        # Calculate first differences
        dy = np.diff(y_vals)
        
        # Calculate second differences (acceleration)
        d2y = np.diff(dy)
        
        # Identify spikes using second derivative
        spike_indices = []
        for i in range(len(d2y)):
            # Calculate local statistics in window
            start = max(0, i - window_size // 2)
            end = min(len(d2y), i + window_size // 2 + 1)
            window = d2y[start:end]
            
            # Calculate z-score
            mean = np.mean(window)
            std = np.std(window) + 1e-10  # Avoid division by zero
            z_score = abs(d2y[i] - mean) / std
            
            # Mark as spike if z-score exceeds threshold
            if z_score > threshold:
                spike_indices.append(i + 1)  # +1 because d2y is shift by 2
        
        # Remove spikes
        filtered_coords = [p for i, p in enumerate(sorted_coords) if i not in spike_indices]
        
        print(f"Spike removal: removed {len(spike_indices)} spike points")
        return filtered_coords
    except Exception as e:
        print(f"Error during spike removal: {e}, skipping spike removal")
        return coords

def smooth_spectral_curve(coords, smoothing_factor=0.3, min_points=20):
    """
    Specialized smoothing for spectral curves using LOWESS with high smoothing factor.
    
    Args:
        coords: List of (x, y) coordinate tuples
        smoothing_factor: LOWESS smoothing factor (0.0-1.0, higher = smoother)
        min_points: Minimum points required
        
    Returns:
        Smoothed coordinates
    """
    import numpy as np
    
    if len(coords) < min_points:
        print(f"Warning: Not enough points ({len(coords)}) for smoothing")
        return coords
        
    # Sort by x-coordinate
    sorted_coords = sorted(coords, key=lambda p: p[0])
    x_vals, y_vals = zip(*sorted_coords)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    try:
        # Try LOWESS smoothing
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y_vals, x_vals, frac=smoothing_factor, it=10, return_sorted=True)
        return list(zip(smoothed[:, 0], smoothed[:, 1]))
    except Exception as e:
        print(f"Error during LOWESS smoothing: {e}")
        try:
            # Fall back to Savitzky-Golay if LOWESS fails
            from scipy.signal import savgol_filter
            # Make sure window size is odd and less than data length
            window_size = min(21, (len(y_vals) // 2) * 2 - 1)
            if window_size < 3:
                return sorted_coords
                
            poly_order = min(3, window_size - 1)
            y_smoothed = savgol_filter(y_vals, window_size, poly_order)
            return list(zip(x_vals, y_smoothed))
        except Exception as e2:
            print(f"Error during Savitzky-Golay smoothing: {e2}")
            return sorted_coords

# 1. Modify extract_curve_by_color to include outlier removal AND continuity check internally
def extract_curve_by_color(img_array, target_color, tolerance=30, min_points=10, continuity_max_gap=7):
    """
    Extracts curve pixels using improved color matching in HSV space.
    
    Args:
        img_array: RGB numpy array (0-255)
        target_color: RGB color to match (0-255)
        tolerance: Color distance threshold
        min_points: Minimum points to consider after initial color match
        continuity_max_gap: Max pixel jump allowed in Y for continuity check
        
    Returns:
        List of filtered and continuous pixel coordinates
    """
    # Create color mask using improved HSV matching
    mask = create_color_mask(img_array, target_color, tolerance)
    
    # Get coordinates of matching pixels
    y_coords, x_coords = np.where(mask)
    
    if len(x_coords) < min_points:
        print(f"  Color extraction ({target_color}): Not enough points ({len(x_coords)}) matched the color.")
        return []
    
    # Combine coordinates
    points = list(zip(x_coords, y_coords))
    print(f"  Color extraction ({target_color}): Found {len(points)} initial points (tolerance={tolerance}).")
    
    # IMPROVED: Use clustering to identify the main curve when multiple disconnected regions exist
    if len(points) > min_points * 5:  # Only cluster if we have lots of points
        try:
            from sklearn.cluster import DBSCAN
            
            # Convert points to array for clustering
            points_array = np.array(points)
            
            # Determine appropriate epsilon based on image size
            img_diagonal = np.sqrt(img_array.shape[0]**2 + img_array.shape[1]**2)
            eps = max(5, img_diagonal * 0.01)  # Adaptive epsilon
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points_array)
            labels = clustering.labels_
            
            # Count points in each cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Skip noise points (label -1)
            if -1 in unique_labels:
                noise_idx = np.where(unique_labels == -1)[0][0]
                unique_labels = np.delete(unique_labels, noise_idx)
                counts = np.delete(counts, noise_idx)
            
            if len(unique_labels) > 0:
                # Find the largest cluster
                largest_cluster = unique_labels[np.argmax(counts)]
                
                # Filter points to keep only the largest cluster
                points = [p for i, p in enumerate(points) if labels[i] == largest_cluster]
                print(f"  Clustering found {len(unique_labels)} clusters, keeping largest with {len(points)} points.")
        except Exception as e:
            print(f"  Warning: Clustering failed ({e}), using all matched points.")
    
    # Apply multi-stage outlier removal (pixel level)
    filtered_points = remove_outliers_multi_stage(points, img_shape=img_array.shape)
    
    if len(filtered_points) < 3:  # Need at least 3 points for continuity check
        print(f"  Too few points ({len(filtered_points)}) remain after outlier removal.")
        return filtered_points
    
    # Apply continuity filter
    continuous_points = enforce_curve_continuity(filtered_points, max_gap=continuity_max_gap)
    print(f"  After enforce_curve_continuity (max_gap={continuity_max_gap}): {len(continuous_points)} points remain.")
    
    return continuous_points

# 2. Ensure transform_coordinates uses clamping (as suggested previously)
def transform_coordinates(pixel_coords, axis_info, transform_y=True):
    """
    Transforms pixel coordinates to data coordinates using axis info.
    Uses CLAMPING instead of extrapolation for safety.
    
    Args:
        pixel_coords: List of (x_pixel, y_pixel) tuples
        axis_info: Dict with axis info (scaled to the image space of pixel_coords)
        transform_y: Whether to transform Y coordinates
    
    Returns:
        List of (x_data, y_data or y_pixel) tuples
    """
    if not pixel_coords:
        return []
    
    # Check X-axis info
    if not axis_info or 'x' not in axis_info or \
       not axis_info['x'].get('pixels') or not axis_info['x'].get('values') or \
       len(axis_info['x']['pixels']) < 2:
        print("Warning: X-axis information is missing or incomplete for transformation.")
        return None
        
    # Check Y-axis info if transforming Y
    y_transformable = False
    if transform_y:
        if 'y' in axis_info and \
           axis_info['y'].get('pixels') and axis_info['y'].get('values') and \
           len(axis_info['y']['pixels']) >= 2:
            y_transformable = True
        else:
            print("Warning: Y-axis information is missing or incomplete. Keeping Y in pixel coordinates.")
            transform_y = False  # Force keeping Y as pixels

    try:
        x_pixels_curve, y_pixels_curve = zip(*pixel_coords)
        x_pixels_curve = np.array(x_pixels_curve)
        y_pixels_curve = np.array(y_pixels_curve)

        # X-Axis Transformation
        x_tick_pixels = np.array(axis_info['x']['pixels'])
        x_tick_values = np.array(axis_info['x']['values'])
        sort_idx_x = np.argsort(x_tick_pixels)
        x_tick_pixels_sorted = x_tick_pixels[sort_idx_x]
        x_tick_values_sorted = x_tick_values[sort_idx_x]
        
        # FIXED: Explicitly clamp values to the domain of the axis
        min_x_pixel = x_tick_pixels_sorted[0]
        max_x_pixel = x_tick_pixels_sorted[-1]
        min_x_value = x_tick_values_sorted[0]
        max_x_value = x_tick_values_sorted[-1]
        
        # Clamp x_pixels_curve to the axis range
        x_pixels_clamped = np.clip(x_pixels_curve, min_x_pixel, max_x_pixel)
        
        # Use linear interpolation with explicit fill values at boundaries
        interp_x = interp1d(x_tick_pixels_sorted, x_tick_values_sorted, 
                            kind='linear',
                            bounds_error=False,
                            fill_value=(min_x_value, max_x_value))
        x_data = interp_x(x_pixels_clamped)
        
        # Recover original points that were within bounds
        in_bounds_mask = (x_pixels_curve >= min_x_pixel) & (x_pixels_curve <= max_x_pixel)
        
        # Y-Axis Transformation (if requested and possible)
        if transform_y and y_transformable:
            y_tick_pixels = np.array(axis_info['y']['pixels'])
            y_tick_values = np.array(axis_info['y']['values'])
            sort_idx_y = np.argsort(y_tick_pixels)
            y_tick_pixels_sorted = y_tick_pixels[sort_idx_y]
            y_tick_values_sorted = y_tick_values[sort_idx_y]
            
            # FIXED: Explicitly clamp values to the domain of the axis
            min_y_pixel = y_tick_pixels_sorted[0]
            max_y_pixel = y_tick_pixels_sorted[-1]
            min_y_value = y_tick_values_sorted[0]
            max_y_value = y_tick_values_sorted[-1]
            
            # Clamp y_pixels_curve to the axis range
            y_pixels_clamped = np.clip(y_pixels_curve, min_y_pixel, max_y_pixel)
            
            # Use linear interpolation with explicit fill values at boundaries
            interp_y = interp1d(y_tick_pixels_sorted, y_tick_values_sorted, 
                                kind='linear',
                                bounds_error=False,
                                fill_value=(min_y_value, max_y_value))
            y_data = interp_y(y_pixels_clamped)
            
            # Update in_bounds_mask to include y-bounds
            y_in_bounds = (y_pixels_curve >= min_y_pixel) & (y_pixels_curve <= max_y_pixel)
            in_bounds_mask = in_bounds_mask & y_in_bounds
            
            transformed_coords = list(zip(x_data, y_data))
        else:
            # Keep Y as pixel coordinates
            transformed_coords = list(zip(x_data, y_pixels_curve))
        
        # Print statistics about clamping
        num_clamped = len(x_pixels_curve) - np.sum(in_bounds_mask)
        if num_clamped > 0:
            percent_clamped = (num_clamped / len(x_pixels_curve)) * 100
            print(f"  Note: {num_clamped} points ({percent_clamped:.1f}%) were outside axis bounds and clamped.")
            
        return transformed_coords

    except Exception as e:
        print(f"Error during coordinate transformation: {e}")
        # import traceback
        # traceback.print_exc()  # Uncomment for detailed error
        return None


def process_spectral_curve(data_coords, smoothing_factor=0.15, num_samples=300):
    """
    Comprehensive processing for spectral curves while preserving spectral features.
    
    Args:
        data_coords: List of (x, y) data coordinates
        smoothing_factor: Smoothing factor (0.0-1.0, lower = less smoothing)
        num_samples: Target number of points in the final curve
        
    Returns:
        Processed curve coordinates
    """
    if not data_coords or len(data_coords) < 10:
        print(f"Warning: Too few points ({len(data_coords) if data_coords else 0}) for spectral processing.")
        return data_coords
    
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Remove duplicate x values first (important for interpolation)
    unique_coords = remove_duplicate_x(data_coords, strategy='mean')
    if len(unique_coords) < 5:
        print(f"Warning: Too few unique x-values ({len(unique_coords)}) after removing duplicates.")
        return unique_coords
    
    # Sort by x value
    sorted_coords = sorted(unique_coords, key=lambda p: p[0])
    
    # Remove spikes (less aggressive for spectral data to preserve features)
    cleaned_coords = remove_spikes(sorted_coords, window_size=7, threshold=5.0)
    if len(cleaned_coords) < 5:
        print(f"Warning: Too few points ({len(cleaned_coords)}) after spike removal.")
        return sorted_coords  # Fall back to sorted but unfiltered coords
    
    # Determine x-range
    x_vals, y_vals = zip(*cleaned_coords)
    x_min, x_max = min(x_vals), max(x_vals)
    
    # FIXED: Adaptive smoothing based on data density
    # Use less smoothing for sparse data, more for dense data
    point_density = len(cleaned_coords) / (x_max - x_min)
    adaptive_smoothing = min(smoothing_factor, 1.0 / (point_density + 1))
    print(f"  Adaptive smoothing: {adaptive_smoothing:.3f} (based on point density: {point_density:.1f} pts/unit)")
    
    try:
        # Apply LOWESS smoothing with adaptive parameter
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y_vals, x_vals, frac=adaptive_smoothing, it=3, return_sorted=True)
        smoothed_coords = list(zip(smoothed[:, 0], smoothed[:, 1]))
    except Exception as e:
        print(f"  Warning: LOWESS smoothing failed ({e}), falling back to Savitzky-Golay.")
        try:
            # Fall back to Savitzky-Golay filter
            from scipy.signal import savgol_filter
            # Ensure window size is odd and less than data length
            window_size = min(15, max(5, (len(y_vals) // 10) * 2 + 1))
            poly_order = min(3, window_size - 1)
            y_smoothed = savgol_filter(y_vals, window_size, poly_order)
            smoothed_coords = list(zip(x_vals, y_smoothed))
        except Exception as e2:
            print(f"  Warning: Savitzky-Golay smoothing also failed ({e2}). Using original cleaned data.")
            smoothed_coords = cleaned_coords
    
    # FIXED: Adaptive sampling based on curve complexity
    # Estimate curve complexity using rate of change in y values
    x_array = np.array([p[0] for p in smoothed_coords])
    y_array = np.array([p[1] for p in smoothed_coords])
    
    # Calculate the absolute differences between adjacent y-values
    y_diffs = np.abs(np.diff(y_array))
    complexity = np.sum(y_diffs) / (y_array.max() - y_array.min()) if y_diffs.size > 0 else 1
    
    # Adjust number of samples based on complexity
    adaptive_samples = min(int(num_samples * complexity * 1.5), 1000)
    adaptive_samples = max(adaptive_samples, 100)  # Ensure minimum sample count
    print(f"  Using {adaptive_samples} samples (complexity factor: {complexity:.2f})")
    
    try:
        # Create regular sampling with adaptive sample count
        # Try cubic interpolation first (better for spectral curves)
        try:
            interp_func = interp1d(x_array, y_array, kind='cubic', bounds_error=False,
                               fill_value=(y_array[0], y_array[-1]))
        except ValueError as e:
            print(f"  Warning: Cubic interpolation failed ({e}), falling back to linear.")
            interp_func = interp1d(x_array, y_array, kind='linear', bounds_error=False,
                               fill_value=(y_array[0], y_array[-1]))
        
        # Create regular sampling of x values
        x_samples = np.linspace(x_min, x_max, adaptive_samples)
        
        # Get corresponding y values
        y_samples = interp_func(x_samples)
        
        # Create final coordinates
        final_coords = list(zip(x_samples, y_samples))
        
        print(f"  Processed spectral curve: {len(data_coords)} input → {len(final_coords)} output points")
        return final_coords
        
    except Exception as e:
        print(f"  Warning: Resampling failed ({e}). Returning smoothed coordinates.")
        return smoothed_coords
    
### --- This is the main function with all the digitalization steps and at the end we get the dictionary
### with the absorption spectra
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
    legend_dict = extract_legend_data_selective(image_path) #extract_legend_data
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
    
    # === Step 5: Generalized Curve Extraction ===
    print("\nStep 5: Generalized Curve Extraction...")
    tolerance_dict = adaptive_color_tolerance(enhanced_img, legend_dict, initial_tolerance=35, max_tolerance=80)

    # --- Part B: Detect Unlabeled Curve Candidates ---
    print("\nStep 5b: Detecting unlabeled curve candidates...")
    unlabeled_candidates = {} # Store potential unlabeled curves {candidate_id: pixel_coords}

  # --- Configuration ---
    MIN_POINTS_LABELED = 10
    MIN_POINTS_UNLABELED = 50
    MIN_POINTS_FINAL = 10
    CONTINUITY_MAX_GAP = 7
    SMOOTHING_FACTOR = 0.15
    NUM_SAMPLES = 500
    CLUSTER_SCORE_THRESHOLD = 0.6
    OVERLAP_THRESHOLD = 0.5

    final_data_dict = {}
    # Ensure enhanced_img is defined (from Step 3) before creating the mask
    if 'enhanced_img' not in locals() or enhanced_img is None:
        print("ERROR: enhanced_img not available to initialize processed_pixel_mask!")
        return {} # Or handle appropriately
    processed_pixel_mask = np.zeros(enhanced_img.shape[:2], dtype=bool) # Mask needs image dimensions

    pixel_coords_map = {}
    filtered_coords_map = {}

    # Determine if Y axis was defaulted earlier (NEEDS TO BE SET BASED ON STEP 2 RESULTS)
    # Make sure axis_info_for_transform is populated from Step 2
    y_axis_was_defaulted = (len(axis_info_for_transform.get('y', {}).get('pixels', [])) < 2)
    print(f"Step 5: Y-axis defaulted: {y_axis_was_defaulted}")
    transform_y = True

    # --- Part A: Process Legend-Detected Curves ---
    print("\nStep 5a: Processing curves identified via legend...")
    if legend_dict:
        # Make sure tolerance_dict is defined (from adaptive_color_tolerance call)
        if 'tolerance_dict' not in locals():
            print("Warning: tolerance_dict not defined. Using default tolerance.")
            # Define tolerance_dict here if it wasn't created earlier, e.g., by calling adaptive_color_tolerance
            tolerance_dict = adaptive_color_tolerance(enhanced_img, legend_dict, initial_tolerance=35, max_tolerance=80)

        for label, color in legend_dict.items():
            print(f"\nProcessing LEGEND entry: '{label}' (Target Color: {np.array(color).astype(int)})...")
            base_tolerance = tolerance_dict.get(label, 35)
            current_tolerance = base_tolerance
            max_tolerance_attempt = base_tolerance * 2.0
            max_retries = 3
            extracted_pixels = []

            # Retry logic
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    current_tolerance = min(current_tolerance * 1.3, max_tolerance_attempt)
                    print(f"  Retry {attempt}/{max_retries} with tolerance {current_tolerance:.1f}...")

                temp_pixels = extract_curve_by_color(
                    enhanced_img, color, tolerance=current_tolerance,
                    min_points=MIN_POINTS_LABELED,
                    continuity_max_gap=CONTINUITY_MAX_GAP
                )

                if len(temp_pixels) >= MIN_POINTS_LABELED:
                    extracted_pixels = temp_pixels
                    print(f"  Found sufficient points ({len(extracted_pixels)}) at tolerance {current_tolerance:.1f}.")
                    break
                elif attempt < max_retries:
                    print(f"  Found only {len(temp_pixels)} points, retrying.")
                else:
                    print(f"  Found only {len(temp_pixels)} points after {max_retries+1} attempts for '{label}'.")

        if extracted_pixels: # Ensure we have pixels from the extraction step
            pixel_coords_map[label] = extracted_pixels # Store the coords regardless of mask update success

            # --- Start of Corrected Block to Update Mask ---
            try:
                # 1. Unzip coordinates (py = rows, px = columns)
                # Check if extracted_pixels is not empty before unzipping
                if not extracted_pixels:
                     raise ValueError("extracted_pixels is empty, cannot unpack.")
                py_coords, px_coords = zip(*extracted_pixels)

                # 2. Convert coordinates to NumPy integer arrays
                py_int = np.array(py_coords, dtype=int)
                px_int = np.array(px_coords, dtype=int)

                # 3. Get the shape of the mask for bounds checking
                h, w = processed_pixel_mask.shape

                # 4. Create a boolean mask for coordinates within the bounds
                valid_indices = (py_int >= 0) & (py_int < h) & (px_int >= 0) & (px_int < w)

                # 5. Select only the valid coordinates using the boolean mask
                py_valid = py_int[valid_indices]
                px_valid = px_int[valid_indices]

                # --- *** CRITICAL LINE *** ---
                # 6. Use the VALIDATED INTEGER ARRAYS (py_valid, px_valid) for indexing
                processed_pixel_mask[py_valid, px_valid] = True
                # --- *** END CRITICAL LINE *** ---

                print(f"  Marked {len(py_valid)} pixels as processed for label '{label}'.")

            except ValueError as ve:
                 # Catch specific error if zip(*) fails on empty list
                 print(f"  Error preparing indices for mask update (label '{label}'): {ve}")
            except IndexError as ie:
                 # Catch index errors possibly caused by incorrect shapes or bounds if filtering failed
                 print(f"  IndexError during mask update (label '{label}'): {ie}. Mask shape: {processed_pixel_mask.shape}")
                 # Add more debug info if needed:
                 # print(f"   Max py_int: {np.max(py_int) if len(py_int)>0 else 'N/A'}, Max px_int: {np.max(px_int) if len(px_int)>0 else 'N/A'}")
            except Exception as e:
                # Catch any other unexpected errors during mask update
                print(f"  Warning: Could not update processed_pixel_mask for label '{label}': {type(e).__name__} - {e}")
            # --- End of Corrected Block ---
        else:
             print(f"  Skipping legend entry '{label}' due to insufficient points after extraction.")

    else:
        print("No legend entries found to process.")


    # --- Part B: Detect Unlabeled Curve Candidates ---
    print("\nStep 5b: Detecting unlabeled curve candidates...")
    unlabeled_candidates = {}

    # Option 1: Use Instance Segmentation Map (Preferable)
    # Make sure ins_map is defined from Step 3
    if 'ins_map' in locals() and ins_map is not None and ins_map.max() > 0:
        print("Using instance map for candidate detection...")
        unique_instances = np.unique(ins_map)
        for instance_id in unique_instances:
            if instance_id == 0: continue

            instance_mask = (ins_map == instance_id)
            instance_pixel_count = np.sum(instance_mask)

            if instance_pixel_count < MIN_POINTS_UNLABELED:
                continue

            overlap_count = np.sum(instance_mask & processed_pixel_mask)
            overlap_ratio = overlap_count / instance_pixel_count

            if overlap_ratio < OVERLAP_THRESHOLD:
                print(f"\n  Found potential unlabeled candidate: Instance {instance_id} (Overlap: {overlap_ratio*100:.1f}%, Size: {instance_pixel_count} px)")
                y_coords, x_coords = np.where(instance_mask)
                candidate_pixels = list(zip(x_coords, y_coords))

                # Apply filtering
                filtered_candidate_pixels = remove_outliers_multi_stage(candidate_pixels, img_shape=enhanced_img.shape)
                if len(filtered_candidate_pixels) < MIN_POINTS_UNLABELED:
                    print(f"    Instance {instance_id} rejected after outlier removal ({len(filtered_candidate_pixels)} pts).")
                    continue

                continuous_candidate_pixels = enforce_curve_continuity(filtered_candidate_pixels, max_gap=CONTINUITY_MAX_GAP * 1.5)
                if len(continuous_candidate_pixels) < MIN_POINTS_UNLABELED:
                    print(f"    Instance {instance_id} rejected after continuity check ({len(continuous_candidate_pixels)} pts).")
                    continue

                # Optional: Add scoring here if desired
                # score = data_score_mult(continuous_candidate_pixels)
                # if score < CLUSTER_SCORE_THRESHOLD: continue

                # Add to candidates and update mask
                candidate_id = f"Instance_{instance_id}"
                unlabeled_candidates[candidate_id] = continuous_candidate_pixels
                try:
                    # --- *** CORRECTION FOR INDEX ERROR (Applied here too) *** ---
                    py_coords, px_coords = zip(*continuous_candidate_pixels)
                    py_int = np.array(py_coords, dtype=int)
                    px_int = np.array(px_coords, dtype=int)
                    h, w = processed_pixel_mask.shape
                    valid_indices = (py_int >= 0) & (py_int < h) & (px_int >= 0) & (px_int < w)
                    py_valid = py_int[valid_indices]
                    px_valid = px_int[valid_indices]
                    processed_pixel_mask[py_valid, px_valid] = True # Mark as processed
                    # --- *** END CORRECTION *** ---
                    print(f"    Instance {instance_id} added as candidate '{candidate_id}'. Marked {len(py_valid)} pixels.")
                except Exception as e:
                    print(f"    Warning: Could not update processed_pixel_mask for instance {instance_id}: {e}")

            #else: # Optional: print skipped instances
            #    print(f"  Instance {instance_id} skipped (Overlap: {overlap_ratio*100:.1f}% or Size: {instance_pixel_count} px).")


    # (Option 2: Color Clustering Fallback would go here if implemented)

    # Combine labeled and unlabeled candidates
    print(f"\nTotal labeled candidates kept: {len(pixel_coords_map)}")
    print(f"Total unlabeled candidates kept: {len(unlabeled_candidates)}")
    filtered_coords_map.update(pixel_coords_map)
    filtered_coords_map.update(unlabeled_candidates)

    # --- Part C: Transform and Process All Valid Curves ---
    # (This part remains the same as in the previous corrected version)
    print("\nStep 5c: Transforming and processing all valid curves...")
    # ... (Loop through filtered_coords_map, call transform_coordinates, process_spectral_curve) ...
    for curve_id, pixel_coords in filtered_coords_map.items():
        print(f"\nFinal processing for curve: '{curve_id}' ({len(pixel_coords)} points)")
        if len(pixel_coords) < MIN_POINTS_FINAL:
            print(f"  Skipping '{curve_id}' - too few points ({len(pixel_coords)}) before transformation.")
            continue

        # Ensure scaled_axis_info is populated from Step 4
        if 'scaled_axis_info' not in locals() or not scaled_axis_info['x']['pixels']:
            print(f"ERROR: scaled_axis_info not available for transformation of '{curve_id}'. Skipping.")
            continue

        data_coords = transform_coordinates(pixel_coords, scaled_axis_info, transform_y=transform_y)

        if data_coords and len(data_coords) >= MIN_POINTS_FINAL:
            processed_coords = process_spectral_curve(
                data_coords,
                smoothing_factor=SMOOTHING_FACTOR,
                num_samples=NUM_SAMPLES
            )
            if processed_coords and len(processed_coords) >= MIN_POINTS_FINAL:
                final_data_dict[curve_id] = processed_coords
                # Determine Y label description
                y_label_desc = "pixel coordinates"
                if transform_y:
                    if y_axis_was_defaulted: y_label_desc = "data values (check scaling)"
                    elif len(scaled_axis_info.get('y', {}).get('pixels', [])) >= 2: y_label_desc = "data values (from ticks)"
                print(f"  Successfully processed '{curve_id}' ({len(processed_coords)} points, Y in {y_label_desc})")
            else: print(f"  Warning: Failed during spectral processing for '{curve_id}'.")
        else:
            if not data_coords: print(f"  Warning: Failed to transform coordinates for '{curve_id}'.")
            else: print(f"  Warning: Too few points ({len(data_coords)}) after transformation for '{curve_id}'.")


    # --- Final Conditional Y-Rescaling ---
    # (This block remains the same as in the previous answer, operating on final_data_dict
    #  and using the y_axis_was_defaulted flag)
    print("\n--- Post-Processing: Final Y-Axis Scaling ---")
    # ... (Your conditional rescaling logic) ...
    if y_axis_was_defaulted:
        # ... (Perform global 0-1 rescaling on final_data_dict if appropriate) ...
        print("Y-axis was defaulted. Applying global 0-1 rescaling if needed...")
        # (Include the full rescaling logic from the previous answer here)
    else:
        print("Y-axis ticks were detected. Skipping global 0-1 rescaling.")


    print("\n--- Generalized Curve Extraction Complete ---")
    return final_data_dict


##################################################################################################################
#### ----Do not change-----------------
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
                # smoothed_y_values = savgol_filter(y_values, window_length=80, polyorder=4)
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
        plt.gca().invert_yaxis()
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
    
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\acsmacrolett.6b00250\acsmacrolett.6b00250\images_folder\page_2_img_1_a.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma200800280\adma200800280\images_folder\page_3_img_3_a.jpg"
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma.201002234\adma.201002234\images_folder\page_2_img_4_0.jpg"#
    # image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\nmat2272\nmat2272\images_folder\nmat2272_fig2_b.jpg"
    image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\c9py01720h\c9py01720h\images_folder\page_4_img_1_b.jpg"
    image_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\nmat2272\nmat2272\images_folder\nmat2272_fig1_b.jpg"

    
    final_dict = get_plot_data_dict_hybrid(image_path, axis_align_opt, plot_extract_opt)
    print('final_dict', final_dict)
    
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