import cv2
import numpy as np
from PIL import Image # <-- Import PIL
from scipy.signal import find_peaks, peak_widths

# Assuming correct relative imports
from .lanenet.instance_segmentation import InstanceSeg as LN_InstanceSeg
from .SpatialEmbeddings.src.instance_segmentation import InstanceSeg as SE_InstanceSeg
from .utils import ComputeGrad
from .optical_flow import OpticalFlow


class PlotDigitizer():
    def __init__(self):
        self.result_dict = {
            "visual": {},
            "data": {},
        }
        self.instance_seg = None
        self.optical_flow = None
        self.img_name = None

    # __len__ might represent state rather than dataset size now
    def __len__(self):
        return 1 if self.instance_seg else 0 # Or return self.instance_seg.__len__() if it's meaningful

    def load_seg(self, seg_type, opt):
        print(f"\nPlotDigitizer: Attempting to load seg_type='{seg_type}'...") # Added newline for clarity
        self.instance_seg = None # Explicitly set to None at start

        if seg_type == "lanenet":
            # self.instance_seg = LN_InstanceSeg(opt) # Add similar try/except/prints if using LaneNet
            pass 
            print("PlotDigitizer: LaneNet loading not fully implemented in this example.")
  
        elif seg_type == "spatialembedding":
            self.instance_seg = SE_InstanceSeg(opt)            

        else:
            # Keep this outside try/except as it's a config error
            raise NotImplementedError(f"Segmentation type '{seg_type}' not supported.")

        # --- Final Check (should be redundant if above checks work, but keep as safeguard) ---
        print("PlotDigitizer: Performing final check on self.instance_seg...")
        if self.instance_seg is None:
            # This path should ideally not be reached if above try/except works
            print("PlotDigitizer: Final check FAILED (self.instance_seg is None).")
            raise RuntimeError(f"Failed to assign segmentation model instance for type '{seg_type}'.")
        elif not hasattr(self.instance_seg, 'model') or self.instance_seg.model is None:
            # This path means assignment happened but model is bad
            print("PlotDigitizer: Final check FAILED (model attribute missing or None).")
            raise RuntimeError(f"Assigned segmentation model instance for type '{seg_type}' has missing/None model attribute.")
        else:
            print(f"PlotDigitizer: Final check PASSED. self.instance_seg is type {type(self.instance_seg)}. load_seg complete.")
            

    def predict_and_process(self, image_path, denoise=False): 

        if self.instance_seg is None: 
            raise RuntimeError("Instance segmentation model (self.instance_seg) is None. Load first using load_seg().")

        # Check if the predict method exists (good practice)
        if not hasattr(self.instance_seg, 'predict'):
            raise NotImplementedError(f"{type(self.instance_seg).__name__} requires a 'predict(image_pil)' method.")

        self.img_name = image_path
        try:
            img_pil = Image.open(self.img_name).convert('RGB')
            print(f"Loaded image from: {self.img_name}")
        except FileNotFoundError:
            print(f"Error: Image file not found at {self.img_name}")
            self.result_dict = {"visual": {}, "data": {}} # Clear results
            return False # Indicate failure
        except Exception as e:
            print(f"Error loading image {self.img_name}: {e}")
            self.result_dict = {"visual": {}, "data": {}} # Clear results
            return False # Indicate failure

        # 2. Run segmentation using the NEW predict method of instance_seg
        try:
            # Calls the predict(img_pil) method you added to SE_InstanceSeg/LN_InstanceSeg
            # Assumes it returns (binary_map_np, instance_map_np) or (None, None) on failure
            bin_img_np, ins_img_np = self.instance_seg.predict(img_pil)

            if bin_img_np is None or ins_img_np is None:
                 print("Segmentation prediction failed. Check logs from segmentation class.")
                 self.result_dict = {"visual": {}, "data": {}}
                 return False # Indicate failure

        except Exception as e:
            print(f"Error during instance segmentation prediction call: {e}")
            self.result_dict = {"visual": {}, "data": {}}
            return False # Indicate failure

        w, h = img_pil.size
        if w > h:
            nw = 512
            nh = int(nw / w * h) if w > 0 else 512
        else:
            nh = 512
            nw = int(nh / h * w) if h > 0 else 512

        img_resized_pil = img_pil.resize((nw, nh), Image.Resampling.LANCZOS)

        try:
            if not isinstance(bin_img_np, np.ndarray) or not isinstance(ins_img_np, np.ndarray):
                 raise TypeError("Segmentation maps received from predict() must be NumPy arrays.")
            seg_map = cv2.resize(bin_img_np.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
            ins_map = cv2.resize(ins_img_np.astype(np.int32), (nw, nh), interpolation=cv2.INTER_NEAREST) # Use appropriate type
        except Exception as e:
             print(f"Error resizing segmentation maps: {e}")
             self.result_dict = {"visual": {}, "data": {}}
             return False

        img_np = np.array(img_resized_pil)

        if denoise:
            try:
                img_denoised_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
                img_denoised_np = cv2.bilateralFilter(img_denoised_np, 15, 75, 75)
                img_proc_np = img_denoised_np # Use denoised version
            except Exception as e:
                 print(f"Warning: Denoising failed: {e}. Using original resized image.")
                 img_proc_np = img_np 
        else:
            img_proc_np = img_np 

        # Create final visual outputs (RGB float, Grayscale inverted float)
        img_rgb = img_proc_np / 255.0
        # Use the processed (potentially denoised) NumPy array for grayscale
        img_gray = 1.0 - cv2.cvtColor(img_proc_np, cv2.COLOR_RGB2GRAY) / 255.0

        # Store results
        targets = ["img_rgb", "img_gray", "seg_map", "ins_map"]
        for target in targets:
            self.result_dict["visual"][target] = eval(target)

        # Estimate linewidth and setup optical flow
        self.linewidth_estimation() # Uses self.result_dict
        # Ensure OpticalFlow handles potential issues if inputs are bad
        try:
            self.optical_flow = OpticalFlow(self.result_dict["visual"]["img_rgb"],
                                            self.result_dict["visual"]["img_gray"],
                                            self.result_dict["visual"]["seg_map"])
        except Exception as e:
             print(f"Error initializing OpticalFlow: {e}")
             self.optical_flow = None # Set to None on failure

        print(f"Finished processing image: {self.img_name}")
        return True # Indicate success


    # --- Linewidth Estimation (ensure checks are present) ---
    def linewidth_estimation(self):
        if "seg_map" not in self.result_dict["visual"]:
            print("Error: Segmentation map not available for linewidth estimation.")
            self.result_dict["data"]["start_ids"] = []
            self.result_dict["data"]["linewidth"] = 1
            self.result_dict["data"]["num_plots"] = 0
            return
        seg_map = self.result_dict["visual"]["seg_map"]
        # Add check for empty seg_map
        if seg_map.size == 0 or seg_map.shape[1] == 0:
            print("Warning: Segmentation map is empty. Cannot estimate linewidth.")
            self.result_dict["data"]["start_ids"] = []
            self.result_dict["data"]["linewidth"] = 1
            self.result_dict["data"]["num_plots"] = 0
            return

        p_width = []
        num_p = []
        for t in range(seg_map.shape[1]):
            # Use height threshold to avoid noise peaks if seg_map isn't purely binary
            peaks, properties = find_peaks(seg_map[:, t], height=0.1) # Adjust height if needed
            num_p.append(len(peaks))
            if len(peaks) > 0:
                try:
                     # Use properties['peak_heights'] if available for better width calc?
                     # rel_height=0.5 is common for FWHM
                     widths, width_heights, left_ips, right_ips = peak_widths(seg_map[:, t], peaks, rel_height=0.5)
                     p_width.extend(list(widths))
                except ValueError as e:
                     print(f"Warning: peak_widths failed for column {t}: {e}")


        if not p_width:
             print("Warning: No valid peak widths found for linewidth estimation.")
             w = 1
             estimated_num_plots = 0
             ids = []
        else:
             w = np.median(p_width)
             value, count = np.unique(num_p, return_counts=True)
             if len(value)==0:
                  estimated_num_plots = 0
                  ids = []
             else:
                  # Simplified: most frequent number of peaks is the estimate
                  estimated_num_plots = value[np.argmax(count)]
                  # Select columns where this number of peaks occurred
                  ids = np.where(np.array(num_p) == estimated_num_plots)[0]

        self.result_dict["data"]["start_ids"] = ids
        self.result_dict["data"]["linewidth"] = w
        self.result_dict["data"]["num_plots"] = estimated_num_plots
        print(f"Estimated linewidth: {w:.2f}")
        print(f"Estimated num of plots: {estimated_num_plots}")
        print(f"Found {len(ids)} candidate start columns.")


    # --- Find Initial Position (ensure checks are present) ---
    def find_init_posi(self, threshold = 2.):
        if "start_ids" not in self.result_dict["data"] or \
           "seg_map" not in self.result_dict["visual"] or \
           "img_gray" not in self.result_dict["visual"]:
            print("Error/Warning: Required data not available for find_init_posi.")
            self.result_dict["data"]["start_posi"] = []
            return

        ids = self.result_dict["data"]["start_ids"]
        if len(ids) == 0:
             print("No candidate start columns (start_ids) found by linewidth_estimation.")
             self.result_dict["data"]["start_posi"] = []
             return

        seg_map = self.result_dict["visual"]["seg_map"]
        img_gray = self.result_dict["visual"]["img_gray"]

        try:
            grads = ComputeGrad(img_gray)
        except Exception as e:
             print(f"Error computing gradient: {e}. Cannot find initial positions.")
             self.result_dict["data"]["start_posi"] = []
             return

        start_posi = []
        max_grads_values = {}
        for idx in ids:
            idx = int(idx) # Ensure index is integer
            if not (0 <= idx < seg_map.shape[1]): continue # Bounds check
            p, _ = find_peaks(seg_map[:, idx], height=0.1) # Use height threshold
            if len(p) == 0: continue

            # Ensure peak indices are valid for the gradient map
            valid_p = p[(p >= 0) & (p < grads.shape[0])]
            if len(valid_p) == 0: continue

            # Ensure gradient map has the expected column index
            if not (0 <= idx < grads.shape[1]): continue

            col_grads = grads[valid_p, idx]
            if len(col_grads) == 0: continue

            max_grad = np.max(np.abs(col_grads))
            max_grads_values[idx] = max_grad
            if max_grad < threshold:
                start_posi.append(idx)

        # Refinement logic
        if 0 < len(start_posi) < 20 and len(max_grads_values) >= len(start_posi): # Ensure we have gradients calculated
            print(f"Found fewer than 20 positions ({len(start_posi)}). Selecting up to 20 lowest gradient positions from candidates.")
            # Sort the original 'ids' that had gradients calculated, based on those gradients
            valid_ids = list(max_grads_values.keys())
            sorted_ids = sorted(valid_ids, key=lambda i: max_grads_values[i])
            start_posi = sorted_ids[:min(20, len(sorted_ids))]
        elif len(start_posi) == 0 and len(max_grads_values) > 0: # If threshold was too strict, take the best ones
             print(f"Threshold {threshold} might be too strict (0 positions found). Selecting up to 20 lowest gradient positions.")
             valid_ids = list(max_grads_values.keys())
             sorted_ids = sorted(valid_ids, key=lambda i: max_grads_values[i])
             start_posi = sorted_ids[:min(20, len(sorted_ids))]


        self.result_dict["data"]["start_posi"] = start_posi
        # Convert final positions to list of ints if they aren't already
        self.result_dict["data"]["start_posi"] = [int(p) for p in self.result_dict["data"]["start_posi"]]
        print(f"Num of final start positions selected: {len(self.result_dict['data']['start_posi'])}")