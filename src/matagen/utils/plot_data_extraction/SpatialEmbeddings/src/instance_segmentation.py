import torch
import torchvision.transforms as transforms # <-- Import transforms
from PIL import Image
import numpy as np # <-- Import numpy
from multiprocessing import freeze_support
# Assuming these are correct relative imports
from .utils.utils import Cluster
from .models import get_model
from .datasets import get_dataset

class InstanceSeg():
    # --- MODIFIED __init__ ---
    def __init__(self, opt):
        """
        Initializes the segmentation model, loads weights, and sets up
        the necessary transform for the predict() method.
        Does NOT load the full dataset by default.
        """
        self.opt = opt
        self.cluster = Cluster()
        self.model = None
        self.device = None
        self.transform = None # To store the image transformation pipeline
        self.test_loader = None # Initialize dataloader as None

        print("Initializing InstanceSeg...")
        self.load_model()       # Loads model and sets self.device
        self._load_transform()  # Sets up self.transform using self.opt directly

        # load_data() is NOT called here anymore. Call explicitly if run(idx) is needed.
        print("InstanceSeg initialized.")

    # --- MODIFIED load_model ---
    def load_model(self):
        """Loads the PyTorch model, weights, and sets the device."""
        opt = self.opt
        print("Loading segmentation model...")
        # Set device
        # Use .get() for safer access to opt dictionary
        if opt.get("cuda", False) and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"Using CUDA device: {self.device}")
        else:
            self.device = torch.device("cpu")
            print(f"Using CPU device: {self.device}")

        # Set model
        try:
            model_name = opt['model']['name']
            model_kwargs = opt['model'].get('kwargs', {}) # Use get for kwargs
            print(f"Creating model '{model_name}' with kwargs: {model_kwargs}")
            model = get_model(model_name, model_kwargs)

            # Load checkpoint before DataParallel if saved without it
            checkpoint_path = opt['checkpoint_path']
            print(f"Resuming model from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location=self.device) # Load to target device

            # Adjust key if necessary (e.g., if saved state['model_state_dict'] or state directly)
            if 'model_state_dict' in state:
                model_state_dict = state['model_state_dict']
            elif 'state_dict' in state:
                 model_state_dict = state['state_dict']
            else:
                 model_state_dict = state # Assume state is the state_dict

            # Handle potential DataParallel prefix ('module.') if checkpoint was saved with it
            if list(model_state_dict.keys())[0].startswith('module.'):
                 print("Removing 'module.' prefix from checkpoint keys...")
                 model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}

            model.load_state_dict(model_state_dict, strict=True)
            print("Model weights loaded successfully.")

            # Apply DataParallel *after* loading state_dict
            # Note: DataParallel might be less efficient than DistributedDataParallel
            self.model = torch.nn.DataParallel(model).to(self.device)
            self.model.eval() # Set model to evaluation mode
            print("Model loaded, set to evaluation mode, and moved to device.")

        except FileNotFoundError:
             print(f"Error: Checkpoint file not found at {opt.get('checkpoint_path', 'N/A')}")
             raise
        except KeyError as e:
             print(f"Error: Missing key in configuration options: {e}")
             raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise # Re-raise error if loading fails

    # --- NEW _load_transform Method ---
    def _load_transform(self):
        """
        Creates the image transformation pipeline based on options directly in self.opt.
        Reads img_height, img_width, norm_mean, norm_std from top level of opt.
        """
        opt = self.opt
        print("Loading image transform...")
        # Use .get() with default values for robustness
        img_height = opt.get('img_height', 256) # Default 256
        img_width = opt.get('img_width', 512)  # Default 512
        mean = opt.get('norm_mean', [0.485, 0.456, 0.406]) # Default ImageNet mean
        std = opt.get('norm_std', [0.229, 0.224, 0.225])   # Default ImageNet std

        try:
             self.transform = transforms.Compose([
                 # Consider adding antialias=True if using recent torchvision
                 # transforms.Resize((img_height, img_width), interpolation=Image.Resampling.BILINEAR, antialias=True),
                 transforms.Resize((img_height, img_width), interpolation=Image.Resampling.BILINEAR),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=std)
             ])
             print(f"Image transform loaded: Resize=({img_height},{img_width}), ToTensor, Normalize(mean={mean}, std={std}).")
        except Exception as e:
             print(f"Error creating transform pipeline: {e}")
             self.transform = None

    # --- predict Method (Assumes you added this previously) ---
    def predict(self, image_pil):
        """
        Runs segmentation on a loaded PIL image.
        Uses self.model, self.device, self.transform initialized in __init__.
        """
        if self.model is None:
            print("Error: predict() called but model not loaded.")
            return None, None
        if self.transform is None:
            print("Error: predict() called but image transform not loaded/defined.")
            return None, None
        if self.device is None:
             print("Error: predict() called but device not set.")
             return None, None

        if not isinstance(image_pil, Image.Image):
             raise TypeError("Input must be a PIL.Image.Image object")

        # Preprocess
        try:
            img_rgb = image_pil.convert("RGB")
            input_tensor = self.transform(img_rgb)
            # Add batch dimension and move to device
            input_batch = input_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
             print(f"Error during image preprocessing: {e}")
             return None, None

        # Inference
        try:
            with torch.no_grad():
                output = self.model(input_batch)
        except Exception as e:
             print(f"Error during model inference: {e}")
             return None, None

        # Post-process
        try:
             # Clustering
             instance_map, predictions = self.cluster.cluster(output[0], threshold=0.9) # Using output[0] for batch size 1
             ins_img_np = instance_map.data.cpu().numpy()

             # Binary map (assuming channel 3 based on original code)
             bin_img_tensor = torch.sigmoid(output[0, 3])
             bin_img_np = bin_img_tensor.data.cpu().numpy()
             # Thresholding
             bin_img_np[bin_img_np > 0.3] = 1.0
             bin_img_np[bin_img_np <= 0.3] = 0.0
             bin_img_np = bin_img_np.astype(np.uint8)

             return bin_img_np, ins_img_np

        except Exception as e:
            print(f"Error during post-processing (cluster/binarization): {e}")
            return None, None


    # --- load_data Method (Now separate, only for run(idx)) ---
    def load_data(self):
        """Loads the dataset and dataloader for index-based access via run()."""
        if self.test_loader is not None:
            print("Dataset already loaded for run(idx).")
            return

        opt = self.opt
        print("Loading dataset for index-based access (run(idx))...")
        # Use .get() for dataset config keys for robustness
        dataset_name = opt.get('dataset', {}).get('name', 'cityscapes') # Default or from opt
        # IMPORTANT: Pass only kwargs expected by the dataset class!
        dataset_kwargs = opt.get('dataset', {}).get('kwargs', {})
        # Filter out keys we moved to top level (img_height, etc.) if they are still present by mistake
        expected_dataset_keys = ['root_dir', 'type'] # Add keys CityscapesDataset expects
        filtered_dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k in expected_dataset_keys}
        print(f"Calling get_dataset('{dataset_name}') with kwargs: {filtered_dataset_kwargs}")

        try:
             # load dataset
             dataset = get_dataset(dataset_name, filtered_dataset_kwargs)

             # If dataset needs transform, it should handle it internally or accept 'transform' kwarg
             # We are not explicitly passing self.transform here.

             test_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=1, # Must be 1 for run(idx)
                                                       shuffle=False,
                                                       num_workers=opt.get('num_workers', 0),
                                                       pin_memory=True if self.device.type == 'cuda' else False)
             self.test_loader = list(test_loader) # Load all samples into memory
             print(f"Dataset loaded with {len(self.test_loader)} samples for run(idx).")
        except Exception as e:
             print(f"Error loading dataset for run(idx): {e}")
             self.test_loader = []


    # --- __len__ Method (Reflects loaded data for run(idx)) ---
    def __len__(self):
        # Returns length of pre-loaded data if available for run(idx)
        return len(self.test_loader) if self.test_loader is not None else 0


    # --- run Method (Original index-based method, now uses predict) ---
    def run(self, idx):
        """
        Processes an image by index using the pre-loaded dataset.
        Requires load_data() to be called first.
        Now internally uses the predict() method for consistency.
        """
        # Ensure data is loaded
        if self.test_loader is None:
            print("Dataset not loaded for run(idx). Call load_data() first.")
            # self.load_data() # Optionally load here, but better to call explicitly before run
            return None, None, None, None # Return None if data not loaded

        if not (0 <= idx < len(self.test_loader)):
             print(f"Error: Index {idx} out of bounds for dataset size {len(self.test_loader)}")
             return None, None, None, None

        # Get original image path/object from the loader
        try:
            sample = self.test_loader[idx]
            im_name = sample.get('im_name')
            # If im_name is a list/tuple, take the first element
            if isinstance(im_name, (list, tuple)):
                im_name = im_name[0] if len(im_name) > 0 else None
        except IndexError:
             print(f"Error accessing sample at index {idx}.")
             return None, None, None, None

        # Load original PIL image (needed for return value consistency)
        img_pil_orig = None
        if im_name:
             try:
                  img_pil_orig = Image.open(im_name).convert("RGB")
             except Exception as e:
                  print(f"Warning: Error opening original image {im_name} for index {idx}: {e}")
                  # Proceed without original PIL image if necessary, predict might still work if sample['image'] exists
        else:
             print(f"Warning: Could not retrieve image name for index {idx}.")
             # Maybe try to get PIL from sample['image'] - complex due to transforms

        # --- Re-use the predict method ---
        # Need a PIL image. Use the original one if loaded, otherwise cannot run predict.
        if img_pil_orig is not None:
            bin_img_np, ins_img_np = self.predict(img_pil_orig)
        else:
             print("Error: Cannot run prediction for run(idx) as original PIL image failed to load or path was missing.")
             bin_img_np, ins_img_np = None, None

        # Return the results along with the original PIL image and name
        return bin_img_np, ins_img_np, img_pil_orig, im_name


# Keep the main execution block if useful for standalone testing
# def main():
#     # Example Usage (requires a sample opt dictionary configured correctly)
#     # opt = { ... }
#     # obj = InstanceSeg(opt)
#     # To use run(idx):
#     # obj.load_data()
#     # if len(obj) > 0: result = obj.run(0)
#     # To use predict:
#     # img = Image.open("path/to/image.png")
#     # bin_map, ins_map = obj.predict(img)
#     pass

# if __name__ == '__main__':
#     freeze_support()
#     main()