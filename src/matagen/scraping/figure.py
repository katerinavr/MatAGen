# import glob
import json
import logging
import os
import pathlib
import time
import warnings

# import cv2
import numpy as np
import torch
# import torch.nn.functional as F
# import torchvision.models.detection
# import torchvision.transforms as T
import yaml
from PIL import Image
# from scipy.special import softmax
# from skimage import io
# from torch.autograd import Variable
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from .figures.models.crnn import CRNN
# from .figures.models.network import resnet152
# from .figures.models.yolov3 import YOLOv3, YOLOv3img
# from .figures.scale import ctc
# from .figures.scale.process import non_max_suppression_malisiewicz
# from .figures.separator import process
from .tool import ExsclaimTool
# from .utilities import boxes
# from .utilities.logging import Printer
# from .utilities.models import load_model_from_checkpoint

from ultralytics import YOLO

def convert_to_rgb(image):
    return image.convert("RGB")
import sys

def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()

class FigureSeparator(ExsclaimTool):
    """
    FigureSeparator object.
    Separate subfigure images from full figure image
    using CNN trained on crowdsourced labeled figures
    Parameters:
    None
    """

    def __init__(self, search_query):
        self.logger = logging.getLogger(__name__)
        self.initialize_query(search_query)
        self._load_model()
        self.exsclaim_json = {}

    def _load_model(self):
        """Load relevant models for the object detection tasks"""
        """Load YOLO model directly from checkpoint"""
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))  # Go up one level
            model_path = os.path.join(project_root,
                'checkpoints',
                'yolov11_finetuned_augmentation_best.pt')
            # model_path = os.path.join(
            #     os.path.dirname(__file__),
            #     "checkpoints/yolov11_finetuned_augmentation_best.pt"
            # )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model directly from .pt file
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(self.device)
            
            # Common YOLO settings if needed
            self.confidence_threshold = 0.25  # Default confidence threshold
            self.image_size = 640  # Default YOLO image size
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

        # Set configuration variables
        # model_path = os.path.dirname(__file__) + "/figures/"
        # configuration_file = model_path + "config/yolov3_default_subfig.cfg"
        # with open(configuration_file, "r") as f:
        #     configuration = yaml.load(f, Loader=yaml.FullLoader)

        # self.image_size = configuration["TEST"]["IMGSIZE"]
        # self.nms_threshold = configuration["TEST"]["NMSTHRE"]
        # self.confidence_threshold = 0.0001
        # self.gpu_id = 1
        # This suppresses warning if user has no CUDA device initialized,
        # which is unneccessary as we are explicitly checking. This may not
        # be necessary in the future, described in:
        # https://github.com/pytorch/pytorch/issues/47038
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        if self.cuda:
            self.logger.info("using cuda")
            # torch.cuda.set_device(device=args.gpu_id)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # # Load object detection model
        # object_detection_model = YOLOv3(configuration["MODEL"])
        # self.object_detection_model = load_model_from_checkpoint(
        #     object_detection_model, "object_detection_model.pt", self.cuda, self.device
        # )
        # # Load text recognition model
        # text_recognition_model = resnet152()
        # self.text_recognition_model = load_model_from_checkpoint(
        #     text_recognition_model, "text_recognition_model.pt", self.cuda, self.device
        # )
        # # Load classification model
        # master_config_file = model_path + "config/yolov3_default_master.cfg"
        # with open(master_config_file, "r") as f:
        #     master_config = yaml.load(f, Loader=yaml.FullLoader)
        # classifier_model = YOLOv3img(master_config["MODEL"])
        # self.classifier_model = load_model_from_checkpoint(
        #     classifier_model, "classifier_model.pt", self.cuda, self.device
        # )
        # # Load scale bar detection model
        # # load an object detection model pre-trained on COCO
        # scale_bar_detection_model = (
        #     torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # )
        # input_features = (
        #     scale_bar_detection_model.roi_heads.box_predictor.cls_score.in_features
        # )
        # number_classes = 3  # background, scale bar, scale bar label
        # scale_bar_detection_model.roi_heads.box_predictor = FastRCNNPredictor(
        #     input_features, number_classes
        # )
        # self.scale_bar_detection_model = load_model_from_checkpoint(
        #     scale_bar_detection_model,
        #     "scale_bar_detection_model.pt",
        #     self.cuda,
        #     self.device,
        # )
        # # Load scale label recognition model
        # parent_dir = pathlib.Path(__file__).resolve(strict=True).parent
        # config_path = parent_dir / "figures" / "config" / "scale_label_reader.json"
        # with open(config_path, "r") as f:
        #     configuration_file = json.load(f)
        # configuration = configuration_file["theta"]
        # scale_label_recognition_model = CRNN(configuration=configuration)
        # self.scale_label_recognition_model = load_model_from_checkpoint(
        #     scale_label_recognition_model,
        #     "scale_label_recognition_model.pt",
        #     self.cuda,
        #     self.device,
        # )

    def _update_exsclaim(self, exsclaim_dict, figure_name, figure_dict):
        figure_name = figure_name.split("/")[-1]
        for master_image in figure_dict["figure_separator_results"][0]["master_images"]:
            exsclaim_dict[figure_name]["master_images"].append(master_image)

        for unassigned in figure_dict["figure_separator_results"][0]["unassigned"]:
            exsclaim_dict[figure_name]["unassigned"]["master_images"].append(unassigned)
        return exsclaim_dict

    def _appendJSON(self, exsclaim_json, figures_separated):
        """Commit updates to EXSCLAIM JSON and updates list of separated figures

        Args:
            results_directory (string): Path to results directory
            exsclaim_json (dict): Updated EXSCLAIM JSON
            figures_separated (set): Figures which have already been separated
        """
        with open(self.results_directory / "exsclaim.json", "w", encoding="utf-8") as f:
            json.dump(exsclaim_json, f, indent=3)
        with open(self.results_directory / "_figures", "a+", encoding="utf-8") as f:
            for figure in figures_separated:
                f.write("%s\n" % figure)

    def run(self, search_query, exsclaim_dict):
        """Run the models relevant to manipulating article figures"""
        self.display_info("Running Figure Separator\n")
        os.makedirs(self.results_directory, exist_ok=True)
        self.exsclaim_json = exsclaim_dict
        t0 = time.time()
        # List figures that have already been separated
        figures_file = self.results_directory / "_figures"
        if os.path.isfile(figures_file):
            with open(figures_file, "r", encoding="utf-8") as f:
                contents = f.readlines()
            figures_separated = {f.strip() for f in contents}
        else:
            figures_separated = set()

        with open(figures_file, "w", encoding="utf-8") as f:
            for figure in figures_separated:
                f.write("%s\n" % str(pathlib.Path(figure).name))
        new_figures_separated = set()

        counter = 1
        figures_path = self.results_directory / "figures"
        figures = [
            figures_path / self.exsclaim_json[figure]["figure_name"]
            for figure in self.exsclaim_json
            if self.exsclaim_json[figure]["figure_name"] not in figures_separated
        ]
        for figure_path in figures:
            self.display_info(
                ">>> ({0} of {1}) ".format(counter, +len(figures))
                + "Extracting images from: "
                + str(figure_path)
            )
            try:
                self.extract_image_objects(figure_path)
                new_figures_separated.add(figure_path.name)
            except Exception:
                if self.print:
                    Printer(
                        (
                            "<!> ERROR: An exception occurred in"
                            " FigureSeparator on figure: {}".format(figure_path)
                        )
                    )
                self.logger.exception(
                    (
                        "<!> ERROR: An exception occurred in"
                        " FigureSeparator on figure: {}".format(figure_path)
                    )
                )

            # Save to file every N iterations (to accomodate restart scenarios)
            if counter % 100 == 0:
                self._appendJSON(self.exsclaim_json, new_figures_separated)
                new_figures_separated = set()
            counter += 1

        t1 = time.time()
        self.display_info(
            ">>> Time Elapsed: {0:.2f} sec ({1} figures)\n".format(
                t1 - t0, int(counter - 1)
            )
        )
        self._appendJSON(self.exsclaim_json, new_figures_separated)
        return self.exsclaim_json

    def extract_image_objects(self, figure_path=str) -> dict:
        """Separate and classify subfigures in an article figure"""

        full_figure_path = figure_path
        img_raw = Image.open(full_figure_path).convert("RGB")
        width, height = img_raw.size
        binary_img = np.zeros((height, width, 1))

        # Get figure name without extension for directory naming
        figure_base_name = pathlib.Path(figure_path).stem

        # Run YOLO detection 
        results = self.yolo_model.predict(
            source=full_figure_path,
            imgsz=608,
            conf=0.1,
            iou=0.45,
            agnostic_nms=False,
        )

        # Initialize figure_json
        figure_name = figure_path.name
        figure_json = self.exsclaim_json.get(figure_name, {})
        figure_json["figure_name"] = figure_name
        figure_json["master_images"] = []

        # Collecting the top box per class
        detections_per_class = {}        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = box.conf[0]
                # Storing only the highest-confidence box per class
                if cls_id not in detections_per_class or conf > detections_per_class[cls_id].conf[0]:
                    detections_per_class[cls_id] = box

        # if no detections, set default box to full figure
        if len(detections_per_class) == 0:
            x1, y1, x2, y2 = 0, 0, width, height
            
            master_image_info = {
            "classification": "subfigure",
            "confidence": 1.0,  # or any sentinel value
            "height": int(y2 - y1),
            "width": int(x2 - x1),
            "geometry": [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x1, "y": y2},
                {"x": x2, "y": y2},
            ],
            "subfigure_label": {
                "text": "0",  # we label with "0" for the full figure, when no subfigures are detected
                "geometry": [
                    {"x": x1, "y": y1},
                    {"x": x2, "y": y1},
                    {"x": x1, "y": y2},
                    {"x": x2, "y": y2},
                    ],
                },
            }

            # Create an output directory
            base_dir = self.results_directory / "images" / figure_base_name
            subfig_dir = base_dir / "0"
            os.makedirs(subfig_dir, exist_ok=True)

            # Save the entire figure as the “0” subfigure
            output_filename = f"{figure_base_name}_0.png"
            img_raw.save(subfig_dir / output_filename)

            figure_json["master_images"].append(master_image_info)
        
        else:

            # Process each final detection
            for cls_id, box in detections_per_class.items():
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Ensure coordinates are within bounds and boxes aren't too small
                x1 = int(min(max(x1, 0), width - 1))
                y1 = int(min(max(y1, 0), height - 1))
                x2 = int(min(max(x2, 0), width))
                y2 = int(min(max(y2, 0), height))
                
                if (x2 - x1 <= 5 or y2 - y1 <= 5):
                    continue

                # Get the label
                label = self.yolo_model.names[cls_id]  # This will be 'a', 'b', 'c', etc.
                
                # Add to binary mask for visualization if small enough
                if (x2 - x1) < 64 and (y2 - y1) < 64:
                    binary_img[y1:y2, x1:x2] = 255

                # Create master_image_info
                master_image_info = {
                    "classification": "subfigure",
                    "confidence": float(conf),
                    "height": int(y2 - y1),
                    "width": int(x2 - x1),
                    "geometry": [
                        {"x": int(x1), "y": int(y1)},
                        {"x": int(x2), "y": int(y1)},
                        {"x": int(x1), "y": int(y2)},
                        {"x": int(x2), "y": int(y2)}
                    ],
                    "subfigure_label": {
                        "text": label,
                        "geometry": [
                            {"x": int(x1), "y": int(y1)},
                            {"x": int(x2), "y": int(y1)},
                            {"x": int(x1), "y": int(y2)},
                            {"x": int(x2), "y": int(y2)}
                        ]
                    }
                }

                # Create output directory structure using figure_base_name (without extension)
                base_dir = self.results_directory / "images" / figure_base_name
                subfig_dir = base_dir / label
                os.makedirs(subfig_dir, exist_ok=True)
                                
                # Crop and save using base name for output filename
                cropped_img = img_raw.crop((x1, y1, x2, y2))
                output_filename = f"{figure_base_name}_{label}.png"
                cropped_img.save(subfig_dir / output_filename)

                figure_json["master_images"].append(master_image_info)

        # Update the JSON
        self.exsclaim_json[figure_name] = figure_json

        # # Detect scale bar lines and labels if needed
        # if hasattr(self, 'determine_scale'):
        #     figure_json = self.determine_scale(full_figure_path, figure_json)

        return figure_json
    


