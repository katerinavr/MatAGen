from mmdet.apis import init_detector, inference_detector
import torch

"""
CNN-based detector trained on plot data
"""
class PlotDetector():
    def __init__(self):
        print("build plot detector ...")
        
    def load_model(self, config_file, checkpoint_file, device='cpu'):
        self.detector = init_detector(config_file, checkpoint_file, device='cpu')
        
    def detection(self, img_path):
        result = inference_detector(self.detector, img_path)
        return result