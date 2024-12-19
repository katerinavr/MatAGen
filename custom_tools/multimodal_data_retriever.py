import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from pydantic import BaseModel
from typing import List


# Define Pydantic models for structured return types
class MultimodalRecord(BaseModel):
    full_caption: str
    caption: str
    caption_summary: List[str]
    image: str
    article_name: str


class MultimodalDatasetResponse(BaseModel):
    records: List[MultimodalRecord]


class MultimodalDatasetProcessor:
    def __init__(self, exsclaim_output_folder: str):
        """
        Initializes the MultimodalDatasetProcessor.

        Args:
            exsclaim_output_folder (str): Path to the folder containing exsclaim.json and figures.
        """
        self.exsclaim_output_folder = Path(exsclaim_output_folder)
        self.img_folder_dir = Path("multimodal_data_folder") / "images_folder"
        os.makedirs(self.img_folder_dir, exist_ok=True)
        self.data = self.load_exsclaim_json()

    def load_exsclaim_json(self) -> dict:
        """
        Loads the exsclaim.json file.

        Returns:
            dict: The content of the exsclaim.json file.
        """
        exsclaim_json = self.exsclaim_output_folder / "exsclaim.json"
        if not exsclaim_json.exists():
            raise FileNotFoundError(f"{exsclaim_json} does not exist.")
        with open(exsclaim_json, 'r') as f:
            return json.load(f)

    def process_image(self, figure_name: str) -> np.ndarray:
        """
        Loads an image from the specified folder and returns it as a numpy array.

        Args:
            figure_name (str): Name of the image file.

        Returns:
            np.ndarray: The loaded image as an array.
        """
        image_path = self.exsclaim_output_folder / "figures" / figure_name
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist.")
        img = Image.open(image_path)
        return np.array(img)

    def crop_image(self, geometry: List[dict], image: np.ndarray) -> np.ndarray:
        """
        Crops an image to the specified geometry.

        Args:
            geometry (list): Coordinates defining the crop area.
            image (np.ndarray): The original image.

        Returns:
            np.ndarray: The cropped image.
        """
        x1, y1 = geometry[0]["x"], geometry[0]["y"]
        x2, y2 = geometry[3]["x"], geometry[3]["y"]
        return image[y1:y2, x1:x2]

    def create_multimodal_dataset(self) -> MultimodalDatasetResponse:
        """
        Creates a dataset with cropped subfigures and their corresponding captions and keywords.

        Returns:
            MultimodalDatasetResponse: A structured response containing the dataset.
        """
        records = []

        for key, value in self.data.items():
            try:
                full_img = self.process_image(value["figure_name"])
            except Exception as e:
                print(f"Error loading image for {key}: {e}")
                continue

            for master_image in value["master_images"]:
                try:
                    geometry = master_image["geometry"]
                    label = master_image["subfigure_label"]["text"]

                    caption = (
                        value["full_caption"]
                        if label == '0'
                        else master_image.get("caption", "")
                    )
                    # Ensure caption is always a string
                    if not isinstance(caption, str):
                        caption = str(caption)

                    caption_summary = (
                        []
                        if label == '0'
                        else master_image.get("keywords", [])
                    )

                    img_name = f"{value['figure_name'].rsplit('.', 1)[0]}_{label}"
                    subfigure = self.crop_image(geometry, full_img)

                    subfigure_img = Image.fromarray(subfigure)
                    if subfigure_img.mode == 'RGBA':
                        subfigure_img = subfigure_img.convert('RGB')
                    file_path = self.img_folder_dir / f"{img_name}.jpg"
                    subfigure_img.save(file_path)

                    records.append(
                        MultimodalRecord(
                            full_caption=value["full_caption"],
                            caption=caption,
                            caption_summary=caption_summary,
                            image=str(file_path),
                            article_name=value["article_name"]
                        )
                    )
                    
                except Exception as e:
                    print(f"Error processing subfigure for {key}: {e}")
                    continue

        return MultimodalDatasetResponse(records=records)


def multimodal_data_retriever_tool(exsclaim_output_folder: str) -> MultimodalDatasetResponse:
    """
    Tool for processing the JSON file and extracting image-text pairs.

    Args:
        exsclaim_output_folder (str): Path to the folder containing the exsclaim.json and figures.

    Returns:
        MultimodalDatasetResponse: A structured response containing the dataset records.
    """
    data_retriever = MultimodalDatasetProcessor(exsclaim_output_folder)
    multimodal_df = data_retriever.create_multimodal_dataset()
    return multimodal_df


if __name__ == "__main__":
    exsclaim_output_folder ="html-scraping"
    processor = MultimodalDatasetProcessor(exsclaim_output_folder)
    multimodal_df = processor.create_multimodal_dataset()
    multimodal_df.to_csv(Path("multimodal_data_folder/retrieved_images.csv"))
    print(multimodal_df)
