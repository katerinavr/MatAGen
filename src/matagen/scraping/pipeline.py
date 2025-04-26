# Code adapted and modified from MaterialEyes
import json
import logging
import os
import pathlib
from pathlib import Path
import numpy as np
from PIL import Image #, ImageDraw, ImageFont
from typing import List
from .figure import FigureSeparator
from .tool import CaptionDistributor, JournalScraper, HTMLScraper, PDFScraper
import sys

def blockPrint():
    sys.stdout = open(os.devnull, "w")

def enablePrint():
    sys.stdout = sys.__stdout__

class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()

class Pipeline:
    """Defines the exsclaim pipeline"""

    def __init__(self, query_path):
        """initialize a Pipeline to run on query path and save to exsclaim path

        Args:
            query_path (dict or path to json): An EXSCLAIM user query JSON
        """
        self.logger = logging.getLogger(__name__)
        self.current_path = pathlib.Path(__file__).resolve().parent
        if "test" == query_path:
            query_path = self.current_path / "tests" / "data" / "nature_test.json"
        if isinstance(query_path, dict):
            self.query_dict = query_path
            self.query_path = ""
        else:
            assert os.path.isfile(
                query_path
            ), "query path must be a dict, query path, or 'test', was {}".format(
                query_path
            )
            self.query_path = query_path
            with open(self.query_path) as f:
                self.query_dict = json.load(f)

        current_dir = Path(self.query_dict.get("output_dir")) 
        self.results_directory = current_dir / self.query_dict["name"]
        os.makedirs(self.results_directory, exist_ok=True)

        # Set up logging
        self.print = False
        for log_output in self.query_dict.get("logging", []):
            if log_output.lower() == "print":
                self.print = True
            else:
                log_output = self.results_directory / log_output
                logging.basicConfig(
                    filename=log_output, filemode="w+", level=logging.INFO, style="{"
                )
        # Check for an existing exsclaim json
        try:
            self.exsclaim_path = self.results_directory / "exsclaim.json"
            with open(self.exsclaim_path, "r") as f:
                # Load configuration file values
                self.exsclaim_dict = json.load(f)
        except Exception:
            self.logger.info("No exsclaim.json file found, starting a new one.")
            # Keep preset values
            self.exsclaim_dict = {}

    def display_info(self, info):
        """Display information to the user as the specified in the query

        Args:
            info (str): A string to display (either to stdout, a log file)
        """
        if self.print:
            Printer(info)
        self.logger.info(info)

    def run(
        self,
        tools=None,
        figure_separator=False,
        caption_distributor=False,
        journal_scraper=False,
        pdf_scraper=False,
        html_scraper=False,
        driver = None
    ):
        """Run EXSCLAIM pipeline on Pipeline instance's query path
        Args:
            tools (list of ExsclaimTools): list of ExsclaimTool objects
                to run on query path in the order they will run. Default
                argument is JournalScraper, CaptionDistributor,
                FigureSeparator
            journal_scraper (boolean): true if JournalScraper should
                be included in tools list. Overriden by a tools argument
            caption_distributor (boolean): true if CaptionDistributor should
                be included in tools list. Overriden by a tools argument
            figure_separator (boolean): true if FigureSeparator should
                be included in tools list. Overriden by a tools argument
        Returns:
            exsclaim_dict (dict): an exsclaim json
        Modifies:
            self.exsclaim_dict
        """
        if tools is None:
            tools = []
            if journal_scraper:
                tools.append(JournalScraper(self.query_dict))
            if pdf_scraper:
                tools.append(PDFScraper(self.query_dict))
            if html_scraper:
               tools.append(HTMLScraper(self.query_dict, driver))
            if caption_distributor:
                tools.append(CaptionDistributor(self.query_dict))
            if figure_separator:
                tools.append(FigureSeparator(self.query_dict))
        # run each ExsclaimTool on search query
        for tool in tools:
            self.exsclaim_dict = tool.run(self.query_dict, self.exsclaim_dict)

        # group unassigned objects
        self.group_objects()

        # Save results as specified
        save_methods = self.query_dict.get("save_format", [])

        if "multimodal_info" in save_methods:
            self.to_multimodal()

        return self.exsclaim_dict

    def assign_captions(self, figure):
            """Assigns all captions to master_images JSONs for single figure

            Args:
                figure (dict): a Figure JSON
            Returns:
                masters (list of dicts): list of master_images JSONs
                unassigned (dict): the updated unassigned JSON
            """
            unassigned = figure.get("unassigned", [])
            masters = []

            captions = unassigned.get("captions", {})
            not_assigned = set([a["label"] for a in captions])

            for index, master_image in enumerate(figure.get("master_images", [])):
                label_json = master_image.get("subfigure_label", {})
                subfigure_label = label_json.get("text", index)
                # remove periods or commas from around subfigure label
                processed_label = subfigure_label.replace(")", "")
                processed_label = processed_label.replace("(", "")
                processed_label = processed_label.replace(".", "")
                paired = False
                for caption_label in captions:
                    # remove periods or commas from around caption label
                    processed_caption_label = caption_label["label"].replace(")", "")
                    processed_caption_label = processed_caption_label.replace("(", "")
                    processed_caption_label = processed_caption_label.replace(".", "")
                    # check if caption label and subfigure label match and caption label
                    # has not already been matched
                    if (
                        processed_caption_label.lower() == processed_label.lower()
                        and processed_caption_label.lower()
                        in [a.lower() for a in not_assigned]
                    ):
                      try:
                        print("caption_label" , caption_label["description"])
                      
                        master_image["caption"] = caption_label["description"]#.replace("\n", " ").strip()
                        master_image["keywords"] = caption_label["keywords"]
                        if self.exsclaim_dict.get("context_retrieval") is True:
                          master_image["context"] = caption_label["context"]
                        if self.exsclaim_dict.get("materials_NER") is True:
                          master_image["materials_ner"] = caption_label["materials_ner"]
                        # master_image["general"] = caption_label["general"]
                        masters.append(master_image)
                        
                        not_assigned.remove(caption_label["label"])
                        # break to next master image if a pairing was found
                        paired = True
                        break
                      except:
                        pass
                if paired:
                    continue
                # no pairing found, create empty fields
                # try: 
                master_image["caption"] = master_image.get("caption", [])
                master_image["keywords"] = master_image.get("keywords", [])
                if self.exsclaim_dict.get("context_retrieval") is True:                    
                    master_image["context"] = master_image.get("context",[])
                if self.exsclaim_dict.get("materials_NER") is True:
                    master_image["materials_ner"] = master_image.get("materials_ner",[])
                # master_image["general"] = master_image.get("general", [])
                masters.append(master_image)
                # except:
                #     pass

            # update unassigned captions
            new_unassigned_captions = []
            for caption_label in captions:
                if caption_label["label"] in not_assigned:
                    new_unassigned_captions.append(caption_label)

            unassigned["captions"] = new_unassigned_captions
            return masters, unassigned

    def group_objects(self):
        """Pair captions with subfigures for each figure in exsclaim json"""
        self.display_info("Matching Image Objects to Caption Text\n")
        counter = 1
        for figure in self.exsclaim_dict:
            self.display_info(
                ">>> ({0} of {1}) ".format(counter, +len(self.exsclaim_dict))
                + "Matching objects from figure: "
                + figure
            )

            figure_json = self.exsclaim_dict[figure]
            masters, unassigned = self.assign_captions(figure_json)

            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            counter += 1
        self.display_info(">>> SUCCESS!\n")
        with open(self.results_directory / "exsclaim.json", "w") as f:
            json.dump(self.exsclaim_dict, f, indent=3)

        return self.exsclaim_dict

    def process_image(self, figure_name: str) -> np.ndarray:
        """
        Loads an image from the specified folder and returns it as a numpy array.

        Args:
        figure_name (str): Name of the image file.

        Returns:
        np.ndarray: The loaded image as an array.
        """
        image_path = self.results_directory / "figures" / figure_name
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

    def to_multimodal(self):
        """Creates a folder named images_folder to upload all the cropped subfigures and a csv file with the image path,
        the caption and the keywords."""
        exsclaim_json = self.exsclaim_dict
        img_folder_dir = self.results_directory / "images_folder"
        os.makedirs(img_folder_dir, exist_ok=True)
        
        records = []

        for key, value in exsclaim_json.items():
            print('key_value',key,value)
            
            try:
                full_img = self.process_image(value["figure_name"])
            # print("full_img", full_img)
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
                    file_path = img_folder_dir / f"{img_name}.jpg"
                    subfigure_img.save(file_path)

                    records.append({
                       "full_caption": value["full_caption"],
                       "caption": caption,
                       "caption_summary": caption_summary,
                       "image": str(file_path),
                       "article_name": value["article_name"]
                        })
                    
                except Exception as e:
                    print(f"Error processing subfigure for {key}: {e}")
                    continue
        output_data = {"records": records}    
        output_path = self.results_directory / "retrieved_image_caption_pairs.json"
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=2)