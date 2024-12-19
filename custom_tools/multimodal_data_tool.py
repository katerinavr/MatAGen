import os
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader, Document

class MultimodalDatasetProcessor:
    def __init__(self, exsclaim_output_folder, api_key):
        self.exsclaim_output_folder = exsclaim_output_folder
        self.api_key = api_key
        self.img_folder_dir = Path("multimodal_data_folder") / "images_folder"
        os.makedirs(self.img_folder_dir, exist_ok=True)
        self.data = self.load_exsclaim_json()
        self.classes = [
            'chemical structures', 'TEM/SEM/AFM images', 'mechanical testing data', 'photographic images',
            'XRD patterns', 'schematic illustration', 'energy level diagrams', 'wearable electronics',
            'absorption spectra', 'TFT and OFET architectures', 'EDX imaging', 'GIWAXS patterns',
            'FTIR spectra', 'electrical characterization', 'NMR spectra', 'Raman spectra', 'XPS spectra'
        ]
        self.openai_mm_llm = OpenAIMultiModal(
            model="gpt-4o", api_key=api_key, max_new_tokens=4096
        )

    def load_exsclaim_json(self):
        """Loads the exsclaim.json file."""
        exsclaim_json = Path(self.exsclaim_output_folder) / "exsclaim.json"
        with open(exsclaim_json, 'r') as f:
            return json.load(f)

    def process_image(self, figure_name):
        """Loads an image from the specified folder and returns it as a numpy array."""
        image_path = os.path.join(self.exsclaim_output_folder, "figures", figure_name)
        # print('image_path', image_path)
        img = Image.open(image_path)
        return np.array(img)

    def crop_image(self, geometry, image):
        """Crops an image to the specified geometry and returns the cropped image as a numpy array."""
        x1, y1 = geometry[0]["x"], geometry[0]["y"]
        x2, y2 = geometry[3]["x"], geometry[3]["y"]
        return image[y1:y2, x1:x2]

    def create_multimodal_dataset(self):
        """Creates a dataset with cropped subfigures and their corresponding captions and keywords."""
        records = []

        for key in self.data.keys():
            full_img = self.process_image(self.data[key]["figure_name"])

            for i, master_image in enumerate(self.data[key]["master_images"]):
                geometry = master_image["geometry"]
                label = master_image["subfigure_label"]["text"]

                if label == '0':
                    caption = self.data[key]["full_caption"]
                    caption_summary = [] #self.data[key]["full_caption"]
                else:
                    caption = master_image.get("caption", "")
                    caption_summary = master_image.get("keywords", "")

                img_name = f"{self.data[key]['figure_name'].rsplit('.', 1)[0]}_{label}"
                subfigure = self.crop_image(geometry, full_img)

                try:
                    subfigure_img = Image.fromarray(subfigure)
                    if subfigure_img.mode == 'RGBA':
                        subfigure_img = subfigure_img.convert('RGB')
                    file_path = self.img_folder_dir / f"{img_name}.jpg"
                    subfigure_img.save(file_path)

                    records.append({
                        'full_caption': self.data[key]["full_caption"],
                        'caption': caption,
                        'caption_summary': caption_summary,
                        'image': str(file_path),
                        'article_name': self.data[key]["article_name"]
                    })

                except Exception as error:
                    print("An exception occurred:", error)

        image_data_df = pd.DataFrame(records)
        image_data_df.to_csv(Path("multimodal_data_folder") / 'multimodal_dataset.csv', index=False)
        return image_data_df

    def evaluate_image_quality(self, image_path):
        """Evaluates the quality of an image using the GPT-4 vision model."""
        image_documents = [ImageDocument(image_path=image_path)]
        attempts = 3
        while attempts > 0:
            try:
                response = self.openai_mm_llm.complete(
                    prompt="""You are an AI trained to evaluate the quality of cropped subfigures from full figures. 
                            Please assess the following aspects of the provided image:
                            1. Clean Edges: Are the edges of the image clean without partial cut-off elements?
                            2. Overall Quality: Does the image maintain high visual and informational quality?                   
                            Return 1 if the image quality is good or 0 if not. 
                            Only 1 and 0 are allowed, do not return any other information.""",
                    image_documents=image_documents
                )
                return int(response.text.strip())
            except Exception as error:
                print("LLM img quality evaluation error:", error)
                attempts -= 1
                if attempts <= 0:
                    print(f"Error: Failed to process paper.")
                    break
                print(f"Error: {str(error)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
                time.sleep(60)
                
                return "None"


    def classify_image(self, image_path, caption):
        """
        Classifies an image and its caption using the GPT-4 vision model.
        """
        try:
            from llama_index.core.schema import ImageDocument
            import mimetypes

            # Verify the file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Create the ImageDocument
            image_document = ImageDocument(image_path=image_path, text=caption)
            print(f"ImageDocument created successfully: {image_document}")
            mime_type, _ = mimetypes.guess_type(image_path)

            print(f"MIME type of the file: {mime_type}")  

            # Simulate API call (replace with actual API logic)
            response = self.openai_mm_llm.complete(
                prompt=f"Classify this image into predefined classes.",
                image_documents=[image_document]
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error in classify_image: {e}")
            return "Error"

    # def classify_image(self, image_path, caption):
    #     """Classifies an image and its caption using the GPT-4 vision model."""
    #     print('image_path', image_path)
    #     cleaned_image_path = f"cleaned_{Path(image_path).name}"
    #     with Image.open(image_path) as img:
    #         img = img.convert("RGB")
    #         img.save(cleaned_image_path, format="JPEG")

        
    #     attempts = 3
    #     while attempts > 0:
    #         try:
    #             image_documents = [ImageDocument(image_path=cleaned_image_path, text=caption)]
    #             response = self.openai_mm_llm.complete(
    #                 prompt=f"Assign one or more of the following tags to the image: {self.classes}. "
    #                     "Return only the predefined classes as a string and do not add any other information or create new classes. "
    #                     "If none of the tags is relevant, assign the tag 'None'.",
    #                 image_documents=image_documents
    #             )
    #             return response.text.strip()
    #         except Exception as error:
    #             print("LLM img classification error:", error)
    #             attempts -= 1
    #             if attempts <= 0:
    #                 print(f"Error: Failed to process paper.")
    #                 break
    #             print(f"Error: {str(error)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
    #             time.sleep(60)
                
    #             return "None"

    def classify_images_with_gpt4(self, dataframe):
        for idx, row in dataframe.iterrows():
            print(f"Processing row {idx}: Image: {row['image']}, Caption: {row['caption']}")
            result = self.classify_image(row['image'], row['caption'])
            print(f"Result: {result}")
        """Classifies images using the GPT-4 vision model and checks the quality of cropped images."""
        # def classify_images(row):
        #     # image_quality = self.evaluate_image_quality(row['image'])
        #     response = self.classify_image(row['image'], row['caption'])
        #     print(response)
        #     return pd.Series([response]) #, image_quality

        # dataframe['gpt4_output'] = dataframe.apply(classify_images, axis=1) #, 'image_quality'
        # dataframe.to_csv("gpt4_multimodal_results.csv", index=False)





if __name__ == "__main__":
    folder = "C:/Users/kvriz/Desktop/DataMiningAgents/custom_tools/external_tools/exsclaim/output/html-scraping"
    # api_key = os.getenv('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')
    api_key = "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"
    processor = MultimodalDatasetProcessor(folder, api_key)

    # multimodal_df = processor.create_multimodal_dataset()

    # multimodal_df.to_csv("gpt4_multimodal_results.csv", index=False)
    multimodal_df = pd.read_csv("gpt4_multimodal_results.csv")
    # print('multimodal_df', multimodal_df)
    # print(multimodal_df['image'].values[0], multimodal_df['caption'].values[0])
    # image_documents = SimpleDirectoryReader(multimodal_df['image'].values[0]).load_data()

    # processor.classify_image(image_documents, multimodal_df['caption'].values[0])
    # processor.classify_images_with_gpt4(multimodal_df)
    caption = "The legend specifies the values of the respective absorption maxima for both high- and low-energy transitions."
    response = processor.classify_image("multimodal_data_folder/images_folder/nmat2272_fig1_b.jpg", caption)
    print(response)



# from pathlib import Path
# from PIL import Image

# def verify_image(image_path):
#     """Verify that the image exists and is in a valid format."""
#     image_file = Path(image_path)
#     if not image_file.is_file():
#         raise ValueError(f"Image path is invalid: {image_path}")
#     try:
#         with Image.open(image_file) as img:
#             img.verify()
#             print(f"Valid image: {image_path}")
#     except Exception as e:
#         raise ValueError(f"Invalid image file: {image_path}, Error: {e}")
    

# verify_image("multimodal_data_folder/images_folder/nmat2272_fig1_b.jpg")

