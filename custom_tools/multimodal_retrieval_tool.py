import os
import numpy as np 
import pandas as pd
import shutil
from pathlib import Path
from llama_index.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


class MultimodalDatasetRetriever:
    def __init__(self, api_key, prompt):
        self.api_key = api_key
        self.prompt = prompt
        self.multimodal_csv = Path("multimodal_data_folder") /  "gpt4_multimodal_results.csv"
        self.multimodal_data_folder_dir = Path("multimodal_data_folder")
        self.dataframe = pd.read_csv(self.multimodal_csv)
        self.openai_mm_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", api_key=api_key, max_new_tokens=500)

        
    def get_multimodal_retriever(self):
        """Given the multimodal database information it extract the relevant images based on the prompt description.
        Then it creates a separate folder named after the image class and copies the relevant images to the folder for 
        post-processing."""

        gpt4v_retrieved_images = []
        image_title =[]
        
       
        for input_image in self.dataframe.image:
            image_caption = self.dataframe["caption"][self.dataframe["image"] == input_image].values[0]
            image_documents = [ImageDocument(image_path=input_image, text=image_caption)]
            image_title.append(input_image)
            try:
                response_1 = self.openai_mm_llm.complete(
                    prompt=f"""Identify the images that can be better described with the 
                    prompt {self.prompt}. Return only the image path of these images.""", 
                    
                    image_documents=image_documents,
                )
                print(response_1.text)
                gpt4v_retrieved_images.append(response_1)
                
            except Exception as error:
                print("An exception occurred:", error)
                gpt4v_retrieved_images.append(0)
        return gpt4v_retrieved_images
    
    def create_folder(self):
        if prompt=='absorption spectra':
            abs_folder = os.path.join(self.multimodal_data_folder_dir, 'abs_spectra')
            os.makedirs(abs_folder, exist_ok=True)
            img_paths = self.dataframe.image[self.dataframe.gpt4o_output == 'absorption spectra']
            for image_path in img_paths:
                image_name = os.path.basename(image_path)
                dest_path = os.path.join(abs_folder, image_name)
                shutil.copy(image_path, dest_path)
        
        if prompt=='chemical structures':
            abs_folder = os.path.join(self.multimodal_data_folder_dir, 'chemical structures')
            os.makedirs(abs_folder, exist_ok=True)
            img_paths = self.dataframe.image[self.dataframe.gpt4o_output == 'chemical structures']
            for image_path in img_paths:
                image_name = os.path.basename(image_path)
                dest_path = os.path.join(abs_folder, image_name)
                shutil.copy(image_path, dest_path)

        
if __name__ == "__main__":
    prompt = """chemical structures"""
    api_key = os.getenv('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')

    processor = MultimodalDatasetRetriever(api_key, prompt)
    multimodal_df = processor.create_folder()
    print(multimodal_df)
