import autogen
import anthropic
import base64
import httpx
import json
import time
from typing import Dict, List, Tuple
import re
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_tools import plot_extracted_abs
import matplotlib.pyplot as plt
from config.claude_system_prompt import CLAUDE_SYSTEM_PROMPT

                             
class ImageAnalysisAgent:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        
    def get_image_data(self, image_url: str, media_type: str) -> str:
        """Fetch and encode image data"""
        response = httpx.get(image_url)
        return base64.standard_b64encode(response.content).decode("utf-8")
    
    def get_image_data_local(self, image_path: str) -> str:
        """Encode local image data to base64"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_json_from_response(self, text: str) -> dict:
        """Extract and parse JSON from Claude's response"""
        try:
            return json.loads(text)
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {text}")
            raise e
    
    def analyze_image(self, image_path, max_retries: int = 3) -> Dict[str, List[Tuple[float, float]]]: # media_type: str,
        """Analyze image using Claude and return extracted data points"""
        # image_data = self.get_image_data(image_url, media_type)
        # image_data = self.get_image_data_local(image_path)#, media_type)
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        supported_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
        }
        
        extension = image_path.split('.')[-1].lower()
        
        if extension not in supported_types:
            raise ValueError(f"Unsupported image format: {extension}. Supported formats are: jpg, jpeg, png, gif, webp")
        
        media_type = supported_types[extension]
        # media_type = f"image/{image_path.split('.')[-1]}" 
        # print('image_data', image_data)
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    #model="claude-3-5-sonnet-20240620",
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4096,
                    system = CLAUDE_SYSTEM_PROMPT,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": [
                            {   
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                }
                            },
                            {
                                "type": "text",
                                "text": """Extract all the spectral points from the image with maximum precision. 
                                        1. Give me the wavelength and absorbance for each curve.
                                        2. Each curve name is given in the legend and is associated with the exact color of the lines.
                                        3. Remove all the background noise, focusing on the true signal shapes
                                        4. Pay extreme attention to:
                                            - Half width at half maximum
                                            - Peak maxima
                                            - Valley minima
                                            - Spectrum intersection points
                                            - Baseline profiles
                                            - Shoulder features
                                            - Steep gradient region
                                        5. Provide:
                                            - ALL the exact peak positions
                                            - ALL intensity values to high precision of 0.01
                                            - Baseline values at 5 nm intervals
                                        6. VERIFICATION PROCESS:
                                        EXECUTE in order:
                                            a) Look at the original image and compare
                                            b) Generate plot matching original scale
                                            c) VERIFY against original:
                                                - Peak heights MUST match
                                                - Valley depths MUST align
                                                - Crossing points MUST be exact
                                                - Baselines MUST overlay
                                            d) Correct ANY discrepancies
                                            e) Produce final verification plot
                                        7. Return the points in a json format and assign the correct name to the coordinates as shown in the legend. 
                                        8. Only return the JSON object with no additional text or formatting.

                                        Expected Output Format:
                                        {
                                            "Name": {
                                                "wavelength": [350, 355, 360, 365, 370, ...],
                                                "absorbance": [0.00, 0.003, 0.10, 0.103, 0.118, ...]
                                              }
                                        }                                        



                                        PROVIDE numerical data only.
                                        ENSURE all data matches original plot.                                
                                                            
                                """
                            }
                        ]
                    }]
                )
                json_data = self.extract_json_from_response(response.content[0].text)              
                
                return json_data 
                
            except anthropic.InternalServerError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)


def run_analysis_workflow(image_path, anthropic_api_key: str): 
    """ Function to run the abs data extraction using Claude."""
    try:
        # Initialize the image analysis agent
        analyzer = ImageAnalysisAgent(anthropic_api_key)

        image_directory = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)
        # Extract data points from the image
        data_points = analyzer.analyze_image(image_path)
        with open(f"{image_name}_extracted_abs.json", 'w') as f:
                    json.dump(data_points , f, indent=2)        
        fig = plot_extracted_abs.plot_smoothed_absorbance(data_points)
        # fig = plot_extracted_abs.plot_spectra(data_points)
        img_name = f"{image_name}_extracted_abs.png"
        plt.savefig(img_name, dpi=600)
        # plt.show()
        return img_name

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(f"Error type: {type(e)}")



if __name__ == "__main__":
    from src.matagen.config.settings import anthropic_api_key

    # image_url = "https://storage.googleapis.com/polymer_abs_test/Screen%20Shot%202022-11-21%20at%201.32.01%20PM.png"
    image_path = "C:/Users/kvriz/Desktop/DataMiningAgents/test_files/cleaned_nmat2272_fig1_b.jpg" #"C:/Users/kvriz/Desktop/DataMiningAgents/src/nmat2272_fig1_b_gra.jpg"

    run_analysis_workflow(image_path, anthropic_api_key)