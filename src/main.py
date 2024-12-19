# src/main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from agents import *

# chat_result = user_proxy.initiate_chat(manager, message="Extract the images and text from the html files in the html_folder")
# chat_result = user_proxy.initiate_chat(image_segmentation_agent, message="Extract the metadata from the image in the html_folder")
OPENAI_API_KEY = "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"

import autogen
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic
from autogen.agentchat.contrib.img_utils import get_pil_image
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path


    # 4. Identify distinct curves/peaks
    # 5. Extract (wavelength, absorbance) coordinates
    # 6. Return data as list of tuples for each curve
    # 7. Handle multiple overlapping spectra if present
    # 8. Account for axis scales and units    
    # 9. Identify the polymer names that correspond to each spectra based on the color

# First, set up the configuration for Claude
config_list_claude = [
    {
        "model": "claude-3-5-sonnet-20240620",
        "api_key": "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",
        "api_type": "anthropic"
    }
]

def save_to_json(data: dict, image_path: str, output_dir: str = "extracted_data") -> str:
    """
    Save the extracted data to a JSON file.
    
    Args:
        data: Dictionary containing the extracted data
        image_path: Original image path (used for naming)
        output_dir: Directory to save the JSON file
    
    Returns:
        str: Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create filename based on original image name and timestamp
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{image_name}_data_{timestamp}.json"
    json_path = Path(output_dir) / json_filename
    
    # Add metadata to the data
    output_data = {
        "metadata": {
            "source_image": image_path,
            "extraction_date": datetime.now().isoformat(),
            "version": "1.0"
        },
        "data": data
    }
    
    # Save to JSON with pretty formatting
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    return str(json_path)

# Create the image segmentation agent
image_segmentation_agent = autogen.AssistantAgent(
    name="spectra segmentation agent",
    system_message="""You are an expert in analyzing spectroscopy plots and extracting data points.
    For each provided absorption spectrum plot:
    1. Extract all the points from the plot. Give me the tuples of wavelength and absorbance for each curve. 
    2. Provide as many points as possible. 
    3. Return the coordinates in a json format and assign the correct name to the coordinates as shown in the legend.
    
    Return the data in a structured format suitable for JSON export.""",
    llm_config={
        "config_list": config_list_claude,
        "temperature": 0.0,
        "request_timeout": 120,
        "seed": 42,
        "use_cache": True
    }
)

# Create a user proxy agent
abs_segment = autogen.UserProxyAgent(
    name="abs_segment",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=1,
    code_execution_config={
        # "work_dir": "coding",
        "use_docker": False,
    },
)


def load_and_process_image(image_path: str, max_size: tuple = (800, 800)) -> str:
    """
    Load, optimize, and encode image file to base64 string
    Includes resizing and compression to reduce token size
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize image while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to BytesIO with optimization
            buffered = BytesIO()
            img.save(buffered, format="JPEG", 
                    optimize=True, 
                    quality=85)  # Adjust quality as needed
            buffered.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Print size information for debugging
            print(f"Optimized image size: {len(img_str)} characters")
            return img_str
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        raise


def analyze_and_save_spectrum(image: str, output_dir: str = "extracted_data") -> tuple:
    """
    Analyze spectrum and save results to JSON.
    
    Args:
        image_path: Path to the spectrum image
        output_dir: Directory to save JSON output
    
    Returns:
        tuple: (chat_messages, json_file_path)
    """

    # Get the analysis results
    chat_messages = abs_segment.initiate_chat(
        image_segmentation_agent,
        message=f"Please analyze the absorption spectrum in this image and extract all data points: {image}"
    )
    
    # Extract the data from the last assistant message
    # Note: You might need to adjust this based on your actual response format
    try:
        last_message = [msg for msg in chat_messages if msg.get('role') == 'assistant'][-1]
        extracted_data = last_message.get('content', {})
        print('extracted_data', extracted_data)
        
        # Save to JSON
        json_path = save_to_json(extracted_data, image_path, output_dir)
        return chat_messages, json_path
    except Exception as e:
        print(f"Error processing data: {e}")
        return chat_messages, None

# Example usage
if __name__ == "__main__":
    image_path = "./src"
    # image = load_and_process_image(image_path) 
    # print('image', image)
    chat_results, json_file = analyze_and_save_spectrum("https://storage.googleapis.com/polymer_abs_test/Screen%20Shot%202022-11-21%20at%201.32.01%20PM.png")
    if json_file:
        print(f"Results saved to: {json_file}")






# analyze_spectrum('html_folder/nmat2272_fig1_b_gra.jpg')


# from termination_conditions import get_termination_condition
# from autogen_agentchat.teams import RoundRobinGroupChat
# from autogen_agentchat.task import Console

# async def main():
#     agents = create_agents()
#     termination = get_termination_condition()

#     # Define a team with agents
#     agent_team = RoundRobinGroupChat(agents=agents, termination_condition=termination)

#     # Start the interaction
#     stream = agent_team.run_stream(task="What is the weather in New York?")
#     await Console(stream)

# # Entry point for the script
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

# user_proxy.initiate_chat(
#     manager,
#     message="""
# Extract the images and text from the html files in the html_folder
# """,
# )


# chat_result = user_proxy.initiate_chat(assistant, message="Extract the images and text from the html files in the html_folder")

