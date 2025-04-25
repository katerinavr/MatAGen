import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from termcolor import colored
from dataclasses import dataclass
import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str
from matagen.config.settings import OPENAI_API_KEY

# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()


@dataclass
class ModelConfig:
    """Configuration for language models and API keys."""
    openai_api_key: str

    def get_gpt4_config(self) -> List[Dict]:
        """Returns configuration list for GPT-4 model."""
        return [
            {
                'model': 'gpt-4o',
                # 'api_key': self.openai_api_key,
                # 'cache_seed': 42,
                # 'temperature': 0,
                # 'timeout': 120,
            }
        ]
    
config_models = ModelConfig(
            openai_api_key=OPENAI_API_KEY,
        )

config_list = [
    {
        "model": "gpt-4o",
        "api_key": OPENAI_API_KEY,
        "cache_seed": 42,
        "temperature": 0,
        "timeout": 120
    }
]
# config = ModelConfig(
#             openai_api_key=OPENAI_API_KEY,
#         )
# config_list_4v = autogen.config_list_from_json(
#     [{'model': 'gpt-4o',
#        'api_key': OPENAI_API_KEY
#     }],
#     filter_dict={
#         "model": ["gpt-4-vision-preview"],
#     },
# )

# config_list_gpt4 = autogen.config_list_from_json(
#     [{'model': 'gpt-4o',
#        'api_key': OPENAI_API_KEY
#     }],
#     filter_dict={
#         "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
#     },
# )

gpt4_llm_config = {"config_list": [{
                'model': 'gpt-4o',
                'api_key': OPENAI_API_KEY
            }], "cache_seed": 42}


working_dir = "tmp/"

# class FigureVisualizer(ConversableAgent):
#     def __init__(self, n_iters=2, **kwargs):
#         """
#         Initializes a FigureCreator instance.

#         This agent facilitates the creation of visualizations through a collaborative effort among its child agents: commander, coder, and critics.

#         Parameters:
#             - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 2.
#             - **kwargs: keyword arguments for the parent AssistantAgent.
#         """
#         super().__init__(**kwargs)
#         self.register_reply([Agent, None], reply_func=FigureVisualizer._reply_user, position=0)
#         self._n_iters = n_iters

#     def _reply_user(self, messages=None, sender=None, config=None, image=None):
#         if all((messages is None, sender is None)):
#             error_msg = f"Either {messages=} or {sender=} must be provided."
#             logger.error(error_msg)  # noqa: F821
#             raise AssertionError(error_msg)
#         if messages is None:
#             messages = self._oai_messages[sender]

#         user_question = messages[-1]["content"]

#         ### Define the agents
#         commander = AssistantAgent(
#             name="Commander",
#             human_input_mode="NEVER",
#             max_consecutive_auto_reply=10,
#             system_message="Help me run the code, and tell other agents it is in the <img result.jpg> file location.",
#             is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
#             code_execution_config={"last_n_messages": 3, "work_dir": working_dir, "use_docker": False},
#             llm_config=self.llm_config
#         )

#         critics = MultimodalConversableAgent(
#             name="Critics",
#             system_message="""Criticize the input figure. How to replot the figure so it will be better? Find bugs and issues for the figure.
#             Pay attention to the color, format, and presentation. Keep in mind of the reader-friendliness.
#             If you think the figures is good enough, then simply say NO_ISSUES""",
#             llm_config={"config_list": config_models.get_gpt4_config(), "max_tokens": 300},
#             human_input_mode="NEVER",
#             # max_consecutive_auto_reply=1,
#             #     use_docker=False,
#         )

#         coder = AssistantAgent(
#             name="Coder",
#             llm_config=self.llm_config
#         )

#         coder.update_system_message(
#             coder.system_message
#             + "ALWAYS save the figure in `result.jpg` file. Tell other agents it is in the <img result.jpg> file location."
#         )

#         # Data flow begins
#         commander.initiate_chat(coder, message=user_question)
#         img = Image.open(os.path.join(working_dir, image))
#         plt.imshow(img)
#         plt.axis("off")  # Hide the axes
#         plt.show()

#         for i in range(self._n_iters):
#             commander.send(
#                 message=f"Improve <img {os.path.join(working_dir, 'result.jpg')}>",
#                 recipient=critics,
#                 request_reply=True,
#             )

#             feedback = commander._oai_messages[critics][-1]["content"]
#             if feedback.find("NO_ISSUES") >= 0:
#                 break
#             commander.send(
#                 message="Here is the feedback to your figure. Please improve! Save the result to `result.jpg`\n"
#                 + feedback,
#                 recipient=coder,
#                 request_reply=True,
#             )
#             img = Image.open(os.path.join(working_dir, "result.jpg"))
#             plt.imshow(img)
#             plt.axis("off")  # Hide the axes
#             plt.show()

#         return True, os.path.join(working_dir, "result.jpg")
    

class FigureVisualizer(ConversableAgent):
    def __init__(self, n_iters=2, **kwargs):
        """
        Initializes a FigureVisualizer instance.
        
        This agent facilitates the analysis and improvement of visualizations through a collaborative 
        effort among its child agents: commander, critics, and coder.
        
        Parameters:
            - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 2.
            - **kwargs: keyword arguments for the parent ConversableAgent.
        """
        super().__init__(**kwargs)
        self.register_reply([Agent, None], reply_func=FigureVisualizer._reply_user, position=0)
        self._n_iters = n_iters
    
    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg) 
            raise AssertionError(error_msg)
            
        if messages is None:
            messages = self._oai_messages[sender]
        
        # Extract the user's message and determine if it contains an image
        user_message = messages[-1]["content"]
        
        # Look for image path in the user message
        image_path = None
        if isinstance(user_message, str):
            # Simple regex to find image path in format <img path/to/image.jpg>
            import re
            image_match = re.search(r'<img\s+(.*?)>', user_message)
            if image_match:
                image_path = image_match.group(1).strip()
        
        if not image_path:
            return False, "No image was provided in the message. Please include an image using <img path/to/image.jpg> format."

        ### Define the agents
        commander = AssistantAgent(
            name="Commander",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="Help analyze the input image.",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"last_n_messages": 3, "work_dir": working_dir, "use_docker": False},
            llm_config=self.llm_config
        )
        
        critics = MultimodalConversableAgent(
            name="RagAnalyser",
            system_message=f"""Answer with details and high precision to the user query {user_message}. 
                This is the most relevant retrieved context from ChromaDB: retrieved text {context} and 
                retrieved image <img {image_path}>.
                Structure your answers based on the retieved context and the image.
                Provide a comprehensive answer to the user query.
                """,
            llm_config={"config_list": self._get_gpt4_config(), "max_tokens": 300},
            human_input_mode="NEVER",
        )
        
        coder = AssistantAgent(
            name="Coder",
            llm_config=self.llm_config,
            system_message="Implement suggestions to improve the provided visualization. Save improved versions as 'improved_result.jpg'."
        )
        
        # Display the input image
        try:
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis("off")  # Hide the axes
            plt.title("Input Image")
            plt.show()
        except Exception as e:
            return False, f"Error opening image: {str(e)}"
        
        # Start with analysis of the provided image
        # commander.initiate_chat(
        #     critics, 
        #     message=f"Analyze this figure: <img {image_path}>. What improvements would you suggest?"
        # )
        
        # # Get initial feedback
        # feedback = commander._oai_messages[critics][-1]["content"]
        # if "NO_ISSUES" in feedback:
        #     return True, f"Analysis complete. The figure at {image_path} has no issues that need to be addressed."
            
        # Run improvement iterations if needed
        output_path = image_path
        # for i in range(self._n_iters):
            # Send feedback to coder
        commander.send(
            message=f"Save the image <img {output_path}> as 'improved_result.jpg'",
            recipient=coder,
            request_reply=True,
        )
            
        # Check if improved image was created
        improved_path = os.path.join(working_dir, "improved_result.jpg")
        if os.path.exists(improved_path):
            # Display improved image
            img = Image.open(improved_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Improved Image (Iteration {i+1})")
            plt.show()
            
            # Get feedback on improved image
            commander.send(
                message=f"Analyze this improved figure: <img {improved_path}>",
                recipient=critics,
                request_reply=True,
            )
            
            feedback = commander._oai_messages[critics][-1]["content"]
            # if "NO_ISSUES" in feedback:
            #     output_path = improved_path
            #     break
                
            output_path = improved_path
        else:
            # If no improved image was created
            return False, "Coder was unable to create an improved image."
        
        return True, f"Analysis and improvement complete. Final image: {output_path}"
    
    def _get_gpt4_config(self):
        """
        Returns a properly formatted GPT-4 configuration that works with AutoGen.
        This avoids the validation errors with cache_seed, temperature, and timeout.
        """
        # Example of a properly formatted config - modify based on your needs
        return [{
            "model": "gpt-4o",
            "api_key": OPENAI_API_KEY
        }]
