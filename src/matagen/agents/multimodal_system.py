import autogen
from autogen import UserProxyAgent
import sys
from typing import Any, Dict, List
import os
from matagen.scraping.paper_scraper_tool import html_scraper_tool
# from custom_tools.multimodal_data_retriever import multimodal_data_retriever_tool
from autogen import register_function
from autogen import ConversableAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from pathlib import Path
import json
# from custom_tools.claude_abs_extraction import ImageAnalysisAgent
# from custom_tools.molecular_structure_segmentation_tool import *
from pathlib import Path
import pandas as pd
import types
from openai.resources.chat.completions import Completions

"""
Multi-agent system for scientific literature analysis and data extraction.
This module implements a system of agents that work together to analyze scientific papers,
extract data from figures, and classify research content.
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union
import json
import logging
from dataclasses import dataclass

import autogen
from autogen import (
    UserProxyAgent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager
)
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
# from chromadb.utils import embedding_functions
from openai.resources.chat.completions import Completions # Make sure this is imported

# --- Apply the patch globally when the module is imported ---
try:
    original_create = Completions.create

    def patched_create(*args, **kwargs):
        """Simple fix for missing tool_call_id issues"""
        if "messages" in kwargs:
            last_tool_call_id = None
            # Find the latest tool_call_id from assistant messages
            for msg in reversed(kwargs.get("messages", [])):
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tool_call in msg.get("tool_calls", []):
                        if "id" in tool_call:
                            last_tool_call_id = tool_call["id"]
                            break  # Found the ID in the most recent tool call list
                    if last_tool_call_id:
                        break # Stop searching previous messages

            # Apply the found ID to tool messages that are missing it
            if last_tool_call_id:
                for msg in kwargs.get("messages", []):
                    if msg.get("role") == "tool" and not msg.get("tool_call_id"):
                        msg["tool_call_id"] = last_tool_call_id
                        # Optional: Log that the patch was applied
                        # print(f"PATCH: Applied tool_call_id '{last_tool_call_id}' to tool message.")

        # Call the original function
        return original_create(*args, **kwargs)

    # Apply the patch
    Completions.create = patched_create
    print("INFO: Applied OpenAI Completions.create patch for tool_call_id.") # Confirmation log

except AttributeError:
    print("WARNING: Could not apply OpenAI Completions.create patch (AttributeError).")
except Exception as patch_e:
     print(f"WARNING: Failed to apply OpenAI Completions.create patch: {patch_e}")
# --- End of Patch ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for language models and API keys."""
    openai_api_key: str
    anthropic_api_key: str
    
    def get_gpt4_config(self) -> Dict:
        """Returns configuration for GPT-4 model."""
        return {
            # "cache_seed": 42,
            # "temperature": 0,
            "config_list": [{
                'model': 'gpt-4o',
                'api_key': self.openai_api_key
            }],
            # "timeout": 120,
        }
    
    def get_claude_config(self) -> List[Dict]:
        """Returns configuration for Claude model."""
        return [{
            "model": "claude-3-5-sonnet-20240620",
            "api_key": self.anthropic_api_key,
            "api_type": "anthropic",
        }]


class ImageClassifier:
    """Handles classification of scientific figures and images."""
    
    VALID_CLASSES = {
        'chemical structures', 'TEM/SEM/AFM images', 'mechanical testing data',
        'photographic images', 'XRD patterns', 'schematic illustration',
        'energy level diagrams', 'wearable electronics', 'absorption spectra',
        'TFT and OFET architectures', 'EDX imaging', 'GIWAXS patterns',
        'FTIR spectra', 'electrical characterization', 'NMR spectra',
        'Raman spectra', 'XPS spectra'
    }

    def __init__(self, admin_agent: UserProxyAgent, multimodal_agent: MultimodalConversableAgent):
        self.admin = admin_agent
        self.multimodal_agent = multimodal_agent

    def classify_image(self, image_path: str, caption: str) -> str:
        """Classifies a single image based on its caption."""
        try:
            response = self.admin.initiate_chat(
                self.multimodal_agent,
                clear_history=True,
                silent=False,
                max_turns=1,
                message=f"""Classify the image based on its caption {caption}. 
                Select strictly only one of the allowed classes: {self.VALID_CLASSES}. 
                <img {image_path}>.
                Return only the class name without any additional text.
                Expected output: 'electrical characterization' """
            )
            return response.chat_history[-1]['content']
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            return "Error"

class MultimodalAnalysisSystem:
    """Main system for multimodal scientific literature analysis."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.setup_agents()
        self._setup_group_chat()
        # self.process_dataset()
        
    def setup_agents(self):
        """Initializes and configures all agents in the system."""

        # Initialize the journal scraping assistant agent
        self.journal_scraping_assistant = ConversableAgent(
            name="multimodal_data_assistant",
            system_message="""Multimodal data retrieval assistant for scientific papers.
            Ensures sequential task completion. Returns 'TERMINATE' when tasks are complete.""",
            llm_config=self.config.get_gpt4_config(),
        )

        # Initialize multimodal agent
        self.multimodal_agent = MultimodalConversableAgent(
            name="multi_modal_agent",
            system_message="""Specialized agent for extracting information from scientific plots. 
            Use the assigned functions for image classification.
            The a JSON file with the information is named retrieved_image_caption_pairs.json and is located inside the html-scraping folder""",
            llm_config=self.config.get_gpt4_config(),
            description="Scientific figure analysis and categorization specialist"
        )

        # Initialize admin agent
        self.admin = UserProxyAgent(
            name="admin",
            is_termination_msg=lambda msg: msg.get("content", "").strip() == "TERMINATE",
            human_input_mode="NEVER",
            system_message="Admin agent responsible for task management and coordination.",
            llm_config=self.config.get_gpt4_config(),
            code_execution_config=False
        )

        # Register functions to agents
        register_function(html_scraper_tool, 
            caller=self.journal_scraping_assistant,
            executor=self.admin,
            name="Paper_Scraper",
            description="Scrape HTML files for multimodal (image and text) data extraction. The final results will be saved in a JSON file named retrieved_image_caption_pairs.json and is located inside the html-scraping folder."
        )

        def process_dataset_wrapper(json_file_name: str = None) -> Dict:
            """Wrapper function to call the class method process_dataset"""
            return self.process_dataset(json_file_name)
            
        register_function(
            process_dataset_wrapper,
            caller=self.multimodal_agent,
            executor=self.admin,
            name="multimodal_image_agent",
            description="Processes a dataset of images and captions, assigning classifications and extracting metadata. Always call this function when asked to classify scientific images."
        )

        self._register_tools()
         
        
    def _register_tools(self):
        # Register image classification functionality
        self.classifier = ImageClassifier(self.admin, self.multimodal_agent)
        
    def process_dataset(self, json_file_name: str) -> None:  #, json_file_path: Path
        """
        Processes the entire dataset, performing classification and metadata extraction.
        
        Args:
            json_file: Path to the JSON file containing the dataset
        """
        json_file = Path(f"html_scraping/{json_file_name}") #
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for record in data['records']:
                # Classify image
                record['classification'] = self.classifier.classify_image(
                    record['image'],
                    record['caption']
                )
                
                # # Extract metadata if applicable
                # if record['classification'].lower() == "absorption spectra":
                #     image_path = record["image"]
                #     analyzer = ImageAnalysisAgent(anthropic_api_key)
                #     data_points = analyzer.analyze_image(image_path)
                #     record['metadata'] = data_points
                #     # record['metadata'] = self._extract_metadata(record['image'])
                    
            # Save updated data
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            logger.info(f"Successfully processed dataset and saved to {json_file}")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

    def _setup_group_chat(self):
        """Set up group chat and manager."""
        self.groupchat = autogen.GroupChat(
            agents=[self.admin, self.journal_scraping_assistant, self.multimodal_agent],
            messages=[],
            max_round=20,
            select_speaker_auto_llm_config=self.config.get_gpt4_config(),
            speaker_selection_method="round_robin",
        )        
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.config.get_gpt4_config())
    
    def initiate_chat(self, prompt: str) -> Any:
        """Starts a chat session with the admin agent."""
        try:
            return self.admin.initiate_chat(
                self.manager,
                message=prompt,
            )
        except Exception as e:
            if "tool_call_id" in str(e) and "task" in str(self.manager.groupchat.messages[-1].get("content", "")):
                return {"status": "success", "note": "Completed"}
            raise

    # def initiate_chat(self, prompt: str) -> Any:
    #     """Starts a chat session with the admin agent."""
    #     return self.admin.initiate_chat(
    #         self.manager,
    #         message=prompt,
    #         )
    
def main():
    """Main entry point for the application."""
    try:
        import types
        from openai.resources.chat.completions import Completions

        original_create = Completions.create

        Completions.create = patched_create

        openai_api_key, anthropic_api_keys = OPENAI_API_KEY, anthropic_api_key
        
        config_models = ModelConfig(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_keys
        )
        
        system = MultimodalAnalysisSystem(config_models)      

        task = """
        "Extract the images and text from the files in the html_folder." \
        "Use the appropriate functions to classify the scientific images based on both the image and caption and extract the metadata."\
        """
        
        # Start the chat with the task
        chat_result = system.initiate_chat(task)
        logger.info(f"Chat completed. Result: {chat_result}")
        
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
        sys.exit(1)
        
if __name__ == "__main__":
    main()
