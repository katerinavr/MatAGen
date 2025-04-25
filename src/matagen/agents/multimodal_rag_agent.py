from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from typing import Dict, List
import logging
from dataclasses import dataclass
from autogen import (
    UserProxyAgent)
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
import matplotlib.pyplot as plt
from PIL import Image

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
            "cache_seed": 42,
            "temperature": 0,
            "config_list": [{
                'model': 'gpt-4o',
                'api_key': self.openai_api_key
            }],
            "timeout": 120,
        }
    
    def get_claude_config(self) -> List[Dict]:
        """Returns configuration for Claude model."""
        return [{
            "model": "claude-3-5-sonnet-20240620",
            "api_key": self.anthropic_api_key,
            "api_type": "anthropic",
        }]

class MultimodalAgent:
    """Handles tasks related to scientific data containing figures."""

    def __init__(self, admin_agent: UserProxyAgent, multimodal_agent: MultimodalConversableAgent):
        self.admin = admin_agent
        self.multimodal_agent = multimodal_agent

    def multimodal_rag(self, image_path: str, query: str, context:StopIteration) -> str:
        """Gets as an input the human query and the retrieved image or text chunks data from the ChromaDB
            retrieval and provides a comprehensive context aware answer."""
        try:
            response = self.admin.initiate_chat(
                self.multimodal_agent,
                clear_history=True,
                silent=False,
                max_turns=1,
                message=f"""Answer with details and high precision to the user query {query}. 
                This is the most relevant retrieved context from ChromaDB: retrieved text {context} and 
                retrieved image <img {image_path}>.
                Structure your answers based on the retieved context and the image.
                Provide a comprehensive answer to the user query.
                """,
            )
            try:
                img = Image.open(image_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis("off")  # Hide the axes
                plt.title("Input Image")
                plt.show()
            except Exception as e:
                return False, f"Error opening image: {str(e)}"
            return response.chat_history[-1]['content']
        
        except Exception as e:
            logger.error(f"Error with image content {image_path}: {e}")
            return "Error"
        
                # Display the input image

