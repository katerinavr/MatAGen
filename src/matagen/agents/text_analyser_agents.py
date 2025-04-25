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


def run_text_analysis_chat(user_query, context, api_key):
    print('tza')