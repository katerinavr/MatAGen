import panel as pn
import pandas as pd
import io
from PIL import Image
import os
import json
import tempfile
import time
import asyncio
import traceback
import logging # Added import

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Imports ---
# Import directly from matagen package (ensure it's installed via pip install -e .)
try:
    from matagen.agents.multimodal_system import ModelConfig, MultimodalAnalysisSystem
    # from matagen.scraping.xml_scraper import html_scraper_tool
    from matagen.scraping.pdf_scraper import pdf_scraper_tool
    from matagen.scraping.journal_scraper import journal_scraper_tool
    AUTOGEN_AVAILABLE = True
    print("Successfully imported matagen modules.")
except ImportError as e:
    logger.error(f"Fatal: Could not import core matagen modules: {e}. Mocking components.", exc_info=True)
    AUTOGEN_AVAILABLE = False
    # --- Define Mock classes ONLY if the real ones can't be imported ---
    class MockModelConfig:
        def __init__(self, *args, **kwargs): print(f"MockModelConfig Initialized...")
        def get_gpt4_config(self): return {}
    class MockMultimodalAnalysisSystem:
        def __init__(self, config): self.config = config
        async def initiate_chat(self, task):
            print(f"--- MOCK AutoGen --- Task: {task}") ; await asyncio.sleep(3)
            # MOCK: Use corrected path logic for consistency
            project_root_mock = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_json_path_mock = os.path.join(project_root_mock, "outputs", "html_scraping", "retrieved_image_caption_pairs.json")
            os.makedirs(os.path.dirname(output_json_path_mock), exist_ok=True)
            with open(output_json_path_mock, 'w') as f: json.dump({"mock": True, "processed_html_task": task}, f)
            logger.info(f"--- MOCK AutoGen End (Output: {output_json_path_mock}) ---")
            return {"status": "mock_success"}
    # Mock scraper tools if needed
    async def mock_scraper_tool(*args, **kwargs):
        tool_name = kwargs.get("tool_name", "generic")
        logger.warning(f"Using MOCK scraper tool: {tool_name}")
        await asyncio.sleep(1)
        return {"mock_result": True, "image_paths": [], "json_data": {"mock": True, "tool_called": tool_name}, "output_dir": f"mock_outputs_{tool_name}"}

    ModelConfig = MockModelConfig
    MultimodalAnalysisSystem = MockMultimodalAnalysisSystem
    # Assign mocks - Use lambda to capture tool_name for differentiation if needed
    html_scraper_tool = (lambda *a, **kw: mock_scraper_tool(*a, tool_name="html", **kw))
    pdf_scraper_tool = (lambda *a, **kw: mock_scraper_tool(*a, tool_name="pdf", **kw))
    journal_scraper_tool = (lambda *a, **kw: mock_scraper_tool(*a, tool_name="journal", **kw))