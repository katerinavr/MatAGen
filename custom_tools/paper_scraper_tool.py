import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_tools.external_tools.exsclaim.exsclaim.pipeline import Pipeline
from config.settings import OPENAI_API_KEY
from typing import Any, Optional

def html_scraper_tool(folder: str, api_key: Optional[str] = OPENAI_API_KEY) -> dict:
    """
    Runs the HTML scraping pipeline and returns the results as a JSON-compatible dictionary.

    Args:
        folder (str): Path to the folder containing HTML files.
        api_key (Optional[str]): API key for authentication (defaults to OPENAI_API_KEY).

    Returns:
        dict: Results from the scraping pipeline in JSON format.
    """
    test_json = {
        "name": "html-scraping",
        "html_folder": folder,
        "llm": "gpt-3.5-turbo",
        "openai_API": api_key,
        "save_format": ["save_subfigures"],
        "logging": ["print", "exsclaim.log"]
    }

    test_pipeline = Pipeline(test_json)
    results = test_pipeline.run(
        figure_separator=True,
        caption_distributor=True,
        journal_scraper=False,
        html_scraper=True,
        driver=None
    )
    
    return results


def pdf_scraper_tool(folder: str, api_key):
    """Tool for extracting multimodal data from PDF papers"""
    test_json = {
        "name": "omiecs_reviews",
        "pdf_folder": folder,
        "llm": "gpt-3.5-turbo",
        "openai_API": api_key,
        "save_format": ["save_subfigures"],
        "logging": ["print", "exsclaim.log"]
    }
    
    test_pipeline = Pipeline(test_json)
    results = test_pipeline.run(
        figure_separator=True,
        caption_distributor=True,
        journal_scraper=False,
        pdf_scraper=True,
        driver=None
    )
    return results

if __name__ == "__main__":
    api_key = "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"
    folder = "C:/Users/kvriz/Desktop/DataMiningAgents/html_folder" #test_folder
    html_scraper_tool(folder, api_key) 
    # pdf_scraper_tool(folder)