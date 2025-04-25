from matagen.exsclaim.pipeline import Pipeline
from matagen.config.settings import OPENAI_API_KEY
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
        "name": "html_scraping",
        "html_folder": folder,
        "llm": "gpt-3.5-turbo",
        "openai_API": api_key,
        "save_format": ["multimodal_info"],
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