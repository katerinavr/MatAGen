from matagen.exsclaim.pipeline import Pipeline

def pdf_scraper_tool(pdf_file_path: str, output_dir:str, paper_name:str, api_key):
    """
    Runs the HTML scraping pipeline and returns the results as a JSON-compatible dictionary.

    Args:
        folder (str): Path to the folder containing HTML files.
        api_key (Optional[str]): API key for authentication (defaults to OPENAI_API_KEY).

    Returns:
        dict: Results from the scraping pipeline in JSON format.
    """
    test_json = {
        "name": paper_name,
        "pdf_folder": pdf_file_path,
        "output_dir": output_dir,
        "llm": "gpt-4o",
        "openai_API": api_key,
        "save_format": ["multimodal_info"],
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
