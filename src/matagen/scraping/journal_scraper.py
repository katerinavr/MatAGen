from matagen.exsclaim.pipeline import Pipeline

def journal_scraper_tool(journal: str, keywords:str, api_key):
    """Tool for extracting multimodal data from PDF papers"""
    test_json = {
        "name": "omiecs_reviews",
        "keywords":  keywords,
        "journal": "nature",
        "llm": "gpt-3.5-turbo",
        "openai_API": api_key,
        "save_format": ["save_subfigures"],
        "logging": ["print", "exsclaim.log"]
    }
    
    test_pipeline = Pipeline(test_json)
    results = test_pipeline.run(
        figure_separator=True,
        caption_distributor=True,
        journal_scraper=True,
        pdf_scraper=False,
        driver=None
    )
    return results
