import os
from pathlib import Path
import shutil 
from matagen.agents import *
from matagen.scraping.pdf_scraper import pdf_scraper_tool
from matagen.config.settings import OPENAI_API_KEY

api_key = ""
pdf_file_path = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\temp"
output_dir = r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run\adma200800280"
paper_name = "adma200800280"
pdf_scraper_tool(pdf_file_path, output_dir, paper_name, api_key)