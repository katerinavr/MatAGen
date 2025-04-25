import os
from pathlib import Path
import shutil 
from matagen.agents import *
from matagen.scraping.pdf_scraper import pdf_scraper_tool
from matagen.config.settings import OPENAI_API_KEY

pdf_input_dir = r"C:\Users\kvriz\Desktop\DataMiningAgents\data\ecps_pdfs"
output_base_dir =  r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\ecps_run"

# --- Ensure base output directory exists ---
try:
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Base output directory: '{output_base_dir}'")
except OSError as e:
    print(f"Error creating base directory {output_base_dir}: {e}")
    exit() # Exit if we can't create the base output

# --- Iterate through files in the original input directory ---
print(f"Scanning for PDF files in: '{pdf_input_dir}'")

try:
    all_items = os.listdir(pdf_input_dir)
except FileNotFoundError:
    print(f"Error: Input directory not found: {pdf_input_dir}")
    all_items = [] # Prevent further errors

processed_count = 0
error_count = 0

for item_name in all_items:
    original_pdf_path = os.path.join(pdf_input_dir, item_name)

    # Check if it's a file and if it ends with .pdf (case-insensitive)
    if os.path.isfile(original_pdf_path) and item_name.lower().endswith(".pdf"):
        pdf_filename = item_name
        pdf_base_name = os.path.splitext(pdf_filename)[0]
        print(f"\nProcessing PDF: {pdf_filename}")

        # --- Define Paths ---
        # 1. Final output directory for this specific PDF
        final_output_dir = os.path.join(output_base_dir, pdf_base_name)
        # 2. Temporary input directory (create a unique name)
        temp_input_dir = os.path.join(output_base_dir, f"__temp_{pdf_base_name}")

        temp_pdf_path = os.path.join(temp_input_dir, pdf_filename) # Path for the copied PDF

        # --- Create Directories ---
        try:
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"  Ensured final output directory exists: '{final_output_dir}'")
            # Remove previous temp dir if it exists (e.g., from failed previous run)
            if os.path.exists(temp_input_dir):
                 shutil.rmtree(temp_input_dir)
            os.makedirs(temp_input_dir)
            print(f"  Created temporary input directory: '{temp_input_dir}'")
        except OSError as e:
            print(f"  Error creating directories for {pdf_filename}: {e}")
            error_count += 1
            continue # Skip this file

        # --- Prepare and Run Tool ---
        try:
            # Copy the single PDF to the temporary directory
            print(f"  Copying '{pdf_filename}' to temporary directory...")
            shutil.copy2(original_pdf_path, temp_pdf_path) # copy2 preserves metadata
            print(f"  Copy complete.")

            # Call the scraper tool pointing to the temporary directory
            print(f"  Running scraper tool...")
            pdf_scraper_tool(
                pdf_file_path=temp_input_dir,       # <<< Input is the TEMP directory
                output_dir=final_output_dir,      # <<< Output is the FINAL specific directory
                paper_name=pdf_base_name,         # Pass paper name, tool might use it internally
                api_key=OPENAI_API_KEY
            )
            print(f"  Tool finished for '{pdf_filename}'.")
            processed_count += 1

        except Exception as e:
            print(f"  ERROR processing '{pdf_filename}' with the tool: {e}")
            # Optional: Log the full traceback
            # logging.exception(f"Error processing {pdf_filename}")
            error_count += 1

        finally:
            # --- Cleanup: ALWAYS remove the temporary directory ---
            try:
                print(f"  Cleaning up temporary directory: '{temp_input_dir}'")
                if os.path.exists(temp_input_dir):
                    shutil.rmtree(temp_input_dir)
                print(f"  Cleanup complete.")
            except OSError as e:
                print(f"  Warning: Could not remove temporary directory {temp_input_dir}: {e}")

# --- Summary ---
print(f"\n--- Processing Complete ---")
print(f"Successfully processed PDFs (tool ran): {processed_count}")
print(f"Errors encountered during processing: {error_count}")
print(f"Final outputs should be in subdirectories under: '{output_base_dir}'")