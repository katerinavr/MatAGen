import os
from pathlib import Path
import shutil 
from matagen.agents import *
from matagen.scraping.pdf_scraper import pdf_scraper_tool
from matagen.config.settings import OPENAI_API_KEY

# Configuration
pdf_input_dir = r"C:\Users\kvriz\Desktop\DataMiningAgents\data\omiec_pdfs"
output_base_dir =  r"C:\Users\kvriz\Desktop\DataMiningAgents\outputs\omiecs_run"
completion_marker_filename = "retrieved_image_caption_pairs.json" # The file indicating  of the process in a directory


# --- Ensure base output directory exists ---
try:
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Base output directory: '{output_base_dir}'")
except OSError as e:
    print(f"Error creating base directory {output_base_dir}: {e}")
    exit()

# --- Iterate through files in the original input directory ---
print(f"Scanning for PDF files in: '{pdf_input_dir}'")

try:
    all_items = os.listdir(pdf_input_dir)
except FileNotFoundError:
    print(f"Error: Input directory not found: {pdf_input_dir}")
    all_items = []

processed_count = 0
skipped_count = 0
error_count = 0

for item_name in all_items:
    original_pdf_path = os.path.join(pdf_input_dir, item_name)

    if os.path.isfile(original_pdf_path) and item_name.lower().endswith(".pdf"):
        pdf_filename = item_name
        pdf_base_name = os.path.splitext(pdf_filename)[0]

        final_output_dir = os.path.join(output_base_dir, pdf_base_name, pdf_base_name)
        # 2. Path to the specific completion marker file
        completion_marker_path = os.path.join(final_output_dir, completion_marker_filename)

        # --- Check if Already Processed ---
        # Ensure the potential output directory exists before checking inside it
        # This is generally good practice, though if the marker exists, the dir should too.
        # Use exist_ok=True so it doesn't error if the dir is already there.
        try:
            os.makedirs(final_output_dir, exist_ok=True)
        except OSError as e:
            print(f"\nError ensuring output directory {final_output_dir} exists for check: {e}")
            error_count += 1
            continue # Skip this file if we can't even ensure the output dir exists

        # Now, check if the completion marker file is present
        if os.path.exists(completion_marker_path):
            print(f"\nSkipping PDF: {pdf_filename} (Already processed - found '{completion_marker_filename}')")
            skipped_count += 1
            continue  # <<<--- Move to the next PDF file in the input directory
        else:
             # If the marker doesn't exist, proceed with processing
             print(f"\nProcessing PDF: {pdf_filename} (Not previously completed)")

        # --- Proceed with Temporary Directory Setup and Tool Execution ---
        # (This part only runs if the PDF was NOT skipped)
        temp_input_dir = os.path.join(output_base_dir, f"__temp_{pdf_base_name}")
        temp_pdf_path = os.path.join(temp_input_dir, pdf_filename)

        try:
            # Create Temp Dir (remove old one if necessary)
            if os.path.exists(temp_input_dir):
                 shutil.rmtree(temp_input_dir)
            os.makedirs(temp_input_dir)
            # print(f"  Created temporary input directory: '{temp_input_dir}'") # Verbose logging

            # Copy the single PDF to the temporary directory
            # print(f"  Copying '{pdf_filename}' to temporary directory...") # Verbose logging
            shutil.copy2(original_pdf_path, temp_pdf_path)
            # print(f"  Copy complete.") # Verbose logging

            # Call the scraper tool
            print(f"  Running scraper tool for '{pdf_filename}'...")
            pdf_scraper_tool(
                pdf_file_path=temp_input_dir,
                output_dir=final_output_dir,
                paper_name=pdf_base_name,
                api_key=OPENAI_API_KEY
            )
            print(f"  Tool finished for '{pdf_filename}'.")
            processed_count += 1

        except Exception as e:
            print(f"  ERROR processing '{pdf_filename}' with the tool: {e}")
            error_count += 1
        finally:
            # Cleanup: ALWAYS remove the temporary directory
            try:
                # print(f"  Cleaning up temporary directory: '{temp_input_dir}'") # Verbose logging
                if os.path.exists(temp_input_dir):
                    shutil.rmtree(temp_input_dir)
                # print(f"  Cleanup complete.") # Verbose logging
            except OSError as e:
                print(f"  Warning: Could not remove temporary directory {temp_input_dir}: {e}")

# --- Summary ---
print(f"\n--- Processing Complete ---")
print(f"PDFs newly processed in this run: {processed_count}")
print(f"PDFs skipped (already processed): {skipped_count}")
print(f"Errors encountered during processing: {error_count}")
print(f"Final outputs are in subdirectories under: '{output_base_dir}'")