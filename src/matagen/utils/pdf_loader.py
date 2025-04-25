import fitz
import logging 
import os
import time
import tempfile

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pdf_to_txt(pdf_files_value, abs_pdf_temp_path, pdf_input, PDF_OUTPUT_BASE, TEMP_UPLOAD_DIR):
        all_results_summary = []; all_extracted_text_content = {}; all_image_panes = []; all_table_widgets = []
        pdf_temp_file_paths = []

        start_time = time.time(); total_files = len(pdf_files_value)
        for i, pdf_bytes in enumerate(pdf_files_value): # Process each file
            if isinstance(pdf_input.filename, list): original_filename = pdf_input.filename[i] if i < len(pdf_input.filename) else f"file_{i+1}.pdf"
            else: original_filename = pdf_input.filename if total_files == 1 else f"{os.path.splitext(pdf_input.filename)[0]}_{i+1}.pdf"
            base_filename = os.path.splitext(original_filename)[0]; current_file_num = i + 1
            # status_text.value = f'Processing PDF {current_file_num}/{total_files}: {original_filename} ({selected_mode})...'; progress_bar.value = int(5 + (i / total_files) * 90)

            # 1. Save Temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TEMP_UPLOAD_DIR, prefix=f"pdf_in_{int(time.time())}_") as temp_pdf:
                temp_pdf.write(pdf_bytes); pdf_temp_file_path = temp_pdf.name
            pdf_temp_file_paths.append(pdf_temp_file_path); abs_pdf_temp_path = os.path.abspath(pdf_temp_file_path)
            logger.info(f"Saved temp PDF [{current_file_num}/{total_files}]: {abs_pdf_temp_path}")

            safe_filename_base = "".join(c for c in base_filename if c.isalnum() or c in ('_', '-')).rstrip()
            run_output_base = os.path.join(PDF_OUTPUT_BASE, f"{safe_filename_base}_{int(time.time())}")
            os.makedirs(run_output_base, exist_ok=True)
            file_summary = {"filename": original_filename}

            logger.info(f"Starting PDF text extraction for {original_filename}")
            extracted_text = ""; output_txt_path = None
            try:
                    doc = fitz.open(abs_pdf_temp_path)
                    for page_num in range(len(doc)): page = doc.load_page(page_num); extracted_text += page.get_text("text", sort=True) + "\n\n" # Added sort=True
                    doc.close()
                    output_txt_filename = f"{base_filename}_text_only.txt"; output_txt_path = os.path.join(run_output_base, output_txt_filename)
                    with open(output_txt_path, 'w', encoding='utf-8') as txt_file: txt_file.write(extracted_text)
                    logger.info(f"Saved text to: {output_txt_path}")
                    file_summary.update({"status": "Success (Text Only)", "output_text_file": output_txt_path, "char_count": len(extracted_text)})
                    all_extracted_text_content[original_filename] = extracted_text # Store text for chat
            except Exception as text_exc:
                    logger.error(f"PDF text extraction failed for {original_filename}: {text_exc}", exc_info=True)
                    file_summary.update({"status": "Failed (Text Only)", "error": str(text_exc)}); all_extracted_text_content[original_filename] = f"ERROR: {text_exc}"
