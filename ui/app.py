import panel as pn
import panel.chat # chat components
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
import fitz  # PyMuPDF for PDF text extraction
from bs4 import BeautifulSoup # For HTML text extraction


# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Imports ---
try:
    from matagen.agents.multimodal_system import ModelConfig, MultimodalAnalysisSystem
    from matagen.agents.text_analyser_agents import run_text_analysis_chat 
    from matagen.scraping.xml_scraper import html_scraper_tool
    from matagen.scraping.pdf_scraper import pdf_scraper_tool
    from matagen.scraping.journal_scraper import journal_scraper_tool
    AUTOGEN_AVAILABLE = True
    logger.info("Successfully imported matagen modules.")
    if 'run_text_analysis_chat' not in locals():
         async def mock_text_analysis_chat(user_query, context, api_key):
              logger.warning("Using MOCK text analysis chat.")
              await asyncio.sleep(2)
              return f"Mock response to: '{user_query}'. Context length: {len(context)} chars."
         run_text_analysis_chat = mock_text_analysis_chat

except ImportError as e:
    logger.error(f"Fatal: Could not import core matagen modules: {e}. Mocking components.", exc_info=True)
    AUTOGEN_AVAILABLE = False
    # --- Define Mock classes / functions ---
    class MockModelConfig:
        def __init__(self, *args, **kwargs): print(f"MockModelConfig Initialized...")
        def get_gpt4_config(self): return {}
    class MockMultimodalAnalysisSystem:
        def __init__(self, config): self.config = config
        async def initiate_chat(self, task): print(f"--- MOCK AutoGen --- Task: {task}") ; await asyncio.sleep(3); return {"status": "mock_success"}
    async def mock_scraper_tool(*args, **kwargs): logger.warning(f"Using MOCK scraper tool: {kwargs.get('tool_name', 'generic')}"); await asyncio.sleep(1); return {"mock_result": True, "image_paths": [], "json_data": {"mock": True, "tool_called": kwargs.get('tool_name', 'generic'), "extracted_full_text": "Mock text", "extracted_tables": []}, "output_dir": "mock_outputs"}
    async def mock_text_analysis_chat(user_query, context, api_key): logger.warning("Using MOCK text analysis chat."); await asyncio.sleep(2); return f"Mock response to: '{user_query}'. Context length: {len(context)} chars."
    ModelConfig = MockModelConfig; MultimodalAnalysisSystem = MockMultimodalAnalysisSystem
    html_scraper_tool = (lambda *a, **kw: mock_scraper_tool(*a, tool_name="html", **kw))
    pdf_scraper_tool = (lambda *a, **kw: mock_scraper_tool(*a, tool_name="pdf", **kw))
    journal_scraper_tool = (lambda *a, **kw: mock_scraper_tool(*a, tool_name="journal", **kw))
    run_text_analysis_chat = mock_text_analysis_chat

# --- State Management ---
app_session_data = {"extracted_text": {}}

# --- Panel App Code ---
pn.extension('tabulator', 'notifications', 'jsoneditor')

# --- Constants & Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
HTML_OUTPUT_BASE = os.path.join(project_root, "outputs", "html_runs")
PDF_OUTPUT_BASE = os.path.join(project_root, "outputs", "pdf_runs")
JOURNAL_OUTPUT_BASE = os.path.join(project_root, "outputs", "journal_runs")
TEMP_UPLOAD_DIR = os.path.join(project_root, "temp_uploads")
HTML_FIXED_OUTPUT_JSON_PATH = os.path.join(HTML_OUTPUT_BASE, "retrieved_image_caption_pairs.json") # If needed
os.makedirs(HTML_OUTPUT_BASE, exist_ok=True); os.makedirs(PDF_OUTPUT_BASE, exist_ok=True)
os.makedirs(JOURNAL_OUTPUT_BASE, exist_ok=True); os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
logger.info(f"Project Root: {project_root}, Temp Dir: {TEMP_UPLOAD_DIR}, PDF Output: {PDF_OUTPUT_BASE}")

# --- UI Components ---
# Sidebar
openai_key_input = pn.widgets.PasswordInput(name="OpenAI API Key", placeholder="sk-...", width_policy='min')
anthropic_key_input = pn.widgets.PasswordInput(name="Anthropic API Key", placeholder="sk-ant-...", width_policy='min')
sidebar = pn.Column(pn.pane.Markdown("### API Configuration"), openai_key_input, anthropic_key_input, width=250, height_policy='max', styles={"border-right": "1px solid #ddd", "padding-right": "15px"})

# Actions
HTML_MODE_OPTIONS = ['Text Only', 'Text and Image (Scraping)']; PDF_MODE_OPTIONS = ['Text Only', 'Text and Image (Scraping)']
html_mode_radio = pn.widgets.RadioButtonGroup(name='HTML Mode', options=HTML_MODE_OPTIONS, value=HTML_MODE_OPTIONS[1], button_type='default', orientation='vertical')
pdf_mode_radio = pn.widgets.RadioButtonGroup(name='PDF Mode', options=PDF_MODE_OPTIONS, value=PDF_MODE_OPTIONS[1], button_type='default', orientation='vertical')
pdf_input = pn.widgets.FileInput(accept='.pdf', name='Upload PDF(s)', width_policy='max', multiple=True)
process_pdf_button = pn.widgets.Button(name='Process PDF(s)', button_type='primary', disabled=True)
pdf_action_col = pn.Column(pdf_input, pdf_mode_radio, process_pdf_button, sizing_mode='stretch_width')
html_input = pn.widgets.FileInput(accept='.html, .htm', name='Upload HTML', width_policy='max')
process_html_button = pn.widgets.Button(name='Process HTML', button_type='primary', disabled=True)
html_action_col = pn.Column(html_input, html_mode_radio, process_html_button, sizing_mode='stretch_width')
search_input = pn.widgets.TextInput(name='Search Keywords', placeholder='Enter keywords...', width_policy='max')
search_button = pn.widgets.Button(name='Search Literature', button_type='primary', disabled=True)
search_action_col = pn.Column(search_input, search_button, sizing_mode='stretch_width')
actions_bar = pn.Card(pn.Row(html_action_col, pdf_action_col, search_action_col, sizing_mode='stretch_width'), title="Actions", sizing_mode='stretch_width')

# Status
status_text = pn.widgets.StaticText(value='Please enter API Keys and select an action.')
progress_bar = pn.widgets.Progress(name='Progress', value=0, max=100, active=False, bar_color='info', width=200)
status_bar = pn.Row(status_text, progress_bar, sizing_mode='stretch_width')

# Chat Interface
chat_feed = pn.chat.ChatFeed(height=450, auto_scroll_limit=100) # Slightly shorter feed height
# Use TextAreaInput for chat
chat_input = pn.widgets.TextAreaInput(
    placeholder="Ask questions about the extracted text...", name="Chat Input",
    auto_grow=True, max_rows=5, rows=1 # Configure text area size
)
send_button = pn.widgets.Button(name="Send", button_type='primary', icon='send', disabled=True)
chat_spinner = pn.indicators.LoadingSpinner(value=False, width=30, height=30, align='center')
chat_interface = pn.Column(
    pn.pane.Markdown("### Chat with Extracted Text"),
    chat_feed,
    pn.Row(chat_input, send_button), # Input and button in a row
    chat_spinner,
    sizing_mode='stretch_width', margin=(25, 0, 0, 0), visible=True
)

# Results Tabs
TAB_HEIGHT = 300 # Define height for tab content
summary_output = pn.pane.JSON({}, name='JSON Summary', depth=-1, theme='light', height=TAB_HEIGHT, sizing_mode='stretch_width')
text_output = pn.pane.Markdown("### Extracted Text\n---", height=TAB_HEIGHT, sizing_mode='stretch_width',styles={'overflow-y': 'auto'})
image_output_grid = pn.GridBox(ncols=4, sizing_mode='stretch_width')
image_output_col = pn.Column(pn.pane.Markdown("### Extracted Images\n---"), image_output_grid, height=TAB_HEIGHT, sizing_mode='stretch_width')
table_output_inner_col = pn.Column(sizing_mode='stretch_width')
table_output_col = pn.Column(pn.pane.Markdown("### Extracted Tables\n---"), table_output_inner_col, height=TAB_HEIGHT,  sizing_mode='stretch_width')
search_results_output = pn.pane.Markdown("### Search Results\n---", height=TAB_HEIGHT, sizing_mode='stretch_width')
results_tabs = pn.Tabs(
    ("Summary", summary_output), ("Extracted Text", text_output), ("Extracted Images", image_output_col),
    ("Extracted Tables", table_output_col), ("Search Results", search_results_output),
    sizing_mode='stretch_width', dynamic=False
)

# --- UI Input State Check Function ---
def update_button_states(*events):
    if progress_bar.active: return
    keys_present = bool(openai_key_input.value and anthropic_key_input.value)
    html_file_present = html_input.value is not None
    pdf_files_present = pdf_input.value is not None and isinstance(pdf_input.value, list) and len(pdf_input.value) > 0
    search_terms_present = bool(search_input.value)
    chat_input_present = bool(chat_input.value)
    text_context_present = bool(app_session_data["extracted_text"])

    process_html_button.disabled = not (keys_present and html_file_present)
    process_pdf_button.disabled = not (keys_present and pdf_files_present)
    search_button.disabled = not (keys_present and search_terms_present)
    send_button.disabled = not (keys_present and chat_input_present and text_context_present)
    chat_interface.visible = text_context_present # Control chat visibility

    # Update status text
    if not keys_present: status_text.value = "Ready. Please enter API Keys."
    elif not text_context_present: status_text.value = "Ready. Process HTML/PDF ('Text Only' mode) to enable chat."
    elif html_file_present or pdf_files_present or search_terms_present:
         ready_actions = []
         if html_file_present and not process_html_button.disabled: ready_actions.append("HTML")
         if pdf_files_present and not process_pdf_button.disabled: ready_actions.append("PDF(s)")
         if search_terms_present and not search_button.disabled: ready_actions.append("Search")
         status = f"Ready for: {', '.join(ready_actions)}. " if ready_actions else "Ready. "
         status += "Chat enabled." if text_context_present else ""
         status_text.value = status
    else: status_text.value = "Ready. Chat enabled. Upload file(s) or enter search keywords."


# --- Callback Functions ---

async def process_html_callback(event):
    html_file_value = html_input.value; openai_key = openai_key_input.value; anthropic_key = anthropic_key_input.value
    selected_mode = html_mode_radio.value
    if html_file_value is None or not openai_key or not anthropic_key: pn.state.notifications.error('Error: Missing file or API Key(s)!', duration=4000); return

    # Disable UI Controls
    process_html_button.disabled = True; process_pdf_button.disabled = True; search_button.disabled = True; html_input.disabled = True; pdf_input.disabled = True; search_input.disabled = True; openai_key_input.disabled = True; anthropic_key_input.disabled = True; send_button.disabled = True
    status_text.value = f'Processing HTML ({selected_mode})...'; progress_bar.active = True; progress_bar.value = 5
    summary_output.object = {}; text_output.object = "### Extracted Text\n---"; image_output_grid.objects = []; table_output_inner_col.objects = []
    results_tabs.active = 0; temp_html_file_path = None

    try:
        original_filename = html_input.filename if html_input.filename else "uploaded.html"; base_filename = os.path.splitext(original_filename)[0]
        # Save temp file
        safe_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '_')).rstrip()
        temp_html_file_path = os.path.join(TEMP_UPLOAD_DIR, f"html_{int(time.time())}_{safe_filename}")
        with open(temp_html_file_path, 'wb') as temp_html: temp_html.write(html_file_value)
        abs_temp_html_path = os.path.abspath(temp_html_file_path)
        status_text.value = f'Saved temp HTML: {os.path.basename(abs_temp_html_path)}'; progress_bar.value = 10; logger.info(f"Temp HTML: {abs_temp_html_path}")

        if selected_mode == 'Text Only':
            status_text.value = 'Extracting text from HTML...'; progress_bar.value = 30; logger.info(f"HTML text extraction for {original_filename}")
            extracted_text = ""; output_txt_path = None
            try:
                with open(abs_temp_html_path, 'r', encoding='utf-8', errors='ignore') as f: soup = BeautifulSoup(f, 'html.parser')
                body = soup.find('body'); extracted_text = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)
                output_txt_filename = f"{base_filename}_text_only_{int(time.time())}.txt"; os.makedirs(HTML_OUTPUT_BASE, exist_ok=True)
                output_txt_path = os.path.join(HTML_OUTPUT_BASE, output_txt_filename)
                with open(output_txt_path, 'w', encoding='utf-8') as txt_file: txt_file.write(extracted_text)
                logger.info(f"Saved text to: {output_txt_path}")
                progress_bar.value = 90

                app_session_data['extracted_text'] = {original_filename: extracted_text} # Store/overwrite context
                chat_interface.visible = True # Make chat visible
                if not chat_feed.value: chat_feed.append({"user": "System","avatar": "ℹ️","value": "Text context loaded. Ask questions!"}) # Add initial msg

                summary_output.object = {"status": "Success (Text Only)", "input": original_filename, "output": output_txt_path,"chars": len(extracted_text)}
                text_output.object = f"### Extracted Text (HTML)\n---\n**File:** {original_filename}\n\n```\n{extracted_text}\n```"
                results_tabs.active = [p.name for p in results_tabs.objects].index("Extracted Text") # Use updated tab name
            except Exception as text_exc: logger.error(f"HTML text extraction failed: {text_exc}", exc_info=True); raise RuntimeError(f"Failed text extraction: {text_exc}")
            status_text.value = f'HTML text extraction finished!'; pn.state.notifications.success('HTML text extracted!', duration=4000)

        elif selected_mode == 'Text and Image (Scraping)':
            status_text.value = 'Initiating full HTML scraping...'; progress_bar.value = 20; logger.info("Starting full HTML scraping...")
            app_session_data['extracted_text'] = {}; chat_interface.visible = False # Clear context, hide chat

            # AutoGen/Tool Call Logic (Option A: Fixed path assumed)
            task = f"""Process HTML at '{abs_temp_html_path}'. Save results to '{HTML_FIXED_OUTPUT_JSON_PATH}'. Then process that JSON. TERMINATE."""
            logger.debug(f"AutoGen Task: {task}")
            if not AUTOGEN_AVAILABLE: raise RuntimeError("AutoGen module not available.")
            config = ModelConfig(openai_api_key=openai_key, anthropic_api_key=anthropic_key)
            system = MultimodalAnalysisSystem(config); chat_result = await system.initiate_chat(task); logger.info(f"AutoGen chat finished: {chat_result}")
            if not os.path.exists(HTML_FIXED_OUTPUT_JSON_PATH): raise FileNotFoundError(f"Expected HTML output file not found: {HTML_FIXED_OUTPUT_JSON_PATH}")
            await asyncio.sleep(0.5);
            with open(HTML_FIXED_OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f: summary_data = json.load(f)
            logger.info(f"Read HTML output JSON from {HTML_FIXED_OUTPUT_JSON_PATH}")
            summary_output.object = summary_data; results_tabs.active = 0
            status_text.value = f'HTML scraping finished!'; pn.state.notifications.success('HTML scraped!', duration=4000)

        progress_bar.value = 100

    except Exception as e: # Main error handler
        logger.error(f"Error during HTML processing ({selected_mode}): {e}", exc_info=True)
        detailed_error = traceback.format_exc(); status_text.value = f'Error: {str(e)}'; pn.state.notifications.error(f'HTML Processing failed: {str(e)}. Check logs.', duration=6000)
        summary_output.object = {"error": str(e), "mode": selected_mode, "details": detailed_error}
    finally: # Cleanup and UI re-enable
        if temp_html_file_path and os.path.exists(temp_html_file_path):
            try: os.remove(temp_html_file_path); logger.info(f"Cleaned temp file: {temp_html_file_path}")
            except Exception as clean_e: logger.warning(f"Failed to clean temp file {temp_html_file_path}: {clean_e}")
        openai_key_input.disabled = False; anthropic_key_input.disabled = False; html_input.disabled = False; pdf_input.disabled = False; search_input.disabled = False
        progress_bar.active = False ; progress_bar.value = 0
        update_button_states() # Reset button states
#---------------------------------------------------------------------------------------------------------------------------------------------
# PDF processing callback
async def process_pdf_callback(event):
    pdf_files_value = pdf_input.value; openai_key = openai_key_input.value; anthropic_key = anthropic_key_input.value; selected_mode = pdf_mode_radio.value
    if not pdf_files_value or not openai_key or not anthropic_key: pn.state.notifications.error('Error: Missing PDF file(s) or API Key(s)!', duration=4000); return

    # --- Disable UI ---
    process_html_button.disabled = True; process_pdf_button.disabled = True; search_button.disabled = True; html_input.disabled = True; pdf_input.disabled = True; search_input.disabled = True; openai_key_input.disabled = True; anthropic_key_input.disabled = True; send_button.disabled = True
    status_text.value = f'Processing {len(pdf_files_value)} PDF file(s) ({selected_mode})...'; progress_bar.active = True; progress_bar.value = 5
    summary_output.object = {}; text_output.object = "### Extracted Text\n---"; image_output_grid.objects = []; table_output_inner_col.objects = []
    results_tabs.active = 0; app_session_data['extracted_text'] = {}; chat_interface.visible = True

    all_results_summary = []; all_extracted_text_content = {}; all_image_panes = []; all_table_widgets = []
    pdf_temp_file_paths = []

    try:
        start_time = time.time(); total_files = len(pdf_files_value)
        for i, pdf_bytes in enumerate(pdf_files_value): # Process each file
            if isinstance(pdf_input.filename, list): original_filename = pdf_input.filename[i] if i < len(pdf_input.filename) else f"file_{i+1}.pdf"
            else: original_filename = pdf_input.filename if total_files == 1 else f"{os.path.splitext(pdf_input.filename)[0]}_{i+1}.pdf"
            base_filename = os.path.splitext(original_filename)[0]; current_file_num = i + 1
            status_text.value = f'Processing PDF {current_file_num}/{total_files}: {original_filename} ({selected_mode})...'; progress_bar.value = int(5 + (i / total_files) * 90)

            # 1. Save Temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TEMP_UPLOAD_DIR, prefix=f"pdf_in_{int(time.time())}_") as temp_pdf:
                temp_pdf.write(pdf_bytes); pdf_temp_file_path = temp_pdf.name
            pdf_temp_file_paths.append(pdf_temp_file_path); abs_pdf_temp_path = os.path.abspath(pdf_temp_file_path)
            logger.info(f"Saved temp PDF [{current_file_num}/{total_files}]: {abs_pdf_temp_path}")

            safe_filename_base = "".join(c for c in base_filename if c.isalnum() or c in ('_', '-')).rstrip()
            run_output_base = os.path.join(PDF_OUTPUT_BASE, f"{safe_filename_base}_{selected_mode.replace(' ','_')}_{int(time.time())}")
            os.makedirs(run_output_base, exist_ok=True)
            file_summary = {"filename": original_filename, "mode": selected_mode}

            # --- Execute based on mode ---
            if selected_mode == 'Text Only':
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

            elif selected_mode == 'Text and Image (Scraping)':
                logger.info(f"Starting full PDF scraping for {original_filename}")
                if not AUTOGEN_AVAILABLE: raise RuntimeError("Matagen module not available.")
                try:
                    result_dict = await asyncio.to_thread( # Call the backend tool
                         pdf_scraper_tool, pdf_file_path=abs_pdf_temp_path, output_base_dir=run_output_base, api_key=openai_key,
                    )
                    file_summary.update({"status": "Success (Scraping)", **result_dict})
                    logger.info(f"Finished scraping '{original_filename}'. Outputs in: {result_dict.get('output_dir', 'N/A')}")
                    # Process results for UI display
                    json_data = result_dict.get("json_data", {}); image_paths = result_dict.get("image_paths", [])
                    tool_text = json_data.get("extracted_full_text", ""); extracted_tables_data = json_data.get("extracted_tables", [])
                    if tool_text: all_extracted_text_content[original_filename] = tool_text # Store text if tool provided it
                    for img_path in image_paths: # Create image panes...
                         try:
                            with open(img_path, 'rb') as img_f: img_bytes = img_f.read()
                            ext = os.path.splitext(img_path)[1].lower() ; img_pane = None
                            if ext == ".png": img_pane = pn.pane.PNG(img_bytes, width=150, name=os.path.basename(img_path))
                            elif ext in [".jpg", ".jpeg"]: img_pane = pn.pane.JPG(img_bytes, width=150, name=os.path.basename(img_path))
                            if img_pane: all_image_panes.append(img_pane)
                         except Exception as img_e: logger.error(f"Error loading image {img_path}: {img_e}")
                    for i, table_data in enumerate(extracted_tables_data): # Create table widgets...
                        try:
                            df = pd.DataFrame(table_data); table_widget = pn.widgets.Tabulator(df, pagination='local', page_size=10)
                            all_table_widgets.extend([pn.pane.Markdown(f"**Table {i+1} from {original_filename}**"), table_widget])
                        except Exception as table_e: logger.error(f"Error creating table widget: {table_e}")
                except Exception as scrape_exc:
                    logger.error(f"PDF scraping failed for {original_filename}: {scrape_exc}", exc_info=True)
                    file_summary.update({"status": "Failed (Scraping)", "error": str(scrape_exc)})

            all_results_summary.append(file_summary) # Add this file's summary
        # --- End of loop ---

        # --- Update UI with Aggregated Results ---
        progress_bar.value = 95; status_text.value = "Aggregating results..."; await asyncio.sleep(0.1)
        app_session_data['extracted_text'] = all_extracted_text_content # Update global context
        display_text = "### Extracted Text\n---\n" + "\n\n---\n\n".join( f"#### From: {fname}\n\n{text_content}" for fname, text_content in all_extracted_text_content.items())
        summary_output.object = {"processed_files": all_results_summary}; text_output.object = display_text
        image_output_grid.objects = all_image_panes; table_output_inner_col.objects = all_table_widgets

        # Activate appropriate tab
        active_tab_name = "Summary";
        if all_image_panes: active_tab_name = "Extracted Images"
        elif all_table_widgets: active_tab_name = "Extracted Tables"
        elif app_session_data['extracted_text']: active_tab_name = "Extracted Text"
        try: # Set active tab using correct logic
            tab_names = [p.name for p in results_tabs.objects]; target_index = tab_names.index(active_tab_name)
            results_tabs.active = target_index; logger.info(f"Activating tab: '{active_tab_name}'")
        except (ValueError, AttributeError) as e: logger.warning(f"Could not activate tab '{active_tab_name}', defaulting to Summary. Error: {e}"); results_tabs.active = 0

        # if app_session_data['extracted_text']: # Make chat visible if text was extracted
        #     chat_interface.visible = True
        #     if not chat_feed.objects: chat_feed.append({"user": "System","avatar": "ℹ️","value": "Text context loaded. Ask questions!"})

        if app_session_data['extracted_text']:          
          chat_interface.visible = True
          if not chat_feed.objects: # Check if feed is empty
              logger.info("Adding initial message to chat feed.")
              # --- REPLACE THE chat_feed.append({...}) line WITH THIS: ---
              initial_message = pn.chat.ChatMessage(
                object="Text context loaded. Ask me questions about the processed document(s)!", # Content in 'object'
                user="System",
                avatar="ℹ️"
              )
              chat_feed.append(initial_message) # Append the ChatMessage object
          update_button_states()


        end_time = time.time(); status_text.value = f'PDF processing finished: {total_files} file(s) in {end_time - start_time:.2f}s!'; pn.state.notifications.success(f'{total_files} PDF(s) processed ({selected_mode})!', duration=4000); progress_bar.value = 100

    except Exception as e: # Main error handler
        logger.error(f"Error during PDF processing callback: {e}", exc_info=True)
        detailed_error = traceback.format_exc(); status_text.value = f'Error: {str(e)}'; pn.state.notifications.error(f'PDF Processing failed: {str(e)}. Check logs.', duration=6000)
        summary_output.object = {"error": str(e), "details": detailed_error, "processed_files_summary": all_results_summary}
    finally: # Cleanup and UI re-enable
        progress_bar.value = 0; progress_bar.active = False
        logger.info(f"Cleaning up {len(pdf_temp_file_paths)} temporary PDF input file(s)...")
        for path in pdf_temp_file_paths: # Cleanup Temp PDFs
            if path and os.path.exists(path):
                try: os.remove(path)
                except Exception as clean_e: logger.warning(f"Failed to clean temp PDF file {path}: {clean_e}")
        openai_key_input.disabled = False; anthropic_key_input.disabled = False; html_input.disabled = False; pdf_input.disabled = False; search_input.disabled = False
        update_button_states()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

async def search_callback(event):
    """Callback triggered when the Search Literature button is clicked."""
    keywords = search_input.value; openai_key = openai_key_input.value
    if not keywords or not openai_key: pn.state.notifications.error('Error: Search keywords and OpenAI Key required!', duration=4000); return

    # Disable UI
    process_html_button.disabled = True; process_pdf_button.disabled = True; search_button.disabled = True; html_input.disabled = True; pdf_input.disabled = True; search_input.disabled = True; openai_key_input.disabled = True; anthropic_key_input.disabled = True; send_button.disabled = True
    status_text.value = f'Searching literature for: "{keywords}"...'; progress_bar.active = True; progress_bar.value = 10
    search_results_output.object = "### Search Results\n---\n*Searching...*"
    # Activate search tab
    try: search_tab_index = [p.name for p in results_tabs.objects].index("Search Results"); results_tabs.active = search_tab_index
    except (ValueError, AttributeError): results_tabs.active = 0

    try:
        start_time = time.time();
        if not AUTOGEN_AVAILABLE: raise RuntimeError("Matagen module not available.")
        search_results = await asyncio.to_thread(journal_scraper_tool, keywords=keywords, api_key=openai_key)
        logger.info(f"Search finished. Result type: {type(search_results)}"); progress_bar.value = 90

        # Display Results
        formatted_results = f"### Search Results for: \"{keywords}\"\n---\n"
        if isinstance(search_results, list) and len(search_results) > 0 and isinstance(search_results[0], dict):
             formatted_results += f"Found {len(search_results)} potential results:\n\n"
             for i, paper in enumerate(search_results): # Formatting...
                 title = paper.get('title', 'No Title'); link = paper.get('link', '#'); snippet = paper.get('snippet', '')
                 formatted_results += f"{i+1}. **[{title}]({link})**\n"
                 if 'authors' in paper: formatted_results += f"   *Authors: {', '.join(paper['authors'])}*\n"
                 if 'journal' in paper: formatted_results += f"   *Journal: {paper['journal']}*\n"
                 formatted_results += f"   > {snippet}\n\n"
        elif isinstance(search_results, dict): formatted_results += f"```json\n{json.dumps(search_results, indent=2)}\n```"
        else: formatted_results += f"Received unexpected result format:\n```\n{search_results}\n```"

        search_results_output.object = formatted_results; end_time = time.time()
        status_text.value = f"Search complete in {end_time - start_time:.2f}s."; pn.state.notifications.success('Search complete!', duration=4000)

    except Exception as e: # Error Handling
        logger.error(f"Error during literature search: {e}", exc_info=True)
        detailed_error = traceback.format_exc(); status_text.value = f'Error: {str(e)}'; pn.state.notifications.error(f'Search failed: {str(e)}. Check logs.', duration=6000)
        search_results_output.object = f"### Search Results\n---\n**Error:**\n```\n{str(e)}\n{detailed_error}\n```"
    finally: # Re-enable UI
        progress_bar.active = False; progress_bar.value = 0
        openai_key_input.disabled = False; anthropic_key_input.disabled = False; html_input.disabled = False; pdf_input.disabled = False; search_input.disabled = False
        update_button_states()


async def chat_callback(event):
    """Callback triggered when the Send button in chat is clicked."""
    user_message = chat_input.value; openai_key = openai_key_input.value
    if not user_message or not openai_key: pn.state.notifications.warning("Enter message and OpenAI key.", duration=3000); return
    if not app_session_data["extracted_text"]: pn.state.notifications.warning("No text context available.", duration=4000); return

    chat_input.value = ''; send_button.disabled = True # Disable button via state check later? No, disable here.
    chat_spinner.value = True
    chat_feed.append({"user": "User", "value": user_message})

    try:
        # Prepare Context (Combine all extracted text)
        full_context = "\n\n--- End Doc / Start Doc ---\n\n".join(f"Document: {fname}\n\n{text}" for fname, text in app_session_data["extracted_text"].items())
        if not full_context: raise ValueError("Failed to retrieve text context.")
        logger.info(f"Sending chat query. Context length: {len(full_context)} chars.")

        # Call Backend Text Analysis Agent (ensure run_text_analysis_chat is imported or mocked)
        agent_response = await run_text_analysis_chat(user_query=user_message, context=full_context, api_key=openai_key)
        agent_chat_message =  pn.chat.ChatMessage(
            object=agent_response,
            user="Agent",
            avatar="🤖"
            )
        chat_feed.append(agent_chat_message)        
       
        # chat_feed.append({"user": "Agent", "avatar": "🤖", "value": agent_response})
        

    except Exception as e: # Error Handling
        logger.error(f"Error during chat processing: {e}", exc_info=True)
        error_message = f"Sorry, an error occurred: {e}"; chat_feed.append({"user": "System", "avatar": "⚠️", "value": error_message}); pn.state.notifications.error(f"Chat Error: {e}", duration=5000)
    finally: # Re-enable
        chat_spinner.value = False
        update_button_states() # Re-evaluates send_button disabled state


# --- Widget Linking ---
html_input.param.watch(update_button_states, 'value')
pdf_input.param.watch(update_button_states, 'value')
openai_key_input.param.watch(update_button_states, 'value')
anthropic_key_input.param.watch(update_button_states, 'value')
search_input.param.watch(update_button_states, 'value')
chat_input.param.watch(update_button_states, 'value')

# Button clicks
process_html_button.on_click(process_html_callback)
process_pdf_button.on_click(process_pdf_callback)
search_button.on_click(search_callback)
send_button.on_click(chat_callback)


# --- Arrange the Layout ---
# (Using Action Bar with Rows/Columns as per user's code)
actions_bar = pn.Card(pn.Row(html_action_col, pdf_action_col, search_action_col, sizing_mode='stretch_width'), title="Actions", sizing_mode='stretch_width')
main_content = pn.Column(actions_bar, status_bar, pn.layout.Divider(), results_tabs, pn.layout.Divider(margin=(20, 0)), chat_interface, sizing_mode='stretch_width')
app_layout = pn.Column(pn.pane.Markdown("# MMatAGen: Multimodal Materials science mining AGents"), pn.Row(sidebar, main_content, sizing_mode='stretch_width'), sizing_mode='stretch_width')

# --- Initial call & Servable ---
update_button_states()
app_layout.servable(title="Document Analysis App")