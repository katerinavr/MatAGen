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
from matagen.scraping import xml_scraper, pdf_scraper, journal_scraper

try:
    from matagen.agents.multimodal_system import ModelConfig, MultimodalAnalysisSystem
    AUTOGEN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agents module: {e}. AutoGen processing will be mocked.")
    AUTOGEN_AVAILABLE = False
    # --- Mock classes if import fails ---
    class MockModelConfig:
        def __init__(self, *args, **kwargs):
             print(f"MockModelConfig Initialized with keys: OpenAI={kwargs.get('openai_api_key')[:5]}..., Anthropic={kwargs.get('anthropic_api_key')[:5]}...")
        def get_gpt4_config(self): return {}
    class MockMultimodalAnalysisSystem:
        def __init__(self, config): self.config = config
        async def initiate_chat(self, task):
            print(f"--- MOCK AutoGen Start ---")
            print(f"Task: {task}")
            await asyncio.sleep(5) # Simulate work
            output_json_path = "html_scraping/retrieved_image_caption_pairs.json"
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            mock_data = { "records": [ ], "processing_status": "Mock Success" }
            with open(output_json_path, 'w', encoding='utf-8') as f: json.dump(mock_data, f, indent=4)
            print(f"--- MOCK AutoGen End (Output: {output_json_path}) ---")
            return {"status": "mock_success"}
    ModelConfig = MockModelConfig
    MultimodalAnalysisSystem = MockMultimodalAnalysisSystem


# --- Panel App Code ---
pn.extension('tabulator', 'notifications', 'jsoneditor')

# --- Constants ---
OUTPUT_JSON_PATH = "DataMiningAgents/html_scraping/retrieved_image_caption_pairs.json" # Adjusted path? Make sure it's correct relative to where you run panel serve
HTML_TEMP_DIR = "temp_html_uploads" # Directory to store temp HTML files

# Create temporary directories if they don't exist
# Use paths relative to the script file for better portability
script_dir = os.path.dirname(__file__)
output_json_full_path = os.path.join(script_dir, OUTPUT_JSON_PATH)
html_temp_full_path = os.path.join(script_dir, HTML_TEMP_DIR)
os.makedirs(os.path.dirname(output_json_full_path), exist_ok=True)
os.makedirs(html_temp_full_path, exist_ok=True)


# --- UI Components ---
# == Sidebar Components ==
openai_key_input = pn.widgets.PasswordInput(
    name="OpenAI API Key", placeholder="sk-...", width_policy='min'
)
anthropic_key_input = pn.widgets.PasswordInput(
    name="Anthropic API Key", placeholder="sk-ant-...", width_policy='min'
)

sidebar = pn.Column(
    pn.pane.Markdown("### API Configuration"),
    openai_key_input,
    anthropic_key_input,
    width=320, # Fixed width for the sidebar
    height_policy='max',
    styles={"border-right": "1px solid #ddd", "padding-right": "15px"} # Visual separation
)

# == Main Area Components ==
# Action Area
pdf_input = pn.widgets.FileInput(accept='.pdf', name='Upload PDF', width_policy='max')
process_pdf_button = pn.widgets.Button(name='Process PDF', button_type='primary', disabled=True)

html_input = pn.widgets.FileInput(accept='.html, .htm', name='Upload HTML', width_policy='max')
# Button starts disabled, enabled by update_process_button_state
process_html_button = pn.widgets.Button(name='Process HTML', button_type='primary', disabled=True)

search_input = pn.widgets.TextInput(name='Search Keywords', placeholder='Enter keywords...', width_policy='max')
search_button = pn.widgets.Button(name='Search Literature', button_type='primary', disabled=True)

# Status indicators
status_text = pn.widgets.StaticText(value='Please enter API Keys and upload an HTML file.')
progress_bar = pn.widgets.Progress(name='Progress', value=0, max=100, active=False, bar_color='info', width=200)

# Results Area (Tabs)
summary_output = pn.pane.JSON({}, name='JSON Summary', depth=-1, theme='light', height=600, sizing_mode='stretch_width')
text_output = pn.pane.Markdown("### Extracted Text (PDF)\n---", height=400, sizing_mode='stretch_width')
image_output = pn.GridBox(ncols=4, sizing_mode='stretch_width')
table_output = pn.Column(sizing_mode='stretch_width')
search_results_output = pn.pane.Markdown("### Search Results\n---", height=400, sizing_mode='stretch_width')

results_tabs = pn.Tabs(
    ("Summary", summary_output),
    ("PDF Text", text_output),
    ("PDF Images", pn.Column(pn.pane.Markdown("### Extracted Images\n---"), image_output, height=400, sizing_mode='stretch_width')),
    ("PDF Tables", pn.Column(pn.pane.Markdown("### Extracted Tables\n---"), table_output, height=400, sizing_mode='stretch_width')),
    ("Search Results", search_results_output),
    sizing_mode='stretch_width',
    dynamic=False
)

# --- Callback Functions ---
async def process_html_callback(event):
    """Callback triggered when the Process HTML button is clicked."""
    # --- Check Inputs ---
    html_file_value = html_input.value
    openai_key = openai_key_input.value
    anthropic_key = anthropic_key_input.value

    if html_file_value is None:
        pn.state.notifications.error('Error: No HTML file uploaded!', duration=4000)
        return
    if not openai_key or not anthropic_key:
         pn.state.notifications.error('Error: API Key(s) missing!', duration=4000)
         return

    # --- Disable UI ---
    process_html_button.disabled = True
    html_input.disabled = True
    pdf_input.disabled = True
    process_pdf_button.disabled = True
    search_input.disabled = True
    search_button.disabled = True
    openai_key_input.disabled = True # Disable key input during processing
    anthropic_key_input.disabled = True
    status_text.value = 'Processing HTML started...'
    progress_bar.active = True
    progress_bar.value = 5
    summary_output.object = {}
    results_tabs.active = 0

    temp_html_file_path = None

    try:
        html_bytes = html_file_value
        original_filename = html_input.filename if html_input.filename else "uploaded.html"

        # 1. Save uploaded HTML temporarily
        safe_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '_')).rstrip()
        # Use the full path to the temp directory
        temp_html_file_path = os.path.join(html_temp_full_path, f"{int(time.time())}_{safe_filename}")

        with open(temp_html_file_path, 'wb') as temp_html:
            temp_html.write(html_bytes)

        status_text.value = f'Saved temporary HTML: {os.path.basename(temp_html_file_path)}'
        progress_bar.value = 10
        logger.info(f"Temporary HTML file saved at: {temp_html_file_path}")

        # 2. Prepare AutoGen Task (using the correct OUTPUT_JSON_PATH full path)
        task = f"""
        Process the HTML file located at '{temp_html_file_path}'.
        First, use the 'Paper_Scraper' tool to extract image-caption pairs from this HTML file, saving the results to '{output_json_full_path}'. Ensure the directory exists.
        Then, use the 'multimodal_image_agent' tool to process the dataset within '{output_json_full_path}', classifying images and extracting metadata as defined in its capabilities. The results should be updated in the same file ('{output_json_full_path}').
        Report successful completion or any critical errors encountered during the process. Respond with TERMINATE when all steps are finished.
        """
        progress_bar.value = 20
        status_text.value = 'Initiating AutoGen processing...'
        logger.info("Initiating AutoGen chat...")
        logger.debug(f"AutoGen Task: {task}")

        # 3. Instantiate & Run AutoGen using UI keys
        if not AUTOGEN_AVAILABLE:
             raise RuntimeError("AutoGen module not available. Cannot perform processing.")

        # Use keys from UI widgets
        config = ModelConfig(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key
        )
        system = MultimodalAnalysisSystem(config) # Instantiate with UI keys

        # Run the chat asynchronously
        chat_result = await system.initiate_chat(task)
        logger.info(f"AutoGen chat finished. Result: {chat_result}")

        progress_bar.value = 80
        status_text.value = 'AutoGen process complete. Reading summary...'

        # 4. Read JSON result (using the full path)
        if not os.path.exists(output_json_full_path):
             logger.error(f"Output JSON file not found at {output_json_full_path} after AutoGen process.")
             raise FileNotFoundError(f"AutoGen process did not create the expected output file: {output_json_full_path}")

        await asyncio.sleep(0.5)
        with open(output_json_full_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        logger.info(f"Successfully read output JSON from {output_json_full_path}")

        # 5. Update UI
        summary_output.object = summary_data
        results_tabs.active = 0

        status_text.value = f'HTML processing finished successfully for {original_filename}!'
        pn.state.notifications.success('HTML processed!', duration=4000)

    except FileNotFoundError as fnf_e:
        logger.error(f"FileNotFoundError: {fnf_e}", exc_info=True)
        status_text.value = f'Error: Output file not found. {fnf_e}'
        pn.state.notifications.error(f'Processing Error: Output file not found. Check logs.', duration=6000)
    except Exception as e:
        logger.error(f"Error during HTML processing: {e}", exc_info=True)
        detailed_error = traceback.format_exc()
        status_text.value = f'Error during HTML processing: {str(e)}'
        pn.state.notifications.error(f'HTML Processing failed: {str(e)}. Check console logs.', duration=6000)
        summary_output.object = {"error": str(e), "details": detailed_error}
    finally:
        # 6. Cleanup
        if temp_html_file_path and os.path.exists(temp_html_file_path):
            try:
                os.remove(temp_html_file_path)
                logger.info(f"Cleaned up temporary file: {temp_html_file_path}")
            except Exception as clean_e:
                 logger.warning(f"Could not remove temporary file {temp_html_file_path}: {clean_e}")

        # Re-enable UI elements
        process_html_button.disabled = False # Re-enable, state will be updated by watcher
        html_input.disabled = False
        pdf_input.disabled = False
        process_pdf_button.disabled = True # Keep disabled
        search_input.disabled = False
        search_button.disabled = True # Keep disabled
        openai_key_input.disabled = False
        anthropic_key_input.disabled = False
        progress_bar.active = False
        progress_bar.value = 0
        # Call watcher function to correctly set button disabled state after run
        update_process_button_state()


# --- Placeholder Callbacks for other buttons ---
async def process_pdf_callback(event):
    pn.state.notifications.warning("PDF processing not implemented yet.", duration=3000)

async def search_callback(event):
    pn.state.notifications.warning("Literature search not implemented yet.", duration=3000)

# --- Widget Linking ---
# Function to update button state based on multiple inputs
def update_process_button_state(*events):
    """Enable Process HTML button only if keys and file are provided."""
    # Check if processing is already running (button is disabled)
    # This prevents re-enabling the button immediately after clicking if inputs change fast
    if progress_bar.active:
         return

    keys_present = bool(openai_key_input.value and anthropic_key_input.value)
    file_present = html_input.value is not None

    process_html_button.disabled = not (keys_present and file_present)

    # Update status text based on what's missing
    if not keys_present and not file_present:
        status_text.value = "Ready. Please enter API Keys and upload an HTML file."
    elif not keys_present:
        status_text.value = f"API Keys missing. File selected: {html_input.filename}"
    elif not file_present:
        status_text.value = "Ready. API Keys entered. Please upload an HTML file."
    else:
        status_text.value = f"Ready to process HTML: {html_input.filename}"


# Watch multiple parameters
html_input.param.watch(update_process_button_state, 'value')
openai_key_input.param.watch(update_process_button_state, 'value')
anthropic_key_input.param.watch(update_process_button_state, 'value')

# Button clicks
process_html_button.on_click(process_html_callback)
process_pdf_button.on_click(process_pdf_callback)
search_button.on_click(search_callback)

# --- Logging Setup ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Arrange the Layout ---
actions_bar = pn.Card(
    pn.Row(
        pn.Column(html_input, process_html_button, sizing_mode='stretch_width'),
        pn.Column(pdf_input, process_pdf_button, sizing_mode='stretch_width'),
        pn.Column(search_input, search_button, sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    ),
    title="Actions",
    sizing_mode='stretch_width'
)

status_bar = pn.Row(status_text, progress_bar)

# Group main area components
main_content = pn.Column(
    actions_bar,
    status_bar,
    pn.layout.Divider(),
    results_tabs,
    sizing_mode='stretch_width' # Main area stretches
)

# Final Layout combining sidebar and main content
app_layout = pn.Column(
    pn.pane.Markdown("# MMatAGen: Multimodal Materials science mining AGents"),
    pn.Row(
        sidebar,
        main_content,
        sizing_mode='stretch_width' # Make the row stretch
    ),
    sizing_mode='stretch_width' # Make the outer column stretch
)


# Initial call to set button state correctly on load
update_process_button_state()

# To run: panel serve app.py --autoreload --show
app_layout.servable(title="Document Analysis App")