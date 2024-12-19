import sys
import os
import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import io 
import json
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents import *  # Ensure this includes user_proxy and assistant objects


# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Redirect stdout to capture process logs
class LogCapture(io.StringIO):
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.original_stdout


# Layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.H1("HTML Processor Assistant", className="text-center my-4"),
            )
        ),
        # Chat Input Section
        dbc.Row(
            dbc.Col(
                [
                    dbc.Label("Type your request for the assistant:", className="h5"),
                    dcc.Textarea(
                        id="chat-input",
                        placeholder="E.g., 'Extract the images and text from the html files in the html_folder.'",
                        style={"width": "100%", "height": "100px"},
                    ),
                    dbc.Button("Send", id="chat-send", color="primary", className="mt-2"),
                    html.Div(id="chat-status", className="mt-3"),
                ],
                width={"size": 6, "offset": 3},
            ),
            className="my-4",
        ),
        # Log Output Section
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Process Logs", className="card-title"),
                            html.Pre(id="process-logs", style={"whiteSpace": "pre-wrap"}),
                        ]
                    ),
                    className="shadow",
                ),
                width={"size": 10, "offset": 1}
            ),
            className="my-4",
        ),
        # Assistant Output Section
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Assistant's Output", className="card-title"),
                            html.Div(id="assistant-output", className="mt-4"),
                        ]
                    ),
                    className="shadow",
                ),
                width={"size": 10, "offset": 1}
            ),
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#f7f9fc", "minHeight": "100vh", "padding": "20px"},
)

@app.callback(
    [Output("chat-status", "children"), Output("process-logs", "children"), Output("assistant-output", "children")],
    Input("chat-send", "n_clicks"),
    State("chat-input", "value"),
    prevent_initial_call=True,
)
def handle_user_request(n_clicks, message):
    if not message:
        return dbc.Alert("Please type a message for the assistant.", color="warning"), "", ""

    try:
        # Call the assistant function
        print("\n>>> Calling user_proxy.initiate_chat...")
        chat_result = user_proxy.initiate_chat(assistant, message=message)

        # Print details to the terminal for debugging
        print("\n>>> chat_result Type:", type(chat_result))
        print("\n>>> chat_result dir:", dir(chat_result))
        print("\n>>> chat_result repr:", repr(chat_result))

        # Extract chat history from ChatResult object
        if hasattr(chat_result, "chat_history"):
            chat_history = chat_result.chat_history
        else:
            raise TypeError("ChatResult object does not have a 'chat_history' attribute.")

        # Log raw chat history for debugging
        logs_output = f"Assistant Chat History:\n{chat_history}\n"

        # Format chat history for display
        formatted_output = []
        for entry in chat_history:
            role = entry.get("role", "unknown").capitalize()
            name = entry.get("name", "Unknown")
            content = entry.get("content", "No content provided.")

            formatted_output.append(
                html.Div(
                    [
                        html.H5(f"{role} ({name}):"),
                        html.P(content, style={"whiteSpace": "pre-wrap"}),
                    ],
                    className="my-4",
                )
            )

        return (
            dbc.Alert("Request processed successfully!", color="success"),
            logs_output,
            html.Div(formatted_output),
        )

    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error occurred during processing: {str(e)}"
        print("\n>>> Error:", error_message)
        return (
            dbc.Alert(f"Error: {str(e)}", color="danger"),
            f"Error Details:\n{error_message}",
            "",
        )

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)