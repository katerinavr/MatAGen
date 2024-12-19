from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
# from src.agents import *  # Ensure this includes user_proxy and assistant objects

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(html.H1("Interactive Chat Assistant", className="text-center my-4"))
        ),
        # Chat Display Section
        dbc.Row(
            dbc.Col(
                html.Div(
                    id="chat-window",
                    style={
                        "border": "1px solid #ddd",
                        "borderRadius": "10px",
                        "padding": "10px",
                        "height": "400px",
                        "overflowY": "scroll",
                        "backgroundColor": "#f9f9f9",
                    },
                ),
                width={"size": 8, "offset": 2},
            ),
            className="mb-4",
        ),
        # Chat Input Section
        dbc.Row(
            dbc.Col(
                [
                    dcc.Textarea(
                        id="chat-input",
                        placeholder="Type your message here...",
                        style={"width": "100%", "height": "80px"},
                    ),
                    dbc.Button("Send", id="send-btn", color="primary", className="mt-2"),
                ],
                width={"size": 8, "offset": 2},
            ),
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#f7f9fc", "minHeight": "100vh", "padding": "20px"},
)

# Global variable to store chat history
chat_history = []


# Callback for handling user input and assistant response
@app.callback(
    Output("chat-window", "children"),
    Input("send-btn", "n_clicks"),
    State("chat-input", "value"),
    prevent_initial_call=True,
)
def update_chat(n_clicks, user_message):
    global chat_history

    # Append user message to chat history
    if user_message:
        chat_history.append({"role": "user", "name": "You", "content": user_message})

    # Simulate assistant response (replace this with your actual assistant call)
    assistant_response = f"I received your message: {user_message}"
    chat_history.append({"role": "assistant", "name": "Assistant", "content": assistant_response})

    # Generate chat bubbles
    chat_bubbles = []
    for entry in chat_history:
        if entry["role"] == "user":
            bubble = html.Div(
                [
                    html.Div(entry["name"], style={"fontWeight": "bold", "color": "#007bff"}),
                    html.Div(entry["content"], style={"backgroundColor": "#e1f5fe", "borderRadius": "10px", "padding": "10px", "margin": "5px 0"}),
                ],
                style={"textAlign": "right"},
            )
        else:  # Assistant
            bubble = html.Div(
                [
                    html.Div(entry["name"], style={"fontWeight": "bold", "color": "#ff5722"}),
                    html.Div(entry["content"], style={"backgroundColor": "#ffe0b2", "borderRadius": "10px", "padding": "10px", "margin": "5px 0"}),
                ],
                style={"textAlign": "left"},
            )
        chat_bubbles.append(bubble)

    return chat_bubbles


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
