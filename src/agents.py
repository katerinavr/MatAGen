import autogen
from autogen import UserProxyAgent
from autogen import AssistantAgent
import sys
from typing import Annotated
import os
# from config.settings import OPENAI_API_KEY
OPENAI_API_KEY = "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"
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_tools.paper_scraper_tool import html_scraper_tool
from custom_tools.multimodal_data_retriever import multimodal_data_retriever_tool
from autogen import register_function
from autogen import ConversableAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from pathlib import Path

# Configurations of the LLM models
config_list = [{'model': 'gpt-4o', 'api_key': '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'}]

gpt4o_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}
config_list_claude =  [{
        "model": "claude-3-5-sonnet-20240620",
        "api_key": "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",
        "api_type": "anthropic",
    }]

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=config_list[0]['api_key'],
                model_name="text-embedding-3-small"
            )
############################## Literature mining agents ##############################

## Multimodal data mining agents
# The user proxy agent is used for interacting with the assistant agent and executes tool calls.
admin = UserProxyAgent(
    name="admin",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    system_message="admin. You pose the task. Return 'TERMINATE' in the end when everything is over.",
    llm_config = gpt4o_config,
    code_execution_config=False
)

multimodal_data_assistant = AssistantAgent(
    name="multimodal_data_assistant",
    system_message="""You are a helpful AI assistant. 
    You can help with multimodal data retrieval from scientific papers.
    Return 'TERMINATE' when the task is done.
    """,
    llm_config= gpt4o_config,
    
)

admin.register_for_execution(name="HTML_Scraper")(html_scraper_tool)
admin.register_for_execution(name="image-text-retriever")(multimodal_data_retriever_tool)

multimodal_data_assistant.register_for_llm(name="HTML_Scraper", description="Calling the HTML Scraper tool to extract image and text from provided HTML files in a given directory")(html_scraper_tool)
multimodal_data_assistant.register_for_llm(name="image-text-retriever", description="Calling the multimodal data retriever tool to extract image and text from the json file created from the literature scrapping. The folder containing the data is the 'html-scraping'")(multimodal_data_retriever_tool)


## Image-related agents
# Image classification agent
multi_modal_agent = MultimodalConversableAgent(name="multi_modal_agent",
                  system_message='''multi_modal_agent.
                  You extract important information from a scientific image.
                  ''',
                  llm_config=gpt4o_config,
                  description='Classify the images.')

# Image quality agent

# Abs image segmentation agent



## Text-related agents
# assistant = RetrieveAssistantAgent(
#     name="assistant",
#     system_message="assistant. You are a helpful assistant. You retrieve knowledge from a text. You should pay attention to all the details, specially quantitative data.",
#     llm_config=gpt4o_config,
# )

# reviewer = RetrieveAssistantAgent(
#     name="reviewer",
#     system_message='''reviewer. double-check the response from the assistant for correctness. 
# Return 'TERMINATE' in the end when the task is over.''',
#     llm_config=gpt4o_config,
# )

# ragproxyagent = RetrieveUserProxyAgent(
#     human_input_mode="NEVER",
#     name="ragproxyagent",
#     retrieve_config={
#         "task": "qa",
#         "docs_path": "./code_dir/Mishin_Al_Ni.pdf",
#         "embedding_function": openai_ef,
#         "model": "gpt-4o",
#         "overwrite": True,
#         "get_or_create": True,
#     },
#     code_execution_config=False,
# )

# groupchat_rag = autogen.GroupChat(
#     agents=[assistant, reviewer, ragproxyagent, #sequence_retriever,
#                ], messages=[], max_round=20,
#     speaker_selection_method='auto',
# )
# manager_rag = autogen.GroupChatManager(groupchat=groupchat_rag, llm_config=gpt4o_config, 
#     system_message='You dynamically select a speaker.')
# # Topic modelling

# # NER

# # RAG


## General assistant agetns
from pathlib import Path
import pandas as pd

# Define Planner Agent
planner = AssistantAgent(
    name="planner",
    system_message='''You are the Planner Agent. 
Suggest a plan for the given task.

Do not write code.

Make sure your plan includes the necessary tools for each step.

Your plan will be reviewed by "critic."

Use only the tools required to accomplish the task, avoiding unnecessary computations and analyses.

Return "TERMINATE_PLAN" when the plan is approved. 
''',
    llm_config=gpt4o_config,
    description='You develop a plan.'
)

# Define Critic Agent
critic = AssistantAgent(
    name="critic",
    system_message='''You are the Critic Agent.
    
Review the planner's plan for completeness and accuracy.

Ensure the plan does not include unnecessary functions.

Return "TERMINATE_PLAN" when the plan is approved.

Do not execute any functions.''',
    llm_config=gpt4o_config,
    description='You review a plan from planner.'
)


admin_plan = autogen.UserProxyAgent(
    name="admin_plan",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_PLAN"),
    human_input_mode="NEVER",
    system_message="admin_plan. You pose the task.",
    code_execution_config=False,
    llm_config=False,
)

admin_core = autogen.UserProxyAgent(
    name="admin_core",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_ALL"),
    human_input_mode="ALWAYS",
    system_message="admin_core. You pose the task.",
    code_execution_config=False,
    llm_config=False,
)

groupchat_plan = autogen.GroupChat(
    agents=[planner, admin_plan, critic,#sequence_retriever,
               ], messages=[], max_round=200, send_introductions=True,
    #allowed_or_disallowed_speaker_transitions=allowed_transitions_1, speaker_transitions_type='allowed',
    speaker_selection_method='auto',
)

manager_plan = autogen.GroupChatManager(groupchat=groupchat_plan, llm_config = gpt4o_config, 
    system_message='You dynamically select a speaker based on the current and previous conversations.')

def _reset_agents_glob():
    planner.reset()
    critic.reset()

def _clear_history_glob():
    planner.clear_history()
    critic.clear_history()

_reset_agents_glob()
_clear_history_glob()

# Multimodal Agent
multi_modal_agent = MultimodalConversableAgent(
    name="multi_modal_agent",
    system_message='''multi_modal_agent.
    You extract important information from a plot.
    ''',
    llm_config=gpt4o_config,# {"config_list": gpt4o_config, "temperature": 0.0},
    description="Extract important information from the plots."
)

# Function for Classifying Figures
@admin_core.register_for_execution()
@planner.register_for_llm(description='''Use this function to assign tags to images.
This function processes records generated by the previous agent.''')
def classify_figures(records: Annotated[list, "List of dictionaries with 'image' and 'caption' keys"]) -> str:
    """
    Classifies figures based on image and caption data.

    Args:
        records: A list of dictionaries with keys 'image' and 'caption'.

    Returns:
        str: Path to the CSV file with classification results.
    """
    retrieved_images = []

    for record in records:
        image_path = record["image"]
        caption = record["caption"]

        try:
            # Pass the image and caption to the multimodal agent
            response = multi_modal_agent.converse(
                message=f"""
                Given the following prompt: "Classify this image based on {caption}."
                Is this image relevant to the task? Provide the classification or extracted information.
                """,
                images=[image_path]
            )

            # Store the result
            retrieved_images.append({"image": image_path, "response": response.text})
            print(f"Processed: {image_path}, Response: {response.text}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            retrieved_images.append({"image": image_path, "response": "Error"})

    # Save results to CSV
    output_csv = Path("multimodal_data_folder/retrieved_images.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    pd.DataFrame(retrieved_images).to_csv(output_csv, index=False)
    return f"Results saved to {output_csv}"


# Combine the agents 
groupchat = autogen.GroupChat(
    agents=[admin, admin_core, planner , multimodal_data_assistant], messages=[], max_round=50
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})