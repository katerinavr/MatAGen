import autogen
from autogen import UserProxyAgent
from autogen import AssistantAgent
import sys
import os
# from config.settings import OPENAI_API_KEY
OPENAI_API_KEY = "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"
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_tools.paper_scraper_tool import html_scraper_tool

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

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4o_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config={"config_list": config_list},
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config={"config_list": config_list},
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)


critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config={"config_list": config_list},
)


# html_scraper = autogen.AssistantAgent(
#     name="HTML_Scraper",
#     system_message="Calling the HTML Scraper tool to extract image and text from provided HTML files in a given directory",
#     human_input_mode="NEVER",
#     tools = [html_scraper_tool],
#     code_execution_config={
#         "last_n_messages": 3,
#         "work_dir": "html_folder",
#         "use_docker": False,
#     },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
# )
# html_scraper.register_for_llm(name="html_scraper", description="A simple calculator")(html_scraper_tool)


from autogen import register_function
from autogen import ConversableAgent


image_segmentation_agent = autogen.AssistantAgent(
    "Absorption spectra segmentation agent",
    system_message="Extract all the points from a provided plot. Give me tuples of the wavelength and absrobance for each curve.",
    llm_config={
        "config_list": config_list_claude,
    },
)

# Data scrapping agents

assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with data retrieval. "
    "Return 'TERMINATE' when the task is done.",
    # llm_config={"config_list": config_list},
    llm_config={"config_list": [{"model": "gpt-4", "api_key": OPENAI_API_KEY}]},
)
assistant.register_for_llm(name="HTML_Scraper", description="Calling the HTML Scraper tool to extract image and text from provided HTML files in a given directory")(html_scraper_tool)

# The user proxy agent is used for interacting with the assistant agent
# and executes tool calls.
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="HTML_Scraper")(html_scraper_tool)


groupchat = autogen.GroupChat(
    agents=[assistant, user_proxy, planner, executor, critic], messages=[], max_round=50
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})