from langchain_openai import OpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
# # from tools import html_scraper_tool, calculator_tool
import sys

import os

# Ensure the project directory is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pneumatic.paper_scraper_tool import paper_scraper_tool
from pneumatic.calculator_tool import calculator_tool
from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import BaseTool, StructuredTool
import langchain
from langchain.agents import initialize_agent
from llama_index.core import ListIndex
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType


from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool


# class MultiToolAgent:
#     def __init__(self, tools, llm_name, api_key,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,**kwargs):
#         self.llm = OpenAI(api_key=api_key, model=llm_name)
#         self.api_key = api_key


#         memory = ConversationBufferMemory(memory_key="chat_history")
#         self.agent_chain = initialize_agent(
#             tools, self.llm, agent=agent_type, verbose=True, memory=memory, **kwargs
#         )

#     def run(self, prompt):
#         with get_openai_callback() as cb:
#             result = self.agent_chain.run(input=prompt)
#         if self.get_cost:
#             print(cb)
#         return result
    

calculator = StructuredTool.from_function(
    func=calculator_tool,
    name="Calculator",
    description="multiply numbers",
    # args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)
folder = "tmp_folder"

# paper_scraper_tool = StructuredTool.from_function(
#     name="paper_scraper",
#     func=lambda folder: paper_scraper_tool(folder, api_key),
#     description="Scrape the folder with papers and return the images and their descriptions.",
#     return_direct=True
# )

tools = [calculator]

