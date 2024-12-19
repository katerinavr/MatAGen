from langgraph.graph import StateGraph, START, END
from custom_tools.calculator_tool import calculator_tool
# from tools.external_tools.paper_scraper_tool import paper_scraper_tool
from typing import TypedDict
import functools
import operator
from typing import Sequence, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
 
# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str
 
# Add the research agent using the create_agent helper function
research_agent = create_agent(llm, "You are a web researcher.", [wikipedia_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
 
# Add the time agent using the create_agent helper function
currenttime_agent = create_agent(llm, "You can tell the current time at", [datetime_tool])
currenttime_node = functools.partial(agent_node, agent=currenttime_agent, name = "CurrentTime")
 
workflow = StateGraph(AgentState)
 
# Add a "chatbot" node. Nodes represent units of work. They are typically regular python functions.
workflow.add_node("Researcher", research_node)
workflow.add_node("CurrentTime", currenttime_node)
workflow.add_node("supervisor", supervisor_chain)
 
# We want our workers to ALWAYS "report back" to the supervisor when done
for member in members:
    workflow.add_edge(member, "supervisor")
 
# Conditional edges usually contain "if" statements to route to different nodes depending on the current graph state.
# These functions receive the current graph state and return a string or list of strings indicating which node(s) to call next.
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
 
# Add an entry point. This tells our graph where to start its work each time we run it.
workflow.add_edge(START, "supervisor")
 
# To be able to run our graph, call "compile()" on the graph builder. This creates a "CompiledGraph" we can use invoke on our state.
graph_2 = workflow.compile()

# # Define the shared state schema
# class PipelineState(TypedDict):
#     numbers: list
#     folder: str
#     api_key: str
#     result: str

# # Define each node as a standalone function
# def calculator_node(state: PipelineState) -> PipelineState:
#     state["result"] = calculator_tool(state["numbers"])
#     return state

# def paper_scraper_node(state: PipelineState) -> PipelineState:
#     # Make sure we are passing exactly two arguments
#     state["result"] = paper_scraper_tool(state["folder"], state["api_key"])
#     return state

# # Create the StateGraph
# graph = StateGraph(PipelineState, name="My Agentic Pipeline")
# graph.add_node("calculator", calculator_node)
# graph.add_node("paper_scraper", paper_scraper_node)

# # Define execution order
# graph.add_edge(START, "calculator")
# graph.add_edge("calculator", "paper_scraper")
# graph.add_edge("paper_scraper", END)

# # Compile the graph
# compiled_graph = graph.compile()

# # Execute the pipeline
# def execute_pipeline():
#     initial_state = PipelineState(numbers=[2, 3], folder="tmp_folder", api_key="your_api_key", result="")
#     final_state = compiled_graph.invoke(initial_state)
#     print("Pipeline Results:", final_state["result"])

# # Run the pipeline
# if __name__ == "__main__":
#     execute_pipeline()
