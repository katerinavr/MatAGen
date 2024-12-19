from langgraph.graph import StateGraph, START, END
from custom_tools.calculator_tool import calculator_tool
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_tools.paper_scraper_tools import paper_scraper_tool
from typing import TypedDict

# Define the shared state schema
class PipelineState(TypedDict):
    numbers: list
    folder: str
    api_key: str
    result: str

# Define each node as a standalone function
def calculator_node(state: PipelineState) -> PipelineState:
    state["result"] = calculator_tool(state["numbers"])
    return state

def paper_scraper_node(state: PipelineState) -> PipelineState:
    state["result"] = paper_scraper_tool(state["folder"], state["api_key"])
    return state

# Create the StateGraph without the `name` parameter
graph = StateGraph(PipelineState)

# Add nodes to the graph
graph.add_node("calculator", calculator_node)
graph.add_node("paper_scraper", paper_scraper_node)

# Define execution order
graph.add_edge(START, "calculator")
graph.add_edge("calculator", "paper_scraper")
graph.add_edge("paper_scraper", END)

# Compile the graph
compiled_graph = graph.compile()

# Execute the pipeline
def execute_pipeline():
    initial_state = PipelineState(numbers=[2, 3], folder="tmp_folder", api_key="your_api_key", result="")
    final_state = compiled_graph.invoke(initial_state)
    print("Pipeline Results:", final_state["result"])
 
# Run the pipeline
if __name__ == "__main__":
    execute_pipeline()
