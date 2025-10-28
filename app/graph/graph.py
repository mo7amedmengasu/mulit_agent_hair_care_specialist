import sys
import os

# Add the parent directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Create and compile the workflow graph
from langgraph.graph import StateGraph, END
from app.agents.agents import supervisor_agent, image_analysis_agent, scientific_data_agent, products_data_agent, final_answer_agent
from app.utils.schema import SupervisorState
from langchain_core.messages import AIMessage, HumanMessage

from typing import TypedDict, Annotated, List, Literal, Dict, Any

def router(state: SupervisorState) -> Literal["supervisor", "image_analysis_agent", "scientific_data_agent", "products_data_agent", "final_answer_agent", "__end__"]:
    """Routes to the next agent based on the state."""

    next_agent = state.get("next_agent", "supervisor")

    if next_agent == "end" or state.get("task_complete", False):
        return END

    if next_agent in [
        "supervisor",
        "image_analysis_agent",
        "scientific_data_agent",
        "products_data_agent",
        "final_answer_agent",
    ]:
        return next_agent

    return "supervisor"


def create_workflow_graph():
    # Create workflow
    workflow = StateGraph(SupervisorState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("image_analysis_agent", image_analysis_agent)
    workflow.add_node("scientific_data_agent", scientific_data_agent)
    workflow.add_node("products_data_agent", products_data_agent)
    workflow.add_node("final_answer_agent", final_answer_agent)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add conditional edges only from the supervisor
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "image_analysis_agent": "image_analysis_agent",
            "scientific_data_agent": "scientific_data_agent",
            "products_data_agent": "products_data_agent",
            "final_answer_agent": "final_answer_agent",
            END: END,
        }
    )

    # Each agent returns control to the supervisor
    for node in [
        "image_analysis_agent",
        "scientific_data_agent",
        "products_data_agent",
        "final_answer_agent",
    ]:
        workflow.add_edge(node, "supervisor")

    # Compile the graph
    graph = workflow.compile()
    return graph

workflow_graph = create_workflow_graph()
