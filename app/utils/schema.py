# Define the schema for SupervisorState
from pydantic import BaseModel
from typing import List, Annotated
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, END, MessagesState

class SupervisorState(MessagesState):
    """State for the multi-agent system"""
    messages: Annotated[List, add_messages]
    next_agent: str = ""
    Images_data: str = ""
    scientific_data: str = ""
    products_data: str = ""
    final_answer: str = ""
    task_complete: bool = False
    current_task: str = ""