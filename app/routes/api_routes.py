from fastapi import APIRouter, HTTPException
from app.graph.graph import workflow_graph  # Correct import for the workflow graph
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

router = APIRouter()

class TaskRequest(BaseModel):
    task: str

@router.post("/get-final-answer")
def get_final_answer(request: TaskRequest):
    """API endpoint to get the final answer from the workflow graph."""
    try:
        # Invoke the workflow graph with the provided task
        response = workflow_graph.invoke(HumanMessage(content=request.task))

        # Extract the final answer from the response
        final_answer = response.get("final_answer", "No final answer available.")

        return {"final_answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


