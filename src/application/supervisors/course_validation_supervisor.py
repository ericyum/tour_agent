from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

# Define the state for this graph
class CourseValidationState(TypedDict):
    course: List[Dict[str, Any]]
    duration: str
    validation_result: str

# Import the agent
from src.application.agents.course_validation.validation_agent import agent_validate_course

# Create the graph
course_validation_workflow = StateGraph(CourseValidationState)

# Add the nodes
course_validation_workflow.add_node("start_validation", agent_validate_course)

# Set the entry point
course_validation_workflow.set_entry_point("start_validation")

# Add the edge to the end
course_validation_workflow.add_edge('start_validation', END)

# Compile the graph
course_validation_graph = course_validation_workflow.compile()
