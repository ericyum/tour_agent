from langgraph.graph import StateGraph, END
from src.application.core.db_state import DBSearchState
from src.application.agents.db_search.db_search_agent import agent_festival_search
from src.application.agents.db_search.nearby_search_agent import agent_nearby_search

# This node is just a pass-through to kick off the graph
def start_node(state: DBSearchState) -> DBSearchState:
    return state

# Router function to decide which agent to call
def route_db_search(state: DBSearchState):
    if state.get("search_type") == "festival_search":
        return "festival_search_node"
    elif state.get("search_type") == "nearby_search":
        return "nearby_search_node"
    else:
        return "__end__"

# Create the graph
db_workflow = StateGraph(DBSearchState)

# Add the nodes
db_workflow.add_node("start", start_node)
db_workflow.add_node("festival_search_node", agent_festival_search)
db_workflow.add_node("nearby_search_node", agent_nearby_search)

# Set the entry point
db_workflow.set_entry_point("start")

# Add the conditional edges from the start node
db_workflow.add_conditional_edges(
    "start",
    route_db_search,
    {
        "festival_search_node": "festival_search_node",
        "nearby_search_node": "nearby_search_node",
        "__end__": END
    }
)

# Add edges from the agent nodes to the end
db_workflow.add_edge('festival_search_node', END)
db_workflow.add_edge('nearby_search_node', END)

# Compile the graph
db_search_graph = db_workflow.compile()