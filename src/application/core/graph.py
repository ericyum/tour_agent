from langgraph.graph import StateGraph, END
from src.domain.state import LLMGraphState
from src.application.agents.content_validator import agent_content_validator
from src.application.agents.llm_summarizer import agent_llm_summarizer
from src.application.agents.rule_scorer import agent_rule_scorer_on_summary

def route_after_validation(state: LLMGraphState):
    if not state.get("is_relevant"):
        return "__end__"
    return "llm_summarizer"

def route_after_scoring(state: LLMGraphState):
    if state.get("feedback_message") and state.get("re_summarize_count", 0) < 3:
        return "llm_summarizer"
    else:
        return "__end__"

def create_llm_workflow():
    llm_workflow = StateGraph(LLMGraphState)
    llm_workflow.add_node("content_validator", agent_content_validator)
    llm_workflow.add_node("llm_summarizer", agent_llm_summarizer)
    llm_workflow.add_node("rule_scorer", agent_rule_scorer_on_summary)

    llm_workflow.set_entry_point("content_validator")
    llm_workflow.add_conditional_edges(
        "content_validator",
        route_after_validation,
        {"llm_summarizer": "llm_summarizer", "__end__": END},
    )
    llm_workflow.add_edge("llm_summarizer", "rule_scorer")
    llm_workflow.add_conditional_edges(
        "rule_scorer",
        route_after_scoring,
        {"llm_summarizer": "llm_summarizer", "__end__": END},
    )

    return llm_workflow.compile()

# 워크플로우 그래프 인스턴스 생성
app_llm_graph = create_llm_workflow()
