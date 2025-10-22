from typing import TypedDict, List, Dict

class LLMGraphState(TypedDict):
    original_text: str
    keyword: str
    title: str  # 블로그 제목 추가
    log_details: bool
    is_relevant: bool  # 관련성 여부 플래그
    llm_summary: str
    final_judgments: List[Dict]
    feedback_message: str | None
    re_summarize_count: int
    aspect_sentiment_pairs: List[tuple]  # (주체, 감성표현) 쌍을 저장
