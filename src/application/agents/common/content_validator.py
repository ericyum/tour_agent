from src.application.core.state import LLMGraphState
from src.infrastructure.llm_client import get_llm_client

def agent_content_validator(state: LLMGraphState):
    if state["log_details"]:
        print(
            f"\n--- [Agent 0: Content Validator] 블로그 관련성 검증 시작: {state['title']} ---"
        )

    keyword = state["keyword"]
    title = state["title"]
    text = state["original_text"]
    llm = get_llm_client()

    try:
        prompt = f"""당신은 블로그 게시물의 주제를 정확하게 판별하는 전문가입니다. 사용자는 '{keyword}'에 대한 '진짜 후기'를 찾고 있습니다. 아래의 조건에 따라 주어진 블로그 제목과 본문이 검색 의도에 부합하는지 판별해주세요.

[판별 조건]
1. **주제 일치:** 게시물의 '주된 내용'이 '{keyword}'에 대한 경험이나 후기여야 합니다. 단순히 언급만 되거나 부수적인 내용이면 안 됩니다.
2. **유사 행사 제외:** '{keyword}'와 이름이 비슷한 다른 행사(예: '세계 {keyword}')에 대한 후기는 아닌지 확인해야 합니다.
3. **다른 주제 제외:** 게시물의 주된 내용이 '{keyword}'가 아닌, 특정 장소(카페, 식당), 제품, 서비스 등에 대한 비교나 추천이 아닌지 확인해야 합니다. (예: '{keyword} 기념 카페 A, B 비교 후기')

[판별할 정보]
- **제목:** {title}
- **본문 (일부):** {text[:2000]}...

[출력]
위 조건들을 모두 고려했을 때, 이 게시물이 사용자가 찾는 '{keyword}'에 대한 '진짜 후기'가 맞다면 '예'를, 그렇지 않다면 '아니오'를 반환해주세요. '예' 또는 '아니오'로만 대답해야 합니다."""

        response = llm.invoke(prompt)
        answer = response.content.strip()

        if "예" in answer:
            if state["log_details"]:
                print(f"   [검증 성공] 이 블로그는 '{keyword}'에 대한 관련글입니다.")
            return {"is_relevant": True}
        else:
            if state["log_details"]:
                print(f"   [검증 실패] 이 블로그는 '{keyword}'와 관련 없는 내용입니다.")
            return {"is_relevant": False}

    except Exception as e:
        print(f"LLM 내용 검증 중 오류 발생: {e}")
        return {"is_relevant": False}  # 오류 발생 시 관련 없는 것으로 처리
