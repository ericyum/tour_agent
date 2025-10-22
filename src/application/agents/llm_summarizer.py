from src.domain.state import LLMGraphState
from src.domain.knowledge_base import knowledge_base
from src.infrastructure.llm_client import get_llm_client
import re
import ast

def agent_llm_summarizer(state: LLMGraphState):
    if state["log_details"]:
        print("\n--- [Agent 1: LLM Aspect Extractor] 주체-감성 쌍 추출 및 요약 시작 ---")

    text = state["original_text"]
    keyword = state["keyword"]
    feedback_message = state.get("feedback_message")
    re_summarize_count = state.get("re_summarize_count", 0)

    llm = get_llm_client()

    # LLM에게 요약과 '주체-감성' 쌍 추출을 동시에 요청하는 새 프롬프트
    base_prompt_template = '''아래는 '{keyword}'에 대한 블로그 리뷰 본문입니다.

[지시 1: 경험 요약]
본문 전체를 읽고, 글쓴이가 '{keyword}'와 관련하여 경험한 내용 전체에 대한 '긍정적인 점'과 '부정적인 점'을 각각 글머리 기호(bullet points) 형식으로 요약해주세요.
요약 시, 감성 표현(명사, 형용사, 부사, 관용어)과 그 감성을 수식하는 표현(강조어, 완화어, 부정어)을 모두 ****로 감싸주세요. 
예시: "음식이 ****너무**** ****별로****였어요.", "불꽃놀이는 ****황홀****했어요."

[지시 2: 주체-감성 쌍 추출]
본문 내용에서 '감성 표현'과 그 대상이 되는 '주체'를 찾아 아래와 같은 파이썬 리스트 형식으로 정리해주세요.
- 주체: 감성의 대상이 되는 구체적인 명사 또는 명사구 (예: 음식, 주차장, 불꽃놀이)
- 감성 표현: 주체에 대한 감정을 나타내는 핵심 단어 (예: 맛있다, 최악이다, 환상적이다)
- 형식: [("주체1", "감성표현1"), ("주체2", "감성표현2")]

[매우 중요: 최종 출력 형식]
두 섹션으로 나누어, 반드시 아래 형식을 그대로 지켜서 차례대로 출력해주세요. 다른 설명은 절대 추가하지 마세요.

--- 요약 ---
- 긍정적인 점:
  - (여기에 긍정적인 경험 요약)
- 부정적인 점:
  - (여기에 부정적인 경험 요약)

--- 주체-감성 쌍 ---
[("주체1", "감성표현1"), ("주체2", "감성표현2"), ...]


--- 블로그 본문 시작 ---
{text}
--- 블로그 본문 끝 ---'''

    user_prompt = base_prompt_template.format(keyword=keyword, text=text)

    if feedback_message and re_summarize_count < 3:  # 최대 3회 재요약 시도
        user_prompt += f"\n\n[이전 요약에 대한 피드백]\n{feedback_message}\n\n위 피드백을 바탕으로 요약과 주체-감성 쌍을 다시 생성해주세요."
        if state["log_details"]:
            print(f"   [LLM 재요청] 피드백 반영하여 재시도 (시도 횟수: {re_summarize_count + 1})")
    elif re_summarize_count >= 3:
        if state["log_details"]:
            print("   [LLM 재요청 실패] 최대 재요약 횟수 초과. 원본 결과 반환.")
        return {"llm_summary": state["llm_summary"], "aspect_sentiment_pairs": state.get("aspect_sentiment_pairs", []), "feedback_message": None}

    summary = ""
    aspect_pairs = []
    try:
        response = llm.invoke(user_prompt)
        raw_content = response.content.strip()

        # LLM 출력에서 요약과 주체-감성 쌍 부분을 분리
        summary_part_str = "--- 요약 ---"
        aspect_part_str = "--- 주체-감성 쌍 ---"

        if aspect_part_str in raw_content:
            summary_section = raw_content.split(aspect_part_str)[0]
            aspect_section = raw_content.split(aspect_part_str)[1]
            
            summary = summary_section.replace(summary_part_str, '').strip()
            
            # 주체-감성 쌍 문자열에서 리스트 부분만 추출
            match = re.search(r'[[].*[]]', aspect_section, re.DOTALL)
            if match:
                list_str = match.group(0)
                # 보안을 위해 ast.literal_eval을 사용하여 안전하게 파싱
                aspect_pairs = ast.literal_eval(list_str)
        else:
            # 형식을 지키지 않았을 경우, 전체를 요약으로 간주
            summary = raw_content

    except Exception as e:
        print(f"LLM 결과 파싱 또는 API 호출 중 오류 발생: {e}")
        # 오류 발생 시 빈 결과 반환
        summary = state.get("llm_summary", "") # 이전 요약이라도 유지
        aspect_pairs = state.get("aspect_sentiment_pairs", [])

    if state["log_details"]:
        print("--- LLM 핵심 경험 요약 결과 ---")
        print(summary if summary else "[요약할 내용이 없습니다]")
        print("---------------------------")
        print("--- LLM 주체-감성 쌍 추출 결과 ---")
        print(aspect_pairs if aspect_pairs else "[추출된 쌍이 없습니다]")
        print("--------------------------------")

    return {"llm_summary": summary, "aspect_sentiment_pairs": aspect_pairs, "feedback_message": None}
