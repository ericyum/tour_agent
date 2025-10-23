from src.domain.state import LLMGraphState
from src.infrastructure.dynamic_scorer import SimpleScorer

def agent_rule_scorer_on_summary(state: LLMGraphState):
    if state["log_details"]:
        print("\n--- [Agent 2: Rule Scorer] 요약 기반 점수 계산 시작 ---")

    summary = state["llm_summary"]
    scorer = SimpleScorer()
    sentences = [s for s in summary.split("\n") if s.strip()]

    final_judgments = []
    is_positive_context = False
    is_negative_context = False

    def is_inconsistent(is_pos, is_neg, score):
        return (is_pos and score < 0.0) or (is_neg and score > 0.0)

    for sentence in sentences:
        if sentence.strip() == "- 긍정적인 점:":
            is_positive_context = True
            is_negative_context = False
            if state["log_details"]:
                print(f"   [필터링] 헤더 문장 제외: {sentence}")
            continue
        elif sentence.strip() == "- 부정적인 점:" or sentence.strip().startswith(
            "- 본문에 언급된 부정적인 점은"
        ):
            is_positive_context = False
            is_negative_context = True
            if state["log_details"]:
                print(f"   [필터링] 헤더 문장 제외: {sentence}")
            continue

        score = scorer.score_sentence(
            sentence,
            is_positive_context=is_positive_context,
            is_negative_context=is_negative_context,
        )

        if is_inconsistent(is_positive_context, is_negative_context, score):
            if state["log_details"]:
                print(
                    f"   [불일치 감지] 1차: {'긍정' if is_positive_context else '부정'} 문맥의 문장이 {score:.2f} 점수. 재계산 시도."
                )

            recalculated_score = scorer.score_sentence(
                sentence,
                is_positive_context=is_positive_context,
                is_negative_context=is_negative_context,
            )

            if is_inconsistent(
                is_positive_context, is_negative_context, recalculated_score
            ):
                feedback_msg = (
                    f"LLM 요약 내용 중 감성 점수 불일치 발생: "
                    f"'{sentence}' 문장은 {'긍정' if is_positive_context else '부정'} 문맥에 있지만, "
                    f"점수는 {recalculated_score:.2f}로 잘못 계산되었습니다. "
                    f"해당 문장의 감성을 다시 평가하고, 감성 표현을 정확히 마킹하여 요약해주세요."
                )
                if state["log_details"]:
                    print(
                        f"   [불일치 지속] 2차: {'긍정' if is_positive_context else '부정'} 문맥의 문장이 {recalculated_score:.2f} 점수. LLM 재요약 요청."
                    )
                return {
                    "feedback_message": feedback_msg,
                    "re_summarize_count": state.get("re_summarize_count", 0) + 1,
                }
            else:
                score = recalculated_score
                if state["log_details"]:
                    print(
                        f"   [일관성 확보] 재계산 후: {'긍정' if is_positive_context else '부정'} 문맥의 문장이 {score:.2f} 점수."
                    )

        verdict = "중립"
        if score > 0.1:
            verdict = "긍정"
        elif score < -0.1:
            verdict = "부정"

        final_judgments.append(
            {"sentence": sentence, "final_verdict": verdict, "score": score}
        )
        if state["log_details"]:
            print(f"   [{verdict}] 점수: {score:.2f} | 문장: {sentence}")

    return {"final_judgments": final_judgments, "feedback_message": None}
