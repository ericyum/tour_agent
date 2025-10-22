import os
import re
from konlpy.tag import Okt
from src.domain.knowledge_base import knowledge_base
from src.infrastructure.llm_client import get_llm_client

okt = Okt()

class SimpleScorer:
    def __init__(self):
        self.kb = knowledge_base
        self.okt = okt
        # LLM 클라이언트를 필요할 때 생성하도록 변경
        self.llm = None

    def _initialize_llm(self):
        if self.llm is None:
            self.llm = get_llm_client()

    def get_dynamic_score(
        self,
        sentence_to_score: str,
        expected_tag: str = None,
        is_positive_context: bool = False,
        is_negative_context: bool = False,
    ) -> float:
        self._initialize_llm()
        if not self.llm:
            return 0.0

        try:
            tag_guidance = ""
            if expected_tag:
                tag_guidance = f"\n[참고] 이 표현은 형태소 분석 결과 '{expected_tag}' 품사로 분류되었습니다. 이 정보를 바탕으로 가장 적절한 카테고리 번호를 선택해주세요."

            context_guidance = "중립"
            if is_positive_context:
                context_guidance = "긍정"
            elif is_negative_context:
                context_guidance = "부정"

            prompt = f"""
            당신은 한국어 신조어, 관용어, 그리고 감성적인 형용사/부사/명사에 능숙한 감성 분석 전문가입니다. 새로운 구절의 감성 점수를 추론해야 합니다.

            [현재 감성 사전의 예시 및 점수 기준]
            - 점수 범위: -2.0 (매우 부정) ~ 2.0 (매우 긍정) 사이의 실수 값으로 추론해주세요.
            - 긍정적인 단어일수록 높은 양수 값, 부정적인 단어일수록 낮은 음수 값, 중립적인 단어는 0에 가까운 값을 부여해주세요.
            - 강도가 강한 감성 표현일수록 절대값이 큰 점수를 부여해주세요.

            1. 긍정/부정 관용어 (점수): {list(self.kb.idioms.items())[:5]}...
            2. 강조 부사 (점수 배율): {list(self.kb.amplifiers.items())[:5]}...
            3. 완화 부사 (점수 배율): {list(self.kb.downtoners.items())[:5]}...
            4. 부정어: {self.kb.negators[:5]}...
            5. 감성 형용사 (점수): {list(self.kb.adjectives.items())[:5]}...
            6. 감성 부사 (점수): {list(self.kb.adverbs.items())[:5]}...
            7. 감성 명사 (점수): {list(self.kb.sentiment_nouns.items())[:5]}...

            분석할 문장: "{sentence_to_score}"

            [문맥 정보]
            이 표현은 전반적으로 '{context_guidance}'적인 문맥에서 나타났습니다. 이 문맥을 반드시 고려하여 점수를 추론해주세요.

            [지시사항]
            1. 위 문장에서, 기존 사전에 없는 새로운 '핵심 감성 표현' 구절(phrase)을 딱 하나만 찾아주세요.
            2. '핵심 감성 표현'은 다른 문장에서도 재사용될 수 있는 일반적인 관용어, 신조어, 또는 감성적인 형용사/부사/명사여야 합니다.
            3. 문장 전체가 아니라, 그 안의 핵심적인 구절만 정확히 추출해야 합니다.
            4. 만약 문장에 재사용 가능한 특별한 감성 표현이 없다면, '없음'이라고 반환해야 합니다.
            5. 추출한 '핵심 감성 표현'이 1~7 중 어떤 카테고리에 속하는지 결정해주세요.
            6. **카테고리 2(강조어) 또는 3(완화어)으로 결정했다면, '점수' 부분에 긍정적인 '점수 배율'을 반환해주세요. (예: 강조어는 1.5, 완화어는 0.5)**
            7. **나머지 카테고리(1, 5, 6, 7)로 결정했다면, 문맥에서 가지는 '최종적인 감성 점수'(-2.0 ~ 2.0)를 '점수' 부분에 반환해주세요.**
            8. **[매우 중요] '{context_guidance}' 문맥에 따라 점수의 부호(+/-)가 결정되어야 합니다. 긍정적 문맥에서는 반드시 양수 점수를, 부정적 문맥에서는 반드시 음수 점수를 부여해야 합니다.**
            9. 만약 추출한 표현이 없더라도, 문장 자체에 긍정 또는 부정 뉘앙스가 있다면, 문맥에 맞는 약간의 긍정/부정 값(예: 0.3 또는 -0.3)을 부여해야 합니다.
            {tag_guidance}

            [답변 형식]
            '카테고리 번호,핵심 감성 표현,점수' 형식으로만 반환해주세요.
            - 예시 1 (긍정 문맥의 새로운 명사): 7,꽉찬,1.2
            - 예시 2 (부정 문맥의 새로운 명사): 7,꽉찬,-1.1
            - 예시 3 (새로운 강조어): 2,훨씬,1.4
            - 예시 4 (감성적이나 특별한 표현 없음): 0,없음,0.3
            """

            response = self.llm.invoke(prompt)
            result = response.content.strip()

            parts = result.split(",")
            if len(parts) != 3:
                return 0.0

            category, phrase, score_str = (
                parts[0].strip(),
                parts[1].strip(),
                parts[2].strip(),
            )
            score = float(score_str)

            # 학습 결과를 파일에 저장하는 로직
            self._update_dictionary(category, phrase, score)
            
            # 점수 반환 로직
            if category in ["1", "5", "6", "7"] and phrase != "없음":
                return score
            elif category == "0" and phrase == "없음" and score != 0.0:
                return score
            else: # 강조어, 완화어 등은 점수 기여분이 0
                return 0.0

        except Exception as e:
            print(f"LLM 점수 추론 중 오류 발생: {e}")
            return 0.0

    def _update_dictionary(self, category: str, phrase: str, score: float):
        if phrase == "없음":
            return

        category_map = {
            "1": ("idioms.csv", "관용어"),
            "2": ("amplifiers.csv", "강조어"),
            "3": ("downtoners.csv", "완화어"),
            "5": ("adjectives.csv", "감성 형용사"),
            "6": ("adverbs.csv", "감성 부사"),
            "7": ("sentiment_nouns.csv", "감성 명사"),
        }

        if category not in category_map:
            return

        file_name, term = category_map[category]
        score_col_name = "multiplier" if category in ["2", "3"] else "score"
        print(f"[학습] 새로운 {term} 발견: {phrase} (값: {score}) -> {file_name}에 추가")

        with open(
            os.path.join(self.kb.dic_path, file_name),
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            f.write(f"\n{phrase},{score}")
        
        # 메모리 상의 사전도 업데이트
        dict_attr = file_name.split('.')[0]
        if hasattr(self.kb, dict_attr):
            dictionary = getattr(self.kb, dict_attr)
            if phrase not in dictionary:
                dictionary[phrase] = []
            dictionary[phrase].append(score)

    def score_sentence(
        self,
        sentence: str,
        is_positive_context: bool = False,
        is_negative_context: bool = False,
    ):
        final_score = 0.0

        if is_positive_context:
            final_score = 0.3
        elif is_negative_context:
            final_score = -0.3

        marked_phrases_with_modifiers = re.findall(
            r"\*\*\*\*([^*]+?)\*\*\*\*(?:\(수식어구:\s*([^)]+?)\))?", sentence
        )

        positive_contribution = 0.0
        negative_contribution = 0.0
        modified_word_scores = {}

        def get_contextual_score(scores: list, is_pos, is_neg):
            if is_pos:
                # 긍정 점수 중 가장 큰 값을 선택 (여러 개일 경우)
                pos_scores = [s for s in scores if s > 0]
                if pos_scores: return max(pos_scores), True
            elif is_neg:
                # 부정 점수 중 가장 작은 값을 선택
                neg_scores = [s for s in scores if s < 0]
                if neg_scores: return min(neg_scores), True
            return None, False

        for phrase, modifier_target in marked_phrases_with_modifiers:
            phrase = phrase.strip()
            modifier_target = modifier_target.strip() if modifier_target else None

            if (
                phrase in self.kb.amplifiers
                or phrase in self.kb.downtoners
                or phrase in self.kb.negators
            ):
                continue

            current_phrase_score = 0.0
            if phrase in self.kb.idioms:
                scores = self.kb.idioms[phrase]
                score, found = get_contextual_score(
                    scores, is_positive_context, is_negative_context
                )
                if found:
                    current_phrase_score = score
                else:
                    current_phrase_score = self.get_dynamic_score(
                        phrase, "Idiom", is_positive_context, is_negative_context
                    )
            else:
                words_in_phrase = self.okt.pos(phrase, norm=True, stem=True)
                for word, tag in words_in_phrase:
                    word_score = 0.0
                    known_word = False
                    scores = []
                    if tag.startswith("Adjective") and word in self.kb.adjectives:
                        scores = self.kb.adjectives[word]
                        known_word = True
                    elif tag.startswith("Adverb") and word in self.kb.adverbs:
                        scores = self.kb.adverbs[word]
                        known_word = True
                    elif tag.startswith("Noun") and word in self.kb.sentiment_nouns:
                        scores = self.kb.sentiment_nouns[word]
                        known_word = True

                    if known_word:
                        score, found = get_contextual_score(
                            scores, is_positive_context, is_negative_context
                        )
                        if found:
                            word_score = score
                        else:
                            word_score = self.get_dynamic_score(
                                word, tag, is_positive_context, is_negative_context
                            )
                    elif not self.kb.is_known_word(word) and (tag.startswith("Adjective") or tag.startswith("Adverb") or tag.startswith("Noun")):
                        word_score = self.get_dynamic_score(
                            word, tag, is_positive_context, is_negative_context
                        )
                    current_phrase_score += word_score

            if modifier_target:
                modified_word_scores[modifier_target] = current_phrase_score
            else:
                if current_phrase_score > 0:
                    positive_contribution += current_phrase_score
                elif current_phrase_score < 0:
                    negative_contribution += current_phrase_score

        for phrase, modifier_target in marked_phrases_with_modifiers:
            phrase = phrase.strip()
            modifier_target = modifier_target.strip() if modifier_target else None

            def get_multiplier(dictionary, p):
                if p in dictionary:
                    return dictionary[p][0]
                return 1.0

            if phrase in self.kb.amplifiers:
                multiplier = get_multiplier(self.kb.amplifiers, phrase)
                if modifier_target and modifier_target in modified_word_scores:
                    modified_word_scores[modifier_target] *= multiplier
            elif phrase in self.kb.downtoners:
                multiplier = get_multiplier(self.kb.downtoners, phrase)
                if modifier_target and modifier_target in modified_word_scores:
                    modified_word_scores[modifier_target] *= multiplier
            elif phrase in self.kb.negators:
                if modifier_target and modifier_target in modified_word_scores:
                    modified_word_scores[modifier_target] *= -1

        for target, score in modified_word_scores.items():
            if score > 0:
                positive_contribution += score
            elif score < 0:
                negative_contribution += score

        final_score += positive_contribution
        final_score += negative_contribution

        # 점수가 0일 때 기본 뉘앙스 점수 유지
        if final_score == 0.0:
            if is_positive_context:
                return 0.3
            elif is_negative_context:
                return -0.3

        return final_score
