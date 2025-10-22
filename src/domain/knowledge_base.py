import os
import pandas as pd

class KnowledgeBase:
    def __init__(self, dic_path="dic"):
        # 상대 경로를 프로젝트 루트 기준으로 변경
        self.dic_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), dic_path)
        self._load_dictionaries()

    def _load_dictionaries(self):
        try:
            def _load_dict_list(file_path, score_col="score"):
                df = pd.read_csv(file_path)
                new_dict = {}
                for _, row in df.iterrows():
                    phrase = row["phrase"]
                    score = row[score_col]
                    if phrase not in new_dict:
                        new_dict[phrase] = []
                    new_dict[phrase].append(score)
                return new_dict

            self.idioms = _load_dict_list(os.path.join(self.dic_path, "idioms.csv"))
            self.amplifiers = _load_dict_list(
                os.path.join(self.dic_path, "amplifiers.csv"), score_col="multiplier"
            )
            self.downtoners = _load_dict_list(
                os.path.join(self.dic_path, "downtoners.csv"), score_col="multiplier"
            )
            self.negators = pd.read_csv(os.path.join(self.dic_path, "negators.csv"))[
                "phrase"
            ].tolist()
            self.adjectives = _load_dict_list(
                os.path.join(self.dic_path, "adjectives.csv")
            )
            self.adverbs = _load_dict_list(os.path.join(self.dic_path, "adverbs.csv"))
            self.sentiment_nouns = _load_dict_list(
                os.path.join(self.dic_path, "sentiment_nouns.csv")
            )

        except FileNotFoundError as e:
            print(f"사전 파일 로드 오류: {e}. 빈 사전으로 시작합니다.")
            self.idioms, self.amplifiers, self.downtoners, self.negators = (
                {},
                {},
                {},
                [],
            )
            self.adjectives, self.adverbs, self.sentiment_nouns = {}, {}, {}

    def is_known_word(self, word: str) -> bool:
        return (
            word in self.idioms
            or word in self.amplifiers
            or word in self.downtoners
            or word in self.negators
            or word in self.adjectives
            or word in self.adverbs
            or word in self.sentiment_nouns
        )

# 싱글턴처럼 사용할 knowledge_base 인스턴스
knowledge_base = KnowledgeBase()
