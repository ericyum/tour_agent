# src/infrastructure/reporting/wordclouds.py
import os
import uuid
from wordcloud import WordCloud
import re
import numpy as np
from PIL import Image
import traceback
from collections import defaultdict

# 감성 사전을 불러오기 위해 knowledge_base 임포트
from src.domain.knowledge_base import knowledge_base

# okt는 더 이상 여기서 필요하지 않음
# from konlpy.tag import Okt
# okt = Okt()

# 기존 STOPWORDS는 주체 필터링에 사용될 수 있으므로 유지
STOPWORDS = set([
    '가', '이', '은', '는', '을', '를', '의', '에', '와', '과', '도', '으로', '로', '하다', '되다', '이다', '있다', '없다',
    '것', '수', '등', '그', '저', '이', '저희', '때', '그것', '곳', '더', '또', '및', '좀', '잘', '꼭', '축제', '후기',
    '입니다', '있습니다', '있어요', '없어요', '해요', '하고', '하는', '그리고', '그래서', '그런데', '하지만', '이제',
    '정말', '매우', '아주', '너무', '진짜', '완전', '많이', '전', '안', '못', '위해', '통해', '대한', '때문', '여기',
    '거기', '저기', '오늘', '내일', '어제', '먼저', '다음', '바로', '다시', '한번', '계속', '이후', '이전', '속',
    '위', '아래', '앞', '뒤', '근처', '주변', '사이', '중', '하나', '정도', '관련', '부분', '경우', '생각', '느낌',
    '방문', '사용', '체험', '추천', '가격', '시간', '사람', '분', '곳', '날', '수', '것', '저', '제', '내', '나', '너',
    '우리', '다른', '모든', '여러', '각종', '다양한', '함께', '직접', '역시', '일단', '사실', '주차', '이용', '가능'
])

FONT_PATHS = [
    'C:/gemini_translation/폰트/maplestory_regular.ttf',
    'C:/gemini_translation/폰트/dunggeunmo.ttf',
    'C:/Windows/Fonts/Malgun.ttf',
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
]

def find_font_path():
    for path in FONT_PATHS:
        if os.path.exists(path):
            return path
    print("[WordCloud] Error: No valid font file found in the specified paths.")
    return None

def positive_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(210, 100%, {random_state.randint(25, 55)}%)"

def negative_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(0, 100%, {random_state.randint(30, 60)}%)"

# 함수의 시그니처를 text 대신 aspect_sentiment_pairs를 받도록 변경
def create_sentiment_wordclouds(aspect_sentiment_pairs: list, keyword: str, mask_path: str = None) -> tuple[str | None, str | None]:
    if not aspect_sentiment_pairs:
        return None, None

    try:
        font_path = find_font_path()
        if not font_path:
            return None, None

        # 감성 점수를 합산할 딕셔너리
        positive_scores = defaultdict(float)
        negative_scores = defaultdict(float)

        sentiment_dictionaries = {
            **knowledge_base.adjectives,
            **knowledge_base.adverbs,
            **knowledge_base.sentiment_nouns,
            **knowledge_base.idioms
        }

        # 입력받은 (주체, 감성) 쌍을 순회
        for aspect, sentiment in aspect_sentiment_pairs:
            # 주체가 유효한지 검사
            if not aspect or len(aspect) < 2 or aspect in STOPWORDS or keyword in aspect:
                continue
            
            # 감성어의 점수를 사전에서 조회
            if sentiment in sentiment_dictionaries:
                scores = sentiment_dictionaries[sentiment]
                if not scores: continue
                
                representative_score = max(scores, key=abs)

                if representative_score > 0:
                    positive_scores[aspect] += representative_score
                elif representative_score < 0:
                    negative_scores[aspect] += abs(representative_score)
        
        mask_array = None
        if mask_path and os.path.exists(mask_path):
            try:
                img = Image.open(mask_path).convert("L")
                mask_array = np.array(img, dtype=np.uint8)
            except Exception as e:
                print(f"[WordCloud] Error loading mask image: {e}")
                mask_array = None

        common_wc_args = {
            "font_path": font_path, "width": 800, "height": 800,
            "background_color": 'white', "mask": mask_array,
            "max_words": 100, "contour_width": 1
        }
        
        temp_dir = os.path.join(os.getcwd(), "temp_images")
        os.makedirs(temp_dir, exist_ok=True)
        sanitized_keyword = re.sub(r'[\\/:*?"<>|]', '_', keyword)

        positive_wc_path = None
        if positive_scores:
            wc_pos = WordCloud(**common_wc_args, color_func=positive_color_func, contour_color='blue').generate_from_frequencies(positive_scores)
            positive_wc_path = os.path.join(temp_dir, f"wc_pos_{sanitized_keyword}_{uuid.uuid4()}.png")
            wc_pos.to_file(positive_wc_path)
            print(f"[WordCloud] Positive Aspect WC generated: {positive_wc_path}")

        negative_wc_path = None
        if negative_scores:
            wc_neg = WordCloud(**common_wc_args, color_func=negative_color_func, contour_color='red').generate_from_frequencies(negative_scores)
            negative_wc_path = os.path.join(temp_dir, f"wc_neg_{sanitized_keyword}_{uuid.uuid4()}.png")
            wc_neg.to_file(negative_wc_path)
            print(f"[WordCloud] Negative Aspect WC generated: {negative_wc_path}")

        return positive_wc_path, negative_wc_path

    except Exception as e:
        print(f"[WordCloud] Error during aspect-based WC generation: {e}")
        traceback.print_exc()
        return None, None