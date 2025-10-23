import sqlite3
import json
import os
import gradio as gr
import re
import math
import asyncio
import io
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
# --- Environment Setup (must be first) ---
from src.infrastructure.config.settings import setup_environment, get_google_api_key, Settings
setup_environment()

# --- Matplotlib Backend Setup ---
import matplotlib
matplotlib.use('Agg')

# --- Visualization and NLP Libraries (Assumed to be installed) ---
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from konlpy.tag import Okt
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("Warning: `wordcloud`, `matplotlib`, `konlpy`, `Pillow`, or `numpy` libraries are not installed. Visual analysis will not work.")
    WordCloud, plt, Okt, Image, ImageDraw, ImageFont, np = [None]*7

# --- Custom Module Imports ---
from src.infrastructure.persistence.database import get_db_connection, init_db
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor
from src.infrastructure.external_services.naver_search.naver_review_api import get_naver_trend, search_naver_blog

# --- New LLM-related Imports (for sentiment analysis only) ---
from src.application.core.state import LLMGraphState
from src.domain.knowledge_base import knowledge_base
from src.infrastructure.llm_client import get_llm_client
from src.infrastructure.dynamic_scorer import SimpleScorer
from src.application.agents.common.content_validator import agent_content_validator
from src.application.agents.common.llm_summarizer import agent_llm_summarizer
from src.application.agents.common.rule_scorer import agent_rule_scorer_on_summary
from src.application.core.graph import app_llm_graph
from src.infrastructure.reporting.charts import create_donut_chart, create_stacked_bar_chart, create_sentence_score_bar_chart
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.application.core.utils import get_season, save_df_to_csv, summarize_negative_feedback, create_driver, change_page, get_logger, haversine

# --- Selenium Imports (for sentiment analysis only) ---
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import traceback

# --- NLTK Setup ---
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# --- Constants and Setup ---
script_dir = os.path.dirname(__file__)


CUSTOM_CSS = """#gallery .thumbnail-item { max-width: 250px !important; min-width: 200px !important; flex-grow: 1 !important; }"""

from src.application.core.constants import AREA_CODE_MAP, SIGUNGU_CODE_MAP, CATEGORY_TO_ICON_MAP, NO_IMAGE_URL, PAGE_SIZE

# --- Data Loading and Mappings ---

CAT_NAME_TO_CODE = {'main': {}, 'medium': {}, 'small': {}}
TITLE_TO_CAT_NAMES = {}

COLUMN_TRANSLATIONS = {
    'addr1': '주소', 'addr2': '상세주소', 'tel': '전화번호', 'title': '제목', 'zipcode': '우편번호', 
    'telname': '연락처 이름', 'homepage': '홈페이지', 'overview': '개요', 'sponsor1': '주최자',
    'sponsor1tel': '주최자 연락처', 'eventenddate': '행사 종료일', 'playtime': '공연 시간',
    'eventplace': '행사 장소', 'eventstartdate': '행사 시작일', 'usetimefestival': '이용 요금',
    'sponsor2': '후원사', 'progresstype': '진행 상태', 'festivaltype': '축제 유형',
    'sponsor2tel': '후원사 연락처', 'agelimit': '연령 제한', 'spendtimefestival': '관람 소요시간',
    'festivalgrade': '축제 등급', 'eventhomepage': '행사 홈페이지', 'subevent': '부대 행사',
    'program': '행사 프로그램', 'discountinfofestival': '할인 정보', 'placeinfo': '행사장 위치 안내',
    'bookingplace': '예매처', 'usefee': '이용 요금', 'infocenterculture': '문의 및 안내',
    'usetimeculture': '이용 시간', 'restdateculture': '쉬는 날', 'parkingfee': '주차 요금',
    'parkingculture': '주차 시설', 'chkcreditcardculture': '신용카드 가능 정보',
    'chkbabycarriageculture': '유모차 대여 정보', 'spendtime': '관람 소요시간',
    'accomcountculture': '수용인원', 'scale': '규모', 'chkpetculture': '반려동물 동반 가능 정보',
    'discountinfo': '할인 정보', 'distance': '총 거리', 'schedule': '코스 일정',
    'taketime': '총 소요시간', 'theme': '코스 테마', 'subnum': '세부 코스 번호',
    'subname': '세부 코스명', 'subdetailoverview': '세부 코스 개요', 'firstimage': '대표 이미지',
    'firstimage2': '추가 이미지'
}

settings = Settings()
logger = get_logger(__name__)

# Initialize the database
init_db()

# Load festival categories and maps
def load_festival_categories_and_maps():
    global TITLE_TO_CAT_NAMES
    festivals_dir = os.path.join(script_dir, "festivals")
    all_categories = {}
    try:
        for filename in os.listdir(festivals_dir):
            if filename.endswith(".json"):
                with open(os.path.join(festivals_dir, filename), 'r', encoding='utf-8') as f:
                    all_categories.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading festival categories: {e}")
        return {}

    for main_name, med_dict in all_categories.items():
        for med_name, small_dict in med_dict.items():
            for small_name, titles in small_dict.items():
                for title in titles:
                    TITLE_TO_CAT_NAMES[title] = (main_name, med_name, small_name)

    db_path = os.path.join(script_dir, "tour.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT title, cat1, cat2, cat3 FROM festivals WHERE cat1 IS NOT NULL AND cat2 IS NOT NULL AND cat3 IS NOT NULL")
    db_festivals = cursor.fetchall()
    conn.close()

    for row in db_festivals:
        title, code1, code2, code3 = row
        if title in TITLE_TO_CAT_NAMES:
            name1, name2, name3 = TITLE_TO_CAT_NAMES[title]
            CAT_NAME_TO_CODE['main'][name1] = code1
            CAT_NAME_TO_CODE['medium'][name2] = code2
            CAT_NAME_TO_CODE['small'][name3] = code3
    
    print("[app.py] Category name-to-code maps created.")
    return all_categories

ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()
naver_supervisor = NaverReviewSupervisor()
okt = Okt() if Okt else None

def get_korean_font():
    if plt is None: return None
    try:
        font_path = font_manager.findfont(font_manager.FontProperties(family='Malgun Gothic'))
        if os.path.exists(font_path):
            return font_path
    except:
        pass
    font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    for font in font_list:
        if 'gothic' in font.lower() or 'gulim' in font.lower() or 'apple' in font.lower():
            return font
    print("Warning: Korean font not found. Visualization text may be broken.")
    return None

KOREAN_FONT_PATH = get_korean_font()



from src.application.supervisors.db_search_supervisor import db_search_graph

def display_page(results, page):
    page = int(page)
    start_index = (page - 1) * PAGE_SIZE
    end_index = start_index + PAGE_SIZE
    page_results = results[start_index:end_index]
    gallery_output = [(item[1], item[0]) for item in page_results]
    total_pages = math.ceil(len(results) / PAGE_SIZE)
    return gallery_output, f"{page} / {total_pages}"

def display_paginated_gallery(results, page_str, direction):
    page = int(page_str.split('/')[0].strip())
    total_pages = math.ceil(len(results) / PAGE_SIZE)
    new_page = page + direction
    if 1 <= new_page <= total_pages:
        return display_page(results, new_page)
    return gr.update(), gr.update()

def search_festival_in_db(festival_name):
    if not festival_name: return None
    db_path = os.path.join(script_dir, "tour.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM festivals WHERE title = ?", (festival_name,))
        festival = cursor.fetchone()
        return dict(festival) if festival else None
    finally:
        conn.close()

def display_festival_details(evt: gr.SelectData, results, page_str):
    page = int(page_str.split('/')[0].strip())
    global_index = (page - 1) * PAGE_SIZE + evt.index
    selected_title = results[global_index][0]
    details = search_festival_in_db(selected_title)

    if not details:
        return gr.update(value="정보를 찾을 수 없습니다."), None, None

    details_list = []
    exclude_cols = [
        'id', 'contentid', 'contenttypeid', 'lDongRegnCd', 'lDongSignguCd', 
        'lclsSystm1', 'lclsSystm2', 'lclsSystm3', 'mlevel', 'cpyrhtDivCd',
        'areacode', 'cat1', 'cat2', 'cat3', 'createdtime', 'mapx', 'mapy', 
        'modifiedtime', 'sigungucode'
    ]
    for key, value in details.items():
        if key in exclude_cols:
            continue
        if value is not None and str(value).strip() != '':
            display_key = COLUMN_TRANSLATIONS.get(key, key)
            details_list.append(f"**{display_key}**: {value}")
    
    details_text = "\n\n".join(details_list)

    return gr.update(value=details_text), details.get('title'), details



async def get_naver_review_info(festival_name, num_reviews):
    if not festival_name: 
        yield gr.update(value="먼저 축제를 선택해주세요.", visible=True), gr.update(visible=False)
        return
    
    yield gr.update(value=f"{festival_name} 후기 검색 중... ({num_reviews}개)", visible=True), gr.update(visible=True, open=True)
    summary, _ = await naver_supervisor.get_review_summary_and_tips(festival_name, num_reviews=num_reviews)
    yield gr.update(value=summary, visible=True), gr.update(visible=True, open=True)




async def generate_trend_graphs(festival_name):
    if not festival_name:
        yield gr.update(visible=False), gr.update(value="축제를 선택해주세요.", visible=True), None, None
        return

    if plt is None:
        yield gr.update(visible=True, open=True), gr.update(value="`matplotlib` 라이브러리가 설치되지 않았습니다.", visible=True), None, None
        return

    yield gr.update(visible=True, open=True), gr.update(value="트렌드 그래프 생성 중...", visible=True), None, None

    font_properties = font_manager.FontProperties(fname=KOREAN_FONT_PATH) if KOREAN_FONT_PATH else None
    details = search_festival_in_db(festival_name)

    # --- 1. 1-Year Trend Graph ---
    today = datetime.today()
    start_date_yearly = today - timedelta(days=365)
    trend_data_yearly = get_naver_trend(festival_name, start_date_yearly, today)
    
    fig_trend_yearly, ax_yearly = plt.subplots(figsize=(10, 5))
    if trend_data_yearly: # Check if list is not empty
        df = pd.DataFrame(trend_data_yearly)
        df['period'] = pd.to_datetime(df['period'])
        ax_yearly.plot(df['period'], df['ratio'])
        ax_yearly.set_title(f"'{festival_name}' 최근 1년 검색량 트렌드", fontproperties=font_properties, fontsize=16)
        ax_yearly.tick_params(axis='x', rotation=30)
    else:
        ax_yearly.text(0.5, 0.5, "트렌드 데이터 없음", ha='center', va='center', fontproperties=font_properties)
    plt.tight_layout()
    buf_trend_yearly = io.BytesIO()
    fig_trend_yearly.savefig(buf_trend_yearly, format='png')
    trend_image_yearly = Image.open(buf_trend_yearly)
    plt.close(fig_trend_yearly)

    # --- 2. Event-Period Trend Graph ---
    fig_trend_event, ax_event = plt.subplots(figsize=(10, 5))
    if details and details.get('eventstartdate'):
        try:
            date_str = str(int(details.get('eventstartdate')))
        except (ValueError, TypeError):
            date_str = str(details.get('eventstartdate'))
        
        center_date = pd.to_datetime(date_str, errors='coerce')
        
        if pd.notna(center_date):
            graph_start = center_date - timedelta(days=7)
            graph_end = center_date + timedelta(days=7)

            trend_data_event = get_naver_trend(festival_name, graph_start, graph_end)
            if trend_data_event: # Check if list is not empty
                df_event = pd.DataFrame(trend_data_event)
                df_event['period'] = pd.to_datetime(df_event['period'])
                ax_event.plot(df_event['period'], df_event['ratio'])
                ax_event.axvline(x=center_date, color='r', linestyle='--', label='Festival Start')
                ax_event.legend()
                ax_event.tick_params(axis='x', rotation=30)
            else:
                 ax_event.text(0.5, 0.5, "기간 트렌드 데이터 없음", ha='center', va='center', fontproperties=font_properties)
        else:
            ax_event.text(0.5, 0.5, "날짜 형식 오류", ha='center', va='center', fontproperties=font_properties)
    else:
        ax_event.text(0.5, 0.5, "축제 시작일 정보 없음", ha='center', va='center', fontproperties=font_properties)
    
    ax_event.set_title(f"'{festival_name}' 축제 시작일 중심 트렌드", fontproperties=font_properties, fontsize=16)
    plt.tight_layout()
    buf_trend_event = io.BytesIO()
    fig_trend_event.savefig(buf_trend_event, format='png')
    trend_image_event = Image.open(buf_trend_event)
    plt.close(fig_trend_event)

    yield gr.update(visible=True, open=True), gr.update(visible=False), trend_image_yearly, trend_image_event



async def generate_word_cloud(festival_name, num_reviews):
    if not festival_name:
        yield gr.update(visible=False), gr.update(value="축제를 선택해주세요.", visible=True), None
        return

    if WordCloud is None or Okt is None or np is None:
        yield gr.update(visible=True, open=True), gr.update(value="`wordcloud`, `konlpy`, 또는 `numpy` 라이브러리가 설치되지 않았습니다.", visible=True), None
        return

    yield gr.update(visible=True, open=True), gr.update(value=f"워드 클라우드 생성 중... ({num_reviews}개)", visible=True), None

    main_cat_tuple = TITLE_TO_CAT_NAMES.get(festival_name)
    main_cat = main_cat_tuple[0] if main_cat_tuple else None
    icon_name = None
    if main_cat:
        normalized_cat_name = main_cat.replace(" ", "")
        icon_name = CATEGORY_TO_ICON_MAP.get(normalized_cat_name)

    mask_array = None
    if icon_name:
        path = os.path.join(script_dir, "assets", f"{icon_name}.png")
        if os.path.exists(path):
            try:
                icon_image = Image.open(path).convert("L")
                mask_array = np.array(icon_image)
                mask_array = 255 - mask_array # This inverts the mask
                print(f"DEBUG: Mask array min value: {mask_array.min()}, max value: {mask_array.max()}")
                non_white_pixels = np.sum(mask_array < 255)
                total_pixels = mask_array.size
                print(f"DEBUG: Percentage of non-white pixels in mask: {non_white_pixels / total_pixels * 100:.2f}%")
            except Exception as e:
                print(f"Error loading mask image: {e}")

    stopwords = {'축제', '오늘', '여기', '저희', '이번', '진짜', '정말', '완전', '후기', '위해', '때문', '하나'}
    _, review_texts = await naver_supervisor.get_review_summary_and_tips(festival_name, num_reviews=num_reviews, return_full_text=True)
    
    wc_image = None
    if review_texts:
        nouns = [word for text in review_texts for word in okt.nouns(text) if len(word) > 1 and word not in stopwords]
        counts = Counter(nouns)
        if counts:
            wc = WordCloud(font_path=KOREAN_FONT_PATH, background_color="white", mask=mask_array, contour_color='steelblue', contour_width=1).generate_from_frequencies(counts)
            wc_image = wc.to_image()
    
    if wc_image is None:
        wc_image = Image.new('RGB', (800, 400), 'white')
        draw = ImageDraw.Draw(wc_image)
        try: font = ImageFont.truetype(KOREAN_FONT_PATH, 20)
        except: font = ImageFont.load_default()
        draw.text((300, 180), "추출된 단어 없음", font=font, fill="black")

    yield gr.update(visible=True, open=True), gr.update(visible=False), wc_image


async def analyze_sentiment(festival_name, num_reviews):
    outputs_to_clear = [
        gr.update(open=True),  # sentiment_accordion
        "",  # sentiment_status
        gr.update(visible=False),  # sentiment_negative_summary
        gr.update(visible=False),  # sentiment_overall_chart
        gr.update(visible=False),  # sentiment_summary
        gr.update(visible=False),  # sentiment_overall_csv
        gr.update(visible=False),  # sentiment_spring_chart
        gr.update(visible=False),  # sentiment_summer_chart
        gr.update(visible=False),  # sentiment_autumn_chart
        gr.update(visible=False),  # sentiment_winter_chart
        gr.update(visible=False),  # sentiment_spring_pos_wc
        gr.update(visible=False),  # sentiment_spring_neg_wc
        gr.update(visible=False),  # sentiment_summer_pos_wc
        gr.update(visible=False),  # sentiment_summer_neg_wc
        gr.update(visible=False),  # sentiment_autumn_pos_wc
        gr.update(visible=False),  # sentiment_autumn_neg_wc
        gr.update(visible=False),  # sentiment_winter_pos_wc
        gr.update(visible=False),  # sentiment_winter_neg_wc
        None,  # sentiment_df_output
        None,  # blog_results_df_state
        None,  # blog_judgments_state
        1,  # sentiment_blog_page_num_input
        "/ 1",  # sentiment_blog_total_pages_output
        gr.update(visible=False),  # sentiment_blog_list_csv
        gr.update(visible=False),  # sentiment_individual_summary
        gr.update(visible=False),  # sentiment_individual_donut_chart
        gr.update(visible=False),  # sentiment_individual_score_chart
        gr.update(visible=False, open=False),  # sentiment_blog_detail_accordion
    ]

    if not festival_name:
        outputs_to_clear[1] = "축제를 선택해주세요."
        yield tuple(outputs_to_clear)
        return

    try:
        outputs_to_clear[1] = "블로그 검색 중..."
        yield tuple(outputs_to_clear)

        search_keyword = f"{festival_name} 후기"

        blog_results_list = []
        all_negative_sentences = []
        seasonal_aspect_pairs = {"봄": [], "여름": [], "가을": [], "겨울": [], "정보없음": []}
        seasonal_texts = {"봄": [], "여름": [], "가을": [], "겨울": [], "정보없음": []}
        seasonal_data = {"봄": {"pos": 0, "neg": 0}, "여름": {"pos": 0, "neg": 0}, "가을": {"pos": 0, "neg": 0}, "겨울": {"pos": 0, "neg": 0}, "정보없음": {"pos": 0, "neg": 0}}
        total_pos, total_neg, total_strong_pos, total_strong_neg = 0, 0, 0, 0
        
        api_results = search_naver_blog(search_keyword, display=num_reviews + 10)
        if not api_results:
            outputs_to_clear[1] = f"'{search_keyword}'에 대한 네이버 블로그를 찾을 수 없습니다."
            yield tuple(outputs_to_clear)
            return

        candidate_blogs = []
        for item in api_results:
            if "blog.naver.com" in item["link"]:
                item['title'] = re.sub(r'<[^>]+>', '', item['title']).strip()
                if item['title'] and item["link"]:
                    candidate_blogs.append(item)
            if len(candidate_blogs) >= num_reviews:
                break
        
        if not candidate_blogs:
            outputs_to_clear[1] = f"'{search_keyword}'에 대한 유효한 블로그 후보를 찾지 못했습니다."
            yield tuple(outputs_to_clear)
            return

        valid_blogs_data = []
        blog_judgments_list = []
        all_negative_sentences = []
        seasonal_aspect_pairs = {"봄": [], "여름": [], "가을": [], "겨울": [], "정보없음": []}
        seasonal_texts = {"봄": [], "여름": [], "가을": [], "겨울": [], "정보없음": []}
        seasonal_data = {"봄": {"pos": 0, "neg": 0}, "여름": {"pos": 0, "neg": 0}, "가을": {"pos": 0, "neg": 0}, "겨울": {"pos": 0, "neg": 0}, "정보없음": {"pos": 0, "neg": 0}}
        total_pos, total_neg, total_strong_pos, total_strong_neg = 0, 0, 0, 0

        for i, blog_data in enumerate(candidate_blogs):
            outputs_to_clear[1] = f"블로그 분석 중... ({len(valid_blogs_data)}/{num_reviews} 완료, {i+1}/{len(candidate_blogs)} 확인)"
            yield tuple(outputs_to_clear)

            try:
                content, _ = await naver_supervisor._scrape_blog_content(blog_data["link"])
                if not content or "오류" in content or "찾을 수 없습니다" in content:
                    continue

                max_content_length = 30000
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "... (내용 일부 생략)"

                final_state = app_llm_graph.invoke({
                    "original_text": content, "keyword": festival_name, "title": blog_data["title"],
                    "log_details": False, "re_summarize_count": 0, "is_relevant": False
                })

                if not final_state or not final_state.get("is_relevant"):
                    continue

                judgments = final_state.get("final_judgments", [])
                if not judgments:
                    continue

                season = get_season(blog_data.get('postdate', ''))
                seasonal_texts[season].append(content)
                
                aspect_pairs = final_state.get("aspect_sentiment_pairs", [])
                if aspect_pairs:
                    seasonal_aspect_pairs[season].extend(aspect_pairs)

                blog_judgments_list.append(judgments)
                pos_count = sum(1 for res in judgments if res["final_verdict"] == "긍정")
                neg_count = sum(1 for res in judgments if res["final_verdict"] == "부정")
                
                strong_pos_count = sum(1 for res in judgments if res["final_verdict"] == "긍정" and res["score"] >= 1.0)
                strong_neg_count = sum(1 for res in judgments if res["final_verdict"] == "부정" and res["score"] < -1.0)

                total_pos += pos_count
                total_neg += neg_count
                total_strong_pos += strong_pos_count
                total_strong_neg += strong_neg_count
                all_negative_sentences.extend([res["sentence"] for res in judgments if res["final_verdict"] == "부정"])
                
                seasonal_data[season]["pos"] += pos_count
                seasonal_data[season]["neg"] += neg_count
                
                sentiment_frequency = pos_count + neg_count
                sentiment_score = ((strong_pos_count - strong_neg_count) / sentiment_frequency * 50 + 50) if sentiment_frequency > 0 else 50.0
                pos_perc = (pos_count/sentiment_frequency*100) if sentiment_frequency > 0 else 0.0
                neg_perc = (neg_count/sentiment_frequency*100) if sentiment_frequency > 0 else 0.0

                blog_results_list.append({
                    "블로그 제목": blog_data["title"], "링크": blog_data["link"], "감성 빈도": sentiment_frequency,
                    "감성 점수": f"{sentiment_score:.1f}", "긍정 문장 수": pos_count, "부정 문장 수": neg_count,
                    "긍정 비율 (%)": f"{pos_perc:.1f}", "부정 비율 (%)": f"{neg_perc:.1f}",
                    "긍/부정 문장 요약": "\n---\n".join([f"[{res['final_verdict']}] {res['sentence']}" for res in judgments])
                })
                valid_blogs_data.append(blog_data)
            except Exception as e:
                print(f"블로그 분석 중 오류 ({festival_name}, {blog_data.get('link', 'N/A')}): {e}")
                traceback.print_exc()
                continue
        
        if not valid_blogs_data:
            outputs_to_clear[1] = f"'{festival_name}'에 대한 유효한 후기 블로그를 찾지 못했습니다."
            yield tuple(outputs_to_clear)
            return

        total_sentiment_frequency = total_pos + total_neg
        total_sentiment_score = ((total_strong_pos - total_strong_neg) / total_sentiment_frequency * 50 + 50) if total_sentiment_frequency > 0 else 50.0

        neg_summary_text = summarize_negative_feedback(all_negative_sentences)
        overall_summary_text = f"""- **긍정 문장 수**: {total_pos}개
- **부정 문장 수**: {total_neg}개
- **감성어 빈도 (긍정+부정)**: {total_sentiment_frequency}개
- **감성 점수**: {total_sentiment_score:.1f}점 (0~100점)"""

        summary_df = pd.DataFrame([{'축제명': festival_name, '감성 빈도': total_sentiment_frequency, '감성 점수': f"{total_sentiment_score:.1f}", '긍정 문장 수': total_pos, '부정 문장 수': total_neg}])
        summary_csv = save_df_to_csv(summary_df, "overall_summary", festival_name)
        blog_df = pd.DataFrame(blog_results_list)
        blog_list_csv = save_df_to_csv(blog_df, "blog_list", festival_name)
        
        initial_page_df, current_page, total_pages_str = change_page(blog_df, 1)

        seasonal_pos_wc_paths = {}
        seasonal_neg_wc_paths = {}
        for season, pairs in seasonal_aspect_pairs.items():
            season_en = CATEGORY_TO_ICON_MAP.get(get_season(pairs[0]['postdate'] if pairs and 'postdate' in pairs[0] else '2000-01-01'))
            if pairs and season_en:
                mask_path = os.path.abspath(os.path.join(script_dir, "assets", f"mask_{season_en}.png"))
                pos_path, neg_path = create_sentiment_wordclouds(pairs, f"{festival_name}_{season}", mask_path=mask_path)
                seasonal_pos_wc_paths[season] = pos_path
                seasonal_neg_wc_paths[season] = neg_path
            else:
                seasonal_pos_wc_paths[season] = None
                seasonal_neg_wc_paths[season] = None

        yield (
            gr.update(visible=True, open=True),
            "분석 완료",
            gr.update(value=neg_summary_text, visible=bool(neg_summary_text)),
            gr.update(value=create_donut_chart(total_pos, total_neg, f'{festival_name} 전체 후기 요약'), visible=True),
            gr.update(value=overall_summary_text, visible=True),
            gr.update(value=summary_csv, visible=summary_csv is not None),
            gr.update(value=create_stacked_bar_chart(seasonal_data.get("봄", {}).get("pos", 0), seasonal_data.get("봄", {}).get("neg", 0), "봄 시즌"), visible=seasonal_data.get("봄", {}).get("pos", 0) > 0 or seasonal_data.get("봄", {}).get("neg", 0) > 0),
            gr.update(value=create_stacked_bar_chart(seasonal_data.get("여름", {}).get("pos", 0), seasonal_data.get("여름", {}).get("neg", 0), "여름 시즌"), visible=seasonal_data.get("여름", {}).get("pos", 0) > 0 or seasonal_data.get("여름", {}).get("neg", 0) > 0),
            gr.update(value=create_stacked_bar_chart(seasonal_data.get("가을", {}).get("pos", 0), seasonal_data.get("가을", {}).get("neg", 0), "가을 시즌"), visible=seasonal_data.get("가을", {}).get("pos", 0) > 0 or seasonal_data.get("가을", {}).get("neg", 0) > 0),
            gr.update(value=create_stacked_bar_chart(seasonal_data.get("겨울", {}).get("pos", 0), seasonal_data.get("겨울", {}).get("neg", 0), "겨울 시즌"), visible=seasonal_data.get("겨울", {}).get("pos", 0) > 0 or seasonal_data.get("겨울", {}).get("neg", 0) > 0),
            gr.update(value=seasonal_pos_wc_paths.get("봄"), visible=seasonal_pos_wc_paths.get("봄") is not None),
            gr.update(value=seasonal_neg_wc_paths.get("봄"), visible=seasonal_neg_wc_paths.get("봄") is not None),
            gr.update(value=seasonal_pos_wc_paths.get("여름"), visible=seasonal_pos_wc_paths.get("여름") is not None),
            gr.update(value=seasonal_neg_wc_paths.get("여름"), visible=seasonal_neg_wc_paths.get("여름") is not None),
            gr.update(value=seasonal_pos_wc_paths.get("가을"), visible=seasonal_pos_wc_paths.get("가을") is not None),
            gr.update(value=seasonal_neg_wc_paths.get("가을"), visible=seasonal_neg_wc_paths.get("가을") is not None),
            gr.update(value=seasonal_pos_wc_paths.get("겨울"), visible=seasonal_pos_wc_paths.get("겨울") is not None),
            gr.update(value=seasonal_neg_wc_paths.get("겨울"), visible=seasonal_neg_wc_paths.get("겨울") is not None),
            initial_page_df, 
            blog_df, 
            blog_judgments_list, 
            current_page, 
            total_pages_str,
            gr.update(value=blog_list_csv, visible=blog_list_csv is not None),
            gr.update(visible=False), # individual_summary
            gr.update(visible=False), # individual_donut_chart
            gr.update(visible=False), # individual_score_chart
            gr.update(visible=False, open=False) # individual_detail_accordion
        )

    except Exception as e:
        print(f"감성 분석 중 예외 발생: {e}")
        traceback.print_exc()
        outputs_to_clear[1] = f"분석 중 오류 발생: {e}"
        yield tuple(outputs_to_clear)







# --- Core Logic Functions ---





# --- Gradio Interface ---

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# 축제 정보 검색 에이전트")

    results_state = gr.State([])
    page_state = gr.State(1)
    selected_festival_state = gr.State()
    selected_festival_details_state = gr.State()
    recommended_facilities_state = gr.State([])
    recommended_courses_state = gr.State([])

    with gr.Group():
        with gr.Row():
            area_dropdown = gr.Dropdown(label="시/도", choices=["전체"] + sorted(list(AREA_CODE_MAP.keys())), value="전체", interactive=True)
            sigungu_dropdown = gr.Dropdown(label="시/군/구", choices=["전체"], value="전체", interactive=True)
        with gr.Row():
            main_cat_dropdown = gr.Dropdown(label="대분류", choices=["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.keys())), value="전체", interactive=True)
            medium_cat_dropdown = gr.Dropdown(label="중분류", choices=["전체"], value="전체", interactive=True)
            small_cat_dropdown = gr.Dropdown(label="소분류", choices=["전체"], value="전체", interactive=True)
    
    search_btn = gr.Button("검색", variant="primary")

    with gr.Column(visible=False) as results_area:
        festival_gallery = gr.Gallery(label="축제 목록", show_label=False, elem_id="gallery", columns=4, height="auto", object_fit="contain")
        with gr.Row(variant="panel"):
            prev_button = gr.Button("◀ 이전")
            page_display = gr.Textbox(value="1 / 1", label="페이지", interactive=False, container=False, scale=1)
            next_button = gr.Button("다음 ▶")

    with gr.Accordion("축제 상세 정보", open=False) as details_accordion:
        festival_details_output = gr.Markdown()

    with gr.Accordion("좌표 기반 추천", open=False, visible=False) as recommend_accordion:
        with gr.Row():
            recommend_radius_slider = gr.Slider(minimum=100, maximum=20000, value=5000, step=100, label="반경 (미터)", interactive=True)
            recommend_btn = gr.Button("추천 받기", variant="primary")
        
        with gr.Row(visible=False) as ranking_controls: # Hide until there are results
            ranking_reviews_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="순위용 리뷰 수", interactive=True)
            ranking_top_n_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="표시할 순위 수", interactive=True)
            rank_facilities_btn = gr.Button("관광 시설 순위 매기기")
            rank_courses_btn = gr.Button("관광 코스 순위 매기기")

        recommend_status = gr.Textbox(label="상태", interactive=False, visible=False)
        gr.Markdown("### 추천 관광 시설")
        recommend_facilities_gallery = gr.Gallery(label="추천 관광 시설", show_label=False, elem_id="recommend_facilities_gallery", columns=4, height="auto", object_fit="contain")
        facility_ranking_report = gr.Markdown(visible=False)
        gr.Markdown("### 추천 관광 코스")
        recommend_courses_gallery = gr.Gallery(label="추천 관광 코스", show_label=False, elem_id="recommend_courses_gallery", columns=4, height="auto", object_fit="contain")
        course_ranking_report = gr.Markdown(visible=False)
        with gr.Accordion("추천 장소 상세 정보", open=False, visible=False) as recommend_details_accordion:
            recommend_details_output = gr.Markdown()

    with gr.Accordion("Naver 후기 요약 및 꿀팁", open=False, visible=False) as naver_review_accordion:
        with gr.Row():
            num_reviews_naver_summary = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="분석할 후기 수 (네이버 요약)", interactive=True)
            naver_search_btn = gr.Button("네이버 후기 요약 검색", variant="primary")
        naver_review_output = gr.Markdown()
    
    with gr.Accordion("검색량 트렌드 그래프", open=False, visible=False) as trend_accordion:
        trend_graph_btn = gr.Button("트렌드 그래프 생성", variant="primary")
        trend_status = gr.Textbox(label="상태", interactive=False)
        with gr.Row():
            trend_plot_yearly = gr.Image(label="최근 1년 검색량 트렌드")
            trend_plot_event = gr.Image(label="축제 기간 중심 트렌드")

    with gr.Accordion("워드 클라우드", open=False, visible=False) as wordcloud_accordion:
        with gr.Row():
            num_reviews_wordcloud = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="분석할 후기 수 (워드클라우드)", interactive=True)
            word_cloud_btn = gr.Button("워드 클라우드 생성", variant="primary")
        wordcloud_status = gr.Textbox(label="상태", interactive=False)
        wordcloud_plot = gr.Image(label="축제의 주요 핵심 요소들")

    # --- Gradio Interface ---

    # State variables for pagination and individual blog details
    blog_results_df_state = gr.State()
    blog_judgments_state = gr.State()
    individual_blog_page_num_state = gr.State(1)

    with gr.Accordion("감성 분석", open=False, visible=False) as sentiment_accordion:
        with gr.Row():
            num_reviews_slider = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="분석할 후기 수", interactive=True)
            run_sentiment_btn = gr.Button("감성 분석 실행", variant="primary")
        
        sentiment_status = gr.Textbox(label="분석 상태", interactive=False)
        
        with gr.Accordion("종합 분석 결과", open=True):
            sentiment_summary = gr.Markdown(label="종합 분석 상세", visible=False)
            sentiment_overall_csv = gr.File(label="종합 분석 (CSV) 다운로드", visible=False)
            sentiment_overall_chart = gr.Plot(label="전체 후기 요약", visible=False)
            sentiment_negative_summary = gr.Markdown(label="주요 불만 사항 요약", visible=False)
            
            with gr.Accordion("계절별 상세 분석", open=False):
                with gr.Row():
                    sentiment_spring_chart = gr.Plot(label="봄 시즌", visible=False, scale=1)
                    sentiment_spring_pos_wc = gr.Image(label="봄 긍정 워드클라우드", visible=False, scale=1)
                    sentiment_spring_neg_wc = gr.Image(label="봄 부정 워드클라우드", visible=False, scale=1)
                with gr.Row():
                    sentiment_summer_chart = gr.Plot(label="여름 시즌", visible=False, scale=1)
                    sentiment_summer_pos_wc = gr.Image(label="여름 긍정 워드클라우드", visible=False, scale=1)
                    sentiment_summer_neg_wc = gr.Image(label="여름 부정 워드클라우드", visible=False, scale=1)
                with gr.Row():
                    sentiment_autumn_chart = gr.Plot(label="가을 시즌", visible=False, scale=1)
                    sentiment_autumn_pos_wc = gr.Image(label="가을 긍정 워드클라우드", visible=False, scale=1)
                    sentiment_autumn_neg_wc = gr.Image(label="가을 부정 워드클라우드", visible=False, scale=1)
                with gr.Row():
                    sentiment_winter_chart = gr.Plot(label="겨울 시즌", visible=False, scale=1)
                    sentiment_winter_pos_wc = gr.Image(label="겨울 긍정 워드클라우드", visible=False, scale=1)
                    sentiment_winter_neg_wc = gr.Image(label="겨울 부정 워드클라우드", visible=False, scale=1)

        gr.Markdown("### 개별 블로그 분석 결과")
        sentiment_df_output = gr.DataFrame(
            headers=["블로그 제목", "링크", "감성 빈도", "감성 점수", "긍정 문장 수", "부정 문장 수", "긍정 비율 (%)", "부정 비율 (%)", "긍/부정 문장 요약"],
            datatype=["str", "str", "number", "str", "number", "number", "str", "str", "str"],
            label="개별 블로그 분석 결과", wrap=True, interactive=True
        )
        with gr.Row():
            sentiment_blog_page_num_input = gr.Number(value=1, label="페이지 번호", interactive=True, scale=1)
            sentiment_blog_total_pages_output = gr.Textbox(value="/ 1", label="전체 페이지", interactive=False, container=False, scale=1)
            sentiment_blog_list_csv = gr.File(label="전체 블로그 목록(CSV) 다운로드", visible=False, scale=2)

        with gr.Accordion("개별 블로그 상세 분석 (표에서 행 선택)", open=False, visible=False) as sentiment_blog_detail_accordion:
            sentiment_individual_summary = gr.Textbox(label="긍/부정 문장 요약", visible=False, interactive=False, lines=10)
            with gr.Row():
                sentiment_individual_donut_chart = gr.Plot(label="개별 블로그 긍/부정 비율", visible=False)
                sentiment_individual_score_chart = gr.Plot(label="문장별 감성 점수", visible=False)

    # --- Event Handlers ---

    def get_trend_score(keyword):
        if not keyword:
            return 0

        today = datetime.today()
        start_date = today - timedelta(days=90) # Last 90 days
        trend_data = get_naver_trend(keyword, start_date, today)
        
        if not trend_data:
            return 0
            
        df = pd.DataFrame(trend_data)
        if 'ratio' in df.columns and not df['ratio'].empty:
            # Return the average ratio as the score
            return df['ratio'].mean()
        
        return 0

    async def get_sentiment_score(keyword, num_reviews):
        if not keyword:
            return 50.0, [] # Return a neutral score and empty list

        search_keyword = f"{keyword} 후기"
        
        api_results = search_naver_blog(search_keyword, display=num_reviews + 10)
        if not api_results:
            return 50.0, []

        candidate_blogs = []
        for item in api_results:
            if "blog.naver.com" in item["link"]:
                item['title'] = re.sub(r'<[^>]+>', '', item['title']).strip()
                if item['title'] and item["link"]:
                    candidate_blogs.append(item)
            if len(candidate_blogs) >= num_reviews:
                break
        
        if not candidate_blogs:
            return 50.0, []

        total_strong_pos = 0
        total_strong_neg = 0
        total_sentiment_frequency = 0
        all_positive_judgments = []

        for blog_data in candidate_blogs:
            try:
                content, _ = await naver_supervisor._scrape_blog_content(blog_data["link"])
                if not content or "오류" in content or "찾을 수 없습니다" in content:
                    continue

                max_content_length = 30000
                if len(content) > max_content_length:
                    content = content[:max_content_length]

                final_state = app_llm_graph.invoke({
                    "original_text": content, "keyword": keyword, "title": blog_data["title"],
                    "log_details": False, "re_summarize_count": 0, "is_relevant": False
                })

                if not final_state or not final_state.get("is_relevant"):
                    continue

                judgments = final_state.get("final_judgments", [])
                if not judgments:
                    continue

                all_positive_judgments.extend([j for j in judgments if j["final_verdict"] == "긍정"])

                pos_count = sum(1 for res in judgments if res["final_verdict"] == "긍정")
                neg_count = sum(1 for res in judgments if res["final_verdict"] == "부정")
                strong_pos_count = sum(1 for res in judgments if res["final_verdict"] == "긍정" and res["score"] >= 1.0)
                strong_neg_count = sum(1 for res in judgments if res["final_verdict"] == "부정" and res["score"] < -1.0)

                total_strong_pos += strong_pos_count
                total_strong_neg += strong_neg_count
                total_sentiment_frequency += (pos_count + neg_count)

            except Exception as e:
                print(f"Error getting sentiment for '{keyword}': {e}")
                continue
        
        if total_sentiment_frequency == 0:
            return 50.0, []
            
        sentiment_score = ((total_strong_pos - total_strong_neg) / total_sentiment_frequency * 50 + 50)
        return sentiment_score, all_positive_judgments

    async def summarize_trend_reasons(keyword):
        if not keyword:
            return "키워드가 없어 트렌드 분석 불가"

        today = datetime.today()
        start_date = today - timedelta(days=90)
        trend_data = get_naver_trend(keyword, start_date, today)

        if not trend_data:
            return "트렌드 데이터 없음"

        df = pd.DataFrame(trend_data)
        data_str = df.to_string()

        llm = get_llm_client(temperature=0.2)
        prompt = f"""
        다음은 '{keyword}'에 대한 최근 90일간의 네이버 검색량 트렌드 데이터입니다.
        데이터(날짜별 관심도 비율)를 기반으로, 검색량 트렌드의 특징을 1~2줄로 요약해주세요.
        예: '최근 한 달간 관심도가 꾸준히 증가하고 있습니다.' 또는 '특정 날짜에 검색량이 급증하는 패턴을 보입니다.'

        데이터:
        {data_str}
        """
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error summarizing trend: {e}")
            return "트렌드 분석 중 오류 발생"

    async def summarize_sentiment_reasons(positive_judgments, keyword):
        if not positive_judgments:
            return "긍정 리뷰가 없어 분석 불가"

        sentences = [j['sentence'] for j in positive_judgments]
        sentences_str = "\n- ".join(sentences[:20])

        llm = get_llm_client(temperature=0.2)
        prompt = f"""
        다음은 '{keyword}'에 대한 블로그 리뷰에서 추출된 긍정적인 문장들입니다.
        이 문장들을 바탕으로, 사용자들이 주로 어떤 점을 칭찬하는지 핵심적인 이유 1~2가지를 요약해주세요.
        예: '깨끗한 시설과 다양한 먹거리에 대한 칭찬이 많습니다.' 또는 '아이들이 즐길 수 있는 체험 프로그램이 좋은 평가를 받았습니다.'

        긍정 문장 목록:
        - {sentences_str}
        """
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error summarizing sentiment: {e}")
            return "감성 분석 이유 요약 중 오류 발생"


    async def rank_facilities(facilities_list, num_reviews, top_n, progress=gr.Progress()):
        if not facilities_list:
            return [], "시설 목록이 비어있습니다.", gr.update(value=[]), "", gr.update(visible=False), gr.update(visible=False)

        async def process_facility(facility):
            title = facility.get('title', '')
            trend_score = get_trend_score(title)
            sentiment_score, positive_judgments = await get_sentiment_score(title, num_reviews)
            
            trend_reason, sentiment_reason = await asyncio.gather(
                summarize_trend_reasons(title),
                summarize_sentiment_reasons(positive_judgments, title)
            )
            
            facility['trend_score'] = round(trend_score, 2)
            facility['sentiment_score'] = round(sentiment_score, 2)
            facility['ranking_score'] = round((trend_score * 0.5) + (sentiment_score * 0.5), 2)
            facility['trend_reason'] = trend_reason
            facility['sentiment_reason'] = sentiment_reason
            return facility

        tasks = [process_facility(f) for f in facilities_list]
        ranked_facilities = []
        for task in progress.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="시설 점수 계산 중"):
            result_facility = await task
            ranked_facilities.append(result_facility)

        ranked_facilities.sort(key=lambda x: x.get('ranking_score', 0), reverse=True)
        gallery_output = [(item.get('firstimage', NO_IMAGE_URL) or NO_IMAGE_URL, f"점수: {item.get('ranking_score')} - {item['title']}") for item in ranked_facilities]
        report_update = await generate_full_report(ranked_facilities, top_n)
        
        return ranked_facilities, "순위 계산 완료!", gallery_output, report_update

    async def rank_courses(courses_list, num_reviews, top_n, progress=gr.Progress()):
        if not courses_list:
            return [], "코스 목록이 비어있습니다.", gr.update(value=[]), "", gr.update(visible=False), gr.update(visible=False)

        async def process_course(course):
            course_title = course.get('title', '')
            sub_points = course.get('sub_points', [])
            if not sub_points:
                course['ranking_score'] = 0
                return course

            sub_point_trend_scores = []
            sub_point_sentiment_scores = []
            all_positive_judgments = []

            for sub_point in sub_points:
                sub_title = sub_point.get('subname', '')
                if not sub_title:
                    continue
                
                trend_score = get_trend_score(sub_title)
                sentiment_score, positive_judgments = await get_sentiment_score(sub_title, num_reviews)
                sub_point_trend_scores.append(trend_score)
                sub_point_sentiment_scores.append(sentiment_score)
                all_positive_judgments.extend(positive_judgments)
            
            if not sub_point_trend_scores:
                course['ranking_score'] = 0
                course['trend_score'] = 0
                course['sentiment_score'] = 0
                course['trend_reason'] = "세부 코스 정보 부족"
                course['sentiment_reason'] = "세부 코스 정보 부족"
            else:
                avg_trend_score = sum(sub_point_trend_scores) / len(sub_point_trend_scores)
                avg_sentiment_score = sum(sub_point_sentiment_scores) / len(sub_point_sentiment_scores)
                course['trend_score'] = round(avg_trend_score, 2)
                course['sentiment_score'] = round(avg_sentiment_score, 2)
                course['ranking_score'] = round((avg_trend_score * 0.5) + (avg_sentiment_score * 0.5), 2)
                
                trend_reason, sentiment_reason = await asyncio.gather(
                    summarize_trend_reasons(course_title),
                    summarize_sentiment_reasons(all_positive_judgments, course_title)
                )
                course['trend_reason'] = trend_reason
                course['sentiment_reason'] = sentiment_reason
            
            return course

        tasks = [process_course(c) for c in courses_list]
        ranked_courses = []
        for task in progress.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="코스 점수 계산 중"):
            result_course = await task
            ranked_courses.append(result_course)

        ranked_courses.sort(key=lambda x: x.get('ranking_score', 0), reverse=True)
        gallery_output = [(item.get('firstimage', NO_IMAGE_URL) or NO_IMAGE_URL, f"점수: {item.get('ranking_score')} - {item['title']}") for item in ranked_courses]
        report_md, report_visible, header_visible = generate_full_report(ranked_courses, top_n)

        return ranked_courses, "순위 계산 완료!", gallery_output, report_md, report_visible, header_visible

    async def generate_full_report(ranked_list, top_n):
        top_n = int(top_n)
        if not ranked_list or not any(item.get('ranking_score', 0) > 0 for item in ranked_list[:top_n]):
            return gr.update(value="스코어링된 항목이 없습니다.", visible=True)

        # Generate the new comparative summary
        comparative_summary = await generate_comparative_summary(ranked_list[:top_n])

        report_parts = [f"## 🏆 최종 순위 분석\n{comparative_summary}", "---"]
        top_items = ranked_list[:top_n]
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for i, item in enumerate(top_items):
            if i >= len(medals):
                rank_indicator = f"{i+1}위"
            else:
                rank_indicator = medals[i]

            title = item.get('title', 'N/A')
            total_score = item.get('ranking_score', 'N/A')
            trend_score = item.get('trend_score', 'N/A')
            sentiment_score = item.get('sentiment_score', 'N/A')
            image_url = item.get('firstimage', NO_IMAGE_URL) or NO_IMAGE_URL
            trend_reason = item.get('trend_reason', '분석 정보 없음')
            sentiment_reason = item.get('sentiment_reason', '분석 정보 없음')

            report_parts.append(f"### {rank_indicator} {i+1}위: {title} (종합 점수: {total_score})")
            report_parts.append(f"![{title}]({image_url})\n")
            report_parts.append(f"- **트렌드 점수**: {trend_score}")
            report_parts.append(f"- **감성 점수**: {sentiment_score}")
            report_parts.append(f"**📈 트렌드 분석**: {trend_reason}")
            report_parts.append(f"**❤️ 감성 분석**: {sentiment_reason}")
            report_parts.append("---")

        report_md = "\n\n".join(report_parts)
        return gr.update(value=report_md, visible=True)

    async def generate_comparative_summary(ranked_list):
        llm = get_llm_client(temperature=0.3)
        
        # Format the data for the prompt
        data_for_prompt = []
        for item in ranked_list:
            data_for_prompt.append({
                "title": item.get('title'),
                "ranking_score": item.get('ranking_score'),
                "trend_score": item.get('trend_score'),
                "sentiment_score": item.get('sentiment_score'),
                "trend_reason": item.get('trend_reason'),
                "sentiment_reason": item.get('sentiment_reason')
            })
        
        prompt = f"""
        당신은 여행 추천 데이터 분석가입니다. 아래에 트렌드 점수와 감성 점수를 종합하여 순위를 매긴 관광지 목록이 있습니다. 
        이 데이터를 바탕으로, 1위가 왜 1위를 차지했는지 다른 순위와 비교하여 최종 결론을 2-3문장으로 요약해주세요.
        단순히 점수가 높다는 사실만 언급하지 말고, 각 점수의 의미(트렌드=화제성, 감성=실제 만족도)를 해석하고, 다른 장소와 비교하여 설득력 있는 이유를 제시해야 합니다.

        [데이터]
        {json.dumps(data_for_prompt, ensure_ascii=False, indent=2)}

        [요약 예시]
        "최종 분석 결과, A가 1위를 차지했습니다. 비록 B가 실제 방문객의 만족도(감성 점수)는 더 높았지만, A는 압도적인 화제성(트렌드 점수)과 준수한 만족도를 바탕으로 가장 균형 잡힌 추천 장소로 선정되었습니다. 반면 C는 높은 화제성에도 불구하고 긍정적인 피드백이 부족하여 순위가 밀렸습니다."
        """
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating comparative summary: {e}")
            return "최종 분석 요약 생성 중 오류가 발생했습니다."

    def update_sigungu(area):
        choices = ["전체"] + sorted(list(SIGUNGU_CODE_MAP.get(area, {}).keys())) if area != "전체" else ["전체"]
        return gr.update(choices=choices, value="전체")

    def update_medium_cat(main_cat):
        choices = ["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.get(main_cat, {}).keys())) if main_cat != "전체" else ["전체"]
        return gr.update(choices=choices, value="전체")

    def update_small_cat(main_cat, medium_cat):
        choices = ["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.get(main_cat, {}).get(medium_cat, {}).keys())) if main_cat != "전체" and medium_cat != "전체" else ["전체"]
        return gr.update(choices=choices, value="전체")

    area_dropdown.change(fn=update_sigungu, inputs=area_dropdown, outputs=sigungu_dropdown)
    main_cat_dropdown.change(fn=update_medium_cat, inputs=main_cat_dropdown, outputs=medium_cat_dropdown)
    medium_cat_dropdown.change(fn=update_small_cat, inputs=[main_cat_dropdown, medium_cat_dropdown], outputs=small_cat_dropdown)

    def run_search_and_display(area, sigungu, main_cat, medium_cat, small_cat):
        initial_state = {
            "search_type": "festival_search",
            "area": area, "sigungu": sigungu, "main_cat": main_cat,
            "medium_cat": medium_cat, "small_cat": small_cat
        }
        final_state = db_search_graph.invoke(initial_state)
        results = final_state.get("results", [])
        total_pages = math.ceil(len(results) / PAGE_SIZE)
        
        gallery, page_str_updated = display_page(results, 1)
        return results, gallery, f"1 / {total_pages}", gr.update(visible=len(results) > 0)

    def run_nearby_search(festival_details, radius_meters):
        if not festival_details or not festival_details.get('mapx') or not festival_details.get('mapy'):
            return [], [], gr.update(value="축제 좌표 정보가 없어 추천할 수 없습니다.", visible=True), [], [], gr.update(visible=False)

        initial_state = {
            "search_type": "nearby_search",
            "latitude": festival_details.get('mapy'),
            "longitude": festival_details.get('mapx'),
            "radius": radius_meters
        }
        final_state = db_search_graph.invoke(initial_state)
        
        facilities_recs = final_state.get("recommended_facilities", [])
        courses_recs = final_state.get("recommended_courses", [])

        if not facilities_recs and not courses_recs:
            return [], [], gr.update(value=f"{radius_meters}m 내에 추천할 장소가 없습니다.", visible=True), [], [], gr.update(visible=False)

        facility_gallery_output = [(item.get('firstimage', NO_IMAGE_URL) or NO_IMAGE_URL, item['title']) for item in facilities_recs]
        course_gallery_output = [(item.get('firstimage', NO_IMAGE_URL) or NO_IMAGE_URL, item['title']) for item in courses_recs]

        return facilities_recs, courses_recs, facility_gallery_output, course_gallery_output, gr.update(visible=False), gr.update(visible=True)

    search_btn.click(
        fn=run_search_and_display,
        inputs=[area_dropdown, sigungu_dropdown, main_cat_dropdown, medium_cat_dropdown, small_cat_dropdown],
        outputs=[results_state, festival_gallery, page_display, results_area]
    )

    prev_button.click(fn=lambda r, p: display_paginated_gallery(r, p, -1), inputs=[results_state, page_display], outputs=[festival_gallery, page_display])
    next_button.click(fn=lambda r, p: display_paginated_gallery(r, p, 1), inputs=[results_state, page_display], outputs=[festival_gallery, page_display])

    festival_gallery.select(
        fn=display_festival_details,
        inputs=[results_state, page_display],
        outputs=[
            festival_details_output, selected_festival_state, selected_festival_details_state
        ]
    ).then(
        fn=lambda: (
            gr.update(open=True), # details_accordion
            gr.update(visible=True), # recommend_accordion
            gr.update(visible=True), # naver_review_accordion
            gr.update(visible=True), # trend_accordion
            gr.update(visible=True), # wordcloud_accordion
            gr.update(visible=True)  # sentiment_accordion
        ),
        outputs=[
            details_accordion,
            recommend_accordion,
            naver_review_accordion,
            trend_accordion,
            wordcloud_accordion,
            sentiment_accordion
        ]
    )
    

    naver_search_btn.click(
        fn=get_naver_review_info,
        inputs=[selected_festival_state, num_reviews_naver_summary],
        outputs=[naver_review_output, naver_review_accordion]
    )

    trend_graph_btn.click(
        fn=generate_trend_graphs,
        inputs=[selected_festival_state],
        outputs=[trend_accordion, trend_status, trend_plot_yearly, trend_plot_event]
    )

    word_cloud_btn.click(
        fn=generate_word_cloud,
        inputs=[selected_festival_state, num_reviews_wordcloud],
        outputs=[wordcloud_accordion, wordcloud_status, wordcloud_plot]
    )

    run_sentiment_btn.click(
        fn=analyze_sentiment,
        inputs=[selected_festival_state, num_reviews_slider],
        outputs=[
            sentiment_accordion,
            sentiment_status,
            sentiment_negative_summary,
            sentiment_overall_chart,
            sentiment_summary,
            sentiment_overall_csv,
            sentiment_spring_chart,
            sentiment_summer_chart,
            sentiment_autumn_chart,
            sentiment_winter_chart,
            sentiment_spring_pos_wc,
            sentiment_spring_neg_wc,
            sentiment_summer_pos_wc,
            sentiment_summer_neg_wc,
            sentiment_autumn_pos_wc,
            sentiment_autumn_neg_wc,
            sentiment_winter_pos_wc,
            sentiment_winter_neg_wc,
            sentiment_df_output,
            blog_results_df_state,
            blog_judgments_state,
            sentiment_blog_page_num_input,
            sentiment_blog_total_pages_output,
            sentiment_blog_list_csv,
            sentiment_individual_summary,
            sentiment_individual_donut_chart,
            sentiment_individual_score_chart,
            sentiment_blog_detail_accordion,
        ]
    )

    rank_facilities_btn.click(
        fn=rank_facilities,
        inputs=[recommended_facilities_state, ranking_reviews_slider, ranking_top_n_slider],
        outputs=[
            recommended_facilities_state,
            recommend_status,
            recommend_facilities_gallery,
            facility_ranking_report
        ]
    )

    rank_courses_btn.click(
        fn=rank_courses,
        inputs=[recommended_courses_state, ranking_reviews_slider, ranking_top_n_slider],
        outputs=[
            recommended_courses_state,
            recommend_status,
            recommend_courses_gallery,
            course_ranking_report
        ]
    )

    recommend_btn.click(
        fn=run_nearby_search,
        inputs=[selected_festival_details_state, recommend_radius_slider],
        outputs=[
            recommended_facilities_state, 
            recommended_courses_state,
            recommend_facilities_gallery, 
            recommend_courses_gallery, 
            recommend_status,
            ranking_controls
        ]
    )

    # Event handlers for pagination and individual blog details within sentiment analysis tab
    sentiment_blog_page_num_input.change(
        fn=lambda df, page_num: change_page(df, page_num),
        inputs=[blog_results_df_state, sentiment_blog_page_num_input],
        outputs=[sentiment_df_output, sentiment_blog_page_num_input, sentiment_blog_total_pages_output]
    )

    def handle_df_select(evt: gr.SelectData, page_num: int, df: pd.DataFrame, judgments: list):
        BLOG_PAGE_SIZE = 10
        page_num = page_num or 1
        
        global_idx = (int(page_num) - 1) * BLOG_PAGE_SIZE + evt.index[0]

        if df is None or df.empty or judgments is None or not isinstance(judgments, list) or global_idx >= len(judgments):
            return gr.update(), gr.update(), gr.update(), gr.update()

        judgments_for_row = judgments[global_idx]
        
        if not isinstance(judgments_for_row, list):
            return gr.update(), gr.update(), gr.update(), gr.update()

        donut_chart = create_donut_chart(
            sum(1 for j in judgments_for_row if isinstance(j, dict) and j.get('final_verdict') == '긍정'),
            sum(1 for j in judgments_for_row if isinstance(j, dict) and j.get('final_verdict') == '부정'),
            f"{df.iloc[global_idx]['블로그 제목'][:20]}... 긍/부정 비율"
        )

        score_chart = create_sentence_score_bar_chart(
            judgments_for_row,
            f"{df.iloc[global_idx]['블로그 제목'][:20]}... 문장별 점수"
        )
        
        summary_text = df.iloc[global_idx]['긍/부정 문장 요약']

        return gr.update(value=donut_chart, visible=True), gr.update(value=score_chart, visible=True), gr.update(value=summary_text, visible=True), gr.update(open=True, visible=True)

    sentiment_df_output.select(
        fn=handle_df_select,
        inputs=[sentiment_blog_page_num_input, blog_results_df_state, blog_judgments_state],
        outputs=[
            sentiment_individual_donut_chart,
            sentiment_individual_score_chart,
            sentiment_individual_summary,
            sentiment_blog_detail_accordion
        ]
    )

    def display_recommend_details(evt: gr.SelectData, recommended_results):
        if not evt or not recommended_results:
            return gr.update(visible=False), gr.update(visible=False)

        selected_item = recommended_results[evt.index]
        
        details = []
        exclude_cols = [
            'type', 'id', 'contentid', 'contenttypeid', 'lDongRegnCd', 'lDongSignguCd', 
            'lclsSystm1', 'lclsSystm2', 'lclsSystm3', 'mlevel', 'cpyrhtDivCd',
            'areacode', 'cat1', 'cat2', 'cat3', 'createdtime', 'mapx', 'mapy', 
            'modifiedtime', 'sigungucode', 'sub_points' # also exclude the sub_points list itself from the main loop
        ]

        # Handle main info for both facilities and courses
        for key, value in selected_item.items():
            if key in exclude_cols:
                continue
            if value is not None and str(value).strip() != '':
                display_key = COLUMN_TRANSLATIONS.get(key, key)
                details.append(f"**{display_key}**: {value}")

        # If it's a course, handle the sub_points specially
        if 'sub_points' in selected_item and selected_item['sub_points']:
            sub_points = selected_item['sub_points']
            
            all_subnames = [sp.get('subname') for sp in sub_points if sp.get('subname')]
            if all_subnames:
                # Remove duplicates while preserving order
                unique_subnames = list(dict.fromkeys(all_subnames))
                details.append(f"**세부 코스명**: {', '.join(unique_subnames)}")

            all_overviews = [sp.get('subdetailoverview') for sp in sub_points if sp.get('subdetailoverview')]
            if all_overviews:
                overview_list_str = [f"{i+1}. {desc}" for i, desc in enumerate(all_overviews)]
                details.append(f"**세부 코스개요**:\n" + "\n".join(overview_list_str))

        details_text = "\n\n".join(details)

        return gr.update(value=details_text), gr.update(visible=True, open=True)

    recommend_facilities_gallery.select(
        fn=display_recommend_details,
        inputs=[recommended_facilities_state],
        outputs=[recommend_details_output, recommend_details_accordion]
    )

    recommend_courses_gallery.select(
        fn=display_recommend_details,
        inputs=[recommended_courses_state],
        outputs=[recommend_details_output, recommend_details_accordion]
    )

if __name__ == "__main__":

    ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()
    
    demo.launch(allowed_paths=["assets"])