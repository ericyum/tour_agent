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
from src.infrastructure.config.settings import setup_environment, get_google_api_key
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
from src.infrastructure.persistence.database import load_data_to_db
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
from src.application.core.utils import get_season, save_df_to_csv, summarize_negative_feedback, create_driver, change_page

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
NO_IMAGE_URL = "https://placehold.co/300x200?text=No+Image"
PAGE_SIZE = 16

CUSTOM_CSS = """#gallery .thumbnail-item { max-width: 250px !important; min-width: 200px !important; flex-grow: 1 !important; }"""

# --- Data Loading and Mappings ---

AREA_CODE_MAP = {
    "서울": 1, "인천": 2, "대전": 3, "대구": 4, "광주": 5, "부산": 6, "울산": 7, 
    "세종특별자치시": 8, "경기도": 31, "강원특별자치도": 32, "충청북도": 33, 
    "충청남도": 34, "경상북도": 35, "경상남도": 36, "전북특별자치도": 37, 
    "전라남도": 38, "제주특별자치도": 39
}

SIGUNGU_CODE_MAP = {
    "서울": { "강남구": 1, "강동구": 2, "강북구": 3, "강서구": 4, "관악구": 5, "광진구": 6, "구로구": 7, "금천구": 8, "노원구": 9, "도봉구": 10, "동대문구": 11, "동작구": 12, "마포구": 13, "서대문구": 14, "서초구": 15, "성동구": 16, "성북구": 17, "송파구": 18, "양천구": 19, "영등포구": 20, "용산구": 21, "은평구": 22, "종로구": 23, "중구": 24, "중랑구": 25 },
    "인천": { "강화군": 1, "계양구": 2, "미추홀구": 3, "남동구": 4, "동구": 5, "부평구": 6, "서구": 7, "연수구": 8, "옹진군": 9, "중구": 10 },
    "대전": { "대덕구": 1, "동구": 2, "서구": 3, "유성구": 4, "중구": 5 },
    "대구": { "남구": 1, "달서구": 2, "달성군": 3, "동구": 4, "북구": 5, "서구": 6, "수성구": 7, "중구": 8, "군위군": 9 },
    "광주": { "광산구": 1, "남구": 2, "동구": 3, "북구": 4, "서구": 5 },
    "부산": { "강서구": 1, "금정구": 2, "기장군": 3, "남구": 4, "동구": 5, "동래구": 6, "부산진구": 7, "북구": 8, "사상구": 9, "사하구": 10, "서구": 11, "수영구": 12, "연제구": 13, "영도구": 14, "중구": 15, "해운대구": 16 },
    "울산": { "중구": 1, "남구": 2, "동구": 3, "북구": 4, "울주군": 5 },
    "세종특별자치시": { "세종특별자치시": 1 },
    "경기도": { "가평군": 1, "고양시": 2, "과천시": 3, "광명시": 4, "광주시": 5, "구리시": 6, "군포시": 7, "김포시": 8, "남양주시": 9, "동두천시": 10, "부천시": 11, "성남시": 12, "수원시": 13, "시흥시": 14, "안산시": 15, "안성시": 16, "안양시": 17, "양주시": 18, "양평군": 19, "여주시": 20, "연천군": 21, "오산시": 22, "용인시": 23, "의왕시": 24, "의정부시": 25, "이천시": 26, "파주시": 27, "평택시": 28, "포천시": 29, "하남시": 30, "화성시": 31 },
    "강원특별자치도": { "강릉시": 1, "고성군": 2, "동해시": 3, "삼척시": 4, "속초시": 5, "양구군": 6, "양양군": 7, "영월군": 8, "원주시": 9, "인제군": 10, "정선군": 11, "철원군": 12, "춘천시": 13, "태백시": 14, "평창군": 15, "홍천군": 16, "화천군": 17, "횡성군": 18 },
    "충청북도": { "괴산군": 1, "단양군": 2, "보은군": 3, "영동군": 4, "옥천군": 5, "음성군": 6, "제천시": 7, "증평군": 8, "진천군": 9, "청주시": 10, "충주시": 11 },
    "충청남도": { "계룡시": 1, "공주시": 2, "금산군": 3, "논산시": 4, "당진시": 5, "보령시": 6, "부여군": 7, "서산시": 8, "서천군": 9, "아산시": 10, "예산군": 11, "천안시": 12, "청양군": 13, "태안군": 14, "홍성군": 15 },
    "경상북도": { "경산시": 1, "경주시": 2, "고령군": 3, "구미시": 4, "김천시": 5, "문경시": 6, "봉화군": 7, "상주시": 8, "성주군": 9, "안동시": 10, "영덕군": 11, "영양군": 12, "영주시": 13, "영천시": 14, "예천군": 15, "울릉군": 16, "울진군": 17, "의성군": 18, "청도군": 19, "청송군": 20, "칠곡군": 21, "포항시": 22 },
    "경상남도": { "거제시": 1, "거창군": 2, "고성군": 3, "김해시": 4, "남해군": 5, "밀양시": 6, "사천시": 7, "산청군": 8, "양산시": 9, "의령군": 10, "진주시": 11, "창녕군": 12, "창원시": 13, "통영시": 14, "하동군": 15, "함안군": 16, "함양군": 17, "합천군": 18 },
    "전북특별자치도": { "고창군": 1, "군산시": 2, "김제시": 3, "남원시": 4, "무주군": 5, "부안군": 6, "순창군": 7, "완주군": 8, "익산시": 9, "임실군": 10, "장수군": 11, "전주시": 12, "정읍시": 13, "진안군": 14 },
    "전라남도": { "강진군": 1, "고흥군": 2, "곡성군": 3, "광양시": 4, "구례군": 5, "나주시": 6, "담양군": 7, "목포시": 8, "무안군": 9, "보성군": 10, "순천시": 11, "신안군": 12, "여수시": 13, "영광군": 14, "영암군": 15, "완도군": 16, "장성군": 17, "장흥군": 18, "진도군": 19, "함평군": 20, "해남군": 21, "화순군": 22 },
    "제주특별자치도": { "제주시": 1, "서귀포시": 2 }
}

CATEGORY_TO_ICON_MAP = {
    "도시와지역이벤트": "mask_city",
    "문화예술과공연": "mask_happy",
    "산업과지식": "mask_industry",
    "자연과계절": "mask_nature",
    "전통과역사": "mask_tradition",
    "지역특산물과음식": "mask_food",
    "체험과레저": "mask_sport"
}

CAT_NAME_TO_CODE = {'main': {}, 'medium': {}, 'small': {}}
TITLE_TO_CAT_NAMES = {}

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

# --- Core Logic Functions ---

def search_festivals(area, sigungu, main_cat, medium_cat, small_cat):
    # Step 1: Primary filtering by location from DB
    loc_where_clauses = []
    loc_params = []

    if area and area != "전체":
        area_code = AREA_CODE_MAP.get(area)
        if area_code:
            loc_where_clauses.append("areacode = ?")
            loc_params.append(area_code)
            if sigungu and sigungu != "전체":
                sigungu_code = SIGUNGU_CODE_MAP.get(area, {}).get(sigungu)
                if sigungu_code:
                    loc_where_clauses.append("sigungucode = ?")
                    loc_params.append(sigungu_code)

    db_path = os.path.join(script_dir, "tour.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT title, firstimage FROM festivals"
    if loc_where_clauses:
        query += " WHERE " + " AND ".join(loc_where_clauses)
        cursor.execute(query, loc_params)
    else:
        cursor.execute(query)
        
    db_results = cursor.fetchall()
    conn.close()

    # Step 2: Secondary filtering by category using pre-built map from JSON
    is_cat_filtered = main_cat != "전체" or medium_cat != "전체" or small_cat != "전체"
    if not is_cat_filtered:
        final_results = db_results
    else:
        final_results = []
        for row in db_results:
            title = row[0]
            # TITLE_TO_CAT_NAMES maps a title to its (main, medium, small) category names
            cat_names = TITLE_TO_CAT_NAMES.get(title)
            if not cat_names:
                continue # This festival title is not in our JSON category mapping

            # Check if the festival's categories match the filter
            main_match = (main_cat == "전체" or main_cat == cat_names[0])
            medium_match = (medium_cat == "전체" or medium_cat == cat_names[1])
            small_match = (small_cat == "전체" or small_cat == cat_names[2])

            if main_match and medium_match and small_match:
                final_results.append(row)

    results = sorted([(row[0], row[1] or NO_IMAGE_URL) for row in final_results])
    total_pages = math.ceil(len(results) / PAGE_SIZE)
    return results, 1, f"1 / {total_pages}", gr.update(visible=len(results) > 0)

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

    if not details: return "정보를 찾을 수 없습니다.", "", "", "", None

    basic_info = f"**축제명**: {details.get('title', 'N/A')}\n**주소**: {details.get('addr1', 'N/A')}\n**전화번호**: {details.get('tel', 'N/A')}"
    detailed_info = f"**시작일**: {details.get('eventstartdate', 'N/A')}\n**종료일**: {details.get('eventenddate', 'N/A')}\n**행사 장소**: {details.get('eventplace', 'N/A')}"
    overview_text = details.get('overview', '정보 없음')
    event_content_text = details.get('행사내용', '정보 없음')
    return basic_info, detailed_info, overview_text, event_content_text, selected_title



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


# --- Gradio Interface ---

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# 축제 정보 검색 에이전트")

    results_state = gr.State([])
    page_state = gr.State(1)
    selected_festival_state = gr.State()

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
        basic_info_output = gr.Markdown()
        detailed_info_output = gr.Markdown()
        overview_output = gr.Markdown()
        content_output = gr.Markdown()

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
        results, page, page_str, is_visible = search_festivals(area, sigungu, main_cat, medium_cat, small_cat)
        gallery, page_str_updated = display_page(results, page)
        return results, gallery, page_str_updated, is_visible

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
            basic_info_output, detailed_info_output, overview_output, content_output, selected_festival_state
        ]
    ).then(
        fn=lambda: (
            gr.update(open=True), # details_accordion
            gr.update(visible=True), # naver_review_accordion
            gr.update(visible=True), # trend_accordion
            gr.update(visible=True), # wordcloud_accordion
            gr.update(visible=True)  # sentiment_accordion
        ),
        outputs=[
            details_accordion,
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

if __name__ == "__main__":
    print("\n[app.py] Forcing database creation/update...")
    load_data_to_db()
    print("[app.py] Database creation/update complete.")
    ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()
    
    demo.launch(allowed_paths=["assets"])