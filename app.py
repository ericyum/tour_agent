import shutil
import requests
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
from src.infrastructure.config.settings import (
    setup_environment,
    get_google_api_key,
    Settings,
)

setup_environment()

# --- Matplotlib Backend Setup ---
import matplotlib

matplotlib.use("Agg")

# --- Visualization and NLP Libraries (Assumed to be installed) ---
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from konlpy.tag import Okt
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print(
        "Warning: `wordcloud`, `matplotlib`, `konlpy`, `Pillow`, or `numpy` libraries are not installed. Visual analysis will not work."
    )
    WordCloud, plt, Okt, Image, ImageDraw, ImageFont, np = [None] * 7

# --- Custom Module Imports ---
from src.infrastructure.persistence.database import get_db_connection, init_db
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor
from src.infrastructure.external_services.naver_search.naver_review_api import (
    get_naver_trend,
    search_naver_blog,
)

# --- New LLM-related Imports (for sentiment analysis only) ---
from src.application.core.state import LLMGraphState
from src.domain.knowledge_base import knowledge_base
from src.infrastructure.llm_client import get_llm_client
from src.infrastructure.dynamic_scorer import SimpleScorer
from src.application.agents.common.content_validator import agent_content_validator
from src.application.agents.common.llm_summarizer import agent_llm_summarizer
from src.application.agents.common.rule_scorer import agent_rule_scorer_on_summary
from src.application.core.graph import app_llm_graph
from src.infrastructure.reporting.charts import (
    create_donut_chart,
    create_stacked_bar_chart,
    create_sentence_score_bar_chart,
)
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.application.core.utils import (
    get_season,
    save_df_to_csv,
    summarize_negative_feedback,
    create_driver,
    change_page,
    get_logger,
    haversine,
)

# --- Selenium Imports (for sentiment analysis only) ---
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import traceback

# --- NLTK Setup ---
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# --- Constants and Setup ---
script_dir = os.path.dirname(__file__)


CUSTOM_CSS = """#gallery .thumbnail-item { max-width: 250px !important; min-width: 200px !important; flex-grow: 1 !important; }"""

from src.application.core.constants import (
    AREA_CODE_MAP,
    SIGUNGU_CODE_MAP,
    CATEGORY_TO_ICON_MAP,
    NO_IMAGE_URL,
    PAGE_SIZE,
)

# --- Data Loading and Mappings ---

CAT_NAME_TO_CODE = {"main": {}, "medium": {}, "small": {}}
TITLE_TO_CAT_NAMES = {}

COLUMN_TRANSLATIONS = {
    "addr1": "주소",
    "addr2": "상세주소",
    "tel": "전화번호",
    "title": "제목",
    "zipcode": "우편번호",
    "telname": "연락처 이름",
    "homepage": "홈페이지",
    "overview": "개요",
    "sponsor1": "주최자",
    "sponsor1tel": "주최자 연락처",
    "eventenddate": "행사 종료일",
    "playtime": "공연 시간",
    "eventplace": "행사 장소",
    "eventstartdate": "행사 시작일",
    "usetimefestival": "이용 요금",
    "sponsor2": "후원사",
    "progresstype": "진행 상태",
    "festivaltype": "축제 유형",
    "sponsor2tel": "후원사 연락처",
    "agelimit": "연령 제한",
    "spendtimefestival": "관람 소요시간",
    "festivalgrade": "축제 등급",
    "eventhomepage": "행사 홈페이지",
    "subevent": "부대 행사",
    "program": "행사 프로그램",
    "discountinfofestival": "할인 정보",
    "placeinfo": "행사장 위치 안내",
    "bookingplace": "예매처",
    "usefee": "이용 요금",
    "infocenterculture": "문의 및 안내",
    "usetimeculture": "이용 시간",
    "restdateculture": "쉬는 날",
    "parkingfee": "주차 요금",
    "parkingculture": "주차 시설",
    "chkcreditcardculture": "신용카드 가능 정보",
    "chkbabycarriageculture": "유모차 대여 정보",
    "spendtime": "관람 소요시간",
    "accomcountculture": "수용인원",
    "scale": "규모",
    "chkpetculture": "반려동물 동반 가능 정보",
    "discountinfo": "할인 정보",
    "distance": "총 거리",
    "schedule": "코스 일정",
    "taketime": "총 소요시간",
    "theme": "코스 테마",
    "subnum": "세부 코스 번호",
    "subname": "세부 코스명",
    "subdetailoverview": "세부 코스 개요",
    "firstimage": "대표 이미지",
    "firstimage2": "추가 이미지",
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
                with open(
                    os.path.join(festivals_dir, filename), "r", encoding="utf-8"
                ) as f:
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
    cursor.execute(
        "SELECT title, cat1, cat2, cat3 FROM festivals WHERE cat1 IS NOT NULL AND cat2 IS NOT NULL AND cat3 IS NOT NULL"
    )
    db_festivals = cursor.fetchall()
    conn.close()

    for row in db_festivals:
        title, code1, code2, code3 = row
        if title in TITLE_TO_CAT_NAMES:
            name1, name2, name3 = TITLE_TO_CAT_NAMES[title]
            CAT_NAME_TO_CODE["main"][name1] = code1
            CAT_NAME_TO_CODE["medium"][name2] = code2
            CAT_NAME_TO_CODE["small"][name3] = code3

    print("[app.py] Category name-to-code maps created.")
    return all_categories


ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()
naver_supervisor = NaverReviewSupervisor()
okt = Okt() if Okt else None


def get_korean_font():
    if plt is None:
        return None
    try:
        font_path = font_manager.findfont(
            font_manager.FontProperties(family="Malgun Gothic")
        )
        if os.path.exists(font_path):
            return font_path
    except:
        pass
    font_list = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in font_list:
        if (
            "gothic" in font.lower()
            or "gulim" in font.lower()
            or "apple" in font.lower()
        ):
            return font
    print("Warning: Korean font not found. Visualization text may be broken.")
    return None


KOREAN_FONT_PATH = get_korean_font()

from src.application.use_cases.analysis_use_case import AnalysisUseCase

analysis_use_case = AnalysisUseCase(
    naver_supervisor=naver_supervisor,
    font_path=KOREAN_FONT_PATH,
    title_to_cat_map=TITLE_TO_CAT_NAMES,
    cat_to_icon_map=CATEGORY_TO_ICON_MAP,
    script_dir=script_dir
)

from src.application.use_cases.sentiment_analysis_use_case import SentimentAnalysisUseCase

sentiment_analysis_use_case = SentimentAnalysisUseCase(
    naver_supervisor=naver_supervisor,
    script_dir=script_dir
)

from src.application.use_cases.ranking_use_case import RankingUseCase

ranking_use_case = RankingUseCase(
    naver_supervisor=naver_supervisor
)


from src.application.supervisors.db_search_supervisor import db_search_graph


def display_page(results, page):
    page = int(page)
    start_index = (page - 1) * PAGE_SIZE
    end_index = start_index + PAGE_SIZE
    page_results = results[start_index:end_index]
    
    gallery_output = []
    for item in page_results:
        image = item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL
        title = item.get("title", "제목 없음")
        if "ranking_score" in item:
            title = f"점수: {item.get('ranking_score', 'N/A')} - {title}"
        gallery_output.append((image, title))

    total_pages = math.ceil(len(results) / PAGE_SIZE)
    page_buttons_updates = update_pagination_buttons(page, total_pages)
    return gallery_output, f"{page} / {total_pages}", *page_buttons_updates


def display_paginated_gallery(results, page_str, direction):
    page = int(page_str.split('/')[0].strip())
    total_pages = math.ceil(len(results) / PAGE_SIZE)
    new_page = page + direction
    if 1 <= new_page <= total_pages:
        gallery_output, page_display_str, *page_buttons_updates = display_page(results, new_page)
        return gallery_output, page_display_str, *page_buttons_updates
    else:
        # If out of bounds, return the current page's content
        # This means if new_page < 1, return page 1. If new_page > total_pages, return total_pages.
        if new_page < 1:
            current_page_to_display = 1
        else: # new_page > total_pages
            current_page_to_display = total_pages
        
        gallery_output, page_display_str, *page_buttons_updates = display_page(results, current_page_to_display)
        return gallery_output, page_display_str, *page_buttons_updates

def update_pagination_buttons(current_page, total_pages):
    max_buttons = 5
    start_page = max(1, current_page - (max_buttons // 2))
    end_page = min(total_pages, start_page + max_buttons - 1)

    if end_page - start_page + 1 < max_buttons:
        start_page = max(1, end_page - max_buttons + 1)

    button_updates = []
    for i in range(1, max_buttons + 1):
        page_num = start_page + i - 1
        if page_num <= end_page:
            button_updates.append(gr.update(value=str(page_num), visible=True, variant="primary" if page_num == current_page else "secondary"))
        else:
            button_updates.append(gr.update(visible=False, value="")) # Set value to empty string when invisible
    
    return button_updates # This will be a list of 5 gr.update objects


from src.application.services.festival_service import get_festival_details_by_title


def display_festival_details(evt: gr.SelectData, results, page_str):
    page = int(page_str.split("/")[0].strip())
    global_index = (page - 1) * PAGE_SIZE + evt.index
    
    selected_item = results[global_index]
    # The title in the results dict is clean, even if the gallery shows a score
    original_title = selected_item.get("title", "")
    
    details = get_festival_details_by_title(original_title)

    if not details:
        return gr.update(value="정보를 찾을 수 없습니다."), None, None

    details_list = []
    # Add the new scores to the details view if they exist
    score_keys = {
        "ranking_score": "종합 순위 점수",
        "time_score": "시기성 점수",
        "sentiment_score": "만족도 점수",
        "quarterly_trend_score": "최근 화제성(90일)",
        "yearly_trend_score": "연간 꾸준함(365일)"
    }
    for key, display_name in score_keys.items():
        if key in selected_item:
            details_list.append(f"**{display_name}**: {selected_item[key]}")

    if details_list: # Add a separator if scores were added
        details_list.append("---")

    exclude_cols = [
        "id", "contentid", "contenttypeid", "lDongRegnCd", "lDongSignguCd",
        "lclsSystm1", "lclsSystm2", "lclsSystm3", "mlevel", "cpyrhtDivCd",
        "areacode", "cat1", "cat2", "cat3", "createdtime", "mapx", "mapy",
        "modifiedtime", "sigungucode", "ranking_score", "time_score", 
        "sentiment_score", "quarterly_trend_score", "yearly_trend_score"
    ]
    for key, value in details.items():
        if key in exclude_cols:
            continue
        if value is not None and str(value).strip() != "":
            display_key = COLUMN_TRANSLATIONS.get(key, key)
            details_list.append(f"**{display_key}**: {value}")

    details_text = "\n\n".join(details_list)

    return gr.update(value=details_text), details.get("title"), details


async def get_naver_review_info(festival_name, num_reviews):
    if not festival_name:
        yield gr.update(value="먼저 축제를 선택해주세요.", visible=True), gr.update(
            visible=False
        )
        return

    yield gr.update(
        value=f"{festival_name} 후기 검색 중... ({num_reviews}개)", visible=True
    ), gr.update(visible=True, open=True)
    summary, _ = await naver_supervisor.get_review_summary_and_tips(
        festival_name, num_reviews=num_reviews
    )
    yield gr.update(value=summary, visible=True), gr.update(visible=True, open=True)


async def handle_generate_trend_graphs(festival_name):
    yield gr.update(visible=True, open=True), gr.update(value="트렌드 그래프 생성 중...", visible=True), None, None
    trend_image_yearly, trend_image_event, status_message = await analysis_use_case.generate_trend_graphs(festival_name)
    if trend_image_yearly is None and trend_image_event is None:
         yield gr.update(visible=True, open=True), gr.update(value=status_message, visible=True), None, None
    else:
        yield gr.update(visible=True, open=True), gr.update(visible=False), trend_image_yearly, trend_image_event


async def handle_generate_word_cloud(festival_name, num_reviews):
    yield gr.update(visible=True, open=True), gr.update(value=f"워드 클라우드 생성 중... ({num_reviews}개)", visible=True), None
    wc_image, status_message = await analysis_use_case.generate_word_cloud(festival_name, num_reviews)
    if wc_image is None:
        yield gr.update(visible=True, open=True), gr.update(value=status_message, visible=True), None
    else:
        yield gr.update(visible=True, open=True), gr.update(visible=False), wc_image


async def handle_scrape_images(festival_name, num_blogs):
    local_image_paths, urls = await analysis_use_case.scrape_festival_images(festival_name, num_blogs)
    return gr.update(value=local_image_paths, visible=True), gr.update(visible=True, open=True), urls


async def handle_analyze_sentiment(festival_name, num_reviews):
    outputs_to_clear = [
        gr.update(open=True), gr.update(value=""), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), None, None, None, 1, "/ 1",
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False, open=False),
    ]
    
    if not festival_name:
        outputs_to_clear[1] = gr.update(value="축제를 선택해주세요.")
        yield tuple(outputs_to_clear)
        return

    try:
        outputs_to_clear[1] = gr.update(value="블로그 검색 및 분석 중...")
        yield tuple(outputs_to_clear)

        result = await sentiment_analysis_use_case.analyze_sentiment(festival_name, num_reviews)

        initial_page_df, current_page, total_pages_str = change_page(result["blog_df"], 1)

        seasonal_charts = result["seasonal_charts"]
        seasonal_pos_wc = result["seasonal_pos_wc_paths"]
        seasonal_neg_wc = result["seasonal_neg_wc_paths"]

        yield (
            gr.update(visible=True, open=True), # sentiment_accordion
            "분석 완료", # sentiment_status
            gr.update(value=result["neg_summary_text"], visible=bool(result["neg_summary_text"])), # sentiment_negative_summary
            gr.update(value=result["overall_chart"], visible=True), # sentiment_overall_chart
            gr.update(value=result["overall_summary_text"], visible=True), # sentiment_summary
            gr.update(value=result["summary_csv_path"], visible=result["summary_csv_path"] is not None), # sentiment_overall_csv
            
            gr.update(value=seasonal_charts.get("봄"), visible="봄" in seasonal_charts),
            gr.update(value=seasonal_charts.get("여름"), visible="여름" in seasonal_charts),
            gr.update(value=seasonal_charts.get("가을"), visible="가을" in seasonal_charts),
            gr.update(value=seasonal_charts.get("겨울"), visible="겨울" in seasonal_charts),

            gr.update(value=seasonal_pos_wc.get("봄"), visible="봄" in seasonal_pos_wc),
            gr.update(value=seasonal_neg_wc.get("봄"), visible="봄" in seasonal_neg_wc),
            gr.update(value=seasonal_pos_wc.get("여름"), visible="여름" in seasonal_pos_wc),
            gr.update(value=seasonal_neg_wc.get("여름"), visible="여름" in seasonal_neg_wc),
            gr.update(value=seasonal_pos_wc.get("가을"), visible="가을" in seasonal_pos_wc),
            gr.update(value=seasonal_neg_wc.get("가을"), visible="가을" in seasonal_neg_wc),
            gr.update(value=seasonal_pos_wc.get("겨울"), visible="겨울" in seasonal_pos_wc),
            gr.update(value=seasonal_neg_wc.get("겨울"), visible="겨울" in seasonal_neg_wc),
            
            initial_page_df, # sentiment_df_output
            result["blog_df"], # blog_results_df_state
            result["blog_judgments_list"], # blog_judgments_state
            current_page, # sentiment_blog_page_num_input
            total_pages_str, # sentiment_blog_total_pages_output
            gr.update(value=result["blog_list_csv_path"], visible=result["blog_list_csv_path"] is not None), # sentiment_blog_list_csv
            
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, open=False),
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
    total_pages_state = gr.State(1) # New state to store total pages
    recommended_facilities_state = gr.State([])
    recommended_courses_state = gr.State([])

    with gr.Group():
        with gr.Row():
            area_dropdown = gr.Dropdown(
                label="시/도",
                choices=["전체"] + sorted(list(AREA_CODE_MAP.keys())),
                value="전체",
                interactive=True,
            )
            sigungu_dropdown = gr.Dropdown(
                label="시/군/구", choices=["전체"], value="전체", interactive=True
            )
        with gr.Row():
            main_cat_dropdown = gr.Dropdown(
                label="대분류",
                choices=["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.keys())),
                value="전체",
                interactive=True,
            )
            medium_cat_dropdown = gr.Dropdown(
                label="중분류", choices=["전체"], value="전체", interactive=True
            )
            small_cat_dropdown = gr.Dropdown(
                label="소분류", choices=["전체"], value="전체", interactive=True
            )
            status_radio = gr.Radio(
                label="진행 상태",
                choices=["전체", "축제 진행중", "진행 예정", "종료된 축제"],
                value="전체",
                interactive=True,
            )
    with gr.Row():
        search_btn = gr.Button("검색", variant="primary", scale=1)
        rank_festivals_btn = gr.Button("축제 순위 보기", scale=1)
        num_reviews_festival_ranking = gr.Slider(
            minimum=1,
            maximum=50,
            value=10,
            step=1,
            label="축제 순위용 리뷰 수",
            interactive=True,
            scale=2,
        )
        festival_ranking_top_n_slider = gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="표시할 순위 수",
            interactive=True,
            scale=1,
        )

    festival_ranking_report = gr.Markdown(visible=False)

    with gr.Column(visible=False) as results_area:
        festival_gallery = gr.Gallery(
            label="축제 목록",
            show_label=False,
            elem_id="gallery",
            columns=4,
            height="auto",
            object_fit="contain",
        )
        with gr.Row(variant="panel"):
            first_page_button = gr.Button("<<", size="sm")
            prev_button = gr.Button("◀ 이전", size="sm")
            page_button_1 = gr.Button("1", visible=False, size="sm")
            page_button_2 = gr.Button("2", visible=False, size="sm")
            page_button_3 = gr.Button("3", visible=False, size="sm")
            page_button_4 = gr.Button("4", visible=False, size="sm")
            page_button_5 = gr.Button("5", visible=False, size="sm")
            next_button = gr.Button("다음 ▶", size="sm")
            last_page_button = gr.Button(">>")
        with gr.Row():
            page_input = gr.Number(label="페이지 이동", value=1, interactive=True, scale=1)
            page_display = gr.Textbox(value="1 / 1", label="현재 페이지", interactive=False, container=False, scale=1)

    with gr.Accordion("축제 상세 정보", open=False) as details_accordion:
        festival_details_output = gr.Markdown()
        with gr.Row():
            num_blogs_for_images = gr.Slider(
                minimum=1,
                maximum=100,
                value=5,
                step=1,
                label="이미지 수집 대상 블로그 수",
                interactive=True,
            )
            image_collect_button = gr.Button("이미지 수집하기")

    with gr.Accordion(
        "이미지 모아보기", open=False, visible=False
    ) as image_gallery_accordion:
        image_gallery = gr.Gallery(
            label="수집된 이미지",
            show_label=False,
            columns=4,
            height="auto",
            object_fit="contain",
        )
        scraped_urls_output = gr.Textbox(label="Scraped Image URLs")

    with gr.Accordion(
        "좌표 기반 추천", open=False, visible=False
    ) as recommend_accordion:
        with gr.Row():
            recommend_radius_slider = gr.Slider(
                minimum=100,
                maximum=20000,
                value=5000,
                step=100,
                label="반경 (미터)",
                interactive=True,
            )
            recommend_btn = gr.Button("추천 받기", variant="primary")

        with gr.Row(visible=False) as ranking_controls:  # Hide until there are results
            ranking_reviews_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="순위용 리뷰 수",
                interactive=True,
            )
            ranking_top_n_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="표시할 순위 수",
                interactive=True,
            )
            rank_facilities_btn = gr.Button("관광 시설 순위 매기기")
            rank_courses_btn = gr.Button("관광 코스 순위 매기기")

        recommend_status = gr.Textbox(label="상태", interactive=False, visible=False)
        gr.Markdown("### 추천 관광 시설")
        recommend_facilities_gallery = gr.Gallery(
            label="추천 관광 시설",
            show_label=False,
            elem_id="recommend_facilities_gallery",
            columns=4,
            height="auto",
            object_fit="contain",
        )
        facility_ranking_report = gr.Markdown(visible=False)
        gr.Markdown("### 추천 관광 코스")
        recommend_courses_gallery = gr.Gallery(
            label="추천 관광 코스",
            show_label=False,
            elem_id="recommend_courses_gallery",
            columns=4,
            height="auto",
            object_fit="contain",
        )
        course_ranking_report = gr.Markdown(visible=False)
        with gr.Accordion(
            "추천 장소 상세 정보", open=False, visible=False
        ) as recommend_details_accordion:
            recommend_details_output = gr.Markdown()

    with gr.Accordion(
        "Naver 후기 요약 및 꿀팁", open=False, visible=False
    ) as naver_review_accordion:
        with gr.Row():
            num_reviews_naver_summary = gr.Slider(
                minimum=1,
                maximum=100,
                value=5,
                step=1,
                label="분석할 후기 수 (네이버 요약)",
                interactive=True,
            )
            naver_search_btn = gr.Button("네이버 후기 요약 검색", variant="primary")
        naver_review_output = gr.Markdown()

    with gr.Accordion(
        "검색량 트렌드 그래프", open=False, visible=False
    ) as trend_accordion:
        trend_graph_btn = gr.Button("트렌드 그래프 생성", variant="primary")
        trend_status = gr.Textbox(label="상태", interactive=False)
        with gr.Row():
            trend_plot_yearly = gr.Image(label="최근 1년 검색량 트렌드")
            trend_plot_event = gr.Image(label="축제 기간 중심 트렌드")

    with gr.Accordion(
        "워드 클라우드", open=False, visible=False
    ) as wordcloud_accordion:
        with gr.Row():
            num_reviews_wordcloud = gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="분석할 후기 수 (워드클라우드)",
                interactive=True,
            )
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
            num_reviews_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=10,
                step=1,
                label="분석할 후기 수",
                interactive=True,
            )
            run_sentiment_btn = gr.Button("감성 분석 실행", variant="primary")

        sentiment_status = gr.Textbox(label="분석 상태", interactive=False)

        with gr.Accordion("종합 분석 결과", open=True):
            sentiment_summary = gr.Markdown(label="종합 분석 상세", visible=False)
            sentiment_overall_csv = gr.File(
                label="종합 분석 (CSV) 다운로드", visible=False
            )
            sentiment_overall_chart = gr.Plot(label="전체 후기 요약", visible=False)
            sentiment_negative_summary = gr.Markdown(
                label="주요 불만 사항 요약", visible=False
            )

            with gr.Accordion("계절별 상세 분석", open=False):
                with gr.Row():
                    sentiment_spring_chart = gr.Plot(
                        label="봄 시즌", visible=False, scale=1
                    )
                    sentiment_spring_pos_wc = gr.Image(
                        label="봄 긍정 워드클라우드", visible=False, scale=1
                    )
                    sentiment_spring_neg_wc = gr.Image(
                        label="봄 부정 워드클라우드", visible=False, scale=1
                    )
                with gr.Row():
                    sentiment_summer_chart = gr.Plot(
                        label="여름 시즌", visible=False, scale=1
                    )
                    sentiment_summer_pos_wc = gr.Image(
                        label="여름 긍정 워드클라우드", visible=False, scale=1
                    )
                    sentiment_summer_neg_wc = gr.Image(
                        label="여름 부정 워드클라우드", visible=False, scale=1
                    )
                with gr.Row():
                    sentiment_autumn_chart = gr.Plot(
                        label="가을 시즌", visible=False, scale=1
                    )
                    sentiment_autumn_pos_wc = gr.Image(
                        label="가을 긍정 워드클라우드", visible=False, scale=1
                    )
                    sentiment_autumn_neg_wc = gr.Image(
                        label="가을 부정 워드클라우드", visible=False, scale=1
                    )
                with gr.Row():
                    sentiment_winter_chart = gr.Plot(
                        label="겨울 시즌", visible=False, scale=1
                    )
                    sentiment_winter_pos_wc = gr.Image(
                        label="겨울 긍정 워드클라우드", visible=False, scale=1
                    )
                    sentiment_winter_neg_wc = gr.Image(
                        label="겨울 부정 워드클라우드", visible=False, scale=1
                    )

        gr.Markdown("### 개별 블로그 분석 결과")
        sentiment_df_output = gr.DataFrame(
            headers=[
                "블로그 제목",
                "링크",
                "감성 빈도",
                "감성 점수",
                "긍정 문장 수",
                "부정 문장 수",
                "긍정 비율 (%)",
                "부정 비율 (%)",
                "긍/부정 문장 요약",
            ],
            datatype=[
                "str",
                "str",
                "number",
                "str",
                "number",
                "number",
                "str",
                "str",
                "str",
            ],
            label="개별 블로그 분석 결과",
            wrap=True,
            interactive=True,
        )
        with gr.Row():
            sentiment_blog_page_num_input = gr.Number(
                value=1, label="페이지 번호", interactive=True, scale=1
            )
            sentiment_blog_total_pages_output = gr.Textbox(
                value="/ 1",
                label="전체 페이지",
                interactive=False,
                container=False,
                scale=1,
            )
            sentiment_blog_list_csv = gr.File(
                label="전체 블로그 목록(CSV) 다운로드", visible=False, scale=2
            )

        with gr.Accordion(
            "개별 블로그 상세 분석 (표에서 행 선택)", open=False, visible=False
        ) as sentiment_blog_detail_accordion:
            sentiment_individual_summary = gr.Textbox(
                label="긍/부정 문장 요약", visible=False, interactive=False, lines=10
            )
            with gr.Row():
                sentiment_individual_donut_chart = gr.Plot(
                    label="개별 블로그 긍/부정 비율", visible=False
                )
                sentiment_individual_score_chart = gr.Plot(
                    label="문장별 감성 점수", visible=False
                )

    # --- Event Handlers ---

    page_button_1.click(
        fn=lambda r, btn_value: display_page(r, int(btn_value)),
        inputs=[results_state, page_button_1],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )
    page_button_2.click(
        fn=lambda r, btn_value: display_page(r, int(btn_value)),
        inputs=[results_state, page_button_2],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )
    page_button_3.click(
        fn=lambda r, btn_value: display_page(r, int(btn_value)),
        inputs=[results_state, page_button_3],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )
    page_button_4.click(
        fn=lambda r, btn_value: display_page(r, int(btn_value)),
        inputs=[results_state, page_button_4],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )
    page_button_5.click(
        fn=lambda r, btn_value: display_page(r, int(btn_value)),
        inputs=[results_state, page_button_5],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )

    async def handle_rank_facilities(facilities_list, num_reviews, top_n, progress=gr.Progress()):
        ranked_facilities, status, gallery, report = await ranking_use_case.rank_places(
            places_list=facilities_list,
            num_reviews=num_reviews,
            top_n=top_n,
            progress=progress,
            is_course=False
        )
        return ranked_facilities, status, gallery, gr.update(value=report, visible=True)
    
    async def handle_rank_courses(courses_list, num_reviews, top_n, progress=gr.Progress()):
        ranked_courses, status, gallery, report = await ranking_use_case.rank_places(
            places_list=courses_list,
            num_reviews=num_reviews,
            top_n=top_n,
            progress=progress,
            is_course=True
        )
        return ranked_courses, status, gallery, gr.update(value=report, visible=True)
    def update_sigungu(area):
        choices = (
            ["전체"] + sorted(list(SIGUNGU_CODE_MAP.get(area, {}).keys()))
            if area != "전체"
            else ["전체"]
        )
        return gr.update(choices=choices, value="전체")

    def update_medium_cat(main_cat):
        choices = (
            ["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.get(main_cat, {}).keys()))
            if main_cat != "전체"
            else ["전체"]
        )
        return gr.update(choices=choices, value="전체")

    def update_small_cat(main_cat, medium_cat):
        choices = (
            ["전체"]
            + sorted(
                list(
                    ALL_FESTIVAL_CATEGORIES.get(main_cat, {}).get(medium_cat, {}).keys()
                )
            )
            if main_cat != "전체" and medium_cat != "전체"
            else ["전체"]
        )
        return gr.update(choices=choices, value="전체")

    area_dropdown.change(
        fn=update_sigungu, inputs=area_dropdown, outputs=sigungu_dropdown
    )
    main_cat_dropdown.change(
        fn=update_medium_cat, inputs=main_cat_dropdown, outputs=medium_cat_dropdown
    )
    medium_cat_dropdown.change(
        fn=update_small_cat,
        inputs=[main_cat_dropdown, medium_cat_dropdown],
        outputs=small_cat_dropdown,
    )

    def run_search_and_display(area, sigungu, main_cat, medium_cat, small_cat, status):
        results_for_state = [] # Explicit initialization
        total_pages = 1 # Explicit initialization
        initial_state = {
            "search_type": "festival_search",
            "area": area, "sigungu": sigungu, "main_cat": main_cat,
            "medium_cat": medium_cat, "small_cat": small_cat
        }
        final_state = db_search_graph.invoke(initial_state)
        all_results = final_state.get("results", [])
        # festival is (title, firstimage, eventstartdate, eventenddate)
        # Convert tuple results from DB to a list of dictionaries
        all_results_dicts = [
            {"title": row[0], "firstimage": row[1], "eventstartdate": row[2], "eventenddate": row[3]}
            for row in all_results
        ]

        filtered_by_status = []
        today = datetime.today().strftime(
            "%Y%m%d"
        )  # Get today's date in YYYYMMDD format

        for festival in all_results_dicts:
            # festival is now a dictionary
            event_start_date_str = str(festival.get("eventstartdate", "")).split('.')[0]
            event_end_date_str = str(festival.get("eventenddate", "")).split('.')[0]

            is_ongoing = False
            is_upcoming = False
            is_ended = False  # New flag

            if event_start_date_str and event_end_date_str:
                if len(event_start_date_str) == 8 and len(event_end_date_str) == 8:
                    if event_start_date_str <= today <= event_end_date_str:
                        is_ongoing = True
                    elif today < event_start_date_str:
                        is_upcoming = True
                    elif today > event_end_date_str:
                        is_ended = True
            
            if status == "전체":
                filtered_by_status.append(festival)
            elif status == "축제 진행중" and is_ongoing:
                filtered_by_status.append(festival)
            elif status == "진행 예정" and is_upcoming:
                filtered_by_status.append(festival)
            elif status == "종료된 축제" and is_ended:
                filtered_by_status.append(festival)

        print(f"DEBUG: run_search_and_display - len(all_results): {len(all_results_dicts)}")
        print(f"DEBUG: run_search_and_display - len(filtered_by_status): {len(filtered_by_status)}")

        # results_for_state should be the list of dicts
        results_for_state = sorted(filtered_by_status, key=lambda x: x['title'])

        total_pages = math.ceil(len(results_for_state) / PAGE_SIZE)
        
        gallery, page_str_updated, page_button_1_update, page_button_2_update, page_button_3_update, page_button_4_update, page_button_5_update = display_page(results_for_state, 1)
        return results_for_state, gallery, page_str_updated, gr.update(visible=len(results_for_state) > 0), gr.update(value=total_pages), page_button_1_update, page_button_2_update, page_button_3_update, page_button_4_update, page_button_5_update

    def run_nearby_search(festival_details, radius_meters):
        if (
            not festival_details
            or not festival_details.get("mapx")
            or not festival_details.get("mapy")
        ):
            return (
                [],
                [],
                gr.update(
                    value="축제 좌표 정보가 없어 추천할 수 없습니다.", visible=True
                ),
                [],
                [],
                gr.update(visible=False),
            )

        initial_state = {
            "search_type": "nearby_search",
            "latitude": festival_details.get("mapy"),
            "longitude": festival_details.get("mapx"),
            "radius": radius_meters,
        }
        final_state = db_search_graph.invoke(initial_state)

        facilities_recs = final_state.get("recommended_facilities", [])
        courses_recs = final_state.get("recommended_courses", [])

        if not facilities_recs and not courses_recs:
            return (
                [],
                [],
                gr.update(
                    value=f"{radius_meters}m 내에 추천할 장소가 없습니다.", visible=True
                ),
                [],
                [],
                gr.update(visible=False),
            )

        facility_gallery_output = [
            (item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"])
            for item in facilities_recs
        ]
        course_gallery_output = [
            (item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"])
            for item in courses_recs
        ]

        return (
            facilities_recs,
            courses_recs,
            facility_gallery_output,
            course_gallery_output,
            gr.update(visible=False),
            gr.update(visible=True),
        )

    search_btn.click(
        fn=run_search_and_display,
        inputs=[area_dropdown, sigungu_dropdown, main_cat_dropdown, medium_cat_dropdown, small_cat_dropdown, status_radio],
        outputs=[results_state, festival_gallery, page_display, results_area, total_pages_state, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )

    prev_button.click(fn=lambda r, p: display_paginated_gallery(r, p, -1), inputs=[results_state, page_display], outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5])
    next_button.click(fn=lambda r, p: display_paginated_gallery(r, p, 1), inputs=[results_state, page_display], outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5])

    festival_gallery.select(
        fn=display_festival_details,
        inputs=[results_state, page_display],
        outputs=[
            festival_details_output,
            selected_festival_state,
            selected_festival_details_state,
        ],
    ).then(
        fn=lambda: (
            gr.update(open=True),  # details_accordion
            gr.update(visible=True),  # image_gallery_accordion
            gr.update(visible=True),  # recommend_accordion
            gr.update(visible=True),  # naver_review_accordion
            gr.update(visible=True),  # trend_accordion
            gr.update(visible=True),  # wordcloud_accordion
            gr.update(visible=True),  # sentiment_accordion
        ),
        outputs=[
            details_accordion,
            image_gallery_accordion,
            recommend_accordion,
            naver_review_accordion,
            trend_accordion,
            wordcloud_accordion,
            sentiment_accordion,
        ],
    )

    image_collect_button.click(
        fn=handle_scrape_images,
        inputs=[selected_festival_state, num_blogs_for_images],
        outputs=[image_gallery, image_gallery_accordion, scraped_urls_output],
    )

    naver_search_btn.click(
        fn=get_naver_review_info,
        inputs=[selected_festival_state, num_reviews_naver_summary],
        outputs=[naver_review_output, naver_review_accordion],
    )

    trend_graph_btn.click(
        fn=handle_generate_trend_graphs,
        inputs=[selected_festival_state],
        outputs=[trend_accordion, trend_status, trend_plot_yearly, trend_plot_event],
    )

    word_cloud_btn.click(
        fn=handle_generate_word_cloud,
        inputs=[selected_festival_state, num_reviews_wordcloud],
        outputs=[wordcloud_accordion, wordcloud_status, wordcloud_plot],
    )

    run_sentiment_btn.click(
        fn=handle_analyze_sentiment,
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
        ],
    )

    rank_facilities_btn.click(
        fn=handle_rank_facilities,
        inputs=[
            recommended_facilities_state,
            ranking_reviews_slider,
            ranking_top_n_slider,
        ],
        outputs=[
            recommended_facilities_state,
            recommend_status,
            recommend_facilities_gallery,
            facility_ranking_report,
        ],
    )

    rank_courses_btn.click(
        fn=handle_rank_courses,
        inputs=[
            recommended_courses_state,
            ranking_reviews_slider,
            ranking_top_n_slider,
        ],
        outputs=[
            recommended_courses_state,
            recommend_status,
            recommend_courses_gallery,
            course_ranking_report,
        ],
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
            ranking_controls,
        ],
    )

    first_page_button.click(
        fn=lambda r: display_page(r, 1),
        inputs=[results_state],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )

    last_page_button.click(
        fn=lambda r: display_page(r, math.ceil(len(r) / PAGE_SIZE) if r else 1),
        inputs=[results_state],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )

    page_input.submit(
        fn=lambda r, pn: display_page(r, pn),
        inputs=[results_state, page_input],
        outputs=[festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5]
    )
    sentiment_blog_page_num_input.change(
        fn=lambda df, page_num: change_page(df, page_num),
        inputs=[blog_results_df_state, sentiment_blog_page_num_input],
        outputs=[
            sentiment_df_output,
            sentiment_blog_page_num_input,
            sentiment_blog_total_pages_output,
        ],
    )

    def handle_df_select(
        evt: gr.SelectData, page_num: int, df: pd.DataFrame, judgments: list
    ):
        BLOG_PAGE_SIZE = 10
        page_num = page_num or 1

        global_idx = (int(page_num) - 1) * BLOG_PAGE_SIZE + evt.index[0]

        if (
            df is None
            or df.empty
            or judgments is None
            or not isinstance(judgments, list)
            or global_idx >= len(judgments)
        ):
            return gr.update(), gr.update(), gr.update(), gr.update()

        judgments_for_row = judgments[global_idx]

        if not isinstance(judgments_for_row, list):
            return gr.update(), gr.update(), gr.update(), gr.update()

        donut_chart = create_donut_chart(
            sum(
                1
                for j in judgments_for_row
                if isinstance(j, dict) and j.get("final_verdict") == "긍정"
            ),
            sum(
                1
                for j in judgments_for_row
                if isinstance(j, dict) and j.get("final_verdict") == "부정"
            ),
            f"{df.iloc[global_idx]['블로그 제목'][:20]}... 긍/부정 비율",
        )

        score_chart = create_sentence_score_bar_chart(
            judgments_for_row,
            f"{df.iloc[global_idx]['블로그 제목'][:20]}... 문장별 점수",
        )

        summary_text = df.iloc[global_idx]["긍/부정 문장 요약"]

        return (
            gr.update(value=donut_chart, visible=True),
            gr.update(value=score_chart, visible=True),
            gr.update(value=summary_text, visible=True),
            gr.update(open=True, visible=True),
        )

    sentiment_df_output.select(
        fn=handle_df_select,
        inputs=[
            sentiment_blog_page_num_input,
            blog_results_df_state,
            blog_judgments_state,
        ],
        outputs=[
            sentiment_individual_donut_chart,
            sentiment_individual_score_chart,
            sentiment_individual_summary,
            sentiment_blog_detail_accordion,
        ],
    )

    def display_recommend_details(evt: gr.SelectData, recommended_results):
        if not evt or not recommended_results:
            return gr.update(visible=False), gr.update(visible=False)

        selected_item = recommended_results[evt.index]

        details = []
        exclude_cols = [
            "type",
            "id",
            "contentid",
            "contenttypeid",
            "lDongRegnCd",
            "lDongSignguCd",
            "lclsSystm1",
            "lclsSystm2",
            "lclsSystm3",
            "mlevel",
            "cpyrhtDivCd",
            "areacode",
            "cat1",
            "cat2",
            "cat3",
            "createdtime",
            "mapx",
            "mapy",
            "modifiedtime",
            "sigungucode",
            "sub_points",  # also exclude the sub_points list itself from the main loop
        ]

        # Handle main info for both facilities and courses
        for key, value in selected_item.items():
            if key in exclude_cols:
                continue
            if value is not None and str(value).strip() != "":
                display_key = COLUMN_TRANSLATIONS.get(key, key)
                details.append(f"**{display_key}**: {value}")

        # If it's a course, handle the sub_points specially
        if "sub_points" in selected_item and selected_item["sub_points"]:
            sub_points = selected_item["sub_points"]

            all_subnames = [sp.get("subname") for sp in sub_points if sp.get("subname")]
            if all_subnames:
                # Remove duplicates while preserving order
                unique_subnames = list(dict.fromkeys(all_subnames))
                details.append(f"**세부 코스명**: {', '.join(unique_subnames)}")

            all_overviews = [
                sp.get("subdetailoverview")
                for sp in sub_points
                if sp.get("subdetailoverview")
            ]
            if all_overviews:
                overview_list_str = [
                    f"{i+1}. {desc}" for i, desc in enumerate(all_overviews)
                ]
                details.append(f"**세부 코스개요**:\n" + "\n".join(overview_list_str))

        details_text = "\n\n".join(details)

        return gr.update(value=details_text), gr.update(visible=True, open=True)

    recommend_facilities_gallery.select(
        fn=display_recommend_details,
        inputs=[recommended_facilities_state],
        outputs=[recommend_details_output, recommend_details_accordion],
    )

    recommend_courses_gallery.select(
        fn=display_recommend_details,
        inputs=[recommended_courses_state],
        outputs=[recommend_details_output, recommend_details_accordion],
    )

    async def handle_rank_festivals(festivals_list, num_reviews, top_n, progress=gr.Progress()):
        if not festivals_list:
            # Return empty updates for all outputs
            empty_updates = [[]] + [gr.update(value=[])] + ["1 / 1"] + [gr.update(visible=False)] * 5 + [gr.update(value="", visible=False)]
            return tuple(empty_updates)

        ranked_festivals, report_md = await ranking_use_case.rank_festivals(
            festivals_list=festivals_list,
            num_reviews=num_reviews,
            top_n=top_n,
            progress=progress,
        )
        
        gallery_output, page_display_str, *page_buttons_updates = display_page(ranked_festivals, 1)
        
        report_update = gr.update(value=report_md, visible=True)

        return ranked_festivals, gallery_output, page_display_str, *page_buttons_updates, report_update

    rank_festivals_btn.click(
        fn=handle_rank_festivals,
        inputs=[results_state, num_reviews_festival_ranking, festival_ranking_top_n_slider],
        outputs=[results_state, festival_gallery, page_display, page_button_1, page_button_2, page_button_3, page_button_4, page_button_5, festival_ranking_report]
    )

if __name__ == "__main__":

    ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()

    demo.launch(allowed_paths=["assets", "temp_img"])
