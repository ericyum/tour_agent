import gradio as gr
import pandas as pd
import asyncio
import math
import os
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# --- Custom Module Imports ---
from src.infrastructure.config.loader import (
    ICON_MAP, KOREAN_FONT_PATH, TITLE_TO_CAT_NAMES, ALL_FESTIVAL_CATEGORIES, FESTIVAL_INFO_LOOKUP, BEST_IMAGES_MAP
)
from src.application.core.constants import (
    CATEGORY_TO_ICON_MAP, NO_IMAGE_URL, PAGE_SIZE, COLUMN_TRANSLATIONS, AREA_CODE_MAP, SIGUNGU_CODE_MAP
)
from src.application.services.festival_service import get_festival_details_by_title
from src.application.agents.precaution_agent import PrecautionAgent
from src.application.supervisors.db_search_supervisor import db_search_graph
from src.application.supervisors.course_validation_supervisor import course_validation_graph
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor
from src.application.use_cases.analysis_use_case import AnalysisUseCase
from src.application.use_cases.sentiment_analysis_use_case import SentimentAnalysisUseCase
from src.application.use_cases.ranking_use_case import RankingUseCase
from src.application.core.utils import change_page
from src.infrastructure.reporting.charts import create_donut_chart, create_sentence_score_bar_chart

# --- Object Instantiation ---
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
naver_supervisor = NaverReviewSupervisor()
precaution_agent = PrecautionAgent()

analysis_use_case = AnalysisUseCase(
    naver_supervisor=naver_supervisor, font_path=KOREAN_FONT_PATH,
    title_to_cat_map=TITLE_TO_CAT_NAMES, cat_to_icon_map=CATEGORY_TO_ICON_MAP, script_dir=script_dir
)
sentiment_analysis_use_case = SentimentAnalysisUseCase(naver_supervisor=naver_supervisor, script_dir=script_dir)
ranking_use_case = RankingUseCase(naver_supervisor=naver_supervisor)

# --- Asset Path Functions ---
def get_local_asset_path(asset_type: str, festival_name: str) -> str | None:
    base_dir = os.path.join(script_dir, asset_type)
    if not os.path.exists(base_dir):
        return None
    if asset_type == 'best_images_and_icons/icons':
        icon_filename = ICON_MAP.get(festival_name)
        if icon_filename:
            file_path = os.path.join(base_dir, icon_filename)
            if os.path.exists(file_path):
                return file_path
        return None
    if asset_type == 'best_images_and_icons/best_images':
        image_filename = BEST_IMAGES_MAP.get(festival_name)
        if image_filename:
            file_path = os.path.join(base_dir, image_filename)
            if os.path.exists(file_path):
                return file_path
        return None
    return None

def get_local_best_image_path(festival_name: str) -> str | None:
    return get_local_asset_path('best_images_and_icons/best_images', festival_name)

def get_local_icon_path(festival_name: str) -> str | None:
    return get_local_asset_path('best_images_and_icons/icons', festival_name)

# --- UI Logic Functions ---
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
            button_updates.append(gr.update(visible=False, value=""))
    return button_updates

def display_page(results, page):
    page = int(page)
    start_index = (page - 1) * PAGE_SIZE
    end_index = start_index + PAGE_SIZE
    page_results = results[start_index:end_index]
    gallery_output = []
    for item in page_results:
        festival_title = item.get("title", "")
        local_image = get_local_best_image_path(festival_title)
        image = local_image if local_image else (item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL)
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
        current_page_to_display = 1 if new_page < 1 else total_pages
        gallery_output, page_display_str, *page_buttons_updates = display_page(results, current_page_to_display)
        return gallery_output, page_display_str, *page_buttons_updates

# --- Event Handler Functions ---
def run_search_and_display(area, sigungu, main_cat, medium_cat, small_cat, status):
    initial_state = {"search_type": "festival_search", "area": area, "sigungu": sigungu, "main_cat": main_cat, "medium_cat": medium_cat, "small_cat": small_cat}
    final_state = db_search_graph.invoke(initial_state)
    all_results = final_state.get("results", [])
    all_results_dicts = [{
        "title": row[0],
        "firstimage": row[1],
        "eventstartdate": row[2],
        "eventenddate": row[3]
    } for row in all_results]
    filtered_by_status = []
    today = datetime.today().strftime("%Y%m%d")
    for festival in all_results_dicts:
        event_start_date_str = str(festival.get("eventstartdate", "")).split('.')[0]
        event_end_date_str = str(festival.get("eventenddate", "")).split('.')[0]
        is_ongoing, is_upcoming, is_ended = False, False, False
        if event_start_date_str and event_end_date_str and len(event_start_date_str) == 8 and len(event_end_date_str) == 8:
            if event_start_date_str <= today <= event_end_date_str:
                is_ongoing = True
            elif today < event_start_date_str:
                is_upcoming = True
            elif today > event_end_date_str:
                is_ended = True
        if status == "전체" or (status == "축제 진행중" and is_ongoing) or (status == "진행 예정" and is_upcoming) or (status == "종료된 축제" and is_ended):
            filtered_by_status.append(festival)
    results_for_state = sorted(filtered_by_status, key=lambda x: x['title'])
    total_pages = math.ceil(len(results_for_state) / PAGE_SIZE) if results_for_state else 1
    gallery, page_str_updated, *page_buttons_updates = display_page(results_for_state, 1)
    return results_for_state, gallery, page_str_updated, gr.update(visible=len(results_for_state) > 0), gr.update(value=total_pages), *page_buttons_updates

async def display_festival_details_and_precautions(evt: gr.SelectData, results, page_str):
    page = int(page_str.split("/")[0].strip())
    global_index = (page - 1) * PAGE_SIZE + evt.index
    if global_index >= len(results) or global_index < 0:
        # Add a placeholder for the new accordion output
        yield gr.update(value="선택된 축제 정보를 찾을 수 없습니다."), None, None, gr.update(visible=False), None, None, None, None, None, None, None, None
        return
    selected_item = results[global_index]
    original_title = selected_item.get("title", "")
    details = get_festival_details_by_title(original_title)
    if not details:
        # Add a placeholder for the new accordion output
        yield gr.update(value="정보를 찾을 수 없습니다."), None, None, gr.update(visible=False), None, None, None, None, None, None, None, None
        return
    details_list = []
    local_best_image = get_local_best_image_path(original_title)
    # Removed the display of local_best_image as per user request.
    local_icon = get_local_icon_path(original_title)
    if local_icon:
        gradio_served_path = f"/gradio_api/file={Path(local_icon).as_posix()}"
        details_list.append(f'<img src="{gradio_served_path}" width="100">')
    score_keys = {"ranking_score": "종합 순위 점수", "time_score": "시기성 점수", "sentiment_score": "만족도 점수", "quarterly_trend_score": "최근 화제성(90일)", "yearly_trend_score": "연간 꾸준함(365일)"}
    for key, display_name in score_keys.items():
        if key in selected_item:
            details_list.append(f"**{display_name}**: {selected_item[key]}")
    if details_list:
        details_list.append("---")
    exclude_cols = ["id", "contentid", "contenttypeid", "lDongRegnCd", "lDongSignguCd", "lclsSystm1", "lclsSystm2", "lclsSystm3", "mlevel", "cpyrhtDivCd", "areacode", "cat1", "cat2", "cat3", "createdtime", "mapx", "mapy", "modifiedtime", "sigungucode", "ranking_score", "time_score", "sentiment_score", "quarterly_trend_score", "yearly_trend_score"]
    for key, value in details.items():
        if key not in exclude_cols and value is not None and str(value).strip() != "":
            display_key = COLUMN_TRANSLATIONS.get(key, key)
            details_list.append(f"**{display_key}**: {value}")
    details_text = "\n\n".join(details_list)
    # Add gr.update(open=True) for the details_accordion
    yield gr.update(value=details_text), original_title, details, gr.update(value="⏳ AI가 맞춤형 에티켓을 생성 중입니다...", visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), selected_item, gr.update(open=True)
    precautions_text = ""
    if FESTIVAL_INFO_LOOKUP and original_title in FESTIVAL_INFO_LOOKUP:
        festival_info = FESTIVAL_INFO_LOOKUP[original_title]
        detailed_cat = festival_info.get('detailed_category')
        prohibited_behaviors = festival_info.get('prohibited_behaviors')
        if detailed_cat or prohibited_behaviors:
            precautions_text = await precaution_agent.generate_precautions(original_title, detailed_cat, prohibited_behaviors)
    if not precautions_text:
        precautions_text = "이 축제에 대한 세부 주의사항 정보가 없습니다."
    # Add gr.update(open=True) for the details_accordion
    yield gr.update(value=details_text), original_title, details, gr.update(value=precautions_text, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), selected_item, gr.update(open=True)

async def get_naver_review_info(festival_name, num_reviews):
    if not festival_name:
        yield gr.update(value="먼저 축제를 선택해주세요."), gr.update(visible=False)
        return
    yield gr.update(value=f"{festival_name} 후기 검색 중... ({num_reviews}개)", visible=True), gr.update(visible=True, open=True)
    summary, _ = await naver_supervisor.get_review_summary_and_tips(festival_name, num_reviews=num_reviews)
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
    outputs_to_clear = [gr.update(open=True)] + [gr.update(value="")] + [gr.update(visible=False)] * 21 + [None, None, None, 1, "/ 1"] + [gr.update(visible=False)] * 5
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
        distribution_description = f"이 분포표는 **{len(result['blog_df'])}**개의 블로그 후기를 분석한 결과입니다. 각 후기에서 긍정/부정 문장을 추출하고, 해당 문장들의 감성 점수를 기반으로 전체적인 만족도 분포를 시각화한 것입니다."
        outlier_description = f"총 **{result['total_score_count']}**개의 감성 점수 중 **{result['outlier_count']}**개의 이상치가 발견되어 만족도 레벨 계산에서 제외되었습니다."
        yield (
            gr.update(visible=True, open=True), "분석 완료",
            gr.update(value=result["neg_summary_text"], visible=bool(result["neg_summary_text"])),
            gr.update(value=result["overall_chart"], visible=True),
            gr.update(value=result["distribution_chart"], visible=result["distribution_chart"] is not None),
            gr.update(value=distribution_description, visible=True),
            gr.update(value=result["outlier_chart"], visible=result["outlier_chart"] is not None),
            gr.update(value=outlier_description, visible=True),
            gr.update(value=result["positive_keywords_html"], visible=bool(result["positive_keywords_html"])),
            gr.update(value=result["overall_summary_text"], visible=True),
            gr.update(value=result["summary_csv_path"], visible=result["summary_csv_path"] is not None),
            *[gr.update(value=seasonal_charts.get(s), visible=s in seasonal_charts) for s in ["봄", "여름", "가을", "겨울"]],
            *[item for s in ["봄", "여름", "가을", "겨울"] for item in (gr.update(value=seasonal_pos_wc.get(s), visible=s in seasonal_pos_wc), gr.update(value=seasonal_neg_wc.get(s), visible=s in seasonal_neg_wc))],
            initial_page_df, result["blog_df"], result["blog_judgments_list"], current_page, total_pages_str,
            gr.update(value=result["blog_list_csv_path"], visible=result["blog_list_csv_path"] is not None),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, open=False),
        )
        return
    except Exception as e:
        print(f"감성 분석 중 예외 발생: {e}")
        traceback.print_exc()
        outputs_to_clear[1] = f"분석 중 오류 발생: {e}"
        yield tuple(outputs_to_clear)
        return

async def handle_rank_festivals(festivals_list, num_reviews, top_n, progress=gr.Progress()):
    if not festivals_list:
        return [], gr.update(value=[]), "1 / 1", *([gr.update(visible=False)] * 5), gr.update(value="", visible=False)
    ranked_festivals, report_md = await ranking_use_case.rank_festivals(festivals_list=festivals_list, num_reviews=num_reviews, top_n=top_n, progress=progress)
    gallery_output, page_display_str, *page_buttons_updates = display_page(ranked_festivals, 1)
    report_update = gr.update(value=report_md, visible=True)
    return ranked_festivals, gallery_output, page_display_str, *page_buttons_updates, report_update

def run_nearby_search(festival_details, radius_meters):
    if not festival_details or not festival_details.get("mapx") or not festival_details.get("mapy"):
        return [], [], [], gr.update(value="좌표 정보가 없어 추천할 수 없습니다.", visible=True), [], [], [], gr.update(visible=False)
    initial_state = {"search_type": "nearby_search", "latitude": festival_details.get("mapy"), "longitude": festival_details.get("mapx"), "radius": radius_meters, "current_festival_id": festival_details.get("contentid")}
    final_state = db_search_graph.invoke(initial_state)
    facilities_recs = final_state.get("recommended_facilities", [])
    courses_recs = final_state.get("recommended_courses", [])
    festivals_recs = final_state.get("recommended_festivals", [])
    if not facilities_recs and not courses_recs and not festivals_recs:
        return [], [], [], gr.update(value=f"{radius_meters}m 내에 추천할 장소가 없습니다.", visible=True), [], [], [], gr.update(visible=False)
    facility_gallery = [(item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"]) for item in facilities_recs]
    course_gallery = [(item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"]) for item in courses_recs]
    festival_gallery = [(item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"]) for item in festivals_recs]
    return facilities_recs, courses_recs, festivals_recs, gr.update(visible=False), facility_gallery, course_gallery, festival_gallery, gr.update(visible=True)

async def handle_rank_places(places_list, num_reviews, top_n, is_course, progress=gr.Progress()):
    ranked_places, status, gallery, report = await ranking_use_case.rank_places(places_list=places_list, num_reviews=num_reviews, top_n=top_n, progress=progress, is_course=is_course)
    return ranked_places, status, gallery, gr.update(value=report, visible=True)

def handle_item_selection(evt: gr.SelectData, items: list):
    """Handles the selection of an item from any gallery, updates the temp state, and shows details."""
    if not evt or not items or evt.index >= len(items):
        return None, gr.update(value="오류: 항목을 찾을 수 없습니다."), gr.update(visible=False)
    selected_item = items[evt.index]
    details_list = []
    exclude_cols = ["id", "contentid", "contenttypeid", "lDongRegnCd", "lDongSignguCd", "lclsSystm1", "lclsSystm2", "lclsSystm3", "mlevel", "cpyrhtDivCd", "areacode", "cat1", "cat2", "cat3", "createdtime", "mapx", "mapy", "modifiedtime", "sigungucode", "ranking_score", "time_score", "sentiment_score", "quarterly_trend_score", "yearly_trend_score", "sub_points", "trend_reason", "sentiment_reason"]
    for key, value in selected_item.items():
        if key not in exclude_cols and value is not None and str(value).strip() != "":
            display_key = COLUMN_TRANSLATIONS.get(key, key)
            details_list.append(f"**{display_key}**: {value}")
    if "sub_points" in selected_item and selected_item["sub_points"]:
        sub_points = selected_item["sub_points"]
        all_subnames = [sp.get("subname") for sp in sub_points if sp.get("subname")]
        if all_subnames:
            unique_subnames = list(dict.fromkeys(all_subnames))
            details_list.append(f"**세부 코스명**: {', '.join(unique_subnames)}")
        all_overviews = [sp.get("subdetailoverview") for sp in sub_points if sp.get("subdetailoverview")]
        if all_overviews:
            overview_list_str = [f"{i+1}. {desc}" for i, desc in enumerate(all_overviews)]
            details_list.append(f"**세부 코스개요**:\n" + "\n".join(overview_list_str))
    details_text = "\n\n".join(details_list)
    return selected_item, gr.update(value=details_text), gr.update(visible=True, open=True)

def add_to_my_course(item_to_add, current_course: list):
    if item_to_add and not any(c['title'] == item_to_add.get('title') for c in current_course):
        current_course.append(item_to_add)
    course_html = "<h3>🚗 나만의 코스</h3>"
    if not current_course:
        course_html += "<p>나만의 코스에 항목을 추가해보세요.</p>"
    else:
        for i, item in enumerate(current_course):
            details_html = ""
            exclude_cols = ["id", "contentid", "contenttypeid", "lDongRegnCd", "lDongSignguCd", "lclsSystm1", "lclsSystm2", "lclsSystm3", "mlevel", "cpyrhtDivCd", "areacode", "cat1", "cat2", "cat3", "createdtime", "mapx", "mapy", "modifiedtime", "sigungucode", "distance", "distance_score", "quarterly_trend_score", "yearly_trend_score", "sentiment_score", "ranking_score", "trend_reason", "sentiment_reason", "sub_points"]
            for key, value in item.items():
                if key not in exclude_cols and value and str(value).strip():
                    display_key = COLUMN_TRANSLATIONS.get(key, key)
                    details_html += f"<b>{display_key}</b>: {value}<br>"
            course_html += f'''<details><summary><b>{i+1}. {item.get('title', '이름 없음')}</b></summary><div style="padding: 10px; border: 1px solid #eee; margin-top: 5px;">{details_html}</div></details>'''
    return current_course, course_html

def clear_my_course():
    return [], "<h3>🚗 나만의 코스</h3><p>나만의 코스에 항목을 추가해보세요.</p>"

async def handle_validate_course(course, duration):
    if not course or not duration:
        yield "코스나 여행 기간을 입력해주세요."
        return
    yield "검증 중입니다... ⏳"
    initial_state = {"course": course, "duration": duration}
    loop = asyncio.get_running_loop()
    final_state = await loop.run_in_executor(None, course_validation_graph.invoke, initial_state)
    yield final_state.get("validation_result", "검증 결과를 가져오는 데 실패했습니다.")

def handle_df_select(evt: gr.SelectData, page_num: int, df: pd.DataFrame, judgments: list):
    BLOG_PAGE_SIZE = 10
    page_num = page_num or 1
    global_idx = (int(page_num) - 1) * BLOG_PAGE_SIZE + evt.index[0]
    if df is None or df.empty or judgments is None or not isinstance(judgments, list) or global_idx >= len(judgments):
        return gr.update(), gr.update(), gr.update(), gr.update()
    judgments_for_row = judgments[global_idx]
    if not isinstance(judgments_for_row, list):
        return gr.update(), gr.update(), gr.update(), gr.update()
    donut_chart = create_donut_chart(sum(1 for j in judgments_for_row if isinstance(j, dict) and j.get("final_verdict") == "긍정"), sum(1 for j in judgments_for_row if isinstance(j, dict) and j.get("final_verdict") == "부정"), f"{df.iloc[global_idx]['블로그 제목'][:20]}... 긍/부정 비율")
    score_chart = create_sentence_score_bar_chart(judgments_for_row, f"{df.iloc[global_idx]['블로그 제목'][:20]}... 문장별 점수")
    summary_text = df.iloc[global_idx]["긍/부정 문장 요약"]
    return gr.update(value=donut_chart, visible=True), gr.update(value=score_chart, visible=True), gr.update(value=summary_text, visible=True), gr.update(open=True, visible=True)