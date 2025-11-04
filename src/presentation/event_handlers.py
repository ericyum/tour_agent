import gradio as gr
import pandas as pd
import asyncio
import math
import os
import re
import traceback  # <--- [ 1. traceback 임포트 ]
from datetime import datetime, timedelta
from pathlib import Path
import base64

# --- Custom Module Imports (기존과 동일) ---
from src.infrastructure.config.loader import (
    ICON_MAP,
    KOREAN_FONT_PATH,
    TITLE_TO_CAT_NAMES,
    ALL_FESTIVAL_CATEGORIES,
    FESTIVAL_INFO_LOOKUP,
    BEST_IMAGES_MAP,
    DF_SPLIT,
    DF_CAMERA,  # <--- [ 2. 1단계에서 로드한 CSV 데이터 임포트 ]
)
from src.application.core.constants import (
    CATEGORY_TO_ICON_MAP,
    NO_IMAGE_URL,
    PAGE_SIZE,
    COLUMN_TRANSLATIONS,
    AREA_CODE_MAP,
    SIGUNGU_CODE_MAP,
)
from src.application.services.festival_service import get_festival_details_by_title
from src.application.agents.precaution_agent import PrecautionAgent
from src.application.supervisors.db_search_supervisor import db_search_graph
from src.application.supervisors.course_validation_supervisor import (
    course_validation_graph,
)
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor
from src.application.use_cases.analysis_use_case import AnalysisUseCase
from src.application.use_cases.sentiment_analysis_use_case import (
    SentimentAnalysisUseCase,
)
from src.application.use_cases.ranking_use_case import RankingUseCase

# <--- [ 3. 3단계에서 만든 RenderingUseCase 임포트 ] ---
from src.application.use_cases.rendering_use_case import RenderingUseCase

# --- [ 신규 추가 ] ---
# charts.py와 utils.py에서 필요한 함수들을 임포트합니다.
from src.infrastructure.reporting.charts import (
    create_donut_chart,
    create_sentence_score_bar_chart,
)
from src.application.core.utils import change_page

# --- [ 신규 추가 끝 ] ---


# --- Object Instantiation (기존과 동일) ---
script_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
naver_supervisor = NaverReviewSupervisor()
precaution_agent = PrecautionAgent()

analysis_use_case = AnalysisUseCase(
    naver_supervisor=naver_supervisor,
    font_path=KOREAN_FONT_PATH,
    title_to_cat_map=TITLE_TO_CAT_NAMES,
    cat_to_icon_map=CATEGORY_TO_ICON_MAP,
    script_dir=script_dir,
)
sentiment_analysis_use_case = SentimentAnalysisUseCase(
    naver_supervisor=naver_supervisor, script_dir=script_dir
)
ranking_use_case = RankingUseCase(naver_supervisor=naver_supervisor)

# <--- [ 4. RenderingUseCase 인스턴스 생성 ] ---
rendering_use_case = RenderingUseCase(df_split=DF_SPLIT, df_camera=DF_CAMERA)
# <--- [ 인스턴스 생성 끝 ] ---


# --- [ 신규 ] Navigation Handlers ---
# (이하 기존 코드들... display_page, display_paginated_gallery 등... 수정 없음)
# ... (기존 코드 생략) ...


def get_local_asset_path(asset_type: str, festival_name: str) -> str | None:
    base_dir = os.path.join(script_dir, asset_type)
    if not os.path.exists(base_dir):
        return None
    if asset_type == "best_images_and_icons/icons":
        icon_filename = ICON_MAP.get(festival_name)
        if icon_filename:
            file_path = os.path.join(base_dir, icon_filename)
            if os.path.exists(file_path):
                return file_path
        return None
    if asset_type == "best_images_and_icons/best_images":
        image_filename = BEST_IMAGES_MAP.get(festival_name)
        if image_filename:
            file_path = os.path.join(base_dir, image_filename)
            if os.path.exists(file_path):
                return file_path
        return None
    return None


def get_local_best_image_path(festival_name: str) -> str | None:
    return get_local_asset_path("best_images_and_icons/best_images", festival_name)


def get_local_icon_path(festival_name: str) -> str | None:
    return get_local_asset_path("best_images_and_icons/icons", festival_name)


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
            button_updates.append(
                gr.update(
                    value=str(page_num),
                    visible=True,
                    variant="primary" if page_num == current_page else "secondary",
                )
            )
        else:
            button_updates.append(gr.update(visible=False, value=""))
    return button_updates


def display_page(results, page):
    """[*** 수정됨 ***]
    페이지에 맞는 갤러리와 '체크박스 목록'을 함께 반환합니다.
    """
    print(
        f"DEBUG: Entering display_page. Results length: {len(results) if results else 0}, Page: {page}"
    )
    page = int(page)
    start_index = (page - 1) * PAGE_SIZE
    end_index = start_index + PAGE_SIZE
    page_results = results[start_index:end_index]
    gallery_output = []

    # [수정] 현재 페이지의 축제 제목 목록 생성
    checkbox_choices = []

    for item in page_results:
        festival_title = item.get("title", "제목 없음")
        local_image = get_local_best_image_path(festival_title)
        image = (
            local_image
            if local_image
            else (item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL)
        )
        title = festival_title
        if "ranking_score" in item:
            title = f"점수: {item.get('ranking_score', 'N/A')} - {title}"
        gallery_output.append((image, title))
        checkbox_choices.append(festival_title)  # [수정] 체크박스 목록에 추가

    total_pages = math.ceil(len(results) / PAGE_SIZE)
    print(f"DEBUG: total_pages calculated in display_page: {total_pages}")
    page_buttons_updates = update_pagination_buttons(page, total_pages)

    # [수정] 체크박스 업데이트 객체 생성 (선택된 값은 비움)
    checkbox_update = gr.update(choices=checkbox_choices, value=[])

    # [수정] 반환값에 체크박스 업데이트 추가 (총 5(버튼) + 1(갤러리) + 1(페이지표시) + 1(체크박스) = 8개)
    # *page_buttons_updates가 5개이므로, gallery_output, f"{page} / {total_pages}" 2개, checkbox_update 1개
    return (
        gallery_output,
        f"{page} / {total_pages}",
        *page_buttons_updates,
        checkbox_update,
    )


def display_paginated_gallery(results, page_str, direction):
    """[*** 수정됨 ***]
    display_page의 변경된 반환값을 모두 전달합니다.
    """
    print(
        f"DEBUG: Entering display_paginated_gallery. Results length: {len(results) if results else 0}, Page String: {page_str}, Direction: {direction}"
    )
    page = int(page_str.split("/")[0].strip())
    total_pages = math.ceil(len(results) / PAGE_SIZE)
    new_page = page + direction
    if 1 <= new_page <= total_pages:
        # [수정] checkbox_update 반환값 추가 (총 8개)
        gallery_output, page_display_str, *page_buttons_updates, checkbox_update = (
            display_page(results, new_page)
        )
        # [수정] 반환값에 체크박스 업데이트 추가
        return gallery_output, page_display_str, *page_buttons_updates, checkbox_update
    else:
        current_page_to_display = 1 if new_page < 1 else total_pages
        # [수정] checkbox_update 반환값 추가 (총 8개)
        gallery_output, page_display_str, *page_buttons_updates, checkbox_update = (
            display_page(results, current_page_to_display)
        )
        # [수정] 반환값에 체크박스 업데이트 추가
        return gallery_output, page_display_str, *page_buttons_updates, checkbox_update


# --- [ 수정 ] Event Handler Functions ---


def run_search_and_display(area, sigungu, main_cat, medium_cat, small_cat, status):
    """(기존 핸들러 - 내부 로직 수정)
    DB 검색을 실행하고 첫 페이지 결과를 반환합니다.
    (주의: 이 함수는 이제 run_search_and_populate_ranking_cbox에 의해서만 사용됨)
    """
    initial_state = {
        "search_type": "festival_search",
        "area": area,
        "sigungu": sigungu,
        "main_cat": main_cat,
        "medium_cat": medium_cat,
        "small_cat": small_cat,
    }
    final_state = db_search_graph.invoke(initial_state)
    all_results = final_state.get("results", [])
    all_results_dicts = [
        {
            "title": row[0],
            "firstimage": row[1],
            "eventstartdate": row[2],
            "eventenddate": row[3],
        }
        for row in all_results
    ]

    # ... (기존 필터링 로직) ...
    filtered_by_status = []
    today = datetime.today().strftime("%Y%m%d")
    for festival in all_results_dicts:
        event_start_date_str = str(festival.get("eventstartdate", "")).split(".")[0]
        event_end_date_str = str(festival.get("eventenddate", "")).split(".")[0]
        is_ongoing, is_upcoming, is_ended = False, False, False
        if (
            event_start_date_str
            and event_end_date_str
            and len(event_start_date_str) == 8
            and len(event_end_date_str) == 8
        ):
            if event_start_date_str <= today <= event_end_date_str:
                is_ongoing = True
            elif today < event_start_date_str:
                is_upcoming = True
            elif today > event_end_date_str:
                is_ended = True
        if (
            status == "전체"
            or (status == "축제 진행중" and is_ongoing)
            or (status == "진행 예정" and is_upcoming)
            or (status == "종료된 축제" and is_ended)
        ):
            filtered_by_status.append(festival)

    results_for_state = sorted(filtered_by_status, key=lambda x: x["title"])
    total_pages = (
        math.ceil(len(results_for_state) / PAGE_SIZE) if results_for_state else 1
    )

    # [수정] display_page를 호출하여 첫 페이지 갤러리와 체크박스를 가져옴
    gallery, page_str_updated, *page_buttons_updates, checkbox_update = display_page(
        results_for_state, 1
    )

    # [수정] 반환값 확장 (총 9개)
    return (
        results_for_state,
        gallery,
        page_str_updated,
        gr.update(visible=len(results_for_state) > 0),
        total_pages,
        *page_buttons_updates,
        checkbox_update,  # 1페이지에 맞는 체크박스
    )


async def display_festival_details_and_precautions(
    evt: gr.SelectData, results, page_str
):
    """(*** 수정됨 ***)"""
    page = int(page_str.split("/")[0].strip())
    global_index = (page - 1) * PAGE_SIZE + evt.index
    if global_index >= len(results) or global_index < 0:
        yield gr.update(
            value="선택된 축제 정보를 찾을 수 없습니다."
        ), None, None, gr.update(visible=False), None
        return

    selected_item = results[global_index]
    original_title = selected_item.get("title", "")
    details = get_festival_details_by_title(original_title)
    if not details:
        yield gr.update(value="정보를 찾을 수 없습니다."), None, None, gr.update(
            visible=False
        ), None
        return

    details_list = []

    # [*** 수정된 로직 ***]
    # 1. get_local_icon_path는 'C:\project\best_images_and_icons\icons\icon.png' 같은 *절대 경로*를 반환
    local_icon_abs_path = get_local_icon_path(original_title)

    if local_icon_abs_path:
        try:
            # 2. 이미지 파일을 읽어서 base64로 인코딩
            with open(local_icon_abs_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            # 3. 파일 확장자에 따라 올바른 Mime type 결정
            mime_type = "image/png"  # 기본값
            if local_icon_abs_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif local_icon_abs_path.lower().endswith(".gif"):
                mime_type = "image/gif"

            # 4. Base64 데이터를 src에 직접 삽입하는 HTML 태그 생성
            details_list.append(
                f'<img src="data:{mime_type};base64,{encoded_string}" width="200">'
            )

        except Exception as e:
            # 파일 읽기 또는 인코딩 실패 시
            print(f"Error base64 encoding icon: {e}")
            pass  # 아이콘 없이 진행

    # [기존 경로 로직 삭제]
    # if local_icon:
    #     gradio_served_path = f"/file={Path(local_icon).as_posix()}"
    #     details_list.append(f'<img src="{gradio_served_path}" width="200">')

    score_keys = {
        "ranking_score": "종합 순위 점수",
        "time_score": "시기성 점수",
        "sentiment_score": "만족도 점수",
        "quarterly_trend_score": "최근 화제성(90일)",
        "yearly_trend_score": "연간 꾸준함(365일)",
    }
    for key, display_name in score_keys.items():
        if key in selected_item:
            details_list.append(f"**{display_name}**: {selected_item[key]}")
    if details_list:
        details_list.append("---")

    exclude_cols = [
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
        "ranking_score",
        "time_score",
        "sentiment_score",
        "quarterly_trend_score",
        "yearly_trend_score",
    ]
    for key, value in details.items():
        if key not in exclude_cols and value is not None and str(value).strip() != "":
            display_key = COLUMN_TRANSLATIONS.get(key, key)
            details_list.append(f"**{display_key}**: {value}")

    details_text = "\n\n".join(details_list)

    # 1차 반환 (즉각적인 정보)
    # [수정] 4번째 반환값(precautions_output)을 즉시 visible=True로 설정
    yield gr.update(value=details_text), original_title, details, gr.update(
        value="⏳ AI가 맞춤형 에티켓을 생성 중입니다...", visible=True
    ), details  # [수정] selected_item 대신 details(전체 DB 정보)를 반환

    # 2차 반환 (LLM 생성 정보)
    precautions_text = ""
    if FESTIVAL_INFO_LOOKUP and original_title in FESTIVAL_INFO_LOOKUP:
        festival_info = FESTIVAL_INFO_LOOKUP[original_title]
        detailed_cat = festival_info.get("detailed_category")
        prohibited_behaviors = festival_info.get("prohibited_behaviors")
        if detailed_cat or prohibited_behaviors:
            precautions_text = await precaution_agent.generate_precautions(
                original_title, detailed_cat, prohibited_behaviors
            )
    if not precautions_text:
        precautions_text = "이 축제에 대한 세부 주의사항 정보가 없습니다."

    yield gr.update(value=details_text), original_title, details, gr.update(
        value=precautions_text, visible=True
    ), details  # [수정] selected_item 대신 details(전체 DB 정보)를 반환


# --- [ 신규 ] Wrapper Handlers for New UI ---


def run_search_and_populate_ranking_cbox(
    area, sigungu, main_cat, medium_cat, small_cat, status
):
    """[신규]
    기존 검색 핸들러를 호출하고, 그 결과로 랭킹 선택용 체크박스를 채웁니다.
    """
    # [수정] run_search_and_display가 이제 (9개)를 반환
    (
        results_for_state,
        gallery,
        page_str_updated,
        results_area_update,
        total_pages,
        *page_buttons_updates,  # 5개
        checkbox_update,
    ) = run_search_and_display(area, sigungu, main_cat, medium_cat, small_cat, status)

    # [수정] 반환값에 checkbox_update 추가 (총 9개)
    return (
        results_for_state,
        gallery,
        page_str_updated,
        results_area_update,
        total_pages,
        *page_buttons_updates,
        checkbox_update,
    )


async def handle_gallery_select_and_show_details(evt: gr.SelectData, results, page_str):
    """[*** 수정됨 ***]
    Gradio 경고(UserWarning)를 해결하기 위해 반환값 개수를 7개로 수정합니다.
    """
    # 1. 상세 정보 로드 (비동기 생성자 사용)
    details_outputs = {}
    async for output in display_festival_details_and_precautions(
        evt, results, page_str
    ):
        details_outputs = {
            "details_text": output[0],
            "original_title": output[1],
            "details_dict": output[2],
            "precautions_text": output[3],
            "selected_item": output[
                4
            ],  # [수정] 이제 selected_item은 details 딕셔너리임
        }

    # 2. 페이지 이동 (탭 ID "details" 반환)
    page_nav_output = gr.update(selected="details")

    # 3. 상세 페이지 탭 초기화 (선택된 탭 없음)
    tab_select_output = gr.update(selected=None)

    # 4. 모든 UI 업데이트 반환 (총 7개)
    return (
        details_outputs["details_text"],  # 1: festival_details_output
        details_outputs["original_title"],  # 2: selected_festival_state
        details_outputs["details_dict"],  # 3: selected_festival_details_state
        details_outputs["precautions_text"],  # 4: precautions_output
        details_outputs["selected_item"],  # 5: temp_selection_state
        page_nav_output,  # 6: top_level_tabs
        tab_select_output,  # 7: detail_tabs_container
    )


# --- (이하 모든 기존 핸들러는 수정 없이 그대로 사용) ---


async def get_naver_review_info(festival_name, num_reviews):
    """(기존 핸들러 - 수정 없음)"""
    if not festival_name:
        yield gr.update(value="먼저 축제를 선택해주세요.")
        return
    yield gr.update(
        value=f"{festival_name} 후기 검색 중... ({num_reviews}개)", visible=True
    )
    summary, _ = await naver_supervisor.get_review_summary_and_tips(
        festival_name, num_reviews=num_reviews
    )
    yield gr.update(value=summary, visible=True)


async def handle_generate_trend_graphs(festival_name):
    """(기존 핸들러 - 출력값만 수정)"""
    yield gr.update(value="트렌드 그래프 생성 중...", visible=True), None, None
    trend_image_yearly, trend_image_event, status_message = (
        await analysis_use_case.generate_trend_graphs(festival_name)
    )
    if trend_image_yearly is None and trend_image_event is None:
        yield gr.update(value=status_message, visible=True), None, None
    else:
        yield gr.update(visible=False), trend_image_yearly, trend_image_event


async def handle_generate_word_cloud(festival_name, num_reviews):
    """(기존 핸들러 - 출력값만 수정)"""
    yield gr.update(
        value=f"워드 클라우드 생성 중... ({num_reviews}개)", visible=True
    ), None
    wc_image, status_message = await analysis_use_case.generate_word_cloud(
        festival_name, num_reviews
    )
    if wc_image is None:
        yield gr.update(value=status_message, visible=True), None
    else:
        yield gr.update(visible=False), wc_image


async def handle_scrape_images(festival_name, num_blogs):
    """(기존 핸들러 - 출력값만 수정)"""
    local_image_paths, urls = await analysis_use_case.scrape_festival_images(
        festival_name, num_blogs
    )
    return gr.update(value=local_image_paths, visible=True), urls


async def handle_analyze_sentiment(festival_name, num_reviews):
    """(기존 핸들러 - 수정 없음)"""
    outputs_to_clear = (
        [gr.update(value="")]
        + [gr.update(visible=False)] * 21
        + [None, None, None, 1, "/ 1"]
        + [gr.update(visible=False)] * 5
    )
    if not festival_name:
        outputs_to_clear[0] = gr.update(value="축제를 선택해주세요.")
        yield tuple(outputs_to_clear)
        return
    try:
        outputs_to_clear[0] = gr.update(value="블로그 검색 및 분석 중...")
        yield tuple(outputs_to_clear)
        result = await sentiment_analysis_use_case.analyze_sentiment(
            festival_name, num_reviews
        )
        initial_page_df, current_page, total_pages_str = change_page(
            result["blog_df"], 1
        )
        seasonal_charts = result["seasonal_charts"]
        seasonal_pos_wc = result["seasonal_pos_wc_paths"]
        seasonal_neg_wc = result["seasonal_neg_wc_paths"]
        distribution_description = f"이 분포표는 **{len(result['blog_df'])}**개의 블로그 후기를 분석한 결과입니다..."
        outlier_description = f"총 **{result['total_score_count']}**개의 감성 점수 중 **{result['outlier_count']}**개의 이상치가 발견..."

        yield (
            "분석 완료",
            gr.update(
                value=result["neg_summary_text"],
                visible=bool(result["neg_summary_text"]),
            ),
            gr.update(value=result["overall_chart"], visible=True),
            gr.update(
                value=result["distribution_chart"],
                visible=result["distribution_chart"] is not None,
            ),
            gr.update(value=distribution_description, visible=True),
            gr.update(
                value=result["outlier_chart"],
                visible=result["outlier_chart"] is not None,
            ),
            gr.update(value=outlier_description, visible=True),
            gr.update(
                value=result["positive_keywords_html"],
                visible=bool(result["positive_keywords_html"]),
            ),
            gr.update(value=result["overall_summary_text"], visible=True),
            gr.update(
                value=result["summary_csv_path"],
                visible=result["summary_csv_path"] is not None,
            ),
            *[
                gr.update(value=seasonal_charts.get(s), visible=s in seasonal_charts)
                for s in ["봄", "여름", "가을", "겨울"]
            ],
            *[
                item
                for s in ["봄", "여름", "가을", "겨울"]
                for item in (
                    gr.update(
                        value=seasonal_pos_wc.get(s), visible=s in seasonal_pos_wc
                    ),
                    gr.update(
                        value=seasonal_neg_wc.get(s), visible=s in seasonal_neg_wc
                    ),
                )
            ],
            initial_page_df,
            result["blog_df"],
            result["blog_judgments_list"],
            current_page,
            total_pages_str,
            gr.update(
                value=result["blog_list_csv_path"],
                visible=result["blog_list_csv_path"] is not None,
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, open=False),
        )
        return
    except Exception as e:
        print(f"감성 분석 중 예외 발생: {e}")
        traceback.print_exc()
        outputs_to_clear[0] = f"분석 중 오류 발생: {e}"
        yield tuple(outputs_to_clear)
        return


async def handle_rank_festivals(
    festivals_list, num_reviews, top_n, progress=gr.Progress()
):
    """[*** 수정됨 ***]
    랭킹 결과를 표시할 때도 display_page를 호출하고,
    페이지네이션된 체크박스 목록을 반환합니다.
    """
    print(
        f"DEBUG: Entering handle_rank_festivals. Festivals list length: {len(festivals_list) if festivals_list else 0}, Num reviews: {num_reviews}, Top N: {top_n}"
    )
    if not festivals_list:
        print("DEBUG: festivals_list is empty in handle_rank_festivals.")
        # [수정] 반환값에 1개(checkbox_update) 추가 (총 9개)
        return (
            [],
            gr.update(value=[]),
            "1 / 1",
            *([gr.update(visible=False)] * 5),
            gr.update(value="", visible=False),
            1,
            gr.update(choices=[], value=[]),  # 빈 체크박스
        )
    ranked_festivals, report_md = await ranking_use_case.rank_festivals(
        festivals_list=festivals_list,
        num_reviews=num_reviews,
        top_n=top_n,
        progress=progress,
    )
    print(f"DEBUG: ranked_festivals length after ranking: {len(ranked_festivals)}")

    # [수정] 랭킹 결과 1페이지에 대한 갤러리/체크박스 가져오기
    gallery_output, page_display_str, *page_buttons_updates, checkbox_update = (
        display_page(ranked_festivals, 1)
    )

    total_pages = (
        math.ceil(len(ranked_festivals) / PAGE_SIZE) if ranked_festivals else 1
    )
    report_update = gr.update(value=report_md, visible=True)

    # [수정] 반환값에 체크박스 업데이트 추가 (총 9개)
    return (
        ranked_festivals,
        gallery_output,
        page_display_str,
        *page_buttons_updates,
        report_update,
        total_pages,
        checkbox_update,  # 1페이지에 맞는 체크박스
    )


async def handle_rank_selected_festivals(
    all_results_state, selected_titles, num_reviews, top_n, progress=gr.Progress()
):
    """[*** 수정됨 ***] (Q1 답변)
    handle_rank_festivals의 변경된 반환값을 모두 전달합니다.
    """
    if not selected_titles:
        empty_updates = (
            [],
            gr.update(value=[]),
            "1 / 1",
            *([gr.update(visible=False)] * 5),
            gr.update(value="순위를 매길 축제를 선택해주세요.", visible=True),
            1,
            gr.update(choices=[], value=[]),  # 빈 체크박스
        )
        return empty_updates

    festivals_to_rank = [
        item for item in all_results_state if item.get("title") in selected_titles
    ]

    # [수정] handle_rank_festivals가 이제 9개를 반환
    (
        ranked_festivals,
        gallery_output,
        page_display_str,
        *page_buttons_updates,
        report_update,
        total_pages,
        checkbox_update,
    ) = await handle_rank_festivals(
        festivals_list=festivals_to_rank,
        num_reviews=num_reviews,
        top_n=top_n,
        progress=progress,
    )

    # [수정] 반환값에 체크박스 업데이트 추가 (총 9개)
    return (
        ranked_festivals,
        gallery_output,
        page_display_str,
        *page_buttons_updates,
        report_update,
        total_pages,
        checkbox_update,  # 1페이지에 맞는 체크박스
    )


def run_nearby_search(festival_details, radius_meters):
    """(기존 핸들러 - 수정 없음)"""
    if (
        not festival_details
        or not festival_details.get("mapx")
        or not festival_details.get("mapy")
    ):
        return (
            [],
            [],
            [],
            gr.update(value="좌표 정보가 없어 추천할 수 없습니다.", visible=True),
            [],
            [],
            [],
            gr.update(visible=False),
        )
    initial_state = {
        "search_type": "nearby_search",
        "latitude": festival_details.get("mapy"),
        "longitude": festival_details.get("mapx"),
        "radius": radius_meters,
        "current_festival_id": festival_details.get("contentid"),
    }
    final_state = db_search_graph.invoke(initial_state)
    facilities_recs = final_state.get("recommended_facilities", [])
    courses_recs = final_state.get("recommended_courses", [])
    festivals_recs = final_state.get("recommended_festivals", [])
    if not facilities_recs and not courses_recs and not festivals_recs:
        return (
            [],
            [],
            [],
            gr.update(
                value=f"{radius_meters}m 내에 추천할 장소가 없습니다.", visible=True
            ),
            [],
            [],
            [],
            gr.update(visible=False),
        )
    facility_gallery = [
        (item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"])
        for item in facilities_recs
    ]
    course_gallery = [
        (item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"])
        for item in courses_recs
    ]
    festival_gallery = [
        (item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL, item["title"])
        for item in festivals_recs
    ]
    return (
        facilities_recs,
        courses_recs,
        festivals_recs,
        gr.update(visible=False),
        facility_gallery,
        course_gallery,
        festival_gallery,
        gr.update(visible=True),
    )


async def handle_rank_places(
    places_list, num_reviews, top_n, is_course, progress=gr.Progress()
):
    """(기존 핸들러 - 수정 없음)"""
    ranked_places, status, gallery, report = await ranking_use_case.rank_places(
        places_list=places_list,
        num_reviews=num_reviews,
        top_n=top_n,
        progress=progress,
        is_course=is_course,
    )
    return ranked_places, status, gallery, gr.update(value=report, visible=True)


def handle_item_selection(evt: gr.SelectData, items: list):
    """(기존 핸들러 - 수정 없음)
    추천 목록(시설, 코스, 축제)에서 항목 선택 시 임시 상태(temp_selection_state)를 업데이트합니다.
    """
    if not evt or not items or evt.index >= len(items):
        return (
            None,
            gr.update(value="오류: 항목을 찾을 수 없습니다."),
            gr.update(visible=False),
        )
    selected_item = items[evt.index]
    details_list = []
    exclude_cols = [
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
        "ranking_score",
        "time_score",
        "sentiment_score",
        "quarterly_trend_score",
        "yearly_trend_score",
        "sub_points",
        "trend_reason",
        "sentiment_reason",
    ]
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
        all_overviews = [
            sp.get("subdetailoverview")
            for sp in sub_points
            if sp.get("subdetailoverview")
        ]
        if all_overviews:
            overview_list_str = [
                f"{i+1}. {desc}" for i, desc in enumerate(all_overviews)
            ]
            details_list.append(f"**세부 코스개요**:\n" + "\n".join(overview_list_str))
    details_text = "\n\n".join(details_list)
    return (
        selected_item,
        gr.update(value=details_text),
        gr.update(visible=True, open=True),
    )


def add_to_my_course(item_to_add, current_course: list):
    """[수정됨]
    '내 코스'에 항목을 추가하고, 사용자에게 알림을 표시합니다.
    """
    if item_to_add and not any(
        c["title"] == item_to_add.get("title") for c in current_course
    ):
        current_course.append(item_to_add)
        gr.Info(
            f"'{item_to_add.get('title')}' 항목을 내 코스에 추가했습니다!"
        )  # 알림 추가
    elif not item_to_add:
        gr.Warning("추가할 항목이 없습니다. 먼저 항목을 선택해주세요.")  # 경고 추가
    else:
        gr.Info(
            f"'{item_to_add.get('title')}' 항목은 이미 코스에 있습니다."
        )  # 알림 추가

    course_html = "<h3>🚗 나만의 코스</h3>"
    if not current_course:
        course_html += "<p>나만의 코스에 항목을 추가해보세요.</p>"
    else:
        for i, item in enumerate(current_course):
            details_html = ""
            exclude_cols = [
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
                "distance",
                "distance_score",
                "quarterly_trend_score",
                "yearly_trend_score",
                "sentiment_score",
                "ranking_score",
                "trend_reason",
                "sentiment_reason",
                "sub_points",
            ]
            for key, value in item.items():
                if key not in exclude_cols and value and str(value).strip():
                    display_key = COLUMN_TRANSLATIONS.get(key, key)
                    details_html += f"<b>{display_key}</b>: {value}<br>"
            course_html += f"""<details><summary><b>{i+1}. {item.get('title', '이름 없음')}</b></summary><div style="padding: 10px; border: 1px solid #eee; margin-top: 5px;">{details_html}</div></details>"""
    return current_course, course_html


def clear_my_course():
    """(기존 핸들러 - 수정 없음)"""
    return [], "<h3>🚗 나만의 코스</h3><p>나만의 코스에 항목을 추가해보세요.</p>"


async def handle_validate_course(course, duration):
    """(기존 핸들러 - 수정 없음)"""
    if not course or not duration:
        yield "코스나 여행 기간을 입력해주세요."
        return
    yield "검증 중입니다... ⏳"
    initial_state = {"course": course, "duration": duration}
    loop = asyncio.get_running_loop()
    final_state = await loop.run_in_executor(
        None, course_validation_graph.invoke, initial_state
    )
    yield final_state.get("validation_result", "검증 결과를 가져오는 데 실패했습니다.")


def handle_df_select(
    evt: gr.SelectData, page_num: int, df: pd.DataFrame, judgments: list
):
    """(기존 핸들로 - 수정 없음)"""
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
        judgments_for_row, f"{df.iloc[global_idx]['블로그 제목'][:20]}... 문장별 점수"
    )
    summary_text = df.iloc[global_idx]["긍/부정 문장 요약"]
    return (
        gr.update(value=donut_chart, visible=True),
        gr.update(value=score_chart, visible=True),
        gr.update(value=summary_text, visible=True),
        gr.update(open=True, visible=True),
    )


# --- [ 5. 신규 핸들러 함수 추가 ] ---
async def handle_generate_rendering(
    festival_details, progress=gr.Progress(track_tqdm=True)
):
    """AI 렌더링 탭의 실행 버튼 핸들러"""
    if not festival_details:
        # [수정] 'return'을 'yield'로 변경
        yield (
            gr.update(value="오류: 축제를 먼저 선택해주세요.", visible=True),
            None,
            None,
        )
        return  # yield로 값을 보낸 후, 함수를 종료하기 위해 bare return 사용

    # festival_details (selected_festival_details_state)는 이제 DB에서 조회한 전체 dict입니다.
    # handle_gallery_select_and_show_details 핸들러에서
    # output[2](details_dict)와 output[4](selected_item)가 모두
    # DB 조회 결과(details)를 받도록 수정되었기 때문입니다.

    yield gr.update(value="렌더링 중... (최대 1-2분 소요)", visible=True), None, None

    try:
        # 3단계에서 만든 UseCase 호출
        result_paths = await rendering_use_case.generate_festival_renderings(
            festival_details, progress=progress.update
        )

        rep_img_path = result_paths.get("representative")
        cond_img_paths = result_paths.get("conditional", [])

        # 갤러리 형식으로 변환: (이미지경로, 캡션)
        cond_gallery = [
            (path, os.path.basename(path)) for path in cond_img_paths if path
        ]

        yield gr.update(value="렌더링 완료!", visible=True), rep_img_path, cond_gallery

    except Exception as e:
        print(f"--- 렌더링 핸들러 오류 ---")
        print(f"축제: {festival_details.get('title', 'N/A')}")
        print(f"오류: {e}")
        traceback.print_exc()
        print(f"------------------------")

        # [수정] 'return'을 'yield'로 변경
        yield gr.update(value=f"렌더링 실패: {e}", visible=True), None, None


# --- [ 신규 핸들러 추가 끝 ] ---
