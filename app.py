import sqlite3
import json
import os
import gradio as gr
import re
import math
from setup_database import load_data_to_db
from naver_review_supervisor import NaverReviewSupervisor
import asyncio

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

CAT_NAME_TO_CODE = {'main': {}, 'medium': {}, 'small': {}}

def load_festival_categories_and_maps():
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

    title_to_cat_names = {}
    for main_name, med_dict in all_categories.items():
        for med_name, small_dict in med_dict.items():
            for small_name, titles in small_dict.items():
                for title in titles:
                    title_to_cat_names[title] = (main_name, med_name, small_name)

    db_path = os.path.join(script_dir, "tour.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT title, cat1, cat2, cat3 FROM festivals WHERE cat1 IS NOT NULL AND cat2 IS NOT NULL AND cat3 IS NOT NULL")
    db_festivals = cursor.fetchall()
    conn.close()

    for row in db_festivals:
        title, code1, code2, code3 = row
        if title in title_to_cat_names:
            name1, name2, name3 = title_to_cat_names[title]
            CAT_NAME_TO_CODE['main'][name1] = code1
            CAT_NAME_TO_CODE['medium'][name2] = code2
            CAT_NAME_TO_CODE['small'][name3] = code3
    
    print("[app.py] Category name-to-code maps created.")
    return all_categories

ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()

# --- Search and Display Logic ---

def search_festivals(area, sigungu, main_cat, medium_cat, small_cat):
    where_clauses = []
    params = []

    if area and area != "전체":
        where_clauses.append("areacode = ?")
        params.append(AREA_CODE_MAP.get(area))
        if sigungu and sigungu != "전체":
            where_clauses.append("sigungucode = ?")
            params.append(SIGUNGU_CODE_MAP.get(area, {}).get(sigungu))

    if main_cat and main_cat != "전체":
        cat1_code = CAT_NAME_TO_CODE['main'].get(main_cat)
        if cat1_code:
            where_clauses.append("cat1 = ?")
            params.append(cat1_code)
        if medium_cat and medium_cat != "전체":
            cat2_code = CAT_NAME_TO_CODE['medium'].get(medium_cat)
            if cat2_code:
                where_clauses.append("cat2 = ?")
                params.append(cat2_code)
                if small_cat and small_cat != "전체":
                    cat3_code = CAT_NAME_TO_CODE['small'].get(small_cat)
                    if cat3_code:
                        where_clauses.append("cat3 = ?")
                        params.append(cat3_code)
    
    db_path = os.path.join(script_dir, "tour.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT title, firstimage FROM festivals"
    valid_params = [p for p in params if p is not None]

    if where_clauses and valid_params:
        query += " WHERE " + " AND ".join(where_clauses)
        cursor.execute(query, valid_params)
    else:
        cursor.execute(query)
        
    results = [(row[0], row[1] or NO_IMAGE_URL) for row in cursor.fetchall()]
    conn.close()

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

def change_page(results, page_str, direction):
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
        if festival:
            return dict(festival)
        else:
            return None
    finally:
        conn.close()

# Initialize the NaverReviewSupervisor
naver_supervisor = NaverReviewSupervisor()

def display_festival_details(evt: gr.SelectData, results, page_str):
    page = int(page_str.split('/')[0].strip())
    global_index = (page - 1) * PAGE_SIZE + evt.index
    selected_title = results[global_index][0]
    details = search_festival_in_db(selected_title)
    
    if not details: return "정보를 찾을 수 없습니다.", "", "", ""

    basic_info = f"**축제명**: {details.get('title', 'N/A')}\n**주소**: {details.get('addr1', 'N/A')}\n**전화번호**: {details.get('tel', 'N/A')}"
    detailed_info = f"**시작일**: {details.get('eventstartdate', 'N/A')}\n**종료일**: {details.get('eventenddate', 'N/A')}\n**행사 장소**: {details.get('eventplace', 'N/A')}"
    overview_text = details.get('overview', '정보 없음')
    event_content_text = details.get('행사내용', '정보 없음')
    return basic_info, detailed_info, overview_text, event_content_text

async def get_naver_review_info(evt: gr.SelectData, results, page_str):
    if not evt: yield gr.update(value="갤러리에서 축제를 선택해주세요.", visible=True), gr.update(visible=False); return
    page = int(page_str.split('/')[0].strip())
    global_index = (page - 1) * PAGE_SIZE + evt.index
    festival_name = results[global_index][0]
    yield gr.update(value=f"{festival_name} 후기 검색 중...", visible=True), gr.update(visible=True, open=True)
    summary, _ = await naver_supervisor.get_review_summary_and_tips(festival_name)
    yield gr.update(value=summary, visible=True), gr.update(visible=True, open=True)

# --- Gradio Interface ---

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# 축제 정보 검색 에이전트")

    results_state = gr.State([])
    page_state = gr.State(1)
    
    with gr.Group():
        with gr.Row():
            area_dropdown = gr.Dropdown(label="시/도", choices=["전체"] + sorted(list(AREA_CODE_MAP.keys())), interactive=True)
            sigungu_dropdown = gr.Dropdown(label="시/군/구", choices=["전체"], interactive=True)
        with gr.Row():
            main_cat_dropdown = gr.Dropdown(label="대분류", choices=["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.keys())), interactive=True)
            medium_cat_dropdown = gr.Dropdown(label="중분류", choices=["전체"], interactive=True)
            small_cat_dropdown = gr.Dropdown(label="소분류", choices=["전체"], interactive=True)
    
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
        naver_review_output = gr.Markdown()

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

    prev_button.click(fn=lambda r, p: change_page(r, p, -1), inputs=[results_state, page_display], outputs=[festival_gallery, page_display])
    next_button.click(fn=lambda r, p: change_page(r, p, 1), inputs=[results_state, page_display], outputs=[festival_gallery, page_display])

    festival_gallery.select(
        fn=display_festival_details,
        inputs=[results_state, page_display],
        outputs=[basic_info_output, detailed_info_output, overview_output, content_output]
    ).then(lambda: gr.update(open=True), outputs=details_accordion)
    
    festival_gallery.select(fn=get_naver_review_info, inputs=[results_state, page_display], outputs=[naver_review_output, naver_review_accordion])

if __name__ == "__main__":
    print("\n[app.py] Forcing database creation/update...")
    load_data_to_db()
    print("[app.py] Database creation/update complete.")
    ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()
    
    demo.launch()