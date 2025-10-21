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
from setup_database import load_data_to_db
from naver_review_supervisor import NaverReviewSupervisor
from modules.naver_search.naver_review import get_naver_trend

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
        final_clauses = [clause for i, clause in enumerate(where_clauses) if i < len(valid_params)]
        if final_clauses:
            query += " WHERE " + " AND ".join(final_clauses)
            cursor.execute(query, valid_params)
        else:
            cursor.execute(query)
    else:
        cursor.execute(query)
        
    results = sorted([(row[0], row[1] or NO_IMAGE_URL) for row in cursor.fetchall()])
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
        return dict(festival) if festival else None
    finally:
        conn.close()

def display_festival_details(evt: gr.SelectData, results, page_str):
    page = int(page_str.split('/')[0].strip())
    global_index = (page - 1) * PAGE_SIZE + evt.index
    selected_title = results[global_index][0]
    details = search_festival_in_db(selected_title)

    if not details: return "정보를 찾을 수 없습니다.", "", "", "", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    basic_info = f"**축제명**: {details.get('title', 'N/A')}\n**주소**: {details.get('addr1', 'N/A')}\n**전화번호**: {details.get('tel', 'N/A')}"
    detailed_info = f"**시작일**: {details.get('eventstartdate', 'N/A')}\n**종료일**: {details.get('eventenddate', 'N/A')}\n**행사 장소**: {details.get('eventplace', 'N/A')}"
    overview_text = details.get('overview', '정보 없음')
    event_content_text = details.get('행사내용', '정보 없음')
    return basic_info, detailed_info, overview_text, event_content_text, selected_title, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

async def get_naver_review_info(festival_name):
    if not festival_name: 
        yield gr.update(value="먼저 축제를 선택해주세요.", visible=True), gr.update(visible=False)
        return
    
    yield gr.update(value=f"{festival_name} 후기 검색 중...", visible=True), gr.update(visible=True, open=True)
    summary, _ = await naver_supervisor.get_review_summary_and_tips(festival_name)
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
    if trend_data_yearly:
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
            if trend_data_event:
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

async def generate_word_cloud(festival_name):
    if not festival_name:
        yield gr.update(visible=False), gr.update(value="축제를 선택해주세요.", visible=True), None
        return

    if WordCloud is None or Okt is None or np is None:
        yield gr.update(visible=True, open=True), gr.update(value="`wordcloud`, `konlpy`, 또는 `numpy` 라이브러리가 설치되지 않았습니다.", visible=True), None
        return

    yield gr.update(visible=True, open=True), gr.update(value="워드 클라우드 생성 중...", visible=True), None

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
                mask_array = 255 - mask_array
            except Exception as e:
                print(f"Error loading mask image: {e}")

    stopwords = {'축제', '오늘', '여기', '저희', '이번', '진짜', '정말', '완전', '후기', '위해', '때문', '하나'}
    _, review_texts = await naver_supervisor.get_review_summary_and_tips(festival_name, num_reviews=20, return_full_text=True)
    
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
    if not festival_name:
        yield gr.update(visible=False), "축제를 선택해주세요.", "", pd.DataFrame()
        return

    if naver_supervisor.llm is None:
        yield gr.update(visible=True, open=True), "LLM이 초기화되지 않았습니다.", "", pd.DataFrame()
        return

    yield gr.update(visible=True, open=True), "후기 수집 및 분석 중...", "", pd.DataFrame()

    _, review_texts_with_meta = await naver_supervisor.get_review_summary_and_tips(festival_name, num_reviews=int(num_reviews), return_full_text=True, return_meta=True)

    if not review_texts_with_meta:
        yield gr.update(visible=True, open=True), "분석할 후기 내용이 없습니다.", "", pd.DataFrame()
        return

    blog_results = []
    total_sentiments = []
    for i, blog in enumerate(review_texts_with_meta):
        yield gr.update(visible=True, open=True), f"후기 {i+1}/{len(review_texts_with_meta)} 분석 중...", "", pd.DataFrame(blog_results) if blog_results else pd.DataFrame()
        
        judgments = await naver_supervisor.get_sentiment_for_text(blog['content'])
        if not judgments:
            continue

        total_sentiments.extend(judgments)
        pos_count = sum(1 for j in judgments if j['classification'] == '긍정')
        neg_count = sum(1 for j in judgments if j['classification'] == '부정')
        total_count = pos_count + neg_count
        avg_score = sum(j['score'] for j in judgments) / len(judgments) if judgments else 5.0

        blog_results.append({
            "블로그 제목": blog['title'],
            "평균 점수": f"{avg_score:.2f}",
            "긍정 문장 수": pos_count,
            "부정 문장 수": neg_count,
            "긍정 비율 (%)": f"{(pos_count / total_count * 100):.1f}" if total_count > 0 else "0.0"
        })

    if not blog_results:
        yield gr.update(visible=True, open=True), "감성 분석 결과를 얻지 못했습니다.", "", pd.DataFrame()
        return

    # Aggregate results
    overall_avg_score = sum(s['score'] for s in total_sentiments) / len(total_sentiments) if total_sentiments else 5.0
    overall_class_counts = Counter(s['classification'] for s in total_sentiments)
    
    summary_report = f"### '{festival_name}' 후기 감성 분석 요약\n\n"
    summary_report += f"- **총 분석 후기 수**: {len(blog_results)}개\n"
    summary_report += f"- **전체 문장 평균 점수**: {overall_avg_score:.2f} / 10.0\n"
    summary_report += f"- **감성 분포**: 긍정 {overall_class_counts.get('긍정', 0)}개, 부정 {overall_class_counts.get('부정', 0)}개, 중립 {overall_class_counts.get('중립', 0)}개"

    yield gr.update(visible=True, open=True), "", summary_report, pd.DataFrame(blog_results)

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
        with gr.Row():
            naver_search_btn = gr.Button("네이버 후기 요약 검색", visible=False)
            trend_graph_btn = gr.Button("트렌드 그래프 생성", visible=False)
            word_cloud_btn = gr.Button("워드 클라우드 생성", visible=False)
            # Button removed, new accordion added below

    with gr.Accordion("Naver 후기 요약 및 꿀팁", open=False, visible=False) as naver_review_accordion:
        naver_review_output = gr.Markdown()
    
    with gr.Accordion("검색량 트렌드 그래프", open=False, visible=False) as trend_accordion:
        trend_status = gr.Textbox(label="상태", interactive=False)
        with gr.Row():
            trend_plot_yearly = gr.Image(label="최근 1년 검색량 트렌드")
            trend_plot_event = gr.Image(label="축제 기간 중심 트렌드")

    with gr.Accordion("워드 클라우드", open=False, visible=False) as wordcloud_accordion:
        wordcloud_status = gr.Textbox(label="상태", interactive=False)
        wordcloud_plot = gr.Image(label="축제의 주요 핵심 요소들")

    with gr.Accordion("감성 분석", open=False, visible=False) as sentiment_accordion:
        with gr.Row():
            num_reviews_slider = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="분석할 후기 수")
            run_sentiment_btn = gr.Button("감성 분석 실행", variant="primary")
        sentiment_status = gr.Textbox(label="상태", interactive=False)
        sentiment_summary = gr.Markdown(label="분석 요약")
        sentiment_df_output = gr.DataFrame(label="블로그별 상세 분석")

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
        outputs=[basic_info_output, detailed_info_output, overview_output, content_output, selected_festival_state, naver_search_btn, trend_graph_btn, word_cloud_btn, sentiment_accordion]
    ).then(lambda: gr.update(open=True), outputs=details_accordion)
    
    naver_search_btn.click(
        fn=get_naver_review_info, 
        inputs=[selected_festival_state],
        outputs=[naver_review_output, naver_review_accordion]
    )

    trend_graph_btn.click(
        fn=generate_trend_graphs,
        inputs=[selected_festival_state],
        outputs=[trend_accordion, trend_status, trend_plot_yearly, trend_plot_event]
    )

    word_cloud_btn.click(
        fn=generate_word_cloud,
        inputs=[selected_festival_state],
        outputs=[wordcloud_accordion, wordcloud_status, wordcloud_plot]
    )

    run_sentiment_btn.click(
        fn=analyze_sentiment,
        inputs=[selected_festival_state, num_reviews_slider],
        outputs=[sentiment_accordion, sentiment_status, sentiment_summary, sentiment_df_output]
    )

if __name__ == "__main__":
    print("\n[app.py] Forcing database creation/update...")
    load_data_to_db()
    print("[app.py] Database creation/update complete.")
    ALL_FESTIVAL_CATEGORIES = load_festival_categories_and_maps()
    
    demo.launch(allowed_paths=["assets"])