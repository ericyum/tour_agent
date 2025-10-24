# src/application/use_cases/analysis_use_case.py

import os
import shutil
import requests
import io
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

# Visualization and NLP
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from wordcloud import WordCloud
from konlpy.tag import Okt

# Custom Module Imports
from src.infrastructure.external_services.naver_search.naver_review_api import get_naver_trend, search_naver_blog
from src.application.services.festival_service import get_festival_details_by_title
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor


class AnalysisUseCase:
    def __init__(self, naver_supervisor: NaverReviewSupervisor, font_path: str, title_to_cat_map: dict, cat_to_icon_map: dict, script_dir: str):
        self.naver_supervisor = naver_supervisor
        self.font_path = font_path
        self.title_to_cat_map = title_to_cat_map
        self.cat_to_icon_map = cat_to_icon_map
        self.script_dir = script_dir
        self.okt = Okt()

    async def generate_trend_graphs(self, festival_name: str):
        if not festival_name:
            return None, None, "축제를 선택해주세요."

        if plt is None:
            return None, None, "`matplotlib` 라이브러리가 설치되지 않았습니다."

        font_properties = font_manager.FontProperties(fname=self.font_path) if self.font_path else None
        details = get_festival_details_by_title(festival_name)

        # --- 1. 1-Year Trend Graph ---
        today = datetime.today()
        start_date_yearly = today - timedelta(days=365)
        trend_data_yearly = get_naver_trend(festival_name, start_date_yearly, today)

        fig_trend_yearly, ax_yearly = plt.subplots(figsize=(10, 5))
        if trend_data_yearly:
            df = pd.DataFrame(trend_data_yearly)
            df["period"] = pd.to_datetime(df["period"])
            ax_yearly.plot(df["period"], df["ratio"])
            ax_yearly.set_title(f"'{festival_name}' 최근 1년 검색량 트렌드", fontproperties=font_properties, fontsize=16)
            ax_yearly.tick_params(axis="x", rotation=30)
        else:
            ax_yearly.text(0.5, 0.5, "트렌드 데이터 없음", ha="center", va="center", fontproperties=font_properties)
        plt.tight_layout()
        buf_trend_yearly = io.BytesIO()
        fig_trend_yearly.savefig(buf_trend_yearly, format="png")
        trend_image_yearly = Image.open(buf_trend_yearly)
        plt.close(fig_trend_yearly)

        # --- 2. Event-Period Trend Graph ---
        fig_trend_event, ax_event = plt.subplots(figsize=(10, 5))
        graph_title = f"'{festival_name}' 축제 시작일 중심 트렌드"
        
        if details and details.get("eventstartdate"):
            date_val = details.get("eventstartdate")
            date_str = str(date_val).split('.')[0]
            center_date = pd.to_datetime(date_str, errors="coerce")

            if pd.notna(center_date):
                trend_data_event = None
                search_date = center_date
                
                # Try to get data for the current year if it's not in the future
                is_future_event = center_date > today
                if not is_future_event:
                    graph_start = center_date - timedelta(days=7)
                    graph_end = center_date + timedelta(days=7)
                    trend_data_event = get_naver_trend(festival_name, graph_start, graph_end)

                # If no data was found (either because it's a future event or data is sparse), fallback to previous years
                if trend_data_event is None:
                    reason_text = "미래 행사" if is_future_event else "데이터 부족"
                    for i in range(1, 4):  # Try for the last 3 years
                        search_year = center_date - pd.DateOffset(years=i)
                        graph_start = search_year - timedelta(days=7)
                        graph_end = search_year + timedelta(days=7)

                        # Do not search for data in the future (a safeguard)
                        if graph_end > today:
                            continue

                        trend_data_event = get_naver_trend(festival_name, graph_start, graph_end)
                        if trend_data_event:
                            search_date = search_year
                            year_text = "작년" if i == 1 else f"{i}년 전"
                            graph_title += f" ({year_text} 데이터, 사유: {reason_text})"
                            break
                
                if trend_data_event:
                    df_event = pd.DataFrame(trend_data_event)
                    df_event["period"] = pd.to_datetime(df_event["period"])
                    
                    # Adjust the date of the plot to match the original festival year for comparison
                    if search_date != center_date:
                        time_diff = center_date - search_date
                        df_event["period"] = df_event["period"] + time_diff

                    ax_event.plot(df_event["period"], df_event["ratio"])
                    ax_event.axvline(x=center_date, color="r", linestyle="--", label="Festival Start")
                    ax_event.legend()
                    ax_event.tick_params(axis="x", rotation=30)
                else:
                    ax_event.text(0.5, 0.5, "기간 트렌드 데이터 없음 (최근 3년간 데이터 부족)", ha="center", va="center", fontproperties=font_properties)
            else:
                ax_event.text(0.5, 0.5, "날짜 형식 오류", ha="center", va="center", fontproperties=font_properties)
        else:
            ax_event.text(0.5, 0.5, "축제 시작일 정보 없음", ha="center", va="center", fontproperties=font_properties)

        ax_event.set_title(graph_title, fontproperties=font_properties, fontsize=16)
        plt.tight_layout()
        buf_trend_event = io.BytesIO()
        fig_trend_event.savefig(buf_trend_event, format="png")
        trend_image_event = Image.open(buf_trend_event)
        plt.close(fig_trend_event)

        return trend_image_yearly, trend_image_event, "트렌드 그래프 생성 완료"

    async def generate_word_cloud(self, festival_name: str, num_reviews: int):
        if not festival_name:
            return None, "축제를 선택해주세요."

        if WordCloud is None or self.okt is None or np is None:
            return None, "`wordcloud`, `konlpy`, 또는 `numpy` 라이브러리가 설치되지 않았습니다."

        main_cat_tuple = self.title_to_cat_map.get(festival_name)
        main_cat = main_cat_tuple[0] if main_cat_tuple else None
        icon_name = None
        if main_cat:
            normalized_cat_name = main_cat.replace(" ", "")
            icon_name = self.cat_to_icon_map.get(normalized_cat_name)

        mask_array = None
        if icon_name:
            path = os.path.join(self.script_dir, "assets", "themes", f"{icon_name}.png")
            if os.path.exists(path):
                try:
                    icon_image = Image.open(path).convert("L")
                    mask_array = np.array(icon_image)
                    mask_array = 255 - mask_array
                except Exception as e:
                    print(f"Error loading mask image: {e}")

        stopwords = {"축제", "오늘", "여기", "저희", "이번", "진짜", "정말", "완전", "후기", "위해", "때문", "하나"}
        _, review_texts = await self.naver_supervisor.get_review_summary_and_tips(
            festival_name, num_reviews=num_reviews, return_full_text=True
        )

        wc_image = None
        if review_texts:
            nouns = [
                word
                for text in review_texts
                for word in self.okt.nouns(text)
                if len(word) > 1 and word not in stopwords
            ]
            counts = Counter(nouns)
            if counts:
                wc = WordCloud(
                    font_path=self.font_path,
                    background_color="white",
                    mask=mask_array,
                    contour_color="steelblue",
                    contour_width=1,
                ).generate_from_frequencies(counts)
                wc_image = wc.to_image()

        if wc_image is None:
            wc_image = Image.new("RGB", (800, 400), "white")
            draw = ImageDraw.Draw(wc_image)
            try:
                font = ImageFont.truetype(self.font_path, 20)
            except:
                font = ImageFont.load_default()
            draw.text((300, 180), "추출된 단어 없음", font=font, fill="black")

        return wc_image, "워드 클라우드 생성 완료"

    async def scrape_festival_images(self, festival_name: str, num_blogs: int):
        if not festival_name:
            return [], ""

        image_save_dir = os.path.join(self.script_dir, "temp_img")
        if os.path.exists(image_save_dir):
            shutil.rmtree(image_save_dir)
        os.makedirs(image_save_dir, exist_ok=True)

        all_image_urls = []
        start_index = 1
        max_results_to_scan = 100
        display_count = 10
        consecutive_skips = 0
        found_blogs_with_images = 0
        target_blog_count = num_blogs

        while found_blogs_with_images < target_blog_count and start_index < max_results_to_scan:
            blog_reviews = search_naver_blog(f"{festival_name} 후기", display=display_count, start=start_index)
            if not blog_reviews:
                break

            for review in blog_reviews:
                if found_blogs_with_images >= target_blog_count:
                    break
                if consecutive_skips >= 3:
                    break

                link = review.get("link")
                if link and "blog.naver.com" in link:
                    text_content, image_urls = await self.naver_supervisor._scrape_blog_content(link)
                    if text_content and "본문 내용을 찾을 수 없습니다" not in text_content and image_urls:
                        is_relevant = await self.naver_supervisor._is_relevant_review(
                            festival_name, review.get("title", ""), text_content
                        )
                        if is_relevant:
                            all_image_urls.extend(image_urls)
                            found_blogs_with_images += 1
                            consecutive_skips = 0
                        else:
                            consecutive_skips += 1
                    else:
                        consecutive_skips += 1
                else:
                    consecutive_skips += 1
            
            if consecutive_skips >= 3:
                break
            
            start_index += display_count

        local_image_paths = []
        for i, img_url in enumerate(all_image_urls):
            try:
                response = requests.get(img_url, stream=True, timeout=10)
                response.raise_for_status()
                file_ext = os.path.splitext(img_url.split("?")[0])[-1]
                if not file_ext or len(file_ext) > 5:
                    file_ext = ".jpg"
                file_name = f"image_{i+1}{file_ext}"
                file_path = os.path.join(image_save_dir, file_name)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                local_image_paths.append(file_path)
            except requests.exceptions.RequestException as e:
                print(f"이미지 다운로드 실패: {img_url}, 오류: {e}")
                continue

        return local_image_paths, "\n".join(all_image_urls)
