# src/application/use_cases/sentiment_analysis_use_case.py

import os
import re
import pandas as pd
from datetime import datetime
import traceback

# Custom Module Imports
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor
from src.infrastructure.external_services.naver_search.naver_review_api import search_naver_blog
from src.application.core.graph import app_llm_graph
from src.application.core.utils import get_season, save_df_to_csv, summarize_negative_feedback
from src.infrastructure.reporting.charts import create_donut_chart, create_stacked_bar_chart
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.application.core.constants import CATEGORY_TO_ICON_MAP

class SentimentAnalysisUseCase:
    def __init__(self, naver_supervisor: NaverReviewSupervisor, script_dir: str):
        self.naver_supervisor = naver_supervisor
        self.script_dir = script_dir

    async def analyze_sentiment(self, festival_name: str, num_reviews: int):
        if not festival_name:
            raise ValueError("축제를 선택해주세요.")

        search_keyword = f"{festival_name} 후기"
        
        blog_results_list = []
        blog_judgments_list = []
        all_negative_sentences = []
        seasonal_aspect_pairs = {"봄": [], "여름": [], "가을": [], "겨울": [], "정보없음": []}
        seasonal_data = {"봄": {"pos": 0, "neg": 0}, "여름": {"pos": 0, "neg": 0}, "가을": {"pos": 0, "neg": 0}, "겨울": {"pos": 0, "neg": 0}, "정보없음": {"pos": 0, "neg": 0}}
        total_pos, total_neg, total_strong_pos, total_strong_neg = 0, 0, 0, 0
        
        start_index = 1
        max_results_to_scan = 100 
        display_count = 20
        consecutive_skips = 0

        while len(blog_results_list) < num_reviews and start_index < max_results_to_scan:
            api_results = search_naver_blog(
                search_keyword, 
                display=display_count, 
                start=start_index
            )
            
            if not api_results:
                break

            candidate_blogs = []
            for item in api_results:
                if "blog.naver.com" in item["link"]:
                    item["title"] = re.sub(r"<[^>]+>", "", item["title"]).strip()
                    if item["title"] and item["link"]:
                        candidate_blogs.append(item)
            
            for blog_data in candidate_blogs:
                if len(blog_results_list) >= num_reviews:
                    break
                try:
                    content, _ = await self.naver_supervisor._scrape_blog_content(blog_data["link"])
                    if not content or "오류" in content or "찾을 수 없습니다" in content:
                        consecutive_skips += 1
                        continue

                    max_content_length = 30000
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "... (내용 일부 생략)"

                    final_state = app_llm_graph.invoke({
                        "original_text": content, "keyword": festival_name, "title": blog_data["title"],
                        "log_details": False, "re_summarize_count": 0, "is_relevant": False,
                    })

                    if not final_state or not final_state.get("is_relevant"):
                        consecutive_skips += 1
                        continue

                    judgments = final_state.get("final_judgments", [])
                    if not judgments:
                        consecutive_skips += 1
                        continue

                    consecutive_skips = 0

                    season = get_season(blog_data.get("postdate", ""))
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
                    pos_perc = (pos_count / sentiment_frequency * 100) if sentiment_frequency > 0 else 0.0
                    neg_perc = (neg_count / sentiment_frequency * 100) if sentiment_frequency > 0 else 0.0

                    blog_results_list.append({
                        "블로그 제목": blog_data["title"], "링크": blog_data["link"], "감성 빈도": sentiment_frequency,
                        "감성 점수": f"{sentiment_score:.1f}", "긍정 문장 수": pos_count, "부정 문장 수": neg_count,
                        "긍정 비율 (%)": f"{pos_perc:.1f}", "부정 비율 (%)": f"{neg_perc:.1f}",
                        "긍/부정 문장 요약": "\n---\n".join([f"[{res['final_verdict']}] {res['sentence']}" for res in judgments]),
                    })
                except Exception as e:
                    print(f"블로그 분석 중 오류 ({festival_name}, {blog_data.get('link', 'N/A')}): {e}")
                    traceback.print_exc()
                    consecutive_skips += 1
                    continue
                
                if consecutive_skips >= 3:
                    break
            
            if len(blog_results_list) >= num_reviews or consecutive_skips >= 3:
                break
            
            start_index += display_count

        if not blog_results_list:
            raise ValueError(f"'{festival_name}'에 대한 유효한 후기 블로그를 찾지 못했습니다.")

        total_sentiment_frequency = total_pos + total_neg
        total_sentiment_score = ((total_strong_pos - total_strong_neg) / total_sentiment_frequency * 50 + 50) if total_sentiment_frequency > 0 else 50.0

        neg_summary_text = summarize_negative_feedback(all_negative_sentences)
        overall_summary_text = f"- **긍정 문장 수**: {total_pos}개\n- **부정 문장 수**: {total_neg}개\n- **감성어 빈도 (긍정+부정)**: {total_sentiment_frequency}개\n- **감성 점수**: {total_sentiment_score:.1f}점 (0~100점)"

        summary_df = pd.DataFrame([{"축제명": festival_name, "감성 빈도": total_sentiment_frequency, "감성 점수": f"{total_sentiment_score:.1f}", "긍정 문장 수": total_pos, "부정 문장 수": total_neg}])
        summary_csv_path = save_df_to_csv(summary_df, "overall_summary", festival_name)
        
        blog_df = pd.DataFrame(blog_results_list)
        blog_list_csv_path = save_df_to_csv(blog_df, "blog_list", festival_name)

        overall_chart = create_donut_chart(total_pos, total_neg, f"{festival_name} 전체 후기 요약")
        
        seasonal_charts = {season: create_stacked_bar_chart(data["pos"], data["neg"], f"{season} 시즌") for season, data in seasonal_data.items() if data["pos"] > 0 or data["neg"] > 0}

        seasonal_pos_wc_paths = {}
        seasonal_neg_wc_paths = {}
        for season, pairs in seasonal_aspect_pairs.items():
            # Correctly determine season_en from the season name itself
            season_en_map = {"봄": "spring", "여름": "summer", "가을": "fall", "겨울": "winter"}
            season_en = season_en_map.get(season)
            
            if pairs and season_en:
                # Construct the mask path using an absolute path from the script directory
                mask_path = os.path.abspath(os.path.join(self.script_dir, "assets", "seasons", f"mask_{season_en}.png"))
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file not found at {mask_path}")
                    mask_path = None
                
                pos_path, neg_path = create_sentiment_wordclouds(pairs, f"{festival_name}_{season}", mask_path=mask_path)
                seasonal_pos_wc_paths[season] = pos_path
                seasonal_neg_wc_paths[season] = neg_path

        return {
            "neg_summary_text": neg_summary_text,
            "overall_summary_text": overall_summary_text,
            "summary_csv_path": summary_csv_path,
            "blog_df": blog_df,
            "blog_judgments_list": blog_judgments_list,
            "blog_list_csv_path": blog_list_csv_path,
            "overall_chart": overall_chart,
            "seasonal_charts": seasonal_charts,
            "seasonal_pos_wc_paths": seasonal_pos_wc_paths,
            "seasonal_neg_wc_paths": seasonal_neg_wc_paths,
        }
