# src/application/use_cases/sentiment_analysis_use_case.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Custom Module Imports
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor
from src.infrastructure.external_services.naver_search.naver_review_api import search_naver_blog
from src.application.core.graph import app_llm_graph
from src.application.core.utils import get_season, save_df_to_csv, summarize_negative_feedback
from src.infrastructure.reporting.charts import create_donut_chart, create_stacked_bar_chart, create_score_distribution_histogram, create_outlier_boxplot
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.application.core.constants import CATEGORY_TO_ICON_MAP

class SentimentAnalysisUseCase:
    def __init__(self, naver_supervisor: NaverReviewSupervisor, script_dir: str):
        self.naver_supervisor = naver_supervisor
        self.script_dir = script_dir

    def _calculate_satisfaction_boundaries(self, scores: list) -> dict:
        if not scores:
            return {
                "boundaries": {},
                "filtered_scores": [],
                "outliers": [],
            }

        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_scores = [s for s in scores if lower_bound <= s <= upper_bound]
        outliers = [s for s in scores if s < lower_bound or s > upper_bound]
        
        if not filtered_scores:
            filtered_scores = scores

        mean = np.mean(filtered_scores)
        std = np.std(filtered_scores)

        boundaries = {
            'mean': mean,
            'std': std,
            'very_dissatisfied_upper': mean - 1.5 * std,
            'dissatisfied_upper': mean - 0.5 * std,
            'neutral_upper': mean + 0.5 * std,
            'satisfied_upper': mean + 1.5 * std,
        }
        return {
            "boundaries": boundaries,
            "filtered_scores": filtered_scores,
            "outliers": outliers,
        }

    def _map_score_to_level(self, score: float, boundaries: dict) -> int:
        if not boundaries:
            return 3 # Default to neutral if no boundaries
        if score < boundaries['very_dissatisfied_upper']:
            return 1 # 매우 불만족
        elif score < boundaries['dissatisfied_upper']:
            return 2 # 불만족
        elif score < boundaries['neutral_upper']:
            return 3 # 보통
        elif score < boundaries['satisfied_upper']:
            return 4 # 만족
        else:
            return 5 # 매우 만족

    async def analyze_sentiment(self, festival_name: str, num_reviews: int):
        if not festival_name:
            raise ValueError("축제를 선택해주세요.")

        search_keyword = f"{festival_name} 후기"
        
        blog_results_list = []
        blog_judgments_list = []
        all_scores = []
        all_negative_sentences = []
        seasonal_data = {"봄": {"pos": 0, "neg": 0}, "여름": {"pos": 0, "neg": 0}, "가을": {"pos": 0, "neg": 0}, "겨울": {"pos": 0, "neg": 0}, "정보없음": {"pos": 0, "neg": 0}}
        total_pos, total_neg = 0, 0
        
        start_index = 1
        max_results_to_scan = 100 
        display_count = 20
        consecutive_skips = 0

        while len(blog_results_list) < num_reviews and start_index < max_results_to_scan:
            api_results = search_naver_blog(search_keyword, display=display_count, start=start_index)
            if not api_results:
                break

            candidate_blogs = [item for item in api_results if "blog.naver.com" in item["link"]]
            for blog_data in candidate_blogs:
                if len(blog_results_list) >= num_reviews: break
                try:
                    content, _ = await self.naver_supervisor._scrape_blog_content(blog_data["link"])
                    if not content or "오류" in content or "찾을 수 없습니다" in content:
                        consecutive_skips += 1
                        continue

                    content = content[:30000]

                    final_state = app_llm_graph.invoke({
                        "original_text": content, "keyword": festival_name,
                        "title": re.sub(r"<[^>]+>", "", blog_data["title"]).strip(),
                        "log_details": True,
                    })

                    if not final_state or not final_state.get("is_relevant") or not final_state.get("final_judgments"):
                        consecutive_skips += 1
                        continue
                    
                    consecutive_skips = 0
                    judgments = final_state.get("final_judgments", [])
                    blog_judgments_list.append(judgments)
                    all_scores.extend([j['score'] for j in judgments])
                    
                    blog_results_list.append({
                        "블로그 제목": re.sub(r"<[^>]+>", "", blog_data["title"]).strip(),
                        "링크": blog_data["link"],
                        "postdate": blog_data.get("postdate", ""),
                        "judgments": judgments,
                    })

                except Exception as e:
                    print(f"블로그 분석 중 오류 ({festival_name}, {blog_data.get('link', 'N/A')}): {e}")
                    traceback.print_exc()
                    consecutive_skips += 1
                    continue
                
                if consecutive_skips >= 3: break
            if len(blog_results_list) >= num_reviews or consecutive_skips >= 3: break
            start_index += display_count

        if not blog_results_list:
            raise ValueError(f"'{festival_name}'에 대한 유효한 후기 블로그를 찾지 못했습니다.")

        boundary_results = self._calculate_satisfaction_boundaries(all_scores)
        boundaries = boundary_results["boundaries"]
        outliers = boundary_results["outliers"]

        distribution_chart_path = None
        if all_scores and boundaries:
            fig = create_score_distribution_histogram(all_scores, boundaries, f'{festival_name} 감성 점수 분포')
            if fig:
                distribution_chart_path = os.path.join(self.script_dir, "temp_img", f"dist_chart_{festival_name}.png")
                os.makedirs(os.path.dirname(distribution_chart_path), exist_ok=True)
                fig.savefig(distribution_chart_path)
                plt.close(fig)

        outlier_chart_path = None
        if all_scores:
            fig = create_outlier_boxplot(all_scores, f'{festival_name} 감성 점수 이상치')
            if fig:
                outlier_chart_path = os.path.join(self.script_dir, "temp_img", f"outlier_chart_{festival_name}.png")
                os.makedirs(os.path.dirname(outlier_chart_path), exist_ok=True)
                fig.savefig(outlier_chart_path)
                plt.close(fig)

        processed_blog_results = []
        all_satisfaction_levels = []

        for blog in blog_results_list:
            judgments = blog["judgments"]
            blog_satisfaction_levels = []
            
            pos_count = 0
            neg_count = 0

            for j in judgments:
                level = self._map_score_to_level(j['score'], boundaries)
                j['satisfaction_level'] = level
                blog_satisfaction_levels.append(level)
                all_satisfaction_levels.append(level)
                
                if j['final_verdict'] == '긍정':
                    pos_count += 1
                else:
                    neg_count += 1
                    all_negative_sentences.append(j['sentence'])
            
            season = get_season(blog.get("postdate", ""))
            seasonal_data[season]["pos"] += pos_count
            seasonal_data[season]["neg"] += neg_count
            total_pos += pos_count
            total_neg += neg_count

            avg_satisfaction = np.mean(blog_satisfaction_levels) if blog_satisfaction_levels else 3.0
            pos_perc = (pos_count / (pos_count + neg_count) * 100) if (pos_count + neg_count) > 0 else 0
            neg_perc = (neg_count / (pos_count + neg_count) * 100) if (pos_count + neg_count) > 0 else 0

            processed_blog_results.append({
                "블로그 제목": blog["블로그 제목"],
                "링크": blog["링크"],
                "만족도 점수": f"{avg_satisfaction:.2f} / 5",
                "감성 빈도": len(judgments),
                "긍정 문장 수": pos_count,
                "부정 문장 수": neg_count,
                "긍정 비율 (%)": f"{pos_perc:.1f}",
                "부정 비율 (%)": f"{neg_perc:.1f}",
                "긍/부정 문장 요약": "\n---\n".join([f"[{j['final_verdict']}({j['satisfaction_level']}점)] {j['sentence']}" for j in judgments]),
            })

        overall_avg_satisfaction = np.mean(all_satisfaction_levels) if all_satisfaction_levels else 3.0
        neg_summary_text = summarize_negative_feedback(all_negative_sentences)
        overall_summary_text = f"- **총 분석 블로그**: {len(blog_results_list)}개\n- **전체 평균 만족도**: {overall_avg_satisfaction:.2f} / 5.0 점\n- **긍정 문장 수**: {total_pos}개\n- **부정 문장 수**: {total_neg}개"

        summary_df = pd.DataFrame([{"축제명": festival_name, "평균 만족도": f"{overall_avg_satisfaction:.2f}", "긍정 문장 수": total_pos, "부정 문장 수": total_neg}])
        summary_csv_path = save_df_to_csv(summary_df, "overall_summary", festival_name)
        
        blog_df = pd.DataFrame(processed_blog_results)
        blog_list_csv_path = save_df_to_csv(blog_df, "blog_list", festival_name)

        overall_chart = create_donut_chart(total_pos, total_neg, f"{festival_name} 전체 후기 요약")
        seasonal_charts = {season: create_stacked_bar_chart(data["pos"], data["neg"], f"{season} 시즌") for season, data in seasonal_data.items() if data["pos"] > 0 or data["neg"] > 0}

        seasonal_pos_wc_paths = {}
        seasonal_neg_wc_paths = {}

        return {
            "neg_summary_text": neg_summary_text,
            "overall_summary_text": overall_summary_text,
            "summary_csv_path": summary_csv_path,
            "blog_df": blog_df,
            "blog_judgments_list": blog_judgments_list,
            "blog_list_csv_path": blog_list_csv_path,
            "overall_chart": overall_chart,
            "distribution_chart": distribution_chart_path,
            "seasonal_charts": seasonal_charts,
            "seasonal_pos_wc_paths": seasonal_pos_wc_paths,
            "seasonal_neg_wc_paths": seasonal_neg_wc_paths,
            "outlier_chart": outlier_chart_path,
            "total_score_count": len(all_scores),
            "outlier_count": len(outliers),
        }

