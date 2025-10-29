# src/application/use_cases/sentiment_analysis_use_case.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

import json
from collections import Counter

# Custom Module Imports
from src.application.supervisors.naver_review_supervisor import NaverReviewSupervisor
from src.infrastructure.external_services.naver_search.naver_review_api import search_naver_blog
from src.application.core.graph import app_llm_graph
from src.application.core.utils import get_season, save_df_to_csv, summarize_negative_feedback
from src.infrastructure.reporting.charts import create_donut_chart, create_stacked_bar_chart, create_score_distribution_histogram, create_outlier_boxplot
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.application.core.constants import CATEGORY_TO_ICON_MAP
from src.infrastructure.llm_client import get_llm_client
from src.domain.knowledge_base import knowledge_base


class SentimentAnalysisUseCase:
    def __init__(self, naver_supervisor: NaverReviewSupervisor, script_dir: str):
        self.naver_supervisor = naver_supervisor
        self.script_dir = script_dir
        self.llm = get_llm_client(temperature=0.1)

    def _format_positive_keywords_html(self, keywords_data: list, total_reviews: int) -> str:
        if not keywords_data:
            return ""

        # Sort by count descending
        keywords_data.sort(key=lambda x: x.get('count', 0), reverse=True)
        
        max_count = keywords_data[0].get('count', 0) if keywords_data else 0

        html = '<div style="padding: 10px; border: 1px solid #e0e0e0; border-radius: 8px;">'
        html += f'<h3 style="margin-bottom: 15px; font-size: 1.1em;">👍 이런 점이 좋았어요 <span style="font-size: 0.8em; color: #777;">(총 {total_reviews}개 후기 기반)</span></h3>'
        html += '<ul style="list-style-type: none; padding: 0; margin: 0;">'

        # Define some nice icons (using unicode)
        icons = ["😋", "✨", "💖", "👍", "🎉", "💯", "⭐", "💡", "🙌", "😎"]

        for i, item in enumerate(keywords_data[:10]): # Show top 10
            keyword = item.get('keyword', 'N/A')
            count = item.get('count', 0)
            width_percentage = (count / max_count) * 100 if max_count > 0 else 0
            icon = icons[i % len(icons)]

            html += f'''
            <li style="margin-bottom: 8px; position: relative; background-color: #f7f7f7; border-radius: 4px; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; height: 100%; width: {width_percentage}%; background-color: #D6E6FF; z-index: 1;"></div>
                <div style="position: relative; z-index: 2; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.95em; color: #333;">{icon} "{keyword}"</span>
                    <span style="font-size: 0.9em; font-weight: bold; color: #005AAB;">{count}</span>
                </div>
            </li>
            '''
        html += '</ul></div>'
        return html

    async def _generate_positive_keywords_summary(self, aspect_sentiment_pairs: list) -> list:
        if not aspect_sentiment_pairs:
            return []

        # Filter for positive sentiment pairs
        positive_pairs = []
        sentiment_dictionaries = {
            **knowledge_base.adjectives, **knowledge_base.adverbs,
            **knowledge_base.sentiment_nouns, **knowledge_base.idioms
        }
        for aspect, sentiment in aspect_sentiment_pairs:
            if sentiment in sentiment_dictionaries:
                scores = sentiment_dictionaries[sentiment]
                if scores and any(s > 0 for s in scores):
                    positive_pairs.append((aspect, sentiment))
        
        if not positive_pairs:
            return []

        # Use Counter to get initial frequencies
        pair_counts = Counter(positive_pairs)
        # Convert to a list of strings for the LLM prompt
        pairs_str_list = [f"('{p[0]}', '{p[1]}'): {c}회" for p, c in pair_counts.items()]

        prompt = f'''
        당신은 사용자 리뷰에서 핵심 긍정 키워드를 추출하고 그룹화하는 마케팅 분석 전문가입니다.
        아래는 '주체-감성' 쌍과 각 쌍의 언급 횟수 목록입니다.

        [데이터]
        {', '.join(pairs_str_list)}

        [요청]
        1. 의미가 유사한 '주체-감성' 쌍들을 하나의 대표 키워드로 그룹화해주세요.
           (예: ('음식', '맛있다'), ('음식', '훌륭하다') -> "음식이 맛있어요")
           (예: ('직원', '친절하다'), ('사장님', '친절하다') -> "직원이 친절해요")
        2. 각 대표 키워드에 몇 개의 원본 쌍이 포함되었는지 합산하여 `count`를 계산해주세요.
        3. 최종 결과는 사용자가 이해하기 쉬운 자연스러운 문장 형태의 `keyword`와 `count`를 포함하는 JSON 리스트 형식으로 반환해주세요.
        4. 가장 많이 언급된 순서로 정렬해주세요.
        5. 다른 설명 없이 JSON 리스트만 출력해주세요.

        [출력 형식 예시]
        [
            {{"keyword": "음식이 맛있어요", "count": 21}},
            {{"keyword": "재료가 신선해요", "count": 6}},
            {{"keyword": "매장이 넓어요", "count": 6}},
            {{"keyword": "직원이 친절해요", "count": 5}}
        ]
        '''
        try:
            response = await self.llm.ainvoke(prompt)
            # Extract JSON from the response
            json_str_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_str_match:
                return json.loads(json_str_match.group())
            return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error generating positive keywords summary: {e}")
            return []

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
        all_aspect_sentiment_pairs = []  # <--- Aggregate all pairs here
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
                    aspect_pairs = final_state.get("aspect_sentiment_pairs", [])
                    blog_judgments_list.append(judgments)
                    all_scores.extend([j['score'] for j in judgments])
                    all_aspect_sentiment_pairs.extend(aspect_pairs)
                    
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

        # Generate "What I liked" summary
        positive_keywords_data = await self._generate_positive_keywords_summary(all_aspect_sentiment_pairs)
        positive_keywords_html = self._format_positive_keywords_html(positive_keywords_data, len(blog_results_list))

        return {
            "positive_keywords_html": positive_keywords_html,
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

