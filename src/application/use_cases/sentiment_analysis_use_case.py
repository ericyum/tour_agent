# src/application/use_cases/sentiment_analysis_use_case.py

import os
import re
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

import json
from collections import Counter

# Custom Module Imports
from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent
from src.infrastructure.external_services.naver_search.naver_review_api import (
    search_naver_blog,
)
from src.application.core.graph import app_llm_graph
from src.application.core.utils import (
    save_df_to_csv,
    summarize_negative_feedback,
)
from src.infrastructure.reporting.charts import (
    create_donut_chart,
    create_stacked_bar_chart,
    create_satisfaction_level_bar_chart,
    create_absolute_score_line_chart,
    create_outlier_boxplot,
)
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.application.core.constants import CATEGORY_TO_ICON_MAP
from src.infrastructure.llm_client import get_llm_client
from src.domain.knowledge_base import knowledge_base


class SentimentAnalysisUseCase:
    def __init__(self, naver_supervisor: NaverReviewAgent, script_dir: str):
        self.naver_supervisor = naver_supervisor
        self.script_dir = script_dir
        self.llm = get_llm_client(temperature=0.1)

    async def _generate_distribution_interpretation(self, counts: dict, total_sentences: int, boundaries: dict, avg_score: float) -> str:
        if total_sentences == 0:
            return "분석할 문장이 없습니다."

        prompt = f"""
        You are an expert data analyst specializing in customer feedback analysis. Your task is to write a comprehensive, easy-to-understand analysis of a festival's sentiment distribution in Korean, based on the data provided. The analysis must be objective, data-driven, and strictly follow the requested markdown format.

        **Input Data:**
        - Total Sentences Analyzed: {total_sentences}
        - Sentence Counts per Satisfaction Level: {counts}
        - Absolute Average Score (from -5.0 to 5.0): {avg_score:.2f}
        - Satisfaction Level Boundaries:
            - Very Dissatisfied: < {boundaries['very_dissatisfied_upper']:.2f}
            - Dissatisfied: < {boundaries['dissatisfied_upper']:.2f}
            - Neutral: < {boundaries['neutral_upper']:.2f}
            - Satisfied: < {boundaries['satisfied_upper']:.2f}
            - Very Satisfied: >= {boundaries['satisfied_upper']:.2f}

        **Instructions:**
        1.  **Role:** Act as a professional data analyst.
        2.  **Tone:** Write in clear, objective, and helpful Korean.
        3.  **Format:** You MUST follow this markdown format exactly. Do not add any other sections.

            ```markdown
            ### 📊 만족도 분포 종합 분석

            **주요 지표:**
            - **분석 문장 수**: {total_sentences}개
            - **평균 만족도 점수**: {avg_score:.2f}점 (5점 만점)
            - **만족도 Level 기준**: (매우 불만족 < {boundaries['very_dissatisfied_upper']:.2f} < 불만족 < {boundaries['dissatisfied_upper']:.2f} < 보통 < {boundaries['neutral_upper']:.2f} < 만족 < {boundaries['satisfied_upper']:.2f} < 매우 만족)

            **분포 형태 분석:**
            [Based on the sentence counts, describe the shape of the distribution. Use one of the following patterns:
            - **압도적 긍정 (J-커브형):** If '매우 만족' and '만족' counts are overwhelmingly dominant.
            - **양극화 (U-커브형):** If '매우 만족' and '매우 불만족' have the highest counts, and '보통' has a low count.
            - **다양한 평가 (평평한 분포):** If counts are spread out across all levels without a clear single peak.
            - **보통 중심 (종형 분포):** If '보통' has the highest count, with other levels tapering off.
            - **부정적 평가 (L-커브형):** If '매우 불만족' and '불만족' counts are overwhelmingly dominant.
            Provide a brief, one-sentence description of the shape in Korean.]

            **종합 해석:**
            [Synthesize all the information into a final conclusion in Korean. CRITICALLY, interpret the distribution shape in the context of the **Absolute Average Score**.
            - If the average score is very high (e.g., > 2.5), explain that even the '보통(Neutral)' category likely represents positive opinions, making the overall feedback very strong.
            - If the average score is very low (e.g., < 0.0), explain that even '보통(Neutral)' might indicate dissatisfaction.
            - If the distribution is polarized, recommend looking into the specific causes.
            - If the distribution is diverse/flat, mention that visitors had varied experiences with different aspects of the event.
            Provide a 2-3 sentence summary in Korean.]
            ```
        4.  **Constraint:** Generate only the markdown content as requested. Do not add any introductory or concluding remarks outside of the specified format.
        """
        try:
            response = await self.llm.ainvoke(prompt)
            # Extract markdown content if the LLM wraps it in ```markdown
            match = re.search(r"```markdown\n(.*)```", response.content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return response.content.strip()
        except Exception as e:
            print(f"Error generating LLM-based distribution interpretation: {e}")
            return "자동 분석 생성 중 오류가 발생했습니다."

    def _format_positive_keywords_html(
        self, keywords_data: list, total_reviews: int
    ) -> str:
        if not keywords_data:
            return ""

        # Sort by count descending
        keywords_data.sort(key=lambda x: x.get("count", 0), reverse=True)

        max_count = keywords_data[0].get("count", 0) if keywords_data else 0

        html = '<div style="padding: 10px; border: 1px solid #e0e0e0; border-radius: 8px;">'
        html += f'<h3 style="margin-bottom: 15px; font-size: 1.1em;">👍 이런 점이 좋았어요 <span style="font-size: 0.8em; color: #777;">(총 {total_reviews}개 후기 기반)</span></h3>'
        html += '<ul style="list-style-type: none; padding: 0; margin: 0;">'

        # Define some nice icons (using unicode)
        icons = ["😋", "✨", "💖", "👍", "🎉", "💯", "⭐", "💡", "🙌", "😎"]

        for i, item in enumerate(keywords_data[:10]):  # Show top 10
            keyword = item.get("keyword", "N/A")
            count = item.get("count", 0)
            width_percentage = (count / max_count) * 100 if max_count > 0 else 0
            icon = icons[i % len(icons)]

            html += f"""
            <li style="margin-bottom: 8px; position: relative; background-color: #f7f7f7; border-radius: 4px; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; height: 100%; width: {width_percentage}%; background-color: #D6E6FF; z-index: 1;"></div>
                <div style="position: relative; z-index: 2; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.95em; color: #333;">{icon} "{keyword}"</span>
                    <span style="font-size: 0.9em; font-weight: bold; color: #005AAB;">{count}</span>
                </div>
            </li>
            """
        html += "</ul></div>"
        return html

    async def _generate_positive_keywords_summary(
        self, aspect_sentiment_pairs: list
    ) -> list:
        if not aspect_sentiment_pairs:
            return []

        # Filter for positive sentiment pairs
        positive_pairs = []
        sentiment_dictionaries = {
            **knowledge_base.adjectives,
            **knowledge_base.adverbs,
            **knowledge_base.sentiment_nouns,
            **knowledge_base.idioms,
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
        pairs_str_list = [
            f"('{p[0]}', '{p[1]}'): {c}회" for p, c in pair_counts.items()
        ]

        prompt = f"""
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
        """
        try:
            response = await self.llm.ainvoke(prompt)
            # Extract JSON from the response
            json_str_match = re.search(r"\[.*\]", response.content, re.DOTALL)
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

        # Handle case where all scores are identical
        if np.isclose(std, 0):
            std = 0.1

        boundaries = {
            "mean": mean,
            "std": std,
            "very_dissatisfied_upper": mean - 1.5 * std,
            "dissatisfied_upper": mean - 0.5 * std,
            "neutral_upper": mean + 0.5 * std,
            "satisfied_upper": mean + 1.5 * std,
        }
        return {
            "boundaries": boundaries,
            "filtered_scores": filtered_scores,
            "outliers": outliers,
        }

    def _map_score_to_level(self, score: float, boundaries: dict) -> int:
        if not boundaries:
            return 3  # Default to neutral if no boundaries
        if score < boundaries["very_dissatisfied_upper"]:
            return 1  # 매우 불만족
        elif score < boundaries["dissatisfied_upper"]:
            return 2  # 불만족
        elif score < boundaries["neutral_upper"]:
            return 3  # 보통
        elif score < boundaries["satisfied_upper"]:
            return 4  # 만족
        else:
            return 5  # 매우 만족

    def _remove_leading_year(self, festival_name: str) -> str:
        # Regex to find a four-digit year at the beginning of the string, optionally followed by '년' and a space
        # e.g., "2025 국민고향 남해 마시고 RUN 후기" -> "국민고향 남해 마시고 RUN 후기"
        # e.g., "2024년 서울 불꽃축제" -> "서울 불꽃축제"
        # e.g., "2023-2024 겨울 축제" -> "2024 겨울 축제" (only removes the first year if followed by space)
        cleaned_name = re.sub(r"^\d{4}\s*년?\s*", "", festival_name).strip()
        return cleaned_name if cleaned_name else festival_name # Return original if only year was present or no change

    async def analyze_sentiment(self, festival_name: str, num_reviews: int):
        if not festival_name:
            raise ValueError("축제를 선택해주세요.")

        # Preprocess festival_name to remove leading year
        processed_festival_name = self._remove_leading_year(festival_name)
        print(f"Original festival name: {festival_name}, Processed: {processed_festival_name}")

        search_keyword = f"{processed_festival_name} 후기"

        blog_results_list = []
        blog_judgments_list = []
        all_scores = []
        all_negative_sentences = []
        all_aspect_sentiment_pairs = []
        total_pos, total_neg = 0, 0

        start_index = 1
        max_results_to_scan = 100
        display_count = 20
        consecutive_skips = 0

        while (
            len(blog_results_list) < num_reviews and start_index < max_results_to_scan
        ):
            # 동기 API 호출을 별도 스레드에서 실행
            api_results = await asyncio.to_thread(
                search_naver_blog, search_keyword, display_count, start_index
            )
            if not api_results:
                break

            candidate_blogs = [
                item for item in api_results if "blog.naver.com" in item["link"]
            ]
            for blog_data in candidate_blogs:
                if len(blog_results_list) >= num_reviews:
                    break
                try:
                    content, _ = await self.naver_supervisor._scrape_blog_content(
                        blog_data["link"]
                    )
                    if (
                        not content
                        or "오류" in content
                        or "찾을 수 없습니다" in content
                    ):
                        consecutive_skips += 1
                        continue

                    content = content[:30000]

                    # 동기 블로킹 호출을 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
                    final_state = await asyncio.to_thread(
                        app_llm_graph.invoke,
                        {
                            "original_text": content,
                            "keyword": processed_festival_name,
                            "title": re.sub(r"<[^>]+>", "", blog_data["title"]).strip(),
                            "log_details": True,
                        }
                    )

                    if (
                        not final_state
                        or not final_state.get("is_relevant")
                        or not final_state.get("final_judgments")
                    ):
                        consecutive_skips += 1
                        continue

                    consecutive_skips = 0
                    judgments = final_state.get("final_judgments", [])
                    aspect_pairs = final_state.get("aspect_sentiment_pairs", [])
                    blog_judgments_list.append(judgments)
                    all_scores.extend([j["score"] for j in judgments])
                    all_aspect_sentiment_pairs.extend(aspect_pairs)

                    blog_results_list.append(
                        {
                            "블로그 제목": re.sub(
                                r"<[^>]+>", "", blog_data["title"]
                            ).strip(),
                            "링크": blog_data["link"],
                            "postdate": blog_data.get("postdate", ""),
                            "judgments": judgments,
                        }
                    )

                except Exception as e:
                    print(
                        f"블로그 분석 중 오류 ({festival_name}, {blog_data.get('link', 'N/A')}): {e}"
                    )
                    traceback.print_exc()
                    consecutive_skips += 1
                    continue

                if consecutive_skips >= 3:
                    break
            if len(blog_results_list) >= num_reviews or consecutive_skips >= 3:
                break
            start_index += display_count

        if not blog_results_list:
            raise ValueError(
                f"'{festival_name}'에 대한 유효한 후기 블로그를 찾지 못했습니다."
            )

        boundary_results = self._calculate_satisfaction_boundaries(all_scores)
        boundaries = boundary_results["boundaries"]
        outliers = boundary_results["outliers"]

        processed_blog_results = []
        all_satisfaction_levels = []

        for blog in blog_results_list:
            judgments = blog["judgments"]
            blog_satisfaction_levels = []

            pos_count = 0
            neg_count = 0

            for j in judgments:
                level = self._map_score_to_level(j["score"], boundaries)
                j["satisfaction_level"] = level
                blog_satisfaction_levels.append(level)
                all_satisfaction_levels.append(level)

                if j["final_verdict"] == "긍정":
                    pos_count += 1
                else:
                    neg_count += 1
                    all_negative_sentences.append(j["sentence"])

            total_pos += pos_count
            total_neg += neg_count

            avg_satisfaction = (
                np.mean(blog_satisfaction_levels) if blog_satisfaction_levels else 3.0
            )
            pos_perc = (
                (pos_count / (pos_count + neg_count) * 100)
                if (pos_count + neg_count) > 0
                else 0
            )
            neg_perc = (
                (neg_count / (pos_count + neg_count) * 100)
                if (pos_count + neg_count) > 0
                else 0
            )

            processed_blog_results.append(
                {
                    "블로그 제목": blog["블로그 제목"],
                    "링크": blog["링크"],
                    "감성 빈도": len(judgments),
                    "감성 점수": f"{avg_satisfaction:.2f} / 5",
                    "긍정 문장 수": pos_count,
                    "부정 문장 수": neg_count,
                    "긍정 비율 (%)": f"{pos_perc:.1f}",
                    "부정 비율 (%)": f"{neg_perc:.1f}",
                    "긍/부정 문장 요약": "<br>---<br>".join(
                        [
                            f"[{j['final_verdict']}({j['satisfaction_level']}점)] {j['sentence']}"
                            for j in judgments
                        ]
                    ),
                    "만족도 점수": f"{avg_satisfaction:.2f} / 5",  # Keep for backwards compatibility
                }
            )

        overall_avg_satisfaction = (
            np.mean(all_satisfaction_levels) if all_satisfaction_levels else 3.0
        )
        
        # --- New: Satisfaction Level Counting and Interpretation ---
        level_map = {1: "매우 불만족", 2: "불만족", 3: "보통", 4: "만족", 5: "매우 만족"}
        satisfaction_counts = Counter(
            [level_map.get(level, "보통") for level in all_satisfaction_levels]
        )
        
        distribution_chart_fig = None
        if all_satisfaction_levels:
            fig = create_satisfaction_level_bar_chart(
                satisfaction_counts, f"{festival_name} 상대적 만족도 분포"
            )
            if fig:
                distribution_chart_fig = fig
                # Save for debugging/legacy use if needed, but don't close
                distribution_chart_path = os.path.join(
                    self.script_dir, "temp_img", f"dist_chart_{festival_name}.png"
                )
                os.makedirs(os.path.dirname(distribution_chart_path), exist_ok=True)
                fig.savefig(distribution_chart_path)
                # plt.close(fig) # Removed to pass fig object

        absolute_chart_fig = None
        if all_scores:
            fig = create_absolute_score_line_chart(
                all_scores, f"{festival_name} 절대 점수 분포"
            )
            if fig:
                absolute_chart_fig = fig
                absolute_chart_path = os.path.join(
                    self.script_dir, "temp_img", f"abs_chart_{festival_name}.png"
                )
                os.makedirs(os.path.dirname(absolute_chart_path), exist_ok=True)
                fig.savefig(absolute_chart_path)
                # plt.close(fig) # Removed to pass fig object

        distribution_description = await self._generate_distribution_interpretation(
            satisfaction_counts, len(all_satisfaction_levels), boundaries, overall_avg_satisfaction
        )
        # --- End New ---

        outlier_chart_fig = None
        if all_scores:
            fig = create_outlier_boxplot(
                all_scores, f"{festival_name} 감성 점수 이상치"
            )
            if fig:
                outlier_chart_fig = fig
                outlier_chart_path = os.path.join(
                    self.script_dir, "temp_img", f"outlier_chart_{festival_name}.png"
                )
                os.makedirs(os.path.dirname(outlier_chart_path), exist_ok=True)
                fig.savefig(outlier_chart_path)
                # plt.close(fig) # Removed to pass fig object

        neg_summary_text = summarize_negative_feedback(all_negative_sentences)
        overall_summary_text = f"- **총 분석 블로그**: {len(blog_results_list)}개\n- **전체 평균 만족도**: {overall_avg_satisfaction:.2f} / 5.0 점\n- **긍정 문장 수**: {total_pos}개\n- **부정 문장 수**: {total_neg}개"

        summary_df = pd.DataFrame(
            [
                {
                    "축제명": festival_name,
                    "평균 만족도": f"{overall_avg_satisfaction:.2f}",
                    "긍정 문장 수": total_pos,
                    "부정 문장 수": total_neg,
                }
            ]
        )
        summary_csv_path = save_df_to_csv(summary_df, "overall_summary", festival_name)

        blog_df = pd.DataFrame(processed_blog_results)
        blog_list_csv_path = save_df_to_csv(blog_df, "blog_list", festival_name)

        overall_chart = create_donut_chart(
            total_pos, total_neg, f"{festival_name} 전체 후기 요약"
        )

        # Generate "What I liked" summary
        positive_keywords_data = await self._generate_positive_keywords_summary(
            all_aspect_sentiment_pairs
        )
        positive_keywords_html = self._format_positive_keywords_html(
            positive_keywords_data, len(blog_results_list)
        )

        print(f"DEBUG: all_aspect_sentiment_pairs: {all_aspect_sentiment_pairs}") # DEBUG PRINT

        # Prepare chart data for frontend rendering
        donut_data = {
            "positive": total_pos,
            "negative": total_neg,
        }

        satisfaction_data = {
            "labels": ["매우 불만족", "불만족", "보통", "만족", "매우 만족"],
            "counts": [satisfaction_counts.get(label, 0) for label in ["매우 불만족", "불만족", "보통", "만족", "매우 만족"]],
        }

        # Prepare absolute score distribution data
        bins = [-np.inf, -2.0, -1.0, 0.0, 1.0, 2.0, np.inf]
        labels_abs = ['매우 부정 (<-2)', '부정 (-2~-1)', '약간 부정 (-1~0)', '약간 긍정 (0~1)', '긍정 (1~2)', '매우 긍정 (>2)']
        hist, _ = np.histogram(all_scores, bins=bins) if all_scores else (np.zeros(len(labels_abs)), None)
        absolute_data = {
            "labels": labels_abs,
            "counts": hist.tolist(),
        }

        # Prepare outlier data
        q1 = np.percentile(all_scores, 25) if all_scores else 0
        q3 = np.percentile(all_scores, 75) if all_scores else 0
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        median = np.median(all_scores) if all_scores else 0
        outlier_data = {
            "min": float(np.min(all_scores)) if all_scores else 0,
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "max": float(np.max(all_scores)) if all_scores else 0,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outliers": [float(s) for s in all_scores if s < lower_bound or s > upper_bound] if all_scores else [],
        }

        return {
            "positive_keywords_html": positive_keywords_html,
            "neg_summary_text": neg_summary_text,
            "overall_summary_text": overall_summary_text,
            "summary_csv_path": summary_csv_path,
            "blog_df": blog_df,
            "blog_judgments_list": blog_judgments_list,
            "blog_list_csv_path": blog_list_csv_path,
            "overall_chart": overall_chart,
            "distribution_chart": distribution_chart_fig,
            "absolute_chart": absolute_chart_fig,
            "distribution_description": distribution_description,
            "outlier_chart": outlier_chart_fig,
            "total_score_count": len(all_scores),
            "outlier_count": len(outliers),
            "all_aspect_sentiment_pairs": all_aspect_sentiment_pairs,
            # Chart data for frontend
            "donut_data": donut_data,
            "satisfaction_data": satisfaction_data,
            "absolute_data": absolute_data,
            "outlier_data": outlier_data,
        }
