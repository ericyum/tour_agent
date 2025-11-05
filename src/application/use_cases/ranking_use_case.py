# src/application/use_cases/ranking_use_case.py

import asyncio
import json
import re
import pandas as pd
from datetime import datetime, timedelta

# Custom Module Imports
from application.agents.naver_review.naver_review_agent import NaverReviewAgent
from src.infrastructure.external_services.naver_search.naver_review_api import (
    get_naver_trend,
    search_naver_blog,
)
from src.application.core.graph import app_llm_graph
from src.infrastructure.llm_client import get_llm_client
from src.application.core.constants import NO_IMAGE_URL


class RankingUseCase:
    def __init__(self, naver_supervisor: NaverReviewAgent):
        self.naver_supervisor = naver_supervisor

    def _get_trend_score(self, keyword: str, days: int) -> float:
        if not keyword:
            return 0.0
        today = datetime.today()
        start_date = today - timedelta(days=days)
        trend_data = get_naver_trend(keyword, start_date, today)
        if not trend_data:
            return 0.0
        df = pd.DataFrame(trend_data)
        if "ratio" in df.columns and not df["ratio"].empty:
            return df["ratio"].mean()
        return 0.0

    async def _get_sentiment_score(
        self, keyword: str, num_reviews: int
    ) -> tuple[float, list]:
        if not keyword:
            return 50.0, []

        search_keyword = f"{keyword} 후기"
        api_results = search_naver_blog(
            search_keyword, display=num_reviews + 5
        )  # Add buffer
        if not api_results:
            return 50.0, []

        candidate_blogs = []
        for item in api_results:
            if "blog.naver.com" in item["link"]:
                item["title"] = re.sub(r"<[^>]+>", "", item["title"]).strip()
                if item["title"] and item["link"]:
                    candidate_blogs.append(item)
            if len(candidate_blogs) >= num_reviews:
                break

        if not candidate_blogs:
            return 50.0, []

        total_strong_pos = 0
        total_strong_neg = 0
        total_sentiment_frequency = 0
        all_positive_judgments = []

        for blog_data in candidate_blogs:
            try:
                content, _ = await self.naver_supervisor._scrape_blog_content(
                    blog_data["link"]
                )
                if not content or "오류" in content or "찾을 수 없습니다" in content:
                    continue

                max_content_length = 30000
                if len(content) > max_content_length:
                    content = content[:max_content_length]

                final_state = app_llm_graph.invoke(
                    {
                        "original_text": content,
                        "keyword": keyword,
                        "title": blog_data["title"],
                        "log_details": True,
                        "re_summarize_count": 0,
                        "is_relevant": False,
                    }
                )

                if not final_state or not final_state.get("is_relevant"):
                    continue

                judgments = final_state.get("final_judgments", [])
                if not judgments:
                    continue

                all_positive_judgments.extend(
                    [j for j in judgments if j["final_verdict"] == "긍정"]
                )
                pos_count = sum(
                    1 for res in judgments if res["final_verdict"] == "긍정"
                )
                neg_count = sum(
                    1 for res in judgments if res["final_verdict"] == "부정"
                )
                strong_pos_count = sum(
                    1
                    for res in judgments
                    if res["final_verdict"] == "긍정" and res["score"] >= 1.0
                )
                strong_neg_count = sum(
                    1
                    for res in judgments
                    if res["final_verdict"] == "부정" and res["score"] < -1.0
                )

                total_strong_pos += strong_pos_count
                total_strong_neg += strong_neg_count
                total_sentiment_frequency += pos_count + neg_count
            except Exception:
                continue

        if total_sentiment_frequency == 0:
            return 50.0, []

        sentiment_score = (
            total_strong_pos - total_strong_neg
        ) / total_sentiment_frequency * 50 + 50
        return sentiment_score, all_positive_judgments

    async def _summarize_trend_reasons(self, keyword: str) -> str:
        if not keyword:
            return "키워드가 없어 트렌드 분석 불가"
        today = datetime.today()
        start_date = today - timedelta(days=90)
        trend_data = get_naver_trend(keyword, start_date, today)
        if not trend_data:
            return "트렌드 데이터 없음"
        df = pd.DataFrame(trend_data)
        data_str = df.to_string()
        llm = get_llm_client(temperature=0.2)
        prompt = f"""
        다음은 '{keyword}'에 대한 최근 90일간의 네이버 검색량 트렌드 데이터입니다.
        
        [데이터]
        {data_str}

        [요청]
        1. 데이터(날짜별 관심도 비율)를 기반으로, 검색량 트렌드의 핵심 특징을 1-2줄로 요약해주세요.
        2. 만약 주말/평일 간의 명확한 차이나 특정 시점의 급등 같은 패턴이 보인다면, 방문 시 참고할 만한 팁을 한 문장 추가해주세요.

        [출력 예시 1 - 꾸준한 경우]
        "최근 한 달간 관심도가 꾸준히 증가하고 있어, 현재 가장 주목받는 시기입니다."
        
        [출력 예시 2 - 주말 편중의 경우]
        "주로 주말에 관심도가 급증하는 경향을 보입니다. 여유로운 방문을 원한다면 평일 방문을 고려해볼 수 있습니다."
        """
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            return "트렌드 분석 중 오류 발생"

    async def _summarize_sentiment_reasons(
        self, positive_judgments: list, keyword: str
    ) -> str:
        if not positive_judgments:
            return "긍정 리뷰가 없어 분석 불가"
        sentences = [j["sentence"] for j in positive_judgments]
        sentences_str = "\n- ".join(sentences[:20])
        llm = get_llm_client(temperature=0.2)
        prompt = f"""
        다음은 '{keyword}'에 대한 블로그 리뷰에서 추출된 긍정적인 문장들입니다.
        이 문장들을 바탕으로, 사용자들이 주로 어떤 점을 칭찬하는지 핵심적인 이유 1~2가지를 요약해주세요.

        [긍정 문장 목록]
        - {sentences_str}

        [출력 규칙]
        - "네, ...해드리겠습니다" 와 같은 서론이나 불필요한 설명은 절대 포함하지 마세요.
        - 분석된 핵심 이유에 대한 요약 내용만 바로 작성해주세요.

        [좋은 출력 예시]
        '깨끗한 시설과 다양한 먹거리에 대한 칭찬이 많습니다.'
        '실감 나고 인상적인 미디어 아트와, 사진 찍기 좋게 아름답게 꾸며진 공간에 대한 만족도가 높습니다.'
        """
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            return "감성 분석 이유 요약 중 오류 발생"

    async def _generate_score_explanation(self, item: dict, is_festival: bool) -> str:
        llm = get_llm_client(temperature=0.3)

        # Prepare data for the prompt
        data = {
            "title": item.get("title"),
            "sentiment_score": item.get("sentiment_score"),
            "sentiment_reason": item.get("sentiment_reason"),
            "quarterly_trend_score": item.get("quarterly_trend_score"),
            "yearly_trend_score": item.get("yearly_trend_score"),
            "trend_reason": item.get("trend_reason"),
        }
        today_str = datetime.today().strftime("%Y년 %m월 %d일")

        if is_festival:
            data["time_score"] = item.get("time_score")
            # Add festival dates to the data payload
            start_date = item.get("eventstartdate")
            end_date = item.get("eventenddate")
            if start_date and end_date:
                try:
                    data["festival_period"] = (
                        f"{datetime.strptime(str(start_date), '%Y%m%d').strftime('%Y년 %m월 %d일')}부터 {datetime.strptime(str(end_date), '%Y%m%d').strftime('%Y년 %m월 %d일')}까지"
                    )
                except (ValueError, TypeError):
                    pass  # Ignore if dates are not in the expected format

            score_definitions = """
            - ❤️ 만족도 점수: 실제 방문객들의 긍정적인 리뷰가 얼마나 많은지를 나타내요. (100점에 가까울수록 만족도가 높음)
            - 📅 시기성 점수: 지금이 이 축제를 방문하기 얼마나 좋은 시기인지를 알려줘요. (100점에 가까울수록 축제가 현재 진행중이거나 곧 시작한다는 의미)
            - 🔥 최근 화제성 (90일): 지난 3개월 동안 사람들이 얼마나 많이 검색했는지를 보여줘요.
            - 🗓️ 연간 꾸준함 (365일): 1년 내내 얼마나 꾸준히 인기가 있었는지를 나타내요.
            """
            # New instruction for the prompt
            timeliness_instruction = f"""
        - '시기성 점수'를 설명할 때는 오늘 날짜({today_str})와 축제 기간({data.get('festival_period', '알 수 없음')})을 함께 언급하며 왜 방문하기 좋은 시기인지(또는 아닌지) 구체적으로 설명해주세요.
        """
        else:
            data["distance_score"] = item.get("distance_score")
            raw_distance_m = item.get("distance")
            if raw_distance_m is not None:
                data["distance_in_km"] = f"{(raw_distance_m / 1000):.2f}km"

            score_definitions = """
            - ❤️ 만족도 점수: 실제 방문객들의 긍정적인 리뷰가 얼마나 많은지를 나타내요. (100점에 가까울수록 만족도가 높음)
            - 📍 거리 점수: 원래 찾으려던 축제 장소에서 얼마나 가까운지를 알려줘요. (100점에 가까울수록 가까움)
            - 🔥 최근 화제성 (90일): 지난 3개월 동안 사람들이 얼마나 많이 검색했는지를 보여줘요.
            - 🗓️ 연간 꾸준함 (365일): 1년 내내 얼마나 꾸준히 인기가 있었는지를 나타내요.
            """
            timeliness_instruction = f"""
        - '거리 점수'를 설명할 때는 실제 거리({data.get('distance_in_km', '알 수 없음')})를 함께 언급하며 얼마나 가까운지(또는 먼지) 구체적으로 설명해주세요.
        """  # No special instruction for distance

        prompt = f"""
        당신은 데이터를 쉽고 친절하게 설명해주는 분석가입니다. 아래는 '{data["title"]}'의 분석 데이터입니다.

        [점수 의미]
        {score_definitions}

        [분석 데이터]
        {json.dumps(data, ensure_ascii=False, indent=2)}

        [요청]
        위 데이터를 바탕으로, 각 항목에 대해 사용자가 이해하기 쉽게 구체적인 설명을 생성해주세요.
        아래 규칙을 반드시 지켜주세요.

        1. 각 항목을 설명할 때, 점수와 함께 그 점수가 의미하는 바를 `sentiment_reason`과 `trend_reason`을 활용하여 풀어서 설명해주세요.
        2. '만족도 점수'를 설명할 때는 `sentiment_reason`에 있는 구체적인 칭찬 이유를 반드시 포함해주세요.
        {timeliness_instruction}
        3. 딱딱한 보고서 형식이 아닌, 친구에게 말하듯 친절하고 부드러운 어투를 사용해주세요.
        4. "[분석]" 이라는 단어나 머리글을 사용하지 마세요.
        5. 각 항목을 설명하는 내용을 Markdown 리스트(-) 형식으로 작성해주세요.

        [좋은 출력 예시 (축제)]
        - ❤️ **만족도**: 83.33점! 방문객들이 '깨끗한 시설과 다양한 먹거리' 때문에 아주 만족했어요.
        - 📅 **시기성**: 100점! 축제가 2024년 10월 25일부터 2024년 11월 3일까지인데, 오늘이 10월 26일이니까 지금 바로 축제를 즐길 수 있는 완벽한 시기예요.
        - 🔥 **화제성**: 최근 3개월간 관심도가 16.25점으로 다소 낮지만, '특정 날짜에만 관심이 집중되는 패턴'을 보여요. 조용한 방문을 원한다면 지금이 기회일 수 있어요. 1년 내내 꾸준한 관심도는 13.47점으로, 아는 사람만 아는 숨은 명소일 가능성이 높아요.
        
        [좋은 출력 예시 (장소)]
        - ❤️ **만족도**: 90점! '직원들이 친절하고 주차장이 넓다'는 점에서 방문객들의 만족도가 매우 높아요.
        - 📍 **거리**: 85점! 원래 가려던 축제 장소에서 약 1.25km 떨어져 있어 함께 둘러보기 좋아요.
        - 🔥 **화제성**: '주말에 검색량이 급증하는 경향'을 보여요. 여유롭게 즐기고 싶다면 평일에 방문하는 걸 추천해요.
        """
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            return "점수 설명 생성 중 오류가 발생했습니다."

    async def _generate_comparative_summary(
        self, ranked_list: list, is_festival: bool
    ) -> str:
        llm = get_llm_client(temperature=0.3)

        # Dynamically create the data structure for the prompt based on context
        data_for_prompt = []
        for item in ranked_list:
            data = {
                "title": item.get("title"),
                "ranking_score": item.get("ranking_score"),
                "sentiment_score": item.get("sentiment_score"),
                "sentiment_reason": item.get("sentiment_reason"),
                "quarterly_trend_score": item.get("quarterly_trend_score"),
                "yearly_trend_score": item.get("yearly_trend_score"),
                "trend_reason": item.get("trend_reason"),
            }
            if is_festival:
                data["time_score"] = item.get("time_score")
            else:
                data["distance_score"] = item.get("distance_score")
            data_for_prompt.append(data)

        context_specific_meaning = (
            """
        - 시기성 점수: 현재 시점에서 얼마나 방문하기 좋은 시기인지를 나타냅니다. (100점에 가까울수록 현재 진행중이거나 곧 시작함을 의미)
        """
            if is_festival
            else """
        - 거리 점수: 선택한 축제 장소에서 얼마나 가까운지를 나타냅니다. (100점에 가까울수록 가까움을 의미)
        """
        )

        prompt = f"""
        당신은 친절하고 통찰력 있는 여행 추천 데이터 분석가입니다. 아래는 여러 점수를 종합하여 순위를 매긴 관광지 또는 축제 목록입니다.

        [각 점수의 의미]
        - 만족도 점수: 실제 방문객들이 리뷰에서 남긴 만족도를 나타냅니다. (100점에 가까울수록 긍정적 평가가 많음)
        - 최근 화제성 (90일): 최근 3개월간 대중의 관심이 얼마나 뜨거운지를 보여줍니다.
        - 연간 꾸준함 (365일): 지난 1년간 얼마나 꾸준히 관심을 받았는지를 보여줍니다.
        {context_specific_meaning}

        [분석 데이터]
        {json.dumps(data_for_prompt, ensure_ascii=False, indent=2)}

        [요청]
        위 데이터와 각 점수의 의미를 바탕으로, 1위가 왜 최고의 선택인지 사용자 친화적으로 설명해주세요. 아래 규칙을 반드시 지켜주세요.
        
        1. "1위는 OOO입니다" 라고 시작하지 말고, "이번 추천에서는 OOO이(가) 가장 높은 점수를 받았네요!" 와 같이 친구처럼 자연스럽게 시작해주세요.
        2. 각 점수가 왜 높은지(또는 낮은지) `sentiment_reason`과 `trend_reason`을 인용하여 구체적인 이유를 설명해주세요.
        3. 다른 순위와 비교하여 1위가 가진 강점을 부각해주세요.
        4. 2~3 문장의 간결하고 설득력 있는 요약문으로 마무리해주세요.

        [좋은 요약 예시]
        "이번 추천에서는 OOO이(가) 가장 높은 점수를 받았네요! 무엇보다 실제 방문객들이 '아이들과 즐길 거리가 많다'는 점에서 높은 만족도를 보였고, 최근 3개월간 검색량이 꾸준히 증가하며 뜨거운 관심을 받고 있다는 점이 큰 강점입니다. 2위인 XXX에 비해 연간 꾸준함은 조금 낮지만, 지금 당장 방문하기 좋은 시기라는 점과 높은 만족도를 고려했을 때 최고의 선택이라고 할 수 있습니다."
        """
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            return "최종 분석 요약 생성 중 오류가 발생했습니다."

    async def generate_ranking_report(
        self, ranked_list: list, top_n: int, is_festival: bool
    ) -> str:
        top_n = int(top_n)
        if not ranked_list or not any(
            item.get("ranking_score", 0) > 0 for item in ranked_list[:top_n]
        ):
            return "스코어링된 항목이 없습니다."

        comparative_summary = await self._generate_comparative_summary(
            ranked_list[:top_n], is_festival
        )
        report_parts = [f"## 🏆 최종 순위 분석\n{comparative_summary}", "---"]
        top_items = ranked_list[:top_n]
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for i, item in enumerate(top_items):
            rank_indicator = medals[i] if i < len(medals) else f"{i+1}위"
            title = item.get("title", "N/A")
            total_score = item.get("ranking_score", "N/A")
            image_url = item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL

            # Generate the user-friendly explanation for the scores
            explanation = await self._generate_score_explanation(item, is_festival)

            report_parts.append(
                f"### {rank_indicator} {i+1}위: {title} (종합 점수: {total_score})"
            )
            report_parts.append(f"![{title}]({image_url})\n")
            report_parts.append(explanation)
            report_parts.append("---")

        return "\n\n".join(report_parts)

    async def rank_places(
        self,
        places_list: list,
        num_reviews: int,
        top_n: int,
        progress,
        is_course: bool = False,
    ):
        if not places_list:
            return [], "목록이 비어있습니다.", [], ""

        # Find max distance for normalization
        distances = [
            p.get("distance", 0) for p in places_list if p.get("distance") is not None
        ]
        max_dist = max(distances) if distances else 0

        async def process_place(place):
            # 1. Calculate Distance Score
            distance = place.get("distance")
            distance_score = 0
            if distance is not None:
                if max_dist > 0:
                    distance_score = 1 - (distance / max_dist)
                else:
                    distance_score = 1.0
            place["distance_score"] = round(distance_score * 100, 2)

            # 2. Calculate Trend and Sentiment Scores
            if is_course:
                course_title = place.get("title", "")
                sub_points = place.get("sub_points", [])
                if not sub_points:
                    place["ranking_score"] = 0
                    return place

                q_trend_scores, y_trend_scores, sentiment_scores, all_judgments = (
                    [],
                    [],
                    [],
                    [],
                )
                for sub in sub_points:
                    sub_title = sub.get("subname", "")
                    if not sub_title:
                        continue
                    q_trend_scores.append(self._get_trend_score(sub_title, days=90))
                    y_trend_scores.append(self._get_trend_score(sub_title, days=365))
                    s_score, judgments = await self._get_sentiment_score(
                        sub_title, num_reviews
                    )
                    sentiment_scores.append(s_score)
                    all_judgments.extend(judgments)

                place["quarterly_trend_score"] = (
                    round(sum(q_trend_scores) / len(q_trend_scores), 2)
                    if q_trend_scores
                    else 0
                )
                place["yearly_trend_score"] = (
                    round(sum(y_trend_scores) / len(y_trend_scores), 2)
                    if y_trend_scores
                    else 0
                )
                place["sentiment_score"] = (
                    round(sum(sentiment_scores) / len(sentiment_scores), 2)
                    if sentiment_scores
                    else 50
                )

                trend_reason, sentiment_reason = await asyncio.gather(
                    self._summarize_trend_reasons(course_title),
                    self._summarize_sentiment_reasons(all_judgments, course_title),
                )
            else:
                title = place.get("title", "")
                quarterly_trend_score = self._get_trend_score(title, days=90)
                yearly_trend_score = self._get_trend_score(title, days=365)
                sentiment_score, judgments = await self._get_sentiment_score(
                    title, num_reviews
                )

                place["quarterly_trend_score"] = round(quarterly_trend_score, 2)
                place["yearly_trend_score"] = round(yearly_trend_score, 2)
                place["sentiment_score"] = round(sentiment_score, 2)

                trend_reason, sentiment_reason = await asyncio.gather(
                    self._summarize_trend_reasons(title),
                    self._summarize_sentiment_reasons(judgments, title),
                )

            # 3. Calculate Final Weighted Score
            w_dist = 0.3
            w_sentiment = 0.4
            w_trend_quarterly = 0.2
            w_trend_yearly = 0.1

            final_score = (
                (place["distance_score"] * w_dist)
                + (place["sentiment_score"] * w_sentiment)
                + (place["quarterly_trend_score"] * w_trend_quarterly)
                + (place["yearly_trend_score"] * w_trend_yearly)
            )
            place["ranking_score"] = round(final_score, 2)

            place["trend_reason"] = trend_reason
            place["sentiment_reason"] = sentiment_reason
            return place

        tasks = [process_place(p) for p in places_list]
        ranked_places = []
        for task in progress.tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="장소 점수 계산 중"
        ):
            ranked_places.append(await task)

        ranked_places.sort(key=lambda x: x.get("ranking_score", 0), reverse=True)
        gallery_output = [
            (
                item.get("firstimage", NO_IMAGE_URL) or NO_IMAGE_URL,
                f"점수: {item.get('ranking_score')} - {item['title']}",
            )
            for item in ranked_places
        ]
        report_md = await self.generate_ranking_report(
            ranked_places, top_n, is_festival=False
        )

        return ranked_places, "순위 계산 완료!", gallery_output, report_md

    def _get_time_score(self, start_date_str: str, end_date_str: str) -> float:
        if not start_date_str or not end_date_str:
            return 0.0

        today = datetime.today().date()

        try:
            original_start_date = datetime.strptime(
                str(start_date_str).split(".")[0], "%Y%m%d"
            ).date()
            original_end_date = datetime.strptime(
                str(end_date_str).split(".")[0], "%Y%m%d"
            ).date()
        except (ValueError, TypeError):
            return 0.0

        is_projected = False  # Flag to track if the date is an assumption

        # Case 1: Festival is currently ongoing.
        if original_start_date <= today <= original_end_date:
            return 1.0

        # Case 2: Festival has already ended.
        if original_end_date < today:
            is_projected = True  # Mark this as a projection
            prospective_start_date = original_start_date
            while prospective_start_date <= today:
                prospective_start_date = (
                    pd.to_datetime(prospective_start_date) + pd.DateOffset(years=1)
                ).date()

            days_until_start = (prospective_start_date - today).days

        # Case 3: Festival is in the future.
        else:  # original_start_date > today
            days_until_start = (original_start_date - today).days

        # Apply tiered scoring
        score = 0.0
        if 0 <= days_until_start <= 7:
            score = 0.9
        elif 8 <= days_until_start <= 30:
            score = 0.6
        elif 31 <= days_until_start <= 90:
            score = 0.3
        else:  # More than 90 days away
            score = 0.1

        # Apply penalty if the date was projected
        if is_projected:
            penalty_multiplier = 0.8  # 20% penalty
            score *= penalty_multiplier

        return score

    async def rank_festivals(
        self, festivals_list: list, num_reviews: int, top_n: int, progress
    ):
        if not festivals_list:
            return [], ""

        async def process_festival(festival):
            title = festival.get("title", "")
            start_date = festival.get("eventstartdate")
            end_date = festival.get("eventenddate")

            # 1. Calculate Time Score
            time_score = self._get_time_score(start_date, end_date)
            festival["time_score"] = round(time_score * 100, 2)

            # 2. Calculate Trend and Sentiment Scores
            quarterly_trend_score = self._get_trend_score(title, days=90)
            yearly_trend_score = self._get_trend_score(title, days=365)
            sentiment_score, judgments = await self._get_sentiment_score(
                title, num_reviews
            )

            festival["quarterly_trend_score"] = round(quarterly_trend_score, 2)
            festival["yearly_trend_score"] = round(yearly_trend_score, 2)
            festival["sentiment_score"] = round(sentiment_score, 2)

            # Get reasons for scores
            trend_reason, sentiment_reason = await asyncio.gather(
                self._summarize_trend_reasons(title),
                self._summarize_sentiment_reasons(judgments, title),
            )
            festival["trend_reason"] = trend_reason
            festival["sentiment_reason"] = sentiment_reason

            # 3. Calculate Final Weighted Score
            w_time = 0.6
            w_sentiment = 0.2
            w_trend_quarterly = 0.1
            w_trend_yearly = 0.1

            final_score = (
                (festival["time_score"] * w_time)
                + (festival["sentiment_score"] * w_sentiment)
                + (festival["quarterly_trend_score"] * w_trend_quarterly)
                + (festival["yearly_trend_score"] * w_trend_yearly)
            )
            festival["ranking_score"] = round(final_score, 2)

            return festival

        tasks = [process_festival(f) for f in festivals_list]
        ranked_festivals = []
        for task in progress.tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="축제 순위 계산 중"
        ):
            ranked_festivals.append(await task)

        ranked_festivals.sort(key=lambda x: x.get("ranking_score", 0), reverse=True)

        report_md = await self.generate_ranking_report(
            ranked_festivals, top_n, is_festival=True
        )

        return ranked_festivals, report_md
