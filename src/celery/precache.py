# src/celery/precache.py
"""
전수 캐싱 Celery 태스크

배포 후 자동으로 모든 축제에 대한 분석을 미리 수행하여 캐시합니다.
사용자 요청 시 즉시 응답할 수 있도록 합니다.

실행 방법:
    # Celery Worker 실행 (별도 터미널)
    celery -A src.celery.app worker --loglevel=info --pool=solo

    # 수동 실행 (테스트용)
    python -c "from src.celery.precache import precache_all_festivals; precache_all_festivals.delay()"

    # Celery Beat로 자동 실행 (배포 시)
    celery -A src.celery.app beat --loglevel=info
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# 프로젝트 루트를 path에 추가 (src/celery에서 2단계 상위)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.celery.app import celery_app
from src.infrastructure.cache_manager import (
    save_to_cache,
    load_from_cache,
    save_task_progress,
    save_task_result,
)
from src.infrastructure.rate_limiter import rate_limiter
from src.infrastructure.config.loader import (
    KOREAN_FONT_PATH,
    TITLE_TO_CAT_NAMES,
)
from src.application.core.constants import CATEGORY_TO_ICON_MAP

# 스크립트 디렉토리 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_async(coro):
    """비동기 함수를 동기적으로 실행"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_all_festival_names() -> List[str]:
    """DB에서 모든 축제 이름 목록을 가져옵니다."""
    from src.infrastructure.persistence.database import get_db_connection, release_connection, get_cursor

    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute("SELECT title FROM festivals WHERE title IS NOT NULL ORDER BY title")
        rows = cursor.fetchall()
        return [row["title"] for row in rows if row["title"]]
    finally:
        cursor.close()
        release_connection(conn)


def get_active_festival_names() -> List[str]:
    """현재 진행 중이거나 곧 시작할 축제 목록을 가져옵니다."""
    from src.infrastructure.persistence.database import get_db_connection, release_connection, get_cursor

    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        today = datetime.today().strftime("%Y%m%d")
        # 현재 진행 중이거나 30일 내 시작 예정인 축제
        future_date = (datetime.today() + timedelta(days=30)).strftime("%Y%m%d")

        cursor.execute("""
            SELECT title FROM festivals
            WHERE title IS NOT NULL
            AND (
                (eventstartdate <= %s AND eventenddate >= %s)
                OR (eventstartdate >= %s AND eventstartdate <= %s)
            )
            ORDER BY eventstartdate
        """, (today, today, today, future_date))

        rows = cursor.fetchall()
        return [row["title"] for row in rows if row["title"]]
    finally:
        cursor.close()
        release_connection(conn)


# ============================================================================
# 개별 축제 캐싱 태스크
# ============================================================================

@celery_app.task(bind=True, name="precache_festival_unit")
def precache_festival_unit(
    self,
    festival_name: str,
    num_reviews: int = 50,
    skip_if_cached: bool = True
) -> Dict[str, Any]:
    """
    단일 축제에 대한 모든 분석을 수행하고 캐시합니다.

    이 태스크는 "단위 캐싱"의 핵심입니다.
    축제 하나에 대해 최대 num_reviews로 분석하여 저장하면,
    사용자가 더 적은 수의 리뷰를 요청해도 이 캐시 데이터를 활용할 수 있습니다.

    Args:
        festival_name: 축제 이름
        num_reviews: 분석할 최대 리뷰 수 (기본값: 50)
        skip_if_cached: 이미 캐시된 경우 스킵할지 여부
    """
    task_id = self.request.id
    result = {
        "festival_name": festival_name,
        "status": "started",
        "cached_items": [],
        "errors": []
    }

    logger.info(f"[PreCache] 시작: {festival_name}")

    try:
        # 1. 감성분석 캐싱
        try:
            cached = load_from_cache("sentiment", festival_name=festival_name, num_reviews=num_reviews)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] 감성분석 이미 캐시됨: {festival_name}")
                result["cached_items"].append("sentiment (already cached)")
            else:
                logger.info(f"[PreCache] 감성분석 시작: {festival_name}")
                from src.application.use_cases.sentiment_analysis_use_case import SentimentAnalysisUseCase
                from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent

                naver_agent = NaverReviewAgent()
                use_case = SentimentAnalysisUseCase(naver_agent, script_dir=SCRIPT_DIR)

                sentiment_result = run_async(
                    use_case.analyze_sentiment(festival_name, num_reviews)
                )

                if sentiment_result:
                    save_to_cache(sentiment_result, "sentiment", festival_name=festival_name, num_reviews=num_reviews)
                    result["cached_items"].append("sentiment")
                    logger.info(f"[PreCache] 감성분석 완료: {festival_name}")

        except Exception as e:
            error_msg = f"감성분석 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 2. 워드클라우드 캐싱
        try:
            cached = load_from_cache("wordcloud", festival_name=festival_name, num_reviews=num_reviews)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] 워드클라우드 이미 캐시됨: {festival_name}")
                result["cached_items"].append("wordcloud (already cached)")
            else:
                logger.info(f"[PreCache] 워드클라우드 시작: {festival_name}")
                from src.application.use_cases.analysis_use_case import AnalysisUseCase
                from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent
                from io import BytesIO
                import base64

                naver_agent = NaverReviewAgent()
                analysis_use_case = AnalysisUseCase(
                    naver_supervisor=naver_agent,
                    font_path=KOREAN_FONT_PATH,
                    title_to_cat_map=TITLE_TO_CAT_NAMES,
                    cat_to_icon_map=CATEGORY_TO_ICON_MAP,
                    script_dir=SCRIPT_DIR,
                )

                wc_image, message = run_async(
                    analysis_use_case.generate_word_cloud(festival_name, num_reviews)
                )

                wc_result = {"festival_name": festival_name, "message": message}
                if wc_image:
                    buffered = BytesIO()
                    wc_image.save(buffered, format="PNG")
                    wc_result["wordcloud_image"] = base64.b64encode(buffered.getvalue()).decode("utf-8")

                save_to_cache(wc_result, "wordcloud", festival_name=festival_name, num_reviews=num_reviews)
                result["cached_items"].append("wordcloud")
                logger.info(f"[PreCache] 워드클라우드 완료: {festival_name}")

        except Exception as e:
            error_msg = f"워드클라우드 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 3. 트렌드 분석 캐싱
        try:
            cached = load_from_cache("trend", festival_name=festival_name)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] 트렌드 이미 캐시됨: {festival_name}")
                result["cached_items"].append("trend (already cached)")
            else:
                logger.info(f"[PreCache] 트렌드 분석 시작: {festival_name}")
                from src.application.use_cases.analysis_use_case import AnalysisUseCase
                from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent

                naver_agent = NaverReviewAgent()
                analysis_use_case = AnalysisUseCase(
                    naver_supervisor=naver_agent,
                    font_path=KOREAN_FONT_PATH,
                    title_to_cat_map=TITLE_TO_CAT_NAMES,
                    cat_to_icon_map=CATEGORY_TO_ICON_MAP,
                    script_dir=SCRIPT_DIR,
                )

                trend_result = run_async(
                    analysis_use_case.generate_trend_graphs(festival_name)
                )

                if trend_result:
                    save_to_cache(trend_result, "trend", festival_name=festival_name)
                    result["cached_items"].append("trend")
                    logger.info(f"[PreCache] 트렌드 분석 완료: {festival_name}")

        except Exception as e:
            error_msg = f"트렌드 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 4. 개별 랭킹 점수 캐싱 (단위 캐싱의 핵심!)
        try:
            cached = load_from_cache("festival_score", festival_name=festival_name, num_reviews=num_reviews)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] 개별 점수 이미 캐시됨: {festival_name}")
                result["cached_items"].append("festival_score (already cached)")
            else:
                logger.info(f"[PreCache] 개별 점수 계산 시작: {festival_name}")

                score_result = run_async(
                    _calculate_festival_unit_score(festival_name, num_reviews)
                )

                if score_result:
                    save_to_cache(score_result, "festival_score", festival_name=festival_name, num_reviews=num_reviews)
                    result["cached_items"].append("festival_score")
                    logger.info(f"[PreCache] 개별 점수 계산 완료: {festival_name}")

        except Exception as e:
            error_msg = f"개별 점수 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 5. 후기 요약 캐싱
        try:
            cached = load_from_cache("review_summary", festival_name=festival_name, num_reviews=num_reviews)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] 후기 요약 이미 캐시됨: {festival_name}")
                result["cached_items"].append("review_summary (already cached)")
            else:
                logger.info(f"[PreCache] 후기 요약 시작: {festival_name}")
                from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent

                naver_agent = NaverReviewAgent()
                summary_result = run_async(
                    naver_agent.get_review_summary_and_tips(festival_name, num_reviews=num_reviews)
                )

                if summary_result:
                    save_to_cache({"summary": summary_result}, "review_summary", festival_name=festival_name, num_reviews=num_reviews)
                    result["cached_items"].append("review_summary")
                    logger.info(f"[PreCache] 후기 요약 완료: {festival_name}")

        except Exception as e:
            error_msg = f"후기 요약 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 6. 이미지 스크래핑 캐싱
        try:
            cached = load_from_cache("images", festival_name=festival_name)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] 이미지 이미 캐시됨: {festival_name}")
                result["cached_items"].append("images (already cached)")
            else:
                logger.info(f"[PreCache] 이미지 스크래핑 시작: {festival_name}")
                from src.application.use_cases.analysis_use_case import AnalysisUseCase
                from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent

                naver_agent = NaverReviewAgent()
                analysis_use_case = AnalysisUseCase(
                    naver_supervisor=naver_agent,
                    font_path=KOREAN_FONT_PATH,
                    title_to_cat_map=TITLE_TO_CAT_NAMES,
                    cat_to_icon_map=CATEGORY_TO_ICON_MAP,
                    script_dir=SCRIPT_DIR,
                )

                image_paths, image_urls = run_async(
                    analysis_use_case.scrape_festival_images(festival_name, num_blogs=5)
                )

                if image_paths or image_urls:
                    save_to_cache({"image_paths": image_paths, "image_urls": image_urls}, "images", festival_name=festival_name)
                    result["cached_items"].append("images")
                    logger.info(f"[PreCache] 이미지 스크래핑 완료: {festival_name} ({len(image_paths)}개)")

        except Exception as e:
            error_msg = f"이미지 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 7. 주변 추천 캐싱
        try:
            cached = load_from_cache("nearby", festival_name=festival_name)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] 주변 추천 이미 캐시됨: {festival_name}")
                result["cached_items"].append("nearby (already cached)")
            else:
                logger.info(f"[PreCache] 주변 추천 시작: {festival_name}")
                from src.application.services.festival_service import get_festival_details_by_title
                from src.application.agents.db_search.nearby_search_agent import agent_nearby_search

                festival_info = get_festival_details_by_title(festival_name)
                if festival_info and festival_info.get("mapx") and festival_info.get("mapy"):
                    state = {
                        "latitude": float(festival_info["mapy"]),
                        "longitude": float(festival_info["mapx"]),
                        "radius": 5.0,  # 5km 반경
                        "current_festival_id": festival_info.get("contentid")
                    }
                    nearby_result = agent_nearby_search(state)

                    if nearby_result:
                        save_to_cache(nearby_result, "nearby", festival_name=festival_name)
                        result["cached_items"].append("nearby")
                        logger.info(f"[PreCache] 주변 추천 완료: {festival_name}")
                else:
                    logger.warning(f"[PreCache] 주변 추천 스킵 (좌표 없음): {festival_name}")

        except Exception as e:
            error_msg = f"주변 추천 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 8. AI 주의사항 캐싱
        try:
            cached = load_from_cache("precautions", festival_name=festival_name)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] AI 주의사항 이미 캐시됨: {festival_name}")
                result["cached_items"].append("precautions (already cached)")
            else:
                logger.info(f"[PreCache] AI 주의사항 시작: {festival_name}")
                from src.application.agents.precaution_agent import PrecautionAgent
                from src.infrastructure.config.loader import FESTIVAL_INFO_LOOKUP

                festival_info = FESTIVAL_INFO_LOOKUP.get(festival_name, {})
                detailed_category = festival_info.get("detailed_category", "")
                prohibited_behaviors = festival_info.get("prohibited_behaviors", "")

                if detailed_category or prohibited_behaviors:
                    precaution_agent = PrecautionAgent()
                    precaution_result = run_async(
                        precaution_agent.generate_precautions(festival_name, detailed_category, prohibited_behaviors)
                    )

                    if precaution_result:
                        save_to_cache(precaution_result, "precautions", festival_name=festival_name)
                        result["cached_items"].append("precautions")
                        logger.info(f"[PreCache] AI 주의사항 완료: {festival_name}")
                else:
                    logger.warning(f"[PreCache] AI 주의사항 스킵 (카테고리 정보 없음): {festival_name}")

        except Exception as e:
            error_msg = f"AI 주의사항 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        # 9. AI 렌더링 캐싱
        try:
            cached = load_from_cache("ai_rendering", festival_name=festival_name)
            if cached and skip_if_cached:
                logger.info(f"[PreCache] AI 렌더링 이미 캐시됨: {festival_name}")
                result["cached_items"].append("ai_rendering (already cached)")
            else:
                logger.info(f"[PreCache] AI 렌더링 시작: {festival_name}")
                from src.application.use_cases.rendering_use_case import RenderingUseCase
                from src.application.services.festival_service import get_festival_details_by_title
                from src.infrastructure.config.loader import DF_SPLIT, DF_CAMERA

                festival_details = get_festival_details_by_title(festival_name)
                if festival_details:
                    rendering_use_case = RenderingUseCase(
                        df_split=DF_SPLIT,
                        df_camera=DF_CAMERA,
                        script_dir=SCRIPT_DIR
                    )

                    rendering_result = run_async(
                        rendering_use_case.generate_festival_renderings(festival_details)
                    )

                    if rendering_result:
                        save_to_cache(rendering_result, "ai_rendering", festival_name=festival_name)
                        result["cached_items"].append("ai_rendering")
                        logger.info(f"[PreCache] AI 렌더링 완료: {festival_name}")
                else:
                    logger.warning(f"[PreCache] AI 렌더링 스킵 (축제 정보 없음): {festival_name}")

        except Exception as e:
            error_msg = f"AI 렌더링 오류: {str(e)}"
            logger.error(f"[PreCache] {error_msg}")
            result["errors"].append(error_msg)

        result["status"] = "completed"
        logger.info(f"[PreCache] 완료: {festival_name} - {result['cached_items']}")

    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        logger.error(f"[PreCache] 실패: {festival_name} - {e}")

    return result


async def _calculate_festival_unit_score(festival_name: str, num_reviews: int) -> Dict[str, Any]:
    """
    개별 축제의 랭킹 점수를 계산합니다.

    이 점수들은 나중에 사용자가 여러 축제를 선택해서 랭킹을 요청할 때
    합산되어 사용됩니다.
    """
    from src.application.use_cases.ranking_use_case import RankingUseCase
    from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent
    from src.application.services.festival_service import get_festival_details_by_title

    # 축제 상세 정보 가져오기
    festival_info = get_festival_details_by_title(festival_name)
    if not festival_info:
        return None

    naver_agent = NaverReviewAgent()
    ranking_use_case = RankingUseCase(naver_agent)

    # 각 점수 계산
    start_date = festival_info.get("eventstartdate")
    end_date = festival_info.get("eventenddate")

    # 시간 점수
    time_score = ranking_use_case._get_time_score(str(start_date) if start_date else "", str(end_date) if end_date else "")

    # 트렌드 점수
    quarterly_trend_score = await ranking_use_case._get_trend_score(festival_name, days=90)
    yearly_trend_score = await ranking_use_case._get_trend_score(festival_name, days=365)

    # 감성 점수
    sentiment_score, positive_judgments = await ranking_use_case._get_sentiment_score(festival_name, num_reviews)

    # 이유 요약
    trend_reason = await ranking_use_case._summarize_trend_reasons(festival_name)
    sentiment_reason = await ranking_use_case._summarize_sentiment_reasons(positive_judgments, festival_name)

    return {
        "festival_name": festival_name,
        "time_score": round(time_score * 100, 2),
        "sentiment_score": round(sentiment_score, 2),
        "quarterly_trend_score": round(quarterly_trend_score, 2),
        "yearly_trend_score": round(yearly_trend_score, 2),
        "trend_reason": trend_reason,
        "sentiment_reason": sentiment_reason,
        "positive_judgments": positive_judgments[:20],  # 상위 20개만 저장
        "calculated_at": datetime.now().isoformat(),
    }


# ============================================================================
# 전체 축제 캐싱 태스크
# ============================================================================

@celery_app.task(bind=True, name="precache_all_festivals")
def precache_all_festivals(
    self,
    num_reviews: int = 50,
    skip_if_cached: bool = True,
    active_only: bool = False,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    모든 축제에 대해 사전 캐싱을 수행합니다.

    Args:
        num_reviews: 분석할 최대 리뷰 수
        skip_if_cached: 이미 캐시된 항목 스킵 여부
        active_only: 현재 활성 축제만 처리할지 여부
        limit: 처리할 최대 축제 수 (테스트용)
    """
    task_id = self.request.id

    # 축제 목록 가져오기
    if active_only:
        festivals = get_active_festival_names()
        logger.info(f"[PreCache] 활성 축제 {len(festivals)}개 로드됨")
    else:
        festivals = get_all_festival_names()
        logger.info(f"[PreCache] 전체 축제 {len(festivals)}개 로드됨")

    if limit:
        festivals = festivals[:limit]
        logger.info(f"[PreCache] {limit}개로 제한됨")

    total = len(festivals)
    results = {
        "total": total,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "details": []
    }

    save_task_progress(task_id, 0, "running", f"전수 캐싱 시작: {total}개 축제")

    for i, festival_name in enumerate(festivals):
        try:
            # 진행률 업데이트
            progress = int((i / total) * 100)
            save_task_progress(
                task_id, progress, "running",
                f"처리 중: {festival_name} ({i+1}/{total})"
            )

            # 개별 축제 캐싱 (직접 호출, 서브태스크 아님)
            result = precache_festival_unit(
                festival_name,
                num_reviews=num_reviews,
                skip_if_cached=skip_if_cached
            )

            if result["status"] == "completed":
                if any("already cached" in item for item in result["cached_items"]):
                    results["skipped"] += 1
                else:
                    results["completed"] += 1
            else:
                results["failed"] += 1

            results["details"].append({
                "festival": festival_name,
                "status": result["status"],
                "cached": result["cached_items"],
                "errors": result["errors"]
            })

        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "festival": festival_name,
                "status": "failed",
                "errors": [str(e)]
            })
            logger.error(f"[PreCache] 축제 처리 실패: {festival_name} - {e}")

    save_task_progress(task_id, 100, "completed", f"전수 캐싱 완료: 성공 {results['completed']}, 실패 {results['failed']}")
    save_task_result(task_id, results)

    logger.info(f"[PreCache] 전수 캐싱 완료: 성공={results['completed']}, 실패={results['failed']}, 스킵={results['skipped']}")

    return results


# ============================================================================
# 스케줄 태스크 (매일 실행)
# ============================================================================

@celery_app.task(name="daily_precache_active_festivals")
def daily_precache_active_festivals():
    """
    매일 실행되는 태스크: 현재 활성 축제들을 우선적으로 캐싱합니다.
    """
    logger.info("[DailyPreCache] 일일 활성 축제 캐싱 시작")
    return precache_all_festivals.delay(
        num_reviews=50,
        skip_if_cached=True,
        active_only=True
    )


@celery_app.task(name="weekly_precache_all_festivals")
def weekly_precache_all_festivals():
    """
    주간 실행되는 태스크: 모든 축제를 캐싱합니다.
    """
    logger.info("[WeeklyPreCache] 주간 전체 축제 캐싱 시작")
    return precache_all_festivals.delay(
        num_reviews=50,
        skip_if_cached=True,
        active_only=False
    )
