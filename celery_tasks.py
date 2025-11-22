# celery_tasks.py
"""
Celery 비동기 작업 정의

장시간 실행되는 분석 작업들을 백그라운드에서 처리합니다.
각 작업은 진행률을 Redis에 저장하여 프론트엔드에서 실시간 조회 가능합니다.
"""

import os
import sys
import asyncio
from typing import Dict, Any, Optional

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery_app import celery_app
from src.infrastructure.cache_manager import (
    save_task_progress,
    save_task_result,
    save_to_cache,
    load_from_cache,
)


def run_async(coro):
    """비동기 함수를 동기적으로 실행"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ============================================================================
# 감성 분석 작업
# ============================================================================

@celery_app.task(bind=True, name="analyze_sentiment")
def analyze_sentiment_task(self, festival_name: str, num_reviews: int = 10) -> Dict[str, Any]:
    """
    감성 분석 백그라운드 작업

    Args:
        festival_name: 축제 이름
        num_reviews: 분석할 리뷰 개수

    Returns:
        감성 분석 결과
    """
    task_id = self.request.id

    try:
        # 1. 캐시 확인
        save_task_progress(task_id, 5, "running", "캐시 확인 중...")
        cached = load_from_cache("sentiment", festival_name=festival_name, num_reviews=num_reviews)
        if cached:
            save_task_progress(task_id, 100, "completed", "캐시에서 결과 로드 완료")
            save_task_result(task_id, cached)
            return cached

        # 2. 분석 시작
        save_task_progress(task_id, 10, "running", "블로그 검색 중...")

        # Use Case 임포트 (지연 임포트로 순환 참조 방지)
        from src.application.use_cases.sentiment_analysis_use_case import SentimentAnalysisUseCase
        from src.application.agents.naver_review.naver_supervisor import NaverReviewSupervisor

        naver_supervisor = NaverReviewSupervisor()
        use_case = SentimentAnalysisUseCase(naver_supervisor)

        # 진행률 콜백 함수
        def progress_callback(current: int, total: int, message: str):
            # 10% ~ 80% 사이에서 진행률 업데이트
            progress = 10 + int((current / total) * 70)
            save_task_progress(task_id, progress, "running", message)

        # 3. 비동기 분석 실행
        result = run_async(
            use_case.analyze_sentiment_with_progress(
                festival_name,
                num_reviews,
                progress_callback=progress_callback
            )
        )

        # 4. 워드클라우드 생성
        save_task_progress(task_id, 85, "running", "워드클라우드 생성 중...")
        from src.application.use_cases.analysis_use_case import AnalysisUseCase
        from io import BytesIO
        import base64

        analysis_use_case = AnalysisUseCase(naver_supervisor)
        wc_image, _ = run_async(
            analysis_use_case.generate_word_cloud(festival_name, num_reviews)
        )

        wordcloud_base64 = None
        if wc_image:
            buffered = BytesIO()
            wc_image.save(buffered, format="PNG")
            wordcloud_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 5. 결과 구성
        save_task_progress(task_id, 95, "running", "결과 정리 중...")

        final_result = {
            "festival_name": result.get("festival_name", festival_name),
            "total_reviews": result.get("total_reviews", 0),
            "satisfaction_scores": result.get("satisfaction_scores", []),
            "avg_score": result.get("avg_score", 0),
            "median_score": result.get("median_score", 0),
            "std_score": result.get("std_score", 0),
            "satisfaction_level": result.get("satisfaction_level", ""),
            "lower_bound": result.get("lower_bound"),
            "upper_bound": result.get("upper_bound"),
            "satisfaction_distribution_chart": result.get("satisfaction_distribution_chart"),
            "absolute_score_chart": result.get("absolute_score_chart"),
            "outliers_box_chart": result.get("outliers_box_chart"),
            "distribution_interpretation": result.get("distribution_interpretation", ""),
            "positive_keywords_summary": result.get("positive_keywords_summary", ""),
            "wordcloud_image": wordcloud_base64,
        }

        # 6. 캐시 저장
        save_to_cache(final_result, "sentiment", festival_name=festival_name, num_reviews=num_reviews)

        # 7. 작업 완료
        save_task_progress(task_id, 100, "completed", "분석 완료!")
        save_task_result(task_id, final_result)

        return final_result

    except Exception as e:
        error_msg = str(e)
        save_task_progress(task_id, 0, "failed", f"오류 발생: {error_msg}")
        save_task_result(task_id, {"error": error_msg})
        raise


# ============================================================================
# 랭킹 분석 작업
# ============================================================================

@celery_app.task(bind=True, name="analyze_ranking")
def analyze_ranking_task(
    self,
    festivals: list,
    num_reviews: int = 5,
    top_n: int = 3
) -> Dict[str, Any]:
    """
    축제 랭킹 분석 백그라운드 작업

    Args:
        festivals: 축제 이름 목록
        num_reviews: 축제당 분석할 리뷰 개수
        top_n: 상위 N개 축제

    Returns:
        랭킹 분석 결과
    """
    task_id = self.request.id

    try:
        # 1. 캐시 확인
        save_task_progress(task_id, 5, "running", "캐시 확인 중...")
        sorted_festivals = tuple(sorted(festivals))
        cached = load_from_cache(
            "ranking",
            festivals=sorted_festivals,
            num_reviews=num_reviews,
            top_n=top_n
        )
        if cached:
            save_task_progress(task_id, 100, "completed", "캐시에서 결과 로드 완료")
            save_task_result(task_id, cached)
            return cached

        # 2. 분석 시작
        save_task_progress(task_id, 10, "running", "축제 분석 준비 중...")

        from src.application.use_cases.ranking_use_case import RankingUseCase
        from src.application.agents.naver_review.naver_supervisor import NaverReviewSupervisor

        naver_supervisor = NaverReviewSupervisor()
        use_case = RankingUseCase(naver_supervisor)

        # 진행률 콜백
        def progress_callback(current: int, total: int, festival_name: str):
            progress = 10 + int((current / total) * 80)
            save_task_progress(task_id, progress, "running", f"{festival_name} 분석 중... ({current}/{total})")

        # 3. 분석 실행
        result = run_async(
            use_case.rank_festivals_with_progress(
                festivals,
                num_reviews,
                top_n,
                progress_callback=progress_callback
            )
        )

        # 4. 캐시 저장
        save_to_cache(
            result,
            "ranking",
            festivals=sorted_festivals,
            num_reviews=num_reviews,
            top_n=top_n
        )

        # 5. 작업 완료
        save_task_progress(task_id, 100, "completed", "랭킹 분석 완료!")
        save_task_result(task_id, result)

        return result

    except Exception as e:
        error_msg = str(e)
        save_task_progress(task_id, 0, "failed", f"오류 발생: {error_msg}")
        save_task_result(task_id, {"error": error_msg})
        raise


# ============================================================================
# AI 렌더링 작업
# ============================================================================

@celery_app.task(bind=True, name="render_ai_image")
def render_ai_image_task(
    self,
    festival_name: str,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    AI 이미지 렌더링 백그라운드 작업

    Args:
        festival_name: 축제 이름
        prompt: 커스텀 프롬프트 (선택)

    Returns:
        렌더링 결과
    """
    task_id = self.request.id

    try:
        # 1. 캐시 확인
        save_task_progress(task_id, 5, "running", "캐시 확인 중...")
        cached = load_from_cache("render", festival_name=festival_name, prompt=prompt or "default")
        if cached:
            save_task_progress(task_id, 100, "completed", "캐시에서 결과 로드 완료")
            save_task_result(task_id, cached)
            return cached

        # 2. 렌더링 시작
        save_task_progress(task_id, 20, "running", "AI 이미지 생성 중...")

        from src.application.use_cases.ai_render_use_case import AIRenderUseCase
        use_case = AIRenderUseCase()

        # 3. 렌더링 실행
        save_task_progress(task_id, 40, "running", "Vision 모델 처리 중...")
        result = run_async(use_case.render_festival_image(festival_name, prompt))

        # 4. 캐시 저장
        save_task_progress(task_id, 90, "running", "결과 저장 중...")
        save_to_cache(result, "render", festival_name=festival_name, prompt=prompt or "default")

        # 5. 작업 완료
        save_task_progress(task_id, 100, "completed", "이미지 생성 완료!")
        save_task_result(task_id, result)

        return result

    except Exception as e:
        error_msg = str(e)
        save_task_progress(task_id, 0, "failed", f"오류 발생: {error_msg}")
        save_task_result(task_id, {"error": error_msg})
        raise


# ============================================================================
# 워드클라우드 작업
# ============================================================================

@celery_app.task(bind=True, name="generate_wordcloud")
def generate_wordcloud_task(
    self,
    festival_name: str,
    num_reviews: int = 20
) -> Dict[str, Any]:
    """
    워드클라우드 생성 백그라운드 작업
    """
    task_id = self.request.id

    try:
        # 1. 캐시 확인
        save_task_progress(task_id, 5, "running", "캐시 확인 중...")
        cached = load_from_cache("wordcloud", festival_name=festival_name, num_reviews=num_reviews)
        if cached:
            save_task_progress(task_id, 100, "completed", "캐시에서 결과 로드 완료")
            save_task_result(task_id, cached)
            return cached

        # 2. 생성 시작
        save_task_progress(task_id, 20, "running", "블로그 리뷰 수집 중...")

        from src.application.use_cases.analysis_use_case import AnalysisUseCase
        from src.application.agents.naver_review.naver_supervisor import NaverReviewSupervisor
        from io import BytesIO
        import base64

        naver_supervisor = NaverReviewSupervisor()
        use_case = AnalysisUseCase(naver_supervisor)

        # 3. 워드클라우드 생성
        save_task_progress(task_id, 50, "running", "워드클라우드 생성 중...")
        wc_image, message = run_async(use_case.generate_word_cloud(festival_name, num_reviews))

        result = {"festival_name": festival_name, "message": message}

        if wc_image:
            buffered = BytesIO()
            wc_image.save(buffered, format="PNG")
            result["wordcloud_image"] = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 4. 캐시 저장
        save_task_progress(task_id, 90, "running", "결과 저장 중...")
        save_to_cache(result, "wordcloud", festival_name=festival_name, num_reviews=num_reviews)

        # 5. 작업 완료
        save_task_progress(task_id, 100, "completed", "워드클라우드 생성 완료!")
        save_task_result(task_id, result)

        return result

    except Exception as e:
        error_msg = str(e)
        save_task_progress(task_id, 0, "failed", f"오류 발생: {error_msg}")
        save_task_result(task_id, {"error": error_msg})
        raise
