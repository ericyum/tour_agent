# src/infrastructure/rate_limiter.py
"""
API Rate Limiter

API 호출 시 과도한 요청으로 인한 차단을 방지하기 위한 속도 제한 유틸리티.
네이버 API, Google Gemini API 등 외부 서비스 호출 시 사용합니다.

사용 예시:
    from src.infrastructure.rate_limiter import rate_limiter

    # 네이버 API 호출 전
    rate_limiter.wait("naver_search")
    result = search_naver_blog(query)

    # Gemini API 호출 전
    rate_limiter.wait("gemini")
    response = llm.invoke(prompt)
"""

import time
import threading
from typing import Dict, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe Rate Limiter

    각 API 유형별로 마지막 호출 시간을 추적하고,
    최소 간격이 지나지 않았으면 대기합니다.
    """

    # API별 최소 대기 시간 (초)
    DEFAULT_DELAYS = {
        "naver_search": 3.0,      # 네이버 검색 API: 3초
        "naver_blog_scrape": 5.0, # 네이버 블로그 스크래핑: 5초
        "naver_trend": 2.0,       # 네이버 트렌드 API: 2초
        "gemini": 4.0,            # Google Gemini: 4초 (분당 15회 = 4초 간격)
        "gemini_vision": 6.0,     # Gemini Vision: 6초 (더 보수적)
        "default": 1.0,           # 기본값
    }

    # 429 에러 시 백오프 설정
    BACKOFF_CONFIG = {
        "initial_delay": 60,      # 첫 번째 재시도: 60초
        "max_delay": 3600,        # 최대 대기: 1시간
        "multiplier": 2,          # 지수 백오프 배수
    }

    def __init__(self, custom_delays: Optional[Dict[str, float]] = None):
        self._last_call_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._backoff_counts: Dict[str, int] = {}

        # 커스텀 딜레이 설정 병합
        self.delays = self.DEFAULT_DELAYS.copy()
        if custom_delays:
            self.delays.update(custom_delays)

    def wait(self, api_type: str = "default", extra_delay: float = 0) -> float:
        """
        API 호출 전 필요한 만큼 대기

        Args:
            api_type: API 유형 (naver_search, gemini 등)
            extra_delay: 추가 대기 시간 (초)

        Returns:
            실제 대기한 시간 (초)
        """
        min_delay = self.delays.get(api_type, self.delays["default"]) + extra_delay

        with self._lock:
            now = time.time()
            last_call = self._last_call_times.get(api_type, 0)
            elapsed = now - last_call

            if elapsed < min_delay:
                wait_time = min_delay - elapsed
                logger.debug(f"[RateLimiter] {api_type}: {wait_time:.2f}초 대기")
                time.sleep(wait_time)
                actual_wait = wait_time
            else:
                actual_wait = 0

            self._last_call_times[api_type] = time.time()
            return actual_wait

    def wait_with_backoff(self, api_type: str, error_occurred: bool = False) -> float:
        """
        에러 발생 시 지수 백오프 적용

        Args:
            api_type: API 유형
            error_occurred: 이전 호출에서 에러가 발생했는지

        Returns:
            대기한 시간 (초)
        """
        if error_occurred:
            # 백오프 카운트 증가
            with self._lock:
                count = self._backoff_counts.get(api_type, 0) + 1
                self._backoff_counts[api_type] = count

            # 지수 백오프 계산
            delay = min(
                self.BACKOFF_CONFIG["initial_delay"] * (self.BACKOFF_CONFIG["multiplier"] ** (count - 1)),
                self.BACKOFF_CONFIG["max_delay"]
            )

            logger.warning(f"[RateLimiter] {api_type}: 에러 발생, {delay}초 백오프 대기 (시도 #{count})")
            time.sleep(delay)
            return delay
        else:
            # 성공 시 백오프 카운트 리셋
            with self._lock:
                self._backoff_counts[api_type] = 0
            return self.wait(api_type)

    def reset_backoff(self, api_type: str):
        """특정 API의 백오프 카운트 리셋"""
        with self._lock:
            self._backoff_counts[api_type] = 0

    def get_stats(self) -> Dict[str, any]:
        """Rate limiter 상태 조회"""
        with self._lock:
            return {
                "last_call_times": self._last_call_times.copy(),
                "backoff_counts": self._backoff_counts.copy(),
                "configured_delays": self.delays.copy(),
            }


# 전역 인스턴스
rate_limiter = RateLimiter()


def with_rate_limit(api_type: str = "default"):
    """
    Rate limiting 데코레이터

    사용 예시:
        @with_rate_limit("naver_search")
        def search_naver_blog(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter.wait(api_type)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            rate_limiter.wait(api_type)
            return await func(*args, **kwargs)

        # 비동기 함수인지 확인
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def with_retry_and_rate_limit(api_type: str = "default", max_retries: int = 3):
    """
    Rate limiting + 재시도 로직 데코레이터

    429 에러 등 발생 시 지수 백오프로 재시도합니다.

    사용 예시:
        @with_retry_and_rate_limit("gemini", max_retries=3)
        def call_gemini(prompt):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    if attempt == 0:
                        rate_limiter.wait(api_type)
                    else:
                        rate_limiter.wait_with_backoff(api_type, error_occurred=True)

                    result = func(*args, **kwargs)
                    rate_limiter.reset_backoff(api_type)
                    return result

                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # 재시도 가능한 에러인지 확인
                    is_retryable = any(keyword in error_str for keyword in [
                        "429", "rate limit", "too many requests",
                        "quota", "exceeded", "timeout"
                    ])

                    if not is_retryable or attempt == max_retries:
                        raise

                    logger.warning(f"[RateLimiter] {api_type}: 재시도 {attempt + 1}/{max_retries}")

            raise last_error

        return wrapper

    return decorator
