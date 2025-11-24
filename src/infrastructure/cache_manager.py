# src/infrastructure/cache_manager.py
"""
캐시 관리 모듈 (Redis 우선, 파일 fallback)

정확하게 같은 조건(파라미터)으로 요청할 경우 캐시된 결과를 반환하여
API 호출 비용과 시간을 절약합니다.

캐시 대상:
- 트렌드 분석 (Naver DataLab API 호출)
- 감성 분석 (블로그 스크래핑 + LLM 분석)
- 워드클라우드 (이미지 생성)
- 리뷰 요약 (블로그 스크래핑 + LLM 요약)
- 주의사항 (LLM 생성)
- 랭킹 (여러 축제 분석)
- AI 렌더링 (Vision 모델 이미지 생성)
- 코스 검증 (지오코딩 + LLM 분석)
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import traceback

# Redis 설정
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_AVAILABLE = False
redis_client = None

try:
    import redis
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("✅ Redis 연결 성공")
except Exception as e:
    print(f"⚠️ Redis 연결 실패, 파일 캐시 사용: {e}")
    REDIS_AVAILABLE = False

# 파일 캐시 설정 (Redis fallback)
CACHE_DIR = "cache"
CACHE_EXPIRY_DAYS = 30  # 캐시 만료 기간 (일)
CACHE_EXPIRY_SECONDS = CACHE_EXPIRY_DAYS * 24 * 60 * 60

# 엔드포인트별 만료 시간 (초)
CACHE_EXPIRY_BY_TYPE = {
    "trend": 7 * 24 * 60 * 60,  # 7일
    "sentiment": 30 * 24 * 60 * 60,  # 30일
    "wordcloud": 30 * 24 * 60 * 60,  # 30일
    "review_summary": 30 * 24 * 60 * 60,  # 30일
    "precautions": 60 * 24 * 60 * 60,  # 60일
    "ranking": 30 * 24 * 60 * 60,  # 30일
    "render": 30 * 24 * 60 * 60,  # 30일
    "course": 7 * 24 * 60 * 60,  # 7일
    "festival_score": 30 * 24 * 60 * 60,  # 30일 (단위 캐싱용 개별 축제 점수)
}


def get_cache_key(*args, **kwargs) -> str:
    """
    입력 파라미터로 유니크한 캐시 키 생성

    Args:
        *args: 위치 인자들
        **kwargs: 키워드 인자들

    Returns:
        str: MD5 해시 키
    """
    key_parts = [str(arg) for arg in args]
    for k in sorted(kwargs.keys()):
        key_parts.append(f"{k}={kwargs[k]}")

    key_str = "_".join(key_parts)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def get_cache_expiry(*args) -> int:
    """캐시 타입에 따른 만료 시간 반환"""
    if args:
        cache_type = str(args[0])
        return CACHE_EXPIRY_BY_TYPE.get(cache_type, CACHE_EXPIRY_SECONDS)
    return CACHE_EXPIRY_SECONDS


# ============================================================================
# Redis 캐시 함수
# ============================================================================

def _redis_load(cache_key: str) -> Optional[Dict[str, Any]]:
    """Redis에서 캐시 로드"""
    try:
        data = redis_client.get(f"cache:{cache_key}")
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        print(f"⚠️ Redis 로드 실패: {e}")
        return None


def _redis_save(data: Dict[str, Any], cache_key: str, expiry_seconds: int) -> bool:
    """Redis에 캐시 저장"""
    try:
        redis_client.setex(
            f"cache:{cache_key}",
            expiry_seconds,
            json.dumps(data, ensure_ascii=False)
        )
        return True
    except Exception as e:
        print(f"⚠️ Redis 저장 실패: {e}")
        return False


# ============================================================================
# 파일 캐시 함수 (Fallback)
# ============================================================================

def get_cache_path(cache_key: str) -> str:
    """캐시 파일 경로 반환"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cache_key}.json")


def is_cache_valid(cache_path: str) -> bool:
    """캐시 파일이 유효한지 확인"""
    if not os.path.exists(cache_path):
        return False
    file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    expiry_date = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)
    return file_mtime > expiry_date


def _file_load(cache_key: str) -> Optional[Dict[str, Any]]:
    """파일에서 캐시 로드"""
    try:
        cache_path = get_cache_path(cache_key)
        if not is_cache_valid(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 파일 캐시 로드 실패: {e}")
        return None


def _file_save(data: Dict[str, Any], cache_key: str) -> bool:
    """파일에 캐시 저장"""
    try:
        cache_path = get_cache_path(cache_key)
        cacheable_data = _make_json_serializable(data)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cacheable_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ 파일 캐시 저장 실패: {e}")
        return False


# ============================================================================
# 메인 캐시 API
# ============================================================================

def load_from_cache(*args, **kwargs) -> Optional[Dict[str, Any]]:
    """
    캐시에서 데이터 로드 (Redis 우선, 파일 fallback)

    Args:
        *args: 캐시 키 생성에 사용할 위치 인자
        **kwargs: 캐시 키 생성에 사용할 키워드 인자

    Returns:
        Optional[Dict]: 캐시된 데이터 또는 None
    """
    cache_key = get_cache_key(*args, **kwargs)

    # Redis 시도
    if REDIS_AVAILABLE:
        data = _redis_load(cache_key)
        if data:
            print(f"✅ Redis 캐시 HIT: {args} {kwargs}")
            return data

    # 파일 캐시 fallback
    data = _file_load(cache_key)
    if data:
        print(f"✅ 파일 캐시 HIT: {args} {kwargs}")
        # Redis가 가용하면 Redis에도 저장
        if REDIS_AVAILABLE:
            expiry = get_cache_expiry(*args)
            _redis_save(data, cache_key, expiry)
        return data

    return None


def save_to_cache(data: Dict[str, Any], *args, **kwargs) -> None:
    """
    데이터를 캐시에 저장 (Redis와 파일 모두)

    Args:
        data: 저장할 데이터 (dict)
        *args: 캐시 키 생성에 사용할 위치 인자
        **kwargs: 캐시 키 생성에 사용할 키워드 인자
    """
    try:
        cache_key = get_cache_key(*args, **kwargs)
        expiry = get_cache_expiry(*args)
        cacheable_data = _make_json_serializable(data)

        # Redis에 저장
        if REDIS_AVAILABLE:
            if _redis_save(cacheable_data, cache_key, expiry):
                print(f"💾 Redis 캐시 저장: {args} {kwargs}")

        # 파일에도 저장 (백업)
        if _file_save(cacheable_data, cache_key):
            print(f"💾 파일 캐시 저장: {args} {kwargs}")

    except Exception as e:
        print(f"⚠️ 캐시 저장 실패 (작업은 계속 진행됨): {e}")
        traceback.print_exc()


def _make_json_serializable(obj: Any) -> Any:
    """객체를 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (datetime,)):
        return obj.isoformat() if obj else None
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


# ============================================================================
# 캐시 관리 유틸리티
# ============================================================================

def clear_cache() -> int:
    """모든 캐시 삭제"""
    count = 0

    # Redis 캐시 삭제
    if REDIS_AVAILABLE:
        try:
            keys = redis_client.keys("cache:*")
            if keys:
                count += redis_client.delete(*keys)
                print(f"🗑️ Redis 캐시 {count}개 삭제")
        except Exception as e:
            print(f"⚠️ Redis 캐시 삭제 실패: {e}")

    # 파일 캐시 삭제
    try:
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith(".json"):
                    os.remove(os.path.join(CACHE_DIR, filename))
                    count += 1
            print(f"🗑️ 파일 캐시 {count}개 삭제 완료")
    except Exception as e:
        print(f"⚠️ 파일 캐시 삭제 실패: {e}")

    return count


def clear_expired_cache() -> int:
    """만료된 캐시 파일만 삭제 (파일 캐시용, Redis는 자동 만료)"""
    count = 0
    try:
        if not os.path.exists(CACHE_DIR):
            return 0

        expiry_date = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)

        for filename in os.listdir(CACHE_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(CACHE_DIR, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

                if file_mtime < expiry_date:
                    os.remove(filepath)
                    count += 1

        print(f"🗑️ 만료된 캐시 파일 {count}개 삭제 완료")
        return count
    except Exception as e:
        print(f"⚠️ 만료된 캐시 삭제 실패: {e}")
        return count


def get_cache_stats() -> Dict[str, Any]:
    """캐시 통계 정보 반환"""
    stats = {
        "redis_available": REDIS_AVAILABLE,
        "redis_keys": 0,
        "file_cache_files": 0,
        "file_cache_size_mb": 0,
    }

    # Redis 통계
    if REDIS_AVAILABLE:
        try:
            keys = redis_client.keys("cache:*")
            stats["redis_keys"] = len(keys)
        except Exception:
            pass

    # 파일 캐시 통계
    try:
        if os.path.exists(CACHE_DIR):
            total_size = 0
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(CACHE_DIR, filename)
                    stats["file_cache_files"] += 1
                    total_size += os.path.getsize(filepath)
            stats["file_cache_size_mb"] = round(total_size / (1024 * 1024), 2)
    except Exception:
        pass

    return stats


# ============================================================================
# 작업 진행률 저장 (Redis 전용)
# ============================================================================

def save_task_progress(task_id: str, progress: int, status: str, message: str = "") -> bool:
    """작업 진행률 저장 (Redis 전용)"""
    if not REDIS_AVAILABLE:
        return False

    try:
        data = {
            "progress": progress,
            "status": status,
            "message": message,
            "updated_at": datetime.now().isoformat(),
        }
        redis_client.setex(
            f"task:{task_id}:progress",
            3600,  # 1시간 만료
            json.dumps(data)
        )
        return True
    except Exception as e:
        print(f"⚠️ 진행률 저장 실패: {e}")
        return False


def get_task_progress(task_id: str) -> Optional[Dict[str, Any]]:
    """작업 진행률 조회"""
    if not REDIS_AVAILABLE:
        return None

    try:
        data = redis_client.get(f"task:{task_id}:progress")
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        print(f"⚠️ 진행률 조회 실패: {e}")
        return None


def save_task_result(task_id: str, result: Dict[str, Any], expiry_seconds: int = 3600) -> bool:
    """작업 결과 저장 (Redis 전용)"""
    if not REDIS_AVAILABLE:
        return False

    try:
        redis_client.setex(
            f"task:{task_id}:result",
            expiry_seconds,
            json.dumps(_make_json_serializable(result), ensure_ascii=False)
        )
        return True
    except Exception as e:
        print(f"⚠️ 결과 저장 실패: {e}")
        return False


def get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """작업 결과 조회"""
    if not REDIS_AVAILABLE:
        return None

    try:
        data = redis_client.get(f"task:{task_id}:result")
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        print(f"⚠️ 결과 조회 실패: {e}")
        return None
