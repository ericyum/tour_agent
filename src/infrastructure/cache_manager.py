# src/infrastructure/cache_manager.py
"""
캐시 관리 모듈

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
from typing import Optional, Dict, Any, List
import traceback


# 캐시 설정
CACHE_DIR = "cache"
CACHE_EXPIRY_DAYS = 30  # 캐시 만료 기간 (일)


def get_cache_key(*args, **kwargs) -> str:
    """
    입력 파라미터로 유니크한 캐시 키 생성

    Args:
        *args: 위치 인자들
        **kwargs: 키워드 인자들

    Returns:
        str: MD5 해시 키

    Example:
        >>> get_cache_key("벚꽃축제", num_reviews=10)
        'a1b2c3d4e5f6...'
    """
    # 모든 인자를 문자열로 변환하여 결합
    key_parts = [str(arg) for arg in args]
    # kwargs를 정렬하여 순서에 상관없이 같은 키 생성
    for k in sorted(kwargs.keys()):
        key_parts.append(f"{k}={kwargs[k]}")

    key_str = "_".join(key_parts)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def get_cache_path(cache_key: str) -> str:
    """캐시 파일 경로 반환"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cache_key}.json")


def is_cache_valid(cache_path: str) -> bool:
    """
    캐시 파일이 유효한지 확인 (존재 여부 및 만료 기간)

    Args:
        cache_path: 캐시 파일 경로

    Returns:
        bool: 캐시가 유효하면 True
    """
    if not os.path.exists(cache_path):
        return False

    # 캐시 파일 수정 시간 확인
    file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    expiry_date = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)

    return file_mtime > expiry_date


def load_from_cache(*args, **kwargs) -> Optional[Dict[str, Any]]:
    """
    캐시에서 데이터 로드

    Args:
        *args: 캐시 키 생성에 사용할 위치 인자
        **kwargs: 캐시 키 생성에 사용할 키워드 인자

    Returns:
        Optional[Dict]: 캐시된 데이터 또는 None

    Example:
        >>> data = load_from_cache("trend", festival_name="벚꽃축제")
    """
    try:
        cache_key = get_cache_key(*args, **kwargs)
        cache_path = get_cache_path(cache_key)

        if not is_cache_valid(cache_path):
            return None

        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            print(f"✅ 캐시 HIT: {args} {kwargs}")
            return cached_data
    except Exception as e:
        print(f"⚠️ 캐시 로드 실패: {e}")
        return None


def save_to_cache(data: Dict[str, Any], *args, **kwargs) -> None:
    """
    데이터를 캐시에 저장

    Args:
        data: 저장할 데이터 (dict)
        *args: 캐시 키 생성에 사용할 위치 인자
        **kwargs: 캐시 키 생성에 사용할 키워드 인자

    Example:
        >>> save_to_cache({"score": 85}, "trend", festival_name="벚꽃축제")
    """
    try:
        cache_key = get_cache_key(*args, **kwargs)
        cache_path = get_cache_path(cache_key)

        # JSON 직렬화 가능한 형태로 변환
        cacheable_data = _make_json_serializable(data)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cacheable_data, f, ensure_ascii=False, indent=2)
            print(f"💾 캐시 저장: {args} {kwargs}")
    except Exception as e:
        print(f"⚠️ 캐시 저장 실패 (작업은 계속 진행됨): {e}")
        traceback.print_exc()


def _make_json_serializable(obj: Any) -> Any:
    """
    객체를 JSON 직렬화 가능한 형태로 변환

    Args:
        obj: 변환할 객체

    Returns:
        JSON 직렬화 가능한 객체
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (datetime,)):
        return obj.isoformat() if obj else None
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # 직렬화할 수 없는 타입은 문자열로 변환
        return str(obj)


def clear_cache() -> int:
    """
    모든 캐시 파일 삭제

    Returns:
        int: 삭제된 파일 개수
    """
    count = 0
    try:
        if not os.path.exists(CACHE_DIR):
            return 0

        for filename in os.listdir(CACHE_DIR):
            if filename.endswith(".json"):
                os.remove(os.path.join(CACHE_DIR, filename))
                count += 1

        print(f"🗑️ 캐시 파일 {count}개 삭제 완료")
        return count
    except Exception as e:
        print(f"⚠️ 캐시 삭제 실패: {e}")
        return count


def clear_expired_cache() -> int:
    """
    만료된 캐시 파일만 삭제

    Returns:
        int: 삭제된 파일 개수
    """
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
    """
    캐시 통계 정보 반환

    Returns:
        Dict: 캐시 파일 개수, 총 크기 등
    """
    try:
        if not os.path.exists(CACHE_DIR):
            return {"total_files": 0, "total_size_mb": 0}

        total_files = 0
        total_size = 0

        for filename in os.listdir(CACHE_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(CACHE_DIR, filename)
                total_files += 1
                total_size += os.path.getsize(filepath)

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
    except Exception as e:
        print(f"⚠️ 캐시 통계 조회 실패: {e}")
        return {"total_files": 0, "total_size_mb": 0}
