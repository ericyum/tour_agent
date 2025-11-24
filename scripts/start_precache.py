#!/usr/bin/env python
# start_precache.py
"""
수동으로 전수 캐싱을 시작하는 스크립트

사용법:
    # 전체 축제 캐싱 시작
    python start_precache.py

    # 활성 축제만 캐싱
    python start_precache.py --active-only

    # 테스트용 (5개만)
    python start_precache.py --limit 5

    # 이미 캐시된 항목도 다시 캐싱
    python start_precache.py --force

Docker 환경에서:
    docker-compose exec celery-worker python start_precache.py
"""

import argparse
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="전수 캐싱 시작")
    parser.add_argument("--active-only", action="store_true", help="현재 활성 축제만 캐싱")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 축제 수 (테스트용)")
    parser.add_argument("--force", action="store_true", help="이미 캐시된 항목도 다시 캐싱")
    parser.add_argument("--num-reviews", type=int, default=50, help="분석할 리뷰 수 (기본: 50)")
    parser.add_argument("--sync", action="store_true", help="비동기 대신 동기적으로 실행 (디버깅용)")

    args = parser.parse_args()

    print("=" * 60)
    print("FestMoment 전수 캐싱 시작")
    print("=" * 60)
    print(f"옵션:")
    print(f"  - 활성 축제만: {args.active_only}")
    print(f"  - 제한: {args.limit or '없음'}")
    print(f"  - 강제 재캐싱: {args.force}")
    print(f"  - 리뷰 수: {args.num_reviews}")
    print(f"  - 동기 실행: {args.sync}")
    print("=" * 60)

    if args.sync:
        # 동기적으로 직접 실행 (디버깅용)
        from precache_tasks import precache_all_festivals
        result = precache_all_festivals(
            num_reviews=args.num_reviews,
            skip_if_cached=not args.force,
            active_only=args.active_only,
            limit=args.limit
        )
        print("\n결과:")
        print(f"  - 완료: {result['completed']}")
        print(f"  - 실패: {result['failed']}")
        print(f"  - 스킵: {result['skipped']}")
    else:
        # Celery 태스크로 비동기 실행
        from precache_tasks import precache_all_festivals

        task = precache_all_festivals.delay(
            num_reviews=args.num_reviews,
            skip_if_cached=not args.force,
            active_only=args.active_only,
            limit=args.limit
        )

        print(f"\n태스크가 시작되었습니다!")
        print(f"태스크 ID: {task.id}")
        print(f"\n진행 상황 확인:")
        print(f"  - Flower 대시보드: http://localhost:5555")
        print(f"  - 또는: celery -A celery_app inspect active")


if __name__ == "__main__":
    main()
