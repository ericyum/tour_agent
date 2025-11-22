# celery_app.py
"""
Celery 비동기 작업 큐 설정

장시간 실행되는 분석 작업을 백그라운드에서 처리합니다.
사용자가 브라우저를 닫아도 작업은 계속 실행되고 결과가 캐시됩니다.

실행 방법:
    celery -A celery_app worker --loglevel=info --pool=solo  # Windows
    celery -A celery_app worker --loglevel=info              # Linux/Mac

모니터링 (Flower):
    celery -A celery_app flower --port=5555
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Redis URL 설정
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Celery 앱 생성
celery_app = Celery(
    "festmoment",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["celery_tasks"],  # 작업 모듈
)

# Celery 설정
celery_app.conf.update(
    # 작업 설정
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,

    # 결과 설정
    result_expires=3600,  # 결과 1시간 후 만료

    # 작업 실행 설정
    task_acks_late=True,  # 작업 완료 후 ACK
    task_reject_on_worker_lost=True,  # 워커 손실 시 작업 재시도
    task_time_limit=600,  # 10분 타임아웃
    task_soft_time_limit=540,  # 9분 soft 타임아웃

    # 동시성 설정
    worker_prefetch_multiplier=1,  # 한 번에 하나씩 가져옴
    worker_concurrency=2,  # 동시 작업 수

    # 재시도 설정
    task_default_retry_delay=60,  # 1분 후 재시도
    task_max_retries=3,  # 최대 3번 재시도
)


if __name__ == "__main__":
    celery_app.start()
