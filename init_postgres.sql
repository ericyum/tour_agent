-- FestMoment PostgreSQL 초기화 스크립트
-- 이 스크립트는 Docker PostgreSQL 컨테이너가 처음 시작될 때 자동으로 실행됩니다.
-- 메인 테이블 스키마는 애플리케이션의 database.py init_db() 함수에서 생성됩니다.

-- UUID 확장 (필요시 사용)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 전문 검색용 확장 (한글 검색 개선)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 기본 인덱스 생성을 위한 함수 (애플리케이션 시작 후 실행)
-- 참고: 실제 테이블은 Python 애플리케이션에서 생성되므로
-- 여기서는 확장 기능만 활성화합니다.

-- 연결 풀링 최적화를 위한 설정
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '128MB';

-- 로그 메시지
DO $$
BEGIN
    RAISE NOTICE 'FestMoment PostgreSQL 초기화 완료. 테이블은 애플리케이션 시작 시 생성됩니다.';
END $$;
