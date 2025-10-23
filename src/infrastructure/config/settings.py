import os
from dotenv import load_dotenv

def setup_environment():
    """
    환경 변수 및 로깅 설정을 초기화합니다.
    """
    # gRPC 로깅 수준 설정 (불필요한 ALTS 로그 메시지 숨기기)
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    # KoNLPy/JPype 로딩 전, JVM 인코딩 설정 (Windows 환경에서 필수)
    os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

    # .env 파일 로드
    # 프로젝트 루트에 있는 .env 파일을 찾도록 경로 수정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(project_root, ".env")
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, encoding="utf-8")
    else:
        print(f".env 파일을 찾을 수 없습니다: {dotenv_path}. API 키가 환경 변수에 설정되었는지 확인하세요.")

def get_naver_api_keys():
    """네이버 API 키를 반환합니다."""
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(".env 파일에 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 설정하세요.")
    return client_id, client_secret

def get_google_api_key():
    """Google API 키를 반환합니다."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(".env 파일에 GOOGLE_API_KEY를 설정하세요.")
    return api_key

def get_naver_trend_api_keys():
    """네이버 트렌드 API 키를 반환합니다."""
    client_id = os.getenv("NAVER_TREND_CLIENT_ID")
    client_secret = os.getenv("NAVER_TREND_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(".env 파일에 NAVER_TREND_CLIENT_ID와 NAVER_TREND_CLIENT_SECRET을 설정하세요.")
    return client_id, client_secret

# 초기 환경 설정 실행
setup_environment()
