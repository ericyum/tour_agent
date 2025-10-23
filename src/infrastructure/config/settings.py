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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dotenv_path = os.path.join(project_root, ".env")
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, encoding="utf-8")
    else:
        print(f".env 파일을 찾을 수 없습니다: {dotenv_path}. API 키가 환경 변수에 설정되었는지 확인하세요.")

class Settings:
    """
    애플리케이션 설정을 관리하는 클래스.
    환경 변수에서 필요한 값들을 로드합니다.
    """
    def __init__(self):
        self.NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
        self.NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.NAVER_TREND_CLIENT_ID = os.getenv("NAVER_TREND_CLIENT_ID")
        self.NAVER_TREND_CLIENT_SECRET = os.getenv("NAVER_TREND_CLIENT_SECRET")

        # 필요한 경우 여기에 다른 설정 값들을 추가합니다.

        self._validate_settings()

    def _validate_settings(self):
        if not self.NAVER_CLIENT_ID or not self.NAVER_CLIENT_SECRET:
            print("경고: NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET이 설정되지 않았습니다.")
        if not self.GOOGLE_API_KEY:
            print("경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
        if not self.NAVER_TREND_CLIENT_ID or not self.NAVER_TREND_CLIENT_SECRET:
            print("경고: NAVER_TREND_CLIENT_ID 또는 NAVER_TREND_CLIENT_SECRET이 설정되지 않았습니다.")

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