from langchain_google_genai import ChatGoogleGenerativeAI
from src.infrastructure.config.settings import get_google_api_key

def get_llm_client(temperature: float = 0.0, model: str = "gemini-2.5-pro") -> ChatGoogleGenerativeAI:
    """Google Generative AI LLM 클라이언트를 반환합니다."""
    try:
        api_key = get_google_api_key()
        return ChatGoogleGenerativeAI(temperature=temperature, model=model, google_api_key=api_key)
    except Exception as e:
        print(f"LLM 초기화 오류: {e}. GOOGLE_API_KEY가 .env 파일에 설정되었는지 확인하세요.")
        # 여기서 None을 반환하거나, 예외를 다시 발생시킬 수 있습니다.
        # 예외를 다시 발생시키면 앱 시작 시 문제를 명확히 알 수 있습니다.
        raise e
