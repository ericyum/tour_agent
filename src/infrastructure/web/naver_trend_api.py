import requests
import os
import json
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

NAVER_TREND_CLIENT_ID = os.getenv("NAVER_TREND_CLIENT_ID")
NAVER_TREND_CLIENT_SECRET = os.getenv("NAVER_TREND_CLIENT_SECRET")

def get_naver_trend(keyword, start_date, end_date):
    """네이버 데이터랩 검색어 트렌드 API를 호출하고 결과를 반환합니다."""
    if not NAVER_TREND_CLIENT_ID or not NAVER_TREND_CLIENT_SECRET:
        print("네이버 트렌드 API 인증 정보가 .env 파일에 설정되지 않았습니다.")
        return None

    headers = {
        "X-Naver-Client-Id": NAVER_TREND_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_TREND_CLIENT_SECRET,
        "Content-Type": "application/json"
    }

    body = {
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "timeUnit": "date",
        "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
    }

    try:
        response = requests.post("https://openapi.naver.com/v1/datalab/search", headers=headers, data=json.dumps(body))
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('results') or not data['results'][0].get('data'):
            return None
            
        return data['results'][0]['data']

    except requests.exceptions.RequestException as e:
        print(f"네이버 트렌드 API 호출 오류: {e}")
        return None
    except Exception as e:
        print(f"트렌드 데이터 처리 중 오류: {e}")
        return None
