import os
import requests
import re
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

def get_naver_api_keys():
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Warning: NAVER_CLIENT_ID or NAVER_CLIENT_SECRET not set in environment variables.")
        return None, None
    return client_id, client_secret

def clean_html(raw_html):
    """HTML 태그를 제거하는 간단한 함수"""
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()

def search_naver_blog(query, display=5):
    """네이버 블로그 검색 API를 호출하고 결과를 반환합니다."""
    client_id, client_secret = get_naver_api_keys()
    if not client_id or not client_secret:
        return []

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    
    params = {
        "query": query,
        "display": display,
        "sort": "sim"  # 관련도순 정렬
    }

    try:
        response = requests.get("https://openapi.naver.com/v1/search/blog.json", headers=headers, params=params)
        response.raise_for_status()  # 오류 발생 시 예외 처리
        
        data = response.json()
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": clean_html(item.get("title", "")),
                "description": clean_html(item.get("description", "")),
                "link": item.get("link", ""),
                "postdate": item.get("postdate", "")
            })
        return results

    except requests.exceptions.RequestException as e:
        print(f"네이버 블로그 API 호출 오류: {e}")
        return []
    except Exception as e:
        print(f"블로그 데이터 처리 중 오류: {e}")
        return []
