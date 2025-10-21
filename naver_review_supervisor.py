import os
import requests
import json
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain_openai import ChatOpenAI
import asyncio
import nltk
import re

# Ensure NLTK punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv() # Load environment variables from .env file

# --- Configuration for Naver API ---
def get_naver_api_keys():
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Warning: NAVER_CLIENT_ID or NAVER_CLIENT_SECRET not set in environment variables.")
        return None, None
    return client_id, client_secret

# --- Naver Blog Search (from TourLens/modules/naver_search/naver_review.py) ---
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

# --- Blog Content Scraper (from TourLens/modules/naver_search/search.py) ---
async def scrape_blog_content(url: str) -> tuple[str, list[str]]:
    """
    Playwright를 사용하여 주어진 URL의 블로그 본문 텍스트와 이미지 URL들을 스크래핑합니다.
    """
    text_content = ""
    image_urls = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)

            main_frame = page
            try:
                main_frame_element = await page.wait_for_selector(
                    "iframe#mainFrame", timeout=5000
                )
                main_frame = await main_frame_element.content_frame()
                if main_frame is None:
                    main_frame = page
            except Exception:
                main_frame = page

            content_selectors = [
                "div.se-main-container",
                "div.post-view",
                "#postViewArea",
            ]
            content_element = None
            for selector in content_selectors:
                try:
                    await main_frame.wait_for_selector(selector, timeout=5000)
                    content_element = await main_frame.query_selector(selector)
                    if content_element:
                        text_content = await content_element.inner_text()
                        if text_content.strip():
                            break  # 내용 찾았으면 중단
                except Exception:
                    continue

            await browser.close()

            if not text_content.strip():
                return (
                    "본문 내용을 찾을 수 없습니다. (지원되지 않는 블로그 구조일 수 있습니다)",
                    [],
                )

            return text_content, image_urls # image_urls is not used in this context, but kept for consistency

    except Exception as e:
        return f"페이지에 접근하는 중 오류가 발생했습니다: {e}", []

# --- LLM-based Summarization and Tip Extraction (adapted from TourLens/modules/naver_search/search.py) ---
class NaverReviewSupervisor:
    def __init__(self):
        self.llm = None
        self.initialize_llm()

    def initialize_llm(self):
        if "OPENAI_API_KEY" not in os.environ:
            print("Warning: OPENAI_API_KEY not set in environment variables. LLM functions will be disabled.")
            self.llm = None
        else:
            try:
                self.llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-mini")
            except Exception as e:
                print(f"Error initializing ChatOpenAI model: {e}. LLM functions will be disabled.")
                self.llm = None

    async def summarize_blog_contents_stream(self, reviews_data):
        if not self.llm:
            yield "오류: LLM이 초기화되지 않았습니다. OPENAI_API_KEY를 확인해주세요."
            return
        if not reviews_data:
            yield "요약할 내용이 없습니다."
            return

        full_text = ""
        for i, review in enumerate(reviews_data):
            content = review.get("content", "")
            if (
                "본문 내용을 찾을 수 없습니다" in content
                or "페이지에 접근하는 중 오류" in content
            ):
                continue
            full_text += f"--- 블로그 후기 {i+1} ---\n\n{content}\n\n"

        if not full_text.strip():
            yield "요약할 유효한 블로그 본문이 없습니다."
            return

        initial_prompt = f"""
        다음은 하나의 주제에 대한 여러 네이버 블로그 후기 내용입니다.
        이 후기들을 종합하여 해당 주제(행사, 장소 등)의 주요 특징, 방문객들의 전반적인 반응, 긍정적인 점과 아쉬운 점을 중심으로 상세하게 요약해주세요.
        특히, 방문객들이 해당 행사를 즐긴 후 근처의 다른 음식점, 카페, 볼거리, 즐길거리 등을 이어서 방문했다는 내용이 있다면 그 부분도 놓치지 말고 요약에 포함해주세요.
        각 블로그의 내용을 단순히 나열하는 것이 아니라, 전체적인 관점에서 정보를 종합하고 재구성하여 전달해야 합니다.

        --- 전체 후기 내용 ---
        {full_text}
        ---
        
        위 내용을 바탕으로 상세 요약:
        """

        try:
            initial_summary = self.llm.invoke(initial_prompt).content
        except Exception as e:
            yield f"OpenAI API 1차 요약 중 오류가 발생했습니다: {e}"
            return

        filtering_prompt = f"""
        아래는 여러 블로그 후기를 바탕으로 생성된 1차 요약본입니다.
        이 요약본에서, 공식 관광 사이트에서는 얻기 힘든 '실제 방문객들의 주관적인 경험'과 관련된 내용만을 추출해주세요.

        추출한 내용을 다음 소주제들에 맞춰 최대한 상세하고 다양하게 분류하고 정리해주세요. 각 소주제에 해당하는 내용이 없다면 그 소주제는 결과에서 생략해주세요.

        - **추천 방문 대상 (누구와 함께 가면 좋을까?)**: (예: "커플 데이트 코스로 완벽해요", "아이들과 함께라면 교육적이고 즐거워요")
        - **가성비 및 만족도 (비용 대비 경험)**: (예: "입장료가 아깝지 않았어요", "무료인데도 볼거리가 풍성해서 만족도 높았어요")
        - **혼잡도 및 체감 대기 시간**: (예: "주말 오후에 방문했더니 사람이 너무 많아 제대로 보기 힘들었어요")
        - **추천 방문 시간대 및 요일**: (예: "야간 개장 때 조명이 켜지니 분위기가 훨씬 좋았어요")
        - **날씨별 방문 팁 (날씨가 경험에 미치는 영향)**: (예: "비 오는 날 방문하니 한옥의 운치가 더해져 좋았어요")
        - **어린이/가족 특화 정보**: (예: "유모차 끌고 다니기 편했어요", "수유실이 잘 되어있어요")
        - **편의시설 및 청결도**: (예: "화장실이 깨끗하고 관리가 잘 되어 있었어요")
        - **현장 직원 및 안내 친절도**: (예: "안내해주시는 분이 정말 친절하고 설명도 잘해주셔서 이해하기 쉬웠어요")
        - **기념품 및 굿즈 후기**: (예: "여기서만 살 수 있는 특별한 기념품이 많아요")
        - **놓치기 쉬운 숨은 팁 (아는 사람만 아는 꿀팁)**: (예: "입구 팸플릿 뒷면의 할인 쿠폰을 꼭 챙기세요")
        - **방문 전 기대치 vs 실제 경험**: (예: "SNS 사진만 보고 갔는데 생각보다 규모가 작아서 실망했어요")
        - **재방문 의사 및 추천 지수**: (예: "내년에도 꼭 다시 방문하고 싶어요")
        - **교육적 가치 및 정보성**: (예: "건축에 대해 몰랐던 새로운 사실을 많이 알게 되어 유익했어요")
        - **유사 행사/장소와의 비교**: (예: "작년 행사보다 프로그램이 더 다채로워졌어요")
        - **행사장 내 동선 및 이동 편의성**: (예: "전시관 사이의 거리가 멀지 않아 둘러보기 편했어요")
        - **현장 이벤트 및 체험 프로그램 상세 후기**: (예: "건축가와의 대화 프로그램이 가장 인상 깊었어요")
        - **행사장 분위기 (BGM, 조명 등)**: (예: "잔잔한 국악 배경음악이 한옥 분위기와 잘 어울렸어요")
        - **어르신/장애인 접근성**: (예: "계단이 많아서 거동이 불편하신 부모님이 힘들어하셨어요")
        - **외국인 방문객 시선**: (예: "영문 설명이 부족해서 아쉬웠어요")
        - **총평: 이 행사를 한 문장으로 요약한다면?**: (블로거가 내린 핵심적인 한 줄 평)

        공식 정보(행사 기간, 장소, 프로그램 목록, 가격 등)는 모두 제외하고, 오직 방문객들의 목소리가 담긴 내용만을 뽑아서 위의 형식에 맞춰 새롭게 정리해주세요.

        --- 1차 요약본 ---
        {initial_summary}
        ---

        --- 방문객 경험 중심 요약 (소주제별 분류) ---
        """

        try:
            answer_stream = self.llm.stream(filtering_prompt)
            full_filtered_summary = ""
            for chunk in answer_stream:
                full_filtered_summary += chunk.content
                yield full_filtered_summary

        except Exception as e:
            yield f"OpenAI API 2차 요약 중 오류가 발생했습니다: {e}"

    async def get_review_summary_and_tips(self, festival_name, num_reviews=5):
        if not self.llm:
            return "LLM이 초기화되지 않아 후기 요약/꿀팁 기능을 사용할 수 없습니다. OPENAI_API_KEY를 확인해주세요.", ""

        print(f"Searching Naver blogs for reviews of '{festival_name}'...")
        blog_results_meta = search_naver_blog(festival_name + " 축제 후기", display=num_reviews)
        
        if not blog_results_meta:
            return "Naver 후기를 찾을 수 없습니다.", ""

        # Scrape full content for each blog
        scraped_contents = []
        tasks = []
        for review_meta in blog_results_meta:
            link = review_meta.get("link")
            if link and "blog.naver.com" in link:
                tasks.append(scrape_blog_content(link))
            else:
                tasks.append(asyncio.sleep(0, result=("네이버 블로그가 아니므로 내용을 가져오지 않습니다.", [])))
        
        scraped_results = await asyncio.gather(*tasks)

        reviews_with_content = []
        for i, (content, _) in enumerate(scraped_results):
            if content and "본문 내용을 찾을 수 없습니다" not in content and "페이지에 접근하는 중 오류" not in content:
                reviews_with_content.append({"title": blog_results_meta[i].get("title", ""), "content": content})

        if not reviews_with_content:
            return "유효한 블로그 본문을 스크래핑할 수 없습니다.", ""

        # Summarize and extract tips using LLM
        # Since Gradio expects a single string for output, we'll run the stream to completion
        summary_generator = self.summarize_blog_contents_stream(reviews_with_content)
        full_summary = ""
        try:
            async for chunk in summary_generator:
                full_summary = chunk
        except TypeError: # Handle case where it's not an async generator (e.g., error string)
            full_summary = next(summary_generator) if isinstance(summary_generator, (list, tuple)) else summary_generator

        # For tips, we can extract them from the full_summary if the prompt is designed for it
        # Or, if a separate tip extraction prompt is needed, it would go here.
        # For now, the summary itself contains the categorized tips.
        tips = full_summary # The summary is already structured with tips
        
        return full_summary, tips
