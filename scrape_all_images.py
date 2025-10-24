import os
import pandas as pd
import requests
import re
import logging
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# --------------------------------------------------------------------------
# 환경 설정 (API 키 로드 등을 위해 가장 먼저 실행되어야 합니다)
# --------------------------------------------------------------------------
from src.infrastructure.config.settings import setup_environment

print("Setting up environment...")
setup_environment()

# LangSmith 로깅 비활성화 (대량 작업 시 불필요한 API 호출 방지)
# LANGCHAIN_ 관련 모든 환경 변수를 제거하여 로깅을 완전히 비활성화합니다.
langchain_vars = [key for key in os.environ if key.startswith("LANGCHAIN_")]
if langchain_vars:
    logging.info(f"Disabling LangSmith tracing for this script by unsetting: {langchain_vars}")
    for key in langchain_vars:
        del os.environ[key]

from src.infrastructure.llm_client import get_llm_client
from src.infrastructure.external_services.naver_search.naver_review_api import search_naver_blog

# --------------------------------------------------------------------------
# 상수 정의
# --------------------------------------------------------------------------
FESTIVAL_CSV_PATH = "database/축제공연행사csv.CSV"
OUTPUT_ROOT_DIR = "C:\\Users\\SBA\\github\\tour_agent\\imagesForAllFestivals"
BLOGS_TO_SCRAPE_PER_FESTIVAL = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------------
# 동기(Sync) 방식으로 재작성된 핵심 로직
# --------------------------------------------------------------------------

def scrape_blog_content_sync(url: str) -> tuple[str | None, list]:
    """Playwright의 동기 API를 사용하여 블로그 콘텐츠를 스크래핑합니다."""
    text_content = None
    image_urls = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)

            main_frame = page
            try:
                main_frame_element = page.wait_for_selector("iframe#mainFrame", timeout=5000)
                main_frame = main_frame_element.content_frame()
                if main_frame is None:
                    main_frame = page
            except PlaywrightTimeoutError:
                main_frame = page

            content_selectors = ["div.se-main-container", "div.post-view", "#postViewArea"]
            for selector in content_selectors:
                try:
                    # 콘텐츠가 로드될 때까지 기다리는 로직을 추가합니다.
                    main_frame.wait_for_selector(selector, timeout=3000) 
                    content_element = main_frame.query_selector(selector)
                    if content_element:
                        text_content = content_element.inner_text()
                        if text_content and text_content.strip():
                            images = content_element.query_selector_all("img")
                            for img in images:
                                src = img.get_attribute("data-lazy-src") or img.get_attribute("data-src") or img.get_attribute("src")
                                if src and src.startswith("http") and "storep-phinf.pstatic.net" not in src and "static.map" not in src:
                                    if src not in image_urls:
                                        image_urls.append(src)
                            break # 콘텐츠를 찾았으므로 루프 종료
                except PlaywrightTimeoutError:
                    continue # 현재 선택자가 없으면 다음으로 넘어감
            
            browser.close()
    except Exception as e:
        logging.error(f"Playwright error scraping {url}: {e}")
        return None, []
    
    return text_content, image_urls

def is_relevant_review_sync(llm, festival_name: str, blog_title: str, blog_content: str) -> bool:
    """LLM을 사용하여 블로그가 관련 후기인지 동기적으로 확인합니다."""
    prompt = f"""당신은 블로그 게시물의 주제를 정확하게 판별하는 전문가입니다.
사용자는 '{festival_name}' 축제에 대한 '진짜 후기'를 찾고 있습니다.
아래의 조건에 따라 주어진 블로그 제목과 본문이 검색 의도에 부합하는지 판별해주세요.

[판별 조건]
1.  **주제 일치:** 게시물의 '주된 내용'이 '{festival_name}' 축제에 대한 경험이나 후기여야 합니다.
2.  **유사 행사 제외:** '{festival_name}'와 이름이 비슷한 다른 행사에 대한 후기는 아니어야 합니다.
3.  **다른 주제 제외:** 주된 내용이 특정 장소(카페, 식당), 제품, 서비스 등에 대한 비교나 추천이 아니어야 합니다.
4.  **충분한 내용:** 내용이 너무 부실하여 실제 경험을 파악하기 어려운 경우, 관련성이 낮다고 판단합니다.

[판별할 정보]
-   **제목:** {blog_title}
-   **본문 (일부):** {blog_content[:2000]}...

[출력]
위 조건들을 모두 고려했을 때, 이 게시물이 사용자가 찾는 '{festival_name}' 축제에 대한 '진짜 후기'가 맞다면 '예'를, 그렇지 않다면 '아니오'를 반환해주세요. '예' 또는 '아니오'로만 대답해야 합니다."""
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        return "예" in answer
    except Exception as e:
        logging.error(f"LLM validation failed: {e}")
        return False

def scrape_and_download_images(llm, festival_name: str, output_dir: str):
    logging.info(f"Collecting image URLs for '{festival_name}'...")
    all_image_urls = []
    start_index = 1
    max_results_to_scan = 100
    display_count = 10
    consecutive_skips = 0
    found_blogs_with_images = 0

    while found_blogs_with_images < BLOGS_TO_SCRAPE_PER_FESTIVAL and start_index < max_results_to_scan:
        blog_reviews = search_naver_blog(f"{festival_name} 후기", display=display_count, start=start_index)
        if not blog_reviews:
            logging.warning(f"No more blog results for '{festival_name}' at start_index {start_index}.")
            break

        for review in blog_reviews:
            if found_blogs_with_images >= BLOGS_TO_SCRAPE_PER_FESTIVAL or consecutive_skips >= 5:
                break

            link = review.get("link")
            if link and "blog.naver.com" in link:
                text_content, image_urls = scrape_blog_content_sync(link)
                if text_content and image_urls:
                    if is_relevant_review_sync(llm, festival_name, review.get("title", ""), text_content):
                        all_image_urls.extend(image_urls)
                        found_blogs_with_images += 1
                        consecutive_skips = 0
                        logging.info(f"  [{found_blogs_with_images}/{BLOGS_TO_SCRAPE_PER_FESTIVAL}] Found {len(image_urls)} images from relevant blog: {link}")
                    else:
                        consecutive_skips += 1
                else:
                    consecutive_skips += 1
            else:
                consecutive_skips += 1
            time.sleep(0.5) # Short delay between blog scrapes
        
        if consecutive_skips >= 5:
            logging.warning(f"Skipped 5 consecutive blogs. Moving to next task for '{festival_name}'.")
            break
        
        start_index += display_count
        time.sleep(1)

    logging.info(f"Found a total of {len(all_image_urls)} unique image URLs for '{festival_name}'.")

    if not all_image_urls:
        logging.warning(f"No images to download for '{festival_name}'.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Downloading images to '{output_dir}'...")

    for i, img_url in enumerate(list(set(all_image_urls))): # Download unique URLs only
        try:
            response = requests.get(img_url, stream=True, timeout=15)
            response.raise_for_status()
            
            file_ext = os.path.splitext(img_url.split("?")[0])[-1] or ".jpg"
            file_name = "image_{:03d}{}".format(i + 1, file_ext)
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, "wb") as f:
                f.write(response.content)

        except requests.exceptions.RequestException as e:
            logging.error(f"  Failed to download {img_url}: {e}")
            continue
        time.sleep(0.1)

def main():
    logging.info("--- Starting Image Scraper for All Festivals (Sync Version) ---")
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    
    try:
        df = pd.read_csv(FESTIVAL_CSV_PATH, encoding='cp949')
        if 'title' not in df.columns:
            logging.error(f"'title' column not found in {FESTIVAL_CSV_PATH}")
            return
        festival_titles = df['title'].dropna().unique().tolist()
        logging.info(f"Found {len(festival_titles)} unique festivals to process.")
    except Exception as e:
        logging.error(f"Error reading festival CSV: {e}")
        return

    llm = get_llm_client()

    for i, title in enumerate(festival_titles):
        sanitized_title = re.sub(r'[\\/*?:\"<>|]', '_', title).strip()
        festival_output_dir = os.path.join(OUTPUT_ROOT_DIR, sanitized_title)
        
        logging.info(f"\n--------------------------------------------------\nProcessing festival {{i+1}}/{len(festival_titles)}: {title}\n--------------------------------------------------")
        
        if os.path.exists(festival_output_dir) and os.listdir(festival_output_dir):
            logging.warning(f"Output directory '{festival_output_dir}' already exists and is not empty. Skipping.")
            continue

        scrape_and_download_images(llm, title, festival_output_dir)
        
        logging.info("Waiting for 3 seconds before next festival...")
        time.sleep(3)

    logging.info("\n==============================\n Mission Complete \n==============================")

if __name__ == "__main__":
    main()