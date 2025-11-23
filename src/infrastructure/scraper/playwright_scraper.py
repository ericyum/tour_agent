"""
Playwright 스크래퍼 - 별도 프로세스에서 실행되는 독립 스크립트
Windows에서 Uvicorn 멀티워커와 호환되도록 subprocess로 호출됨
"""
import sys
import json
import argparse


def scrape_blog_content(url: str) -> dict:
    """
    Playwright를 사용하여 네이버 블로그 본문과 이미지 URL을 스크래핑합니다.
    """
    from playwright.sync_api import sync_playwright

    text_content = ""
    image_urls = []
    error = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)

            main_frame = page
            try:
                main_frame_element = page.wait_for_selector(
                    "iframe#mainFrame", timeout=5000
                )
                main_frame = main_frame_element.content_frame()
                if main_frame is None:
                    main_frame = page
            except Exception:
                main_frame = page

            content_selectors = [
                "div.se-main-container",
                "div.post-view",
                "#postViewArea",
            ]

            for selector in content_selectors:
                try:
                    main_frame.wait_for_selector(selector, timeout=5000)
                    content_element = main_frame.query_selector(selector)
                    if content_element:
                        text_content = content_element.inner_text()
                        if text_content.strip():
                            # 이미지 찾기
                            images = content_element.query_selector_all("img")
                            for img in images:
                                lazy_src = img.get_attribute("data-lazy-src")
                                data_src = img.get_attribute("data-src")
                                regular_src = img.get_attribute("src")

                                src = lazy_src or data_src or regular_src

                                if src and src.startswith("http"):
                                    # Filter out emoticons/stickers
                                    if "storep-phinf.pstatic.net" in src and "ogq_" in src:
                                        continue
                                    # Filter out map images
                                    if "simg.pstatic.net" in src and "static.map" in src:
                                        continue

                                    if src not in image_urls:
                                        image_urls.append(src)
                            break
                except Exception:
                    continue

            browser.close()

            if not text_content.strip():
                text_content = "본문 내용을 찾을 수 없습니다. (지원되지 않는 블로그 구조일 수 있습니다)"

    except Exception as e:
        error = str(e)
        text_content = f"페이지에 접근하는 중 오류가 발생했습니다: {e}"

    return {
        "text_content": text_content,
        "image_urls": image_urls,
        "error": error
    }


def main():
    # Windows cp949 인코딩 문제 해결 - UTF-8로 강제 설정
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Playwright Blog Scraper")
    parser.add_argument("url", help="URL to scrape")
    args = parser.parse_args()

    result = scrape_blog_content(args.url)

    # JSON으로 결과 출력 (stdout)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
