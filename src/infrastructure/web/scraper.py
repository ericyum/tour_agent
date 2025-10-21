from playwright.async_api import async_playwright

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
