import os
import requests
import re
from dotenv import load_dotenv
from src.infrastructure.external_services.naver_search.naver_review_api import search_naver_blog
from playwright.async_api import async_playwright # Original scraper used playwright
from src.infrastructure.llm_client import get_llm_client # Added LLM client import

load_dotenv()

class NaverReviewSupervisor:
    def __init__(self):
        self.llm = get_llm_client() # Initialize LLM client

    async def get_review_summary_and_tips(self, festival_name, num_reviews=5, return_full_text=False, return_meta=False):
        search_query = f"{festival_name} 후기"
        print(f"Searching Naver blogs for reviews of '{search_query}'...")

        reviews_with_content = []
        consecutive_skips = 0
        start_index = 1
        max_results_to_scan = 100  # Scan up to 100 results
        display_count = 20 # Fetch 20 at a time

        while len(reviews_with_content) < num_reviews and start_index < max_results_to_scan:
            blog_results_meta = search_naver_blog(search_query, display=display_count, start=start_index)
            
            if not blog_results_meta:
                break

            for review_meta in blog_results_meta:
                if len(reviews_with_content) >= num_reviews:
                    break
                
                if consecutive_skips >= 3:
                    print(f"DEBUG: Skipped 3 consecutive blogs. Proceeding with {len(reviews_with_content)} reviews.")
                    break

                link = review_meta.get("link")
                if link and "blog.naver.com" in link:
                    text_content, _ = await self._scrape_blog_content(link)
                    if text_content and "본문 내용을 찾을 수 없습니다" not in text_content and "페이지에 접근하는 중 오류" not in text_content:
                        is_relevant = await self._is_relevant_review(festival_name, review_meta.get("title", ""), text_content)
                        if is_relevant:
                            print(f"DEBUG: Scraped content for '{review_meta.get('title')}': {text_content[:200]}...")
                            reviews_with_content.append({"title": review_meta.get("title", ""), "content": text_content, "link": review_meta.get("link", "")})
                            consecutive_skips = 0
                        else:
                            print(f"DEBUG: Blog '{review_meta.get('title')}' deemed irrelevant by LLM validator. Skipping.")
                            consecutive_skips += 1
                    else:
                        print(f"DEBUG: Failed to scrape content for '{review_meta.get('title')}' or content was empty/error. Skipping.")
                        consecutive_skips += 1
                else:
                    consecutive_skips += 1
            
            if consecutive_skips >= 3:
                break

            start_index += display_count

        if not reviews_with_content:
            return "유효한 블로그 본문을 스크래핑할 수 없습니다.", []

        if return_full_text:
            if return_meta:
                return "", reviews_with_content
            else:
                full_texts = [review['content'] for review in reviews_with_content]
                return "", full_texts

        llm_generated_summary, _ = await self._llm_summarize_reviews(festival_name, reviews_with_content)
        return llm_generated_summary, llm_generated_summary

    async def _scrape_blog_content(self, url: str) -> tuple[str, list[str]]:
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
                                # 이미지 찾기
                                images = await content_element.query_selector_all("img")
                                for img in images:
                                    # 네이버 블로그는 lazy loading을 사용하므로 data-lazy-src, data-src, src 순으로 확인
                                    lazy_src = await img.get_attribute("data-lazy-src")
                                    data_src = await img.get_attribute("data-src")
                                    regular_src = await img.get_attribute("src")

                                    src = lazy_src or data_src or regular_src

                                    if (
                                        src
                                        and src.startswith("http")
                                    ):
                                        # Filter out emoticons/stickers
                                        if "storep-phinf.pstatic.net" in src and "ogq_" in src:
                                            continue
                                        # Filter out map images
                                        if "simg.pstatic.net" in src and "static.map" in src:
                                            continue

                                        # 썸네일 파라미터(?type=...)를 포함하여 원본 이미지 URL 확보
                                        cleaned_src = src
                                        if cleaned_src not in image_urls:
                                            image_urls.append(cleaned_src)
                                break  # 내용과 이미지를 찾았으면 중단
                    except Exception:
                        continue

                await browser.close()

                if not text_content.strip():
                    return (
                        "본문 내용을 찾을 수 없습니다. (지원되지 않는 블로그 구조일 수 있습니다)",
                        [],
                    )

                return text_content, image_urls

        except Exception as e:
            return f"페이지에 접근하는 중 오류가 발생했습니다: {e}", []

    async def _is_relevant_review(self, festival_name: str, blog_title: str, blog_content: str) -> bool:
        prompt = f"""당신은 블로그 게시물의 주제를 정확하게 판별하는 전문가입니다.
사용자는 '{festival_name}' 축제에 대한 '진짜 후기'를 찾고 있습니다.
아래의 조건에 따라 주어진 블로그 제목과 본문이 검색 의도에 부합하는지 판별해주세요.

[판별 조건]
1.  **주제 일치:** 게시물의 '주된 내용'이 '{festival_name}' 축제에 대한 경험이나 후기여야 합니다. 단순히 언급만 되거나 부수적인 내용이면 안 됩니다.
2.  **유사 행사 제외:** '{festival_name}'와 이름이 비슷한 다른 행사(예: '세계 {festival_name}')에 대한 후기는 아닌지 확인해야 합니다.
3.  **다른 주제 제외:** 게시물의 주된 내용이 '{festival_name}' 축제가 아닌, 특정 장소(카페, 식당), 제품, 서비스 등에 대한 비교나 추천이 아닌지 확인해야 합니다. (예: '{festival_name} 기념 카페 A, B 비교 후기')
4.  **충분한 내용:** 게시물 본문이 너무 짧거나 내용이 부실하여 실제 경험을 파악하기 어려운 경우, 관련성이 낮다고 판단합니다.

[판별할 정보]
-   **제목:** {blog_title}
-   **본문 (일부):** {blog_content[:2000]}...

[출력]
위 조건들을 모두 고려했을 때, 이 게시물이 사용자가 찾는 '{festival_name}' 축제에 대한 '진짜 후기'가 맞다면 '예'를, 그렇지 않다면 '아니오'를 반환해주세요. '예' 또는 '아니오'로만 대답해야 합니다."""
        try:
            response = await self.llm.ainvoke(prompt)
            answer = response.content.strip()
            if "예" in answer:
                print(f"DEBUG: Validation successful for '{blog_title}'. It's a relevant review.")
                return True
            else:
                print(f"DEBUG: Validation failed for '{blog_title}'. Not a relevant review.")
                return False
        except Exception as e:
            print(f"DEBUG: LLM validation failed for '{blog_title}': {e}")
            return False # Assume not relevant if validation fails

    async def get_sentiment_for_text(self, text: str):
        # This function was not part of the original NaverReviewSupervisor
        # It will be implemented by the new LLM-driven sentiment analysis
        return []

    async def _llm_summarize_reviews(self, festival_name: str, reviews_with_content: list) -> tuple[str, str]:
        import traceback # Added import for traceback
        combined_content = "\n\n--- 다음 블로그 ---\n\n".join([f"제목: {r['title']}\n내용: {r['content']}" for r in reviews_with_content])
        
        print(f"DEBUG: Combined content sent to LLM (first 500 chars):\n{combined_content[:500]}...") # Print first 500 chars
        
        
        prompt = f"""당신은 축제 리뷰를 분석하여 핵심 정보를 추출하는 전문가입니다.
아래는 '{festival_name}' 축제에 대한 여러 블로그 리뷰 내용입니다.
이 리뷰들을 종합하여 다음 소주제별로 정보를 분류하고 요약해주세요.
각 소주제는 반드시 제공된 제목과 괄호 안의 설명을 포함하여 작성해야 합니다.
내용이 없는 소주제는 "정보 없음"이라고 명시해주세요.

--- 방문객 경험 중심 요약 (소주제별 분류) ---
추천 방문 대상 (누구와 함께 가면 좋을까?)
[여기에 내용 요약]

가성비 및 만족도 (비용 대비 경험)
[여기에 내용 요약]

혼잡도 및 체감 대기 시간
[여기에 내용 요약]

추천 방문 시간대 및 요일
[여기에 내용 요약]

날씨별 방문 팁 (날씨가 경험에 미치는 영향)
[여기에 내용 요약]

어린이/가족 특화 정보
[여기에 내용 요약]

편의시설 및 청결도
[여기에 내용 요약]

현장 직원 및 안내 친절도
[여기에 내용 요약]

기념품 및 굿즈 후기
[여기에 내용 요약]

놓치기 쉬운 숨은 팁 (아는 사람만 아는 꿀팁)
[여기에 내용 요약]

방문 전 기대치 vs 실제 경험
[여기에 내용 요약]

재방문 의사 및 추천 지수
[여기에 내용 요약]

교육적 가치 및 정보성
[여기에 내용 요약]

유사 행사/장소와의 비교
[여기에 내용 요약]

행사장 내 동선 및 이동 편의성
[여기에 내용 요약]

현장 이벤트 및 체험 프로그램 상세 후기
[여기에 내용 요약]

행사장 분위기 (BGM, 조명 등)
[여기에 내용 요약]

어르신/장애인 접근성
[여기에 내용 요약]

외국인 방문객 시선
[여기에 내용 요약]

총평: 이 행사를 한 문장으로 요약한다면?
[여기에 총평 요약]

[블로그 리뷰 내용]
{combined_content}

[출력 형식]
위에서 제시된 소주제별 분류에 따라 내용을 채워서 출력해주세요.
각 소주제 제목은 그대로 유지하고, 그 아래에 요약된 내용을 작성해주세요.
내용이 없는 소주제는 "정보 없음"이라고 명시해주세요.
"""
        try:
            response = await self.llm.ainvoke(prompt)
            raw_summary = response.content.strip()

            print(f"DEBUG: Raw summary from LLM:\n{raw_summary}\n--- END RAW SUMMARY ---") # Print raw LLM output

            sections_data = {}
            
            # Define the expected headings for parsing, including markdown
            headings_with_markdown = [
                "**추천 방문 대상 (누구와 함께 가면 좋을까?)**",
                "**가성비 및 만족도 (비용 대비 경험)**",
                "**혼잡도 및 체감 대기 시간**",
                "**추천 방문 시간대 및 요일**",
                "**날씨별 방문 팁 (날씨가 경험에 미치는 영향)**",
                "**어린이/가족 특화 정보**",
                "**편의시설 및 청결도**",
                "**현장 직원 및 안내 친절도**",
                "**기념품 및 굿즈 후기**",
                "**놓치기 쉬운 숨은 팁 (아는 사람만 아는 꿀팁)**",
                "**방문 전 기대치 vs 실제 경험**",
                "**재방문 의사 및 추천 지수**",
                "**교육적 가치 및 정보성**",
                "**유사 행사/장소와의 비교**",
                "**행사장 내 동선 및 이동 편의성**",
                "**현장 이벤트 및 체험 프로그램 상세 후기**",
                "**행사장 분위기 (BGM, 조명 등)**",
                "**어르신/장애인 접근성**",
                "**외국인 방문객 시선**",
                "**총평: 이 행사를 한 문장으로 요약한다면?**"
            ]
            
            # Add the main header for initial split
            full_text_to_parse = raw_summary.replace("--- 방문객 경험 중심 요약 (소주제별 분류) ---", "").strip()

            # Iterate through headings to extract content
            for i, heading in enumerate(headings_with_markdown):
                start_idx = full_text_to_parse.find(heading)
                if start_idx == -1:
                    sections_data[heading] = "정보 없음"
                    continue
                
                content_start_idx = start_idx + len(heading)
                
                # Find the next heading to determine the end of the current section's content
                next_heading_idx = -1
                for j in range(i + 1, len(headings_with_markdown)):
                    temp_idx = full_text_to_parse.find(headings_with_markdown[j], content_start_idx)
                    if temp_idx != -1:
                        next_heading_idx = temp_idx
                        break
                
                if next_heading_idx != -1:
                    content = full_text_to_parse[content_start_idx:next_heading_idx].strip()
                else:
                    content = full_text_to_parse[content_start_idx:].strip()
                
                # Remove "[여기에 내용 요약]" placeholder if LLM didn't fill it
                if content == "[여기에 내용 요약]":
                    content = "정보 없음"
                
                sections_data[heading] = content.strip()
            
            final_summary_parts = []
            for heading in headings_with_markdown:
                content = sections_data.get(heading, "정보 없음").strip()
                
                # If content is "정보 없음", skip this section unless it's the overall summary
                if content == "정보 없음" and not heading.startswith("**총평"):
                    continue

                # Remove the markdown from the heading for display in the final summary construction
                display_heading = heading.replace("**", "")
                
                if display_heading.startswith("총평"):
                    final_summary_parts.append(f"**{display_heading}**\n{content}\n")
                else:
                    final_summary_parts.append(f"**{display_heading}**\n{content}\n\n")
            
            final_summary = f"**{festival_name} 축제 방문객 경험 중심 요약**\n\n" + "".join(final_summary_parts)
            
            return final_summary, raw_summary # Return the formatted summary and the raw summary for potential debugging
        except Exception as e:
            print(f"LLM 요약 중 오류 발생: {e}")
            traceback.print_exc() # Print traceback for debugging
            return "LLM 요약 생성 중 오류가 발생했습니다.", "LLM 요약 생성 중 오류가 발생했습니다."