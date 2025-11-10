import gradio as gr
import functools

# Import data loaders and constants
from src.infrastructure.config.loader import ALL_FESTIVAL_CATEGORIES
from src.application.core.constants import (
    AREA_CODE_MAP,
    SIGUNGU_CODE_MAP,
    PAGE_SIZE,
)

# Import handlers and callbacks
from src.presentation import event_handlers
from src.presentation import callbacks


def create_ui():
    """Creates and returns the Gradio UI Blocks."""
    with gr.Blocks(
        css="""#gallery .thumbnail-item { max-width: 250px !important; min-width: 200px !important; flex-grow: 1 !important; }
               #detail_tabs .tabs-container { min-height: 500px; }
               #detail_tabs .tab-nav button { font-size: 1.1em; padding: 10px; }
            """
    ) as demo:
        # --- [1] State Variables (전역 상태) ---
        # Page Navigation
        # current_page_state = gr.State("home") # [수정] 탭의 selected 속성으로 대체

        # Search & Results
        results_state = gr.State([])
        total_pages_state = gr.State(1)

        # Details
        selected_festival_state = gr.State()
        selected_festival_details_state = gr.State()
        temp_selection_state = gr.State()  # '내 코스' 추가용 임시 상태

        # Recommendations
        recommended_facilities_state = gr.State([])
        recommended_courses_state = gr.State([])
        recommended_festivals_state = gr.State([])
        recommended_total_pages_state = gr.State(1)

        # My Course
        my_course_state = gr.State([])

        # Sentiment Analysis
        blog_results_df_state = gr.State()
        blog_judgments_state = gr.State()

        # --- [2] UI Definition (페이지 구조) ---
        gr.Markdown("#  Festival Moment 🎪")

        # --- (A) Top Navigation Bar (상단 네비게이션 바) ---
        with gr.Tabs(selected="home") as top_level_tabs:
            with gr.TabItem("🏠 홈", id="home"):
                gr.Markdown(
                    """
                    ## 🌟 Festival Moment에 오신 것을 환영합니다!

                    Festival Moment는 당신의 완벽한 축제 및 여행 경험을 위한 AI 기반 파트너입니다.

                    - **🔍 검색:** 전국의 모든 축제 정보를 원하는 조건으로 검색하세요.
                    - **🏆 순위:** AI가 검색된 축제들의 화제성과 만족도를 분석하여 똑똑한 순위를 매겨줍니다.
                    - **✨ 상세 분석:** 축제를 클릭하여 트렌드, 방문객 후기(감성 분석, 워드 클라우드), 주변 추천까지 한눈에 확인하세요.
                    - **🗺️ 내 코스:** 마음에 드는 축제와 장소들을 '내 코스'에 담고, AI에게 여행 일정이 현실적인지 검증받으세요!

                    ---
                    **시작하려면 '검색' 탭을 눌러주세요.**
                    """
                )

            with gr.TabItem("🔍 검색", id="search"):
                gr.Markdown("## 🔍 축제 검색")
                # --- (기존 검색 필터 UI) ---
                with gr.Group():
                    with gr.Row():
                        area_dropdown = gr.Dropdown(
                            label="시/도",
                            choices=["전체"] + sorted(list(AREA_CODE_MAP.keys())),
                            value="전체",
                            interactive=True,
                        )
                        sigungu_dropdown = gr.Dropdown(
                            label="시/군/구",
                            choices=["전체"],
                            value="전체",
                            interactive=True,
                        )
                    with gr.Row():
                        main_cat_dropdown = gr.Dropdown(
                            label="대분류",
                            choices=["전체"]
                            + sorted(list(ALL_FESTIVAL_CATEGORIES.keys())),
                            value="전체",
                            interactive=True,
                        )
                        medium_cat_dropdown = gr.Dropdown(
                            label="중분류",
                            choices=["전체"],
                            value="전체",
                            interactive=True,
                        )
                        small_cat_dropdown = gr.Dropdown(
                            label="소분류",
                            choices=["전체"],
                            value="전체",
                            interactive=True,
                        )
                        status_radio = gr.Radio(
                            label="진행 상태",
                            choices=["전체", "축제 진행중", "진행 예정", "종료된 축제"],
                            value="전체",
                            interactive=True,
                        )
                with gr.Row():
                    search_btn = gr.Button("검색", variant="primary", scale=1)

                # --- (검색 결과 UI - 기존 results_area) ---
                with gr.Column(visible=False) as results_area:
                    gr.Markdown("--- \n ## 📊 검색 결과")
                    # --- (Q1 답변: 순위 매기기 기능) ---
                    with gr.Accordion("🏆 AI 순위 매기기", open=False):
                        with gr.Tabs():
                            with gr.TabItem("전체 순위 매기기"):
                                with gr.Row():
                                    rank_festivals_btn = gr.Button(
                                        "검색 결과 전체 순위 보기", scale=2
                                    )
                                    num_reviews_festival_ranking = gr.Slider(
                                        minimum=1,
                                        maximum=50,
                                        value=10,
                                        step=1,
                                        label="축제 순위용 리뷰 수",
                                        interactive=True,
                                        scale=2,
                                    )
                                    festival_ranking_top_n_slider = gr.Slider(
                                        minimum=1,
                                        maximum=5,
                                        value=3,
                                        step=1,
                                        label="표시할 순위 수",
                                        interactive=True,
                                        scale=1,
                                    )
                                festival_ranking_report = gr.Markdown(visible=False)

                            with gr.TabItem("선택하여 순위 매기기 (Q1)"):
                                selected_festivals_checkbox = gr.CheckboxGroup(
                                    label="순위 매길 축제 선택 (현재 페이지만 표시됩니다)",
                                    choices=[],
                                )
                                with gr.Row():
                                    rank_selected_btn = gr.Button(
                                        "선택 항목 순위 매기기", scale=2
                                    )
                                    num_reviews_festival_ranking_sel = gr.Slider(
                                        minimum=1,
                                        maximum=50,
                                        value=10,
                                        step=1,
                                        label="축제 순위용 리뷰 수",
                                        interactive=True,
                                        scale=2,
                                    )
                                    festival_ranking_top_n_slider_sel = gr.Slider(
                                        minimum=1,
                                        maximum=5,
                                        value=3,
                                        step=1,
                                        label="표시할 순위 수",
                                        interactive=True,
                                        scale=1,
                                    )
                                festival_ranking_report_sel = gr.Markdown(visible=False)

                    gr.Markdown("### 🖼️ 축제 목록 (클릭하여 상세 정보 보기)")
                    festival_gallery = gr.Gallery(
                        label="축제 목록",
                        show_label=False,
                        elem_id="gallery",
                        columns=4,
                        height="auto",
                        object_fit="contain",
                    )
                    with gr.Row(variant="panel"):
                        first_page_button = gr.Button("<<", size="sm")
                        prev_button = gr.Button("◀ 이전", size="sm")
                        page_buttons = [
                            gr.Button(str(i), visible=False, size="sm")
                            for i in range(1, 6)
                        ]
                        next_button = gr.Button("다음 ▶", size="sm")
                        last_page_button = gr.Button(">>")
                    with gr.Row():
                        page_input = gr.Number(
                            label="페이지 이동", value=1, interactive=True, scale=1
                        )
                        page_display = gr.Textbox(
                            value="1 / 1",
                            label="현재 페이지",
                            interactive=False,
                            container=False,
                            scale=1,
                        )

            with gr.TabItem("✨ 상세 정보", id="details"):
                # [*** 수정됨 ***] Q2: 뒤로가기 버튼
                back_to_search_btn = gr.Button("◀ 검색 목록으로 돌아가기")

                gr.Markdown("## ✨ 축제 상세 정보")
                festival_details_output = gr.Markdown()
                precautions_output = gr.Markdown(
                    label="AI 기반 에티켓 가이드", visible=False
                )
                add_to_my_course_btn_details = gr.Button(
                    "🗺️ 이 축제를 내 코스에 추가", variant="primary"
                )

                gr.Markdown("--- \n ## 💡 AI 추가 분석 (원하는 탭 클릭)")

                with gr.Tabs(elem_id="detail_tabs") as detail_tabs_container:

                    # --- Detail Tab 1: Image Gallery ---
                    with gr.TabItem(label="📸 이미지", id="image") as image_tab:
                        with gr.Row():
                            num_blogs_for_images = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=5,
                                step=1,
                                label="이미지 수집 대상 블로그 수",
                                interactive=True,
                            )
                            image_collect_button = gr.Button("이미지 수집하기")
                        image_gallery = gr.Gallery(
                            label="수집된 이미지",
                            show_label=False,
                            columns=4,
                            height="auto",
                            object_fit="contain",
                        )
                        scraped_urls_output = gr.Textbox(
                            label="Scraped Image URLs", visible=False
                        )

                    # --- Detail Tab 2: Trend Graph ---
                    with gr.TabItem(label="📈 트렌드", id="trend") as trend_tab:
                        trend_graph_btn = gr.Button(
                            "트렌드 그래프 생성", variant="primary"
                        )
                        trend_status = gr.Textbox(label="상태", interactive=False)
                        with gr.Row():
                            trend_plot_yearly = gr.Image(label="최근 1년 검색량 트렌드")
                            trend_plot_event = gr.Image(label="축제 기간 중심 트렌드")

                    # --- Detail Tab 3: Word Cloud ---
                    with gr.TabItem(
                        label="☁️ 워드클라우드", id="wordcloud"
                    ) as wordcloud_tab:
                        with gr.Row():
                            num_reviews_wordcloud = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=20,
                                step=1,
                                label="분석할 후기 수 (워드클라우드)",
                                interactive=True,
                            )
                            word_cloud_btn = gr.Button(
                                "워드 클라우드 생성", variant="primary"
                            )
                        wordcloud_status = gr.Textbox(label="상태", interactive=False)
                        wordcloud_plot = gr.Image(label="축제의 주요 핵심 요소들")

                    # --- Detail Tab 4: Sentiment Analysis ---
                    with gr.TabItem(
                        label="❤️ 감성 분석", id="sentiment"
                    ) as sentiment_tab:
                        with gr.Row():
                            num_reviews_slider = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=10,
                                step=1,
                                label="분석할 후기 수",
                                interactive=True,
                            )
                            run_sentiment_btn = gr.Button(
                                "감성 분석 실행", variant="primary"
                            )
                        sentiment_status = gr.Textbox(
                            label="분석 상태", interactive=False
                        )
                        # (기존 감성 분석 Accordion 내용들)
                        with gr.Accordion("종합 분석 결과", open=True):
                            sentiment_summary = gr.Markdown(
                                label="종합 분석 상세", visible=False
                            )
                            sentiment_overall_csv = gr.File(
                                label="종합 분석 (CSV) 다운로드", visible=False
                            )
                            sentiment_positive_keywords = gr.HTML(visible=False)
                            with gr.Row():
                                sentiment_overall_chart = gr.Plot(
                                    label="전체 후기 요약 (긍/부정)", visible=False, scale=1
                                )
                            with gr.Row():
                                sentiment_distribution_chart = gr.Image(
                                    label="상대적 만족도 분포 (막대)", visible=False, scale=1
                                )
                                sentiment_absolute_chart = gr.Image(
                                    label="절대 점수 분포 (꺾은선)", visible=False, scale=1
                                )
                            sentiment_distribution_description = gr.Markdown(
                                visible=False
                            )
                            with gr.Row():
                                outlier_chart = gr.Image(
                                    label="이상치 탐지 결과", visible=False, scale=1
                                )
                            outlier_description = gr.Markdown(visible=False)
                            sentiment_negative_summary = gr.Markdown(
                                label="주요 불만 사항 요약", visible=False
                            )
                        gr.Markdown("### 개별 블로그 분석 결과")
                        sentiment_df_output = gr.DataFrame(
                            headers=[
                                "블로그 제목",
                                "링크",
                                "감성 빈도",
                                "감성 점수",
                                "긍정 문장 수",
                                "부정 문장 수",
                                "긍정 비율 (%)",
                                "부정 비율 (%)",
                                "긍/부정 문장 요약",
                            ],
                            datatype=[
                                "str",
                                "str",
                                "number",
                                "str",
                                "number",
                                "number",
                                "str",
                                "str",
                                "str",
                            ],
                            label="개별 블로그 분석 결과",
                            wrap=True,
                            interactive=True,
                        )
                        with gr.Row():
                            sentiment_blog_page_num_input = gr.Number(
                                value=1, label="페이지 번호", interactive=True, scale=1
                            )
                            sentiment_blog_total_pages_output = gr.Textbox(
                                value="/ 1",
                                label="전체 페이지",
                                interactive=False,
                                container=False,
                                scale=1,
                            )
                            sentiment_blog_list_csv = gr.File(
                                label="전체 블로그 목록(CSV) 다운로드",
                                visible=False,
                                scale=2,
                            )
                        with gr.Accordion(
                            "개별 블로그 상세 분석 (표에서 행 선택)",
                            open=False,
                            visible=False,
                        ) as sentiment_blog_detail_accordion:
                            sentiment_individual_summary = gr.Markdown(
                                label="긍/부정 문장 요약", visible=False
                            )
                            with gr.Row():
                                sentiment_individual_donut_chart = gr.Plot(
                                    label="개별 블로그 긍/부정 비율", visible=False
                                )
                                sentiment_individual_score_chart = gr.Plot(
                                    label="문장별 감성 점수", visible=False
                                )

                    # --- Detail Tab 5: Naver Review ---
                    with gr.TabItem(
                        label="📝 후기 요약", id="naver_review"
                    ) as naver_review_tab:
                        with gr.Row():
                            num_reviews_naver_summary = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=5,
                                step=1,
                                label="분석할 후기 수 (네이버 요약)",
                                interactive=True,
                            )
                            naver_search_btn = gr.Button(
                                "네이버 후기 요약 검색", variant="primary"
                            )
                        naver_review_output = gr.Markdown()

                    # --- Detail Tab 6: Recommendation ---
                    with gr.TabItem(
                        label="📍 주변 추천", id="recommend"
                    ) as recommend_tab:
                        with gr.Row():
                            recommend_radius_slider = gr.Slider(
                                minimum=100,
                                maximum=20000,
                                value=5000,
                                step=100,
                                label="반경 (미터)",
                                interactive=True,
                            )
                            recommend_btn = gr.Button("추천 받기", variant="primary")

                        with gr.Row(visible=False) as ranking_controls:
                            ranking_reviews_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="순위용 리뷰 수",
                                interactive=True,
                            )
                            ranking_top_n_slider = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=3,
                                step=1,
                                label="표시할 순위 수",
                                interactive=True,
                            )
                            rank_facilities_btn = gr.Button("관광 시설 순위 매기기")
                            rank_courses_btn = gr.Button("관광 코스 순위 매기기")
                            rank_festivals_rec_btn = gr.Button("추천 축제 순위 매기기")

                        recommend_status = gr.Textbox(
                            label="상태", interactive=False, visible=False
                        )

                        gr.Markdown("### 추천 관광 시설")
                        recommend_facilities_gallery = gr.Gallery(
                            label="추천 관광 시설",
                            show_label=False,
                            elem_id="recommend_facilities_gallery",
                            columns=4,
                            height="auto",
                            object_fit="contain",
                        )
                        facility_ranking_report = gr.Markdown(visible=False)

                        gr.Markdown("### 추천 관광 코스")
                        recommend_courses_gallery = gr.Gallery(
                            label="추천 관광 코스",
                            show_label=False,
                            elem_id="recommend_courses_gallery",
                            columns=4,
                            height="auto",
                            object_fit="contain",
                        )
                        course_ranking_report = gr.Markdown(visible=False)

                        gr.Markdown("### 추천 축제")
                        recommend_festivals_gallery = gr.Gallery(
                            label="추천 축제",
                            show_label=False,
                            elem_id="recommend_festivals_gallery",
                            columns=4,
                            height="auto",
                            object_fit="contain",
                        )
                        with gr.Row(variant="panel"):
                            rec_first_page_button = gr.Button("<<", size="sm")
                            rec_prev_button = gr.Button("◀ 이전", size="sm")
                            rec_page_buttons = [
                                gr.Button(str(i), visible=False, size="sm")
                                for i in range(1, 6)
                            ]
                            rec_next_button = gr.Button("다음 ▶", size="sm")
                            rec_last_page_button = gr.Button(">>")
                        with gr.Row():
                            rec_page_input = gr.Number(
                                label="페이지 이동", value=1, interactive=True, scale=1
                            )
                            rec_page_display = gr.Textbox(
                                value="1 / 1",
                                label="현재 페이지",
                                interactive=False,
                                container=False,
                                scale=1,
                            )
                        festival_ranking_rec_report = gr.Markdown(visible=False)

                        with gr.Accordion(
                            "추천 장소 상세 정보 (항목 선택 시 표시)",
                            open=False,
                            visible=False,
                        ) as recommend_details_accordion:
                            recommend_details_output = gr.Markdown()
                            # '내 코스에 넣기' 버튼 추가 (요청 사항)
                            add_to_my_course_btn_reco = gr.Button(
                                "🗺️ 선택한 추천 장소를 내 코스에 추가"
                            )

                    # --- [ 신규 ] 렌더링 탭 ---
                    with gr.TabItem(
                        label="🎨 AI 렌더링", id="rendering"
                    ) as rendering_tab:
                        gr.Markdown("### 🤖 AI 기반 축제 렌더링")
                        gr.Markdown(
                            "AI가 축제 현장의 위성사진을 바탕으로 대표 장면과 다양한 테마의 장면을 렌더링합니다. "
                            "**실행 시 1~2분 정도 소요될 수 있습니다.** (Gemini API 호출)"
                        )
                        rendering_btn = gr.Button("AI 렌더링 실행", variant="primary")
                        rendering_status = gr.Textbox(
                            label="상태", interactive=False, visible=False
                        )

                        with gr.Accordion("1. 대표 렌더링 (대표 계절/시간)", open=True):
                            representative_rendering_output = gr.Image(
                                label="대표 렌더링", type="filepath"
                            )

                        with gr.Accordion("2. 조건별 렌더링 (테마별 장면)", open=True):
                            conditional_rendering_output = gr.Gallery(
                                label="조건별 렌더링",
                                columns=2,
                                height="auto",
                                object_fit="contain",
                            )
                    # --- [ 신규 ] 탭 추가 끝 ---

            with gr.TabItem("🗺️ 내 코스", id="my_course"):
                gr.Markdown("## 🗺️ 나만의 코스")
                my_course_output = gr.Markdown("나만의 코스에 항목을 추가해보세요.")
                with gr.Row():
                    clear_my_course_btn = gr.Button("나만의 코스 비우기")

                gr.Markdown("--- \n ### 📅 코스 현실성 검증")
                with gr.Row():
                    trip_duration_input = gr.Textbox(
                        label="총 여행 기간을 입력하세요 (예: 2박 3일, 당일치기)",
                        placeholder="예: 1박 2일",
                    )
                    validate_course_btn = gr.Button("검증하기", variant="primary")
                course_validation_output = gr.Markdown()

        # --- [3] Event Handlers Binding ---

        # --- (A) Navigation Handlers (페이지 이동) ---
        # [*** 수정됨 ***] '뒤로가기' 버튼 클릭 시 'search' 탭으로 이동
        back_to_search_btn.click(
            fn=lambda: gr.update(
                selected="search"
            ),  # 'search' 탭 ID를 선택하도록 gr.update 객체 반환
            inputs=None,
            outputs=[top_level_tabs],  # top_level_tabs의 selected 값을 업데이트
        )

        # --- (B-2) Search Page Handlers ---
        # (드롭다운 핸들러 - 수정 없음)
        update_sigungu_with_map = functools.partial(
            callbacks.update_sigungu, sigungu_code_map=SIGUNGU_CODE_MAP
        )
        area_dropdown.change(
            fn=update_sigungu_with_map, inputs=area_dropdown, outputs=sigungu_dropdown
        )
        update_medium_cat_with_map = functools.partial(
            callbacks.update_medium_cat,
            all_festival_categories=ALL_FESTIVAL_CATEGORIES,
        )
        main_cat_dropdown.change(
            fn=update_medium_cat_with_map,
            inputs=main_cat_dropdown,
            outputs=medium_cat_dropdown,
        )
        update_small_cat_with_map = functools.partial(
            callbacks.update_small_cat,
            all_festival_categories=ALL_FESTIVAL_CATEGORIES,
        )
        medium_cat_dropdown.change(
            fn=update_small_cat_with_map,
            inputs=[main_cat_dropdown, medium_cat_dropdown],
            outputs=small_cat_dropdown,
        )

        # 검색 버튼
        search_btn.click(
            fn=event_handlers.run_search_and_populate_ranking_cbox,
            inputs=[
                area_dropdown,
                sigungu_dropdown,
                main_cat_dropdown,
                medium_cat_dropdown,
                small_cat_dropdown,
                status_radio,
            ],
            outputs=[
                results_state,
                festival_gallery,
                page_display,
                results_area,
                total_pages_state,
            ]
            + page_buttons
            + [selected_festivals_checkbox],
        )

        # [*** 수정됨 ***]
        # 페이지네이션/랭킹 버튼이 체크박스도 업데이트하도록 수정
        checkbox_output = selected_festivals_checkbox
        pagination_outputs = [
            festival_gallery,
            page_display,
            *page_buttons,
            checkbox_output,
        ]
        ranking_outputs = [
            results_state,
            festival_gallery,
            page_display,
            *page_buttons,
            festival_ranking_report,
            total_pages_state,
            checkbox_output,
        ]
        ranking_sel_outputs = [
            results_state,
            festival_gallery,
            page_display,
            *page_buttons,
            festival_ranking_report_sel,
            total_pages_state,
            checkbox_output,
        ]

        # 전체 랭킹 버튼
        rank_festivals_btn.click(
            fn=event_handlers.handle_rank_festivals,
            inputs=[
                results_state,
                num_reviews_festival_ranking,
                festival_ranking_top_n_slider,
            ],
            outputs=ranking_outputs,
        )

        # 선택 랭킹 버튼
        rank_selected_btn.click(
            fn=event_handlers.handle_rank_selected_festivals,
            inputs=[
                results_state,
                selected_festivals_checkbox,
                num_reviews_festival_ranking_sel,
                festival_ranking_top_n_slider_sel,
            ],
            outputs=ranking_sel_outputs,
        )

        # 페이지네이션 버튼
        for btn in page_buttons:
            btn.click(
                fn=event_handlers.display_page,
                inputs=[results_state, btn],
                outputs=pagination_outputs,
            )
        page_input.submit(
            fn=event_handlers.display_page,
            inputs=[results_state, page_input],
            outputs=pagination_outputs,
        )
        prev_button.click(
            fn=event_handlers.display_paginated_gallery,
            inputs=[results_state, page_display, gr.State(-1)],
            outputs=pagination_outputs,
        )
        next_button.click(
            fn=event_handlers.display_paginated_gallery,
            inputs=[results_state, page_display, gr.State(1)],
            outputs=pagination_outputs,
        )
        first_page_button.click(
            fn=event_handlers.display_page,
            inputs=[results_state, gr.State(1)],
            outputs=pagination_outputs,
        )
        last_page_button.click(
            fn=lambda r, tp: event_handlers.display_page(r, int(tp)),
            inputs=[results_state, total_pages_state],
            outputs=pagination_outputs,
        )

        # --- (B-3) Details Page Handlers ---

        # (갤러리 클릭 -> 상세 페이지 이동)
        details_page_basic_outputs = [
            festival_details_output,
            selected_festival_state,
            selected_festival_details_state,
            precautions_output,
            temp_selection_state,  # '내 코스' 추가를 위한 상태
        ]

        festival_gallery.select(
            fn=event_handlers.handle_gallery_select_and_show_details,
            inputs=[results_state, page_display],
            outputs=details_page_basic_outputs
            + [top_level_tabs, detail_tabs_container],
        )

        # (상세 페이지 각 탭 내부의 기능 버튼 - 기존 바인딩과 동일)
        image_collect_button.click(
            fn=event_handlers.handle_scrape_images,
            inputs=[selected_festival_state, num_blogs_for_images],
            outputs=[image_gallery, scraped_urls_output],
        )
        naver_search_btn.click(
            fn=event_handlers.get_naver_review_info,
            inputs=[selected_festival_state, num_reviews_naver_summary],
            outputs=[naver_review_output],
        )
        trend_graph_btn.click(
            fn=event_handlers.handle_generate_trend_graphs,
            inputs=[selected_festival_state],
            outputs=[trend_status, trend_plot_yearly, trend_plot_event],
        )
        word_cloud_btn.click(
            fn=event_handlers.handle_generate_word_cloud,
            inputs=[selected_festival_state, num_reviews_wordcloud],
            outputs=[wordcloud_status, wordcloud_plot],
        )
        run_sentiment_btn.click(
            fn=event_handlers.handle_analyze_sentiment,
            inputs=[selected_festival_state, num_reviews_slider],
            outputs=[
                sentiment_status,
                sentiment_negative_summary,
                sentiment_overall_chart,
                sentiment_distribution_chart,
                sentiment_absolute_chart,
                sentiment_distribution_description,
                outlier_chart,
                outlier_description,
                sentiment_positive_keywords,
                sentiment_summary,
                sentiment_overall_csv,
                sentiment_df_output,
                blog_results_df_state,
                blog_judgments_state,
                sentiment_blog_page_num_input,
                sentiment_blog_total_pages_output,
                sentiment_blog_list_csv,
                sentiment_individual_summary,
                sentiment_individual_donut_chart,
                sentiment_individual_score_chart,
                sentiment_blog_detail_accordion,
            ],
        )

        # (추천 탭 기능 - 기존 바인딩과 동일)
        recommend_btn.click(
            fn=event_handlers.run_nearby_search,
            inputs=[selected_festival_details_state, recommend_radius_slider],
            outputs=[
                recommended_facilities_state,
                recommended_courses_state,
                recommended_festivals_state,
                recommend_status,
                recommend_facilities_gallery,
                recommend_courses_gallery,
                recommend_festivals_gallery,
                ranking_controls,
            ],
        )
        rank_facilities_btn.click(
            fn=functools.partial(event_handlers.handle_rank_places, is_course=False),
            inputs=[
                recommended_facilities_state,
                ranking_reviews_slider,
                ranking_top_n_slider,
            ],
            outputs=[
                recommended_facilities_state,
                recommend_status,
                recommend_facilities_gallery,
                facility_ranking_report,
            ],
        )
        rank_courses_btn.click(
            fn=functools.partial(event_handlers.handle_rank_places, is_course=True),
            inputs=[
                recommended_courses_state,
                ranking_reviews_slider,
                ranking_top_n_slider,
            ],
            outputs=[
                recommended_courses_state,
                recommend_status,
                recommend_courses_gallery,
                course_ranking_report,
            ],
        )
        rank_festivals_rec_btn.click(
            fn=event_handlers.handle_rank_festivals,
            inputs=[
                recommended_festivals_state,
                ranking_reviews_slider,
                ranking_top_n_slider,
            ],
            outputs=[
                recommended_festivals_state,
                recommend_festivals_gallery,
                rec_page_display,
            ]
            + rec_page_buttons
            + [
                festival_ranking_rec_report,
                recommended_total_pages_state,
                gr.State(None),
            ],  # [수정] 체크박스 출력을 위한 더미값
        )

        # (추천 탭 페이지네이션 - [수정] 체크박스 더미값 추가)
        rec_pagination_outputs = [
            recommend_festivals_gallery,
            rec_page_display,
            *rec_page_buttons,
            gr.State(None),  # 체크박스 출력을 위한 더미값
        ]
        for btn in rec_page_buttons:
            btn.click(
                fn=event_handlers.display_page,
                inputs=[recommended_festivals_state, btn],
                outputs=rec_pagination_outputs,
            )
        rec_page_input.submit(
            fn=event_handlers.display_page,
            inputs=[recommended_festivals_state, rec_page_input],
            outputs=rec_pagination_outputs,
        )
        rec_prev_button.click(
            fn=event_handlers.display_paginated_gallery,
            inputs=[recommended_festivals_state, rec_page_display, gr.State(-1)],
            outputs=rec_pagination_outputs,
        )
        rec_next_button.click(
            fn=event_handlers.display_paginated_gallery,
            inputs=[recommended_festivals_state, rec_page_display, gr.State(1)],
            outputs=rec_pagination_outputs,
        )
        rec_first_page_button.click(
            fn=event_handlers.display_page,
            inputs=[recommended_festivals_state, gr.State(1)],
            outputs=rec_pagination_outputs,
        )
        rec_last_page_button.click(
            fn=lambda r, tp: event_handlers.display_page(r, int(tp)),
            inputs=[recommended_festivals_state, recommended_total_pages_state],
            outputs=rec_pagination_outputs,
        )

        # (추천 탭 아이템 선택 -> 상세 정보 표시 - 기존 바인딩과 동일)
        recommend_facilities_gallery.select(
            fn=event_handlers.handle_item_selection,
            inputs=[recommended_facilities_state],
            outputs=[
                temp_selection_state,
                recommend_details_output,
                recommend_details_accordion,
            ],
        )
        recommend_courses_gallery.select(
            fn=event_handlers.handle_item_selection,
            inputs=[recommended_courses_state],
            outputs=[
                temp_selection_state,
                recommend_details_output,
                recommend_details_accordion,
            ],
        )
        recommend_festivals_gallery.select(
            fn=event_handlers.handle_item_selection,
            inputs=[recommended_festivals_state],
            outputs=[
                temp_selection_state,
                recommend_details_output,
                recommend_details_accordion,
            ],
        )

        # (감성 분석 탭 - 상세 보기 핸들러)
        sentiment_df_output.select(
            fn=event_handlers.handle_df_select,
            inputs=[
                sentiment_blog_page_num_input,
                blog_results_df_state,
                blog_judgments_state,
            ],
            outputs=[
                sentiment_individual_donut_chart,
                sentiment_individual_score_chart,
                sentiment_individual_summary,
                sentiment_blog_detail_accordion,
            ],
        )

        # --- [ 신규 ] 렌더링 탭 핸들러 ---
        rendering_btn.click(
            fn=event_handlers.handle_generate_rendering,
            inputs=[selected_festival_details_state],  # 현재 선택된 축제 상세 정보
            outputs=[
                rendering_status,  # 상태 텍스트박스
                representative_rendering_output,  # 대표 렌더링 이미지
                conditional_rendering_output,  # 조건별 렌더링 갤러리
            ],
        )
        # --- [ 신규 핸들러 추가 끝 ] ---

        # --- (B-4) My Course Page Handlers ---

        # '내 코스에 추가' 버튼 (2곳)
        add_to_my_course_btn_details.click(  # 1. 상세 페이지
            fn=event_handlers.add_to_my_course,
            inputs=[
                selected_festival_details_state,
                my_course_state,
            ],  # <--- 'selected_festival_details_state'로 변경
            outputs=[my_course_state, my_course_output],
        )
        add_to_my_course_btn_reco.click(  # 2. 추천 탭
            fn=event_handlers.add_to_my_course,
            inputs=[temp_selection_state, my_course_state],
            outputs=[my_course_state, my_course_output],
        )

        # '내 코스' 페이지 기능 (기존 바인딩과 동일)
        clear_my_course_btn.click(
            fn=event_handlers.clear_my_course,
            inputs=[],
            outputs=[my_course_state, my_course_output],
        )
        validate_course_btn.click(
            fn=event_handlers.handle_validate_course,
            inputs=[my_course_state, trip_duration_input],
            outputs=[course_validation_output],
        )

    return demo
