import gradio as gr
import functools

# Import data loaders and constants
from src.infrastructure.config.loader import ALL_FESTIVAL_CATEGORIES
from src.application.core.constants import AREA_CODE_MAP, SIGUNGU_CODE_MAP

# Import handlers and callbacks
from src.presentation import event_handlers
from src.presentation import callbacks

def create_ui():
    """Creates and returns the Gradio UI Blocks."""
    with gr.Blocks(css="""#gallery .thumbnail-item { max-width: 250px !important; min-width: 200px !important; flex-grow: 1 !important; }""") as demo:
        gr.Markdown("# 축제 정보 검색 에이전트")

        # --- State Variables ---
        results_state = gr.State([])
        selected_festival_state = gr.State()
        selected_festival_details_state = gr.State()
        total_pages_state = gr.State(1)
        recommended_facilities_state = gr.State([])
        recommended_courses_state = gr.State([])
        recommended_festivals_state = gr.State([])
        my_course_state = gr.State([])
        temp_selection_state = gr.State()
        blog_results_df_state = gr.State()
        blog_judgments_state = gr.State()

        # --- UI Definition ---
        with gr.Group():
            with gr.Row():
                area_dropdown = gr.Dropdown(label="시/도", choices=["전체"] + sorted(list(AREA_CODE_MAP.keys())), value="전체", interactive=True)
                sigungu_dropdown = gr.Dropdown(label="시/군/구", choices=["전체"], value="전체", interactive=True)
            with gr.Row():
                main_cat_dropdown = gr.Dropdown(label="대분류", choices=["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.keys())), value="전체", interactive=True)
                medium_cat_dropdown = gr.Dropdown(label="중분류", choices=["전체"], value="전체", interactive=True)
                small_cat_dropdown = gr.Dropdown(label="소분류", choices=["전체"], value="전체", interactive=True)
                status_radio = gr.Radio(label="진행 상태", choices=["전체", "축제 진행중", "진행 예정", "종료된 축제"], value="전체", interactive=True)
        with gr.Row():
            search_btn = gr.Button("검색", variant="primary", scale=1)
            rank_festivals_btn = gr.Button("축제 순위 보기", scale=1)
            num_reviews_festival_ranking = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="축제 순위용 리뷰 수", interactive=True, scale=2)
            festival_ranking_top_n_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="표시할 순위 수", interactive=True, scale=1)

        festival_ranking_report = gr.Markdown(visible=False)

        with gr.Column(visible=False) as results_area:
            festival_gallery = gr.Gallery(label="축제 목록", show_label=False, elem_id="gallery", columns=4, height="auto", object_fit="contain")
            with gr.Row(variant="panel"):
                first_page_button = gr.Button("<<", size="sm")
                prev_button = gr.Button("◀ 이전", size="sm")
                page_buttons = [gr.Button(str(i), visible=False, size="sm") for i in range(1, 6)]
                next_button = gr.Button("다음 ▶", size="sm")
                last_page_button = gr.Button(">>")
            with gr.Row():
                page_input = gr.Number(label="페이지 이동", value=1, interactive=True, scale=1)
                page_display = gr.Textbox(value="1 / 1", label="현재 페이지", interactive=False, container=False, scale=1)
        
        with gr.Accordion("축제 상세 정보", open=False) as details_accordion:
            festival_details_output = gr.Markdown()
            precautions_output = gr.Markdown(label="AI 기반 에티켓 가이드", visible=False)
            with gr.Row():
                num_blogs_for_images = gr.Slider(minimum=1, maximum=100, value=5, step=1, label="이미지 수집 대상 블로그 수", interactive=True)
                image_collect_button = gr.Button("이미지 수집하기")

        with gr.Accordion("이미지 모아보기", open=False, visible=False) as image_gallery_accordion:
            image_gallery = gr.Gallery(label="수집된 이미지", show_label=False, columns=4, height="auto", object_fit="contain")
            scraped_urls_output = gr.Textbox(label="Scraped Image URLs")

        with gr.Accordion("좌표 기반 추천", open=False, visible=False) as recommend_accordion:
            with gr.Row():
                recommend_radius_slider = gr.Slider(minimum=100, maximum=20000, value=5000, step=100, label="반경 (미터)", interactive=True)
                recommend_btn = gr.Button("추천 받기", variant="primary")
            with gr.Row(visible=False) as ranking_controls:
                ranking_reviews_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="순위용 리뷰 수", interactive=True)
                ranking_top_n_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="표시할 순위 수", interactive=True)
                rank_facilities_btn = gr.Button("관광 시설 순위 매기기")
                rank_courses_btn = gr.Button("관광 코스 순위 매기기")
                rank_festivals_rec_btn = gr.Button("추천 축제 순위 매기기")
            recommend_status = gr.Textbox(label="상태", interactive=False, visible=False)
            gr.Markdown("### 추천 관광 시설")
            recommend_facilities_gallery = gr.Gallery(label="추천 관광 시설", show_label=False, elem_id="recommend_facilities_gallery", columns=4, height="auto", object_fit="contain")
            facility_ranking_report = gr.Markdown(visible=False)
            gr.Markdown("### 추천 관광 코스")
            recommend_courses_gallery = gr.Gallery(label="추천 관광 코스", show_label=False, elem_id="recommend_courses_gallery", columns=4, height="auto", object_fit="contain")
            course_ranking_report = gr.Markdown(visible=False)
            gr.Markdown("### 추천 축제")
            recommend_festivals_gallery = gr.Gallery(label="추천 축제", show_label=False, elem_id="recommend_festivals_gallery", columns=4, height="auto", object_fit="contain")
            festival_ranking_rec_report = gr.Markdown(visible=False)
            with gr.Accordion("추천 장소 상세 정보", open=False, visible=False) as recommend_details_accordion:
                recommend_details_output = gr.Markdown()

        with gr.Accordion("Naver 후기 요약 및 꿀팁", open=False, visible=False) as naver_review_accordion:
            with gr.Row():
                num_reviews_naver_summary = gr.Slider(minimum=1, maximum=100, value=5, step=1, label="분석할 후기 수 (네이버 요약)", interactive=True)
                naver_search_btn = gr.Button("네이버 후기 요약 검색", variant="primary")
            naver_review_output = gr.Markdown()

        with gr.Accordion("검색량 트렌드 그래프", open=False, visible=False) as trend_accordion:
            trend_graph_btn = gr.Button("트렌드 그래프 생성", variant="primary")
            trend_status = gr.Textbox(label="상태", interactive=False)
            with gr.Row():
                trend_plot_yearly = gr.Image(label="최근 1년 검색량 트렌드")
                trend_plot_event = gr.Image(label="축제 기간 중심 트렌드")

        with gr.Accordion("워드 클라우드", open=False, visible=False) as wordcloud_accordion:
            with gr.Row():
                num_reviews_wordcloud = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="분석할 후기 수 (워드클라우드)", interactive=True)
                word_cloud_btn = gr.Button("워드 클라우드 생성", variant="primary")
            wordcloud_status = gr.Textbox(label="상태", interactive=False)
            wordcloud_plot = gr.Image(label="축제의 주요 핵심 요소들")

        with gr.Accordion("감성 분석", open=False, visible=False) as sentiment_accordion:
            with gr.Row():
                num_reviews_slider = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="분석할 후기 수", interactive=True)
                run_sentiment_btn = gr.Button("감성 분석 실행", variant="primary")
            sentiment_status = gr.Textbox(label="분석 상태", interactive=False)
            with gr.Accordion("종합 분석 결과", open=True):
                sentiment_summary = gr.Markdown(label="종합 분석 상세", visible=False)
                sentiment_overall_csv = gr.File(label="종합 분석 (CSV) 다운로드", visible=False)
                sentiment_positive_keywords = gr.HTML(visible=False)
                with gr.Row():
                    sentiment_overall_chart = gr.Plot(label="전체 후기 요약", visible=False, scale=1)
                    sentiment_distribution_chart = gr.Image(label="만족도 점수 분포", visible=False, scale=1)
                sentiment_distribution_description = gr.Markdown(visible=False)
                with gr.Row():
                    outlier_chart = gr.Image(label="이상치 탐지 결과", visible=False, scale=1)
                outlier_description = gr.Markdown(visible=False)
                sentiment_negative_summary = gr.Markdown(label="주요 불만 사항 요약", visible=False)
                with gr.Accordion("계절별 상세 분석", open=False):
                    with gr.Row():
                        sentiment_spring_chart = gr.Plot(label="봄 시즌", visible=False, scale=1)
                        sentiment_spring_pos_wc = gr.Image(label="봄 긍정 워드클라우드", visible=False, scale=1)
                        sentiment_spring_neg_wc = gr.Image(label="봄 부정 워드클라우드", visible=False, scale=1)
                    with gr.Row():
                        sentiment_summer_chart = gr.Plot(label="여름 시즌", visible=False, scale=1)
                        sentiment_summer_pos_wc = gr.Image(label="여름 긍정 워드클라우드", visible=False, scale=1)
                        sentiment_summer_neg_wc = gr.Image(label="여름 부정 워드클라우드", visible=False, scale=1)
                    with gr.Row():
                        sentiment_autumn_chart = gr.Plot(label="가을 시즌", visible=False, scale=1)
                        sentiment_autumn_pos_wc = gr.Image(label="가을 긍정 워드클라우드", visible=False, scale=1)
                        sentiment_autumn_neg_wc = gr.Image(label="가을 부정 워드클라우드", visible=False, scale=1)
                    with gr.Row():
                        sentiment_winter_chart = gr.Plot(label="겨울 시즌", visible=False, scale=1)
                        sentiment_winter_pos_wc = gr.Image(label="겨울 긍정 워드클라우드", visible=False, scale=1)
                        sentiment_winter_neg_wc = gr.Image(label="겨울 부정 워드클라우드", visible=False, scale=1)
            gr.Markdown("### 개별 블로그 분석 결과")
            sentiment_df_output = gr.DataFrame(headers=["블로그 제목", "링크", "감성 빈도", "감성 점수", "긍정 문장 수", "부정 문장 수", "긍정 비율 (%)", "부정 비율 (%)", "긍/부정 문장 요약"], datatype=["str", "str", "number", "str", "number", "number", "str", "str", "str"], label="개별 블로그 분석 결과", wrap=True, interactive=True)
            with gr.Row():
                sentiment_blog_page_num_input = gr.Number(value=1, label="페이지 번호", interactive=True, scale=1)
                sentiment_blog_total_pages_output = gr.Textbox(value="/ 1", label="전체 페이지", interactive=False, container=False, scale=1)
                sentiment_blog_list_csv = gr.File(label="전체 블로그 목록(CSV) 다운로드", visible=False, scale=2)
            with gr.Accordion("개별 블로그 상세 분석 (표에서 행 선택)", open=False, visible=False) as sentiment_blog_detail_accordion:
                sentiment_individual_summary = gr.Textbox(label="긍/부정 문장 요약", visible=False, interactive=False, lines=10)
                with gr.Row():
                    sentiment_individual_donut_chart = gr.Plot(label="개별 블로그 긍/부정 비율", visible=False)
                    sentiment_individual_score_chart = gr.Plot(label="문장별 감성 점수", visible=False)

        with gr.Accordion("나만의 코스", open=True) as my_course_accordion:
            my_course_output = gr.Markdown("나만의 코스에 항목을 추가해보세요.")
            with gr.Row():
                add_to_my_course_btn = gr.Button("선택한 항목 '나만의 코스'에 추가")
                clear_my_course_btn = gr.Button("나만의 코스 비우기")
            gr.Markdown("---")
            gr.Markdown("### 📅 코스 현실성 검증")
            with gr.Row():
                trip_duration_input = gr.Textbox(label="총 여행 기간을 입력하세요 (예: 2박 3일, 당일치기)", placeholder="예: 1박 2일")
                validate_course_btn = gr.Button("검증하기", variant="primary")
            course_validation_output = gr.Markdown()

        # --- Event Handlers Binding ---
        update_sigungu_with_map = functools.partial(callbacks.update_sigungu, sigungu_code_map=SIGUNGU_CODE_MAP)
        area_dropdown.change(fn=update_sigungu_with_map, inputs=area_dropdown, outputs=sigungu_dropdown)
        update_medium_cat_with_map = functools.partial(callbacks.update_medium_cat, all_festival_categories=ALL_FESTIVAL_CATEGORIES)
        main_cat_dropdown.change(fn=update_medium_cat_with_map, inputs=main_cat_dropdown, outputs=medium_cat_dropdown)
        update_small_cat_with_map = functools.partial(callbacks.update_small_cat, all_festival_categories=ALL_FESTIVAL_CATEGORIES)
        medium_cat_dropdown.change(fn=update_small_cat_with_map, inputs=[main_cat_dropdown, medium_cat_dropdown], outputs=small_cat_dropdown)

        search_btn.click(fn=event_handlers.run_search_and_display, inputs=[area_dropdown, sigungu_dropdown, main_cat_dropdown, medium_cat_dropdown, small_cat_dropdown, status_radio], outputs=[results_state, festival_gallery, page_display, results_area, total_pages_state] + page_buttons)
        rank_festivals_btn.click(fn=event_handlers.handle_rank_festivals, inputs=[results_state, num_reviews_festival_ranking, festival_ranking_top_n_slider], outputs=[results_state, festival_gallery, page_display] + page_buttons + [festival_ranking_report])

        for btn in page_buttons:
            btn.click(fn=event_handlers.display_page, inputs=[results_state, btn], outputs=[festival_gallery, page_display] + page_buttons)
        page_input.submit(fn=event_handlers.display_page, inputs=[results_state, page_input], outputs=[festival_gallery, page_display] + page_buttons)
        prev_button.click(fn=event_handlers.display_paginated_gallery, inputs=[results_state, page_display, gr.State(-1)], outputs=[festival_gallery, page_display] + page_buttons)
        next_button.click(fn=event_handlers.display_paginated_gallery, inputs=[results_state, page_display, gr.State(1)], outputs=[festival_gallery, page_display] + page_buttons)
        first_page_button.click(fn=event_handlers.display_page, inputs=[results_state, gr.State(1)], outputs=[festival_gallery, page_display] + page_buttons)
        last_page_button.click(fn=lambda r, tp: event_handlers.display_page(r, tp), inputs=[results_state, total_pages_state], outputs=[festival_gallery, page_display] + page_buttons)

        festival_gallery.select(fn=event_handlers.display_festival_details_and_precautions, inputs=[results_state, page_display], outputs=[
            festival_details_output, selected_festival_state, selected_festival_details_state, precautions_output,
            naver_review_accordion, trend_accordion, wordcloud_accordion, sentiment_accordion, recommend_accordion, image_gallery_accordion,
            temp_selection_state, details_accordion
        ])

        image_collect_button.click(fn=event_handlers.handle_scrape_images, inputs=[selected_festival_state, num_blogs_for_images], outputs=[image_gallery, image_gallery_accordion, scraped_urls_output])
        naver_search_btn.click(fn=event_handlers.get_naver_review_info, inputs=[selected_festival_state, num_reviews_naver_summary], outputs=[naver_review_output, naver_review_accordion])
        trend_graph_btn.click(fn=event_handlers.handle_generate_trend_graphs, inputs=[selected_festival_state], outputs=[trend_accordion, trend_status, trend_plot_yearly, trend_plot_event])
        word_cloud_btn.click(fn=event_handlers.handle_generate_word_cloud, inputs=[selected_festival_state, num_reviews_wordcloud], outputs=[wordcloud_accordion, wordcloud_status, wordcloud_plot])
        run_sentiment_btn.click(fn=event_handlers.handle_analyze_sentiment, inputs=[selected_festival_state, num_reviews_slider], outputs=[
            sentiment_accordion, sentiment_status, sentiment_negative_summary, sentiment_overall_chart, sentiment_distribution_chart, sentiment_distribution_description, outlier_chart, outlier_description, sentiment_positive_keywords, sentiment_summary, sentiment_overall_csv,
            sentiment_spring_chart, sentiment_summer_chart, sentiment_autumn_chart, sentiment_winter_chart,
            sentiment_spring_pos_wc, sentiment_spring_neg_wc, sentiment_summer_pos_wc, sentiment_summer_neg_wc, sentiment_autumn_pos_wc, sentiment_autumn_neg_wc, sentiment_winter_pos_wc, sentiment_winter_neg_wc,
            sentiment_df_output, blog_results_df_state, blog_judgments_state, sentiment_blog_page_num_input, sentiment_blog_total_pages_output, sentiment_blog_list_csv,
            sentiment_individual_summary, sentiment_individual_donut_chart, sentiment_individual_score_chart, sentiment_blog_detail_accordion
        ])

        recommend_btn.click(fn=event_handlers.run_nearby_search, inputs=[selected_festival_details_state, recommend_radius_slider], outputs=[
            recommended_facilities_state, recommended_courses_state, recommended_festivals_state, recommend_status, recommend_facilities_gallery, recommend_courses_gallery, recommend_festivals_gallery, ranking_controls
        ])
        rank_facilities_btn.click(fn=functools.partial(event_handlers.handle_rank_places, is_course=False), inputs=[recommended_facilities_state, ranking_reviews_slider, ranking_top_n_slider], outputs=[recommended_facilities_state, recommend_status, recommend_facilities_gallery, facility_ranking_report])
        rank_courses_btn.click(fn=functools.partial(event_handlers.handle_rank_places, is_course=True), inputs=[recommended_courses_state, ranking_reviews_slider, ranking_top_n_slider], outputs=[recommended_courses_state, recommend_status, recommend_courses_gallery, course_ranking_report])
        rank_festivals_rec_btn.click(fn=event_handlers.handle_rank_festivals, inputs=[recommended_festivals_state, ranking_reviews_slider, ranking_top_n_slider], outputs=[recommended_festivals_state, recommend_festivals_gallery, festival_ranking_rec_report])

        # Handlers for selecting items from recommendation galleries
        recommend_facilities_gallery.select(fn=event_handlers.handle_item_selection, inputs=[recommended_facilities_state], outputs=[temp_selection_state, recommend_details_output, recommend_details_accordion])
        recommend_courses_gallery.select(fn=event_handlers.handle_item_selection, inputs=[recommended_courses_state], outputs=[temp_selection_state, recommend_details_output, recommend_details_accordion])
        recommend_festivals_gallery.select(fn=event_handlers.handle_item_selection, inputs=[recommended_festivals_state], outputs=[temp_selection_state, recommend_details_output, recommend_details_accordion])

        add_to_my_course_btn.click(fn=event_handlers.add_to_my_course, inputs=[temp_selection_state, my_course_state], outputs=[my_course_state, my_course_output])
        clear_my_course_btn.click(fn=event_handlers.clear_my_course, inputs=[], outputs=[my_course_state, my_course_output])
        validate_course_btn.click(fn=event_handlers.handle_validate_course, inputs=[my_course_state, trip_duration_input], outputs=[course_validation_output])

        sentiment_df_output.select(fn=event_handlers.handle_df_select, inputs=[sentiment_blog_page_num_input, blog_results_df_state, blog_judgments_state], outputs=[
            sentiment_individual_donut_chart, sentiment_individual_score_chart, sentiment_individual_summary, sentiment_blog_detail_accordion
        ])

    return demo