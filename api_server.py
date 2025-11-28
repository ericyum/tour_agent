import sys
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64
from io import BytesIO
import re

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import matplotlib.pyplot as plt
from PIL import Image

# Environment and Initial Setup
from src.infrastructure.config.settings import setup_environment

setup_environment()

# Import the database initializer
from src.infrastructure.persistence.database import init_db, get_db_connection, release_connection, get_cursor

# Initialize database BEFORE importing loader (which queries DB at import time)
init_db()

# Import configurations and utilities
from src.infrastructure.config.loader import (
    ICON_MAP,
    KOREAN_FONT_PATH,
    TITLE_TO_CAT_NAMES,
    ALL_FESTIVAL_CATEGORIES,
    FESTIVAL_INFO_LOOKUP,
    BEST_IMAGES_MAP,
    DF_SPLIT,
    DF_CAMERA,
)
from src.application.core.constants import (
    CATEGORY_TO_ICON_MAP,
    NO_IMAGE_URL,
    PAGE_SIZE,
    AREA_CODE_MAP,
    SIGUNGU_CODE_MAP,
)

# Import services and agents
from src.application.services.festival_service import get_festival_details_by_title
from src.application.agents.precaution_agent import PrecautionAgent
from src.application.supervisors.db_search_supervisor import db_search_graph
from src.application.supervisors.course_validation_supervisor import (
    course_validation_graph,
)
from src.application.agents.naver_review.naver_review_agent import NaverReviewAgent
from src.application.use_cases.analysis_use_case import AnalysisUseCase
from src.application.use_cases.sentiment_analysis_use_case import (
    SentimentAnalysisUseCase,
)
from src.application.use_cases.ranking_use_case import RankingUseCase
from src.application.use_cases.rendering_use_case import RenderingUseCase
from src.application.services.course_service import get_course_details_by_title
from src.application.services.facility_service import get_facility_details_by_title
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.infrastructure.cache_manager import (
    load_from_cache,
    save_to_cache,
    get_task_progress,
    get_task_result,
    get_cache_stats,
    REDIS_AVAILABLE,
)
from src.application.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
    revoke_refresh_token,
    revoke_all_user_tokens,
    get_current_user_from_token,
    require_auth,
    require_admin,
)
from src.application.services import qna_service

# Initialize FastAPI app
app = FastAPI(
    title="FestMoment API",
    description="AI-powered Festival Guide Service",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://festmoment.co.kr",
        "https://www.festmoment.co.kr",
        "https://festmoment.onrender.com",
        "https://festmoment-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
script_dir = os.path.dirname(os.path.abspath(__file__))

# Mount static files for images and icons
best_images_path = os.path.join(script_dir, "best_images_and_icons")
temp_images_path = os.path.join(script_dir, "temp_img")
os.makedirs(temp_images_path, exist_ok=True)  # Ensure temp_img directory exists

if os.path.exists(best_images_path):
    app.mount(
        "/api/assets/best_images_and_icons",
        StaticFiles(directory=best_images_path),
        name="assets",
    )
app.mount("/api/temp_img", StaticFiles(directory=temp_images_path), name="temp_img")

naver_supervisor = NaverReviewAgent()
precaution_agent = PrecautionAgent()

analysis_use_case = AnalysisUseCase(
    naver_supervisor=naver_supervisor,
    font_path=KOREAN_FONT_PATH,
    title_to_cat_map=TITLE_TO_CAT_NAMES,
    cat_to_icon_map=CATEGORY_TO_ICON_MAP,
    script_dir=script_dir,
)
sentiment_analysis_use_case = SentimentAnalysisUseCase(
    naver_supervisor=naver_supervisor, script_dir=script_dir
)
ranking_use_case = RankingUseCase(naver_supervisor=naver_supervisor)
rendering_use_case = RenderingUseCase(df_split=DF_SPLIT, df_camera=DF_CAMERA)


# Pydantic Models for Request/Response
class SearchRequest(BaseModel):
    area: Optional[str] = "전체"
    sigungu: Optional[str] = "전체"
    main_cat: Optional[str] = "전체"
    medium_cat: Optional[str] = "전체"
    small_cat: Optional[str] = "전체"
    status: Optional[str] = "전체"
    page: int = 1


class FestivalResponse(BaseModel):
    title: str
    image: str
    start_date: Optional[str]
    end_date: Optional[str]


class RankingRequest(BaseModel):
    festivals: List[str]
    num_reviews: int = 10
    top_n: int = 3


class CourseValidationRequest(BaseModel):
    course: List[Dict[str, Any]]
    duration: str


class NearbySearchRequest(BaseModel):
    latitude: float
    longitude: float
    radius: float
    current_festival_id: Optional[str] = None


class SentimentChartResponse(BaseModel):
    donut_chart: Optional[str] = None
    satisfaction_chart: Optional[str] = None
    wordcloud_positive: Optional[str] = None
    wordcloud_negative: Optional[str] = None
    absolute_chart: Optional[str] = None
    outlier_chart: Optional[str] = None
    # Chart data for frontend rendering
    donut_data: Optional[Dict[str, Any]] = None
    satisfaction_data: Optional[Dict[str, Any]] = None
    absolute_data: Optional[Dict[str, Any]] = None
    outlier_data: Optional[Dict[str, Any]] = None


class SentimentAnalysisResponse(BaseModel):
    summary: str
    positive_count: int
    negative_count: int
    neutral_count: int
    charts: SentimentChartResponse
    blog_results: List[Dict[str, Any]]
    blog_list_csv_path: Optional[str] = None
    positive_keywords: Optional[str] = None
    negative_summary: Optional[str] = None
    outlier_description: Optional[str] = None
    total_score_count: Optional[int] = None
    outlier_count: Optional[int] = None
    blog_judgments_list: Optional[List[List[Dict[str, Any]]]] = None
    overall_summary_text: Optional[str] = None


# Helper to convert images to base64
def fig_to_base64(fig):
    if fig is None:
        return None
    buf = BytesIO()
    if isinstance(fig, plt.Figure):
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
    elif isinstance(fig, Image.Image):
        fig.save(buf, format="PNG")
    elif isinstance(fig, str) and os.path.exists(fig):
        try:
            with open(fig, "rb") as f:
                buf.write(f.read())
        except IOError:
            return None
    else:
        return None
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# API Endpoints


@app.on_event("startup")
async def startup_event():
    """Server startup event"""
    print("✅ FestMoment API Server Started")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "FestMoment API", "version": "1.0.0"}


@app.get("/api/config/areas")
async def get_areas():
    """Get all available areas"""
    return {"areas": ["전체"] + sorted(list(AREA_CODE_MAP.keys()))}


@app.get("/api/config/sigungus")
async def get_sigungus(area: str = Query(...)):
    """Get sigungus for a specific area"""
    if area == "전체":
        return {"sigungus": ["전체"]}
    return {"sigungus": ["전체"] + sorted(list(SIGUNGU_CODE_MAP.get(area, {}).keys()))}


@app.get("/api/config/categories")
async def get_categories():
    """Get all festival categories"""
    return {"main_categories": ["전체"] + sorted(list(ALL_FESTIVAL_CATEGORIES.keys()))}


@app.get("/api/config/categories/medium")
async def get_medium_categories(main_cat: str = Query(...)):
    """Get medium categories for a main category"""
    if main_cat == "전체":
        return {"medium_categories": ["전체"]}
    return {
        "medium_categories": ["전체"]
        + sorted(list(ALL_FESTIVAL_CATEGORIES.get(main_cat, {}).keys()))
    }


@app.get("/api/config/categories/small")
async def get_small_categories(
    main_cat: str = Query(...), medium_cat: str = Query(...)
):
    """Get small categories for a medium category"""
    if main_cat == "전체" or medium_cat == "전체":
        return {"small_categories": ["전체"]}
    return {
        "small_categories": ["전체"]
        + sorted(
            list(ALL_FESTIVAL_CATEGORIES.get(main_cat, {}).get(medium_cat, {}).keys())
        )
    }


@app.post("/api/festivals/search")
async def search_festivals(request: SearchRequest):
    """Search festivals with filters"""
    try:
        # Use db_search_graph
        state = {
            "search_type": "festival_search",
            "area": request.area,
            "sigungu": request.sigungu,
            "main_cat": request.main_cat,
            "medium_cat": request.medium_cat,
            "small_cat": request.small_cat,
            "results": None,
        }

        result_state = db_search_graph.invoke(state)
        results = result_state.get("results", [])

        # Filter by status if needed
        if request.status != "전체":
            today = datetime.now().strftime("%Y%m%d")
            filtered = []
            for title, image, start_date, end_date in results:
                if request.status == "축제 진행중":
                    if start_date and end_date and start_date <= today <= end_date:
                        filtered.append((title, image, start_date, end_date))
                elif request.status == "진행 예정":
                    if start_date and start_date > today:
                        filtered.append((title, image, start_date, end_date))
                elif request.status == "종료된 축제":
                    if end_date and end_date < today:
                        filtered.append((title, image, start_date, end_date))
            results = filtered

        # Pagination
        total = len(results)
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE if total > 0 else 1
        start_idx = (request.page - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_results = results[start_idx:end_idx]

        festivals = [
            {
                "title": title,
                "image": image or NO_IMAGE_URL,
                "start_date": start_date,
                "end_date": end_date,
            }
            for title, image, start_date, end_date in page_results
        ]

        return {
            "festivals": festivals,
            "total": total,
            "page": request.page,
            "total_pages": total_pages,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/festivals/{festival_name}")
async def get_festival_details(festival_name: str):
    """Get detailed information for a specific festival"""
    try:
        details = get_festival_details_by_title(festival_name)
        if not details:
            raise HTTPException(status_code=404, detail="Festival not found")

        # Get icon and best image paths
        icon_path = get_local_icon_path(festival_name)
        best_image_path = get_local_best_image_path(festival_name)

        return {
            "details": details,
            "icon_path": icon_path,
            "best_image_path": best_image_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/courses/{course_title}")
async def get_course_details(course_title: str):
    """Get detailed information for a specific course"""
    try:
        details = get_course_details_by_title(course_title)
        if not details:
            raise HTTPException(status_code=404, detail="Course not found")
        return {"details": details}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/facilities/{facility_title}")
async def get_facility_details(facility_title: str):
    """Get detailed information for a specific facility"""
    try:
        details = get_facility_details_by_title(facility_title)
        if not details:
            raise HTTPException(status_code=404, detail="Facility not found")
        return {"details": details}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/api/festivals/{festival_name}/trend")
async def get_festival_trend(festival_name: str):
    """Get trend graphs for a festival"""
    try:
        # 캐시 확인
        cached_data = load_from_cache("trend", festival_name=festival_name)
        if cached_data:
            return cached_data

        yearly_img, event_img, message = await analysis_use_case.generate_trend_graphs(
            festival_name
        )

        if not yearly_img and not event_img:
            result = {"message": message, "yearly_trend": None, "event_trend": None}
            save_to_cache(result, "trend", festival_name=festival_name)
            return result

        # Convert PIL images to base64
        yearly_b64 = None
        if yearly_img:
            buffered = BytesIO()
            yearly_img.save(buffered, format="PNG")
            yearly_b64 = base64.b64encode(buffered.getvalue()).decode()

        event_b64 = None
        if event_img:
            buffered = BytesIO()
            event_img.save(buffered, format="PNG")
            event_b64 = base64.b64encode(buffered.getvalue()).decode()

        result = {
            "yearly_trend": (
                f"data:image/png;base64,{yearly_b64}" if yearly_b64 else None
            ),
            "event_trend": f"data:image/png;base64,{event_b64}" if event_b64 else None,
            "message": message,
        }

        # 캐시 저장
        save_to_cache(result, "trend", festival_name=festival_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/festivals/{festival_name}/sentiment", response_model=SentimentAnalysisResponse
)
async def get_sentiment_analysis(
    festival_name: str, num_reviews: int = Query(10, ge=1, le=50)
):
    """Get sentiment analysis for a festival"""
    try:
        # 캐시 확인
        cached_data = load_from_cache("sentiment", festival_name=festival_name, num_reviews=num_reviews)
        if cached_data:
            return SentimentAnalysisResponse(**cached_data)

        result = await sentiment_analysis_use_case.analyze_sentiment(
            festival_name, num_reviews
        )

        # --- Wordcloud Masking Logic ---
        mask_path = None
        info = FESTIVAL_INFO_LOOKUP.get(festival_name)

        print(f"[WordCloud] Festival: {festival_name}")
        print(f"[WordCloud] Info found: {info is not None}")

        if info:
            print(f"[WordCloud] Info keys: {list(info.keys())}")
            eventstartdate = info.get("eventstartdate")
            print(f"[WordCloud] eventstartdate: {eventstartdate}")

            if eventstartdate:
                try:
                    start_date_str = str(eventstartdate)
                    print(f"[WordCloud] start_date_str: {start_date_str}")
                    month = int(start_date_str[4:6])
                    print(f"[WordCloud] Month: {month}")

                    if 3 <= month <= 5:
                        mask_filename = "mask_spring.png"
                    elif 6 <= month <= 8:
                        mask_filename = "mask_summer.png"
                    elif 9 <= month <= 11:
                        mask_filename = "mask_fall.png"
                    else: # 12, 1, 2
                        mask_filename = "mask_winter.png"

                    print(f"[WordCloud] Selected mask: {mask_filename}")
                    potential_path = os.path.join(script_dir, "assets", "seasons", mask_filename)
                    print(f"[WordCloud] Checking path: {potential_path}")

                    if os.path.exists(potential_path):
                        mask_path = potential_path
                        print(f"[WordCloud] ✓ Using mask: {mask_path}")
                    else:
                        print(f"[WordCloud] ✗ Mask not found at: {potential_path}")

                except (ValueError, IndexError) as e:
                    print(f"[WordCloud] Error parsing date: {e}")
                    import traceback
                    traceback.print_exc()
                    mask_path = None
            else:
                print(f"[WordCloud] No eventstartdate in info")
        else:
            print(f"[WordCloud] Festival not found in FESTIVAL_INFO_LOOKUP")

        print(f"[WordCloud] Calling create_sentiment_wordclouds with mask_path: {mask_path}")
        pos_wordcloud, neg_wordcloud = create_sentiment_wordclouds(
            result["all_aspect_sentiment_pairs"], festival_name, mask_path=mask_path
        )
        print(f"[WordCloud] Wordclouds generated successfully")

        # Extract counts from the summary text
        summary_text = result.get("overall_summary_text", "")
        pos_match = re.search(r"긍정 문장 수: (\d+)", summary_text)
        neg_match = re.search(r"부정 문장 수: (\d+)", summary_text)

        positive_count = int(pos_match.group(1)) if pos_match else 0
        negative_count = int(neg_match.group(1)) if neg_match else 0

        # Convert full DataFrame to list of dicts for the response
        blog_results = []
        if "blog_df" in result and not result["blog_df"].empty:
            # Replace NaN with None for JSON compatibility
            df_cleaned = result["blog_df"].replace({float('nan'): None})
            blog_results = df_cleaned.to_dict(orient="records")

        # Create outlier description like archive_gradio
        outlier_description = None
        if result.get("total_score_count") and result.get("outlier_count") is not None:
            outlier_description = f"총 **{result['total_score_count']}**개의 감성 점수 중 **{result['outlier_count']}**개의 이상치가 발견되었습니다."

        response = SentimentAnalysisResponse(
            summary=result.get("distribution_description", "요약 정보 없음"),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=0,  # Neutral count is not explicitly calculated in the use case
            charts=SentimentChartResponse(
                donut_chart=fig_to_base64(result.get("overall_chart")),
                satisfaction_chart=fig_to_base64(result.get("distribution_chart")),
                wordcloud_positive=fig_to_base64(pos_wordcloud),
                wordcloud_negative=fig_to_base64(neg_wordcloud),
                absolute_chart=fig_to_base64(result.get("absolute_chart")),
                outlier_chart=fig_to_base64(result.get("outlier_chart")),
                # Add chart data for frontend rendering
                donut_data=result.get("donut_data"),
                satisfaction_data=result.get("satisfaction_data"),
                absolute_data=result.get("absolute_data"),
                outlier_data=result.get("outlier_data"),
            ),
            blog_results=blog_results,
            blog_list_csv_path=result.get("blog_list_csv_path"),
            positive_keywords=result.get("positive_keywords_html"),
            negative_summary=result.get("neg_summary_text"),
            outlier_description=outlier_description,
            total_score_count=result.get("total_score_count"),
            outlier_count=result.get("outlier_count"),
            blog_judgments_list=result.get("blog_judgments_list"),
            overall_summary_text=result.get("overall_summary_text"),
        )

        # 캐시 저장
        save_to_cache(response.dict(), "sentiment", festival_name=festival_name, num_reviews=num_reviews)
        return response
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/festivals/{festival_name}/images")
async def scrape_images(festival_name: str, num_blogs: int = Query(5, ge=1, le=20)):
    """Scrape images from Naver blogs for a festival"""
    try:
        local_image_paths, _ = await analysis_use_case.scrape_festival_images(
            festival_name, num_blogs
        )
        # Convert local paths to server-relative URLs
        server_urls = [
            f"/api/temp_img/{os.path.basename(p)}" for p in local_image_paths
        ]
        return {"image_urls": server_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/festivals/{festival_name}/wordcloud")
async def get_wordcloud(festival_name: str, num_reviews: int = Query(20, ge=1, le=100)):
    """Generate a word cloud for a festival"""
    try:
        # 캐시 확인
        cached_data = load_from_cache("wordcloud", festival_name=festival_name, num_reviews=num_reviews)
        if cached_data:
            return cached_data

        wc_image, message = await analysis_use_case.generate_word_cloud(
            festival_name, num_reviews
        )
        if not wc_image:
            raise HTTPException(status_code=404, detail=message)

        buffered = BytesIO()
        wc_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        result = {"wordcloud": f"data:image/png;base64,{img_b64}", "message": message}

        # 캐시 저장
        save_to_cache(result, "wordcloud", festival_name=festival_name, num_reviews=num_reviews)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/festivals/{festival_name}/review-summary")
async def get_review_summary(
    festival_name: str, num_reviews: int = Query(5, ge=1, le=50)
):
    """Get an AI-generated summary of Naver blog reviews"""
    try:
        # 캐시 확인
        cached_data = load_from_cache("review_summary", festival_name=festival_name, num_reviews=num_reviews)
        if cached_data:
            return cached_data

        summary, _ = await naver_supervisor.get_review_summary_and_tips(
            festival_name, num_reviews=num_reviews
        )
        result = {"summary": summary}

        # 캐시 저장
        save_to_cache(result, "review_summary", festival_name=festival_name, num_reviews=num_reviews)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/festivals/{festival_name}/precautions")
async def get_precautions(festival_name: str):
    """Get AI-generated precautions for a festival"""
    try:
        # 캐시 확인
        cached_data = load_from_cache("precautions", festival_name=festival_name)
        if cached_data:
            return cached_data

        print(f"[Precautions] Requested for: '{festival_name}'")
        print(f"[Precautions] Total festivals in lookup: {len(FESTIVAL_INFO_LOOKUP)}")

        info = FESTIVAL_INFO_LOOKUP.get(festival_name)
        if not info:
            # Try to find similar festival names for debugging
            similar = [k for k in FESTIVAL_INFO_LOOKUP.keys() if festival_name in k or k in festival_name]
            if similar:
                print(f"[Precautions] Festival not found. Similar names: {similar[:5]}")
            else:
                print(f"[Precautions] Festival not found. No similar names.")
            result = {"precautions": "이 축제에 대한 특별한 주의사항 정보가 없습니다."}
            save_to_cache(result, "precautions", festival_name=festival_name)
            return result

        detailed_category = info.get("detailed_category", "")
        prohibited_behaviors = info.get("prohibited_behaviors", "")

        print(f"[Precautions] detailed_category: {detailed_category if detailed_category else 'None'}")
        print(f"[Precautions] prohibited_behaviors: {prohibited_behaviors[:100] if prohibited_behaviors else 'None'}...")

        if not detailed_category and not prohibited_behaviors:
            print(f"[Precautions] No precaution data available for this festival")
            result = {"precautions": "이 축제에 대한 특별한 주의사항 정보가 없습니다."}
            save_to_cache(result, "precautions", festival_name=festival_name)
            return result

        precautions = await precaution_agent.generate_precautions(
            festival_name, detailed_category, prohibited_behaviors
        )

        result = {"precautions": precautions}

        # 캐시 저장
        save_to_cache(result, "precautions", festival_name=festival_name)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/festivals/ranking")
async def rank_festivals(request: RankingRequest):
    """Rank selected festivals based on sentiment and trend analysis"""
    try:
        # 캐시 확인 (축제 목록을 정렬하여 순서에 상관없이 동일한 캐시 키 생성)
        sorted_festivals = sorted(request.festivals)
        cached_data = load_from_cache(
            "ranking",
            festivals=tuple(sorted_festivals),
            num_reviews=request.num_reviews,
            top_n=request.top_n
        )
        if cached_data:
            return cached_data

        # Fetch full festival details from database for each festival name
        conn = get_db_connection()
        cursor = get_cursor(conn)

        try:
            festivals_data = []
            for festival_name in request.festivals:
                cursor.execute(
                    """
                    SELECT * FROM festivals WHERE title = %s
                """,
                    (festival_name,),
                )
                row = cursor.fetchone()
                if row:
                    # Convert to dict (RealDictCursor returns dict-like rows)
                    festival_dict = dict(row)
                    festivals_data.append(festival_dict)
        finally:
            cursor.close()
            release_connection(conn)

        if not festivals_data:
            raise HTTPException(
                status_code=404, detail="선택한 축제를 찾을 수 없습니다"
            )

        ranked_festivals, analysis = await ranking_use_case.rank_festivals(
            festivals_data, request.num_reviews, request.top_n
        )
        result = {"ranked_festivals": ranked_festivals, "analysis": analysis}

        # 캐시 저장
        save_to_cache(
            result,
            "ranking",
            festivals=tuple(sorted_festivals),
            num_reviews=request.num_reviews,
            top_n=request.top_n
        )
        return result
    except Exception as e:
        import traceback

        print(f"Ranking error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/festivals/{festival_name}/render")
async def render_festival_image(festival_name: str):
    """Generate AI-rendered image for a festival"""
    try:
        # 캐시 확인
        cached_data = load_from_cache("render", festival_name=festival_name)
        if cached_data:
            return cached_data

        print(f"[Rendering] Requested for: '{festival_name}'")

        # 1. Get festival details
        details = get_festival_details_by_title(festival_name)
        if not details:
            raise HTTPException(status_code=404, detail="Festival not found")

        # 2. Call the rendering use case
        generated_paths = await rendering_use_case.generate_festival_renderings(details)

        # 3. Process representative image
        representative_image = None
        rep_path = generated_paths.get("representative")
        if rep_path and os.path.exists(rep_path):
            representative_image = {
                "image_base64": fig_to_base64(rep_path),
                "prompt": f"AI-generated representative rendering of the '{festival_name}' festival."
            }
            print(f"[Rendering] Success! Representative image generated at {rep_path}")

        # 4. Process conditional images
        conditional_images = []
        cond_paths = generated_paths.get("conditional", [])
        for i, cond_path in enumerate(cond_paths):
            if cond_path and os.path.exists(cond_path):
                # Extract condition name from filename, e.g., "조건_1_야간_취식_aerial.png" -> "야간_취식"
                filename = os.path.basename(cond_path)
                parts = filename.split('_')
                prompt_info = "conditional scene"
                if len(parts) > 2:
                    prompt_info = " ".join(parts[2:-1]) # Get the parts between index and angle

                conditional_images.append({
                    "image_base64": fig_to_base64(cond_path),
                    "prompt": f"Conditional rendering for '{prompt_info}' at the '{festival_name}' festival."
                })
                print(f"[Rendering] Success! Conditional image {i+1} generated at {cond_path}")

        if not representative_image and not conditional_images:
             raise HTTPException(status_code=500, detail="Failed to generate any images.")

        result = {
            "representative_image": representative_image,
            "conditional_images": conditional_images
        }

        # 캐시 저장
        save_to_cache(result, "render", festival_name=festival_name)
        return result

    except Exception as e:
        print(f"[Rendering] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/course/validate")
async def validate_course(request: CourseValidationRequest):
    """Validate and optimize a travel course"""
    try:
        # 캐시 확인 (course를 JSON 문자열로 변환하여 캐시 키 생성)
        import json
        course_key = json.dumps(request.course, sort_keys=True, ensure_ascii=False)
        cached_data = load_from_cache("course_validate", course=course_key, duration=request.duration)
        if cached_data:
            return cached_data

        state = {
            "course": request.course,
            "duration": request.duration,
            "validation_result": "",
        }

        result_state = course_validation_graph.invoke(state)

        result = {"validation_result": result_state.get("validation_result", "")}

        # 캐시 저장
        save_to_cache(result, "course_validate", course=course_key, duration=request.duration)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/nearby/search")
async def search_nearby(request: NearbySearchRequest):
    """Search for nearby facilities, courses, and festivals"""
    try:
        state = {
            "search_type": "nearby_search",
            "latitude": request.latitude,
            "longitude": request.longitude,
            "radius": request.radius,
            "current_festival_id": request.current_festival_id,
            "recommended_facilities": None,
            "recommended_courses": None,
            "recommended_festivals": None,
        }

        result_state = db_search_graph.invoke(state)

        return {
            "facilities": result_state.get("recommended_facilities", []),
            "courses": result_state.get("recommended_courses", []),
            "festivals": result_state.get("recommended_festivals", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/assets/{asset_type}/{filename}")
async def get_asset(asset_type: str, filename: str):
    """Serve local asset files (icons, images)"""
    try:
        asset_path = os.path.join(script_dir, asset_type, filename)
        if not os.path.exists(asset_path):
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(asset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MVP Feedback & Analytics Endpoints
# ============================================================================

class FeedbackSubmission(BaseModel):
    page_url: str
    festival_name: Optional[str] = None
    rating: int  # 1 (thumbs down) or 5 (thumbs up)
    comment: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


class FeatureRatingSubmission(BaseModel):
    festival_name: str
    feature_name: str  # e.g., "sentiment_analysis", "wordcloud", "ai_rendering"
    rating: int  # 1-5 stars
    session_id: Optional[str] = None


class UserEventSubmission(BaseModel):
    event_category: str  # e.g., "Festival", "AI", "Course"
    event_action: str  # e.g., "SearchExecuted", "SentimentAnalysisClicked"
    event_label: Optional[str] = None  # e.g., festival name
    session_id: Optional[str] = None
    page_url: Optional[str] = None
    user_id: Optional[int] = None  # Authenticated user ID
    guest_id: Optional[str] = None  # Guest identifier (e.g., "Guest1")
    username: Optional[str] = None  # Username for authenticated users


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """Submit general page feedback (A: Simple Feedback Form)"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                """
                INSERT INTO feedback (page_url, festival_name, rating, comment, user_agent, session_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    feedback.page_url,
                    feedback.festival_name,
                    feedback.rating,
                    feedback.comment,
                    feedback.user_agent,
                    feedback.session_id,
                ),
            )
            result = cursor.fetchone()
            conn.commit()
            feedback_id = result["id"]
        finally:
            cursor.close()
            release_connection(conn)
        return {"success": True, "id": feedback_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feature-rating")
async def submit_feature_rating(rating: FeatureRatingSubmission):
    """Submit feature-specific satisfaction rating (B: Feature Satisfaction Survey)"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                """
                INSERT INTO feature_ratings (festival_name, feature_name, rating, session_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (rating.festival_name, rating.feature_name, rating.rating, rating.session_id),
            )
            result = cursor.fetchone()
            conn.commit()
            rating_id = result["id"]
        finally:
            cursor.close()
            release_connection(conn)
        return {"success": True, "id": rating_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics/event")
async def track_event(event: UserEventSubmission):
    """Track user behavior events (C: User Behavior Tracking)"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                """
                INSERT INTO user_events (event_category, event_action, event_label, session_id, page_url, user_id, guest_id, username)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    event.event_category,
                    event.event_action,
                    event.event_label,
                    event.session_id,
                    event.page_url,
                    event.user_id,
                    event.guest_id,
                    event.username,
                ),
            )
            result = cursor.fetchone()
            conn.commit()
            event_id = result["id"]
        finally:
            cursor.close()
            release_connection(conn)
        return {"success": True, "id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/feedback")
async def get_all_feedback():
    """Admin: Get all feedback submissions"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                """
                SELECT id, page_url, festival_name, rating, comment, timestamp, user_agent, session_id
                FROM feedback
                ORDER BY timestamp DESC
                """
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
            release_connection(conn)

        feedback_list = []
        for row in rows:
            feedback_list.append({
                "id": row["id"],
                "page_url": row["page_url"],
                "festival_name": row["festival_name"],
                "rating": row["rating"],
                "comment": row["comment"],
                "timestamp": str(row["timestamp"]) if row["timestamp"] else None,
                "user_agent": row["user_agent"],
                "session_id": row["session_id"],
            })
        return {"feedback": feedback_list, "total": len(feedback_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/feature-ratings")
async def get_all_feature_ratings():
    """Admin: Get all feature ratings"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                """
                SELECT feature_name, AVG(rating) as avg_rating, COUNT(*) as count
                FROM feature_ratings
                GROUP BY feature_name
                ORDER BY avg_rating DESC
                """
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
            release_connection(conn)

        ratings_list = []
        for row in rows:
            ratings_list.append({
                "feature_name": row["feature_name"],
                "avg_rating": round(float(row["avg_rating"]), 2) if row["avg_rating"] else 0,
                "count": row["count"],
            })
        return {"feature_ratings": ratings_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics")
async def get_analytics_summary():
    """Admin: Get analytics summary"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)

        try:
            # Event statistics
            cursor.execute(
                """
                SELECT event_category, event_action, COUNT(*) as count
                FROM user_events
                GROUP BY event_category, event_action
                ORDER BY count DESC
                LIMIT 20
                """
            )
            events = cursor.fetchall()

            # Popular festivals
            cursor.execute(
                """
                SELECT event_label, COUNT(*) as count
                FROM user_events
                WHERE event_category = 'Festival' AND event_label IS NOT NULL
                GROUP BY event_label
                ORDER BY count DESC
                LIMIT 10
                """
            )
            popular_festivals = cursor.fetchall()

            # Total counts
            cursor.execute("SELECT COUNT(*) as cnt FROM feedback")
            total_feedback = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM feature_ratings")
            total_ratings = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM user_events")
            total_events = cursor.fetchone()["cnt"]
        finally:
            cursor.close()
            release_connection(conn)

        return {
            "summary": {
                "total_feedback": total_feedback,
                "total_ratings": total_ratings,
                "total_events": total_events,
            },
            "top_events": [
                {"category": row["event_category"], "action": row["event_action"], "count": row["count"]} for row in events
            ],
            "popular_festivals": [
                {"festival_name": row["event_label"], "views": row["count"]} for row in popular_festivals
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/qna")
async def get_qna_statistics():
    """Admin: Get Q&A statistics for dashboard"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)

        try:
            # Total counts
            cursor.execute("SELECT COUNT(*) as cnt FROM questions")
            total_questions = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM answers")
            total_answers = cursor.fetchone()["cnt"]

            # Unanswered questions count
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM questions q
                WHERE NOT EXISTS (SELECT 1 FROM answers a WHERE a.question_id = q.id)
            """)
            unanswered_questions = cursor.fetchone()["cnt"]

            # Questions by festival (top 10)
            cursor.execute("""
                SELECT festival_name, COUNT(*) as count
                FROM questions
                GROUP BY festival_name
                ORDER BY count DESC
                LIMIT 10
            """)
            questions_by_festival = cursor.fetchall()

            # Recent questions (latest 10)
            cursor.execute("""
                SELECT
                    q.id, q.festival_name, q.title, q.created_at, q.views,
                    u.username,
                    (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
                FROM questions q
                JOIN users u ON q.user_id = u.id
                ORDER BY q.created_at DESC
                LIMIT 10
            """)
            recent_questions = cursor.fetchall()

            # Recent answers (latest 10)
            cursor.execute("""
                SELECT
                    a.id, a.content, a.created_at, a.is_accepted,
                    u.username, u.role,
                    q.title as question_title, q.festival_name
                FROM answers a
                JOIN users u ON a.user_id = u.id
                JOIN questions q ON a.question_id = q.id
                ORDER BY a.created_at DESC
                LIMIT 10
            """)
            recent_answers = cursor.fetchall()
        finally:
            cursor.close()
            release_connection(conn)

        return {
            "summary": {
                "total_questions": total_questions,
                "total_answers": total_answers,
                "unanswered_questions": unanswered_questions,
                "answer_rate": round((total_answers / total_questions * 100), 1) if total_questions > 0 else 0,
            },
            "questions_by_festival": [
                {"festival_name": row["festival_name"], "count": row["count"]} for row in questions_by_festival
            ],
            "recent_questions": [
                {
                    "id": row["id"],
                    "festival_name": row["festival_name"],
                    "title": row["title"],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                    "views": row["views"],
                    "author": row["username"],
                    "answer_count": row["answer_count"],
                }
                for row in recent_questions
            ],
            "recent_answers": [
                {
                    "id": row["id"],
                    "content": row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                    "is_accepted": bool(row["is_accepted"]),
                    "author": row["username"],
                    "author_role": row["role"],
                    "question_title": row["question_title"],
                    "festival_name": row["festival_name"],
                }
                for row in recent_answers
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Q&A Endpoints
# ============================================================================

class QuestionCreate(BaseModel):
    title: str
    content: str


class QuestionUpdate(BaseModel):
    title: str
    content: str


class AnswerCreate(BaseModel):
    content: str


class AnswerUpdate(BaseModel):
    content: str


@app.get("/api/qna/questions")
async def get_all_questions(limit: int = 50, offset: int = 0):
    """Get all questions (paginated)"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)

        try:
            cursor.execute(
                """
                SELECT
                    q.id, q.title, q.content, q.views, q.created_at, q.updated_at,
                    u.username, u.full_name,
                    (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
                FROM questions q
                JOIN users u ON q.user_id = u.id
                ORDER BY q.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()

            cursor.execute("SELECT COUNT(*) as cnt FROM questions")
            total = cursor.fetchone()["cnt"]

            questions = []
            for row in rows:
                questions.append({
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                    "views": row["views"],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                    "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
                    "author": {
                        "username": row["username"],
                        "full_name": row["full_name"],
                    },
                    "answer_count": row["answer_count"],
                })

            return {"questions": questions, "total": total}
        finally:
            cursor.close()
            release_connection(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qna/questions/{question_id}")
async def get_question(question_id: int):
    """Get a single question with its answers"""
    from src.application.services.qna_service import get_question_by_id

    result = get_question_by_id(question_id)
    if not result:
        raise HTTPException(status_code=404, detail="질문을 찾을 수 없습니다")
    return result


@app.post("/api/qna/questions")
async def create_question(
    question: QuestionCreate,
    authorization: str = Header(None)
):
    """Create a new question (requires authentication)"""
    from src.application.services.qna_service import create_question as create_q

    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

    if not question.title.strip() or not question.content.strip():
        raise HTTPException(status_code=400, detail="제목과 내용을 입력해주세요")

    question_id = create_q(
        festival_name="일반",  # Default category for general questions
        user_id=user["user_id"],
        title=question.title.strip(),
        content=question.content.strip()
    )

    return {"id": question_id, "message": "질문이 등록되었습니다"}


@app.put("/api/qna/questions/{question_id}")
async def update_question(
    question_id: int,
    question: QuestionUpdate,
    authorization: str = Header(None)
):
    """Update a question (only by author)"""
    from src.application.services.qna_service import update_question as update_q, get_question_by_id

    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

    existing = get_question_by_id(question_id)
    if not existing:
        raise HTTPException(status_code=404, detail="질문을 찾을 수 없습니다")

    if existing["user_id"] != user["user_id"] and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="수정 권한이 없습니다")

    update_q(question_id, question.title.strip(), question.content.strip())
    return {"message": "질문이 수정되었습니다"}


@app.delete("/api/qna/questions/{question_id}")
async def delete_question(
    question_id: int,
    authorization: str = Header(None)
):
    """Delete a question (only by author or admin)"""
    from src.application.services.qna_service import delete_question as delete_q, get_question_by_id

    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

    existing = get_question_by_id(question_id)
    if not existing:
        raise HTTPException(status_code=404, detail="질문을 찾을 수 없습니다")

    if existing["user_id"] != user["user_id"] and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="삭제 권한이 없습니다")

    delete_q(question_id)
    return {"message": "질문이 삭제되었습니다"}


@app.post("/api/qna/questions/{question_id}/answers")
async def create_answer(
    question_id: int,
    answer: AnswerCreate,
    authorization: str = Header(None)
):
    """Create an answer for a question (admin only)"""
    from src.application.services.qna_service import create_answer as create_a, get_question_by_id

    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="관리자만 답변을 작성할 수 있습니다")

    existing = get_question_by_id(question_id)
    if not existing:
        raise HTTPException(status_code=404, detail="질문을 찾을 수 없습니다")

    if not answer.content.strip():
        raise HTTPException(status_code=400, detail="답변 내용을 입력해주세요")

    answer_id = create_a(question_id, user["user_id"], answer.content.strip())
    return {"id": answer_id, "message": "답변이 등록되었습니다"}


@app.put("/api/qna/answers/{answer_id}")
async def update_answer(
    answer_id: int,
    answer: AnswerUpdate,
    authorization: str = Header(None)
):
    """Update an answer (only by author)"""
    from src.application.services.qna_service import update_answer as update_a

    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

    # Get answer to check ownership
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute("SELECT user_id FROM answers WHERE id = %s", (answer_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="답변을 찾을 수 없습니다")
        if row["user_id"] != user["user_id"] and user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="수정 권한이 없습니다")
    finally:
        cursor.close()
        release_connection(conn)

    update_a(answer_id, answer.content.strip())
    return {"message": "답변이 수정되었습니다"}


@app.delete("/api/qna/answers/{answer_id}")
async def delete_answer(
    answer_id: int,
    authorization: str = Header(None)
):
    """Delete an answer (only by author or admin)"""
    from src.application.services.qna_service import delete_answer as delete_a

    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

    # Get answer to check ownership
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute("SELECT user_id FROM answers WHERE id = %s", (answer_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="답변을 찾을 수 없습니다")
        if row["user_id"] != user["user_id"] and user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="삭제 권한이 없습니다")
    finally:
        cursor.close()
        release_connection(conn)

    delete_a(answer_id)
    return {"message": "답변이 삭제되었습니다"}


@app.post("/api/qna/answers/{answer_id}/accept")
async def accept_answer(
    answer_id: int,
    authorization: str = Header(None)
):
    """Accept an answer (only by question author)"""
    from src.application.services.qna_service import accept_answer as accept_a

    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")

    # Get answer and question to check ownership
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute(
            """
            SELECT a.question_id, q.user_id as question_author_id
            FROM answers a
            JOIN questions q ON a.question_id = q.id
            WHERE a.id = %s
            """,
            (answer_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="답변을 찾을 수 없습니다")
        if row["question_author_id"] != user["user_id"] and user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="채택 권한이 없습니다")
        question_id = row["question_id"]
    finally:
        cursor.close()
        release_connection(conn)

    accept_a(answer_id, question_id)
    return {"message": "답변이 채택되었습니다"}


# ============================================================================
# Authentication & User Management Endpoints
# ============================================================================

class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


@app.post("/api/auth/register")
async def register(user: UserRegister):
    """Register a new user"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)

        try:
            # Check if username or email already exists
            cursor.execute(
                "SELECT id FROM users WHERE username = %s OR email = %s",
                (user.username, user.email),
            )
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Username or email already exists")

            # Hash password and create user
            password_hash = hash_password(user.password)
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user.username, user.email, password_hash, user.full_name, "user"),
            )
            result = cursor.fetchone()
            conn.commit()
            user_id = result["id"]

            # Get created user
            cursor.execute(
                "SELECT id, username, email, full_name, role, created_at FROM users WHERE id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            release_connection(conn)

        # Create access token + refresh token
        access_token = create_access_token({
            "user_id": row["id"],
            "username": row["username"],
            "role": row["role"],
        })
        refresh_token, refresh_expires_at = create_refresh_token(row["id"])

        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 900,  # 15 minutes in seconds
            "user": {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "created_at": str(row["created_at"]) if row["created_at"] else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    """Login a user"""
    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)

        try:
            # Get user by username
            cursor.execute(
                """
                SELECT id, username, email, password_hash, full_name, role, created_at
                FROM users
                WHERE username = %s
                """,
                (credentials.username,),
            )
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=401, detail="Invalid username or password")

            # Verify password
            if not verify_password(credentials.password, row["password_hash"]):
                raise HTTPException(status_code=401, detail="Invalid username or password")

            # Update last login
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                (row["id"],),
            )
            conn.commit()
        finally:
            cursor.close()
            release_connection(conn)

        # Create access token + refresh token
        access_token = create_access_token({
            "user_id": row["id"],
            "username": row["username"],
            "role": row["role"],
        })
        refresh_token, refresh_expires_at = create_refresh_token(row["id"])

        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 900,  # 15 minutes in seconds
            "user": {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "created_at": str(row["created_at"]) if row["created_at"] else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/me")
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current authenticated user"""
    user_payload = require_auth(authorization)

    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                """
                SELECT id, username, email, full_name, role, created_at, last_login
                FROM users
                WHERE id = %s
                """,
                (user_payload["user_id"],),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            release_connection(conn)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "full_name": row["full_name"],
            "role": row["role"],
            "created_at": str(row["created_at"]) if row["created_at"] else None,
            "last_login": str(row["last_login"]) if row["last_login"] else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/refresh")
async def refresh_access_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        user_id = verify_refresh_token(request.refresh_token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

        # Get user info
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                "SELECT id, username, role FROM users WHERE id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            release_connection(conn)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        # Create new access token
        access_token = create_access_token({
            "user_id": row["id"],
            "username": row["username"],
            "role": row["role"],
        })

        return {
            "success": True,
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 900,  # 15 minutes in seconds
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/logout")
async def logout(request: LogoutRequest):
    """Logout by revoking refresh token"""
    try:
        success = revoke_refresh_token(request.refresh_token)
        return {"success": success, "message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/logout-all")
async def logout_all_devices(authorization: Optional[str] = Header(None)):
    """Logout from all devices by revoking all user's refresh tokens"""
    user_payload = require_auth(authorization)

    try:
        success = revoke_all_user_tokens(user_payload["user_id"])
        return {"success": success, "message": "Logged out from all devices"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/auth/profile")
async def update_profile(
    update: UserUpdate,
    authorization: Optional[str] = Header(None)
):
    """Update user profile"""
    user_payload = require_auth(authorization)

    try:
        conn = get_db_connection()
        cursor = get_cursor(conn)

        try:
            # Update user (build query with numbered placeholders for PostgreSQL)
            updates = []
            params = []
            param_idx = 1
            if update.full_name is not None:
                updates.append(f"full_name = ${param_idx}")
                params.append(update.full_name)
                param_idx += 1
            if update.email is not None:
                updates.append(f"email = ${param_idx}")
                params.append(update.email)
                param_idx += 1

            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")

            params.append(user_payload["user_id"])
            # Use %s placeholders for psycopg2
            query = f"UPDATE users SET {', '.join([u.replace('$' + str(i+1), '%s') for i, u in enumerate(updates)])} WHERE id = %s"
            cursor.execute(query, params)
            conn.commit()

            # Get updated user
            cursor.execute(
                "SELECT id, username, email, full_name, role, created_at FROM users WHERE id = %s",
                (user_payload["user_id"],),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            release_connection(conn)

        return {
            "success": True,
            "user": {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "created_at": str(row["created_at"]) if row["created_at"] else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/google")
async def google_login(request: dict):
    """Login or register with Google OAuth"""
    try:
        from src.application.services.auth_service import verify_google_token, get_or_create_google_user

        # Get Google ID token from request
        id_token = request.get("credential")
        if not id_token:
            raise HTTPException(status_code=400, detail="No credential provided")

        # Verify token with Google
        google_user_info = await verify_google_token(id_token)
        if not google_user_info:
            raise HTTPException(status_code=401, detail="Invalid Google token")

        # Get or create user
        user = get_or_create_google_user(google_user_info)

        # Create tokens
        access_token = create_access_token({
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"]
        })
        refresh_token, refresh_expires_at = create_refresh_token(user["id"])

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 900,  # 15 minutes in seconds
            "user": user
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/auth/account")
async def delete_account(authorization: Optional[str] = Header(None)):
    """
    Delete user account and all associated data
    Works for both local and Google OAuth accounts
    """
    user_payload = require_auth(authorization)

    try:
        from src.application.services.auth_service import delete_user_account

        success = delete_user_account(user_payload["user_id"])
        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "success": True,
            "message": "Account successfully deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Q&A Board Endpoints
# ============================================================================

class QuestionCreate(BaseModel):
    festival_name: str
    title: str
    content: str


class QuestionUpdate(BaseModel):
    title: str
    content: str


class AnswerCreate(BaseModel):
    content: str


class AnswerUpdate(BaseModel):
    content: str


@app.get("/api/qna/festival/{festival_name}")
async def get_festival_questions(festival_name: str, limit: int = 50, offset: int = 0):
    """Get all questions for a festival"""
    try:
        questions = qna_service.get_questions_for_festival(festival_name, limit, offset)
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qna/question/{question_id}")
async def get_question(question_id: int):
    """Get a question with its answers"""
    try:
        question = qna_service.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        return question
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qna/question")
async def create_question(
    question: QuestionCreate,
    authorization: Optional[str] = Header(None)
):
    """Create a new question (requires authentication)"""
    user_payload = require_auth(authorization)

    try:
        question_id = qna_service.create_question(
            question.festival_name,
            user_payload["user_id"],
            question.title,
            question.content,
        )
        return {"success": True, "question_id": question_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/qna/question/{question_id}")
async def update_question(
    question_id: int,
    update: QuestionUpdate,
    authorization: Optional[str] = Header(None)
):
    """Update a question (author only)"""
    user_payload = require_auth(authorization)

    try:
        # Check ownership
        question = qna_service.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        if question["user_id"] != user_payload["user_id"]:
            raise HTTPException(status_code=403, detail="You can only edit your own questions")

        success = qna_service.update_question(question_id, update.title, update.content)
        if not success:
            raise HTTPException(status_code=404, detail="Question not found")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/qna/question/{question_id}")
async def delete_question(
    question_id: int,
    authorization: Optional[str] = Header(None)
):
    """Delete a question (author or admin)"""
    user_payload = require_auth(authorization)

    try:
        # Check ownership or admin
        question = qna_service.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        if question["user_id"] != user_payload["user_id"] and user_payload["role"] != "admin":
            raise HTTPException(status_code=403, detail="Only author or admin can delete")

        success = qna_service.delete_question(question_id)
        if not success:
            raise HTTPException(status_code=404, detail="Question not found")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qna/question/{question_id}/answer")
async def create_answer(
    question_id: int,
    answer: AnswerCreate,
    authorization: Optional[str] = Header(None)
):
    """Create an answer to a question (admin only)"""
    user_payload = require_auth(authorization)

    if user_payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="관리자만 답변을 작성할 수 있습니다")

    try:
        answer_id = qna_service.create_answer(
            question_id,
            user_payload["user_id"],
            answer.content,
        )
        return {"success": True, "answer_id": answer_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/qna/answer/{answer_id}")
async def update_answer(
    answer_id: int,
    update: AnswerUpdate,
    authorization: Optional[str] = Header(None)
):
    """Update an answer (author only)"""
    user_payload = require_auth(authorization)

    try:
        # Check ownership
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute("SELECT user_id FROM answers WHERE id = %s", (answer_id,))
            row = cursor.fetchone()
        finally:
            cursor.close()
            release_connection(conn)

        if not row:
            raise HTTPException(status_code=404, detail="Answer not found")

        if row["user_id"] != user_payload["user_id"]:
            raise HTTPException(status_code=403, detail="You can only edit your own answers")

        success = qna_service.update_answer(answer_id, update.content)
        if not success:
            raise HTTPException(status_code=404, detail="Answer not found")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/qna/answer/{answer_id}")
async def delete_answer(
    answer_id: int,
    authorization: Optional[str] = Header(None)
):
    """Delete an answer (author or admin)"""
    user_payload = require_auth(authorization)

    try:
        # Check ownership or admin
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute("SELECT user_id FROM answers WHERE id = %s", (answer_id,))
            row = cursor.fetchone()
        finally:
            cursor.close()
            release_connection(conn)

        if not row:
            raise HTTPException(status_code=404, detail="Answer not found")

        if row["user_id"] != user_payload["user_id"] and user_payload["role"] != "admin":
            raise HTTPException(status_code=403, detail="Only author or admin can delete")

        success = qna_service.delete_answer(answer_id)
        if not success:
            raise HTTPException(status_code=404, detail="Answer not found")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qna/answer/{answer_id}/accept")
async def accept_answer(
    answer_id: int,
    authorization: Optional[str] = Header(None)
):
    """Accept an answer (question author only)"""
    user_payload = require_auth(authorization)

    try:
        # Get answer and question
        conn = get_db_connection()
        cursor = get_cursor(conn)
        try:
            cursor.execute(
                """
                SELECT a.question_id, q.user_id
                FROM answers a
                JOIN questions q ON a.question_id = q.id
                WHERE a.id = %s
                """,
                (answer_id,),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            release_connection(conn)

        if not row:
            raise HTTPException(status_code=404, detail="Answer not found")

        question_id = row["question_id"]
        question_author_id = row["user_id"]

        if question_author_id != user_payload["user_id"]:
            raise HTTPException(status_code=403, detail="Only question author can accept answers")

        success = qna_service.accept_answer(answer_id, question_id)
        if not success:
            raise HTTPException(status_code=404, detail="Answer not found")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/questions")
async def get_my_questions(authorization: Optional[str] = Header(None)):
    """Get current user's questions"""
    user_payload = require_auth(authorization)

    try:
        questions = qna_service.get_user_questions(user_payload["user_id"])
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/answers")
async def get_my_answers(authorization: Optional[str] = Header(None)):
    """Get current user's answers"""
    user_payload = require_auth(authorization)

    try:
        answers = qna_service.get_user_answers(user_payload["user_id"])
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def get_local_icon_path(festival_name: str) -> Optional[str]:
    """Get local icon path for a festival"""
    base_dir = os.path.join(script_dir, "best_images_and_icons", "icons")
    if not os.path.exists(base_dir):
        return None
    icon_filename = ICON_MAP.get(festival_name)
    if icon_filename:
        file_path = os.path.join(base_dir, icon_filename)
        if os.path.exists(file_path):
            return f"/api/assets/best_images_and_icons/icons/{icon_filename}"
    return None


def get_local_best_image_path(festival_name: str) -> Optional[str]:
    """Get local best image path for a festival"""
    base_dir = os.path.join(script_dir, "best_images_and_icons", "best_images")
    if not os.path.exists(base_dir):
        return None
    image_filename = BEST_IMAGES_MAP.get(festival_name)
    if image_filename:
        file_path = os.path.join(base_dir, image_filename)
        if os.path.exists(file_path):
            return f"/api/assets/best_images_and_icons/best_images/{image_filename}"
    return None


# ============================================================================
# Async Task API (비동기 작업 - 진행률 추적)
# ============================================================================

class AsyncTaskRequest(BaseModel):
    festival_name: str
    num_reviews: int = 10


class AsyncRankingRequest(BaseModel):
    festivals: List[str]
    num_reviews: int = 5
    top_n: int = 3


@app.get("/api/system/status")
async def get_system_status():
    """시스템 상태 확인 (Redis, Celery 연결 상태)"""
    status = {
        "redis_available": REDIS_AVAILABLE,
        "cache_stats": get_cache_stats(),
        "celery_available": False,
    }

    # Celery 상태 확인
    try:
        from src.celery.app import celery_app
        inspect = celery_app.control.inspect()
        active = inspect.active()
        status["celery_available"] = active is not None
        status["celery_workers"] = list(active.keys()) if active else []
    except Exception as e:
        status["celery_error"] = str(e)

    return status


@app.post("/api/async/sentiment/start")
async def start_async_sentiment_analysis(request: AsyncTaskRequest):
    """비동기 감성 분석 시작"""
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="비동기 작업을 사용하려면 Redis가 필요합니다. 동기 API를 사용해주세요."
        )

    try:
        from src.celery.tasks import analyze_sentiment_task

        # Celery 작업 시작
        task = analyze_sentiment_task.delay(
            request.festival_name,
            request.num_reviews
        )

        return {
            "task_id": task.id,
            "status": "started",
            "message": f"'{request.festival_name}' 감성 분석이 시작되었습니다.",
            "poll_url": f"/api/async/task/{task.id}/progress",
            "result_url": f"/api/async/task/{task.id}/result",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/async/ranking/start")
async def start_async_ranking_analysis(request: AsyncRankingRequest):
    """비동기 랭킹 분석 시작"""
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="비동기 작업을 사용하려면 Redis가 필요합니다. 동기 API를 사용해주세요."
        )

    try:
        from src.celery.tasks import analyze_ranking_task

        task = analyze_ranking_task.delay(
            request.festivals,
            request.num_reviews,
            request.top_n
        )

        return {
            "task_id": task.id,
            "status": "started",
            "message": f"{len(request.festivals)}개 축제 랭킹 분석이 시작되었습니다.",
            "poll_url": f"/api/async/task/{task.id}/progress",
            "result_url": f"/api/async/task/{task.id}/result",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/async/wordcloud/start")
async def start_async_wordcloud(request: AsyncTaskRequest):
    """비동기 워드클라우드 생성 시작"""
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="비동기 작업을 사용하려면 Redis가 필요합니다. 동기 API를 사용해주세요."
        )

    try:
        from src.celery.tasks import generate_wordcloud_task

        task = generate_wordcloud_task.delay(
            request.festival_name,
            request.num_reviews
        )

        return {
            "task_id": task.id,
            "status": "started",
            "message": f"'{request.festival_name}' 워드클라우드 생성이 시작되었습니다.",
            "poll_url": f"/api/async/task/{task.id}/progress",
            "result_url": f"/api/async/task/{task.id}/result",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/async/task/{task_id}/progress")
async def get_async_task_progress(task_id: str):
    """비동기 작업 진행률 조회"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis가 필요합니다.")

    progress_data = get_task_progress(task_id)

    if not progress_data:
        # Celery에서 직접 상태 확인
        try:
            from src.celery.app import celery_app
            from celery.result import AsyncResult

            result = AsyncResult(task_id, app=celery_app)

            if result.state == "PENDING":
                return {
                    "task_id": task_id,
                    "progress": 0,
                    "status": "pending",
                    "message": "작업이 대기 중입니다...",
                }
            elif result.state == "STARTED":
                return {
                    "task_id": task_id,
                    "progress": 5,
                    "status": "running",
                    "message": "작업이 시작되었습니다...",
                }
            elif result.state == "SUCCESS":
                return {
                    "task_id": task_id,
                    "progress": 100,
                    "status": "completed",
                    "message": "작업이 완료되었습니다.",
                }
            elif result.state == "FAILURE":
                return {
                    "task_id": task_id,
                    "progress": 0,
                    "status": "failed",
                    "message": str(result.result),
                }
            else:
                return {
                    "task_id": task_id,
                    "progress": 0,
                    "status": result.state.lower(),
                    "message": "작업 상태 확인 중...",
                }
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {task_id}")

    return {
        "task_id": task_id,
        **progress_data
    }


@app.get("/api/async/task/{task_id}/result")
async def get_async_task_result(task_id: str):
    """비동기 작업 결과 조회"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis가 필요합니다.")

    # 먼저 Redis에서 결과 확인
    result_data = get_task_result(task_id)
    if result_data:
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result_data,
        }

    # Celery에서 결과 확인
    try:
        from src.celery.app import celery_app
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=celery_app)

        if result.ready():
            if result.successful():
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result.result,
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(result.result),
                }
        else:
            # 아직 완료되지 않음
            progress = get_task_progress(task_id)
            return {
                "task_id": task_id,
                "status": "running",
                "progress": progress.get("progress", 0) if progress else 0,
                "message": "작업이 아직 진행 중입니다. 잠시 후 다시 시도해주세요.",
            }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {task_id}")


# ============================================================================
# Admin: Precache Trigger Endpoints (for Cloud Scheduler)
# ============================================================================

class PrecacheTriggerRequest(BaseModel):
    """Precache 트리거 요청"""
    active_only: bool = True
    limit: Optional[int] = None
    num_reviews: int = 50
    api_key: str  # 보안을 위한 API 키


@app.post("/api/admin/trigger-precache")
async def trigger_precache(request: PrecacheTriggerRequest):
    """
    Precache 작업을 트리거합니다.
    Cloud Scheduler에서 호출하도록 설계되었습니다.

    보안: api_key가 환경변수 PRECACHE_API_KEY와 일치해야 합니다.
    """
    import os

    # API 키 검증
    expected_key = os.environ.get("PRECACHE_API_KEY", "")
    if not expected_key or request.api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Redis 연결 확인
    if not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Redis가 연결되지 않아 비동기 작업을 실행할 수 없습니다."
        )

    try:
        from src.celery.precache import precache_all_festivals

        # Celery 태스크 실행
        task = precache_all_festivals.delay(
            num_reviews=request.num_reviews,
            skip_if_cached=True,
            active_only=request.active_only,
            limit=request.limit
        )

        return {
            "status": "started",
            "task_id": task.id,
            "message": f"Precache 작업이 시작되었습니다. (active_only={request.active_only})",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Precache 시작 실패: {str(e)}")


@app.get("/api/admin/precache-status/{task_id}")
async def get_precache_status(task_id: str, api_key: str = Query(...)):
    """Precache 작업 상태를 조회합니다."""
    import os

    # API 키 검증
    expected_key = os.environ.get("PRECACHE_API_KEY", "")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    progress = get_task_progress(task_id)
    result = get_task_result(task_id)

    if result:
        return {"task_id": task_id, "status": "completed", "result": result}
    elif progress:
        return {"task_id": task_id, **progress}
    else:
        return {"task_id": task_id, "status": "unknown"}


if __name__ == "__main__":
    import uvicorn
    import multiprocessing

    # 워커 수 계산 (CPU 코어 수 기반, 최소 2, 최대 4)
    # Playwright를 sync_api + ThreadPoolExecutor로 변경하여 Windows에서도 멀티 워커 지원
    workers = min(max(multiprocessing.cpu_count(), 2), 4)

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        timeout_keep_alive=30,
        access_log=True,
    )
