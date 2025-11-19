import sys
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64
from io import BytesIO
import re

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import matplotlib.pyplot as plt
from PIL import Image

# Environment and Initial Setup
from src.infrastructure.config.settings import setup_environment

setup_environment()

# Import the database initializer
from src.infrastructure.persistence.database import init_db, get_db_connection

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
from application.agents.naver_review.naver_review_agent import NaverReviewAgent
from src.application.use_cases.analysis_use_case import AnalysisUseCase
from src.application.use_cases.sentiment_analysis_use_case import (
    SentimentAnalysisUseCase,
)
from src.application.use_cases.ranking_use_case import RankingUseCase
from src.application.use_cases.rendering_use_case import RenderingUseCase
from src.application.services.course_service import get_course_details_by_title
from src.application.services.facility_service import get_facility_details_by_title
from src.infrastructure.reporting.wordclouds import create_sentiment_wordclouds
from src.infrastructure.cache_manager import load_from_cache, save_to_cache

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
    ],  # React dev servers
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
    """Initialize database on startup"""
    if not os.path.exists(os.path.join(script_dir, "tour.db")):
        init_db()
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
        cursor = conn.cursor()

        festivals_data = []
        for festival_name in request.festivals:
            cursor.execute(
                """
                SELECT * FROM festivals WHERE title = ?
            """,
                (festival_name,),
            )
            row = cursor.fetchone()
            if row:
                # Convert sqlite3.Row to dict
                festival_dict = {key: row[key] for key in row.keys()}
                festivals_data.append(festival_dict)

        conn.close()

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
