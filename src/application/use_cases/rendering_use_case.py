import os
import requests
import pandas as pd
import re
import asyncio
import traceback

# 'from google import generativeai' 대신 표준 방식 사용
import google.generativeai as genai
from google.generativeai import types

from PIL import Image
import io

# 설정 및 LLM 클라이언트 임포트
from src.infrastructure.config.settings import get_google_maps_key, get_google_api_key
from src.infrastructure.llm_client import get_llm_client


class RenderingUseCase:
    def __init__(self, df_split: pd.DataFrame, df_camera: pd.DataFrame):
        """
        AI 렌더링 유스케이스를 초기화합니다.

        Args:
            df_split: 'festival_condition_split.csv'에서 로드된 DataFrame
            df_camera: 'festivals_camera_angle_all.csv'에서 로드된 DataFrame
        """
        # [참고] self.client는 gemini-pro-vision이므로 이미지 생성에 사용되지 않습니다.
        # _generate_image 함수는 노트북 코드를 따라 genai.GenerativeModel을 직접 사용합니다.
        self.client = get_llm_client(model="gemini-pro-vision", temperature=0.8)

        try:
            self.maps_api_key = get_google_maps_key()
        except ValueError as e:
            print(f"Warning: {e}. 위성 지도 이미지를 사용할 수 없습니다.")
            self.maps_api_key = None

        self.df_split = df_split
        self.df_camera = df_camera

        # 데이터가 정상적으로 로드되었는지 확인
        if self.df_split.empty or self.df_camera.empty:
            print(
                "!!! CRITICAL ERROR: RenderingUseCase가 빈 CSV 데이터로 초기화되었습니다."
            )
        else:
            print(
                f"[RenderingUseCase] Initialized with {len(df_split)} split rows and {len(df_camera)} camera rows."
            )

    async def _generate_image(
        self, prompt, save_dir, filename, ref_img_bytes=None, retries=2
    ):
        """Gemini를 호출하여 이미지를 생성하고 저장합니다."""
        for attempt in range(retries):
            try:
                parts = [prompt]
                if ref_img_bytes:
                    # 위성 지도를 PNG로 변환하여 참조 이미지로 사용
                    img = Image.open(io.BytesIO(ref_img_bytes)).convert("RGB")
                    png_buffer = io.BytesIO()
                    img.save(png_buffer, format="PNG")

                    # 모델 형식에 맞게 parts 구성
                    parts.append(Image.open(png_buffer))

                api_key = (
                    get_google_api_key()
                )  # .env의 GOOGLE_API_KEY (Vicker 키로 가정)
                genai.configure(api_key=api_key)  # genai 클라이언트 설정

                g_client = None
                g_config = None

                try:
                    # 1순위: 노트북에서 사용한 'gemini-2.5-flash-image' 시도
                    g_client = genai.GenerativeModel(
                        model_name="models/gemini-2.5-flash-image"
                    )
                    # [수정 1] 'response_modalities' 인수 제거
                    g_config = types.GenerationConfig(
                        temperature=0.8,
                    )

                    # [수정 2] .agenerate_content -> .generate_content_async
                    resp = await g_client.generate_content_async(
                        contents=parts, generation_config=g_config
                    )

                except Exception as e:
                    # 2순위: 1순위 모델이 없으면 'gemini-1.5-flash'로 대체 시도
                    print(
                        f"Warning: 'gemini-2.5-flash-image' 모델을 찾을 수 없습니다 ({e}). 'gemini-1.5-flash'로 대체합니다."
                    )

                    # [수정 3] 'gemini-1.5-flash-latest' -> 'gemini-1.5-flash'
                    g_client = genai.GenerativeModel(model_name="gemini-1.5-flash")
                    g_config = types.GenerationConfig(temperature=0.8)

                    # [수정 2] .agenerate_content -> .generate_content_async
                    resp = await g_client.generate_content_async(
                        contents=parts, generation_config=g_config
                    )

                # [노트북 코드(da609b31)의 'resp' 처리 로직을 따름]
                if resp.candidates and resp.candidates[0].content.parts:
                    for p in resp.candidates[0].content.parts:
                        if hasattr(p, "inline_data") and getattr(
                            p.inline_data, "data", None
                        ):
                            img_data = p.inline_data.data
                            os.makedirs(save_dir, exist_ok=True)
                            path = os.path.join(save_dir, filename)
                            with open(path, "wb") as f:
                                f.write(img_data)
                            print(f"✅ 이미지 저장 완료: {path}")
                            return path

                print(
                    f"⚠️ API가 이미지를 반환하지 않음: {filename} (시도 {attempt+1}/{retries})"
                )

            except Exception as e:
                print(
                    f"❌ 이미지 생성 실패 ({filename}) [시도 {attempt+1}/{retries}]: {e}"
                )
                traceback.print_exc()
                await asyncio.sleep(1)

        print(f"❌ 최종 실패: {filename}")
        return None

    async def _get_satellite_image(self, lat, lon):
        """Google Maps Static API로 위성 지도 이미지를 다운로드합니다."""
        if not self.maps_api_key or not lat or not lon:
            print("⚠️ 지도 이미지 없음 (좌표X 또는 키X)")
            return None

        map_url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom=18&size=1024x1024&scale=2&maptype=satellite&key={self.maps_api_key}"
        )
        try:
            # Gradio 앱 환경에서는 asyncio.to_thread를 사용해 동기 I/O를 비동기로 실행합니다.
            response = await asyncio.to_thread(requests.get, map_url, timeout=10)
            response.raise_for_status()
            print(f"🗺 지도 이미지 다운로드 완료: {lat},{lon}")
            return response.content
        except Exception as e:
            print(f"🗺 지도 이미지 다운로드 실패: {e}")
            return None

    async def generate_festival_renderings(self, festival_details: dict, progress=None):
        """축제 정보를 기반으로 대표 렌더링과 조건부 렌더링을 비동기로 생성합니다."""
        fest_name_raw = festival_details.get("title", "")
        lat = festival_details.get("mapy")
        lon = festival_details.get("mapx")
        address = festival_details.get("addr1", "주소 정보 없음")

        if not fest_name_raw or not lat or not lon:
            raise ValueError("축제 정보(이름, 좌표)가 부족합니다.")

        fest_name_lookup = fest_name_raw.strip()

        # 1. 위성 지도 다운로드 (비동기)
        if progress:
            progress(0.1)
        map_bytes = await self._get_satellite_image(lat, lon)

        # 2. CSV에서 렌더링 조건 조회
        row_split = self.df_split[
            self.df_split["Title"].astype(str).str.strip() == fest_name_lookup
        ]
        rows_camera = self.df_camera[
            self.df_camera["FestivalName"].astype(str).str.strip() == fest_name_lookup
        ]

        if row_split.empty or rows_camera.empty:
            print(
                f"'{fest_name_lookup}'에 대한 렌더링 조건 CSV 데이터를 찾을 수 없습니다. (Split: {len(row_split)}, Camera: {len(rows_camera)})"
            )
            # fallback: .str.contains 사용 (더 느리지만 유연함)
            if row_split.empty:
                row_split = self.df_split[
                    self.df_split["Title"]
                    .astype(str)
                    .str.contains(fest_name_lookup, case=False, na=False)
                ]
            if rows_camera.empty:
                rows_camera = self.df_camera[
                    self.df_camera["FestivalName"]
                    .astype(str)
                    .str.contains(fest_name_lookup, case=False, na=False)
                ]

            if row_split.empty or rows_camera.empty:
                raise ValueError(
                    f"'{fest_name_lookup}'에 대한 렌더링 조건 CSV 데이터를 찾을 수 없습니다."
                )

        # 3. 출력 경로 설정 (temp_img 폴더 활용)
        safe_fest_name = re.sub(r'[\\/*?"<>|:\s]+', "_", fest_name_lookup)
        FEST_OUT_DIR = os.path.join(os.getcwd(), "temp_img", safe_fest_name)
        REP_DIR = os.path.join(FEST_OUT_DIR, "1_대표렌더링")
        COND_DIR = os.path.join(FEST_OUT_DIR, "2_조건렌더링")

        generated_paths = {"representative": None, "conditional": []}
        tasks = []

        # 4. 대표 렌더링 작업 준비
        if progress:
            progress(0.3)
        row_split_first = row_split.iloc[0]
        season = str(row_split_first["condition1"]).split(",")[0].strip()
        time = str(row_split_first["condition2"]).split(",")[0].strip()

        is_night = any(k in time for k in ["밤", "야간", "심야"])
        night_addon = ""
        if is_night:
            night_addon = """
- 야간 촬영이므로 노이즈 없이 선명한 노출 유지
- 따뜻한 조명, 반사광, 촛불빛 강조
- 광원 주변은 부드러운 글로우 처리
"""
        prompt_rep = f"""
'{fest_name_lookup}'가 열리는 {address} 현장을,
전문 포토그래퍼가 DSLR 35mm f/1.8로 촬영한 고화질 홍보사진처럼 표현하세요.

요구사항:
- 사람 시점(ground-level)
- {season} {time}의 자연광, 그림자, 하늘색 반영
- 인파는 북적이지 않게, 활기 있게 배치
- 부스·음식·사람 디테일을 따뜻하고 선명하게 묘사
- 간판/한글 텍스트는 흐릿하게 처리 (깨짐 방지)
- DSLR 촬영 톤, soft HDR, warm tone 중심
{night_addon}
- 반드시 이미지 형태로 출력 (텍스트 응답 금지)
"""
        filename_rep = f"대표_{season}_{time}.png".replace(" ", "_").replace("/", "_")
        tasks.append(self._generate_image(prompt_rep, REP_DIR, filename_rep, map_bytes))

        # 5. 조건 렌더링 작업 준비
        if progress:
            progress(0.5)
        for i, cond_row in enumerate(rows_camera.itertuples()):
            cond_name = cond_row.ConditionName
            cond_desc = cond_row.ConditionDesc
            angle = cond_row.camera_angle
            angle_prompt = "거리에서 사람 눈높이로 DSLR로 촬영한 구도"
            if angle == "aerial":
                angle_prompt = "드론 항공 시점에서 부드럽게 내려다보는 구도"
            elif angle == "indoor":
                angle_prompt = "실내 조명 아래 전시장/공연 공간 구도"

            is_night_cond = any(
                kw in cond_desc for kw in ["밤", "야간", "불빛", "조명", "야시장", "빛"]
            )
            night_addon_2 = ""
            if is_night_cond:
                night_addon_2 = """
- 야간 조명 연출이므로 빛 반사와 노출 밸런스 주의
- 광원 주변은 살짝 글로우 처리
"""

            prompt_cond = f"""
한국 축제 '{fest_name_lookup}'의 '{cond_name}' 장면을,
{angle_prompt}로 포토그래퍼가 따뜻한 색감으로 촬영한 현장사진처럼 생성하세요.

🎬 장면 설명: {cond_desc}
📍 위치: {address}

요구사항:
- {angle_prompt} 시점 유지
- 인파/부스/조명/무대 디테일을 사실적으로 묘사
- DSLR 35mm f/1.8의 얕은 심도, HDR 질감
- 간판 한글은 흐릿하게 (깨짐 방지)
{night_addon_2}
- 반드시 이미지 형태로 출력 (텍스트 응답 금지)
"""
            filename_cond = f"조건_{i+1}_{cond_name}_{angle}.png".replace(
                " ", "_"
            ).replace("/", "_")
            tasks.append(
                self._generate_image(prompt_cond, COND_DIR, filename_cond, map_bytes)
            )

        # 6. 모든 이미지 생성 작업 동시 실행
        if progress:
            progress(0.6)
        results = await asyncio.gather(*tasks)

        generated_paths["representative"] = results[0]  # 첫 번째가 대표 렌더링
        generated_paths["conditional"] = [
            path for path in results[1:] if path
        ]  # 나머지가 조건 렌더링

        if progress:
            progress(1.0)
        return generated_paths
