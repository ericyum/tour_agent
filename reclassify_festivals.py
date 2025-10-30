import pandas as pd
import json
import os
import asyncio
import re
from src.infrastructure.llm_client import get_llm_client

FULL_CATEGORY_HIERARCHY = {
    "계절과 자연": {
        "자연경관": ["꽃 축제", "단풍 축제", "눈/얼음 축제", "바다/강 축제", "산/숲 축제"],
        "자연현상": ["별/천문 축제", "일출/일몰 축제", "기타 자연현상"],
        "생태/환경": ["생태 체험", "환경 보호", "기타 생태"]
    },
    "전통과 역사": {
        "역사문화": ["역사 재현", "문화재 야행", "역사 인물 기념"],
        "민속전통": ["민속놀이", "전통 공예/예술", "전통 의례/행사"],
        "지역유산": ["지역 고유 전통", "기타 지역유산"]
    },
    "문화와 예술": {
        "공연예술": ["음악 축제", "연극/뮤지컬", "무용/퍼포먼스"],
        "시각예술": ["미술/조각", "사진/영상", "디자인/공예"],
        "문학/미디어": ["문학/책 축제", "영화/영상 축제", "미디어 아트"],
        "복합문화": ["복합 문화 예술", "다원 예술", "기타 문화예술"]
    },
    "미식과 특산물": {
        "지역특산물": ["농산물 축제", "수산물 축제", "임산물 축제", "기타 특산물"],
        "음식문화": ["음식/요리 축제", "길거리 음식", "세계 음식"],
        "음료/주류": ["커피/차 축제", "맥주/와인 축제", "전통주 축제"]
    },
    "체험과 교육": {
        "과학/기술": ["과학 체험", "기술 교육", "발명/창의"],
        "공예/만들기": ["수공예", "도예/목공예", "DIY/만들기"],
        "농촌/생태": ["농촌 체험", "생태 교육", "자연 학습"],
        "역사/문화": ["역사 교육", "전통 문화 체험", "유적 탐방"]
    },
    "레저와 스포츠": {
        "육상스포츠": ["걷기/달리기", "등산/트레킹", "자전거/마라톤"],
        "수상스포츠": ["카약/래프팅", "서핑/요트", "낚시/해양"],
        "겨울스포츠": ["스키/스노보드", "썰매/스케이트", "빙어/얼음낚시"],
        "e스포츠/게임": ["e스포츠 대회", "보드게임/VR", "캐릭터/애니메이션"],
        "익스트림/아웃도어": ["캠핑/백패킹", "패러글라이딩/짚라인", "기타 아웃도어"]
    },
    "도시와 커뮤니티": {
        "도시이벤트": ["불꽃/빛 축제", "거리 퍼레이드", "야시장/플리마켓"],
        "지역활성화": ["지역 상권", "도시 재생", "커뮤니티 행사"],
        "축제/페스티벌": ["시민 참여", "거리 예술", "기타 도시 축제"]
    },
    "종교와 영성": {
        "불교": ["연등회", "사찰 문화", "불교 의례"],
        "기독교": ["성탄 마켓", "부활절 행사", "기독교 문화"],
        "기타종교": ["기타 종교 행사", "영성/명상", "철학/인문"]
    }
}

# 2. LLM에 한 번에 요청할 축제의 수를 상수로 정의 (API 안정성 및 토큰 제한 고려)
BATCH_SIZE = 200

# JSON 파싱 실패 시 대체 패턴
JSON_FALLBACK_PATTERN = r'\{.*\}'

async def classify_festivals_in_batch(festivals_batch: list, llm_client) -> dict:
    """
    LLM을 사용하여 축제 목록(배치)을 미리 정의된 카테고리로 분류합니다.
    """
    # LLM에 전달할 축제 목록을 간단한 형태로 변환 (제목과 개요만 포함)
    festivals_info = [{"title": f["title"], "overview": f["overview"]} for f in festivals_batch]

    prompt = f"""
    당신은 수많은 한국 축제 데이터를 명확한 기준에 따라 분류하는 데이터 분석 전문가입니다.

    [분류 기준]
    아래 정의된 대분류, 중분류, 소분류를 기준으로, 각 축제가 어떤 대분류, 중분류, 소분류에 가장 적합한지 하나씩 선택해주세요.
    만약 적합한 소분류가 없다면, 해당 중분류의 '기타' 소분류를 선택해주세요.

    <대분류, 중분류 및 소분류 목록>
    {json.dumps(FULL_CATEGORY_HIERARCHY, ensure_ascii=False, indent=2)}

    [분류할 축제 목록]
    {json.dumps(festivals_info, ensure_ascii=False, indent=2)}

    [출력 형식]
    각 축제명과 할당된 대분류, 중분류 및 소분류 카테고리를 키-값 쌍으로 가지는 단일 JSON 객체를 반환해주세요.
    다른 설명이나 추가 텍스트 없이 JSON 객체만 포함해야 합니다.

    예시:
    ```json
    {{
      "축제명1": {{"main_category": "계절과 자연", "middle_category": "자연경관", "sub_category": "꽃 축제"}},
      "축제명2": {{"main_category": "전통과 역사", "middle_category": "역사문화", "sub_category": "역사 재현"}}
    }}
    ```
    """
    try:
        response = await llm_client.ainvoke(prompt)
        # LLM 응답에서 JSON 부분만 정확히 추출
        json_match = re.search(r'```json\n(.*)\n```', response.content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
            return json.loads(json_content)
        else:
            # Fallback to direct parsing if not wrapped in ```json
            json_match_fallback = re.search(JSON_FALLBACK_PATTERN, response.content, re.DOTALL)
            if json_match_fallback:
                return json.loads(json_match_fallback.group())
            else:
                print(f"LLM 응답에서 JSON을 찾을 수 없습니다: {response.content}")
                return {}

    except (json.JSONDecodeError, Exception) as e:
        print(f"LLM 분류 중 오류 발생: {e}")
        return {{}}

async def reclassify_festivals():
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "database", "축제공연행사csv.CSV")
    output_dir = os.path.join(script_dir, "new_festivals_classification")

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, encoding='cp949')
    except FileNotFoundError:
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생: {e}")
        return

    llm_client = get_llm_client()
    
    # 필요한 컬럼만 추출하고, NaN 값을 빈 문자열로 대체
    df_filtered = df[['title', 'overview', 'firstimage', 'eventstartdate', 'eventenddate', 'mapx', 'mapy']].fillna('')
    
    # DataFrame을 딕셔너리 리스트로 변환
    all_festivals_data = df_filtered.to_dict('records')

    # 최종 분류 결과를 저장할 딕셔너리 초기화
    # 대분류 -> 중분류 -> 소분류 -> 축제 리스트
    final_categorized_festivals = {
        main_cat: {
            middle_cat: {
                sub_cat: [] for sub_cat in sub_cats
            } for middle_cat, sub_cats in middle_cats.items()
        } for main_cat, middle_cats in FULL_CATEGORY_HIERARCHY.items()
    }
    final_categorized_festivals["분류 실패"] = [] # 분류에 실패한 경우를 위한 카테고리

    print(f"총 {len(all_festivals_data)}개의 축제를 {BATCH_SIZE}개씩 묶어 분류를 시작합니다.")

    # 전체 축제 데이터를 배치 크기만큼 나누어 처리
    for i in range(0, len(all_festivals_data), BATCH_SIZE):
        batch = all_festivals_data[i:i + BATCH_SIZE]
        
        print(f"  ... {i+1}번부터 {i+len(batch)}번까지의 축제를 처리 중...")
        
        # LLM을 통해 현재 배치 분류
        classified_batch = await classify_festivals_in_batch(batch, llm_client)

        # 분류 결과를 기반으로 최종 딕셔너리에 데이터 추가
        for festival_data in batch:
            title = festival_data['title']
            # LLM의 응답에서 해당 축제의 분류 결과를 가져옴
            assigned_categories = classified_batch.get(title) # { "main_category": "...", "middle_category": "...", "sub_category": "..." }

            if assigned_categories and isinstance(assigned_categories, dict) and \
               'main_category' in assigned_categories and \
               'middle_category' in assigned_categories and \
               'sub_category' in assigned_categories:
                
                main_cat = assigned_categories['main_category']
                middle_cat = assigned_categories['middle_category']
                sub_cat = assigned_categories['sub_category']

                if main_cat in final_categorized_festivals and \
                   middle_cat in final_categorized_festivals[main_cat] and \
                   sub_cat in final_categorized_festivals[main_cat][middle_cat]:
                    final_categorized_festivals[main_cat][middle_cat][sub_cat].append(festival_data)
                else:
                    # LLM이 유효하지 않은 대분류/중분류/소분류를 반환한 경우
                    final_categorized_festivals["분류 실패"].append(festival_data)
                    print(f"    [분류 실패] '{title}' 축제 (할당된 대분류: {main_cat}, 중분류: {middle_cat}, 소분류: {sub_cat}) - 유효하지 않은 카테고리)")
            else:
                # LLM 응답 형식이 잘못되었거나 분류 정보가 누락된 경우
                final_categorized_festivals["분류 실패"].append(festival_data)
                print(f"    [분류 실패] '{title}' 축제 (LLM 응답 형식 오류 또는 분류 누락: {assigned_categories})")

    print("\n최종 분류 완료. JSON 파일을 생성합니다...")

    # 각 대분류별로 JSON 파일 생성 (내부에 중분류, 소분류별로 묶음)
    for main_category, middle_categories_data in final_categorized_festivals.items():
        if main_category == "분류 실패":
            if middle_categories_data: # 분류 실패 항목이 있으면 별도 파일로 저장
                sanitized_category = main_category.replace(" ", "_").replace("/", "_")
                output_file_path = os.path.join(output_dir, f"festivals_type_{sanitized_category}.json")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(middle_categories_data, f, ensure_ascii=False, indent=2)
                print(f"'{main_category}' 카테고리의 축제 {len(middle_categories_data)}개가 {output_file_path}에 저장되었습니다.")
            continue

        # 해당 대분류에 속한 축제가 하나라도 있는지 확인
        has_festivals = any(
            any(festivals for festivals in sub_categories_data.values())
            for sub_categories_data in middle_categories_data.values()
        )

        if not has_festivals:
            continue

        sanitized_main_category = main_category.replace(" ", "_").replace("/", "_")
        output_file_path = os.path.join(output_dir, f"festivals_type_{sanitized_main_category}.json")
        
        # 중분류, 소분류별로 묶인 데이터를 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(middle_categories_data, f, ensure_ascii=False, indent=2)
            
        total_festivals_in_main_cat = sum(
            len(fests) for sub_categories_data in middle_categories_data.values()
            for fests in sub_categories_data.values()
        )
        print(f"'{main_category}' 카테고리의 축제 {total_festivals_in_main_cat}개가 {output_file_path}에 저장되었습니다.")

    print("\n축제 재분류 및 JSON 파일 생성이 완료되었습니다.")

if __name__ == "__main__":
    # asyncio.run()은 스크립트당 한 번만 호출하는 것이 좋습니다.
    # 만약 다른 비동기 작업과 함께 실행해야 한다면, main 함수를 만들어 관리하세요.
    asyncio.run(reclassify_festivals())