import pandas as pd
import json
import os
import asyncio
from src.infrastructure.llm_client import get_llm_client

async def classify_festival_with_llm(festival_name: str, overview: str, llm_client) -> str:
    """
    LLM을 사용하여 축제를 새로운 대분류로 분류합니다.
    """
    prompt = f"""
    당신은 한국의 축제를 전문적으로 분류하는 전문가입니다.
    아래 축제 이름과 개요를 바탕으로, 이 축제가 다음 7~10가지 대분류 중 어디에 속하는지 가장 적절한 하나의 대분류 이름을 반환해주세요.
    만약 적절한 대분류가 없다면 기존 분류에 얽매이지 않고 새로운 대분류 이름을 직접 생성하여 반환해주세요.

    [대분류 후보 (예시, 필요에 따라 더 적절한 분류를 생성해도 됩니다. 새로운 분류가 필요하면 직접 생성해주세요.)]
    1. 문화예술 (음악, 미술, 공연, 문학 등)
    2. 자연경관 (산, 바다, 꽃, 단풍 등 자연을 주제로 한 축제)
    3. 지역특산물 (특정 지역의 농수산물, 음식 등을 주제로 한 축제)
    4. 역사전통 (역사적 사건, 전통 문화, 유적 등을 주제로 한 축제)
    5. 체험레저 (스포츠, 액티비티, 참여형 프로그램 등)
    6. 도시생활 (도시에서 열리는 현대적인 이벤트, 거리 축제 등)
    7. 산업과학 (특정 산업, 기술, 과학 등을 주제로 한 축제)
    8. 종교의례 (특정 종교나 의례를 주제로 한 축제)

    [축제 정보]
    - 축제 이름: {festival_name}
    - 개요: {overview}

    [출력 형식]
    오직 하나의 대분류 이름만 반환해주세요. (예: 문화예술)
    """
    try:
        response = await llm_client.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"LLM 분류 중 오류 발생: {e}")
        return "분류 실패" # Return a specific category for failed classifications

async def verify_category_with_llm(festival_name: str, overview: str, assigned_category: str, all_categories: list, llm_client) -> str:
    """
    LLM을 사용하여 축제의 분류가 적절한지 검증하고, 필요시 재분류합니다.
    """
    all_categories_str = ", ".join(sorted(list(set(all_categories))))
    
    prompt = f"""
    당신은 한국의 축제를 전문적으로 분류하는 전문가입니다.
    아래 축제 이름과 개요, 그리고 현재 할당된 대분류 정보를 바탕으로, 이 축제가 현재 대분류에 적절하게 분류되었는지 검증해주세요.
    만약 현재 대분류가 적절하다면 '적절함'이라고만 반환해주세요.
    만약 적절하지 않다면, 기존에 생성된 대분류 목록({all_categories_str}) 중에서 가장 적절한 대분류 이름을 반환하거나,
    새로운 대분류가 필요하다고 판단되면 새로운 대분류 이름을 직접 생성하여 반환해주세요.

    [축제 정보]
    - 축제 이름: {festival_name}
    - 개요: {overview}
    - 현재 할당된 대분류: {assigned_category}

    [출력 형식]
    오직 하나의 대분류 이름만 반환해주세요. (예: 적절함 또는 문화예술 또는 새로운 분류명)
    """
    try:
        response = await llm_client.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"LLM 검증 중 오류 발생: {e}")
        return assigned_category # 오류 발생 시 기존 분류 유지

async def reclassify_festivals():
    script_dir = os.path.dirname(__file__)
    csv_path = "C:\\Users\\SBA\\github\\tour_agent\\database\\축제공연행사csv.CSV"
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
    
    df_filtered = df[['title', 'overview', 'firstimage', 'eventstartdate', 'eventenddate', 'mapx', 'mapy']].fillna('')

    initial_categorized_festivals = {}
    print(f"총 {len(df_filtered)}개의 축제를 1차 분류합니다...")

    initial_tasks = []
    for index, row in df_filtered.iterrows():
        festival_name = row['title']
        overview = row['overview']
        initial_tasks.append(classify_festival_with_llm(festival_name, overview, llm_client))

    initial_categories = await asyncio.gather(*initial_tasks)

    for index, row in df_filtered.iterrows():
        festival_name = row['title']
        category = initial_categories[index]
        
        festival_data = {
            "title": festival_name,
            "overview": row['overview'],
            "firstimage": row['firstimage'],
            "eventstartdate": row['eventstartdate'],
            "eventenddate": row['eventenddate'],
            "mapx": row['mapx'],
            "mapy": row['mapy'],
        }

        if category not in initial_categorized_festivals:
            initial_categorized_festivals[category] = []
        initial_categorized_festivals[category].append(festival_data)
        # print(f"1차 분류: '{festival_name}' -> '{category}'") # 너무 많은 출력 방지

    print("1차 분류 완료. 이제 분류된 내용을 점검하고 필요시 재분류합니다...")

    final_categorized_festivals = {}
    max_verification_attempts = 2 # 각 축제당 최대 2번의 재분류 시도

    # 모든 1차 분류 카테고리 목록을 준비
    all_current_categories = list(initial_categorized_festivals.keys())

    verification_tasks = []
    festivals_to_process = []

    for initial_category, festivals in initial_categorized_festivals.items():
        for festival_data in festivals:
            festivals_to_process.append((festival_data, initial_category, 0)) # (festival_data, assigned_category, attempt_count)

    while festivals_to_process:
        current_festival_data, current_assigned_category, attempt_count = festivals_to_process.pop(0)
        festival_name = current_festival_data['title']
        overview = current_festival_data['overview']

        if attempt_count >= max_verification_attempts:
            print(f"[최대 시도 횟수 초과] '{festival_name}'은(는) '{current_assigned_category}'로 최종 분류됩니다.")
            if current_assigned_category not in final_categorized_festivals:
                final_categorized_festivals[current_assigned_category] = []
            final_categorized_festivals[current_assigned_category].append(current_festival_data)
            continue

        # 현재까지 생성된 모든 카테고리 목록을 업데이트하여 LLM에 전달
        all_current_categories = list(final_categorized_festivals.keys()) + list(initial_categorized_festivals.keys())
        all_current_categories = list(set(all_current_categories))

        verified_category = await verify_category_with_llm(
            festival_name, overview, current_assigned_category, all_current_categories, llm_client
        )

        if verified_category == '적절함':
            if current_assigned_category not in final_categorized_festivals:
                final_categorized_festivals[current_assigned_category] = []
            final_categorized_festivals[current_assigned_category].append(current_festival_data)
            print(f"[검증 완료] '{festival_name}'은(는) '{current_assigned_category}'에 적절합니다.")
        elif verified_category == current_assigned_category: # LLM이 같은 카테고리를 반환했지만 '적절함'이 아닌 경우 (재시도 필요 없음)
            if current_assigned_category not in final_categorized_festivals:
                final_categorized_festivals[current_assigned_category] = []
            final_categorized_festivals[current_assigned_category].append(current_festival_data)
            print(f"[검증 완료] '{festival_name}'은(는) '{current_assigned_category}'에 적절합니다. (LLM 재확인)")
        else:
            print(f"[재분류 필요] '{festival_name}' ('{current_assigned_category}' -> '{verified_category}') 재분류 시도: {attempt_count + 1}")
            # 재분류가 필요한 경우, 다시 처리 목록에 추가 (시도 횟수 증가)
            festivals_to_process.append((current_festival_data, verified_category, attempt_count + 1))

    print("최종 분류 완료. JSON 파일을 생성합니다...")

    for category, festivals in final_categorized_festivals.items():
        # 파일명에 사용할 수 없는 문자 제거
        sanitized_category = category.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        output_file_path = os.path.join(output_dir, f"festivals_type_{sanitized_category}.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(festivals, f, ensure_ascii=False, indent=2)
        print(f"'{category}' 카테고리의 축제 {len(festivals)}개가 {output_file_path}에 저장되었습니다.")

    print("축제 재분류 및 JSON 파일 생성이 완료되었습니다.")

if __name__ == "__main__":
    asyncio.run(reclassify_festivals())