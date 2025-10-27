from src.infrastructure.llm_client import get_llm_client
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

def agent_validate_course(state: dict) -> dict:
    course = state.get("course")
    duration = state.get("duration")

    if not course or not duration:
        state["validation_result"] = "코스나 여행 기간 정보가 부족하여 검증할 수 없습니다."
        return state

    # 1. Initialize Geocoder
    try:
        geolocator = Nominatim(user_agent="tour_agent_v1")
        # Add a rate limiter to avoid overwhelming the free service
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    except Exception as e:
        state["validation_result"] = f"지오코딩 서비스 초기화 중 오류가 발생했습니다: {e}"
        return state

    # 2. Create a flat list of all points of interest
    all_points = []
    for item in course:
        item_type = "관광지"
        if "eventstartdate" in item: item_type = "축제"
        elif "taketime" in item: item_type = "코스"

        # If it's a Course, deconstruct it
        if item_type == "코스" and "sub_points" in item:
            base_address_area = item.get("addr1", "").split(" ")[0] if item.get("addr1") else ""
            for sp in item["sub_points"]:
                point = {
                    "title": sp.get("subname"),
                    "overview": sp.get("subdetailoverview"),
                    "type": "코스 내 장소",
                    "parent_course_title": item.get("title")
                }
                # Geocode the sub-point
                location = None
                # Try to geocode only if name is present
                if sp.get("subname"):
                    try:
                        # Construct a more specific query for better results
                        query = f'{sp.get("subname")}, {base_address_area}'
                        location = geocode(query)
                    except Exception:
                        location = None # Geocoding failed

                if location:
                    point["latitude"] = location.latitude
                    point["longitude"] = location.longitude
                    point["address"] = location.address
                
                all_points.append(point)
        else: # For Festivals and Facilities
            point = {
                "title": item.get("title"),
                "overview": item.get("overview"),
                "type": item_type,
                "latitude": item.get("mapy"),
                "longitude": item.get("mapx"),
                "address": item.get("addr1"),
                # Pass other relevant details
                "playtime": item.get("playtime") or item.get("usetimeculture"),
                "duration_info": item.get("taketime") or item.get("spendtimefestival") or item.get("spendtime"),
                "period": f'{item.get("eventstartdate")} - {item.get("eventenddate")}' if "eventstartdate" in item else None
            }
            all_points.append(point)

    # Filter out points without coordinates
    valid_points = [p for p in all_points if p.get("latitude") and p.get("longitude")]
    
    if not valid_points:
        state["validation_result"] = "코스에 포함된 장소들의 좌표를 확인할 수 없어 상세 검증을 진행할 수 없습니다."
        return state

    # 3. Create the new, powerful prompt
    points_json = json.dumps(valid_points, ensure_ascii=False, indent=2)
    
    prompt = f'''
    당신은 모든 장소의 물리적 위치(좌표), 상세 정보, 예상 소요 시간, 운영 시간을 종합하여 최적의 여행 경로를 짜는 '루트 최적화 전문 가이드'입니다.

    [여행 정보]
    - 여행 기간: {duration}
    - 방문할 모든 장소 목록 (좌표 및 상세 정보 포함):
    {points_json}

    [최상위 임무]
    위 장소 목록을 모두 방문하는 가장 효율적이고 즐거운 '최적 여행 계획'을 새로 수립해주세요. 사용자가 제시한 순서는 잊고, 전문가의 입장에서 0부터 다시 계획을 세워야 합니다.

    [세부 분석 및 제안 요청]
    1.  **최적 동선 및 일정 수립**:
        -   모든 장소의 위도/경도 값을 바탕으로, 지리적으로 가장 효율적인 방문 순서를 결정해주세요. (참고: 위도 1도 = 약 111km, 경도 1도 = 약 88km)
        -   결정된 순서에 따라, {duration} 기간 동안의 **일자별 추천 여행 일정표**를 구체적으로 작성해주세요.
        -   일정표에는 각 장소의 예상 도착 시간, 관람 시간, 다음 장소로의 예상 이동 시간을 포함해야 합니다. (이동 시간은 거리를 바탕으로 합리적으로 추정해주세요.)
        -   '체력 안배'(활동적인 곳과 정적인 곳의 조화), '운영 시간', '식사 시간'까지 모두 고려하여 가장 현실적인 계획을 세워야 합니다.

    2.  **계획 현실성 평가**: 당신이 수립한 '최적 여행 계획'이 현실적인지 스스로 평가해주세요. 만약 계획이 무리하다면, 어떤 부분을 조정해야 할지(예: 일부 장소 제외, 관람 시간 단축) 대안을 제시해주세요. 반대로 시간이 남는다면, 추가할 만한 주변 장소나 활동을 추천해주세요.

    3.  **최종 조언**: 당신이 제안한 계획을 성공적으로 실행하기 위한 최종 팁(예: 추천 교통수단, 예약 팁 등)을 알려주세요.

    [출력 형식]
    - "👑 루트 최적화 전문가의 추천 여행 계획" 이라는 제목으로 시작해주세요.
    - 가장 먼저, 당신이 수립한 '일자별 추천 여행 일정표'를 명확하게 보여주세요.
    - 그 다음, '계획 현실성 평가'와 '최종 조언'을 순서대로 작성해주세요.
    - 모든 설명은 친구에게 말하듯 친절하고 상세한 어투를 사용해주세요.
    '''

    # 4. Invoke LLM
    llm = get_llm_client(temperature=0.5)
    try:
        response = llm.invoke(prompt)
        state["validation_result"] = response.content
    except Exception as e:
        state["validation_result"] = f"코스 검증 중 오류가 발생했습니다: {e}"

    return state
