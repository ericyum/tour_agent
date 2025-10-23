import sqlite3
from src.application.core.db_state import DBSearchState
from src.infrastructure.persistence.database import get_db_connection
from src.application.core.utils import haversine

def agent_nearby_search(state: DBSearchState) -> DBSearchState:
    latitude = state.get("latitude")
    longitude = state.get("longitude")
    radius = state.get("radius")

    if not all([latitude, longitude, radius]):
        # Should not happen if routed correctly, but as a safeguard
        state["recommended_facilities"] = []
        state["recommended_courses"] = []
        return state

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    facilities = conn.execute("SELECT * FROM facilities WHERE mapx IS NOT NULL AND mapy IS NOT NULL").fetchall()
    courses = conn.execute("SELECT * FROM courses WHERE mapx IS NOT NULL AND mapy IS NOT NULL").fetchall()
    conn.close()

    facilities_recs = []
    for place_row in facilities:
        try:
            distance = haversine(longitude, latitude, place_row['mapx'], place_row['mapy'])
            if distance <= float(radius):
                facilities_recs.append(dict(place_row))
        except (ValueError, TypeError):
            continue

    courses_recs_grouped = {}
    for place_row in courses:
        try:
            distance = haversine(longitude, latitude, place_row['mapx'], place_row['mapy'])
            if distance <= float(radius):
                course_dict = dict(place_row)
                content_id = course_dict['contentid']
                if content_id not in courses_recs_grouped:
                    courses_recs_grouped[content_id] = {
                        'main_info': course_dict,
                        'sub_points': []
                    }
                courses_recs_grouped[content_id]['sub_points'].append(course_dict)
        except (ValueError, TypeError):
            continue

    courses_recs = []
    for content_id, course_group in courses_recs_grouped.items():
        final_course_obj = course_group['main_info']
        final_course_obj['sub_points'] = sorted(course_group['sub_points'], key=lambda x: x.get('subnum', 0))
        courses_recs.append(final_course_obj)
    
    state["recommended_facilities"] = facilities_recs
    state["recommended_courses"] = courses_recs
    
    return state
