import sqlite3
import os
import json
from src.application.core.db_state import DBSearchState
from src.application.core.constants import AREA_CODE_MAP, SIGUNGU_CODE_MAP, NO_IMAGE_URL
from src.infrastructure.persistence.database import get_db_connection

# This function is a helper to load the category mappings, similar to what's in app.py
# In a more advanced architecture, this might be a shared service.
def get_title_to_cat_names_map():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    festivals_dir = os.path.join(project_root, "festivals")
    all_categories = {}
    title_to_cat_names = {}
    try:
        for filename in os.listdir(festivals_dir):
            if filename.endswith(".json"):
                with open(os.path.join(festivals_dir, filename), 'r', encoding='utf-8') as f:
                    all_categories.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading festival categories in agent: {e}")
        return {}

    for main_name, med_dict in all_categories.items():
        for med_name, small_dict in med_dict.items():
            for small_name, titles in small_dict.items():
                for title in titles:
                    title_to_cat_names[title] = (main_name, med_name, small_name)
    return title_to_cat_names

def agent_festival_search(state: DBSearchState) -> DBSearchState:
    area = state.get("area")
    sigungu = state.get("sigungu")
    main_cat = state.get("main_cat")
    medium_cat = state.get("medium_cat")
    small_cat = state.get("small_cat")

    # Step 1: Primary filtering by location from DB
    loc_where_clauses = []
    loc_params = []

    if area and area != "전체":
        area_code = AREA_CODE_MAP.get(area)
        if area_code:
            loc_where_clauses.append("areacode = ?")
            loc_params.append(area_code)
            if sigungu and sigungu != "전체":
                sigungu_code = SIGUNGU_CODE_MAP.get(area, {}).get(sigungu)
                if sigungu_code:
                    loc_where_clauses.append("sigungucode = ?")
                    loc_params.append(sigungu_code)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT title, firstimage FROM festivals"
    if loc_where_clauses:
        query += " WHERE " + " AND ".join(loc_where_clauses)
        cursor.execute(query, loc_params)
    else:
        cursor.execute(query)
        
    db_results = cursor.fetchall()
    conn.close()

    # Step 2: Secondary filtering by category using map
    TITLE_TO_CAT_NAMES = get_title_to_cat_names_map()
    is_cat_filtered = main_cat != "전체" or medium_cat != "전체" or small_cat != "전체"
    
    if not is_cat_filtered:
        final_results_tuples = db_results
    else:
        final_results_tuples = []
        for row in db_results:
            title = row[0]
            cat_names = TITLE_TO_CAT_NAMES.get(title)
            if not cat_names:
                continue

            main_match = (main_cat == "전체" or main_cat == cat_names[0])
            medium_match = (medium_cat == "전체" or medium_cat == cat_names[1])
            small_match = (small_cat == "전체" or small_cat == cat_names[2])

            if main_match and medium_match and small_match:
                final_results_tuples.append(row)

    # Format results into the structure expected by the UI
    results = sorted([(row[0], row[1] or NO_IMAGE_URL) for row in final_results_tuples])
    
    state["results"] = results
    return state