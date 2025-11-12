import sqlite3
from src.application.core.db_state import DBSearchState
from src.infrastructure.persistence.database import get_db_connection
from src.application.core.utils import haversine

def agent_nearby_search(state: DBSearchState) -> DBSearchState:
    latitude = state.get("latitude")
    longitude = state.get("longitude")
    radius = state.get("radius")
    # Get the contentid of the festival to be excluded, if it exists
    current_festival_id = state.get("current_festival_id")


    if not all([latitude, longitude, radius]):
        # Should not happen if routed correctly, but as a safeguard
        state["recommended_facilities"] = []
        state["recommended_courses"] = []
        state["recommended_festivals"] = []
        return state

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    facilities = conn.execute("SELECT * FROM facilities WHERE mapx IS NOT NULL AND mapy IS NOT NULL").fetchall()
    courses = conn.execute("SELECT * FROM courses WHERE mapx IS NOT NULL AND mapy IS NOT NULL").fetchall()
    festivals = conn.execute("SELECT * FROM festivals WHERE mapx IS NOT NULL AND mapy IS NOT NULL").fetchall()
    conn.close()

    facilities_recs = []
    for place_row in facilities:
        try:
            dest_lon, dest_lat = float(place_row['mapx']), float(place_row['mapy'])
            if not (124 < dest_lon < 132 and 33 < dest_lat < 39):
                if (124 < dest_lat < 132 and 33 < dest_lon < 39):
                    dest_lon, dest_lat = dest_lat, dest_lon
            
            distance = haversine(longitude, latitude, dest_lon, dest_lat)
            if distance <= float(radius):
                place_dict = {k: place_row[k] for k in place_row.keys()}
                place_dict['distance'] = distance  # Add distance to the dictionary
                facilities_recs.append(place_dict)
        except (ValueError, TypeError):
            continue

    courses_recs_grouped = {}
    min_course_distances = {}
    for place_row in courses:
        try:
            dest_lon, dest_lat = float(place_row['mapx']), float(place_row['mapy'])
            if not (124 < dest_lon < 132 and 33 < dest_lat < 39):
                if (124 < dest_lat < 132 and 33 < dest_lon < 39):
                    dest_lon, dest_lat = dest_lat, dest_lon

            distance = haversine(longitude, latitude, dest_lon, dest_lat)
            if distance <= float(radius):
                course_dict = {k: place_row[k] for k in place_row.keys()}
                content_id = course_dict['contentid']
                
                # Store the minimum distance for the entire course
                if content_id not in min_course_distances or distance < min_course_distances[content_id]:
                    min_course_distances[content_id] = distance

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
        # Create a new dictionary for the course, using the main_info as a base
        # but ensuring it's a shallow copy to avoid modifying the original dict in sub_points.
        main_info_copy = course_group['main_info'].copy()
        
        # Sort the sub_points
        sorted_sub_points = sorted(course_group['sub_points'], key=lambda x: x.get('subnum', 0))
        
        # Assign the sorted sub_points list to the new course object
        main_info_copy['sub_points'] = sorted_sub_points
        
        # Add the minimum distance
        main_info_copy['distance'] = min_course_distances.get(content_id, float('inf'))
        
        # Append the new, clean object to the results
        courses_recs.append(main_info_copy)

    festivals_recs = []
    for festival_row in festivals:
        # Exclude the current festival from its own recommendation list
        if current_festival_id and festival_row['contentid'] == current_festival_id:
            continue
        try:
            dest_lon, dest_lat = float(festival_row['mapx']), float(festival_row['mapy'])
            if not (124 < dest_lon < 132 and 33 < dest_lat < 39):
                if (124 < dest_lat < 132 and 33 < dest_lon < 39):
                    dest_lon, dest_lat = dest_lat, dest_lon

            distance = haversine(longitude, latitude, dest_lon, dest_lat)
            if distance <= float(radius):
                festival_dict = {k: festival_row[k] for k in festival_row.keys()}
                festival_dict['distance'] = distance
                festivals_recs.append(festival_dict)
        except (ValueError, TypeError):
            continue
    
    state["recommended_facilities"] = sorted(facilities_recs, key=lambda x: x.get('distance', float('inf')))
    state["recommended_courses"] = sorted(courses_recs, key=lambda x: x.get('distance', float('inf')))
    state["recommended_festivals"] = sorted(festivals_recs, key=lambda x: x.get('distance', float('inf')))
    
    return state
