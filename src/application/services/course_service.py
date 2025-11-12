import sqlite3
from typing import Dict, Any, Optional
from src.infrastructure.persistence.database import get_db_connection

def get_course_details_by_title(title: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Fetch the main course details
    cursor.execute("SELECT * FROM courses WHERE title = ?", (title,))
    main_course_row = cursor.fetchone()
    
    if not main_course_row:
        conn.close()
        return None
    
    main_course_details = dict(main_course_row)
    
    # Fetch sub-points for the course
    # Assuming sub-points are also in the 'courses' table and linked by contentid
    # This might need adjustment based on actual DB schema if sub-points are in a different table
    cursor.execute("SELECT * FROM courses WHERE contentid = ? AND subnum IS NOT NULL ORDER BY subnum", (main_course_details['contentid'],))
    sub_points_rows = cursor.fetchall()
    
    main_course_details['sub_points'] = [dict(row) for row in sub_points_rows]
    
    conn.close()
    return main_course_details
