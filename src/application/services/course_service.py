from typing import Dict, Any, Optional
from src.infrastructure.persistence.database import get_db_connection, release_connection, get_cursor

def get_course_details_by_title(title: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        # Fetch the main course details
        cursor.execute("SELECT * FROM courses WHERE title = %s", (title,))
        main_course_row = cursor.fetchone()

        if not main_course_row:
            return None

        main_course_details = dict(main_course_row)

        # Fetch sub-points for the course
        # Assuming sub-points are also in the 'courses' table and linked by contentid
        cursor.execute(
            "SELECT * FROM courses WHERE contentid = %s AND subnum IS NOT NULL ORDER BY subnum",
            (main_course_details['contentid'],)
        )
        sub_points_rows = cursor.fetchall()

        main_course_details['sub_points'] = [dict(row) for row in sub_points_rows]

        return main_course_details
    finally:
        cursor.close()
        release_connection(conn)
