from typing import Dict, Any, Optional
from src.infrastructure.persistence.database import get_db_connection, release_connection, get_cursor

def get_facility_details_by_title(title: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute("SELECT * FROM facilities WHERE title = %s", (title,))
        facility_row = cursor.fetchone()

        if facility_row:
            return dict(facility_row)
        return None
    finally:
        cursor.close()
        release_connection(conn)
