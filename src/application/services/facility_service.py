import sqlite3
from typing import Dict, Any, Optional
from src.infrastructure.persistence.database import get_db_connection

def get_facility_details_by_title(title: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM facilities WHERE title = ?", (title,))
    facility_row = cursor.fetchone()
    
    conn.close()
    
    if facility_row:
        return dict(facility_row)
    return None
