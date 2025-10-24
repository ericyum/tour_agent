from src.infrastructure.persistence.database import get_db_connection

def get_festival_details_by_title(festival_name: str):
    """Fetches all details for a given festival by its title."""
    if not festival_name:
        return None
    conn = get_db_connection()
    try:
        # The connection object already has row_factory set to sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM festivals WHERE title = ?", (festival_name,))
        festival = cursor.fetchone()
        return dict(festival) if festival else None
    finally:
        conn.close()
