from src.infrastructure.persistence.database import get_db_connection, release_connection, get_cursor

def get_festival_details_by_title(festival_name: str):
    """Fetches all details for a given festival by its title."""
    if not festival_name:
        return None
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute("SELECT * FROM festivals WHERE title = %s", (festival_name,))
        festival = cursor.fetchone()
        return dict(festival) if festival else None
    finally:
        cursor.close()
        release_connection(conn)
