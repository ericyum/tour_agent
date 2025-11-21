"""
Migration script to add guest user tracking to user_events and feedback tables
This enables tracking of both authenticated users and guest users (Guest1, Guest2, etc.)
"""

import sqlite3

def migrate():
    conn = sqlite3.connect('tour.db')
    cursor = conn.cursor()

    print("[OK] Starting guest user tracking migration...")

    try:
        # Add user tracking columns to user_events table
        print("[OK] Adding user tracking columns to user_events...")
        cursor.execute("""
            ALTER TABLE user_events ADD COLUMN user_id INTEGER DEFAULT NULL
        """)
        cursor.execute("""
            ALTER TABLE user_events ADD COLUMN guest_id TEXT DEFAULT NULL
        """)
        cursor.execute("""
            ALTER TABLE user_events ADD COLUMN username TEXT DEFAULT NULL
        """)

        # Add user tracking columns to feedback table (if not exists)
        print("[OK] Adding user_id to feedback table...")
        try:
            cursor.execute("""
                ALTER TABLE feedback ADD COLUMN user_id INTEGER DEFAULT NULL
            """)
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("[OK] user_id column already exists in feedback table")
            else:
                raise

        # Create indexes for better query performance
        print("[OK] Creating indexes...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON user_events(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_events_guest_id ON user_events(guest_id)
        """)

        conn.commit()
        print("[OK] Migration completed successfully!")
        print("[OK] Guest users will now be tracked as 'Guest1', 'Guest2', etc.")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
