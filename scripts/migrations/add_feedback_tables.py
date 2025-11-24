"""
Database migration script to add feedback tables to existing tour.db
Run this script once to add the new tables without destroying existing data
"""

import sqlite3
import os

# Get database path
project_root = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_root, "tour.db")

print(f"Connecting to database: {db_path}")

if not os.path.exists(db_path):
    print("ERROR: tour.db not found! Please run database initialization first.")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    print("\n=== Adding feedback tables ===\n")

    # Create feedback table
    print("Creating feedback table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_url TEXT NOT NULL,
            festival_name TEXT,
            rating INTEGER NOT NULL,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_agent TEXT,
            session_id TEXT
        )
    ''')
    print("[OK] Feedback table created")

    # Create feature_ratings table
    print("Creating feature_ratings table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            festival_name TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT
        )
    ''')
    print("[OK] Feature ratings table created")

    # Create user_events table
    print("Creating user_events table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_category TEXT NOT NULL,
            event_action TEXT NOT NULL,
            event_label TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            page_url TEXT
        )
    ''')
    print("[OK] User events table created")

    conn.commit()
    print("\n=== Migration completed successfully! ===")

    # Verify tables were created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\nAll tables in database: {[t[0] for t in tables]}")

except Exception as e:
    print(f"\nERROR: Migration failed: {e}")
    conn.rollback()
    exit(1)
finally:
    conn.close()
    print("\nDatabase connection closed.")
