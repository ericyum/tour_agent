"""
Add refresh_tokens table for secure token management
"""

import sqlite3
import os

project_root = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_root, "tour.db")

print(f"Connecting to database: {db_path}")

if not os.path.exists(db_path):
    print("ERROR: tour.db not found!")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    print("\n=== Adding refresh_tokens table ===\n")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            revoked BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    print("[OK] refresh_tokens table created")

    # Create index for performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token
        ON refresh_tokens(token)
    ''')
    print("[OK] Index created on token column")

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id
        ON refresh_tokens(user_id)
    ''')
    print("[OK] Index created on user_id column")

    conn.commit()
    print("\n=== Migration completed successfully! ===")

except Exception as e:
    print(f"\nERROR: Migration failed: {e}")
    conn.rollback()
    exit(1)
finally:
    conn.close()
    print("\nDatabase connection closed.")
