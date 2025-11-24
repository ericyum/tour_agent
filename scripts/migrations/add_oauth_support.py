"""
Migration script to add OAuth support (Google login) and prepare for account deletion
"""

import sqlite3

def migrate():
    conn = sqlite3.connect('tour.db')
    cursor = conn.cursor()

    print("[OK] Starting OAuth support migration...")

    try:
        # Add OAuth provider columns
        print("[OK] Adding auth_provider column...")
        cursor.execute("""
            ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'local'
        """)

        print("[OK] Adding oauth_id column...")
        cursor.execute("""
            ALTER TABLE users ADD COLUMN oauth_id TEXT DEFAULT NULL
        """)

        print("[OK] Adding oauth_picture column for profile image...")
        cursor.execute("""
            ALTER TABLE users ADD COLUMN oauth_picture TEXT DEFAULT NULL
        """)

        # Update existing users to have 'local' auth_provider
        cursor.execute("""
            UPDATE users SET auth_provider = 'local' WHERE auth_provider IS NULL
        """)

        # Create index for OAuth lookups
        print("[OK] Creating indexes...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_oauth_id ON users(oauth_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_auth_provider ON users(auth_provider)
        """)

        # Note: SQLite doesn't support ALTER COLUMN to make password_hash nullable
        # We'll handle this in application logic (allow NULL for OAuth users)
        print("[OK] Note: password_hash will be NULL for OAuth users (handled in application)")

        conn.commit()
        print("[OK] OAuth migration completed successfully!")
        print("[OK] Supported providers: 'local' (username/password), 'google' (OAuth)")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
