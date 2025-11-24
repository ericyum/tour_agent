"""
Add authentication and Q&A system tables to tour.db
"""

import sqlite3
import os
import hashlib

# Get database path
project_root = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_root, "tour.db")

print(f"Connecting to database: {db_path}")

if not os.path.exists(db_path):
    print("ERROR: tour.db not found!")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    print("\n=== Adding authentication and Q&A tables ===\n")

    # Create users table
    print("Creating users table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME
        )
    ''')
    print("[OK] Users table created")

    # Create questions table (Q&A)
    print("Creating questions table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            festival_name TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            views INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    print("[OK] Questions table created")

    # Create answers table
    print("Creating answers table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            is_accepted BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES questions (id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    print("[OK] Answers table created")

    # Update feedback table to link with users (optional)
    print("Checking if feedback table needs user_id column...")
    cursor.execute("PRAGMA table_info(feedback)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'user_id' not in columns:
        cursor.execute('ALTER TABLE feedback ADD COLUMN user_id INTEGER')
        print("[OK] Added user_id to feedback table")
    else:
        print("[SKIP] user_id already exists in feedback table")

    # Update feature_ratings table to link with users
    print("Checking if feature_ratings table needs user_id column...")
    cursor.execute("PRAGMA table_info(feature_ratings)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'user_id' not in columns:
        cursor.execute('ALTER TABLE feature_ratings ADD COLUMN user_id INTEGER')
        print("[OK] Added user_id to feature_ratings table")
    else:
        print("[SKIP] user_id already exists in feature_ratings table")

    conn.commit()

    # Create default admin account
    print("\n=== Creating default admin account ===")
    admin_password = "admin123"  # Change this in production!
    password_hash = hashlib.sha256(admin_password.encode()).hexdigest()

    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@festmoment.com', password_hash, 'Administrator', 'admin'))
        conn.commit()
        print(f"[OK] Admin account created")
        print(f"     Username: admin")
        print(f"     Password: {admin_password}")
    except sqlite3.IntegrityError:
        print("[SKIP] Admin account already exists")

    # Create sample user account
    print("\n=== Creating sample user account ===")
    user_password = "user123"
    password_hash = hashlib.sha256(user_password.encode()).hexdigest()

    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('testuser', 'user@festmoment.com', password_hash, 'Test User', 'user'))
        conn.commit()
        print(f"[OK] Sample user account created")
        print(f"     Username: testuser")
        print(f"     Password: {user_password}")
    except sqlite3.IntegrityError:
        print("[SKIP] Sample user account already exists")

    print("\n=== Migration completed successfully! ===")

    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\nAll tables: {[t[0] for t in tables]}")

except Exception as e:
    print(f"\nERROR: Migration failed: {e}")
    conn.rollback()
    exit(1)
finally:
    conn.close()
    print("\nDatabase connection closed.")
