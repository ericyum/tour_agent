"""
PostgreSQL Database Module for FestMoment
SQLite에서 PostgreSQL로 전환된 버전
"""

import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the directory of the current script
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define file paths relative to the project root (for CSV data loading)
csv_files = {
    "festivals": os.path.join(project_root, "database", "축제공연행사csv.csv"),
    "facilities": os.path.join(project_root, "database", "문화시설csv.csv"),
    "courses": os.path.join(project_root, "database", "여행코스csv.csv")
}

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://festmoment:festmoment_password@localhost:5432/festmoment")

# Connection pool (thread-safe)
_connection_pool = None


def _parse_database_url(url: str) -> dict:
    """Parse DATABASE_URL into connection parameters"""
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") or "festmoment",
        "user": parsed.username or "festmoment",
        "password": parsed.password or "festmoment_password",
    }


def _get_pool():
    """Get or create connection pool"""
    global _connection_pool
    if _connection_pool is None:
        db_params = _parse_database_url(DATABASE_URL)
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **db_params
        )
    return _connection_pool


def get_db_connection():
    """
    Get a database connection from pool

    Returns a connection with RealDictCursor for dict-like row access
    PostgreSQL은 기본적으로 동시 접근을 지원함 (MVCC)
    """
    try:
        conn = _get_pool().getconn()
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        # Fallback: direct connection
        db_params = _parse_database_url(DATABASE_URL)
        return psycopg2.connect(**db_params)


def release_connection(conn):
    """Return connection to pool"""
    try:
        if _connection_pool:
            _connection_pool.putconn(conn)
    except Exception:
        conn.close()


def get_cursor(conn):
    """Get a cursor with dict-like row access"""
    return conn.cursor(cursor_factory=RealDictCursor)


# 각 테이블의 컬럼 정의
FESTIVALS_COLUMNS = [
    'addr1', 'addr2', 'areacode', 'cat1', 'cat2', 'cat3', 'contentid', 'contenttypeid', 'createdtime',
    'firstimage', 'firstimage2', 'cpyrhtDivCd', 'mapx', 'mapy', 'mlevel', 'modifiedtime', 'sigungucode',
    'tel', 'title', 'zipcode', 'lDongRegnCd', 'lDongSignguCd', 'lclsSystm1', 'lclsSystm2', 'lclsSystm3',
    'telname', 'homepage', 'overview', 'sponsor1', 'sponsor1tel', 'eventenddate', 'playtime', 'eventplace',
    'eventstartdate', 'usetimefestival', 'sponsor2', 'progresstype', 'festivaltype', 'sponsor2tel',
    'agelimit', 'spendtimefestival', 'festivalgrade', 'eventhomepage', 'subevent', 'program',
    'discountinfofestival', 'placeinfo', 'bookingplace'
]

FACILITIES_COLUMNS = [
    'addr1', 'addr2', 'areacode', 'cat1', 'cat2', 'cat3', 'contentid', 'contenttypeid', 'createdtime',
    'mapx', 'mapy', 'mlevel', 'modifiedtime', 'sigungucode', 'title', 'zipcode', 'lDongRegnCd',
    'lDongSignguCd', 'lclsSystm1', 'lclsSystm2', 'lclsSystm3', 'firstimage', 'firstimage2', 'cpyrhtDivCd',
    'tel', 'homepage', 'overview', 'telname', 'usefee', 'infocenterculture', 'usetimeculture',
    'restdateculture', 'parkingfee', 'parkingculture', 'chkcreditcardculture', 'chkbabycarriageculture',
    'spendtime', 'accomcountculture', 'scale', 'chkpetculture', 'discountinfo'
]

COURSES_COLUMNS = [
    'areacode', 'cat1', 'cat2', 'cat3', 'contentid', 'contenttypeid', 'createdtime', 'firstimage',
    'firstimage2', 'cpyrhtDivCd', 'mapx', 'mapy', 'mlevel', 'modifiedtime', 'sigungucode', 'title',
    'lDongRegnCd', 'lDongSignguCd', 'lclsSystm1', 'lclsSystm2', 'lclsSystm3', 'addr1', 'addr2', 'zipcode',
    'overview', 'homepage', 'distance', 'schedule', 'taketime', 'theme', 'subnum', 'subcontentid',
    'subname', 'subdetailoverview', 'subdetailimg'
]

TABLE_COLUMNS_MAP = {
    "festivals": FESTIVALS_COLUMNS,
    "facilities": FACILITIES_COLUMNS,
    "courses": COURSES_COLUMNS
}


def init_db(force_reset=False):
    """Initialize PostgreSQL database with all tables

    Args:
        force_reset: If True, drop all tables and recreate. Default False.
    """
    db_params = _parse_database_url(DATABASE_URL)
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    try:
        # Check if tables already exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'festivals'
            )
        """)
        tables_exist = cursor.fetchone()[0]

        if tables_exist and not force_reset:
            print("Database tables already exist. Skipping initialization.")
            cursor.close()
            conn.close()
            return

        print(f"Initializing PostgreSQL database: {DATABASE_URL}")

        # Drop existing tables (fresh start)
        print("Dropping existing tables...")
        tables_to_drop = [
            'answers', 'questions', 'feature_ratings', 'feedback',
            'user_events', 'refresh_tokens', 'users',
            'festivals', 'facilities', 'courses'
        ]
        for table in tables_to_drop:
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        conn.commit()

        # Create festivals table
        print("Creating festivals table...")
        festivals_cols = []
        for col in FESTIVALS_COLUMNS:
            if col in ['areacode', 'mlevel', 'sigungucode']:
                festivals_cols.append(f"{col} INTEGER")
            elif col in ['mapx', 'mapy']:
                festivals_cols.append(f"{col} DOUBLE PRECISION")
            else:
                festivals_cols.append(f"{col} TEXT")

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS festivals (
                id SERIAL PRIMARY KEY,
                {", ".join(festivals_cols)}
            )
        ''')
        print("Festivals table created.")

        # Create facilities table
        print("Creating facilities table...")
        facilities_cols = []
        for col in FACILITIES_COLUMNS:
            if col in ['areacode', 'mlevel', 'sigungucode']:
                facilities_cols.append(f"{col} INTEGER")
            elif col in ['mapx', 'mapy']:
                facilities_cols.append(f"{col} DOUBLE PRECISION")
            else:
                facilities_cols.append(f"{col} TEXT")

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS facilities (
                id SERIAL PRIMARY KEY,
                {", ".join(facilities_cols)}
            )
        ''')
        print("Facilities table created.")

        # Create courses table
        print("Creating courses table...")
        courses_cols = []
        for col in COURSES_COLUMNS:
            if col in ['areacode', 'mlevel', 'sigungucode']:
                courses_cols.append(f"{col} INTEGER")
            elif col in ['mapx', 'mapy']:
                courses_cols.append(f"{col} DOUBLE PRECISION")
            else:
                courses_cols.append(f"{col} TEXT")

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS courses (
                id SERIAL PRIMARY KEY,
                {", ".join(courses_cols)}
            )
        ''')
        print("Courses table created.")

        # Create users table
        print("Creating users table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                role VARCHAR(20) DEFAULT 'user',
                auth_provider VARCHAR(20) DEFAULT 'local',
                google_id VARCHAR(255),
                profile_picture TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        print("Users table created.")

        # Create refresh_tokens table
        print("Creating refresh_tokens table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_revoked BOOLEAN DEFAULT FALSE
            )
        ''')
        print("Refresh tokens table created.")

        # Create questions table (Q&A)
        print("Creating questions table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id SERIAL PRIMARY KEY,
                festival_name VARCHAR(255) NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                views INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("Questions table created.")

        # Create answers table
        print("Creating answers table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answers (
                id SERIAL PRIMARY KEY,
                question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id),
                content TEXT NOT NULL,
                is_accepted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("Answers table created.")

        # Create feedback table
        print("Creating feedback table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                page_url TEXT NOT NULL,
                festival_name VARCHAR(255),
                rating INTEGER NOT NULL,
                comment TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                session_id VARCHAR(255),
                user_id INTEGER REFERENCES users(id)
            )
        ''')
        print("Feedback table created.")

        # Create feature_ratings table
        print("Creating feature_ratings table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_ratings (
                id SERIAL PRIMARY KEY,
                festival_name VARCHAR(255) NOT NULL,
                feature_name VARCHAR(255) NOT NULL,
                rating INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(255),
                user_id INTEGER REFERENCES users(id)
            )
        ''')
        print("Feature ratings table created.")

        # Create user_events table
        print("Creating user_events table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_events (
                id SERIAL PRIMARY KEY,
                event_category VARCHAR(100) NOT NULL,
                event_action VARCHAR(100) NOT NULL,
                event_label TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(255),
                page_url TEXT,
                user_id INTEGER REFERENCES users(id),
                guest_id VARCHAR(100),
                username VARCHAR(100)
            )
        ''')
        print("User events table created.")

        # Create indexes for performance
        print("Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_festivals_title ON festivals(title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_festivals_areacode ON festivals(areacode)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_festival ON questions(festival_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_user ON questions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_festival ON feedback(festival_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_events_session ON user_events(session_id)")
        print("Indexes created.")

        conn.commit()
        print("Database initialized successfully.")

        # Load data from CSV files
        print("\nLoading data from CSV files...")
        load_data_to_db()

        # Create default admin account
        print("\nCreating default admin account...")
        create_default_users(conn, cursor)
        conn.commit()

    except Exception as e:
        print(f"Error during initialization: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
        print("Database initialization completed.")


def create_default_users(conn, cursor):
    """Create default admin and test user accounts"""
    import hashlib

    # Admin account
    admin_password = "admin123"
    admin_hash = hashlib.sha256(admin_password.encode()).hexdigest()

    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (username) DO NOTHING
        ''', ('admin', 'admin@festmoment.com', admin_hash, 'Administrator', 'admin'))
        print(f"Admin account created (username: admin, password: {admin_password})")
    except Exception as e:
        print(f"Admin account might already exist: {e}")

    # Test user account
    user_password = "user123"
    user_hash = hashlib.sha256(user_password.encode()).hexdigest()

    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (username) DO NOTHING
        ''', ('testuser', 'user@festmoment.com', user_hash, 'Test User', 'user'))
        print(f"Test user account created (username: testuser, password: {user_password})")
    except Exception as e:
        print(f"Test user account might already exist: {e}")


def load_data_to_db():
    """Load CSV data into PostgreSQL using pandas"""
    from sqlalchemy import create_engine

    print(f"Loading data into PostgreSQL...")

    # Create SQLAlchemy engine for pandas
    engine = create_engine(DATABASE_URL)

    try:
        for table_name, file_path in csv_files.items():
            if os.path.exists(file_path):
                print(f"Reading data from {file_path}...")

                # Read CSV with appropriate encoding
                df = pd.read_csv(file_path, encoding='cp949')
                print(f"DataFrame for '{table_name}' has {len(df)} rows and {len(df.columns)} columns.")

                # Convert column names to lowercase (PostgreSQL uses lowercase)
                df.columns = df.columns.str.lower()

                # Filter to schema columns (also lowercase)
                if table_name in TABLE_COLUMNS_MAP:
                    schema_columns = [col.lower() for col in TABLE_COLUMNS_MAP[table_name]]
                    df_filtered = df[[col for col in schema_columns if col in df.columns]]
                    print(f"Filtered DataFrame has {len(df_filtered.columns)} columns.")
                else:
                    df_filtered = df

                # Write to PostgreSQL
                df_filtered.to_sql(table_name, engine, if_exists='append', index=False)
                print(f"Successfully loaded {len(df_filtered)} rows into '{table_name}' table.")
            else:
                print(f"Warning: File not found at {file_path}")

    except Exception as e:
        print(f"Error during data loading: {e}")
        raise
    finally:
        engine.dispose()
        print("Data loading completed.")


def close_pool():
    """Close all connections in pool (call on app shutdown)"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        print("Connection pool closed.")


if __name__ == "__main__":
    init_db()
