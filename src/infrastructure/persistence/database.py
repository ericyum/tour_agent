import pandas as pd
import sqlite3
import os
import re # re 모듈 추가

# Get the directory of the current script
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define file paths relative to the project root
excel_files = {
    "festivals": os.path.join(project_root, "database", "축제공연행사csv.csv"),
    "facilities": os.path.join(project_root, "database", "문화시설csv.csv"),
    "courses": os.path.join(project_root, "database", "여행코스csv.csv")
}

db_path = os.path.join(project_root, "tour.db")

# 각 테이블의 컬럼 정의 (init_db에서 사용될 것)
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

# 컬럼 정의 맵
TABLE_COLUMNS_MAP = {
    "festivals": FESTIVALS_COLUMNS,
    "facilities": FACILITIES_COLUMNS,
    "courses": COURSES_COLUMNS
}

def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    print(f"Attempting to initialize database at: {db_path}")
    # Remove the old database file if it exists to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed old database file: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Connected to database for table creation: {db_path}")

    # Create tables
    print("Creating festivals table...")
    # FESTIVALS_COLUMNS를 사용하여 SQL 쿼리 생성
    festivals_cols_sql = ", ".join([f"{col} TEXT" if col not in ['areacode', 'mapx', 'mapy', 'mlevel', 'sigungucode'] else f"{col} INTEGER" if col in ['areacode', 'mlevel', 'sigungucode'] else f"{col} REAL" for col in FESTIVALS_COLUMNS])
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS festivals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {festivals_cols_sql}
        )
    ''')
    print("Festivals table created.")

    print("Creating facilities table...")
    # FACILITIES_COLUMNS를 사용하여 SQL 쿼리 생성
    facilities_cols_sql = ", ".join([f"{col} TEXT" if col not in ['areacode', 'mapx', 'mapy', 'mlevel', 'sigungucode'] else f"{col} INTEGER" if col in ['areacode', 'mlevel', 'sigungucode'] else f"{col} REAL" for col in FACILITIES_COLUMNS])
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS facilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {facilities_cols_sql}
        )
    ''')
    print("Facilities table created.")

    print("Creating courses table...")
    # COURSES_COLUMNS를 사용하여 SQL 쿼리 생성
    courses_cols_sql = ", ".join([f"{col} TEXT" if col not in ['areacode', 'mapx', 'mapy', 'mlevel', 'sigungucode'] else f"{col} INTEGER" if col in ['areacode', 'mlevel', 'sigungucode'] else f"{col} REAL" for col in COURSES_COLUMNS])
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {courses_cols_sql}
        )
    ''')
    print("Courses table created.")

    conn.commit()
    conn.close()
    print(f"Database '{db_path}' initialized with necessary tables.")

    # Load data after creating tables
    print("Calling load_data_to_db()...\n")
    load_data_to_db()
    print("\nload_data_to_db() finished.")


# Function to load excel data into sqlite
def load_data_to_db():
    print(f"Attempting to load data into database at: {db_path}")
    conn = sqlite3.connect(db_path)
    print(f"Connected to database for data loading: {db_path}")

    try:
        # Process each excel file
        for table_name, file_path in excel_files.items():
            if os.path.exists(file_path):
                print(f"Reading data from {file_path}...")
                # Read csv file into a pandas DataFrame with specified encoding
                df = pd.read_csv(file_path, encoding='cp949') # Added encoding
                print(f"DataFrame for '{table_name}' has {len(df)} rows and {len(df.columns)} columns.")
                
                # 테이블 스키마에 정의된 컬럼만 선택
                if table_name in TABLE_COLUMNS_MAP:
                    schema_columns = TABLE_COLUMNS_MAP[table_name]
                    # DataFrame에 실제로 존재하는 컬럼만 필터링
                    df_filtered = df[[col for col in schema_columns if col in df.columns]]
                    print(f"Filtered DataFrame for '{table_name}' has {len(df_filtered)} rows and {len(df_filtered.columns)} columns (after schema filtering).")
                else:
                    df_filtered = df # 스키마 정의가 없으면 필터링하지 않음

                # Write the DataFrame to the sqlite database
                df_filtered.to_sql(table_name, conn, if_exists='append', index=False) # Changed to 'append'
                print(f"Successfully loaded {len(df_filtered)} rows into '{table_name}' table.")
            else:
                print(f"Error: File not found at {file_path}")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
    finally:
        # Close the database connection
        conn.close()
        print("Database connection closed after data loading.")

if __name__ == "__main__":
    init_db() # Call init_db to create tables and load data
