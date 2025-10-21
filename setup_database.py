import pandas as pd
import sqlite3
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Define file paths relative to the script's directory
excel_files = {
    "festivals": os.path.join(script_dir, "database", "축제공연행사csv.csv"),
    "facilities": os.path.join(script_dir, "database", "문화시설csv.csv"),
    "courses": os.path.join(script_dir, "database", "여행코스csv.csv")
}

db_path = os.path.join(script_dir, "tour.db")

# Function to load excel data into sqlite
def load_data_to_db():
    print(f"Attempting to create/update database at: {db_path}")
    # Remove the old database file if it exists to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed old database file: {db_path}")

    # Create a new database connection
    conn = sqlite3.connect(db_path)
    print(f"Created and connected to database: {db_path}")

    try:
        # Process each excel file
        for table_name, file_path in excel_files.items():
            if os.path.exists(file_path):
                print(f"Reading data from {file_path}...")
                # Read csv file into a pandas DataFrame with specified encoding
                df = pd.read_csv(file_path, encoding='cp949') # Added encoding
                print(f"DataFrame for '{table_name}' has {len(df)} rows and {len(df.columns)} columns.")
                
                # Write the DataFrame to the sqlite database
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Successfully loaded {len(df)} rows into '{table_name}' table.")
            else:
                print(f"Error: File not found at {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    load_data_to_db()
