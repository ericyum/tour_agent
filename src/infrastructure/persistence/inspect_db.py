import sqlite3
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

db_path = os.path.join(script_dir, "tour.db")

print(f"Inspecting database at: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Schema for 'festivals' table:")
cursor.execute("PRAGMA table_info(festivals)")
for row in cursor.fetchall():
    print(row)

conn.close()
