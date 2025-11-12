import os
import json
import re
import pandas as pd
import sqlite3
from matplotlib import font_manager

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# --- Data Loading Functions ---


def load_icon_map():
    icon_map_path = os.path.join(PROJECT_ROOT, "best_images_and_icons", "icon_map.json")
    try:
        with open(icon_map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load icon_map.json. Error: {e}")
        return {}


def load_best_images_map():
    best_images_map_path = os.path.join(
        PROJECT_ROOT, "best_images_and_icons", "best_images_map.json"
    )
    try:
        with open(best_images_map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load best_images_map.json. Error: {e}")
        return {}


def load_festival_categories_and_maps():
    festivals_dir = os.path.join(PROJECT_ROOT, "festivals")
    all_categories = {}
    title_to_cat_names = {}
    cat_name_to_code = {"main": {}, "medium": {}, "small": {}}
    try:
        for filename in os.listdir(festivals_dir):
            if filename.endswith(".json"):
                with open(
                    os.path.join(festivals_dir, filename), "r", encoding="utf-8"
                ) as f:
                    all_categories.update(json.load(f))
    except Exception as e:
        print(f"Error loading festival categories: {e}")
        return {}, {}, {}

    for main_name, med_dict in all_categories.items():
        for med_name, small_dict in med_dict.items():
            for small_name, titles in small_dict.items():
                for title in titles:
                    title_to_cat_names[title] = (main_name, med_name, small_name)

    db_path = os.path.join(PROJECT_ROOT, "tour.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title, cat1, cat2, cat3 FROM festivals WHERE cat1 IS NOT NULL AND cat2 IS NOT NULL AND cat3 IS NOT NULL"
        )
        db_festivals = cursor.fetchall()
        conn.close()
        for row in db_festivals:
            title, code1, code2, code3 = row
            if title in title_to_cat_names:
                name1, name2, name3 = title_to_cat_names[title]
                cat_name_to_code["main"][name1] = code1
                cat_name_to_code["medium"][name2] = code2
                cat_name_to_code["small"][name3] = code3
    except Exception as e:
        print(f"Error reading database for category codes: {e}")

    print("[Loader] Festival category maps created.")
    return all_categories, title_to_cat_names, cat_name_to_code


def get_korean_font():
    try:
        font_path = font_manager.findfont(
            font_manager.FontProperties(family="Malgun Gothic")
        )
        if os.path.exists(font_path):
            return font_path
    except:
        pass
    font_list = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in font_list:
        if (
            "gothic" in font.lower()
            or "gulim" in font.lower()
            or "apple" in font.lower()
        ):
            return font
    print("Warning: Korean font not found. Visualization text may be broken.")
    return None


def load_festival_info_lookup():
    """Loads the precaution info from the classification CSV and DB info."""
    festival_info = {}
    csv_precautions = {}

    # Load from CSV
    try:
        csv_path = os.path.join(PROJECT_ROOT, "festival_final_classification.csv")
        # encoding_sig handles BOM automatically
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # Strip whitespace from festival names
        df['festival_name'] = df['festival_name'].str.strip()

        # Store CSV data separately for matching
        for _, row in df.iterrows():
            csv_name = row['festival_name']
            # Handle NaN values
            detailed_cat = row['detailed_category'] if pd.notna(row['detailed_category']) else ""
            prohibited = row['prohibited_behaviors'] if pd.notna(row['prohibited_behaviors']) else ""

            csv_precautions[csv_name] = {
                "detailed_category": detailed_cat,
                "prohibited_behaviors": prohibited
            }

        print(f"[Loader] Loaded {len(csv_precautions)} festivals from CSV for precautions")
    except FileNotFoundError:
        print(
            "Warning: festival_final_classification.csv not found. Precaution feature will be disabled."
        )
    except Exception as e:
        print(f"Error loading festival_final_classification.csv: {e}")

    # Load from DB and match with CSV
    db_path = os.path.join(PROJECT_ROOT, "tour.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title, eventstartdate, eventenddate, addr1, tel, homepage, mapx, mapy, contentid FROM festivals"
        )
        db_festivals = cursor.fetchall()
        conn.close()

        matched_count = 0
        for row in db_festivals:
            db_title = row[0]
            if db_title:
                db_title = db_title.strip()

                # Initialize entry with DB info
                festival_info[db_title] = {
                    "eventstartdate": row[1],
                    "eventenddate": row[2],
                    "addr1": row[3],
                    "tel": row[4],
                    "homepage": row[5],
                    "mapx": row[6],
                    "mapy": row[7],
                    "contentid": row[8],
                }

                # Try to match with CSV data
                # First try exact match
                if db_title in csv_precautions:
                    festival_info[db_title].update(csv_precautions[db_title])
                    matched_count += 1
                else:
                    # Try partial match (remove year prefix)
                    # e.g., "2023 제20회 축제" -> "제20회 축제"
                    # Remove leading year pattern (e.g., "2023 ", "2024 ")
                    title_without_year = re.sub(r'^\d{4}\s+', '', db_title)

                    if title_without_year != db_title and title_without_year in csv_precautions:
                        festival_info[db_title].update(csv_precautions[title_without_year])
                        matched_count += 1
                        # Debug log for first 5 matches
                        if matched_count <= 5:
                            print(f"[Loader] Matched (year removed): '{db_title}' -> '{title_without_year}'")
                    else:
                        # Try matching by checking if CSV name is contained in DB name
                        matched_in_substring = False
                        for csv_name, csv_data in csv_precautions.items():
                            # Check if CSV name is a substring of DB name (ignoring case)
                            if csv_name in db_title or db_title.replace(' ', '') == csv_name.replace(' ', ''):
                                festival_info[db_title].update(csv_data)
                                matched_count += 1
                                matched_in_substring = True
                                # Debug log for first 5 matches
                                if matched_count <= 5:
                                    print(f"[Loader] Matched (substring): '{db_title}' contains '{csv_name}'")
                                break

        print(f"[Loader] Matched {matched_count} festivals with CSV precautions")
        print(f"[Loader] Total festivals in lookup: {len(festival_info)}")
    except Exception as e:
        print(f"Error loading festival info from database: {e}")
        import traceback
        traceback.print_exc()

    return festival_info


# --- [ 신규 추가된 함수 ] ---
def load_rendering_data():
    """AI 렌더링에 필요한 CSV 데이터를 로드합니다."""
    # CSV 파일들이 'database' 폴더에 있다고 가정합니다.
    split_path = os.path.join(PROJECT_ROOT, "database", "festival_condition_split.csv")
    camera_path = os.path.join(
        PROJECT_ROOT, "database", "festivals_camera_angle_all.csv"
    )

    df_split = pd.DataFrame()
    df_camera = pd.DataFrame()

    try:
        df_split = pd.read_csv(split_path)
        print(f"[Loader] Loaded {len(df_split)} rows from festival_condition_split.csv")
    except FileNotFoundError:
        print(f"!!! CRITICAL WARNING: {split_path} not found. AI Rendering will fail.")
    except Exception as e:
        print(f"Error loading {split_path}: {e}")

    try:
        df_camera = pd.read_csv(camera_path)
        print(
            f"[Loader] Loaded {len(df_camera)} rows from festivals_camera_angle_all.csv"
        )
    except FileNotFoundError:
        print(f"!!! CRITICAL WARNING: {camera_path} not found. AI Rendering will fail.")
    except Exception as e:
        print(f"Error loading {camera_path}: {e}")

    return df_split, df_camera


# --- [ 신규 함수 추가 끝 ] ---


# --- Global Constants Initialized on Import ---

print("Loading application configurations...")

ICON_MAP = load_icon_map()
BEST_IMAGES_MAP = load_best_images_map()
ALL_FESTIVAL_CATEGORIES, TITLE_TO_CAT_NAMES, CAT_NAME_TO_CODE = (
    load_festival_categories_and_maps()
)
KOREAN_FONT_PATH = get_korean_font()
FESTIVAL_INFO_LOOKUP = load_festival_info_lookup()

# --- [ 신규 추가된 전역 변수 ] ---
DF_SPLIT, DF_CAMERA = load_rendering_data()
# --- [ 신규 전역 변수 추가 끝 ] ---

print("Configuration loading complete.")
