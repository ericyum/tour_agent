import os
import json
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
    """Loads the precaution info from the classification CSV."""
    try:
        csv_path = os.path.join(PROJECT_ROOT, "festival_final_classification.csv")
        df = pd.read_csv(csv_path)
        return df.set_index("festival_name")[
            ["detailed_category", "prohibited_behaviors"]
        ].to_dict("index")
    except FileNotFoundError:
        print(
            "Warning: festival_final_classification.csv not found. Precaution feature will be disabled."
        )
        return {}


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
