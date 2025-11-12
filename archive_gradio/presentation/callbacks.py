import gradio as gr

def update_sigungu(area: str, sigungu_code_map: dict):
    """Updates the sigungu dropdown choices based on the selected area."""
    choices = (
        ["전체"] + sorted(list(sigungu_code_map.get(area, {}).keys()))
        if area != "전체"
        else ["전체"]
    )
    return gr.update(choices=choices, value="전체")

def update_medium_cat(main_cat: str, all_festival_categories: dict):
    """Updates the medium category dropdown based on the selected main category."""
    choices = (
        ["전체"] + sorted(list(all_festival_categories.get(main_cat, {}).keys()))
        if main_cat != "전체"
        else ["전체"]
    )
    return gr.update(choices=choices, value="전체")

def update_small_cat(main_cat: str, medium_cat: str, all_festival_categories: dict):
    """Updates the small category dropdown based on the selected medium category."""
    choices = (
        ["전체"]
        + sorted(
            list(
                all_festival_categories.get(main_cat, {}).get(medium_cat, {}).keys()
            )
        )
        if main_cat != "전체" and medium_cat != "전체"
        else ["전체"]
    )
    return gr.update(choices=choices, value="전체")
