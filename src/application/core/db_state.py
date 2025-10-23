from typing import TypedDict, List, Dict, Any

class DBSearchState(TypedDict):
    search_type: str
    
    # Inputs for festival search
    area: str | None
    sigungu: str | None
    main_cat: str | None
    medium_cat: str | None
    small_cat: str | None
    
    # Inputs for nearby search
    latitude: float | None
    longitude: float | None
    radius: float | None
    
    # Results
    results: List[Dict[str, Any]] | None
    recommended_facilities: List[Dict[str, Any]] | None
    recommended_courses: List[Dict[str, Any]] | None
