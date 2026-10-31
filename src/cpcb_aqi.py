"""
CPCB (Central Pollution Control Board - India) National Air Quality Index (NAQI) Module.
Provides deterministic calculation of pollutant sub-indices, overall AQI determination,
and official health advisory statements based on Indian standards.
"""

from typing import Dict, Optional, Tuple, Any

# Breakpoint table based on official CPCB NAQI guidelines:
# Each entry is a list of tuples: (B_lo, B_hi, I_lo, I_hi)
BREAKPOINTS: Dict[str, list] = {
    "PM2.5": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "PM10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "NO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 600.0, 401, 500),
    ],
    "SO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2000.0, 401, 500),
    ],
    "CO": [
        # Note: In mg/m3. If input is given in µg/m3, it will be converted automatically.
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "O3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
    "NH3": [
        (0.0, 200.0, 0, 50),
        (200.1, 400.0, 51, 100),
        (401.0, 800.0, 101, 200),
        (801.0, 1200.0, 201, 300),
        (1201.0, 1800.0, 301, 400),
        (1801.0, 2400.0, 401, 500),
    ]
}

# CPCB Category Metadata
CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Good": {
        "range": (0, 50),
        "color": "#10b981",       # Emerald green
        "badge_bg": "#064e3b",
        "health_impact": "Minimal impact. Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "advisory": "Outdoor activities can be enjoyed normally by all individuals.",
        "mask_recommended": False,
    },
    "Satisfactory": {
        "range": (51, 100),
        "color": "#84cc16",       # Lime green
        "badge_bg": "#365314",
        "health_impact": "Minor breathing discomfort to sensitive individuals (elderly, children, asthmatics).",
        "advisory": "Sensitive people should monitor symptoms; general public can continue outdoor routines.",
        "mask_recommended": False,
    },
    "Moderate": {
        "range": (101, 200),
        "color": "#eab308",       # Amber yellow
        "badge_bg": "#713f12",
        "health_impact": "Breathing discomfort to the people with lungs, asthma and heart diseases.",
        "advisory": "People with lung diseases such as asthma and children should limit prolonged outdoor exertion.",
        "mask_recommended": False,
    },
    "Poor": {
        "range": (201, 300),
        "color": "#f97316",       # Orange
        "badge_bg": "#7c2d12",
        "health_impact": "Breathing discomfort to most people on prolonged exposure; noticeable respiratory strain.",
        "advisory": "Wear N95/FFP2 masks outdoors. Reduce heavy outdoor exercise. Keep windows closed during peak hours.",
        "mask_recommended": True,
    },
    "Very Poor": {
        "range": (301, 400),
        "color": "#ef4444",       # Bright red
        "badge_bg": "#7f1d1d",
        "health_impact": "Respiratory illness on prolonged exposure. Pronounced effect on people with lung and heart diseases.",
        "advisory": "Avoid morning walks and prolonged outdoor physical activities. Sensitive groups must remain indoors.",
        "mask_recommended": True,
    },
    "Severe": {
        "range": (401, 500),
        "color": "#991b1b",       # Deep crimson
        "badge_bg": "#450a0a",
        "health_impact": "Affects healthy people and seriously impacts those with existing diseases. Emergency-level health advisory.",
        "advisory": "Remain indoors. Keep air purifiers active. High-efficiency filtration masks mandatory outside.",
        "mask_recommended": True,
    },
    "Hazardous": {
        "range": (501, 9999),
        "color": "#701a75",       # Purple/Maroon
        "badge_bg": "#4a044e",
        "health_impact": "Beyond standard scale. Severe environmental danger. Life-threatening pollution levels for all demographics.",
        "advisory": "Immediate shelter-in-place with sealed air filtration. Avoid any outdoor presence.",
        "mask_recommended": True,
    }
}


def calculate_sub_index(pollutant: str, concentration: float) -> Optional[float]:
    """
    Computes CPCB sub-index for a specific pollutant concentration using linear interpolation.
    Formula:
        I_p = [ (I_hi - I_lo) / (B_hi - B_lo) ] * (C_p - B_lo) + I_lo
    """
    if concentration is None or concentration < 0:
        return None

    # Handle CO in mg/m3: if value is > 50, likely entered in µg/m3, so convert to mg/m3
    if pollutant == "CO" and concentration > 50:
        concentration = concentration / 1000.0

    table = BREAKPOINTS.get(pollutant)
    if not table:
        return None

    # Below minimum breakpoint
    if concentration < table[0][0]:
        return float(table[0][2])

    for b_lo, b_hi, i_lo, i_hi in table:
        if b_lo <= concentration <= b_hi:
            sub = ((i_hi - i_lo) / (b_hi - b_lo)) * (concentration - b_lo) + i_lo
            return round(sub, 1)

    # Above maximum breakpoint (extrapolate capped to 500 or slightly above)
    last_b_lo, last_b_hi, last_i_lo, last_i_hi = table[-1]
    if concentration > last_b_hi:
        # Linear slope beyond max
        sub = last_i_hi + ((concentration - last_b_hi) / (last_b_hi - last_b_lo)) * (last_i_hi - last_i_lo)
        return round(min(sub, 750.0), 1)

    return None


def get_cpcb_category(aqi_val: float) -> Tuple[str, Dict[str, Any]]:
    """
    Returns the category name and its metadata dictionary for a given AQI value.
    """
    if aqi_val is None or aqi_val < 0:
        return "Unknown", {
            "range": (0, 0),
            "color": "#94a3b8",
            "badge_bg": "#334155",
            "health_impact": "Data unavailable or invalid value.",
            "advisory": "Please verify sensor inputs.",
            "mask_recommended": False,
        }

    val = round(aqi_val)
    if val <= 50:
        return "Good", CATEGORIES["Good"]
    elif val <= 100:
        return "Satisfactory", CATEGORIES["Satisfactory"]
    elif val <= 200:
        return "Moderate", CATEGORIES["Moderate"]
    elif val <= 300:
        return "Poor", CATEGORIES["Poor"]
    elif val <= 400:
        return "Very Poor", CATEGORIES["Very Poor"]
    elif val <= 500:
        return "Severe", CATEGORIES["Severe"]
    else:
        return "Hazardous", CATEGORIES["Hazardous"]


def calculate_overall_cpcb_aqi(pollutant_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculates official CPCB AQI from a dictionary of pollutant values.
    Requirements per CPCB standard:
    - At least 3 pollutants must be monitored.
    - At least one of the 3 pollutants must be PM2.5 or PM10.
    Overall AQI = max(sub_indices).
    """
    sub_indices: Dict[str, float] = {}
    for pollutant, value in pollutant_dict.items():
        if value is not None and value >= 0:
            sub = calculate_sub_index(pollutant, value)
            if sub is not None:
                sub_indices[pollutant] = sub

    has_pm = ("PM2.5" in sub_indices) or ("PM10" in sub_indices)
    valid_count = len(sub_indices)
    is_cpcb_compliant = (valid_count >= 3) and has_pm

    if not sub_indices:
        return {
            "aqi": None,
            "category": "Unknown",
            "meta": CATEGORIES["Good"],
            "sub_indices": {},
            "prominent_pollutant": None,
            "is_cpcb_compliant": False,
            "compliance_note": "No valid pollutant measurements provided."
        }

    # Find highest sub-index
    prominent = max(sub_indices, key=sub_indices.get)
    max_aqi = sub_indices[prominent]
    category, meta = get_cpcb_category(max_aqi)

    if not is_cpcb_compliant:
        note = f"Warning: Only {valid_count} pollutant(s) available"
        if not has_pm:
            note += " (missing PM2.5/PM10 required by CPCB protocol)"
        note += ". Result is an estimated sub-index, not a complete CPCB NAQI calculation."
    else:
        note = f"Fully compliant with CPCB NAQI 3+ pollutant standard. Driven primarily by {prominent}."

    return {
        "aqi": max_aqi,
        "category": category,
        "meta": meta,
        "sub_indices": sub_indices,
        "prominent_pollutant": prominent,
        "is_cpcb_compliant": is_cpcb_compliant,
        "compliance_note": note
    }
