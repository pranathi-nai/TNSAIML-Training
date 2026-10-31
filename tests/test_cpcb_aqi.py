"""
Unit tests for official CPCB NAQI calculation engine.
"""

import pytest
from src.cpcb_aqi import (
    calculate_sub_index,
    get_cpcb_category,
    calculate_overall_cpcb_aqi,
    BREAKPOINTS
)


def test_calculate_sub_index_breakpoints():
    """Verify interpolation exactly matches CPCB NAQI boundary specifications."""
    # PM2.5: 0-30 -> 0-50, 30.1-60 -> 51-100, 60.1-90 -> 101-200
    assert calculate_sub_index("PM2.5", 0.0) == 0.0
    assert calculate_sub_index("PM2.5", 30.0) == 50.0
    assert calculate_sub_index("PM2.5", 60.0) == 100.0
    assert calculate_sub_index("PM2.5", 90.0) == 200.0
    assert calculate_sub_index("PM2.5", 120.0) == 300.0
    assert calculate_sub_index("PM2.5", 250.0) == 400.0

    # PM10: 0-50 -> 0-50, 50.1-100 -> 51-100
    assert calculate_sub_index("PM10", 50.0) == 50.0
    assert calculate_sub_index("PM10", 100.0) == 100.0
    assert calculate_sub_index("PM10", 250.0) == 200.0


def test_invalid_and_negative_pollutant_values():
    """Ensure invalid, None, or negative values return None safely."""
    assert calculate_sub_index("PM2.5", -10.0) is None
    assert calculate_sub_index("PM2.5", None) is None
    assert calculate_sub_index("NON_EXISTENT_POLLUTANT", 50.0) is None


def test_cpcb_categories():
    """Verify correct categorization across standard scale."""
    cat, meta = get_cpcb_category(35.0)
    assert cat == "Good"
    assert meta["mask_recommended"] is False

    cat, meta = get_cpcb_category(85.0)
    assert cat == "Satisfactory"

    cat, meta = get_cpcb_category(150.0)
    assert cat == "Moderate"

    cat, meta = get_cpcb_category(250.0)
    assert cat == "Poor"
    assert meta["mask_recommended"] is True

    cat, meta = get_cpcb_category(350.0)
    assert cat == "Very Poor"

    cat, meta = get_cpcb_category(450.0)
    assert cat == "Severe"

    cat, meta = get_cpcb_category(600.0)
    assert cat == "Hazardous"


def test_calculate_overall_cpcb_aqi():
    """Verify overall AQI is governed by the highest sub-index with compliance checks."""
    pollutants = {
        "PM2.5": 110.0,   # Sub-index ~ 267 (Poor)
        "PM10": 120.0,    # Sub-index ~ 113 (Moderate)
        "NO2": 35.0,      # Sub-index ~ 44 (Good)
        "SO2": 15.0,      # Sub-index ~ 19 (Good)
        "CO": 0.8,        # Sub-index ~ 40 (Good)
        "O3": 45.0        # Sub-index ~ 45 (Good)
    }

    result = calculate_overall_cpcb_aqi(pollutants)

    assert result["is_cpcb_compliant"] is True
    assert result["prominent_pollutant"] == "PM2.5"
    assert result["category"] == "Poor"
    assert result["aqi"] > 200.0


def test_cpcb_non_compliance_warning():
    """CPCB requires at least 3 pollutants including at least 1 PM (PM2.5 or PM10)."""
    # Only 2 gaseous pollutants
    pollutants = {
        "NO2": 45.0,
        "SO2": 25.0
    }
    result = calculate_overall_cpcb_aqi(pollutants)
    assert result["is_cpcb_compliant"] is False
    assert "missing PM2.5/PM10" in result["compliance_note"]
