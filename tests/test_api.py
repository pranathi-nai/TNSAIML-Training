"""
Unit tests for API parameter formatting, mock handling, and preset city registries.
"""

import pytest
from src.api_client import PRESET_CITIES, fetch_open_meteo_air_quality


def test_preset_cities_registry():
    """Verify essential Indian metropolitan hubs are registered with valid coordinates."""
    assert "New Delhi" in PRESET_CITIES
    assert "Mumbai" in PRESET_CITIES
    assert "Bengaluru" in PRESET_CITIES
    assert "Hyderabad" in PRESET_CITIES
    assert "Kolkata" in PRESET_CITIES

    for city, coords in PRESET_CITIES.items():
        assert -90.0 <= coords["lat"] <= 90.0
        assert -180.0 <= coords["lon"] <= 180.0
        assert "state" in coords


def test_api_network_failure_handling(monkeypatch):
    """Ensure graceful fallback when network fails or times out."""
    import requests

    def mock_get(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection timeout simulation")

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_open_meteo_air_quality(28.6139, 77.2090)
    assert result["success"] is False
    assert "timed out" in result["error"]
