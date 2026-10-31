"""
Open-Meteo Air Quality API integration client for AeroSense.
Fetches real-time ambient pollutant measurements and multi-day hourly forecasts.
"""

from typing import Dict, List, Optional, Tuple, Any
import requests
import pandas as pd
from datetime import datetime

# Open-Meteo Air Quality API Endpoint
API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Pre-configured major Indian metropolitan and tier-1 cities
PRESET_CITIES: Dict[str, Dict[str, float]] = {
    "New Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab / Haryana"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"}
}


def fetch_open_meteo_air_quality(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
    timeout_sec: int = 12
) -> Dict[str, Any]:
    """
    Connects to the Open-Meteo Air Quality API and retrieves live measurements
    and hourly forecasts.
    Returns:
        Structured response dictionary with current readings, hourly DataFrame,
        timestamps, units, and status metadata.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "european_aqi",
            "us_aqi"
        ],
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone"
        ],
        "timezone": "auto",
        "forecast_days": forecast_days
    }

    try:
        response = requests.get(API_URL, params=params, timeout=timeout_sec)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API request returned status code {response.status_code}: {response.text}",
                "current": {},
                "hourly_df": pd.DataFrame(),
                "units": {}
            }

        data = response.json()

        # Parse units
        hourly_units = data.get("hourly_units", {})
        current_data = data.get("current", {})
        hourly_data = data.get("hourly", {})

        # Standardized current reading extraction
        current_formatted = {
            "PM2.5": current_data.get("pm2_5"),
            "PM10": current_data.get("pm10"),
            "NO2": current_data.get("nitrogen_dioxide"),
            "SO2": current_data.get("sulphur_dioxide"),
            "CO": current_data.get("carbon_monoxide"),
            "O3": current_data.get("ozone"),
            "us_aqi": current_data.get("us_aqi"),
            "european_aqi": current_data.get("european_aqi"),
            "time": current_data.get("time")
        }

        # Build hourly DataFrame
        if "time" in hourly_data:
            df_hourly = pd.DataFrame({
                "Timestamp": pd.to_datetime(hourly_data["time"]),
                "PM2.5": hourly_data.get("pm2_5", []),
                "PM10": hourly_data.get("pm10", []),
                "NO2": hourly_data.get("nitrogen_dioxide", []),
                "SO2": hourly_data.get("sulphur_dioxide", []),
                "CO": hourly_data.get("carbon_monoxide", []),
                "O3": hourly_data.get("ozone", [])
            })
        else:
            df_hourly = pd.DataFrame()

        return {
            "success": True,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "elevation": data.get("elevation"),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current": current_formatted,
            "hourly_df": df_hourly,
            "units": hourly_units,
            "attribution": "Data provided by Open-Meteo Air Quality API (Copernicus CAMS & GFS model data)."
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "The Open-Meteo API connection timed out. Please verify your internet connection.",
            "current": {},
            "hourly_df": pd.DataFrame(),
            "units": {}
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Network error communicating with Open-Meteo API: {str(e)}",
            "current": {},
            "hourly_df": pd.DataFrame(),
            "units": {}
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error parsing API response: {str(e)}",
            "current": {},
            "hourly_df": pd.DataFrame(),
            "units": {}
        }
