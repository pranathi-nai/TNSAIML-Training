"""
Generator for benchmark environmental dataset for AeroSense.
Generates multi-city daily time-series data with authentic seasonal patterns,
realistic meteorology, and strict CPCB NAQI ground-truth targets.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

# Ensure parent directory is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cpcb_aqi import calculate_overall_cpcb_aqi

np.random.seed(42)

def generate_benchmark_dataset(
    start_date: str = "2022-01-01",
    days: int = 730,
    output_path: str = "data/sample_aqi_data.csv"
):
    cities_profiles = {
        "Delhi": {
            "pm25_base": 110.0, "pm25_var": 45.0, "pm10_base": 210.0, "pm10_var": 70.0,
            "no2_base": 55.0, "so2_base": 18.0, "co_base": 1.6, "o3_base": 42.0,
            "winter_spike": 2.2, "monsoon_drop": 0.45
        },
        "Mumbai": {
            "pm25_base": 52.0, "pm25_var": 20.0, "pm10_base": 105.0, "pm10_var": 35.0,
            "no2_base": 38.0, "so2_base": 16.0, "co_base": 0.95, "o3_base": 30.0,
            "winter_spike": 1.4, "monsoon_drop": 0.35
        },
        "Bengaluru": {
            "pm25_base": 38.0, "pm25_var": 14.0, "pm10_base": 78.0, "pm10_var": 25.0,
            "no2_base": 28.0, "so2_base": 9.0, "co_base": 0.75, "o3_base": 34.0,
            "winter_spike": 1.25, "monsoon_drop": 0.50
        },
        "Hyderabad": {
            "pm25_base": 48.0, "pm25_var": 18.0, "pm10_base": 98.0, "pm10_var": 30.0,
            "no2_base": 32.0, "so2_base": 12.0, "co_base": 0.85, "o3_base": 36.0,
            "winter_spike": 1.35, "monsoon_drop": 0.45
        },
        "Kolkata": {
            "pm25_base": 75.0, "pm25_var": 30.0, "pm10_base": 145.0, "pm10_var": 50.0,
            "no2_base": 44.0, "so2_base": 14.0, "co_base": 1.15, "o3_base": 38.0,
            "winter_spike": 1.85, "monsoon_drop": 0.40
        }
    }

    start = datetime.strptime(start_date, "%Y-%m-%d")
    records = []

    for city, prof in cities_profiles.items():
        for d in range(days):
            current_date = start + timedelta(days=d)
            month = current_date.month

            # Seasonal factors
            # Winter (Nov, Dec, Jan, Feb): high inversion & emissions
            if month in [11, 12, 1, 2]:
                season_mult = prof["winter_spike"]
            # Monsoon (Jul, Aug, Sep): rain washout
            elif month in [7, 8, 9]:
                season_mult = prof["monsoon_drop"]
            # Pre-monsoon / Summer (Mar, Apr, May, Jun): higher dust (PM10)
            elif month in [3, 4, 5, 6]:
                season_mult = 1.1
            else:
                season_mult = 1.0

            # Generate concentrations with natural random variance
            pm25 = max(5.0, float(np.random.normal(prof["pm25_base"] * season_mult, prof["pm25_var"])))
            # PM10 is correlated with PM2.5 plus coarse road/soil dust
            dust_component = max(10.0, float(np.random.normal(50.0, 15.0)))
            pm10 = max(pm25 * 1.3, float(pm25 + dust_component))

            no2 = max(2.0, float(np.random.normal(prof["no2_base"] * (0.8 + 0.4 * (season_mult > 1)), 8.0)))
            so2 = max(1.0, float(np.random.normal(prof["so2_base"], 3.5)))
            co = max(0.1, float(np.random.normal(prof["co_base"] * (0.9 + 0.3 * (season_mult > 1)), 0.25)))
            o3 = max(5.0, float(np.random.normal(prof["o3_base"], 9.0)))

            # Introduce rare missing sensor values (~1.5% probability) typical of real CPCB data
            if np.random.rand() < 0.015:
                so2 = np.nan
            if np.random.rand() < 0.015:
                o3 = np.nan

            pollutant_inputs = {
                "PM2.5": pm25,
                "PM10": pm10,
                "NO2": no2,
                "SO2": so2,
                "CO": co,
                "O3": o3
            }

            # Calculate deterministic CPCB NAQI target
            cpcb_res = calculate_overall_cpcb_aqi(pollutant_inputs)
            aqi_val = cpcb_res["aqi"]
            category = cpcb_res["category"]

            records.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "City": city,
                "PM2.5": round(pm25, 1),
                "PM10": round(pm10, 1),
                "NO2": round(no2, 1),
                "SO2": round(so2, 1) if not np.isnan(so2) else np.nan,
                "CO": round(co, 2),
                "O3": round(o3, 1) if not np.isnan(o3) else np.nan,
                "AQI": round(aqi_val, 1) if aqi_val is not None else np.nan,
                "AQI_Bucket": category
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated benchmark dataset at {output_path}: {len(df)} rows, {len(df.columns)} columns.")

if __name__ == "__main__":
    generate_benchmark_dataset()
