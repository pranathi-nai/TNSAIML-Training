"""
Unit tests for data validation, column normalization, and preprocessing pipelines.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    normalize_dataframe_columns,
    validate_and_clean_dataset,
    prepare_ml_data,
    split_environmental_data
)


def test_column_name_normalization():
    """Verify alias mapping correctly identifies variations of pollutant names."""
    messy_df = pd.DataFrame({
        "pm25_concentration": [45.0, 50.0],
        "Particulate Matter 10": [110.0, 120.0],
        "nitrogen dioxide": [35.0, 40.0],
        "SO2 (ug/m3)": [12.0, 15.0],
        "Carbon Monoxide": [0.8, 1.1],
        "O3": [25.0, 30.0],
        "Air Quality Index": [120, 140],
        "reading_date": ["2023-01-01", "2023-01-02"]
    })

    clean_df, mapping = normalize_dataframe_columns(messy_df)

    assert "PM2.5" in clean_df.columns
    assert "PM10" in clean_df.columns
    assert "NO2" in clean_df.columns
    assert "SO2" in clean_df.columns
    assert "CO" in clean_df.columns
    assert "O3" in clean_df.columns
    assert "AQI" in clean_df.columns
    assert "Date" in clean_df.columns


def test_validate_and_clean_dataset_removes_duplicates_and_negatives():
    """Verify handling of duplicates and impossible negative concentrations."""
    raw_df = pd.DataFrame({
        "PM2.5": [50.0, 50.0, -10.0, 75.0],   # duplicate row 1 and negative in row 2
        "PM10": [100.0, 100.0, 80.0, 150.0],
        "NO2": [25.0, 25.0, 30.0, 45.0],
        "SO2": [10.0, 10.0, 12.0, 15.0],
        "CO": [0.5, 0.5, 0.6, 0.8],
        "O3": [20.0, 20.0, 22.0, 28.0],
        "AQI": [110.0, 110.0, 95.0, 160.0]
    })

    cleaned_df, report = validate_and_clean_dataset(raw_df, min_valid_rows=2)

    assert report["is_valid"] is True
    assert report["duplicate_rows_dropped"] == 1
    assert report["negative_values_handled"] == 1
    # Negative value in PM2.5 should have been converted to NaN
    assert np.isnan(cleaned_df.loc[cleaned_df["NO2"] == 30.0, "PM2.5"].values[0])


def test_prepare_ml_data_prevents_target_leakage():
    """Verify leaky columns like category, bucket, or status are strictly excluded."""
    df = pd.DataFrame({
        "PM2.5": [30.0, 60.0, 90.0],
        "PM10": [50.0, 100.0, 150.0],
        "AQI": [50.0, 100.0, 200.0],
        "AQI_Bucket": ["Good", "Satisfactory", "Moderate"],
        "Air_Quality_Status": ["Clean", "Acceptable", "Polluted"],
        "Sub_Index": [50, 100, 200]
    })

    X, y, feature_names = prepare_ml_data(df)

    assert "AQI_Bucket" not in feature_names
    assert "Air_Quality_Status" not in feature_names
    assert "Sub_Index" not in feature_names
    assert "AQI" not in feature_names
    assert list(feature_names) == ["PM2.5", "PM10"]
    assert len(y) == 3


def test_chronological_data_split():
    """Verify chronological split retains temporal sequence without mixing future dates."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "PM2.5": range(10),
        "AQI": range(10)
    })

    train_df, test_df = split_environmental_data(df, test_size=0.3)

    assert len(train_df) == 7
    assert len(test_df) == 3
    # Check max train date is strictly earlier than min test date
    assert train_df["Date"].max() < test_df["Date"].min()
