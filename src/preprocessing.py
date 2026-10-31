"""
Data preprocessing, schema normalization, missing value handling,
and dataset validation module for AeroSense.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Standard core features used for ML modeling
DEFAULT_FEATURE_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET_COL = "AQI"

# Column alias dictionary for robust matching across Kaggle, CPCB, and sensor CSVs
COLUMN_ALIASES: Dict[str, List[str]] = {
    "PM2.5": [
        "pm2.5", "pm25", "pm2_5", "pm_2_5", "pm 2.5",
        "particulate matter 2.5", "pm2.5 (ug/m3)", "pm2.5 (µg/m³)",
        "pm25_concentration", "pm2.5_ugm3"
    ],
    "PM10": [
        "pm10", "pm_10", "pm 10", "particulate matter 10",
        "pm10 (ug/m3)", "pm10 (µg/m³)", "pm10_concentration", "pm10_ugm3"
    ],
    "NO2": [
        "no2", "nitrogen dioxide", "nitrogendioxide", "no_2",
        "no2 (ug/m3)", "no2 (µg/m³)", "no2_concentration"
    ],
    "SO2": [
        "so2", "sulphur dioxide", "sulfur dioxide", "so_2",
        "so2 (ug/m3)", "so2 (µg/m³)", "so2_concentration"
    ],
    "CO": [
        "co", "carbon monoxide", "carbonmonoxide",
        "co (mg/m3)", "co (ug/m3)", "co (µg/m³)", "co_concentration"
    ],
    "O3": [
        "o3", "ozone", "o_3", "o3 (ug/m3)", "o3 (µg/m³)", "o3_concentration"
    ],
    "AQI": [
        "aqi", "air_quality_index", "air quality index", "overall_aqi",
        "cpcb_aqi", "target", "calculated_aqi"
    ],
    "Date": [
        "date", "datetime", "timestamp", "time", "date_time",
        "reading_date", "day"
    ],
    "City": [
        "city", "location", "station", "station_name", "place",
        "city_name", "monitoring_station"
    ]
}

# Leaky columns that must never be used as predictive features
LEAKY_COLUMNS = {
    "aqi", "aqi_bucket", "aqi_category", "air_quality_status",
    "status", "target", "sub_index", "overall_aqi", "category"
}


def normalize_column_name(raw_name: str) -> Optional[str]:
    """
    Matches a single raw column name against known aliases.
    Strips symbols, extra whitespace, and performs case-insensitive matching.
    """
    clean_str = raw_name.strip().lower()
    # Check exact alias match
    for canonical, aliases in COLUMN_ALIASES.items():
        if clean_str == canonical.lower():
            return canonical
        for alias in aliases:
            if clean_str == alias:
                return canonical

    # Secondary fuzzy check: token matching
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in clean_str:
                return canonical

    return None


def normalize_dataframe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Renames DataFrame columns to standard canonical names where recognized.
    Returns:
        (renamed_df, rename_mapping)
    """
    df = df.copy()
    rename_mapping: Dict[str, str] = {}
    assigned_canonicals = set()

    for col in df.columns:
        canonical = normalize_column_name(str(col))
        if canonical and canonical not in assigned_canonicals:
            rename_mapping[col] = canonical
            assigned_canonicals.add(canonical)

    df.rename(columns=rename_mapping, inplace=True)
    return df, rename_mapping


def validate_and_clean_dataset(
    df: pd.DataFrame,
    min_valid_rows: int = 10
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates, standardizes, and cleans an environmental dataset.
    Performs:
    1. Column name normalization
    2. Duplicate observation removal
    3. Conversion to numeric values and removal of negative values
    4. Date parsing (if present)
    5. Missing value diagnostics
    """
    report: Dict[str, Any] = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "columns_mapped": {},
        "missing_summary": {},
        "duplicate_rows_dropped": 0,
        "negative_values_handled": 0,
        "date_parsed": False,
        "available_features": [],
        "has_target_aqi": False,
        "is_valid": True,
        "error_message": ""
    }

    if df is None or len(df) == 0:
        report["is_valid"] = False
        report["error_message"] = "Dataset is empty."
        return df, report

    # 1. Normalize columns
    df_clean, mapping = normalize_dataframe_columns(df)
    report["columns_mapped"] = mapping

    # Check for core pollutants
    detected_features = [f for f in DEFAULT_FEATURE_COLS if f in df_clean.columns]
    report["available_features"] = detected_features
    report["has_target_aqi"] = TARGET_COL in df_clean.columns

    if len(detected_features) < 2:
        report["is_valid"] = False
        report["error_message"] = (
            f"Insufficient pollutant features found. Recognized {len(detected_features)}: {detected_features}. "
            f"At least 2 core pollutants (e.g. PM2.5, PM10, NO2, SO2, CO, O3) are required."
        )
        return df_clean, report

    # 2. Duplicate detection
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    report["duplicate_rows_dropped"] = initial_len - len(df_clean)

    # 3. Numeric conversion & bounds checking
    numeric_targets = detected_features + ([TARGET_COL] if report["has_target_aqi"] else [])
    negatives_count = 0

    for col in numeric_targets:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        # Handle impossible negative concentrations
        neg_mask = df_clean[col] < 0
        negatives_count += int(neg_mask.sum())
        df_clean.loc[neg_mask, col] = np.nan

        # Extreme physically impossible capping for outliers (>5000 µg/m³ except CO)
        if col != "CO":
            extreme_mask = df_clean[col] > 3000.0
            df_clean.loc[extreme_mask, col] = np.nan
        else:
            extreme_mask = df_clean[col] > 500.0
            df_clean.loc[extreme_mask, col] = np.nan

    report["negative_values_handled"] = negatives_count

    # 4. Date parsing & sorting
    if "Date" in df_clean.columns:
        try:
            df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
            df_clean = df_clean.sort_values(by="Date").reset_index(drop=True)
            report["date_parsed"] = True
        except Exception:
            report["date_parsed"] = False

    # 5. Missing value summary
    missing_summary = {}
    for col in df_clean.columns:
        n_missing = int(df_clean[col].isna().sum())
        missing_summary[col] = {
            "missing_count": n_missing,
            "missing_percentage": round((n_missing / len(df_clean)) * 100, 2)
        }
    report["missing_summary"] = missing_summary

    # Ensure minimum valid rows
    if len(df_clean) < min_valid_rows:
        report["is_valid"] = False
        report["error_message"] = f"Dataset has only {len(df_clean)} rows, fewer than the minimum {min_valid_rows} required."

    return df_clean, report


def prepare_ml_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    drop_unlabeled: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Extracts feature matrix X and target series y while rigorously preventing data leakage.
    Discards leaky columns (such as categories or buckets) and ensures only specified
    pollutant features are passed to the model.
    """
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]

    # Verify no leaky columns accidentally included
    sanitized_features = [
        c for c in feature_cols
        if c.lower() not in LEAKY_COLUMNS and c != TARGET_COL
    ]

    if TARGET_COL in df.columns:
        valid_df = df.copy()
        if drop_unlabeled:
            valid_df = valid_df.dropna(subset=[TARGET_COL])
        X = valid_df[sanitized_features]
        y = valid_df[TARGET_COL]
        return X, y, sanitized_features
    else:
        return df[sanitized_features], None, sanitized_features


def split_environmental_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and test sets.
    If 'Date' column is present and properly sorted, performs a chronological split
    to avoid future-data leakage in air quality forecasts.
    Otherwise, falls back to a reproducible random split.
    """
    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        split_idx = int(len(df) * (1.0 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        return train_df, test_df
    else:
        return train_test_split(df, test_size=test_size, random_state=random_state)
