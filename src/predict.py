"""
Prediction and inference engine for AeroSense.
Connects trained machine-learning pipelines with user inputs, validation,
and health advisory rules.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.cpcb_aqi import get_cpcb_category, calculate_overall_cpcb_aqi


# Typical realistic measurement ranges (for validation and UI guidelines)
POLLUTANT_BOUNDS = {
    "PM2.5": {"min": 0.0, "max": 1000.0, "unit": "µg/m³", "label": "Particulate Matter 2.5"},
    "PM10": {"min": 0.0, "max": 1500.0, "unit": "µg/m³", "label": "Particulate Matter 10"},
    "NO2": {"min": 0.0, "max": 1000.0, "unit": "µg/m³", "label": "Nitrogen Dioxide"},
    "SO2": {"min": 0.0, "max": 1000.0, "unit": "µg/m³", "label": "Sulphur Dioxide"},
    "CO": {"min": 0.0, "max": 100.0, "unit": "mg/m³", "label": "Carbon Monoxide"},
    "O3": {"min": 0.0, "max": 1000.0, "unit": "µg/m³", "label": "Ozone"}
}


def validate_input_values(inputs: Dict[str, float]) -> Tuple[bool, List[str], List[str]]:
    """
    Validates pollutant inputs against plausible physical thresholds.
    Returns:
        (is_valid, error_list, warning_list)
    """
    errors = []
    warnings = []

    for key, val in inputs.items():
        if val is None:
            continue
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            errors.append(f"Invalid non-numeric value for {key}: '{val}'")
            continue

        if val_f < 0:
            errors.append(f"{key} concentration cannot be negative ({val_f}).")

        # Check bounds warnings
        bounds = POLLUTANT_BOUNDS.get(key)
        if bounds:
            if val_f > bounds["max"]:
                warnings.append(
                    f"{key} reading of {val_f} {bounds['unit']} is exceptionally high (typical max is {bounds['max']} {bounds['unit']})."
                )

    # Sanity check: PM2.5 is a subset of PM10
    pm25 = inputs.get("PM2.5")
    pm10 = inputs.get("PM10")
    if pm25 is not None and pm10 is not None:
        if pm25 > pm10 and pm10 > 0:
            warnings.append(
                f"PM2.5 ({pm25} µg/m³) is higher than PM10 ({pm10} µg/m³). In ambient atmosphere, PM2.5 is a physical sub-fraction of PM10."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def predict_air_quality(
    pipeline: Any,
    input_pollutants: Dict[str, float],
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Executes inference using the trained ML pipeline and computes side-by-side
    deterministic CPCB sub-indices for transparency.
    """
    # 1. Validation
    is_valid, errors, warnings = validate_input_values(input_pollutants)
    if not is_valid:
        return {
            "success": False,
            "errors": errors,
            "warnings": warnings,
            "prediction": None
        }

    # 2. Build feature vector aligned with training feature names
    input_row = {}
    for feat in feature_names:
        input_row[feat] = input_pollutants.get(feat, np.nan)

    input_df = pd.DataFrame([input_row])

    # 3. Model Inference
    try:
        raw_pred = pipeline.predict(input_df)[0]
        # Realistic clamping: AQI cannot be negative
        predicted_aqi = max(0.0, round(float(raw_pred), 1))
    except Exception as e:
        return {
            "success": False,
            "errors": [f"Inference error: {str(e)}"],
            "warnings": warnings,
            "prediction": None
        }

    # 4. Get CPCB category and metadata for ML prediction
    ml_category, ml_meta = get_cpcb_category(predicted_aqi)

    # 5. Compute Deterministic CPCB benchmark calculation for comparison
    cpcb_result = calculate_overall_cpcb_aqi(input_pollutants)

    # 6. Compare ML prediction vs CPCB formula
    diff = None
    if cpcb_result["aqi"] is not None:
        diff = round(predicted_aqi - cpcb_result["aqi"], 1)

    return {
        "success": True,
        "errors": [],
        "warnings": warnings,
        "ml_predicted_aqi": predicted_aqi,
        "ml_category": ml_category,
        "ml_meta": ml_meta,
        "cpcb_benchmark": cpcb_result,
        "difference_ml_vs_cpcb": diff,
        "input_features": input_pollutants
    }
