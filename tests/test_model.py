"""
Unit tests for model training pipelines, metric computations, and persistence.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.model import (
    create_model_pipeline,
    compute_metrics,
    train_and_compare_models,
    save_model_artifacts,
    load_model_artifacts
)
from src.predict import predict_air_quality, validate_input_values


def test_create_model_pipeline():
    """Verify pipelines instantiate with correct steps."""
    for m_type in ["ridge", "random_forest", "hist_gbdt"]:
        pipe = create_model_pipeline(m_type)
        assert "imputer" in pipe.named_steps
        assert "regressor" in pipe.named_steps


def test_compute_metrics():
    """Verify regression metrics calculation."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(6.67, 0.05)
    assert metrics["R2"] > 0.95


def test_model_training_and_persistence(tmp_path):
    """Verify end-to-end training and artifact loading."""
    X_train = pd.DataFrame({
        "PM2.5": [20, 50, 80, 120, 160, 220],
        "PM10": [40, 90, 130, 180, 240, 300]
    })
    y_train = pd.Series([35, 80, 140, 210, 270, 330])

    X_test = pd.DataFrame({
        "PM2.5": [30, 100],
        "PM10": [60, 150]
    })
    y_test = pd.Series([50, 180])

    results, best_name, best_pipeline = train_and_compare_models(
        X_train, y_train, X_test, y_test, ["PM2.5", "PM10"]
    )

    assert best_name in results
    assert best_pipeline is not None

    # Test saving
    output_dir = str(tmp_path / "models_test")
    save_model_artifacts(
        pipeline=best_pipeline,
        model_name=best_name,
        feature_names=["PM2.5", "PM10"],
        all_results=results,
        dataset_info={"rows": 8},
        output_dir=output_dir
    )

    # Test loading
    loaded_pipe, meta, status = load_model_artifacts(output_dir)
    assert loaded_pipe is not None
    assert meta["best_model_name"] == best_name

    # Test prediction
    pred_res = predict_air_quality(
        loaded_pipe,
        {"PM2.5": 45.0, "PM10": 85.0},
        ["PM2.5", "PM10"]
    )
    assert pred_res["success"] is True
    assert pred_res["ml_predicted_aqi"] > 0


def test_input_validation():
    """Verify validation boundaries and sanity checks."""
    # Negative input
    is_valid, errors, _ = validate_input_values({"PM2.5": -5.0})
    assert is_valid is False
    assert len(errors) > 0

    # PM2.5 > PM10 produces a scientific warning
    is_valid, errors, warnings = validate_input_values({"PM2.5": 200.0, "PM10": 100.0})
    assert is_valid is True
    assert any("PM2.5" in w for w in warnings)
