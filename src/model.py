"""
Machine learning pipeline definition, training, evaluation,
and persistence module for AeroSense.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_model_pipeline(model_type: str = "random_forest", random_state: int = 42) -> Pipeline:
    """
    Creates an end-to-end scikit-learn Pipeline with median imputation
    fitted strictly on training data to prevent leakage.
    """
    imputer = SimpleImputer(strategy="median")

    if model_type == "ridge":
        scaler = StandardScaler()
        estimator = Ridge(alpha=1.0, random_state=random_state)
        return Pipeline([
            ("imputer", imputer),
            ("scaler", scaler),
            ("regressor", estimator)
        ])

    elif model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        return Pipeline([
            ("imputer", imputer),
            ("regressor", estimator)
        ])

    elif model_type == "hist_gbdt":
        estimator = HistGradientBoostingRegressor(
            max_iter=120,
            max_depth=8,
            learning_rate=0.08,
            random_state=random_state
        )
        return Pipeline([
            ("imputer", imputer),
            ("regressor", estimator)
        ])

    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'ridge', 'random_forest', or 'hist_gbdt'.")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates standard regression performance metrics.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Safe MAPE calculation (avoiding division by zero)
    non_zero_mask = (y_true != 0)
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2)
    }


def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Extracts feature importance scores from the regressor inside the pipeline.
    Uses native tree importance for Random Forest, normalized coefficients for Ridge,
    and permutation importance for HistGradientBoosting.
    """
    regressor = pipeline.named_steps["regressor"]
    importances: Dict[str, float] = {}

    try:
        if hasattr(regressor, "feature_importances_"):
            raw_scores = regressor.feature_importances_
            total = np.sum(raw_scores)
            norm_scores = raw_scores / total if total > 0 else raw_scores
            for name, score in zip(feature_names, norm_scores):
                importances[name] = round(float(score) * 100, 2)

        elif hasattr(regressor, "coef_"):
            raw_scores = np.abs(regressor.coef_)
            total = np.sum(raw_scores)
            norm_scores = raw_scores / total if total > 0 else raw_scores
            for name, score in zip(feature_names, norm_scores):
                importances[name] = round(float(score) * 100, 2)

        elif X_val is not None and y_val is not None:
            # HistGradientBoosting doesn't have feature_importances_ by default; use permutation importance
            perm = permutation_importance(pipeline, X_val, y_val, n_repeats=5, random_state=42)
            raw_scores = np.maximum(perm.importances_mean, 0)
            total = np.sum(raw_scores)
            norm_scores = raw_scores / total if total > 0 else raw_scores
            for name, score in zip(feature_names, norm_scores):
                importances[name] = round(float(score) * 100, 2)
        else:
            # Fallback equal distribution
            for name in feature_names:
                importances[name] = round(100.0 / len(feature_names), 2)

    except Exception:
        for name in feature_names:
            importances[name] = round(100.0 / len(feature_names), 2)

    # Sort descending
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str]
) -> Tuple[Dict[str, Any], str, Pipeline]:
    """
    Trains Ridge Baseline, Random Forest, and HistGradientBoosting regressors.
    Evaluates their performance on the held-out test split,
    selects the best model, and returns comprehensive comparisons.
    """
    candidate_types = [
        ("Baseline Ridge Regression", "ridge"),
        ("Random Forest Regressor", "random_forest"),
        ("Histogram Gradient Boosting", "hist_gbdt")
    ]

    results: Dict[str, Any] = {}
    best_model_name = ""
    best_score = -float("inf")
    best_pipeline = None

    for display_name, m_type in candidate_types:
        pipeline = create_model_pipeline(m_type)
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_metrics = compute_metrics(y_train.to_numpy(), y_train_pred)
        test_metrics = compute_metrics(y_test.to_numpy(), y_test_pred)

        feat_importance = extract_feature_importance(pipeline, feature_names, X_test, y_test)

        results[display_name] = {
            "model_type": m_type,
            "pipeline": pipeline,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feat_importance,
            "predictions_sample": {
                "actual": [round(float(v), 1) for v in y_test.iloc[:50].tolist()],
                "predicted": [round(float(v), 1) for v in y_test_pred[:50].tolist()]
            }
        }

        # Select model with highest test R²
        if test_metrics["R2"] > best_score:
            best_score = test_metrics["R2"]
            best_model_name = display_name
            best_pipeline = pipeline

    return results, best_model_name, best_pipeline


def save_model_artifacts(
    pipeline: Pipeline,
    model_name: str,
    feature_names: List[str],
    all_results: Dict[str, Any],
    dataset_info: Dict[str, Any],
    output_dir: str = "models"
) -> str:
    """
    Serializes the winning pipeline and saves comprehensive JSON metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    pipeline_path = os.path.join(output_dir, "trained_pipeline.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    # Serialize pipeline
    joblib.dump(pipeline, pipeline_path)

    # Prepare serializable metadata
    serializable_results = {}
    for name, data in all_results.items():
        serializable_results[name] = {
            "model_type": data["model_type"],
            "train_metrics": data["train_metrics"],
            "test_metrics": data["test_metrics"],
            "feature_importance": data["feature_importance"],
            "predictions_sample": data["predictions_sample"]
        }

    metadata = {
        "saved_at": datetime.now().isoformat(),
        "best_model_name": model_name,
        "feature_names": feature_names,
        "models_comparison": serializable_results,
        "dataset_info": dataset_info
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return pipeline_path


def load_model_artifacts(models_dir: str = "models") -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]], str]:
    """
    Safely loads the serialized pipeline and metadata.
    Returns:
        (pipeline, metadata, status_message)
    """
    pipeline_path = os.path.join(models_dir, "trained_pipeline.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(pipeline_path) or not os.path.exists(metadata_path):
        return None, None, f"Model artifacts not found in '{models_dir}'. Please train a model first."

    try:
        pipeline = joblib.load(pipeline_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return pipeline, metadata, "Model artifacts loaded successfully."
    except Exception as e:
        return None, None, f"Failed to load model artifacts: {str(e)}"
