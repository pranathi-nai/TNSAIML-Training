"""
Standalone CLI training script for AeroSense ML models.
Executes data preprocessing, cross-model training, chronological evaluation,
and saves the best model pipeline and evaluation metadata.

Usage:
    python scripts/train_model.py --data data/sample_aqi_data.csv --output models/
"""

import os
import sys
import argparse
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    validate_and_clean_dataset,
    prepare_ml_data,
    split_environmental_data,
    DEFAULT_FEATURE_COLS
)
from src.model import train_and_compare_models, save_model_artifacts


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Train and evaluate AeroSense AQI prediction models.")
    parser.add_argument("--data", type=str, default="data/sample_aqi_data.csv", help="Path to input training CSV")
    parser.add_argument("--output", type=str, default="models", help="Directory to save trained model artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion (default: 0.2)")
    args = parser.parse_args()

    print("=" * 65)
    print(">> AeroSense: Machine Learning Model Training Pipeline")
    print("=" * 65)

    if not os.path.exists(args.data):
        print(f"[ERROR] Data file '{args.data}' not found.")
        sys.exit(1)

    print(f"[DATA] Loading dataset from: {args.data}")
    df_raw = pd.read_csv(args.data)
    print(f"       Raw dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 1. Validation & Preprocessing
    print("\n[STEP 1] Cleaning & Normalizing Schema...")
    df_clean, report = validate_and_clean_dataset(df_raw)

    if not report["is_valid"]:
        print(f"[ERROR] Preprocessing failed: {report['error_message']}")
        sys.exit(1)

    print(f"       Columns recognized: {report['columns_mapped']}")
    print(f"       Dropped duplicate rows: {report['duplicate_rows_dropped']}")
    print(f"       Negative/extreme values handled: {report['negative_values_handled']}")
    print(f"       Chronological Date parsed: {report['date_parsed']}")
    print(f"       Available pollutant features: {report['available_features']}")

    if not report["has_target_aqi"]:
        print("[ERROR] Dataset does not contain a recognized ground-truth 'AQI' target column.")
        sys.exit(1)

    # 2. Train-Test Splitting
    print(f"\n[STEP 2] Performing Chronological Data Splitting (Test Size = {args.test_size})...")
    train_df, test_df = split_environmental_data(df_clean, test_size=args.test_size)
    print(f"       Training set: {len(train_df)} observations")
    print(f"       Testing set:  {len(test_df)} observations")

    # 3. Feature Matrix Preparation (Prevent Target Leakage)
    X_train, y_train, feature_names = prepare_ml_data(train_df, report["available_features"])
    X_test, y_test, _ = prepare_ml_data(test_df, report["available_features"])
    print(f"       Model features ({len(feature_names)}): {feature_names}")
    print("       Target: AQI (Continuous continuous index)")

    # 4. Multi-Model Training & Evaluation
    print("\n[STEP 3] Training & Benchmarking Candidate Regressors...")
    results, best_model_name, best_pipeline = train_and_compare_models(
        X_train, y_train, X_test, y_test, feature_names
    )

    print("\n" + "=" * 65)
    print(f"{'Model Architecture':<32} | {'Test MAE':<9} | {'Test RMSE':<9} | {'Test R2':<8}")
    print("-" * 65)
    for name, res in results.items():
        tm = res["test_metrics"]
        star = " *" if name == best_model_name else ""
        print(f"{name + star:<32} | {tm['MAE']:<9.2f} | {tm['RMSE']:<9.2f} | {tm['R2']:<8.4f}")
    print("=" * 65)

    print(f"\n[BEST] Selected Model: {best_model_name}")

    # 5. Save Artifacts
    print(f"\n[STEP 4] Saving Artifacts to '{args.output}'...")
    dataset_info = {
        "source_file": args.data,
        "total_rows": len(df_clean),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": feature_names,
        "date_range": [
            str(df_clean["Date"].min()) if "Date" in df_clean.columns else "N/A",
            str(df_clean["Date"].max()) if "Date" in df_clean.columns else "N/A"
        ],
        "cities": df_clean["City"].unique().tolist() if "City" in df_clean.columns else []
    }

    saved_path = save_model_artifacts(
        pipeline=best_pipeline,
        model_name=best_model_name,
        feature_names=feature_names,
        all_results=results,
        dataset_info=dataset_info,
        output_dir=args.output
    )

    print(f"       Pipeline saved to: {saved_path}")
    print(f"       Metadata saved to: {os.path.join(args.output, 'model_metadata.json')}")
    print("\n[SUCCESS] Training pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
