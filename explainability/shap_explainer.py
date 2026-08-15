"""
explainability/shap_explainer.py
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet"
CHURN_PRED_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "churn_predictions.parquet"
CLV_PRED_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "clv_predictions.parquet"
CHURN_ARTIFACTS_DIR = PROJECT_ROOT / "models" / "churn" / "artifacts"
CLV_ARTIFACTS_DIR = PROJECT_ROOT / "models" / "clv" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TOP_N_FEATURES = 5
NULLABLE_FEATURES = ["avg_review_score", "avg_delivery_days", "avg_delivery_delay"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("shap_explainer")


# Build the model input matrix using the feature columns saved during training
def build_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    # Remove identifiers and raw date fields that are not used as model inputs
    drop_cols = ["customer_unique_id", "recency_days", "first_purchase", "last_purchase"]
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    # Add missing-value indicators and fill missing numerical values
    for col in NULLABLE_FEATURES:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(X[col].median())

    # One-hot encode customer state and restore the exact training feature schema
    X = pd.get_dummies(X, columns=["customer_state"], prefix="state")
    X = X.reindex(columns=feature_cols, fill_value=0)
    return X


# Calculate SHAP values and normalize the output shape across model versions
def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Returns a (n_samples, n_features) array regardless of SHAP/model
    version quirks -- XGBoost binary classifiers sometimes return a 3D
    array (n_samples, n_features, n_classes); the positive-class slice is
    taken in that case."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


# Convert the most influential SHAP features for one customer into JSON
def top_features_json(
    shap_row: np.ndarray, feature_values: pd.Series, feature_names: list, top_n: int
) -> str:
    order = np.argsort(-np.abs(shap_row))[:top_n]
    items = [
        {
            "feature": feature_names[i],
            "shap_value": round(float(shap_row[i]), 4),
            "feature_value": round(float(feature_values.iloc[i]), 4),
        }
        for i in order
    ]
    return json.dumps(items)


# Generate and save a SHAP summary plot showing global feature impact
def plot_summary(shap_values: np.ndarray, X: pd.DataFrame, title: str, out_path: Path):
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Wrote %s", out_path)


# Calculate and save global feature importance using mean absolute SHAP values
def save_global_importance(
    shap_values: np.ndarray, feature_names: list, out_path: Path
) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out_path, index=False)
    logger.info("Top 10 features by mean |SHAP|:\n%s", importance.head(10).to_string(index=False))
    return importance


# Generate global and customer-level SHAP explanations for the churn model
def explain_churn():
    logger.info("Explaining churn model (XGBoost)...")
    features = pd.read_parquet(FEATURES_PATH)
    churn_pred = pd.read_parquet(CHURN_PRED_PATH)
    df = features.merge(
        churn_pred[["customer_unique_id", "segment_id", "churn_probability"]],
        on="customer_unique_id", how="inner",
    )

    # Reconstruct the exact feature matrix used when the churn model was trained
    feature_cols = joblib.load(CHURN_ARTIFACTS_DIR / "feature_columns.joblib")
    X = build_features(df, feature_cols)

    # Load the trained churn model and calculate its SHAP values
    model = joblib.load(CHURN_ARTIFACTS_DIR / "xgb_model.joblib")
    shap_values = compute_shap_values(model, X)

    # Save the churn model's global SHAP plot and feature importance table
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_summary(shap_values, X, "Churn model (XGBoost): SHAP feature impact", FIGURES_DIR / "churn_shap_summary.png")
    save_global_importance(shap_values, list(X.columns), OUTPUT_DIR / "churn_feature_importance.csv")

    # Extract the most influential features for every individual customer
    logger.info("Building per-customer local explanations for %d customers...", len(X))
    top_features = [
        top_features_json(shap_values[i], X.iloc[i], list(X.columns), TOP_N_FEATURES)
        for i in range(len(X))
    ]
    result = df[["customer_unique_id", "churn_probability"]].copy()
    result["shap_top_features"] = top_features

    # Save structured local explanations for downstream narration
    out_path = OUTPUT_DIR / "churn_shap.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d customers)", out_path, len(result))


# Generate global and customer-level SHAP explanations for the CLV model
def explain_clv():
    logger.info("Explaining CLV model (LightGBM)...")
    features = pd.read_parquet(FEATURES_PATH)
    clv_pred = pd.read_parquet(CLV_PRED_PATH)
    df = features.merge(
        clv_pred[["customer_unique_id", "segment_id", "churn_probability", "clv_ml"]],
        on="customer_unique_id", how="inner",
    )

    # Reconstruct the exact feature matrix used when the CLV model was trained
    feature_cols = joblib.load(CLV_ARTIFACTS_DIR / "feature_columns.joblib")
    X = build_features(df, feature_cols)

    # Load the trained CLV model and calculate its SHAP values
    model = joblib.load(CLV_ARTIFACTS_DIR / "lgbm_model.joblib")
    shap_values = compute_shap_values(model, X)

    # Save the CLV model's global SHAP plot and feature importance table
    plot_summary(shap_values, X, "CLV model (LightGBM): SHAP feature impact", FIGURES_DIR / "clv_shap_summary.png")
    save_global_importance(shap_values, list(X.columns), OUTPUT_DIR / "clv_feature_importance.csv")

    # Extract the most influential features for every individual customer
    logger.info("Building per-customer local explanations for %d customers...", len(X))
    top_features = [
        top_features_json(shap_values[i], X.iloc[i], list(X.columns), TOP_N_FEATURES)
        for i in range(len(X))
    ]
    result = df[["customer_unique_id", "clv_ml"]].copy()
    result["shap_top_features"] = top_features

    # Save structured local explanations for downstream narration
    out_path = OUTPUT_DIR / "clv_shap.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d customers)", out_path, len(result))


# Run SHAP analysis for both churn and CLV models
def run():
    explain_churn()
    explain_clv()
    logger.info("SHAP explainability complete.")


if __name__ == "__main__":
    run()