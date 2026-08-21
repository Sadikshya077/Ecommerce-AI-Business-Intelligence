"""
models/churn/evaluate_churn.py

Compares Logistic Regression and XGBoost churn models, selects the
winner by ROC-AUC, and generates churn probabilities for all customers.

Run from project root:
    python -m models.churn.evaluate_churn
"""

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "models"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet"
SEGMENTS_PATH = DATA_DIR / "customer_segments.parquet"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SELECTION_METRIC = "roc_auc"
CLASSIFICATION_THRESHOLD = 0.5
NULLABLE_FEATURES = ["avg_review_score", "avg_delivery_days", "avg_delivery_delay"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate_churn")


def compute_metrics(y_true, proba, threshold=CLASSIFICATION_THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def plot_roc_curves(y_true, logreg_proba, xgb_proba):
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, logreg_proba, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_true, xgb_proba, name="XGBoost", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("Churn model ROC curves")
    ax.legend()
    fig.tight_layout()
    out_path = FIGURES_DIR / "churn_roc_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_confusion_matrices(y_true, logreg_proba, xgb_proba, threshold=CLASSIFICATION_THRESHOLD):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, proba, name in [
        (axes[0], logreg_proba, "Logistic Regression"),
        (axes[1], xgb_proba, "XGBoost"),
    ]:
        pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    fig.suptitle(f"Confusion matrices at threshold={threshold}")
    fig.tight_layout()
    out_path = FIGURES_DIR / "churn_confusion_matrices.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def score_full_population(winner_name: str) -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH)[["customer_unique_id", "segment_id"]]
    df = features.merge(segments, on="customer_unique_id", how="inner")

    feature_cols = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")

    # Must match build_feature_matrix in train_churn.py exactly: segment_id
    # is excluded because it leaks recency_days, which the churn label is
    # thresholded on. See the comment there for the full explanation.
    drop_cols = ["customer_unique_id", "recency_days", "first_purchase", "last_purchase", "segment_id"]
    X = df[[c for c in df.columns if c not in drop_cols]].copy()
    for col in NULLABLE_FEATURES:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(X[col].median())
    X = pd.get_dummies(X, columns=["customer_state"], prefix="state")

    # Align to the exact columns/order the model was trained on -- any
    # category present here but absent at training time (or vice versa)
    # is handled by reindexing with fill_value=0.
    X = X.reindex(columns=feature_cols, fill_value=0)

    if winner_name == "xgboost":
        model = joblib.load(ARTIFACTS_DIR / "xgb_model.joblib")
        proba = model.predict_proba(X)[:, 1]
    else:
        model = joblib.load(ARTIFACTS_DIR / "logreg_model.joblib")
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
        proba = model.predict_proba(scaler.transform(X))[:, 1]

    result = df[["customer_unique_id", "segment_id"]].copy()
    result["churn_probability"] = proba
    return result


def run():
    test_pred = pd.read_parquet(DATA_DIR / "churn_test_predictions.parquet")
    y_true = test_pred["y_true"]

    logreg_metrics = compute_metrics(y_true, test_pred["logreg_proba"])
    xgb_metrics = compute_metrics(y_true, test_pred["xgb_proba"])

    comparison = pd.DataFrame([
        {"model": "logistic_regression", **logreg_metrics},
        {"model": "xgboost", **xgb_metrics},
    ])
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))

    winner_row = comparison.loc[comparison[SELECTION_METRIC].idxmax()]
    winner_name = winner_row["model"]
    logger.info(
        "Selected winner: %s (highest %s = %.4f)", winner_name, SELECTION_METRIC, winner_row[SELECTION_METRIC]
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curves(y_true, test_pred["logreg_proba"], test_pred["xgb_proba"])
    plot_confusion_matrices(y_true, test_pred["logreg_proba"], test_pred["xgb_proba"])

    comparison.to_csv(DATA_DIR / "churn_metrics_comparison.csv", index=False)

    logger.info("Scoring full customer population with winning model (%s)...", winner_name)
    full_predictions = score_full_population(winner_name)
    full_predictions.to_parquet(DATA_DIR / "churn_predictions.parquet", index=False)

    logger.info(
        "Wrote churn_predictions.parquet: %d customers, mean churn_probability=%.3f",
        len(full_predictions), full_predictions["churn_probability"].mean(),
    )
    logger.info("Evaluation complete. Winner: %s", winner_name)


if __name__ == "__main__":
    run()