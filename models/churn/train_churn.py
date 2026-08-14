"""
models/churn/train_churn.py

Trains Logistic Regression and XGBoost churn classifiers and logs
both to MLflow for comparison. Model selection and full-population
scoring happen in evaluate_churn.py -- this script's job is strictly
training and honest, side-by-side logging of both candidates.

Churn label: a customer is labeled churned if recency_days exceeds
the data-driven churn window computed in churn_window.py. recency_days
is used ONLY to build the label and is then dropped from the feature
set -- including it as a feature would make the "prediction" a
tautology, since the label is directly derived from it. The model
instead predicts churn risk from behavioral and satisfaction signals:
purchase frequency, spend, delivery experience, review scores,
category diversity, segment membership, and tenure.

Output:
    data/processed/models/churn_test_predictions.parquet
        customer_unique_id, y_true, logreg_proba, xgb_proba
    models/churn/artifacts/logreg_model.joblib
    models/churn/artifacts/xgb_model.joblib
    models/churn/artifacts/scaler.joblib
    models/churn/artifacts/feature_columns.joblib
    models/churn/artifacts/churn_window_days.txt
    MLflow runs under experiment "churn_prediction"

Run from project root:
    python -m models.churn.train_churn
"""

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from models.churn.churn_window import compute_churn_window

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CUSTOMER_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "customer_segments.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow" / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlflow" / "mlartifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Customer satisfaction/delivery features that can be legitimately missing
# (no review submitted, order not yet delivered within the observation
# window). Flagged with a *_missing indicator and median-imputed, rather
# than silently dropping those customers.
NULLABLE_FEATURES = ["avg_review_score", "avg_delivery_days", "avg_delivery_delay"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_churn")


def load_dataset(churn_window_days: int) -> pd.DataFrame:
    features = pd.read_parquet(CUSTOMER_FEATURES_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH)[["customer_unique_id", "segment_id"]]

    df = features.merge(segments, on="customer_unique_id", how="inner")
    dropped = len(features) - len(df)
    if dropped:
        logger.warning("%d customers had no segment assignment and were dropped", dropped)

    df["churned"] = (df["recency_days"] > churn_window_days).astype(int)
    churn_rate = df["churned"].mean()
    logger.info(
        "Churn label built at window=%d days -- churn rate: %.1f%% (%d / %d customers)",
        churn_window_days, churn_rate * 100, df["churned"].sum(), len(df),
    )
    return df


def build_feature_matrix(df: pd.DataFrame):
    # segment_id is EXCLUDED deliberately, despite being available: Phase 3's
    # K-Means segmentation clustered on recency_days among other features,
    # and segments 0/1 (94% of customers) are separated almost entirely by
    # recency. Since the churn label is itself a recency threshold,
    # including segment_id here would leak the label through a re-encoded
    # version of the same variable rather than genuinely predict churn from
    # independent behavioral signals. frequency and monetary -- the other
    # two RFM inputs to segmentation -- are still included directly below,
    # since those alone don't determine the label.
    drop_cols = [
        "customer_unique_id", "recency_days", "first_purchase", "last_purchase",
        "churned", "segment_id",
    ]
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    for col in NULLABLE_FEATURES:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(X[col].median())

    X = pd.get_dummies(X, columns=["customer_state"], prefix="state")
    y = df["churned"]
    return X, y


def train_logreg(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    proba = model.predict_proba(X_test_scaled)[:, 1]
    return model, scaler, proba


def train_xgb(X_train, y_train, X_test):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


def quick_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def log_to_mlflow(run_name: str, params: dict, metrics: dict, model, flavor: str):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, "model")
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(model, "model")


def run():
    churn_window_days = compute_churn_window()

    df = load_dataset(churn_window_days)
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows | Test: %d rows | Features: %d", len(X_train), len(X_test), X.shape[1])

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name("churn_prediction") is None:
        mlflow.create_experiment("churn_prediction", artifact_location=f"file:{MLFLOW_ARTIFACT_DIR}")
    mlflow.set_experiment("churn_prediction")

    logger.info("Training Logistic Regression...")
    logreg_model, scaler, logreg_proba = train_logreg(X_train, y_train, X_test)
    logreg_metrics = quick_metrics(y_test, logreg_proba)
    logger.info("Logistic Regression: %s", logreg_metrics)
    log_to_mlflow(
        "logistic_regression",
        {"model": "LogisticRegression", "class_weight": "balanced", "churn_window_days": churn_window_days},
        logreg_metrics, logreg_model, "sklearn",
    )

    logger.info("Training XGBoost...")
    xgb_model, xgb_proba = train_xgb(X_train, y_train, X_test)
    xgb_metrics = quick_metrics(y_test, xgb_proba)
    logger.info("XGBoost: %s", xgb_metrics)
    log_to_mlflow(
        "xgboost",
        {"model": "XGBClassifier", "n_estimators": 300, "max_depth": 4,
         "learning_rate": 0.05, "churn_window_days": churn_window_days},
        xgb_metrics, xgb_model, "xgboost",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    test_predictions = pd.DataFrame({
        "customer_unique_id": df.loc[idx_test, "customer_unique_id"].values,
        "y_true": y_test.values,
        "logreg_proba": logreg_proba,
        "xgb_proba": xgb_proba,
    })
    test_predictions.to_parquet(OUTPUT_DIR / "churn_test_predictions.parquet", index=False)

    joblib.dump(logreg_model, ARTIFACTS_DIR / "logreg_model.joblib")
    joblib.dump(xgb_model, ARTIFACTS_DIR / "xgb_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), ARTIFACTS_DIR / "feature_columns.joblib")
    (ARTIFACTS_DIR / "churn_window_days.txt").write_text(str(churn_window_days))

    logger.info("Training complete. Run evaluate_churn.py to compare models and select the final one.")


if __name__ == "__main__":
    run()