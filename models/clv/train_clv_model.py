# Train XGBoost and LightGBM models to predict customer historical spend
# Compare both models and save their predictions, models, and feature schema

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet"
CHURN_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "churn_predictions.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow" / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlflow" / "mlartifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.2
NULLABLE_FEATURES = ["avg_review_score", "avg_delivery_days", "avg_delivery_delay"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_clv_model")


# Load customer features and merge them with churn predictions
def load_dataset() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    churn = pd.read_parquet(CHURN_PATH)[["customer_unique_id", "churn_probability"]]

    # Keep only customers that have both feature data and churn predictions
    df = features.merge(churn, on="customer_unique_id", how="inner")
    dropped = len(features) - len(df)
    if dropped:
        logger.warning("%d customers had no churn prediction and were dropped", dropped)
    return df


# Prepare model inputs while removing target and leakage-prone columns
def build_feature_matrix(df: pd.DataFrame):
    drop_cols = [
        "customer_unique_id", "recency_days", "first_purchase", "last_purchase",
        "segment_id", "monetary",
    ]
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    # Preserve missing-value information and fill missing numerical features
    for col in NULLABLE_FEATURES:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(X[col].median())

    # Convert customer state into model-ready one-hot encoded features
    X = pd.get_dummies(X, columns=["customer_state"], prefix="state")
    y = df["monetary"]
    return X, y


# Calculate root mean squared error between actual and predicted values
def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


# Calculate the main regression evaluation metrics
def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# Train, evaluate, track, and save both CLV regression models
def run():
    # Load the complete training dataset and prepare model features
    df = load_dataset()
    X, y = build_feature_matrix(df)

    # Split customers into training and test sets for model evaluation
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train: %d rows | Test: %d rows | Features: %d", len(X_train), len(X_test), X.shape[1])

    # Configure the local MLflow tracking database and experiment
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name("clv_prediction") is None:
        mlflow.create_experiment("clv_prediction", artifact_location=f"file:{MLFLOW_ARTIFACT_DIR}")
    mlflow.set_experiment("clv_prediction")

    # Train and evaluate the XGBoost regression model
    logger.info("Training XGBoost regressor...")
    xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = evaluate(y_test, xgb_pred)
    logger.info("XGBoost: %s", xgb_metrics)

    # Track the XGBoost model, parameters, and metrics in MLflow
    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({"model": "XGBRegressor", "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05})
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_model, "model")

    # Train and evaluate the LightGBM regression model
    logger.info("Training LightGBM regressor...")
    lgbm_model = LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)
    lgbm_metrics = evaluate(y_test, lgbm_pred)
    logger.info("LightGBM: %s", lgbm_metrics)

    # Track the LightGBM model, parameters, and metrics in MLflow
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params({"model": "LGBMRegressor", "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05})
        mlflow.log_metrics(lgbm_metrics)
        mlflow.lightgbm.log_model(lgbm_model, "model")

    # Create directories for model outputs and saved artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save test predictions from both models for later CLV evaluation
    test_predictions = pd.DataFrame({
        "customer_unique_id": df.loc[idx_test, "customer_unique_id"].values,
        "y_true": y_test.values,
        "xgb_pred": xgb_pred,
        "lgbm_pred": lgbm_pred,
    })
    test_predictions.to_parquet(OUTPUT_DIR / "clv_test_predictions.parquet", index=False)

    # Save trained models and the exact feature schema used during training
    joblib.dump(xgb_model, ARTIFACTS_DIR / "xgb_model.joblib")
    joblib.dump(lgbm_model, ARTIFACTS_DIR / "lgbm_model.joblib")
    joblib.dump(list(X.columns), ARTIFACTS_DIR / "feature_columns.joblib")

    logger.info("Training complete. Run evaluate_clv.py to compare against the formula-based estimate.")


if __name__ == "__main__":
    run()
