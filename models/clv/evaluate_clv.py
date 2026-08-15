# Evaluate ML-based and formula-based CLV estimates
# Compare prediction quality and generate final full-population CLV estimates
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "models"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet"
CHURN_PATH = DATA_DIR / "churn_predictions.parquet"
FORMULA_PATH = DATA_DIR / "clv_formula.parquet"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TOP_DECILE = 0.10
NULLABLE_FEATURES = ["avg_review_score", "avg_delivery_days", "avg_delivery_delay"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate_clv")


# Calculate root mean squared error between actual and predicted values
def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


# Measure how many of the highest-value customers appear in both rankings
def top_decile_overlap(actual: pd.Series, predicted: pd.Series) -> float:
    n_top = max(int(len(actual) * TOP_DECILE), 1)
    actual_top = set(actual.nlargest(n_top).index)
    predicted_top = set(predicted.nlargest(n_top).index)
    return len(actual_top & predicted_top) / n_top


# Evaluate the trained ML CLV models against historical customer spend
def evaluate_ml_models():
    test_pred = pd.read_parquet(DATA_DIR / "clv_test_predictions.parquet")
    y_true = test_pred["y_true"].reset_index(drop=True)

    # Calculate evaluation metrics for each ML model
    rows = []
    for col, name in [("xgb_pred", "xgboost"), ("lgbm_pred", "lightgbm")]:
        pred = test_pred[col].reset_index(drop=True)
        spearman_corr, _ = spearmanr(y_true, pred)
        rows.append({
            "model": name,
            "rmse": rmse(y_true, pred),
            "mae": mean_absolute_error(y_true, pred),
            "r2": r2_score(y_true, pred),
            "spearman": spearman_corr,
            "top_decile_overlap": top_decile_overlap(y_true, pred),
        })
    comparison = pd.DataFrame(rows)
    return comparison, test_pred


# Evaluate the formula-based CLV against realized historical customer spend
def evaluate_formula(test_pred: pd.DataFrame) -> dict:

    # Load formula-based CLV values and match them to the test customers
    formula_df = pd.read_parquet(FORMULA_PATH)[["customer_unique_id", "clv_formula"]]
    merged = test_pred.merge(formula_df, on="customer_unique_id", how="inner")

    # Calculate prediction and ranking metrics for the formula-based estimate
    y_true = merged["y_true"].reset_index(drop=True)
    pred = merged["clv_formula"].reset_index(drop=True)
    spearman_corr, _ = spearmanr(y_true, pred)

    return {
        "model": "formula_based",
        "rmse": rmse(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
        "spearman": spearman_corr,
        "top_decile_overlap": top_decile_overlap(y_true, pred),
    }


# Plot actual spend against the selected ML model and formula-based CLV
def plot_predicted_vs_actual(test_pred: pd.DataFrame, formula_df: pd.DataFrame, winner_col: str, winner_name: str):
    merged = test_pred.merge(formula_df, on="customer_unique_id", how="inner")

    # Create comparison plots using the 99th percentile to reduce outlier distortion
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    lims = [0, merged["y_true"].quantile(0.99)]
    axes[0].scatter(merged["y_true"], merged[winner_col], alpha=0.3, s=10)
    axes[0].plot(lims, lims, color="gray", linestyle="--")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual historical spend (R$)")
    axes[0].set_ylabel(f"Predicted ({winner_name})")
    axes[0].set_title(f"{winner_name}: predicted vs. actual")

    axes[1].scatter(merged["y_true"], merged["clv_formula"], alpha=0.3, s=10, color="tab:orange")
    axes[1].set_xlim(lims)
    axes[1].set_ylim([0, merged["clv_formula"].quantile(0.99)])
    axes[1].set_xlabel("Actual historical spend (R$)")
    axes[1].set_ylabel("Formula-based CLV")
    axes[1].set_title("Formula-based CLV vs. actual\n(expected to run higher -- projects future value)")

    fig.tight_layout()
    out_path = FIGURES_DIR / "clv_predicted_vs_actual.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# Score every customer using the winning ML model and formula-based CLV
def score_full_population(winner_name: str) -> pd.DataFrame:

    # Load customer features and churn predictions for the full population
    features = pd.read_parquet(FEATURES_PATH)
    churn = pd.read_parquet(CHURN_PATH)[["customer_unique_id", "segment_id", "churn_probability"]]
    df = features.merge(churn, on="customer_unique_id", how="inner")

    # Load the exact feature schema used when training the winning model
    feature_cols = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")
    drop_cols = [
        "customer_unique_id", "recency_days", "first_purchase", "last_purchase",
        "segment_id", "monetary",
    ]
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    # Handle missing numerical values while preserving missingness information
    for col in NULLABLE_FEATURES:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(X[col].median())

    # Encode customer state and align features with the trained model schema
    X = pd.get_dummies(X, columns=["customer_state"], prefix="state")
    X = X.reindex(columns=feature_cols, fill_value=0)

    # Load the selected ML model and generate CLV predictions
    model_file = "xgb_model.joblib" if winner_name == "xgboost" else "lgbm_model.joblib"
    model = joblib.load(ARTIFACTS_DIR / model_file)
    df["clv_ml"] = model.predict(X)

    return df[["customer_unique_id", "segment_id", "churn_probability", "monetary", "clv_ml"]]


# Evaluate the models, select the best ML model, and generate final CLV outputs
def run():

    # Evaluate both ML models and the formula-based CLV
    comparison, test_pred = evaluate_ml_models()
    formula_row = evaluate_formula(test_pred)
    comparison = pd.concat([comparison, pd.DataFrame([formula_row])], ignore_index=True)

    logger.info("Model comparison:\n%s", comparison.to_string(index=False))
    logger.info(
        "NOTE: formula_based RMSE/MAE/R2 are approximate -- the formula projects "
        "forward-looking lifetime value, not historical spend, so a direct numeric "
        "comparison to realized monetary is a simplification. Its top_decile_overlap "
        "is the more defensible metric: does it still rank the most valuable "
        "customers correctly, even if the absolute number runs higher."
    )

    # Select the ML model with the lowest RMSE against historical spend
    ml_only = comparison[comparison["model"].isin(["xgboost", "lightgbm"])]
    ml_winner_row = ml_only.loc[ml_only["rmse"].idxmin()]
    ml_winner = ml_winner_row["model"]
    logger.info("ML winner (lowest RMSE against historical spend): %s", ml_winner)

    # Generate the predicted-versus-actual comparison plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    winner_col = "xgb_pred" if ml_winner == "xgboost" else "lgbm_pred"
    formula_df = pd.read_parquet(FORMULA_PATH)[["customer_unique_id", "clv_formula"]]
    plot_predicted_vs_actual(test_pred, formula_df, winner_col, ml_winner)

    # Save the complete model comparison metrics
    comparison.to_csv(DATA_DIR / "clv_metrics_comparison.csv", index=False)

    # Generate CLV predictions for the full customer population
    logger.info("Scoring full customer population with %s and the formula...", ml_winner)
    full_ml = score_full_population(ml_winner)
    full_formula = pd.read_parquet(FORMULA_PATH)[["customer_unique_id", "clv_formula"]]

    # Combine ML and formula-based CLV estimates side by side
    final = full_ml.merge(full_formula, on="customer_unique_id", how="left")
    final.to_parquet(DATA_DIR / "clv_predictions.parquet", index=False)

    logger.info(
        "Wrote clv_predictions.parquet: %d customers | mean clv_ml=%.2f | mean clv_formula=%.2f",
        len(final), final["clv_ml"].mean(), final["clv_formula"].mean(),
    )
    logger.info("CLV evaluation complete.")


if __name__ == "__main__":
    run()
