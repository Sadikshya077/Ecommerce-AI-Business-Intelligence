# Calculate formula-based Customer Lifetime Value using order behavior and churn probability
# Combine customer features with churn predictions to estimate relative remaining customer value

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet"
CHURN_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "churn_predictions.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "models"

CHURN_PROB_CLIP_MIN = 0.02
CHURN_PROB_CLIP_MAX = 0.98

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("clv_formula")


# Calculate formula-based CLV from customer value, frequency, and churn probability
def compute_formula_clv() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)[["customer_unique_id", "avg_order_value", "frequency"]]
    churn = pd.read_parquet(CHURN_PATH)[["customer_unique_id", "churn_probability"]]

    # Combine customer features with their predicted churn probabilities
    df = features.merge(churn, on="customer_unique_id", how="inner")
    dropped = len(features) - len(df)
    if dropped:
        logger.warning("%d customers had no churn prediction and were dropped", dropped)

    # Limit extreme churn probabilities to prevent unrealistic CLV values
    clipped = df["churn_probability"].clip(CHURN_PROB_CLIP_MIN, CHURN_PROB_CLIP_MAX)
    n_clipped = int((df["churn_probability"] != clipped).sum())
    if n_clipped:
        logger.info(
            "Clipped churn_probability to [%.2f, %.2f] for %d customers to avoid unbounded CLV",
            CHURN_PROB_CLIP_MIN, CHURN_PROB_CLIP_MAX, n_clipped,
        )

    # Apply the formula-based CLV calculation
    df["clv_formula"] = df["avg_order_value"] * df["frequency"] * (1 / clipped)
    return df


# Run the CLV calculation and save the resulting customer-level estimates
def run():
    df = compute_formula_clv()
    logger.info(
        "Formula-based CLV computed for %d customers -- mean=%.2f  median=%.2f  max=%.2f",
        len(df), df["clv_formula"].mean(), df["clv_formula"].median(), df["clv_formula"].max(),
    )

    # Create the output directory and save the CLV results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "clv_formula.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    run()
