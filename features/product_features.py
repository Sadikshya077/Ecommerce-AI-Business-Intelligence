"""
features/product_features.py

Builds order-level basket data for association rule mining
(FP-Growth, Phase 3) and basic category summary statistics.

Output:
    data/processed/features/order_baskets.parquet
        Long format, one row per (order_id, category_name_english).
        Phase 3's association_rules script pivots this into a one-hot
        transaction matrix via mlxtend.preprocessing.TransactionEncoder.

    data/processed/features/category_summary.parquet
        One row per category: order count, item count, avg price,
        total revenue. Useful for sanity-checking association rules
        and for dashboard category breakdowns later.

Run from project root:
    python -m features.product_features
"""

import logging
from pathlib import Path

import pandas as pd

from warehouse.db import get_engine

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "features"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("product_features")

QUERY = """
SELECT
    f.order_id,
    f.product_key,
    f.price,
    p.category_name_english
FROM fact_order_item f
JOIN dim_product p ON f.product_key = p.product_key
WHERE p.category_name_english IS NOT NULL AND p.category_name_english != 'unknown'
"""


def build_order_baskets(df: pd.DataFrame) -> pd.DataFrame:
    baskets = df[["order_id", "category_name_english"]].drop_duplicates()
    return baskets


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("category_name_english")
        .agg(
            n_orders=("order_id", "nunique"),
            n_items=("product_key", "count"),
            avg_price=("price", "mean"),
            total_revenue=("price", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    return summary


def run():
    engine = get_engine()
    logger.info("Querying warehouse for order-item/category data...")
    df = pd.read_sql(QUERY, engine)
    logger.info("Pulled %d rows", len(df))

    baskets = build_order_baskets(df)
    logger.info("Built order_baskets: %d rows across %d orders", len(baskets), baskets["order_id"].nunique())

    summary = build_category_summary(df)
    logger.info("Built category_summary: %d categories", len(summary))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baskets.to_parquet(OUTPUT_DIR / "order_baskets.parquet", index=False)
    summary.to_parquet(OUTPUT_DIR / "category_summary.parquet", index=False)
    logger.info("Wrote order_baskets.parquet and category_summary.parquet to %s", OUTPUT_DIR)


if __name__ == "__main__":
    run()
