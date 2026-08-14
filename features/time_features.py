"""
features/time_features.py

Builds a daily sales time series with lag and rolling-window features
for Phase 4's sales forecasting (Prophet baseline, optional XGBoost
comparison).

Output: data/processed/features/daily_sales.parquet
One row per calendar day in the dataset's date range, including days
with zero orders (filled with 0) so lag/rolling windows stay correct.

Run from project root:
    python -m features.time_features
"""

import logging
from pathlib import Path

import pandas as pd

from warehouse.db import get_engine

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "features"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("time_features")

QUERY = """
SELECT
    d.full_date AS order_date,
    f.order_id,
    f.price
FROM fact_order_item f
JOIN dim_date d ON f.order_date_key = d.date_key
"""


def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    # Postgres DATE columns come back via read_sql as plain Python date
    # objects (object dtype), not datetime64 -- convert explicitly or
    # every .dt accessor below breaks.
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    daily = (
        df.groupby("order_date")
        .agg(
            revenue=("price", "sum"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("order_date")
    )

    # Fill any missing calendar days with zero activity so lag/rolling
    # windows are computed over a continuous date range, not just the
    # days that happened to have an order.
    full_range = pd.date_range(daily["order_date"].min(), daily["order_date"].max(), freq="D")
    daily = (
        daily.set_index("order_date")
        .reindex(full_range)
        .fillna(0.0)
        .rename_axis("order_date")
        .reset_index()
    )

    daily["day_of_week"] = daily["order_date"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6])
    daily["month"] = daily["order_date"].dt.month
    daily["quarter"] = daily["order_date"].dt.quarter

    for lag in [1, 7, 30]:
        daily[f"revenue_lag_{lag}"] = daily["revenue"].shift(lag)
        daily[f"order_count_lag_{lag}"] = daily["order_count"].shift(lag)

    for window in [7, 30]:
        daily[f"revenue_rolling_mean_{window}"] = daily["revenue"].rolling(window).mean()
        daily[f"order_count_rolling_mean_{window}"] = daily["order_count"].rolling(window).mean()

    return daily


def run():
    engine = get_engine()
    logger.info("Querying warehouse for daily order data...")
    df = pd.read_sql(QUERY, engine)
    logger.info("Pulled %d rows", len(df))

    daily = build_daily_sales(df)
    logger.info("Built daily_sales: %d days, %d columns", *daily.shape)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "daily_sales.parquet"
    daily.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    run()