"""
features/customer_features.py

Builds customer-level features (RFM, satisfaction, category diversity)
from the PostgreSQL warehouse.

IMPORTANT: Olist's customer_id is generated fresh per order -- it does
NOT identify a returning customer. customer_unique_id is the actual
person. All aggregation here groups by customer_unique_id, never by
customer_id, or every repeat customer will look like a first-time
buyer in every feature.

Output: data/processed/features/customer_features.parquet
One row per customer_unique_id.

Run from project root:
    python -m features.customer_features
"""

import logging
from pathlib import Path

import pandas as pd

from warehouse.db import get_engine

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "features"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("customer_features")

QUERY = """
SELECT
    c.customer_unique_id,
    c.customer_state,
    f.order_id,
    f.price,
    f.freight_value,
    f.payment_value,
    f.payment_installments,
    f.delivery_days,
    f.delivery_delay_days,
    d.full_date AS order_date,
    r.review_score,
    p.category_name_english
FROM fact_order_item f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_date d ON f.order_date_key = d.date_key
LEFT JOIN dim_review r ON f.review_key = r.review_key
LEFT JOIN dim_product p ON f.product_key = p.product_key
"""


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Postgres DATE columns come back via read_sql as plain Python date
    # objects (object dtype), not datetime64 -- convert explicitly or
    # every .dt accessor and Timestamp/Timedelta operation below breaks.
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    reference_date = df["order_date"].max() + pd.Timedelta(days=1)

    # Collapse order-item grain to one row per order first. price/freight are
    # summed across items (multiple items can belong to one order), while
    # payment_value/installments/delivery/review are duplicated across every
    # item of the same order, so take one value rather than summing them.
    order_level = (
        df.groupby(["customer_unique_id", "order_id"])
        .agg(
            order_merchandise_value=("price", "sum"),
            order_freight=("freight_value", "sum"),
            order_payment=("payment_value", "max"),
            order_date=("order_date", "first"),
            order_installments=("payment_installments", "max"),
            order_delivery_days=("delivery_days", "first"),
            order_delivery_delay=("delivery_delay_days", "first"),
            order_review=("review_score", "first"),
        )
        .reset_index()
    )

    features = (
        order_level.groupby("customer_unique_id")
        .agg(
            frequency=("order_id", "nunique"),
            monetary=("order_payment", "sum"),
            avg_order_value=("order_payment", "mean"),
            avg_freight=("order_freight", "mean"),
            first_purchase=("order_date", "min"),
            last_purchase=("order_date", "max"),
            avg_installments=("order_installments", "mean"),
            avg_delivery_days=("order_delivery_days", "mean"),
            avg_delivery_delay=("order_delivery_delay", "mean"),
            avg_review_score=("order_review", "mean"),
        )
        .reset_index()
    )

    features["recency_days"] = (reference_date - features["last_purchase"]).dt.days
    features["customer_tenure_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days

    # Category diversity: how many distinct categories has this customer bought from.
    category_diversity = (
        df.dropna(subset=["category_name_english"])
        .groupby("customer_unique_id")["category_name_english"]
        .nunique()
        .rename("distinct_categories")
        .reset_index()
    )
    features = features.merge(category_diversity, on="customer_unique_id", how="left")
    features["distinct_categories"] = features["distinct_categories"].fillna(0).astype(int)

    # Most common state for this customer (should almost always be a single value).
    state = (
        df.groupby("customer_unique_id")["customer_state"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else "unknown")
        .rename("customer_state")
    )
    features = features.merge(state, on="customer_unique_id", how="left")

    return features


def run():
    engine = get_engine()
    logger.info("Querying warehouse for order-item level customer data...")
    df = pd.read_sql(QUERY, engine)
    logger.info("Pulled %d rows", len(df))

    features = build_customer_features(df)
    logger.info("Built customer_features: %d customers, %d columns", *features.shape)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "customer_features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    run()