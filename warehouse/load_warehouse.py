# Loads the cleaned Parquet output of the etl package into the
# PostgreSQL star schema defined in warehouse/schema.sql.

import logging
from pathlib import Path

import pandas as pd

from db import get_engine

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("load_warehouse")


# ---------------------------------------------------------------------------
# Load cleaned parquet tables
# ---------------------------------------------------------------------------

def load_processed_tables() -> dict:
    names = ["customers", "orders", "order_items", "payments", "reviews", "products", "sellers"]
    tables = {}
    for name in names:
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed file: {path}\n"
                "Run `python -m etl.pipeline` first."
            )
        tables[name] = pd.read_parquet(path)
        logger.info("Loaded %-14s %6d rows", name, len(tables[name]))
    return tables


# ---------------------------------------------------------------------------
# Order-level aggregation -- payments and reviews can have >1 row per order
# ---------------------------------------------------------------------------

def aggregate_payments_to_order(payments: pd.DataFrame) -> pd.DataFrame:
    """Collapse possibly-multiple payment rows per order into one summary row."""
    agg = (
        payments.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=(
                "payment_type",
                lambda x: x.mode().iat[0] if not x.mode().empty else "unknown",
            ),
        )
        .reset_index()
    )
    return agg


def dedupe_reviews_to_order(reviews: pd.DataFrame) -> pd.DataFrame:
    """Guarantee exactly one review row per order_id for the dim_review join."""
    reviews = reviews.copy()
    if "review_creation_date" in reviews.columns:
        reviews = reviews.sort_values("review_creation_date")
    reviews = reviews.drop_duplicates(subset="order_id", keep="last")
    return reviews[["order_id", "review_score"]]


# ---------------------------------------------------------------------------
# Dimension loaders -- each reads its own cleaned table directly, so every
# customer/product/seller gets a dimension row regardless of whether it
# appears in a currently-valid order (keeps dimensions complete).
# ---------------------------------------------------------------------------

def load_dim_customer(engine, customers: pd.DataFrame):
    dim = customers.rename(columns={"customer_zip_code_prefix": "customer_zip_prefix"})[
        ["customer_id", "customer_unique_id", "customer_city", "customer_state", "customer_zip_prefix"]
    ]
    dim.to_sql("dim_customer", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    logger.info("dim_customer: loaded %d rows", len(dim))


def load_dim_product(engine, products: pd.DataFrame):
    dim = products.rename(columns={
        "product_category_name": "category_name",
        "product_category_name_english": "category_name_english",
        "product_weight_g": "weight_g",
        "product_length_cm": "length_cm",
        "product_height_cm": "height_cm",
        "product_width_cm": "width_cm",
    })[["product_id", "category_name", "category_name_english", "weight_g", "length_cm", "height_cm", "width_cm"]]
    dim.to_sql("dim_product", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    logger.info("dim_product: loaded %d rows", len(dim))


def load_dim_seller(engine, sellers: pd.DataFrame):
    dim = sellers.rename(columns={"seller_zip_code_prefix": "seller_zip_prefix"})[
        ["seller_id", "seller_city", "seller_state", "seller_zip_prefix"]
    ]
    dim.to_sql("dim_seller", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    logger.info("dim_seller: loaded %d rows", len(dim))


def load_dim_payment(engine, payments_agg: pd.DataFrame):
    dim = payments_agg[["payment_type"]].dropna().drop_duplicates()
    dim.to_sql("dim_payment", engine, if_exists="append", index=False)
    logger.info("dim_payment: loaded %d rows", len(dim))


def load_dim_review(engine, reviews_order: pd.DataFrame):
    dim = reviews_order.drop_duplicates(subset="order_id")
    dim.to_sql("dim_review", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    logger.info("dim_review: loaded %d rows", len(dim))


def build_dim_date(min_date, max_date) -> pd.DataFrame:
    dates = pd.date_range(min_date, max_date, freq="D")
    dim = pd.DataFrame({"full_date": dates})
    dim["date_key"] = dim["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim["day"] = dim["full_date"].dt.day
    dim["month"] = dim["full_date"].dt.month
    dim["quarter"] = dim["full_date"].dt.quarter
    dim["year"] = dim["full_date"].dt.year
    dim["day_of_week"] = dim["full_date"].dt.dayofweek
    dim["is_weekend"] = dim["day_of_week"].isin([5, 6])
    return dim[["date_key", "full_date", "day", "month", "quarter", "year", "day_of_week", "is_weekend"]]


def load_dim_date(engine, orders: pd.DataFrame):
    min_date = orders["order_purchase_timestamp"].min().normalize()
    max_date = orders["order_purchase_timestamp"].max().normalize()
    dim = build_dim_date(min_date, max_date)
    dim.to_sql("dim_date", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    logger.info("dim_date: loaded %d rows", len(dim))


# ---------------------------------------------------------------------------
# Fact table: build the order-item grain source, then map to surrogate keys
# ---------------------------------------------------------------------------

def build_fact_source(tables: dict, payments_agg: pd.DataFrame, reviews_order: pd.DataFrame) -> pd.DataFrame:
    order_items = tables["order_items"]
    orders = tables["orders"]

    fact = order_items.merge(orders, on="order_id", how="left", suffixes=("", "_order"))
    fact = fact.merge(payments_agg, on="order_id", how="left")
    fact = fact.merge(reviews_order, on="order_id", how="left")

    # transform.py's _clean_orders parses the raw timestamp columns to
    # datetime but doesn't derive delivery_days/delivery_delay_days --
    # computed here instead, from the raw Olist timestamp column names.
    fact["delivery_days"] = (
        fact["order_delivered_customer_date"] - fact["order_purchase_timestamp"]
    ).dt.days
    fact["delivery_delay_days"] = (
        fact["order_delivered_customer_date"] - fact["order_estimated_delivery_date"]
    ).dt.days

    fact["payment_value"] = fact["payment_value"].fillna(0.0)
    fact["payment_installments"] = fact["payment_installments"].fillna(1)
    fact["payment_type"] = fact["payment_type"].fillna("unknown")

    fact["order_date_key"] = pd.to_numeric(
        fact["order_purchase_timestamp"].dt.strftime("%Y%m%d"), errors="coerce"
    ).astype("Int64")

    return fact


def load_fact(engine, fact_source: pd.DataFrame):
    customer_map = pd.read_sql("SELECT customer_key, customer_id FROM dim_customer", engine)
    product_map = pd.read_sql("SELECT product_key, product_id FROM dim_product", engine)
    seller_map = pd.read_sql("SELECT seller_key, seller_id FROM dim_seller", engine)
    payment_map = pd.read_sql("SELECT payment_key, payment_type FROM dim_payment", engine)
    review_map = pd.read_sql("SELECT review_key, order_id FROM dim_review", engine)

    fact = fact_source.merge(customer_map, on="customer_id", how="left")
    fact = fact.merge(product_map, on="product_id", how="left")
    fact = fact.merge(seller_map, on="seller_id", how="left")
    fact = fact.merge(payment_map, on="payment_type", how="left")
    fact = fact.merge(review_map, on="order_id", how="left")

    # Orphan check -- these are genuinely missing parent records, distinct
    # from "no review yet" which is expected and not logged as an issue.
    for col, label in [
        ("customer_key", "customer"), ("product_key", "product"), ("seller_key", "seller"),
    ]:
        n_missing = int(fact[col].isna().sum())
        if n_missing:
            logger.warning("%d order_item rows have no matching %s (orphaned FK)", n_missing, label)

    fact_final = fact[[
        "order_id", "order_item_id", "customer_key", "product_key", "seller_key",
        "payment_key", "order_date_key", "review_key",
        "price", "freight_value", "payment_value", "payment_installments",
        "delivery_days", "delivery_delay_days",
    ]]

    fact_final.to_sql("fact_order_item", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    logger.info("fact_order_item: loaded %d rows", len(fact_final))


def run():
    engine = get_engine()
    tables = load_processed_tables()

    payments_agg = aggregate_payments_to_order(tables["payments"])
    reviews_order = dedupe_reviews_to_order(tables["reviews"])

    logger.info("Loading dimension tables...")
    load_dim_customer(engine, tables["customers"])
    load_dim_product(engine, tables["products"])
    load_dim_seller(engine, tables["sellers"])
    load_dim_payment(engine, payments_agg)
    load_dim_review(engine, reviews_order)
    load_dim_date(engine, tables["orders"])

    logger.info("Building fact source table...")
    fact_source = build_fact_source(tables, payments_agg, reviews_order)

    logger.info("Loading fact table...")
    load_fact(engine, fact_source)

    logger.info("Warehouse load complete.")


if __name__ == "__main__":
    run()


"""
warehouse/load_warehouse.py

Loads the cleaned Parquet output of the etl package into the
PostgreSQL star schema defined in warehouse/schema.sql.

This matches the actual ETL output: 8 separate cleaned tables
(customers, orders, order_items, payments, reviews, products,
sellers, geolocation) rather than one pre-joined wide table.
The order-item grain fact table is built here, in this script,
by joining them.

IMPORTANT -- two things transform.py does NOT handle, which this
script has to account for:

1. payments and reviews can have more than one row per order_id
   (payments: no natural PK, multiple installments/vouchers per
   order; reviews: rare duplicate review submissions). Joining
   either directly onto order_items at native grain would fan out
   the fact table -- each is aggregated down to one row per
   order_id first.

2. validate.py only *warns* about orphaned foreign keys, it does
   not drop the offending rows. The left joins below will leave a
   NULL surrogate key wherever a parent record is genuinely missing
   (e.g. an order_item pointing at a product_id absent from
   products.parquet). NULL foreign keys are allowed by the schema
   so the load will not fail, but this script logs how many rows
   were affected per dimension -- check that output before treating
   the warehouse as ground truth for churn/CLV feature engineering.

ASSUMPTION -- this script uses the original Olist column names
(product_category_name, product_weight_g, customer_zip_code_prefix,
seller_zip_code_prefix, etc.), since your transform.py's cleaning
functions don't appear to rename columns, only cast/fill them. If
any of your _clean_* functions do rename columns, adjust the
`.rename()` calls below to match.

Prerequisites:
    1. PostgreSQL running, reachable via .env
    2. Schema applied: psql -U postgres -d ecommerce_bi -f warehouse\\schema.sql
    3. ETL already run: python -m etl.pipeline
"""
