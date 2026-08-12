"""
etl/transform.py
==================
Stage 3 of the pipeline: clean each raw table.
  - dedupe on primary key
  - standardize date dtypes
  - handle missing values with explicit, documented rules
  - translate product category names to English
  - aggregate geolocation to one row per zip_code_prefix
"""

import pandas as pd

from etl.config import DATE_COLUMNS
from etl.logger import logger
from etl.report import RunReport
# from typing import Dict

def transform(tables: dict[str, pd.DataFrame], report: RunReport) -> dict[str, pd.DataFrame]:
    logger.info("STAGE: transform")

    cleaned = {
        "customers": _clean_customers(tables["customers"]),
        "orders": _clean_orders(tables["orders"]),
        "order_items": _clean_order_items(tables["order_items"]),
        "payments": _clean_payments(tables["payments"]),
        "reviews": _clean_reviews(tables["reviews"]),
        "products": _clean_products(tables["products"], tables["category_translation"]),
        "sellers": _clean_sellers(tables["sellers"]),
        "geolocation": _clean_geolocation(tables["geolocation"]),
    }

    for name, df in cleaned.items():
        report.rows_out[name] = len(df)

    logger.info("  transform complete")
    return cleaned


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _standardize_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _dedupe(df: pd.DataFrame, key) -> pd.DataFrame:
    keys = [key] if isinstance(key, str) else key
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="first")
    dropped = before - len(df)
    if dropped:
        logger.info(f"    dropped {dropped} exact-key duplicate row(s)")
    return df


# ---------------------------------------------------------------------------
# Per-table cleaning functions
# ---------------------------------------------------------------------------

def _clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    return _dedupe(df.copy(), "customer_id")


def _clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    orders = _dedupe(df.copy(), "order_id")
    orders = _standardize_dates(orders, DATE_COLUMNS["orders"])
    # order_status feeds the churn label and forecasting downstream -- normalize it
    orders["order_status"] = orders["order_status"].str.lower().str.strip()
    return orders


def _clean_order_items(df: pd.DataFrame) -> pd.DataFrame:
    order_items = _dedupe(df.copy(), ["order_id", "order_item_id"])
    order_items["price"] = order_items["price"].astype(float)
    order_items["freight_value"] = order_items["freight_value"].astype(float)
    return order_items


def _clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    payments = df.copy()
    # payments has no single-column PK -- an order can have multiple payment
    # rows -- so dedupe on the full row instead of a key subset
    before = len(payments)
    payments = payments.drop_duplicates(keep="first")
    if before != len(payments):
        logger.info(f"    payments: dropped {before - len(payments)} exact duplicate row(s)")
    payments["payment_value"] = payments["payment_value"].astype(float)
    return payments


def _clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    reviews = _dedupe(df.copy(), "review_id")
    reviews = _standardize_dates(reviews, DATE_COLUMNS["reviews"])
    # Free-text review fields are legitimately optional -- keep the "no
    # comment" case explicit as empty string rather than fabricating text
    reviews["review_comment_title"] = reviews["review_comment_title"].fillna("")
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("")
    return reviews


def _clean_products(products_raw: pd.DataFrame, category_translation: pd.DataFrame) -> pd.DataFrame:
    products = _dedupe(products_raw.copy(), "product_id")
    products = products.merge(category_translation, on="product_category_name", how="left")

    # Category is required for association-rule mining / feature engineering;
    # rows with no category (and no translation) are labeled explicitly rather than dropped
    products["product_category_name_english"] = products[
        "product_category_name_english"
    ].fillna("unknown")

    numeric_product_cols = [
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
    ]
    for col in numeric_product_cols:
        if col in products.columns:
            median_val = products[col].median()
            n_missing = products[col].isnull().sum()
            if n_missing:
                logger.info(
                    f"    products.{col}: filling {n_missing} missing value(s) "
                    f"with column median ({median_val:.2f})"
                )
            products[col] = products[col].fillna(median_val)

    return products


def _clean_sellers(df: pd.DataFrame) -> pd.DataFrame:
    return _dedupe(df.copy(), "seller_id")


def _clean_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    # Geolocation has many rows per zip prefix (multiple lat/lng samples).
    # Aggregate to one representative row per prefix so downstream joins
    # don't fan out order/customer row counts.
    geo = df.copy()
    geolocation = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(
            geolocation_lat=("geolocation_lat", "median"),
            geolocation_lng=("geolocation_lng", "median"),
            geolocation_city=(
                "geolocation_city",
                lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0],
            ),
            geolocation_state=(
                "geolocation_state",
                lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0],
            ),
        )
        .reset_index()
    )
    return geolocation
