"""
etl/config.py
==============
Central configuration for the Olist ETL pipeline: paths, expected raw
filenames, date columns to parse, and primary-key definitions per table.

Keeping this in one place means every other etl/ module imports from here
instead of hardcoding paths or table names -- if the raw file layout or
project directory structure changes, this is the only file that changes.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# etl/config.py -> etl/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Raw Olist files
# ---------------------------------------------------------------------------

RAW_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# ---------------------------------------------------------------------------
# Columns that should be parsed as datetimes per table
# ---------------------------------------------------------------------------

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}

# ---------------------------------------------------------------------------
# Primary key(s) expected to be unique and non-null per table
# (payments has no natural PK -- a single order can have multiple payment
# rows -- so it's intentionally excluded here and deduped on the full row
# instead, in transform.py)
# ---------------------------------------------------------------------------

PRIMARY_KEYS = {
    "customers": "customer_id",
    "orders": "order_id",
    "order_items": ["order_id", "order_item_id"],
    "products": "product_id",
    "sellers": "seller_id",
    "reviews": "review_id",
}
