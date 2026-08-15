"""
Sanity-check the raw tables before transforming.
Does NOT mutate data -- only inspects it and records warnings on the
RunReport. Only hard-fails (ETLError) on problems that would make
transformation meaningless, e.g. a required key column missing entirely.
"""

import pandas as pd

from etl.config import PRIMARY_KEYS
from etl.exceptions import ETLError
from etl.logger import logger
from etl.report import RunReport


def validate(tables: dict[str, pd.DataFrame], report: RunReport) -> None:
    logger.info("STAGE: validate")

    _check_primary_keys(tables, report)
    _check_referential_integrity(tables, report)

    logger.info("  validation complete")


def _check_primary_keys(tables: dict[str, pd.DataFrame], report: RunReport) -> None:
    for name, key in PRIMARY_KEYS.items():
        df = tables[name]
        keys = [key] if isinstance(key, str) else key

        for k in keys:
            if k not in df.columns:
                raise ETLError(f"Table '{name}' is missing expected key column '{k}'")

        null_count = df[keys].isnull().any(axis=1).sum()
        if null_count:
            msg = f"{name}: {null_count} row(s) with null primary key ({keys})"
            report.warnings.append(msg)
            logger.warning(f"  {msg}")

        dup_count = df.duplicated(subset=keys).sum()
        if dup_count:
            msg = f"{name}: {dup_count} duplicate row(s) on key ({keys})"
            report.warnings.append(msg)
            logger.warning(f"  {msg}")


def _check_referential_integrity(tables: dict[str, pd.DataFrame], report: RunReport) -> None:
    # order_items / payments / reviews -> orders
    orders_ids = set(tables["orders"]["order_id"])
    for child in ("order_items", "payments", "reviews"):
        child_ids = set(tables[child]["order_id"])
        orphaned = child_ids - orders_ids
        if orphaned:
            msg = f"{child}: {len(orphaned)} order_id(s) not found in orders (orphaned)"
            report.warnings.append(msg)
            logger.warning(f"  {msg}")

    # orders -> customers
    customer_ids = set(tables["customers"]["customer_id"])
    orphaned_customers = set(tables["orders"]["customer_id"]) - customer_ids
    if orphaned_customers:
        msg = f"orders: {len(orphaned_customers)} customer_id(s) not found in customers"
        report.warnings.append(msg)
        logger.warning(f"  {msg}")

    # order_items -> products / sellers
    product_ids = set(tables["products"]["product_id"])
    orphaned_products = set(tables["order_items"]["product_id"]) - product_ids
    if orphaned_products:
        msg = f"order_items: {len(orphaned_products)} product_id(s) not found in products"
        report.warnings.append(msg)
        logger.warning(f"  {msg}")

    seller_ids = set(tables["sellers"]["seller_id"])
    orphaned_sellers = set(tables["order_items"]["seller_id"]) - seller_ids
    if orphaned_sellers:
        msg = f"order_items: {len(orphaned_sellers)} seller_id(s) not found in sellers"
        report.warnings.append(msg)
        logger.warning(f"  {msg}")
