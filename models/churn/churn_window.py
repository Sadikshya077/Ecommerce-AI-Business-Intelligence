"""
models/churn/churn_window.py

Derives a data-driven churn window (N days) rather than choosing one
arbitrarily: computes the distribution of days between consecutive
orders for repeat customers, and sets the churn window at the 90th
percentile of that distribution -- a customer who has gone longer
than 90% of observed repeat-purchase gaps without buying again is
treated as having likely churned.

Cross-checked against the Phase 3 segmentation result: the "lapsed
one-time buyers" segment's average recency should land in a similar
range if the two independent analyses (clustering vs. gap
distribution) are telling a consistent story.

LIMITATION: this dataset is right-censored -- customers whose most
recent purchase falls near the end of the observed window haven't
had a fair chance to make a second purchase yet, and may be
mislabeled "churned" simply because the data collection stopped
early. This is a known limitation of applying a static churn window
to a fixed historical dataset; worth stating explicitly rather than
treating the label as ground truth.

Set CHURN_WINDOW_OVERRIDE to an int to bypass the calculation after
reviewing the diagnostics -- the computed distribution is still
logged either way, so the choice stays documented.

Run standalone for diagnostics only:
    python -m models.churn.churn_window
"""

import logging
from pathlib import Path

import pandas as pd

from warehouse.db import get_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEGMENT_PROFILE_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "segment_profile.parquet"

CHURN_WINDOW_OVERRIDE = None
PERCENTILE = 90

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("churn_window")

QUERY = """
SELECT
    c.customer_unique_id,
    d.full_date AS order_date
FROM fact_order_item f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_date d ON f.order_date_key = d.date_key
"""


def compute_gap_distribution() -> pd.Series:
    engine = get_engine()
    df = pd.read_sql(QUERY, engine)
    df["order_date"] = pd.to_datetime(df["order_date"])

    # One row per distinct order date per customer -- the fact table is
    # order-item grain, so multiple items on the same order would
    # otherwise duplicate the same date.
    order_dates = df.drop_duplicates(subset=["customer_unique_id", "order_date"])
    order_dates = order_dates.sort_values(["customer_unique_id", "order_date"])

    order_dates["gap_days"] = order_dates.groupby("customer_unique_id")["order_date"].diff().dt.days
    gaps = order_dates["gap_days"].dropna()
    return gaps


def compute_churn_window() -> int:
    if CHURN_WINDOW_OVERRIDE is not None:
        logger.info("CHURN_WINDOW_OVERRIDE=%d set -- skipping data-driven calculation.", CHURN_WINDOW_OVERRIDE)
        return CHURN_WINDOW_OVERRIDE

    gaps = compute_gap_distribution()
    n_repeat_gaps = len(gaps)
    logger.info("Computed %d inter-purchase gaps from repeat customers", n_repeat_gaps)

    if n_repeat_gaps == 0:
        raise ValueError(
            "No repeat-purchase gaps found -- cannot compute a data-driven churn "
            "window. Set CHURN_WINDOW_OVERRIDE manually."
        )

    logger.info(
        "Inter-purchase gap distribution (days): median=%.1f  p75=%.1f  p90=%.1f  p95=%.1f  mean=%.1f",
        gaps.median(), gaps.quantile(0.75), gaps.quantile(0.90), gaps.quantile(0.95), gaps.mean(),
    )

    churn_window = int(round(gaps.quantile(PERCENTILE / 100)))
    logger.info("Churn window set to %d days (p%d of repeat-purchase gaps)", churn_window, PERCENTILE)

    if SEGMENT_PROFILE_PATH.exists():
        profile = pd.read_parquet(SEGMENT_PROFILE_PATH)
        lapsed_candidates = profile[profile["avg_frequency"] < 1.5]
        if not lapsed_candidates.empty:
            lapsed_recency = lapsed_candidates["avg_recency_days"].max()
            consistent = abs(lapsed_recency - churn_window) < churn_window * 0.5
            logger.info(
                "Cross-check: Phase 3's lapsed one-time-buyer segment had an "
                "average recency of %.1f days -- %s the computed churn window.",
                lapsed_recency, "consistent with" if consistent else "notably different from",
            )

    return churn_window


if __name__ == "__main__":
    compute_churn_window()
