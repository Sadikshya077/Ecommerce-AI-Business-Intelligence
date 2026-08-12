"""
etl/extract.py
================
Stage 1 of the pipeline: read the raw Olist CSVs from data/raw/ into a dict
of DataFrames. Fails fast (ETLError) if the raw directory or any expected
file is missing, so later stages never run against partial data.
"""

import pandas as pd

from etl.config import RAW_DIR, RAW_FILES
from etl.exceptions import ETLError
from etl.logger import logger
from etl.report import RunReport


def extract(report: RunReport) -> dict[str, pd.DataFrame]:
    """Read all raw CSVs into a dict of DataFrames keyed by table name."""
    logger.info("STAGE: extract")
    if not RAW_DIR.exists():
        raise ETLError(
            f"Raw data directory not found: {RAW_DIR}\n"
            f"Download the Olist dataset and place the CSVs there first."
        )

    tables: dict[str, pd.DataFrame] = {}
    missing = []
    for name, filename in RAW_FILES.items():
        path = RAW_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        df = pd.read_csv(path)
        tables[name] = df
        report.rows_in[name] = len(df)
        logger.info(f"  loaded {filename:45s} rows={len(df)}")

    if missing:
        raise ETLError(
            "Missing required raw file(s): " + ", ".join(missing) +
            f"\nExpected them in {RAW_DIR}"
        )

    return tables
