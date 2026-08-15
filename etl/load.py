"""Writes cleaned tables to data/processed/*.parquet.
Idempotent - reruns overwrite the previous output rather than appending.

Note: this does NOT load into PostgreSQL. That's warehouse/load_warehouse.py's
job, which reads these same parquet files as its input.
"""

import pandas as pd

from etl.config import PROCESSED_DIR, PROJECT_ROOT
from etl.logger import logger


def load(tables: dict[str, pd.DataFrame]) -> None:
    logger.info("STAGE: load")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        out_path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  wrote {out_path.relative_to(PROJECT_ROOT)}  rows={len(df)}")

    logger.info("  load complete")
