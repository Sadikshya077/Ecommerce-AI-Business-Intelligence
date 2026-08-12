"""
etl/pipeline.py
=================
Orchestrates the full ETL run: extract -> validate -> transform -> load.
This is the only module you actually execute.

Run from the project root:

    python -m etl.pipeline
"""

from etl.config import PROJECT_ROOT
from etl.exceptions import ETLError
from etl.extract import extract
from etl.load import load
from etl.logger import logger
from etl.report import RunReport
from etl.transform import transform
from etl.validate import validate


def run_pipeline() -> RunReport:
    report = RunReport()
    try:
        raw_tables = extract(report)
        validate(raw_tables, report)
        cleaned_tables = transform(raw_tables, report)
        load(cleaned_tables)
    except ETLError as e:
        logger.error(f"ETL pipeline stopped: {e}")
        raise
    except Exception:
        logger.exception("Unexpected error during ETL run")
        raise
    return report


def main() -> None:
    logger.info(f"Starting Olist ETL pipeline (project root: {PROJECT_ROOT})")
    report = run_pipeline()
    report.log_summary()
    logger.info("ETL pipeline finished successfully.")


if __name__ == "__main__":
    main()
