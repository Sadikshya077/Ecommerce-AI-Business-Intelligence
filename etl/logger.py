"""
Single shared logger for the pipeline. Every other etl/ module does:

    from etl.logger import logger

so log formatting and level are configured in exactly one place.
"""

import logging
import sys


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("olist_etl")
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # avoid duplicate handlers if imported twice
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


logger = _build_logger()
