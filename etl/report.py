"""
etl/report.py
==============
RunReport: a small mutable container passed through extract -> validate ->
transform so each stage can record row counts and warnings, then pipeline.py
prints a single summary at the end of the run.

Lives in its own module (rather than pipeline.py) so extract.py, validate.py,
and transform.py can import it without importing pipeline.py itself.
"""

from dataclasses import dataclass, field

from etl.logger import logger
# from typing import Dict

@dataclass
class RunReport:
    rows_in: dict = field(default_factory=dict)
    rows_out: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def log_summary(self) -> None:
        logger.info("----- ETL run summary -----")
        for name in self.rows_in:
            before = self.rows_in.get(name, "-")
            after = self.rows_out.get(name, "-")
            logger.info(f"  {name:22s} raw={before!s:>10}  clean={after!s:>10}")
        if self.warnings:
            logger.warning(f"{len(self.warnings)} warning(s) raised during this run:")
            for w in self.warnings:
                logger.warning(f"  - {w}")
        else:
            logger.info("No data-quality warnings.")
