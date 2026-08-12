"""
etl/exceptions.py
===================
Pipeline-specific exception(s). Kept in their own module so any etl/ file
can raise/catch ETLError without importing pipeline.py (which would create
a circular import, since pipeline.py imports from every other module).
"""


class ETLError(Exception):
    """Raised when the pipeline cannot safely continue (e.g. missing raw
    files, or a table missing a required key column)."""
