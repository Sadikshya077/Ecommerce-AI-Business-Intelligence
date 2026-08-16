"""
warehouse/db.py

Shared PostgreSQL connection helper, reused by load_warehouse.py
and later by the FastAPI service.

Reads connection settings from a .env file in the project root
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def get_engine() -> Engine:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")

    if not all([user, password, host, port, db]):
        raise RuntimeError("PostgreSQL configuration is incomplete")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    return create_engine(url)
