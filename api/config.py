"""api/config.py"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# Centralized settings, loaded once and cached -- avoids re-reading .env
# or scattering os.getenv() calls across the codebase. Read in __init__,
# not as class attributes: class attributes are evaluated once at module
# import time and would freeze whatever the env happened to be at that
# moment, making it impossible to reconfigure (e.g. via monkeypatch in
# tests) after the module first loads.
class Settings:
    def __init__(self):
        self.api_key: str = os.getenv("API_KEY", "")
        self.cors_origins: list = os.getenv(
            "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
        ).split(",")
        self.data_dir: Path = PROJECT_ROOT / "data" / "processed" / "models"
        self.project_root: Path = PROJECT_ROOT


@lru_cache
def get_settings() -> Settings:
    return Settings()