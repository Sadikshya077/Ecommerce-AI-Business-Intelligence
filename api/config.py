"""api/config.py"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# Centralized settings, loaded once and cached -- avoids re-reading .env
# or scattering os.getenv() calls across the codebase
class Settings:
    api_key: str = os.getenv("API_KEY", "")
    cors_origins: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    data_dir: Path = PROJECT_ROOT / "data" / "processed" / "models"
    project_root: Path = PROJECT_ROOT


@lru_cache
def get_settings() -> Settings:
    return Settings()
