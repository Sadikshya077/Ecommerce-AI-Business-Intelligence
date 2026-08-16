"""llm/config.py"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# Read in __init__, not as class attributes -- same reasoning as
# api/config.py's Settings: class attributes freeze at module import time
# and can't be reconfigured afterward (e.g. in tests via monkeypatch).
class LLMSettings:
    def __init__(self):
        self.api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.model: str = os.getenv("LLM_MODEL", "claude-sonnet-5")
        self.timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
        self.max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()
