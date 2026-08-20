"""dashboard/api_client.py"""

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Self-contained on purpose (no cross-file imports within dashboard/) --
# Streamlit's sys.path handling for multi-page apps varies across versions
# and invocation styles, so this file loads its own config directly rather
# than depending on an import path assumption that might not hold.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")
TIMEOUT_SECONDS = 15

HEADERS = {"X-API-Key": API_KEY}

logger = logging.getLogger("dashboard.api_client")


# Raised when the backend can't be reached or returns an error -- pages
# catch this and show a clear message instead of letting Streamlit crash
class APIClientError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _get(path: str, params: dict | None = None):
    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout as exc:
        logger.error("Request to %s timed out", url)
        raise APIClientError(
            "The API request timed out. Please try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        logger.error("Request to %s failed: %s", url, exc)
        raise APIClientError(
            f"Could not reach the API at {API_BASE_URL}. Is it running?"
        ) from exc

    if response.status_code == 401:
        raise APIClientError("API key rejected -- check API_KEY in .env")

    if response.status_code == 404:
        raise APIClientError("Requested resource was not found")

    if response.status_code == 503:
        raise APIClientError("This data is temporarily unavailable")

    if not response.ok:
        logger.error(
            "API request failed: %s returned HTTP %s",
            url,
            response.status_code,
        )
        raise APIClientError(
            f"Unexpected API response ({response.status_code})"
        )

    try:
        return response.json()
    except ValueError as exc:
        logger.error("API returned invalid JSON from %s", url)
        raise APIClientError(
            "The API returned an invalid response."
        ) from exc

    return response.json()


def get_segments() -> list:
    return _get("/api/v1/segments")


def get_customer_sample(n: int = 10) -> list:
    return _get("/api/v1/customers/sample", params={"n": n})


def get_churn_risk_leaderboard(n: int = 20) -> list:
    return _get("/api/v1/customers/top-churn-risk", params={"n": n})


def get_clv_leaderboard(n: int = 20) -> list:
    return _get("/api/v1/customers/top-clv", params={"n": n})


def get_customer_profile(customer_id: str) -> dict:
    return _get(f"/api/v1/customers/{customer_id}")


def get_churn(customer_id: str) -> dict:
    return _get(f"/api/v1/churn/{customer_id}")


def get_clv(customer_id: str) -> dict:
    return _get(f"/api/v1/clv/{customer_id}")


def get_insights(customer_id: str) -> dict:
    return _get(f"/api/v1/customers/{customer_id}/insights")


def get_narrative(customer_id: str) -> dict:
    return _get(f"/api/v1/customers/{customer_id}/narrative")


def get_forecast_summary() -> dict:
    return _get("/api/v1/forecast/summary")


def get_forecast_series() -> list:
    return _get("/api/v1/forecast/series")