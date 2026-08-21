"""tests/conftest.py"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies import verify_api_key
from api.main import app

TEST_API_KEY = "test-key-123"


def _shap_json():
    return json.dumps([
        {"feature": "avg_freight", "shap_value": 0.5, "feature_value": 20.0},
        {"feature": "avg_delivery_delay", "shap_value": 0.3, "feature_value": 2.0},
    ])


# A small, fixed synthetic customer dataset -- tests never depend on real
# pipeline output, so they run fast and don't require Postgres, the full
# ETL pipeline, or trained models to exist on disk
def _fake_customer_frame() -> pd.DataFrame:
    shap = _shap_json()
    rows = [
        {
            "customer_unique_id": "cust_low_risk",
            "segment_id": 2,
            "segment_label": "Loyal repeat customers",
            "churn_probability": 0.12,
            "clv_ml": 300.0,
            "clv_formula": 450.0,
            "monetary": 280.0,
            "churn_shap_top_features": shap,
            "clv_shap_top_features": shap,
        },
        {
            "customer_unique_id": "cust_high_risk",
            "segment_id": 0,
            "segment_label": "Lapsed one-time buyers",
            "churn_probability": 0.91,
            "clv_ml": 40.0,
            "clv_formula": 55.0,
            "monetary": 38.0,
            "churn_shap_top_features": shap,
            "clv_shap_top_features": shap,
        },
    ]
    return pd.DataFrame(rows).set_index("customer_unique_id")


_FAKE_FORECAST_SUMMARY = {
    "last_actual_date": "2018-08-01",
    "forecast_horizon_days": 30,
    "prior_30d_actual_revenue": 100000.0,
    "next_30d_predicted_revenue": 105000.0,
    "pct_change": 5.0,
    "trend_direction": "up",
}

_FAKE_SEGMENT_PROFILE = [
    {"segment_id": 2, "segment_label": "Loyal repeat customers", "n_customers": 2881,
     "avg_recency_days": 227.1, "avg_frequency": 2.11, "avg_monetary": 289.6, "pct_of_customers": 3.0},
    {"segment_id": 0, "segment_label": "Lapsed one-time buyers", "n_customers": 38256,
     "avg_recency_days": 394.7, "avg_frequency": 1.0, "avg_monetary": 134.3, "pct_of_customers": 40.1},
]


def _patch_store_load(monkeypatch):
    def fake_load(self):
        self._df = _fake_customer_frame()
        self.forecast_summary = _FAKE_FORECAST_SUMMARY
        self.segment_profile = _FAKE_SEGMENT_PROFILE
        return self

    monkeypatch.setattr("api.data_store.CustomerStore.load", fake_load)


# Customer data present, but the two soft-loaded global signals (forecast,
# segment profile) are unavailable -- exercises the ModelArtifactMissingError
# (503) path without needing to actually delete real pipeline output
@pytest.fixture
def client_missing_optional_data(monkeypatch):
    def fake_load(self):
        self._df = _fake_customer_frame()
        self.forecast_summary = None
        self.segment_profile = []
        return self

    monkeypatch.setattr("api.data_store.CustomerStore.load", fake_load)
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# One customer's SHAP data is deliberately invalid JSON -- exercises the
# PredictionServiceError (500) path
@pytest.fixture
def client_malformed_shap(monkeypatch):
    def fake_load(self):
        rows = [{
            "customer_unique_id": "cust_low_risk",
            "segment_id": 2,
            "segment_label": "Loyal repeat customers",
            "churn_probability": 0.12,
            "clv_ml": 300.0,
            "clv_formula": 450.0,
            "monetary": 280.0,
            "churn_shap_top_features": "{not valid json",
            "clv_shap_top_features": _shap_json(),
        }]
        self._df = pd.DataFrame(rows).set_index("customer_unique_id")
        self.forecast_summary = _FAKE_FORECAST_SUMMARY
        self.segment_profile = _FAKE_SEGMENT_PROFILE
        return self

    monkeypatch.setattr("api.data_store.CustomerStore.load", fake_load)
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# Auth bypassed -- for testing business logic / response shape, not security itself
@pytest.fixture
def client(monkeypatch):
    _patch_store_load(monkeypatch)
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# Real auth enforced -- for testing the security dependency itself
@pytest.fixture
def client_with_auth(monkeypatch):
    _patch_store_load(monkeypatch)
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    from api.config import get_settings
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
def test_api_key():
    return TEST_API_KEY