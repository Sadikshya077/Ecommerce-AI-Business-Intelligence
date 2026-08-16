"""tests/test_health.py"""

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


# Patch CustomerStore.load() with a small synthetic dataset for isolated API tests.
def _patch_store_load(monkeypatch, n_customers=2, forecast=True):
    def fake_load(self):
        ids = [f"cust_{i}" for i in range(n_customers)]
        # Build the DataFrame from columns so the empty dataset still has a valid schema.
        self._df = pd.DataFrame({"customer_unique_id": ids}).set_index("customer_unique_id")
        self.forecast_summary = {"trend_direction": "up"} if forecast else None
        return self

    monkeypatch.setattr("api.data_store.CustomerStore.load", fake_load)


# Verify that the health endpoint returns a successful response.
def test_health_returns_ok(monkeypatch):
    _patch_store_load(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Verify that the health endpoint is accessible without authentication.
def test_health_requires_no_auth(monkeypatch):
    _patch_store_load(monkeypatch)
    with TestClient(app) as client:
        # No X-API-Key header sent at all
        response = client.get("/health")
    assert response.status_code == 200


# Verify that liveness stays healthy even when the data store is empty.
def test_health_succeeds_even_when_store_not_ready(monkeypatch):
    # Liveness must report ok even before the store has real data loaded --
    # that distinction (process alive vs. able to serve real traffic) is
    # exactly what /ready exists to capture separately
    _patch_store_load(monkeypatch, n_customers=0, forecast=False)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"



# Verify that readiness reports ready when customer data is loaded.
def test_readiness_reports_ready_when_data_loaded(monkeypatch):
    _patch_store_load(monkeypatch, n_customers=3)
    with TestClient(app) as client:
        response = client.get("/ready")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["customers_loaded"] == 3
    assert body["forecast_available"] is True



# Verify that readiness reports not_ready when the store has no data.
def test_readiness_reports_not_ready_when_store_empty(monkeypatch):
    _patch_store_load(monkeypatch, n_customers=0, forecast=False)
    with TestClient(app) as client:
        response = client.get("/ready")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "not_ready"
    assert body["customers_loaded"] == 0
    assert body["forecast_available"] is False