"""tests/test_kpi_endpoint.py"""


def test_kpis_endpoint_returns_aggregate_stats(client):
    response = client.get("/api/v1/customers/kpis")
    assert response.status_code == 200
    body = response.json()
    assert body["total_customers"] == 2
    assert 0.0 <= body["avg_churn_probability"] <= 1.0
    # Fixture: cust_high_risk (0.91) exceeds the 0.6 threshold, cust_low_risk (0.12) doesn't
    assert body["high_risk_count"] == 1
    assert body["total_clv_ml"] == body["avg_clv_ml"] * body["total_customers"]
    assert "clv_p75_threshold" in body
    assert "high_value_count" in body


def test_kpis_endpoint_requires_auth(client_with_auth):
    response = client_with_auth.get("/api/v1/customers/kpis")
    assert response.status_code == 401