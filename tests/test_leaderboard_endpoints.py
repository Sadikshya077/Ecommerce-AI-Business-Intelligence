"""tests/test_leaderboard_endpoints.py"""


def test_top_churn_risk_returns_highest_first(client):
    response = client.get("/api/v1/customers/top-churn-risk?n=2")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    # Fixture data: cust_high_risk (0.91) should rank above cust_low_risk (0.12)
    assert body[0]["customer_unique_id"] == "cust_high_risk"
    assert body[0]["churn_probability"] > body[1]["churn_probability"]


def test_top_clv_returns_highest_first(client):
    response = client.get("/api/v1/customers/top-clv?n=2")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    # Fixture data: cust_low_risk (clv_ml=300.0) should rank above cust_high_risk (40.0)
    assert body[0]["customer_unique_id"] == "cust_low_risk"
    assert body[0]["clv_ml"] > body[1]["clv_ml"]


def test_top_churn_risk_respects_n_parameter(client):
    response = client.get("/api/v1/customers/top-churn-risk?n=1")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_leaderboard_endpoints_require_auth(client_with_auth):
    response = client_with_auth.get("/api/v1/customers/top-clv")
    assert response.status_code == 401
