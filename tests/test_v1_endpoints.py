"""tests/test_v1_endpoints.py"""


def test_churn_endpoint_returns_expected_shape(client):
    response = client.get("/api/v1/churn/cust_low_risk")
    assert response.status_code == 200
    body = response.json()
    assert body["customer_unique_id"] == "cust_low_risk"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["segment_id"] == 2
    assert len(body["top_features"]) == 2


def test_churn_endpoint_unknown_customer_returns_404(client):
    response = client.get("/api/v1/churn/does-not-exist")
    assert response.status_code == 404


def test_clv_endpoint_returns_both_estimates(client):
    response = client.get("/api/v1/clv/cust_high_risk")
    assert response.status_code == 200
    body = response.json()
    assert body["clv_ml"] == 40.0
    assert body["clv_formula"] == 55.0


def test_customer_profile_endpoint(client):
    response = client.get("/api/v1/customers/cust_low_risk")
    assert response.status_code == 200
    assert response.json()["segment_label"] == "Loyal repeat customers"


def test_customer_profile_unknown_customer_returns_404(client):
    response = client.get("/api/v1/customers/does-not-exist")
    assert response.status_code == 404


def test_customer_sample_returns_requested_count(client):
    response = client.get("/api/v1/customers/sample?n=1")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_segments_endpoint_lists_all_segments(client):
    response = client.get("/api/v1/segments")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {"segment_id", "segment_label", "n_customers"} <= body[0].keys()


def test_insights_bundles_everything_in_one_call(client):
    response = client.get("/api/v1/customers/cust_low_risk/insights")
    assert response.status_code == 200
    body = response.json()
    assert body["segment_label"] == "Loyal repeat customers"
    assert "churn_top_features" in body
    assert "clv_top_features" in body
    assert body["market_context"]["trend_direction"] == "up"
    assert "association_rules_note" in body
    # Behavioral profile fields, reused from customer_features.parquet
    assert body["frequency"] == 2.0
    assert body["recency_days"] == 150.0
    assert body["avg_order_value"] == 140.0
    assert body["avg_freight"] == 18.5
    assert body["avg_delivery_days"] == 9.0
    assert body["avg_review_score"] == 4.2


def test_insights_returns_null_not_nan_for_missing_optional_fields(client):
    # cust_high_risk has no delivery-time or review data on record -- these
    # must serialize as JSON null, not a literal NaN float
    response = client.get("/api/v1/customers/cust_high_risk/insights")
    assert response.status_code == 200
    body = response.json()
    assert body["avg_delivery_days"] is None
    assert body["avg_review_score"] is None
    # Required behavioral fields are still present and correct
    assert body["frequency"] == 1.0
    assert body["recency_days"] == 410.0


def test_insights_unknown_customer_returns_404(client):
    response = client.get("/api/v1/customers/does-not-exist/insights")
    assert response.status_code == 404


def test_forecast_summary_endpoint(client):
    response = client.get("/api/v1/forecast/summary")
    assert response.status_code == 200
    assert response.json()["trend_direction"] == "up"


def test_unversioned_churn_path_is_404(client):
    response = client.get("/churn/cust_low_risk")
    assert response.status_code == 404


def test_old_v1_prefix_without_api_is_404(client):
    # Confirms the earlier /v1 (no /api) path from Part 1 no longer resolves
    response = client.get("/v1/churn/cust_low_risk")
    assert response.status_code == 404


def test_missing_api_key_is_rejected(client_with_auth):
    response = client_with_auth.get("/api/v1/churn/cust_low_risk")
    assert response.status_code == 401


def test_wrong_api_key_is_rejected(client_with_auth):
    response = client_with_auth.get("/api/v1/churn/cust_low_risk", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_correct_api_key_is_accepted(client_with_auth, test_api_key):
    response = client_with_auth.get("/api/v1/churn/cust_low_risk", headers={"X-API-Key": test_api_key})
    assert response.status_code == 200


def test_health_still_unversioned_and_unauthenticated(client_with_auth):
    response = client_with_auth.get("/health")
    assert response.status_code == 200