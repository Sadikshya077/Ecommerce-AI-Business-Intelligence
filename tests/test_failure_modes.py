"""tests/test_failure_modes.py"""


def test_forecast_summary_returns_503_when_unavailable(client_missing_optional_data):
    response = client_missing_optional_data.get("/api/v1/forecast/summary")
    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    # Safe message only -- no filenames or internal detail in the response body
    assert "sales_forecast_summary.json" not in body["detail"]


def test_segments_returns_503_when_unavailable(client_missing_optional_data):
    response = client_missing_optional_data.get("/api/v1/segments")
    assert response.status_code == 503


def test_insights_degrades_gracefully_when_forecast_unavailable(client_missing_optional_data):
    # market_context should be null, not break the rest of the response --
    # a missing global signal shouldn't take down customer-level data
    response = client_missing_optional_data.get("/api/v1/customers/cust_low_risk/insights")
    assert response.status_code == 200
    assert response.json()["market_context"] is None


def test_malformed_shap_data_returns_safe_500(client_malformed_shap):
    response = client_malformed_shap.get("/api/v1/churn/cust_low_risk")
    assert response.status_code == 500
    body = response.json()
    assert "detail" in body
    # Must not leak the raw parse error, "json", or a traceback
    assert "json" not in body["detail"].lower()
    assert "traceback" not in body["detail"].lower()
    assert "not valid json" not in body["detail"]


def test_clv_endpoint_unaffected_by_other_customers_malformed_data(client_malformed_shap):
    # Only churn_shap_top_features is broken for this customer -- confirms
    # the failure is scoped to what actually failed, not the whole response
    response = client_malformed_shap.get("/api/v1/clv/cust_low_risk")
    assert response.status_code == 200


def test_insights_response_has_complete_schema(client):
    response = client.get("/api/v1/customers/cust_low_risk/insights")
    body = response.json()
    expected_keys = {
        "customer_unique_id", "segment_id", "segment_label", "churn_probability",
        "clv_ml", "clv_formula", "historical_spend", "churn_top_features",
        "clv_top_features", "association_rules_note", "market_context",
    }
    assert expected_keys <= body.keys()
    assert isinstance(body["churn_top_features"], list)
    assert isinstance(body["churn_top_features"][0]["shap_value"], float)
