"""tests/test_narrative_endpoint.py"""

from llm.schemas import NarrativeContent, NarrativeResult


def _fake_narrative_result():
    return NarrativeResult(
        content=NarrativeContent(
            summary="Loyal customer with low churn risk.",
            risk_explanation="Avg freight was an important factor contributing to the model's prediction.",
            key_drivers=["avg freight"],
            recommended_actions=["Offer a loyalty discount"],
            limitations=["SHAP reflects correlation, not causation"],
        ),
        model="gemini-3.6-flash",
        prompt_version="v1",
        generated_at="2026-08-16T00:00:00+00:00",
        latency_ms=250.0,
        input_tokens=120,
        output_tokens=60,
        faithfulness_passed=True,
        faithfulness_notes=[],
    )


def test_narrative_endpoint_returns_generated_narrative(client, monkeypatch):
    monkeypatch.setattr(
        "api.services.narration_service.generate_narrative",
        lambda context: _fake_narrative_result(),
    )
    response = client.get("/api/v1/customers/cust_low_risk/narrative")
    assert response.status_code == 200
    body = response.json()
    assert body["content"]["summary"]
    assert body["faithfulness_passed"] is True
    assert body["model"] == "gemini-3.6-flash"


def test_narrative_endpoint_returns_503_when_llm_fails(client, monkeypatch):
    monkeypatch.setattr(
        "api.services.narration_service.generate_narrative",
        lambda context: None,
    )
    response = client.get("/api/v1/customers/cust_low_risk/narrative")
    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    # Safe message -- no internal detail (customer ID, exception text) leaked
    assert "cust_low_risk" not in body["detail"]


def test_narrative_endpoint_unknown_customer_returns_404(client, monkeypatch):
    monkeypatch.setattr(
        "api.services.narration_service.generate_narrative",
        lambda context: _fake_narrative_result(),
    )
    response = client.get("/api/v1/customers/does-not-exist/narrative")
    assert response.status_code == 404


def test_insights_endpoint_unaffected_by_narration_failure(client, monkeypatch):
    # The core "never a single point of failure" check: /insights must keep
    # working even when narration is completely broken, since it never
    # calls into narration_service at all
    monkeypatch.setattr(
        "api.services.narration_service.generate_narrative",
        lambda context: None,
    )
    response = client.get("/api/v1/customers/cust_low_risk/insights")
    assert response.status_code == 200


def test_narrative_endpoint_requires_auth(client_with_auth):
    response = client_with_auth.get("/api/v1/customers/cust_low_risk/narrative")
    assert response.status_code == 401


def test_narrative_endpoint_serializes_real_shap_objects_correctly(client, monkeypatch):
    # Regression test: mocks only at the true external boundary (call_llm),
    # so this exercises the REAL chain -- build_customer_insights() ->
    # CustomerInsights validation -> model_dump(mode="json") ->
    # build_user_prompt() -> json.dumps(). The earlier tests in this file
    # and in test_llm_narrator.py all mocked closer to the LLM and used
    # hand-written plain-dict contexts, which never exercised this path and
    # is exactly how a real bug here (SHAPFeature objects not being
    # JSON-serializable) went undetected until a live run against real data.
    import json as json_module

    raw_json_text = json_module.dumps({
        "summary": "This is a loyal customer with low churn risk.",
        "risk_explanation": "Avg freight was an important factor contributing to the model's prediction.",
        "key_drivers": ["avg freight"],
        "recommended_actions": ["Offer a loyalty discount"],
        "limitations": ["SHAP reflects correlation, not causation"],
    })

    def fake_call_llm(system_prompt, user_prompt):
        return raw_json_text, 200.0, {"input_tokens": 50, "output_tokens": 30}

    monkeypatch.setattr("llm.narrator.call_llm", fake_call_llm)
    response = client.get("/api/v1/customers/cust_low_risk/narrative")
    assert response.status_code == 200
    assert response.json()["content"]["summary"]