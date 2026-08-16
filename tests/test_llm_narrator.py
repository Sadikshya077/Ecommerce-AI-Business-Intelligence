"""tests/test_llm_narrator.py"""

import json

from llm.faithfulness import check_faithfulness
from llm.narrator import generate_narrative
from llm.schemas import NarrativeContent

SAMPLE_CONTEXT = {
    "customer_unique_id": "cust_low_risk",
    "segment_label": "Loyal repeat customers",
    "churn_probability": 0.12,
    "clv_ml": 300.0,
    "clv_formula": 450.0,
    "churn_top_features": [{"feature": "avg_freight", "shap_value": 0.5, "feature_value": 20.0}],
    "clv_top_features": [{"feature": "avg_order_value", "shap_value": 50.0, "feature_value": 150.0}],
    "association_rules_note": "No meaningful rules found.",
    "market_context": None,
}


def _valid_narrative_json():
    return json.dumps({
        "summary": "This is a loyal customer with low churn risk.",
        "risk_explanation": "Avg freight was an important factor contributing to the model's churn prediction.",
        "key_drivers": ["avg freight", "avg order value"],
        "recommended_actions": ["Offer a loyalty discount"],
        "limitations": ["SHAP values reflect correlation, not causation"],
    })


def test_generate_narrative_returns_none_when_llm_unavailable(monkeypatch):
    from llm.client import LLMClientError

    def fake_call_llm(system_prompt, user_prompt):
        raise LLMClientError("simulated outage")

    monkeypatch.setattr("llm.narrator.call_llm", fake_call_llm)
    assert generate_narrative(SAMPLE_CONTEXT) is None


def test_generate_narrative_returns_none_on_invalid_json(monkeypatch):
    def fake_call_llm(system_prompt, user_prompt):
        return "not valid json at all", 100.0, {"input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr("llm.narrator.call_llm", fake_call_llm)
    assert generate_narrative(SAMPLE_CONTEXT) is None


def test_generate_narrative_returns_none_on_schema_violation(monkeypatch):
    def fake_call_llm(system_prompt, user_prompt):
        # Valid JSON, but missing required fields
        return json.dumps({"summary": "only this field"}), 100.0, {"input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr("llm.narrator.call_llm", fake_call_llm)
    assert generate_narrative(SAMPLE_CONTEXT) is None


def test_generate_narrative_succeeds_with_valid_response(monkeypatch):
    def fake_call_llm(system_prompt, user_prompt):
        return _valid_narrative_json(), 250.0, {"input_tokens": 120, "output_tokens": 60}

    monkeypatch.setattr("llm.narrator.call_llm", fake_call_llm)
    result = generate_narrative(SAMPLE_CONTEXT)

    assert result is not None
    assert result.content.summary
    assert result.latency_ms == 250.0
    assert result.input_tokens == 120
    assert result.prompt_version == "v1"
    assert result.faithfulness_passed is True


def test_faithfulness_flags_causal_language():
    content = NarrativeContent(
        summary="ok",
        risk_explanation="Low frequency caused the customer to churn.",
        key_drivers=["avg freight"],
        recommended_actions=["contact them"],
        limitations=["none"],
    )
    passed, notes = check_faithfulness(content, SAMPLE_CONTEXT)
    assert passed is False
    assert any("causal" in n.lower() for n in notes)


def test_faithfulness_flags_unsupported_feature_mentions():
    content = NarrativeContent(
        summary="ok",
        risk_explanation="Something unrelated drove this.",
        key_drivers=["completely made up feature name"],
        recommended_actions=["contact them"],
        limitations=["none"],
    )
    passed, notes = check_faithfulness(content, SAMPLE_CONTEXT)
    assert passed is False


def test_faithfulness_passes_clean_narrative():
    content = NarrativeContent(
        summary="Loyal customer.",
        risk_explanation="Avg freight was an important factor contributing to the model's prediction.",
        key_drivers=["avg freight", "avg order value"],
        recommended_actions=["Offer a discount"],
        limitations=["SHAP reflects correlation, not causation"],
    )
    passed, notes = check_faithfulness(content, SAMPLE_CONTEXT)
    assert passed is True
    assert notes == []
