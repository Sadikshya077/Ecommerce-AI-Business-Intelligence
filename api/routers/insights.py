"""api/routers/insights.py"""

import json

from fastapi import APIRouter, HTTPException

from api.data_store import load_forecast_summary, store
from api.schemas import CustomerInsights, MarketContext, SHAPFeature

router = APIRouter(prefix="/customer", tags=["insights"])

# Load the forecast summary once when the API starts.
_forecast_summary = load_forecast_summary()

ASSOCIATION_RULES_NOTE = (
    "No statistically meaningful product association rules were found for this "
    "dataset (only 0.7% of orders spanned multiple categories); this signal is "
    "not available for narration."
)


# Return a small sample of real customer records for manual API testing.
@router.get("/sample")
def sample_customers(n: int = 5):
    df = store.sample(n)
    return df[["customer_unique_id", "segment_label", "churn_probability", "clv_ml"]].to_dict(orient="records")


# Return all customer-level insights needed by the LLM narration layer.
@router.get("/{customer_unique_id}/insights", response_model=CustomerInsights)
def get_customer_insights(customer_unique_id: str):
    record = store.get(customer_unique_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_unique_id}' not found")

    # Convert stored SHAP JSON into validated feature objects.
    churn_top_features = [SHAPFeature(**f) for f in json.loads(record["churn_shap_top_features"])]
    clv_top_features = [SHAPFeature(**f) for f in json.loads(record["clv_shap_top_features"])]

    # Include market context when a forecast summary is available.
    market_context = MarketContext(**_forecast_summary) if _forecast_summary else None

    return CustomerInsights(
        customer_unique_id=customer_unique_id,
        segment_id=int(record["segment_id"]),
        segment_label=record["segment_label"],
        churn_probability=record["churn_probability"],
        clv_ml=record["clv_ml"],
        clv_formula=record["clv_formula"],
        historical_spend=record["monetary"],
        churn_top_features=churn_top_features,
        clv_top_features=clv_top_features,
        association_rules_note=ASSOCIATION_RULES_NOTE,
        market_context=market_context,
    )