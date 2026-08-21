"""api/services/insight_service.py"""

from api.data_store import store
from api.schemas.insights import MarketContext
from api.services.customer_service import get_customer_record
from api.services.prediction_service import build_churn_result, build_clv_result

# Reported as a finding (see reports/methodology_phase3.md), not an empty
# field -- the LLM narrator has something honest to say about this signal
ASSOCIATION_RULES_NOTE = (
    "No statistically meaningful product association rules were found for this "
    "dataset (only 0.7% of orders spanned multiple categories); this signal is "
    "not available for narration."
)


# The integrated framework artifact. Combines customer-level behavioral
# predictions (segmentation -> churn -> CLV, a direct dependency chain)
# with product/market-level context (association rules, sales forecast),
# which inform interpretation only and do NOT feed into CLV itself -- see
# reports/methodology_phase4.md for why that boundary is deliberate.
# Routers must not assemble this object themselves; this is the one place
# it's built.
def build_customer_insights(customer_unique_id: str) -> dict:
    record = get_customer_record(customer_unique_id)
    churn = build_churn_result(record)
    clv = build_clv_result(record)

    market_context = MarketContext(**store.forecast_summary) if store.forecast_summary else None

    return {
        "customer_unique_id": customer_unique_id,
        "segment_id": churn["segment_id"],
        "segment_label": record["segment_label"],
        "churn_probability": churn["churn_probability"],
        "clv_ml": clv["clv_ml"],
        "clv_formula": clv["clv_formula"],
        "historical_spend": clv["historical_spend"],
        "churn_top_features": churn["top_features"],
        "clv_top_features": clv["top_features"],
        "association_rules_note": ASSOCIATION_RULES_NOTE,
        "market_context": market_context,
    }
