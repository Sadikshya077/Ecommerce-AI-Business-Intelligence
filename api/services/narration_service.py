"""api/services/narration_service.py"""

from api.exceptions import NarrationUnavailableError
from api.schemas.insights import CustomerInsights
from api.services.insight_service import build_customer_insights
from llm.narrator import generate_narrative
from llm.schemas import NarrativeResult


# Combines the Phase 6 insights context with the Phase 7 narration engine.
# Raises NarrationUnavailableError (not a bare None) when narration fails,
# so the router stays thin and consistent with every other service in this
# codebase. CustomerNotFoundError from build_customer_insights propagates
# up unchanged -- a missing customer is not this service's concern to
# reinterpret.
def get_customer_narrative(customer_unique_id: str) -> NarrativeResult:
    raw_context = build_customer_insights(customer_unique_id)

    # Validate against the same schema /insights returns -- this is the
    # "Validated Insight Context" step in the Phase 7 architecture -- then
    # dump to a fully plain, JSON-safe dict. mode="json" recursively
    # converts nested SHAPFeature/MarketContext model instances into plain
    # dicts; without this, json.dumps() in the prompt builder fails on
    # those nested objects (they aren't JSON-serializable on their own).
    validated_context = CustomerInsights(**raw_context).model_dump(mode="json")

    result = generate_narrative(validated_context)
    if result is None:
        raise NarrationUnavailableError(f"Narration failed for customer {customer_unique_id}")
    return result