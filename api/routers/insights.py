"""api/routers/insights.py"""

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.schemas.insights import CustomerInsights
from api.services.insight_service import build_customer_insights

# Nested under /customers (plural) to match the canonical path
# /api/v1/customers/{customer_id}/insights -- lives in its own router file
# per the canonical structure, but shares the customers/ URL namespace with
# customers.py; no path collision since the two files define different exact routes.
router = APIRouter(prefix="/customers", tags=["insights"], dependencies=[Depends(verify_api_key)])


# The single bundled call the LLM narration layer (Phase 7) uses. All
# assembly happens in insight_service.build_customer_insights -- this
# handler does not construct the response object itself.
@router.get("/{customer_unique_id}/insights", response_model=CustomerInsights)
def get_customer_insights(customer_unique_id: str):
    result = build_customer_insights(customer_unique_id)
    return CustomerInsights(**result)