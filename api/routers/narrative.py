"""api/routers/narrative.py"""

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.services.narration_service import get_customer_narrative
from llm.schemas import NarrativeResult

router = APIRouter(prefix="/customers", tags=["narrative"], dependencies=[Depends(verify_api_key)])


# LLM-generated business narrative for one customer, grounded in the same
# context /customers/{id}/insights returns. This is a SEPARATE endpoint
# deliberately: it's what actually enforces "the LLM must never become a
# single point of failure" -- /insights, /churn, /clv never call into
# narration_service, so they keep working regardless of this endpoint's state.
@router.get("/{customer_unique_id}/narrative", response_model=NarrativeResult)
def get_narrative(customer_unique_id: str):
    return get_customer_narrative(customer_unique_id)
