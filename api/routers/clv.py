"""api/routers/clv.py"""

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.schemas.clv import CLVResponse
from api.services.customer_service import get_customer_record
from api.services.prediction_service import build_clv_result

router = APIRouter(prefix="/clv", tags=["clv"], dependencies=[Depends(verify_api_key)])


@router.get("/{customer_unique_id}", response_model=CLVResponse)
def get_clv(customer_unique_id: str):
    record = get_customer_record(customer_unique_id)
    result = build_clv_result(record)
    return CLVResponse(customer_unique_id=customer_unique_id, **result)