"""api/routers/churn.py"""

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.schemas.churn import ChurnResponse
from api.services.customer_service import get_customer_record
from api.services.prediction_service import build_churn_result

router = APIRouter(prefix="/churn", tags=["churn"], dependencies=[Depends(verify_api_key)])


# Business logic lives in services/ -- this handler only receives the
# request, calls the service, and returns the response model. Not-found
# is raised as CustomerNotFoundError inside get_customer_record and mapped
# to a safe 404 by the centralized handler in exceptions.py.
@router.get("/{customer_unique_id}", response_model=ChurnResponse)
def get_churn(customer_unique_id: str):
    record = get_customer_record(customer_unique_id)
    result = build_churn_result(record)
    return ChurnResponse(customer_unique_id=customer_unique_id, **result)