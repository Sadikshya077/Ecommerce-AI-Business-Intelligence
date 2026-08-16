"""api/routers/customers.py"""

from typing import List

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.schemas.customer import CustomerProfile, CustomerSample
from api.services.customer_service import get_customer_record, sample_customers

router = APIRouter(prefix="/customers", tags=["customers"], dependencies=[Depends(verify_api_key)])


# Registered before the {customer_unique_id} route below so "sample" is
# never matched as a path parameter value
@router.get("/sample", response_model=List[CustomerSample])
def get_sample_customers(n: int = 5):
    return sample_customers(n)


# Lightweight customer profile -- segment, churn, both CLV estimates,
# without the SHAP detail /churn, /clv, or /insights carry
@router.get("/{customer_unique_id}", response_model=CustomerProfile)
def get_customer_profile(customer_unique_id: str):
    record = get_customer_record(customer_unique_id)
    return CustomerProfile(
        customer_unique_id=customer_unique_id,
        segment_id=int(record["segment_id"]),
        segment_label=record["segment_label"],
        churn_probability=record["churn_probability"],
        clv_ml=record["clv_ml"],
        clv_formula=record["clv_formula"],
        historical_spend=record["monetary"],
    )