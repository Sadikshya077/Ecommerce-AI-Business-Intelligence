"""api/routers/customers.py"""

from typing import List

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.schemas.customer import CustomerProfile, CustomerSample
from api.services.customer_service import (
    get_churn_risk_leaderboard,
    get_clv_leaderboard,
    get_customer_record,
    sample_customers,
)

router = APIRouter(prefix="/customers", tags=["customers"], dependencies=[Depends(verify_api_key)])


# All registered before the {customer_unique_id} route below so these
# literal paths are never matched as a path parameter value
@router.get("/sample", response_model=List[CustomerSample])
def get_sample_customers(n: int = 5):
    return sample_customers(n)


# Actual top-N customers by churn risk platform-wide -- distinct from
# /sample, which is random. See api/data_store.py's top_by() for why this
# needed a small backend addition rather than being sorted client-side.
@router.get("/top-churn-risk", response_model=List[CustomerSample])
def get_top_churn_risk(n: int = 20):
    return get_churn_risk_leaderboard(n)


# Actual top-N customers by predicted CLV platform-wide
@router.get("/top-clv", response_model=List[CustomerSample])
def get_top_clv(n: int = 20):
    return get_clv_leaderboard(n)


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