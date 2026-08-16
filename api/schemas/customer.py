"""api/schemas/customer.py"""

from pydantic import BaseModel


# Row shape for GET /customers/sample
class CustomerSample(BaseModel):
    customer_unique_id: str
    segment_label: str
    churn_probability: float
    clv_ml: float


# Response for GET /customers/{customer_unique_id} -- lighter than /insights,
# no SHAP detail
class CustomerProfile(BaseModel):
    customer_unique_id: str
    segment_id: int
    segment_label: str
    churn_probability: float
    clv_ml: float
    clv_formula: float
    historical_spend: float


# Row shape for GET /segments -- Phase 3's k=4 segmentation result
class SegmentProfile(BaseModel):
    segment_id: int
    segment_label: str
    n_customers: int
    avg_recency_days: float
    avg_frequency: float
    avg_monetary: float
    pct_of_customers: float
