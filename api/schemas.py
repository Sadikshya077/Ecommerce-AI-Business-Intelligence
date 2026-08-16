"""api/schemas.py"""

from typing import List, Optional

from pydantic import BaseModel


# Define a SHAP feature with its name, impact, and original value.
class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


# Define the response structure for a customer's churn prediction.
class ChurnResponse(BaseModel):
    customer_unique_id: str
    churn_probability: float
    segment_id: int
    top_features: List[SHAPFeature]


# Define the response structure for a customer's CLV predictions.
class CLVResponse(BaseModel):
    customer_unique_id: str
    clv_ml: float
    clv_formula: float
    historical_spend: float
    top_features: List[SHAPFeature]


# Define the global sales forecast context used in customer insights.
class MarketContext(BaseModel):
    last_actual_date: str
    forecast_horizon_days: int
    prior_30d_actual_revenue: float
    next_30d_predicted_revenue: float
    pct_change: float
    trend_direction: str


# Define the combined response returned by the customer insights endpoint.
class CustomerInsights(BaseModel):
    customer_unique_id: str
    segment_id: int
    segment_label: str
    churn_probability: float
    clv_ml: float
    clv_formula: float
    historical_spend: float
    churn_top_features: List[SHAPFeature]
    clv_top_features: List[SHAPFeature]
    association_rules_note: str
    market_context: Optional[MarketContext] = None