"""api/schemas/insights.py"""

from typing import List, Optional

from pydantic import BaseModel

from api.schemas.common import SHAPFeature


# Global sales-forecast signal, embedded in customer insights as market
# context -- not a per-customer prediction, the same values for everyone
class MarketContext(BaseModel):
    last_actual_date: str
    forecast_horizon_days: int
    prior_30d_actual_revenue: float
    next_30d_predicted_revenue: float
    pct_change: float
    trend_direction: str


# The integrated framework artifact: segmentation, churn, CLV, SHAP for
# both models, behavioral profile, and market context, in one response
class CustomerInsights(BaseModel):
    customer_unique_id: str
    segment_id: int
    segment_label: str
    churn_probability: float
    clv_ml: float
    clv_formula: float
    historical_spend: float
    # Behavioral profile fields, reused as-is from Phase 2's feature
    # engineering output (customer_features.parquet) -- not recalculated
    # here. avg_review_score is Optional: it's a pre-existing, legitimate
    # gap for customers with no submitted review, not introduced by this schema.
    frequency: float
    recency_days: float
    avg_order_value: float
    avg_freight: float
    avg_delivery_days: Optional[float] = None
    avg_review_score: Optional[float] = None
    churn_top_features: List[SHAPFeature]
    clv_top_features: List[SHAPFeature]
    association_rules_note: str
    market_context: Optional[MarketContext] = None