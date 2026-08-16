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
# both models, and market context, in one response
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
