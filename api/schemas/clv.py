"""api/schemas/clv.py"""

from typing import List

from pydantic import BaseModel

from api.schemas.common import SHAPFeature


class CLVResponse(BaseModel):
    customer_unique_id: str
    clv_ml: float
    clv_formula: float
    historical_spend: float
    top_features: List[SHAPFeature]
