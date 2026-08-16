"""api/schemas/churn.py"""

from typing import List

from pydantic import BaseModel

from api.schemas.common import SHAPFeature


class ChurnResponse(BaseModel):
    customer_unique_id: str
    churn_probability: float
    segment_id: int
    top_features: List[SHAPFeature]
