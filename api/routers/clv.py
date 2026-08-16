"""api/routers/clv.py"""

import json

from fastapi import APIRouter, HTTPException

from api.data_store import store
from api.schemas import CLVResponse, SHAPFeature

router = APIRouter(prefix="/clv", tags=["clv"])


# Return formula-based and ML-based CLV with the top SHAP features.
@router.get("/{customer_unique_id}", response_model=CLVResponse)
def get_clv(customer_unique_id: str):
    record = store.get(customer_unique_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_unique_id}' not found")

    # Convert stored SHAP feature JSON into validated response objects.
    top_features = [SHAPFeature(**f) for f in json.loads(record["clv_shap_top_features"])]

    return CLVResponse(
        customer_unique_id=customer_unique_id,
        clv_ml=record["clv_ml"],
        clv_formula=record["clv_formula"],
        historical_spend=record["monetary"],
        top_features=top_features,
    )