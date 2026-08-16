"""api/routers/churn.py"""

import json

from fastapi import APIRouter, HTTPException

from api.data_store import store
from api.schemas import ChurnResponse, SHAPFeature

router = APIRouter(prefix="/churn", tags=["churn"])


# Return churn probability and the top SHAP features for a customer.
@router.get("/{customer_unique_id}", response_model=ChurnResponse)
def get_churn(customer_unique_id: str):
    record = store.get(customer_unique_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_unique_id}' not found")

    # Convert stored SHAP feature JSON into validated response objects.
    top_features = [SHAPFeature(**f) for f in json.loads(record["churn_shap_top_features"])]

    return ChurnResponse(
        customer_unique_id=customer_unique_id,
        churn_probability=record["churn_probability"],
        segment_id=int(record["segment_id"]),
        top_features=top_features,
    )