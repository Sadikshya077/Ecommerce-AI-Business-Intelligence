"""api/routers/forecast.py"""

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends

from api.data_store import store
from api.dependencies import verify_api_key
from api.exceptions import ModelArtifactMissingError
from api.schemas.insights import MarketContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "models"

router = APIRouter(prefix="/forecast", tags=["forecast"], dependencies=[Depends(verify_api_key)])


# Short trend summary -- same content embedded in /customers/{id}/insights
@router.get("/summary", response_model=MarketContext)
def get_forecast_summary():
    if not store.forecast_summary:
        raise ModelArtifactMissingError("sales_forecast_summary.json not generated yet")
    return store.forecast_summary


# Full future forecast series -- for the Streamlit dashboard's forecast
# chart in Phase 8. Read fresh from disk per request rather than held in
# CustomerStore -- it's a larger, less frequently used payload than the
# summary, so keeping it out of the hot-path in-memory data is deliberate.
@router.get("/series")
def get_forecast_series():
    path = DATA_DIR / "sales_forecast.parquet"
    if not path.exists():
        raise ModelArtifactMissingError("sales_forecast.parquet not generated yet")
    df = pd.read_parquet(path)
    df["ds"] = df["ds"].astype(str)
    return df.to_dict(orient="records")