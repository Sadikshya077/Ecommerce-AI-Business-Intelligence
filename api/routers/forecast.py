"""api/routers/forecast.py"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.data_store import load_forecast_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "models"

router = APIRouter(prefix="/forecast", tags=["forecast"])

# Load the forecast summary once when the API starts.
_summary_cache = load_forecast_summary()


# Return the short forecast trend summary.
@router.get("/summary")
def get_forecast_summary():
    if not _summary_cache:
        raise HTTPException(
            status_code=503,
            detail="Sales forecast not yet generated -- run models/forecasting/prophet_forecast.py",
        )
    return _summary_cache


# Return the full forecast series with dates and revenue predictions.
@router.get("/series")
def get_forecast_series():
    path = DATA_DIR / "sales_forecast.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Sales forecast not yet generated -- run models/forecasting/prophet_forecast.py",
        )
    df = pd.read_parquet(path)

    # Convert dates to strings so they can be returned as JSON.
    df["ds"] = df["ds"].astype(str)
    return df.to_dict(orient="records")