"""api/routers/health.py"""

from fastapi import APIRouter
from pydantic import BaseModel

from api.data_store import store

router = APIRouter(tags=["health"])


# Define the response schema for the health endpoint.
class HealthResponse(BaseModel):
    status: str


# Define the response schema for the readiness endpoint.
class ReadinessResponse(BaseModel):
    status: str
    customers_loaded: int
    forecast_available: bool


# Return a simple liveness response without checking application data.
@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


# Check whether customer data and forecast data are available.
@router.get("/ready", response_model=ReadinessResponse)
def readiness_check():
    is_ready = len(store) > 0
    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        customers_loaded=len(store),
        forecast_available=bool(store.forecast_summary),
    )