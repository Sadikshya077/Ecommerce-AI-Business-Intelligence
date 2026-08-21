"""api/schemas/common.py"""

from pydantic import BaseModel


# Liveness response
class HealthResponse(BaseModel):
    status: str


# Readiness response -- distinct from liveness, confirms data is loaded
class ReadinessResponse(BaseModel):
    status: str
    customers_loaded: int
    forecast_available: bool


# Shape of every error response returned by exceptions.py's handlers
class ErrorResponse(BaseModel):
    detail: str


# One SHAP-attributed feature, shared across churn/CLV/insights responses
class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
