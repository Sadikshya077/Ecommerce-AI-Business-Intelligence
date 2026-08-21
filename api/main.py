"""api/main.py"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.data_store import store
from api.exceptions import register_exception_handlers
from api.middleware import RequestLoggingMiddleware
from api.routers import churn, clv, customers, forecast, health, insights, narrative, segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")


# Loads all model outputs into memory once, before the app starts accepting requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    logger.info("Loaded %d customers into the prediction service", len(store))
    yield


settings = get_settings()

app = FastAPI(
    title="Ecommerce AI Business Intelligence API",
    description="Serves churn, CLV, SHAP, and sales-forecast context for the CLV-centric BI framework.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["X-API-Key"],
)
app.add_middleware(RequestLoggingMiddleware)

register_exception_handlers(app)

# Health/readiness stay unversioned (design assumption -- standard practice
# for infra-level liveness checks, distinct from the API contract itself).
# Everything else mounts under /api/v1 per the API versioning requirement.
app.include_router(health.router)
app.include_router(customers.router, prefix="/api/v1")
app.include_router(segments.router, prefix="/api/v1")
app.include_router(churn.router, prefix="/api/v1")
app.include_router(clv.router, prefix="/api/v1")
app.include_router(insights.router, prefix="/api/v1")
app.include_router(narrative.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")