"""api/main.py"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.data_store import store
from api.exceptions import register_exception_handlers
from api.middleware import RequestLoggingMiddleware
from api.routers import health

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")


# Load all model outputs before the API starts accepting requests.
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

# Configure CORS for the allowed frontend origins and request methods.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["X-API-Key"],
)
app.add_middleware(RequestLoggingMiddleware)

# Register the application's custom exception handlers.
register_exception_handlers(app)

# Register the health and readiness endpoints.
app.include_router(health.router)