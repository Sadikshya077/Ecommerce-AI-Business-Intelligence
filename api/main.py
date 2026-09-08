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

from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>E-commerce AI Business Intelligence API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 60px auto;
                padding: 0 25px;
                color: #222;
            }

            h1 {
                margin-bottom: 8px;
            }

            .subtitle {
                color: #666;
                margin-bottom: 30px;
            }

            .status {
                display: inline-block;
                padding: 8px 14px;
                background: #e8f5e9;
                color: #2e7d32;
                border-radius: 6px;
                margin-bottom: 25px;
            }

            .services {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin: 20px 0 30px;
            }

            .service {
                padding: 16px;
                border: 1px solid #ddd;
                border-radius: 8px;
            }

            .links a {
                display: inline-block;
                margin-right: 15px;
                text-decoration: none;
            }

            code {
                background: #f4f4f4;
                padding: 3px 6px;
                border-radius: 4px;
            }
        </style>
    </head>

    <body>
        <h1>E-commerce AI Business Intelligence API</h1>

        <p class="subtitle">
            Customer analytics and AI-powered business intelligence services.
        </p>

        <div class="status">● API Operational</div>

        <h2>Available Services</h2>

        <div class="services">
            <div class="service">Customer Insights</div>
            <div class="service">Churn Risk Prediction</div>
            <div class="service">Customer Lifetime Value</div>
            <div class="service">Customer Segmentation</div>
            <div class="service">SHAP Explanations</div>
            <div class="service">Sales Forecasting</div>
            <div class="service">LLM Narratives</div>
        </div>

        <h2>API Information</h2>

        <p><strong>Version:</strong> 0.1.0</p>
        <p><strong>API Base:</strong> <code>/api/v1</code></p>

        <h2>Documentation</h2>

        <div class="links">
            <a href="/docs">Swagger UI →</a>
            <a href="/redoc">ReDoc →</a>
            <a href="/health">Health Check →</a>
        </div>
    </body>
    </html>
    """


app.include_router(health.router)
app.include_router(customers.router, prefix="/api/v1")
app.include_router(segments.router, prefix="/api/v1")
app.include_router(churn.router, prefix="/api/v1")
app.include_router(clv.router, prefix="/api/v1")
app.include_router(insights.router, prefix="/api/v1")
app.include_router(narrative.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")