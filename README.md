# Explainable AI-Driven Intelligent Business Analytics Framework

An end-to-end, CLV-centric business intelligence system built on the Olist Brazilian e-commerce dataset. The framework integrates data warehousing, multi-algorithm data mining, and machine learning into a single pipeline, explains its predictions with SHAP, narrates them in plain language through a grounded LLM business analyst, and serves the whole system through a FastAPI backend and a Streamlit BI dashboard -- fully containerized for reproducible deployment.

## Overview

Customer Lifetime Value (CLV) is the centerpiece of this framework, not one model among several. Every other component exists to make that estimate better, more explainable, and more actionable:

- **Segmentation** identifies who the customer is (K-Means, k=4)
- **Association rule mining** tests what customers tend to buy together (FP-Growth) -- a rigorous null result on this dataset, reported as a finding rather than hidden
- **Churn prediction** estimates whether a customer will keep buying (Logistic Regression vs. XGBoost), and its output feeds directly into the CLV formula as the survival term
- **CLV prediction** compares a classical formula-based estimate against an ML regressor (XGBoost vs. LightGBM)
- **Sales forecasting** (Prophet) provides market context for interpreting individual CLV numbers
- **SHAP** explains why each model predicted what it did, for every customer, both models
- **A grounded LLM narration layer** (Google Gemini, schema-constrained structured output) translates model results into plain-language business narratives, constrained to only the structured data supplied to it, with a lightweight automated faithfulness check and a documented manual evaluation rubric
- **A FastAPI service** exposes everything through a versioned REST API with typed error handling, authentication, and structured logging
- **A Streamlit dashboard** turns the full pipeline into a browsable BI product: executive overview, segment explorer, churn risk and customer value leaderboards, sales forecast, and a per-customer "Customer 360" view
- **Docker Compose** packages the full runtime (PostgreSQL, MLflow, FastAPI, Streamlit) for one-command reproducibility; pre-built images are published to Docker Hub

## Architecture

```
Olist dataset
      |
      v
ETL pipeline (extract -> validate -> transform -> load)
      |
      v
PostgreSQL data warehouse (star schema)
      |
      v
OLAP analytics (roll-up, drill-down, slice, dice, pivot)
      |
      v
Feature engineering (customer / product / time)
      |
      +-- Segmentation (K-Means) --+
      +-- Association rules (FP-Growth)
      +-- Sales forecast (Prophet)
      |                            |
      |                            v
      +----------------------> Churn prediction (LogReg vs XGBoost)
                                    |
                                    v
                          CLV prediction (formula vs XGBoost/LightGBM)
                                    |
                                    v
                            SHAP explainability
                                    |
                                    v
                        FastAPI service (/api/v1/...)
                             |              |
                             v              v
                  LLM business analyst   Streamlit BI dashboard
                  (grounded narration)   (Overview, Segments, Churn,
                                          Customer Value, Forecast,
                                          Customer 360)
```

Association rules and sales forecasting feed the LLM narration layer and the dashboard as contextual signal, not as direct CLV inputs -- segmentation and churn are the two components with a mathematical dependency into CLV.

## Tech stack

| Layer | Tools |
|---|---|
| ETL / warehouse | Python, pandas, PostgreSQL |
| Data mining | scikit-learn (K-Means), mlxtend (FP-Growth) |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Forecasting | Prophet |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI, Pydantic, pytest |
| LLM narration | Google Gemini API, schema-constrained structured output |
| Dashboard | Streamlit, Plotly |
| Deployment | Docker, Docker Compose |

## Project structure

```
Ecommerce-AI-Business-Intelligence/
|-- data/
|   |-- raw/                       # Olist CSVs (not tracked in git)
|   |-- processed/                 # cleaned parquet + feature/model output (not tracked in git)
|-- etl/                           # extract -> validate -> transform -> load pipeline
|-- warehouse/                     # PostgreSQL star schema + loader + OLAP queries
|-- features/                      # customer / product / time feature engineering
|-- models/
|   |-- segmentation/              # K-Means (k=4)
|   |-- association_rules/         # FP-Growth
|   |-- churn/                     # LogReg vs XGBoost, data-driven churn window
|   |-- clv/                       # formula-based vs XGBoost/LightGBM
|   |-- forecasting/               # Prophet, with tail-trimming for known data drop-off
|-- explainability/                # SHAP, global + per-customer
|-- api/                           # FastAPI service
|   |-- main.py, config.py, dependencies.py, middleware.py, exceptions.py
|   |-- routers/                   # health, customers, segments, churn, clv, insights, narrative, forecast
|   |-- schemas/                   # Pydantic request/response contracts
|   |-- services/                  # business logic layer
|   |-- requirements.txt           # API service dependencies
|-- llm/                           # Grounded LLM narration
|   |-- config.py, prompts.py, client.py, schemas.py, faithfulness.py, narrator.py
|   |-- evaluation/                # stratified sampling + manual scoring tools
|-- dashboard/                     # Streamlit BI application
|   |-- api_client.py              # sole HTTP boundary -- never imports backend internals
|   |-- formatting.py, chart_utils.py, segment_insights.py
|   |-- app.py                     # Overview
|   |-- pages/                     # Segment Explorer, Churn Risk, Customer Value,
|   |                                Sales Forecast, Customer 360
|   |-- requirements.txt           # Dashboard dependencies
|-- docker/                        # Dockerfiles for api, dashboard, mlflow
|-- reports/                       # methodology writeups, figures, LLM evaluation
|-- tests/                         # automated test suite
|-- docker-compose.yml
|-- .env.example
|-- pytest.ini
|-- README.md
```

## Getting started

### Prerequisites

- Python 3.13+
- Docker and Docker Compose (recommended path), or PostgreSQL 16+ for a local (non-Docker) setup
- The [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle
- A free [Google Gemini API key](https://aistudio.google.com/apikey) for the LLM narration layer

### Configuration

```powershell
git clone <repo-url>
cd Ecommerce-AI-Business-Intelligence
copy .env.example .env
```

Edit `.env` with your PostgreSQL credentials, a generated `API_KEY`, and your `GEMINI_API_KEY`.

### Running with Docker

```powershell
docker compose up --build
```

Pre-built images are also published on Docker Hub -- `rubysah32/ecommerce-ai-api`, `rubysah32/ecommerce-ai-dashboard`, and `rubysah32/ecommerce-ai-mlflow` -- for running the stack without cloning or building anything. See "Running from Docker Hub images only" below.

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`
- MLflow: `http://localhost:5000`

The offline analytical pipeline (ETL through SHAP) is intentionally not part of the always-on Compose services -- it runs as a deliberate, inspectable sequence of scripts against the `postgres` service (its port is published to the host), populating `data/processed/`, which the `api` service reads via a mounted volume.

### Running from Docker Hub images only

This works without cloning the repository or building anything -- three published images plus the official `postgres:16` image are all that's needed.

**What you still need**, even with pre-built images:
- A `.env` file (same variables as `.env.example`: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `API_KEY`, `GEMINI_API_KEY`)
- A populated `data/processed/` folder. This is the output of the project's offline analytical pipeline (ETL through SHAP) and is deliberately **not baked into the `api` image** -- keeping the image lean and the pipeline a separate, inspectable step is a design choice, not an oversight. Without it, the containers start and stay healthy, but `/ready` reports no data and customer endpoints return empty results. Obtain this folder either by running the pipeline once (see "Running locally without Docker" above) against the `postgres` container below, or by copying a `data/processed/` folder from someone who already has one.

**Steps:**

1. Create a working folder containing `.env` and, once available, a `data/processed/` subfolder.
2. Save the compose file below as `docker-compose.images.yml` in that same folder.
3. Run:
   ```powershell
   docker compose -f docker-compose.images.yml up
   ```

`postgres` and `mlflow` have no functional dependency from `api` at request time -- the API only ever reads the `data/processed/` volume above, it never queries either live. Both remain in this file because they're part of the documented architecture (and `postgres` is needed if you intend to run the offline pipeline against this same stack), not because `api` talks to them per request.

Once running:
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`
- MLflow: `http://localhost:5000`

### Running locally without Docker

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r api\requirements.txt -r dashboard\requirements.txt

psql -U postgres -c "CREATE DATABASE ecommerce_bi;"
psql -U postgres -d ecommerce_bi -f warehouse\schema.sql

python -m etl.pipeline
python warehouse\load_warehouse.py
python -m features.customer_features
python -m features.product_features
python -m features.time_features
python -m models.segmentation.kmeans_segmentation
python -m models.association_rules.fp_growth
python -m models.churn.train_churn
python -m models.churn.evaluate_churn
python -m models.clv.clv_formula
python -m models.clv.train_clv_model
python -m models.clv.evaluate_clv
python -m models.forecasting.prophet_forecast
python -m explainability.shap_explainer

# Terminal 1
uvicorn api.main:app --reload
# Terminal 2
streamlit run dashboard\app.py
```

### Testing

```powershell
pytest -v
```

The full suite runs against synthetic fixtures with mocked LLM calls -- no live database, trained models, or API credits required.

## Methodology highlights

- **Comparative evaluation is applied selectively.** Churn and CLV -- the two components with a direct mathematical dependency -- are each evaluated across multiple algorithms. Segmentation, forecasting, and association mining use a single literature-standard method appropriate to their supporting role, rather than padding the comparison table.
- **Churn probability feeds the CLV formula directly**: `CLV = avg_order_value x purchase_frequency x (1 / churn_probability)`, evaluated alongside an ML-based CLV regressor for comparison.
- **A feature-leakage issue was identified and corrected during churn model development.** `segment_id`, derived from RFM features including recency, was found to leak the churn label through cluster membership. Removing it brought ROC-AUC down from an implausible 0.98 to a realistic 0.76 -- documented in `reports/methodology_phase4.md` as evidence of methodological rigor, not treated as a setback.
- **Association rule mining returned a genuine, well-supported null result**: only 0.7% of orders span multiple product categories, and no statistically meaningful cross-category rules exist in this dataset at any reasonable support/lift threshold tested.
- **The LLM narration layer is grounded, not free-reasoning.** It receives only structured, schema-validated JSON derived from the SHAP and prediction layers, and is explicitly constrained to reference values present in that input. An automated faithfulness check flags unsupported feature mentions and causal-language violations; this is documented as a lightweight, partial check rather than proof of complete faithfulness -- a full evaluation follows the rubric in `reports/llm_evaluation_rubric.md`.
- **Data validation is non-destructive.** Referential integrity issues (orphaned foreign keys, null or duplicate primary keys) are logged as warnings rather than silently dropped, keeping data quality issues visible and auditable.
- **"High risk" and "high value" customer thresholds are dashboard-level conventions, not model outputs.** No such categorical threshold exists in any trained model; both are defined once, transparently, and labeled as UI conventions rather than presented as ground truth.

## API reference

All endpoints except `/health` and `/ready` require an `X-API-Key` header and live under `/api/v1/`.

| Endpoint | Purpose |
|---|---|
| `GET /customers/{id}` | Lightweight customer profile |
| `GET /customers/{id}/insights` | Bundled segment, churn, CLV, SHAP, behavioral profile, and market context -- the LLM's sole grounding source |
| `GET /customers/{id}/narrative` | Grounded LLM business narrative, served as a separate endpoint so a narration failure never affects analytical endpoints |
| `GET /customers/sample`, `/customers/top-churn-risk`, `/customers/top-clv` | Bulk customer views |
| `GET /customers/kpis` | Population-level KPI aggregates |
| `GET /segments` | Segmentation profile |
| `GET /churn/{id}`, `/clv/{id}` | Individual model outputs with SHAP |
| `GET /forecast/summary`, `/forecast/series` | Sales forecast |

Interactive documentation is available at `/docs` once the API is running.

