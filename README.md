# Explainable AI-Driven Intelligent Business Analytics Framework

An end-to-end, CLV-centric business intelligence system built on the Olist Brazilian e-commerce dataset. The framework integrates data warehousing, multi-algorithm data mining, and machine learning into a single pipeline, explains its predictions with SHAP, narrates them in plain language through a grounded LLM business analyst, and serves the whole thing through a FastAPI backend and a Streamlit BI dashboard.

## Core idea

Customer Lifetime Value (CLV) is the centerpiece of this framework, not one model among several. Every other component exists to make that one estimate better, more explainable, and more actionable:

- **Segmentation** identifies who the customer is
- **Association rule mining** reveals what they tend to buy (tested rigorously; no meaningful cross-category signal was found in this dataset -- see Methodology notes)
- **Churn prediction** estimates whether they'll keep buying -- and its output feeds directly into the CLV formula as the survival term
- **Sales forecasting** provides market context for interpreting individual CLV numbers
- **SHAP** explains why each model predicted what it did, for every customer
- **A grounded LLM narration layer**, constrained to only the structured SHAP/prediction output supplied to it, translates results into plain-language business narratives -- with a lightweight automated faithfulness check and a documented manual evaluation process
- **A FastAPI service** exposes everything through a versioned REST API, with typed error handling, authentication, and structured logging
- **A Streamlit dashboard** turns all of the above into a browsable BI product: executive overview, segment explorer, churn risk and customer value leaderboards, sales forecast, and a full per-customer "Customer 360" view

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
| API | FastAPI, Pydantic, pytest |
| LLM narration | Google Gemini API, schema-constrained structured output |
| Dashboard | Streamlit, Plotly |
| Reproducibility | Docker Compose (planned) |

## Project status

| Phase | Description | Status |
|---|---|---|
| 1 | Data warehouse -- ETL pipeline, PostgreSQL star schema, OLAP queries | Complete |
| 2 | Feature engineering (customer / product / time) | Complete |
| 3 | Segmentation & association rule mining | Complete |
| 4 | Churn -> CLV chain | Complete |
| 5 | SHAP explainability | Complete |
| 6 | FastAPI service (canonical layered architecture, /api/v1) | Complete |
| 7 | LLM narration layer + evaluation harness | Built; manual rubric scoring of sample narratives in progress |
| 8 | Streamlit dashboard (all 6 pages) | Complete |
| 9 | Docker Compose packaging | Not started |
| 10 | Final hardening & research validation review | Not started |

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
|   |-- forecasting/               # Prophet, with tail-trimming for Olist's known data drop-off
|-- explainability/                # SHAP, global + per-customer
|-- api/                           # FastAPI service
|   |-- main.py, config.py, dependencies.py, middleware.py, exceptions.py
|   |-- routers/                   # health, customers, segments, churn, clv, insights, narrative, forecast
|   |-- schemas/                   # Pydantic request/response contracts
|   |-- services/                  # business logic layer (routers never touch models/DB directly)
|-- llm/                           # Grounded LLM narration
|   |-- config.py, prompts.py, client.py, schemas.py, faithfulness.py, narrator.py
|   |-- evaluation/                # stratified sampling + manual scoring tools
|-- dashboard/                     # Streamlit BI application
|   |-- api_client.py              # sole HTTP boundary -- never imports backend internals
|   |-- formatting.py, chart_utils.py, segment_insights.py
|   |-- app.py                     # Overview
|   |-- pages/                     # Segment Explorer, Churn Risk, Customer Value,
|                                     Sales Forecast, Customer 360
|-- docker/                        # Phase 9 (planned)
|-- reports/                       # methodology writeups, figures, LLM evaluation
|-- tests/                         # ~75 automated tests
|-- .env.example
|-- pytest.ini
|-- requirements.txt
|-- README.md
```

## Getting started

### Prerequisites

- Python 3.11+
- PostgreSQL 16+ ([download](https://www.postgresql.org/download/))
- The [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle
- A free [Google Gemini API key](https://aistudio.google.com/apikey) (no credit card required) for the LLM narration layer

### Setup

```powershell
# 1. Clone and enter the repo
git clone <repo-url>
cd Ecommerce-AI-Business-Intelligence

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# edit .env with your PostgreSQL credentials, a generated API_KEY, and your GEMINI_API_KEY

# 5. Download the Olist dataset from Kaggle and place all CSVs in data\raw\

# 6. Create the database and apply the star schema
psql -U postgres -c "CREATE DATABASE ecommerce_bi;"
psql -U postgres -d ecommerce_bi -f warehouse\schema.sql

# 7. Run the full offline pipeline (ETL -> warehouse -> features -> models -> SHAP)
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
```

### Running the application

```powershell
# Terminal 1 -- backend
uvicorn api.main:app --reload

# Terminal 2 -- dashboard
streamlit run dashboard/app.py
```

- API docs: `http://127.0.0.1:8000/docs`
- Dashboard: `http://localhost:8501`
- Health check (no auth): `http://127.0.0.1:8000/health`

### Running tests

```powershell
pytest -v
```

All tests use synthetic fixtures and mocked LLM calls -- no real API calls, no database, no trained models required to run the suite.

## Methodology notes

- **Comparative evaluation is applied selectively** -- churn and CLV, the two models with a direct mathematical dependency, are each evaluated across multiple algorithms. Segmentation, forecasting, and association mining use a single literature-standard method appropriate to their supporting role.
- **Churn probability feeds the CLV formula directly**: `CLV = avg_order_value x purchase_frequency x (1 / churn_probability)`, evaluated alongside an ML-based CLV regressor for comparison.
- **A real leakage bug was caught and fixed during churn model development** -- `segment_id` (derived from RFM features including recency) leaked the churn label through cluster membership. Removing it dropped ROC-AUC from a suspicious 0.98 to a realistic 0.76, documented in `reports/methodology_phase4.md`.
- **Association rule mining returned a genuine null result** -- only 0.7% of orders span multiple categories; no statistically meaningful cross-category rules exist in this dataset. Reported as a finding, not hidden.
- **The LLM narration layer is grounded, not free-reasoning** -- it receives only structured, schema-validated JSON output from the SHAP and prediction layers and is constrained to reference values present in that input. A lightweight automated faithfulness check flags unsupported feature mentions and causal-language violations; this is explicitly documented as a partial check, not proof of complete faithfulness -- full evaluation requires the manual rubric scoring in `reports/llm_evaluation_rubric.md`.
- **Data validation is non-destructive** -- referential integrity issues (orphaned foreign keys, null/duplicate primary keys) are logged as warnings, not silently dropped, so data quality issues remain visible and auditable.
- **"High risk" and "high value" customer thresholds are dashboard conventions, not model outputs** -- no such categorical threshold exists anywhere in the trained models; they're defined once in `dashboard/formatting.py` (churn probability tiers) and via the 75th percentile of CLV (computed server-side in `api/data_store.py`), and labeled as such in the UI.

## API overview

All endpoints except `/health` and `/ready` require an `X-API-Key` header and live under `/api/v1/`:

| Endpoint | Purpose |
|---|---|
| `GET /customers/{id}` | Lightweight customer profile |
| `GET /customers/{id}/insights` | Bundled segment + churn + CLV + SHAP + behavioral profile + market context -- the LLM's sole grounding source |
| `GET /customers/{id}/narrative` | Grounded LLM business narrative (separate from `/insights` so a narration failure never affects analytical endpoints) |
| `GET /customers/sample`, `/customers/top-churn-risk`, `/customers/top-clv` | Bulk customer views |
| `GET /customers/kpis` | Population-level KPI aggregates |
| `GET /segments` | Phase 3 segmentation profile |
| `GET /churn/{id}`, `/clv/{id}` | Individual model outputs with SHAP |
| `GET /forecast/summary`, `/forecast/series` | Sales forecast |

## License

Academic project -- Institute of Engineering (IOE), Thapathali Campus.