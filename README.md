# Explainable AI-Driven Intelligent Business Analytics Framework

An end-to-end, CLV-centric business intelligence system built on the Olist Brazilian e-commerce dataset. The framework integrates data warehousing, multi-algorithm data mining, and machine learning into a single pipeline, explains its predictions with SHAP, and narrates them in plain language through an LLM-based business analyst — all served through a production-inspired stack.

## Core idea

Customer Lifetime Value (CLV) is the centerpiece of this framework, not one model among several. Every other component exists to make that one estimate better and more trustworthy:

- **Segmentation** identifies who the customer is
- **Association rule mining** reveals what they tend to buy
- **Churn prediction** estimates whether they'll keep buying — and its output feeds directly into the CLV formula as the survival term
- **Sales forecasting** provides market context for interpreting individual CLV numbers
- **SHAP** explains why each model predicted what it did
- **An LLM narration layer**, grounded strictly in the SHAP and pipeline output, translates the results into strategic recommendations a non-technical stakeholder can act on

## Architecture

```
Olist dataset
      │
      ▼
ETL pipeline (extract → validate → transform → load)
      │
      ▼
PostgreSQL data warehouse (star schema)
      │
      ▼
OLAP analytics (roll-up, drill-down, slice, dice, pivot)
      │
      ▼
Feature engineering (customer / product / time)
      │
      ├── Segmentation (K-Means) ──┐
      ├── Association rules (FP-Growth)
      ├── Sales forecast (Prophet)
      │                            │
      │                            ▼
      └────────────────────► Churn prediction (LogReg vs XGBoost)
                                    │
                                    ▼
                          CLV prediction (formula vs XGBoost/LightGBM)
                                    │
                                    ▼
                    MLflow experiment tracking · SHAP explainability
                                    │
                                    ▼
                        FastAPI prediction service
                                    │
                                    ▼
                      LLM business analyst (grounded narration)
                                    │
                                    ▼
                     Streamlit BI dashboard + executive reports
```

Association rules and sales forecasting feed the LLM narration layer as contextual signal rather than as direct CLV inputs — segmentation and churn are the two components with a mathematical dependency into CLV.

## Tech stack

| Layer | Tools |
|---|---|
| ETL / warehouse | Python, pandas, PostgreSQL |
| Data mining | scikit-learn (K-Means), mlxtend (FP-Growth) |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Forecasting | Prophet |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| Serving | FastAPI |
| LLM narration | Anthropic API, JSON-constrained prompts |
| Dashboard | Streamlit |
| Reproducibility | Docker Compose |

## Project status

| Phase | Description | Status |
|---|---|---|
| 1 | Data warehouse — ETL pipeline, PostgreSQL star schema, OLAP queries | ✅ Complete |
| 2 | Feature engineering (customer / product / time) | ⏳ Not started |
| 3 | Segmentation & association rule mining | ⏳ Not started |
| 4 | Churn → CLV chain | ⏳ Not started |
| 5 | SHAP explainability | ⏳ Not started |
| 6 | FastAPI prediction service | ⏳ Not started |
| 7 | LLM narration layer | ⏳ Not started |
| 8 | Streamlit dashboard | ⏳ Not started |
| 9 | Docker Compose packaging | ⏳ Not started |

## Project structure

```
Ecommerce-AI-Business-Intelligence/
├── data/
│   ├── raw/                 # Olist CSVs (not tracked in git)
│   └── processed/           # cleaned parquet output (not tracked in git)
├── etl/                     # extract → validate → transform → load pipeline
│   ├── config.py
│   ├── logger.py
│   ├── exceptions.py
│   ├── report.py
│   ├── extract.py
│   ├── validate.py
│   ├── transform.py
│   ├── load.py
│   └── pipeline.py
├── warehouse/                # PostgreSQL star schema + loader + OLAP queries
│   ├── db.py
│   ├── schema.sql
│   ├── load_warehouse.py
│   └── olap_queries.sql
├── features/                  # Phase 2
├── models/                     # Phase 3-4
│   ├── segmentation/
│   ├── association_rules/
│   ├── churn/
│   ├── clv/
│   └── forecasting/
├── explainability/               # Phase 5
├── api/                           # Phase 6
├── llm/                            # Phase 7
├── dashboard/                       # Phase 8
├── docker/                           # Phase 9
├── notebooks/                         # EDA and experimentation only
├── reports/                            # methodology, results, figures
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

- Python 3.11+
- PostgreSQL 16+ ([download](https://www.postgresql.org/download/))
- The [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle

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
# edit .env with your local PostgreSQL credentials

# 5. Download the Olist dataset from Kaggle and place all CSVs in data\raw\

# 6. Create the database
psql -U postgres -c "CREATE DATABASE ecommerce_bi;"

# 7. Apply the star schema
psql -U postgres -d ecommerce_bi -f warehouse\schema.sql

# 8. Run the ETL pipeline
python -m etl.pipeline

# 9. Load the data warehouse
python warehouse\load_warehouse.py
```

### Verifying the setup

Run the sample OLAP queries against the warehouse to confirm everything loaded correctly:

```powershell
psql -U postgres -d ecommerce_bi -f warehouse\olap_queries.sql
```

## Methodology notes

- **Comparative evaluation is applied selectively** — churn and CLV, the two models with a direct mathematical dependency, are each evaluated across multiple algorithms. Segmentation, forecasting, and association mining use a single literature-standard method appropriate to their supporting role.
- **Churn probability feeds the CLV formula directly**: `CLV = avg_order_value × purchase_frequency × (1 / churn_probability)`, evaluated alongside an ML-based CLV regressor for comparison.
- **The LLM narration layer is grounded, not free-reasoning** — it receives only structured JSON output from the SHAP and prediction layers and is constrained to reference values present in that input, avoiding hallucinated business claims.
- **Data validation is non-destructive** — referential integrity issues (orphaned foreign keys, null/duplicate primary keys) are logged as warnings, not silently dropped, so data quality issues remain visible and auditable.