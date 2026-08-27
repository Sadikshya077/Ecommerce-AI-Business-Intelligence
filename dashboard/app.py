"""dashboard/app.py"""

import pandas as pd
import streamlit as st

from api_client import (
    APIClientError,
    get_customer_kpis,
    get_customer_sample,
    get_forecast_summary,
    get_segments,
)
from chart_utils import horizontal_bar_chart, risk_value_scatter
from formatting import currency, percent, risk_category

st.set_page_config(page_title="Ecommerce BI Dashboard", layout="wide", page_icon="\U0001F4CA")

st.title("Business Intelligence Overview")
st.caption("A single view of customer health, revenue outlook, and where to focus retention and growth effort.")

# --- KPIs -----------------------------------------------------------------
try:
    kpis = get_customer_kpis()
except APIClientError as exc:
    st.error(f"Could not load overview metrics: {exc.message}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total customers", f"{kpis['total_customers']:,}")
col2.metric("Avg. predicted CLV", currency(kpis["avg_clv_ml"]))
col3.metric("Avg. churn probability", percent(kpis["avg_churn_probability"]))
col4.metric("High-risk customers", f"{kpis['high_risk_count']:,}")

st.divider()

# --- Business snapshot ------------------------------------------------------
st.subheader("Business snapshot")

high_risk_share = kpis["high_risk_count"] / kpis["total_customers"]
risk_note = (
    f"**{percent(high_risk_share)}** of the customer base ({kpis['high_risk_count']:,} customers) "
    f"is currently classified as high churn risk (predicted churn probability above 60%)."
)
value_note = (
    f"The average predicted lifetime value per customer is **{currency(kpis['avg_clv_ml'])}**, "
    f"against an average historical spend of **{currency(kpis['avg_historical_spend'])}** -- "
    f"reflecting a customer base where most customers have purchased only once so far."
)

try:
    forecast = get_forecast_summary()
    trend_note = (
        f"Revenue is forecast to trend **{forecast['trend_direction']}** over the next 30 days "
        f"({forecast['pct_change']:+.1f}% vs. the prior 30 days), predicting "
        f"**{currency(forecast['next_30d_predicted_revenue'])}**."
    )
except APIClientError:
    trend_note = "Sales forecast is currently unavailable."

st.markdown(f"- {risk_note}")
st.markdown(f"- {value_note}")
st.markdown(f"- {trend_note}")

st.divider()

# --- Segment overview -------------------------------------------------------
st.subheader("Customer segments")
st.caption(
    "Customer segmentation groups customers with similar purchasing behavior "
    "(recency, frequency, and spend) so the business can apply different strategies to each group."
)

try:
    segments = get_segments()
except APIClientError as exc:
    st.warning(f"Segment data unavailable: {exc.message}")
    segments = []

if segments:
    seg_df = pd.DataFrame(segments)
    col_a, col_b = st.columns(2)
    with col_a:
        horizontal_bar_chart(seg_df, "segment_label", "n_customers", title="Customers per segment")
    with col_b:
        horizontal_bar_chart(seg_df, "segment_label", "avg_monetary", title="Avg. historical spend per segment (R$)")

st.divider()

# --- Risk vs. value ----------------------------------------------------------
st.subheader("Risk vs. value")
st.caption(
    "Each point is one customer. The most commercially important group is "
    "**high value, high churn risk** (top-right) -- losing these customers carries the greatest revenue impact."
)

try:
    sample = get_customer_sample(n=300)
    sample_df = pd.DataFrame(sample)
    sample_df["risk_category"] = sample_df["churn_probability"].apply(risk_category)
    risk_value_scatter(
        sample_df, x_col="churn_probability", y_col="clv_ml",
        x_title="Churn probability", y_title="Predicted CLV (R$)",
        x_threshold=0.6, y_threshold=sample_df["clv_ml"].median(),
        hover_col="customer_unique_id",
        title="Churn risk vs. predicted CLV (sample of 300 customers)",
    )
    st.caption(
        "Based on a random sample of 300 customers for readability -- "
        "see the Churn Risk and Customer Value pages for full rankings."
    )
except APIClientError as exc:
    st.info(f"Risk/value view unavailable: {exc.message}")