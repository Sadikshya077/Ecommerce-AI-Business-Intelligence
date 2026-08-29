"""dashboard/pages/Customer_Value.py"""

import pandas as pd
import streamlit as st

from api_client import APIClientError, get_clv_leaderboard, get_customer_kpis, get_segments
from chart_utils import horizontal_bar_chart, risk_value_scatter
from formatting import currency, risk_badge

st.set_page_config(page_title="Customer Value", layout="wide", page_icon="\U0001F4B0")
st.title("Customer Value")
st.caption(
    "Customer Lifetime Value (CLV) estimates the future economic value a customer may generate. "
    "The ML estimate is predictive and should not be interpreted as guaranteed revenue."
)

n = st.slider("Number of customers to analyze", min_value=10, max_value=100, value=30, step=10)

try:
    leaderboard = get_clv_leaderboard(n)
    kpis = get_customer_kpis()
except APIClientError as exc:
    st.error(f"Could not load customer value data: {exc.message}")
    st.stop()

if not leaderboard:
    st.info("No customer data available.")
    st.stop()

df = pd.DataFrame(leaderboard)

# --- KPI cards ----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg. predicted CLV (platform-wide)", currency(kpis["avg_clv_ml"]))
col2.metric("Highest predicted CLV", currency(df["clv_ml"].max()))
col3.metric("Total predicted CLV (platform-wide)", currency(kpis["total_clv_ml"]))
col4.metric("High-value customers (top 25%)", f"{kpis['high_value_count']:,}")
st.caption(
    f'"High value" here means predicted CLV above {currency(kpis["clv_p75_threshold"])} '
    f"(the platform's 75th percentile) -- a dashboard convention, not a model output."
)

st.divider()

# --- Leaderboard ----------------------------------------------------------
st.subheader("Highest predicted value customers")
display_df = df.copy()
display_df["Predicted CLV"] = display_df["clv_ml"].apply(currency)
display_df["Churn risk"] = display_df["churn_probability"].apply(risk_badge)
st.dataframe(
    display_df.rename(columns={"customer_unique_id": "Customer ID", "segment_label": "Segment"})[
        ["Customer ID", "Segment", "Predicted CLV", "Churn risk"]
    ],
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "This view shows the ML-based CLV estimate. The formula-based estimate is available "
    "on the Customer 360 page for individual customers."
)

st.divider()

# --- Critical visualization: value vs. risk -----------------------------
st.subheader("Value vs. churn risk")
st.caption(
    "High-value customers with high churn risk (top-right) deserve priority -- "
    "losing them carries greater potential revenue impact than losing a low-value customer."
)
risk_value_scatter(
    df, x_col="churn_probability", y_col="clv_ml",
    x_title="Churn probability", y_title="Predicted CLV (R$)",
    x_threshold=0.6, y_threshold=df["clv_ml"].median(),
    hover_col="customer_unique_id",
    title="Predicted CLV vs. churn risk",
)

st.divider()

# --- Segment value comparison --------------------------------------------
st.subheader("Value by segment")
try:
    segments = get_segments()
    seg_df = pd.DataFrame(segments)
    horizontal_bar_chart(seg_df, "segment_label", "avg_monetary", title="Avg. historical spend by segment (R$)")
except APIClientError:
    st.info("Segment comparison unavailable.")