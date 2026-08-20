"""dashboard/pages/Churn_Risk.py"""

import pandas as pd
import streamlit as st

from api_client import APIClientError, get_churn_risk_leaderboard
from chart_utils import wrapped_bar_chart

st.set_page_config(page_title="Churn Risk", layout="wide")
st.title("Highest Churn Risk Customers")

n = st.slider("Number of customers to show", min_value=5, max_value=50, value=20, step=5)

try:
    leaderboard = get_churn_risk_leaderboard(n)
except APIClientError as exc:
    st.error(f"Could not load churn risk data: {exc.message}")
    st.stop()

if not leaderboard:
    st.info("No customer data available.")
    st.stop()

df = pd.DataFrame(leaderboard)
df["churn_probability_pct"] = (df["churn_probability"] * 100).round(1)

st.dataframe(
    df.rename(columns={
        "customer_unique_id": "Customer ID",
        "segment_label": "Segment",
        "churn_probability_pct": "Churn risk (%)",
        "clv_ml": "Predicted CLV (R$)",
    })[["Customer ID", "Segment", "Churn risk (%)", "Predicted CLV (R$)"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Segment composition of high-risk customers")
segment_counts = df["segment_label"].value_counts().reset_index()
segment_counts.columns = ["segment_label", "count"]
wrapped_bar_chart(segment_counts, "segment_label", "count", y_axis_title="Customers")
st.caption(f"Segment breakdown of the {len(df)} highest churn-risk customers shown above")