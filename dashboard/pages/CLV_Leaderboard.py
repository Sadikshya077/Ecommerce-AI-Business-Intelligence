"""dashboard/pages/CLV_Leaderboard.py"""

import pandas as pd
import streamlit as st

from api_client import APIClientError, get_clv_leaderboard
from chart_utils import wrapped_bar_chart

st.set_page_config(page_title="CLV Leaderboard", layout="wide")
st.title("Highest Customer Lifetime Value")

n = st.slider("Number of customers to show", min_value=5, max_value=50, value=20, step=5)

try:
    leaderboard = get_clv_leaderboard(n)
except APIClientError as exc:
    st.error(f"Could not load CLV data: {exc.message}")
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
        "clv_ml": "Predicted CLV (R$)",
        "churn_probability_pct": "Churn risk (%)",
    })[["Customer ID", "Segment", "Predicted CLV (R$)", "Churn risk (%)"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Segment composition of highest-value customers")
segment_counts = df["segment_label"].value_counts().reset_index()
segment_counts.columns = ["segment_label", "count"]
wrapped_bar_chart(segment_counts, "segment_label", "count", y_axis_title="Customers")
st.caption(f"Segment breakdown of the {len(df)} highest-CLV customers shown above")

at_risk = df[df["churn_probability"] > 0.5]
if not at_risk.empty:
    st.warning(
        f"{len(at_risk)} of these high-value customers also have churn risk above 50% -- "
        f"worth prioritizing for retention."
    )