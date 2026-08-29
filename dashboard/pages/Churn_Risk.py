"""dashboard/pages/Churn_Risk.py"""

import pandas as pd
import streamlit as st

from api_client import APIClientError, get_churn_risk_leaderboard, get_customer_kpis
from chart_utils import horizontal_bar_chart
from formatting import RISK_THRESHOLDS, currency, percent, risk_badge, risk_category

st.set_page_config(page_title="Churn Risk", layout="wide", page_icon="\u26A0\ufe0f")
st.title("Churn Risk")
st.caption(
    "Churn probability is the model's estimated likelihood that a customer will become inactive "
    "within the defined churn window. It is a prediction, not a certainty."
)

n = st.slider("Number of customers to analyze", min_value=10, max_value=100, value=30, step=10)

try:
    leaderboard = get_churn_risk_leaderboard(n)
    kpis = get_customer_kpis()
except APIClientError as exc:
    st.error(f"Could not load churn risk data: {exc.message}")
    st.stop()

if not leaderboard:
    st.info("No customer data available.")
    st.stop()

df = pd.DataFrame(leaderboard)
df["risk_category"] = df["churn_probability"].apply(risk_category)

# --- KPI cards --------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("High-risk customers (platform-wide)", f"{kpis['high_risk_count']:,}")
col2.metric("Avg. churn probability (platform-wide)", percent(kpis["avg_churn_probability"]))
col3.metric("Highest churn probability", percent(df["churn_probability"].max()))
high_risk_value = df.loc[df["risk_category"] == "High", "clv_ml"].sum()
col4.metric("Value of high-risk customers shown", currency(high_risk_value))

st.caption(
    f"Risk tiers (dashboard convention, not a model output): "
    f"Low (<{percent(RISK_THRESHOLDS['low'])}), "
    f"Medium ({percent(RISK_THRESHOLDS['low'])}-{percent(RISK_THRESHOLDS['medium'])}), "
    f"High (>{percent(RISK_THRESHOLDS['medium'])})."
)

st.divider()

# --- Risk distribution --------------------------------------------------------
st.subheader("Risk distribution")
st.caption("How many of the customers analyzed above fall into each risk tier.")
dist = df["risk_category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
dist.columns = ["risk_category", "count"]
horizontal_bar_chart(dist, "risk_category", "count", title="Customers by risk tier")

st.divider()

# --- Leaderboard --------------------------------------------------------------
st.subheader("Highest churn risk customers")
display_df = df.copy()
display_df["Churn risk"] = display_df["churn_probability"].apply(percent)
display_df["Predicted CLV"] = display_df["clv_ml"].apply(currency)
display_df["Risk"] = display_df["churn_probability"].apply(risk_badge)

st.dataframe(
    display_df.rename(columns={"customer_unique_id": "Customer ID", "segment_label": "Segment"})[
        ["Customer ID", "Segment", "Churn risk", "Predicted CLV", "Risk"]
    ],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --- Priority retention targets --------------------------------------------
st.subheader("Priority retention targets")
st.caption(
    "Customers who are simultaneously high churn risk and high predicted value -- "
    "the most commercially important group to act on."
)

priority = df[
    (df["risk_category"] == "High") & (df["clv_ml"] > df["clv_ml"].median())
].sort_values("clv_ml", ascending=False)

if priority.empty:
    st.info("No customers in this list are both high-risk and above-median value.")
else:
    for _, row in priority.head(10).iterrows():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            c1.write(f"**{row['customer_unique_id']}**")
            c2.write(row["segment_label"])
            c3.write(f"Risk: {percent(row['churn_probability'])}")
            c4.write(f"CLV: {currency(row['clv_ml'])}")
            # Wired for Customer 360 -- functional once that page is added
            if st.button("View Customer 360 \u2192", key=f"churn_view_{row['customer_unique_id']}"):
                st.session_state["selected_customer_id"] = row["customer_unique_id"]
                st.switch_page("pages/5_Customer_360.py")

st.divider()

# --- Segment risk concentration ------------------------------------------
st.subheader("Which segments carry the most churn risk?")
st.caption("Average churn probability among the customers analyzed above, grouped by segment.")
seg_risk = df.groupby("segment_label")["churn_probability"].mean().reset_index()
seg_risk.columns = ["segment_label", "avg_churn_probability"]
horizontal_bar_chart(seg_risk, "segment_label", "avg_churn_probability", title="Avg. churn probability by segment")