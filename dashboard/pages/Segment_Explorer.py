"""dashboard/pages/Segment_Explorer.py"""

import pandas as pd
import plotly.express as px
import streamlit as st
 
from api_client import APIClientError, get_segments
from chart_utils import horizontal_bar_chart
from formatting import format_currency
from segment_insights import interpret_segment
 
st.set_page_config(page_title="Segment Explorer", layout="wide", page_icon="\U0001F465")
st.title("Segment Explorer")
st.caption(
    "Customer segmentation groups customers with similar purchasing behavior "
    "so the business can use different strategies for different groups."
)
 
try:
    segments = get_segments()
except APIClientError as exc:
    st.error(f"Could not load segment data: {exc.message}")
    st.stop()
 
df = pd.DataFrame(segments)
 
# --- Composition ------------------------------------------------------------
st.subheader("Customer base composition")
col1, col2 = st.columns(2)
with col1:
    fig = px.pie(df, names="segment_label", values="n_customers", hole=0.45, title="Share of customers by segment")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    horizontal_bar_chart(df, "segment_label", "avg_monetary", title="Avg. historical spend by segment (R$)")
 
st.divider()
 
# --- Comparison table ---------------------------------------------------
st.subheader("Segment comparison")
comparison = df.copy()
comparison["Customers"] = comparison["n_customers"].apply(lambda x: f"{x:,}")
comparison["% of base"] = comparison["pct_of_customers"].apply(lambda x: f"{x:.1f}%")
comparison["Avg. recency (days)"] = comparison["avg_recency_days"].round(1)
comparison["Avg. frequency"] = comparison["avg_frequency"].round(2)
comparison["Avg. spend"] = comparison["avg_monetary"].apply(format_currency)
st.dataframe(
    comparison.rename(columns={"segment_label": "Segment"})[
        ["Segment", "Customers", "% of base", "Avg. recency (days)", "Avg. frequency", "Avg. spend"]
    ],
    use_container_width=True,
    hide_index=True,
)
 
st.divider()
 
# --- Segment deep-dive ----------------------------------------------------
st.subheader("Segment deep-dive")
selected_label = st.selectbox("Select a segment", df["segment_label"])
row = df[df["segment_label"] == selected_label].iloc[0]
 
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{int(row['n_customers']):,}")
k2.metric("% of base", f"{row['pct_of_customers']:.1f}%")
k3.metric("Avg. recency", f"{row['avg_recency_days']:.0f} days")
k4.metric("Avg. spend", format_currency(row["avg_monetary"]))
 
interpretation, strategy = interpret_segment(row, df)
st.markdown("**Business interpretation**")
st.write(interpretation)
st.markdown("**Suggested business strategy**")
st.info(strategy)
 