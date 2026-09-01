"""dashboard/pages/Sales_Forecast.py"""

import pandas as pd
import streamlit as st

from api_client import APIClientError, get_forecast_series, get_forecast_summary
from chart_utils import forecast_chart
from formatting import format_currency

st.set_page_config(page_title="Sales Forecast", layout="wide", page_icon="\U0001F4C8")
st.title("Sales Forecast")
st.caption(
    "The forecast estimates future revenue based on historical sales patterns. "
    "The shaded region represents forecast uncertainty, not a guaranteed outcome."
)

try:
    summary = get_forecast_summary()
except APIClientError as exc:
    st.error(f"Could not load forecast summary: {exc.message}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prior 30-day revenue", format_currency(summary["prior_30d_actual_revenue"]))
col2.metric("Next 30-day predicted revenue", format_currency(summary["next_30d_predicted_revenue"]))
col3.metric("Expected change", f"{summary['pct_change']:+.1f}%")
col4.metric("Trend", summary["trend_direction"].capitalize())

st.divider()

st.subheader("Predicted daily revenue")

try:
    series = get_forecast_series()
except APIClientError as exc:
    st.error(f"Could not load forecast series: {exc.message}")
    st.stop()

if not series:
    st.info("No forecast data available.")
    st.stop()

df = pd.DataFrame(series)
df["ds"] = pd.to_datetime(df["ds"])
forecast_chart(
    df, date_col="ds", predicted_col="yhat", lower_col="yhat_lower", upper_col="yhat_upper",
    title="Daily revenue forecast with uncertainty band",
)

st.divider()

st.subheader("What this means")
trend = summary["trend_direction"]
pct = summary["pct_change"]

if trend == "up":
    implication = (
        f"Revenue is trending **up** ({pct:+.1f}%), suggesting current demand and retention efforts "
        f"are translating into growth. Worth ensuring inventory and fulfillment capacity can support it."
    )
elif trend == "down":
    implication = (
        f"Revenue is trending **down** ({pct:+.1f}%) -- worth checking whether this aligns with seasonal "
        f"patterns or reflects a genuine decline, alongside the Churn Risk page."
    )
else:
    implication = (
        f"Revenue is expected to stay relatively **flat** ({pct:+.1f}%) over the next 30 days -- "
        f"no major shift in either direction is currently predicted."
    )
st.markdown(implication)