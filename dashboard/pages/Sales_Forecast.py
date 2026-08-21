"""dashboard/pages/Sales_Forecast.py"""

import pandas as pd
import streamlit as st

from api_client import APIClientError, get_forecast_series, get_forecast_summary

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("Sales Forecast")

try:
    summary = get_forecast_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Prior 30d actual revenue", f"R$ {summary['prior_30d_actual_revenue']:,.2f}")
    col2.metric("Next 30d predicted revenue", f"R$ {summary['next_30d_predicted_revenue']:,.2f}")
    col3.metric("Change", f"{summary['pct_change']:+.1f}%")
    st.caption(f"Trend: {summary['trend_direction']} -- last actual data point: {summary['last_actual_date']}")
except APIClientError as exc:
    st.warning(f"Forecast summary unavailable: {exc.message}")

st.divider()

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
df = df.set_index("ds")

st.subheader("Predicted daily revenue")
st.line_chart(df[["yhat_lower", "yhat", "yhat_upper"]])
st.caption("yhat = predicted revenue, with lower/upper confidence bounds")
