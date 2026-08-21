"""dashboard/app.py"""

import streamlit as st

from api_client import APIClientError, get_forecast_summary, get_segments

st.set_page_config(page_title="Ecommerce BI Dashboard", layout="wide")
st.title("Business Intelligence Overview")

# Segment data is required for this page -- a failure here stops the page
# entirely rather than rendering a half-broken KPI row
try:
    segments = get_segments()
except APIClientError as exc:
    st.error(f"Could not load segment data: {exc.message}")
    st.stop()

total_customers = sum(s["n_customers"] for s in segments)
weighted_monetary = sum(s["avg_monetary"] * s["n_customers"] for s in segments) / total_customers

col1, col2, col3 = st.columns(3)
col1.metric("Total customers", f"{total_customers:,}")
col2.metric("Avg. historical spend", f"R$ {weighted_monetary:,.2f}")
col3.metric("Segments", len(segments))

st.subheader("Market context")

# Forecast is optional (Prophet may not have run, or this signal may be
# temporarily unavailable) -- degrade gracefully rather than break the
# whole KPI page over one missing signal
try:
    forecast = get_forecast_summary()
    fcol1, fcol2 = st.columns(2)
    fcol1.metric("Predicted revenue (next 30d)", f"R$ {forecast['next_30d_predicted_revenue']:,.2f}")
    fcol2.metric("Change vs. prior 30d", f"{forecast['pct_change']:+.1f}%")
    st.caption(f"Trend: {forecast['trend_direction']} -- as of {forecast['last_actual_date']}")
except APIClientError:
    st.info("Sales forecast is currently unavailable.")

st.divider()
st.subheader("Segment breakdown")
st.dataframe(
    [
        {
            "Segment": s["segment_label"],
            "Customers": s["n_customers"],
            "% of base": f"{s['pct_of_customers']:.1f}%",
            "Avg. recency (days)": round(s["avg_recency_days"], 1),
            "Avg. frequency": round(s["avg_frequency"], 2),
            "Avg. spend (R$)": round(s["avg_monetary"], 2),
        }
        for s in segments
    ],
    use_container_width=True,
    hide_index=True,
)
