"""dashboard/pages/Segment_Explorer.py"""

import pandas as pd
import streamlit as st

from api_client import APIClientError, get_segments
from chart_utils import wrapped_bar_chart

st.set_page_config(page_title="Segment Explorer", layout="wide")
st.title("Segment Explorer")

try:
    segments = get_segments()
except APIClientError as exc:
    st.error(f"Could not load segment data: {exc.message}")
    st.stop()

df = pd.DataFrame(segments)

wrapped_bar_chart(df, "segment_label", "n_customers", y_axis_title="Customers")
st.caption("Customers per segment")

col1, col2 = st.columns(2)
with col1:
    wrapped_bar_chart(df, "segment_label", "avg_monetary", y_axis_title="Avg. spend (R$)")
    st.caption("Average historical spend per segment (R$)")
with col2:
    wrapped_bar_chart(df, "segment_label", "avg_recency_days", y_axis_title="Avg. recency (days)")
    st.caption("Average recency per segment (days)")

st.divider()
selected = st.selectbox("View segment details", df["segment_label"])
row = df[df["segment_label"] == selected].iloc[0]
st.write(f"**{selected}** -- {row['n_customers']:,} customers ({row['pct_of_customers']:.1f}% of base)")
st.write(
    f"Avg. recency: {row['avg_recency_days']:.1f} days | "
    f"Avg. frequency: {row['avg_frequency']:.2f} | "
    f"Avg. spend: R$ {row['avg_monetary']:.2f}"
)