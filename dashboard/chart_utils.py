"""dashboard/chart_utils.py"""

import textwrap

import plotly.express as px
import streamlit as st


# Wraps long category labels onto multiple lines instead of letting the
# default renderer rotate them vertically or cut them off. Segment names
# like "Lapsed one-time buyers" stay horizontal and fully readable this
# way, matching standard professional chart conventions.
def wrapped_bar_chart(df, category_col: str, value_col: str, y_axis_title: str = None, wrap_width: int = 14):
    labels = ["<br>".join(textwrap.wrap(str(v), wrap_width)) for v in df[category_col]]

    fig = px.bar(x=labels, y=df[value_col])
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=y_axis_title or value_col,
        xaxis=dict(tickangle=0),
        margin=dict(b=90),
    )
    st.plotly_chart(fig, use_container_width=True)
