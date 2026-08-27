"""dashboard/chart_utils.py"""

import textwrap

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from formatting import human_feature_name


# Kept for pages not yet redesigned -- vertical bar with wrapped labels.
# Prefer horizontal_bar_chart() for anything new; it reads better for long
# category names and rankings.
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


# Horizontal bar chart -- the right choice for rankings/leaderboards and
# for long category labels: reads top-to-bottom, never needs rotated or
# wrapped axis text regardless of label length.
def horizontal_bar_chart(df, category_col: str, value_col: str, title: str = None):
    fig = px.bar(df, x=value_col, y=category_col, orientation="h", title=title)
    fig.update_layout(yaxis=dict(autorange="reversed"), yaxis_title=None, xaxis_title=value_col)
    st.plotly_chart(fig, use_container_width=True)


# Scatter plot for two-metric relationships (e.g. churn risk vs. CLV).
# Optional threshold lines mark quadrant boundaries -- e.g. a vertical
# line at the "high risk" cutoff and a horizontal line at median value,
# splitting the chart into the four business-meaningful quadrants.
def risk_value_scatter(df, x_col: str, y_col: str, x_title: str, y_title: str,
                        x_threshold: float = None, y_threshold: float = None,
                        hover_col: str = None, title: str = None):
    fig = px.scatter(df, x=x_col, y=y_col, hover_name=hover_col, title=title, opacity=0.6)
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

    if x_threshold is not None:
        fig.add_vline(x=x_threshold, line_dash="dash", line_color="gray")
    if y_threshold is not None:
        fig.add_hline(y=y_threshold, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)


# Diverging horizontal bar for SHAP feature contributions -- features
# pushing the prediction up vs. down sit on opposite sides of zero, with
# human-readable labels (via formatting.human_feature_name) instead of
# raw snake_case feature names.
def shap_diverging_chart(features: list, title: str = None):
    labels = [human_feature_name(f["feature"]) for f in features]
    values = [f["shap_value"] for f in features]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.2f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="SHAP contribution", yaxis_title=None,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# Forecast chart with a shaded uncertainty band rather than three
# unrelated lines -- predicted revenue is a single visually distinct
# line, with the lower/upper bound rendered as a filled region behind it.
def forecast_chart(df, date_col: str, predicted_col: str, lower_col: str, upper_col: str, title: str = None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(df[date_col]) + list(df[date_col])[::-1],
        y=list(df[upper_col]) + list(df[lower_col])[::-1],
        fill="toself", fillcolor="rgba(31,119,180,0.15)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        name="Forecast uncertainty",
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[predicted_col], mode="lines",
        line=dict(color="#1f77b4", width=2), name="Predicted revenue",
    ))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Revenue (R$)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)