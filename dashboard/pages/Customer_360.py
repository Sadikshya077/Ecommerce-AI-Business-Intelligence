"""dashboard/pages/Customer_360.py"""

import streamlit as st

from api_client import (
    APIClientError,
    get_churn_risk_leaderboard,
    get_clv_leaderboard,
    get_customer_sample,
    get_insights,
    get_narrative,
)
from chart_utils import shap_diverging_chart
from formatting import format_currency, percent, risk_badge, risk_category

st.set_page_config(page_title="Customer 360", layout="wide", page_icon="\U0001F464")
st.title("Customer 360")
st.caption(
    "The complete analytical story for one customer: segment, churn risk, predicted value, "
    "why the model sees them this way, and what the business could do about it."
)

# --- Customer selection ---------------------------------------------------
# True full-text search across ~95k customers isn't practical in a single
# dropdown, so this browses a merged pool of the most commercially
# interesting customers (top risk, top value, plus a sample) -- Streamlit's
# selectbox filters as you type, giving searchable behavior over that pool.
# A direct text entry is also provided for an exact ID from elsewhere
# (e.g. the "View Customer 360" buttons on Churn Risk, via session_state).
st.subheader("Select a customer")

try:
    browse_pool = get_churn_risk_leaderboard(20) + get_clv_leaderboard(20) + get_customer_sample(20)
    seen, options = set(), []
    for c in browse_pool:
        if c["customer_unique_id"] not in seen:
            seen.add(c["customer_unique_id"])
            options.append(c)
except APIClientError:
    options = []

option_labels = {
    f"{c['customer_unique_id'][:8]}... | {c['segment_label']} | "
    f"{risk_category(c['churn_probability'])} risk | {format_currency(c['clv_ml'])} CLV": c["customer_unique_id"]
    for c in options
}

default_id = st.session_state.get("selected_customer_id", "")

col_a, col_b = st.columns([2, 1])
with col_a:
    chosen_label = st.selectbox(
        "Browse customers (top risk, top value, and a sample)",
        options=["-- select --"] + list(option_labels.keys()),
    )
    selected_from_dropdown = option_labels.get(chosen_label)
with col_b:
    manual_id = st.text_input("Or enter a customer ID directly", value=default_id)

customer_id = manual_id.strip() or selected_from_dropdown

if not customer_id:
    st.info("Select a customer above, or enter a customer ID directly, to view their full profile.")
    st.stop()

# --- Load core analytical data (required -- page stops if this fails) -----
try:
    insights = get_insights(customer_id)
except APIClientError as exc:
    st.error(f"Could not load this customer: {exc.message}")
    st.stop()

st.session_state["selected_customer_id"] = customer_id

# --- Header ----------------------------------------------------------------
st.divider()
h1, h2, h3, h4 = st.columns(4)
h1.metric("Customer ID", f"{customer_id[:12]}...")
h2.metric("Segment", insights["segment_label"])
h3.metric("Risk status", risk_badge(insights["churn_probability"]))
h4.metric("Churn probability", percent(insights["churn_probability"]))

# --- Customer profile --------------------------------------------------
st.divider()
st.subheader("Customer profile")

p1, p2, p3 = st.columns(3)
p1.metric("Historical spend", format_currency(insights["historical_spend"]))
p2.metric("Purchase frequency", f"{insights['frequency']:.1f} orders")
p3.metric("Recency", f"{insights['recency_days']:.0f} days since last purchase")

p4, p5, p6 = st.columns(3)
p4.metric("Avg. order value", format_currency(insights["avg_order_value"]))
p5.metric("Avg. freight cost", format_currency(insights["avg_freight"]))
p6.metric(
    "Avg. delivery time",
    f"{insights['avg_delivery_days']:.0f} days" if insights["avg_delivery_days"] is not None else "Not available",
)

if insights["avg_review_score"] is not None:
    st.metric("Avg. review score", f"{insights['avg_review_score']:.1f} / 5")
else:
    st.caption("No review score on record for this customer.")

# --- Churn analysis ----------------------------------------------------
st.divider()
st.subheader("Churn analysis")
c1, c2 = st.columns(2)
c1.metric("Churn probability", percent(insights["churn_probability"]))
c2.metric("Risk category", risk_category(insights["churn_probability"]))
st.write(
    f"This customer's predicted churn probability is **{percent(insights['churn_probability'])}**, "
    f"placing them in the **{risk_category(insights['churn_probability'])}** risk category "
    f"(dashboard convention: Low <30%, Medium 30-60%, High >60%)."
)

# --- CLV analysis --------------------------------------------------------
st.divider()
st.subheader("Customer lifetime value")
v1, v2, v3 = st.columns(3)
v1.metric("ML-based CLV", format_currency(insights["clv_ml"]))
v2.metric("Formula-based CLV", format_currency(insights["clv_formula"]))
v3.metric("Difference", format_currency(insights["clv_formula"] - insights["clv_ml"]))
st.caption(
    "The ML estimate reflects value predicted from this customer's behavior. The formula-based "
    "estimate projects lifetime value using their churn probability, and is typically higher for "
    "low-risk customers since it assumes a longer future relationship. Neither figure is guaranteed revenue."
)

# --- SHAP explainability --------------------------------------------------
st.divider()
st.subheader("Why the model sees this customer this way")
st.caption(
    "SHAP values show which features contributed most to each model's prediction. They describe "
    "statistical association with the outcome, not proof that a feature caused it."
)
s1, s2 = st.columns(2)
with s1:
    shap_diverging_chart(insights["churn_top_features"], title="Churn prediction -- top contributing features")
with s2:
    shap_diverging_chart(insights["clv_top_features"], title="CLV prediction -- top contributing features")

if insights.get("market_context"):
    mc = insights["market_context"]
    st.caption(
        f"Market context: revenue trend is **{mc['trend_direction']}** "
        f"({mc['pct_change']:+.1f}% next 30 days) -- useful context when interpreting this "
        f"customer's numbers against the broader business."
    )

# --- AI Business Insight ------------------------------------------------
st.divider()
st.subheader("\U0001F916 AI Business Insight")
st.caption(
    "AI-generated interpretation based on the analytical results above. "
    "It cannot invent facts -- it narrates what the models already found."
)

try:
    narrative = get_narrative(customer_id)
    content = narrative["content"]

    st.markdown("**Executive summary**")
    st.write(content["summary"])

    st.markdown("**Why this customer is classified this way**")
    st.write(content["risk_explanation"])

    st.markdown("**Key drivers**")
    for driver in content["key_drivers"]:
        st.markdown(f"- {driver}")

    st.markdown("**Recommended actions**")
    for i, action in enumerate(content["recommended_actions"], start=1):
        st.markdown(f"{i}. {action}")

    with st.expander("Limitations"):
        for limitation in content["limitations"]:
            st.markdown(f"- {limitation}")

except APIClientError:
    # Graceful degradation, as required: the page must not break, and
    # every analytical section above remains fully visible either way.
    st.warning(
        "AI narrative temporarily unavailable. The underlying analytical results "
        "above are still accurate and available."
    )