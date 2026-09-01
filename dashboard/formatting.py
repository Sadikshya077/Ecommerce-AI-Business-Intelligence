"""dashboard/formatting.py"""

# Shared formatting so every page presents numbers consistently -- no page
# should ever show a raw float like 0.5272793769836426 to a user.


def format_currency(value) -> str:
    return f"R$ {value:,.2f}"


def percent(value_0_to_1) -> str:
    return f"{value_0_to_1 * 100:.1f}%"


# No "high risk" threshold exists anywhere in the backend -- churn_probability
# is a continuous model output, never categorized upstream. These tiers are
# a dashboard-only UI convention, defined once here for consistent use
# across every page, not a backend or model definition.
RISK_THRESHOLDS = {"low": 0.3, "medium": 0.6}


def risk_category(churn_probability: float) -> str:
    if churn_probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if churn_probability < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def risk_badge(churn_probability: float) -> str:
    category = risk_category(churn_probability)
    icons = {"Low": "\U0001F7E2", "Medium": "\U0001F7E1", "High": "\U0001F534"}
    return f"{icons[category]} {category}"


# Maps raw snake_case feature names to short, human-readable labels for
# charts and narrative display. Falls back to a generic conversion for any
# feature not explicitly listed, so new features never render as blank.
FEATURE_LABELS = {
    "avg_delivery_days": "Avg. delivery time (days)",
    "avg_delivery_delay": "Avg. delivery delay (days)",
    "avg_freight": "Avg. freight cost",
    "avg_order_value": "Avg. order value",
    "avg_installments": "Avg. payment installments",
    "avg_review_score": "Avg. review score",
    "frequency": "Purchase frequency",
    "monetary": "Total historical spend",
    "customer_tenure_days": "Customer tenure (days)",
}


def human_feature_name(raw_name: str) -> str:
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    if raw_name.startswith("state_"):
        return f"Located in {raw_name.replace('state_', '')}"
    return raw_name.replace("_", " ").capitalize()