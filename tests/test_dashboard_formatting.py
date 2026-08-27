"""tests/test_dashboard_formatting.py"""

from dashboard.formatting import currency, human_feature_name, percent, risk_category


def test_currency_formats_with_two_decimals_and_thousands_separator():
    assert currency(1234.5) == "R$ 1,234.50"


def test_percent_formats_fraction_as_percentage():
    assert percent(0.523) == "52.3%"


def test_risk_category_below_low_threshold():
    assert risk_category(0.1) == "Low"


def test_risk_category_between_thresholds():
    assert risk_category(0.45) == "Medium"


def test_risk_category_above_high_threshold():
    assert risk_category(0.8) == "High"


def test_human_feature_name_known_feature():
    assert human_feature_name("avg_freight") == "Avg. freight cost"


def test_human_feature_name_state_feature():
    assert human_feature_name("state_SP") == "Located in SP"


def test_human_feature_name_unknown_feature_falls_back_gracefully():
    assert human_feature_name("some_new_feature") == "Some new feature"