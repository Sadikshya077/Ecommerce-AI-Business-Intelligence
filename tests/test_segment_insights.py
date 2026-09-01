"""tests/test_segment_insights.py"""

import pandas as pd

from dashboard.segment_insights import interpret_segment


def _segments_df():
    return pd.DataFrame([
        {"segment_label": "Lapsed", "avg_frequency": 1.0, "avg_recency_days": 390.0, "avg_monetary": 134.0},
        {"segment_label": "Recent", "avg_frequency": 1.0, "avg_recency_days": 130.0, "avg_monetary": 135.0},
        {"segment_label": "Loyal", "avg_frequency": 2.1, "avg_recency_days": 220.0, "avg_monetary": 290.0},
        {"segment_label": "HighValue", "avg_frequency": 1.0, "avg_recency_days": 240.0, "avg_monetary": 1170.0},
    ])


def test_lapsed_segment_suggests_reactivation():
    df = _segments_df()
    _, strategy = interpret_segment(df.iloc[0], df)
    assert "reactivation" in strategy.lower()


def test_recent_one_time_suggests_second_purchase_incentive():
    df = _segments_df()
    _, strategy = interpret_segment(df.iloc[1], df)
    assert "second-purchase" in strategy.lower()


def test_loyal_segment_suggests_retention_program():
    df = _segments_df()
    _, strategy = interpret_segment(df.iloc[2], df)
    assert "loyalty" in strategy.lower() or "retention" in strategy.lower()


def test_high_value_segment_gets_upsell_note():
    df = _segments_df()
    _, strategy = interpret_segment(df.iloc[3], df)
    assert "upsell" in strategy.lower()
