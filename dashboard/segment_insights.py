"""dashboard/segment_insights.py"""

from formatting import format_currency


# Generates a business interpretation and suggested strategy from a
# segment's actual RFM metrics -- rule-based on the numbers themselves,
# not a hardcoded lookup keyed by segment label, so it stays accurate if
# segment definitions or labels ever change upstream.
def interpret_segment(row, all_segments_df) -> tuple:
    freq = row["avg_frequency"]
    recency = row["avg_recency_days"]
    monetary = row["avg_monetary"]
    avg_monetary_overall = all_segments_df["avg_monetary"].mean()

    frequency_desc = (
        "have purchased only once so far" if freq < 1.3 else f"purchase repeatedly (avg. {freq:.1f} orders)"
    )
    recency_desc = "recently" if recency < 200 else "some time ago, without a recent return"
    if monetary > avg_monetary_overall * 1.3:
        value_desc = "with an above-average order value"
    elif monetary < avg_monetary_overall * 0.8:
        value_desc = "with a below-average order value"
    else:
        value_desc = "with a typical order value"

    interpretation = (
        f"These customers {frequency_desc}, having purchased most {recency_desc} "
        f"({recency:.0f} days on average since their last purchase), {value_desc} "
        f"(avg. {format_currency(monetary)})."
    )

    if freq < 1.3 and recency > 200:
        strategy = (
            "Primary opportunity: **reactivation campaigns** targeting lapsed one-time buyers "
            "to convert them into repeat customers."
        )
    elif freq < 1.3:
        strategy = (
            "Primary opportunity: **second-purchase incentives** while the relationship is "
            "still fresh, before recency lapses further."
        )
    else:
        strategy = (
            "Primary opportunity: **loyalty and retention programs** to protect and extend "
            "an already-repeating relationship."
        )

    if monetary > avg_monetary_overall * 1.3:
        strategy += (
            " Given the above-average order value, **upsell/cross-sell** offers may be more "
            "effective here than pure frequency-based incentives."
        )

    return interpretation, strategy