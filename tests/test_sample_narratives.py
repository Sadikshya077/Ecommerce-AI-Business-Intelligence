"""tests/test_sample_narratives.py"""

import pandas as pd

from llm.evaluation.sample_narratives import stratified_sample


def _fake_population(n_per_segment: dict) -> pd.DataFrame:
    rows = []
    for segment_id, n in n_per_segment.items():
        for i in range(n):
            rows.append({
                "customer_unique_id": f"seg{segment_id}_cust{i}",
                "segment_id": segment_id,
                "churn_probability": i / max(n - 1, 1),
            })
    return pd.DataFrame(rows)


def test_stratified_sample_covers_every_segment():
    df = _fake_population({0: 50, 1: 50, 2: 10, 3: 5})
    sample = stratified_sample(df, per_segment=7)
    assert set(sample["segment_id"].unique()) == {0, 1, 2, 3}


def test_stratified_sample_respects_small_segments():
    # Segment 3 only has 5 customers -- asking for 7 should return all 5,
    # not error or duplicate rows
    df = _fake_population({0: 50, 3: 5})
    sample = stratified_sample(df, per_segment=7)
    seg3 = sample[sample["segment_id"] == 3]
    assert len(seg3) == 5
    assert seg3["customer_unique_id"].nunique() == 5


def test_stratified_sample_spans_the_risk_distribution():
    # Within a segment, the sample should include both low and high
    # churn_probability values, not cluster in the middle
    df = _fake_population({0: 100})
    sample = stratified_sample(df, per_segment=5)
    seg0 = sample[sample["segment_id"] == 0]
    assert seg0["churn_probability"].min() < 0.1
    assert seg0["churn_probability"].max() > 0.9


def test_stratified_sample_total_size_matches_expectation():
    df = _fake_population({0: 50, 1: 50, 2: 50, 3: 50})
    sample = stratified_sample(df, per_segment=7)
    assert len(sample) == 28