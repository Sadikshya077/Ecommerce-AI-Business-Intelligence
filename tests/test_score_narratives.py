"""tests/test_score_narratives.py"""

import pandas as pd

from llm.evaluation.score_narratives import RUBRIC_COLUMNS, summarize


def _fake_scored_df():
    return pd.DataFrame([
        {"faithfulness_1_5": 5, "numerical_accuracy_1_5": 4, "clarity_1_5": 5,
         "actionability_1_5": 3, "non_causal_language_1_5": 5},
        {"faithfulness_1_5": 4, "numerical_accuracy_1_5": 4, "clarity_1_5": 4,
         "actionability_1_5": 2, "non_causal_language_1_5": 5},
    ])


def test_summarize_computes_mean_per_dimension():
    df = _fake_scored_df()
    summary = summarize(df)
    assert len(summary) == len(RUBRIC_COLUMNS)
    faithfulness_row = summary[summary["dimension"] == "faithfulness"].iloc[0]
    assert faithfulness_row["mean"] == 4.5


def test_summarize_flags_low_actionability():
    df = _fake_scored_df()
    summary = summarize(df)
    actionability_row = summary[summary["dimension"] == "actionability"].iloc[0]
    assert actionability_row["mean"] == 2.5
    assert actionability_row["mean"] < 3.5


def test_summarize_reports_min_max_and_count():
    df = _fake_scored_df()
    summary = summarize(df)
    clarity_row = summary[summary["dimension"] == "clarity"].iloc[0]
    assert clarity_row["min"] == 4
    assert clarity_row["max"] == 5
    assert clarity_row["n_scored"] == 2
