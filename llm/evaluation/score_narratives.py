"""llm/evaluation/score_narratives.py"""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "reports" / "llm_evaluation" / "sample_narratives_for_review.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "llm_evaluation" / "evaluation_summary.md"

RUBRIC_COLUMNS = [
    "faithfulness_1_5",
    "numerical_accuracy_1_5",
    "clarity_1_5",
    "actionability_1_5",
    "non_causal_language_1_5",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("score_narratives")


# Loads the reviewed CSV and confirms every row has been scored -- catches
# the common mistake of running this before finishing the manual pass
def load_scored_csv() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run `python -m llm.evaluation.sample_narratives` first."
        )
    df = pd.read_csv(INPUT_PATH)

    unscored_mask = df[RUBRIC_COLUMNS].isna().any(axis=1) | (df[RUBRIC_COLUMNS] == "").any(axis=1)
    n_unscored = int(unscored_mask.sum())
    if n_unscored:
        logger.warning(
            "%d of %d rows have at least one blank rubric column -- "
            "these are excluded from the summary. Finish scoring them for a complete result.",
            n_unscored, len(df),
        )
    return df[~unscored_mask].copy()


# Computes mean, min, max per rubric dimension, plus the automated
# faithfulness pass rate for comparison against the manual scores
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in RUBRIC_COLUMNS:
        scores = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "dimension": col.replace("_1_5", "").replace("_", " "),
            "mean": round(scores.mean(), 2),
            "min": int(scores.min()),
            "max": int(scores.max()),
            "n_scored": len(scores),
        })
    return pd.DataFrame(rows)


def run():
    df = load_scored_csv()
    if df.empty:
        logger.error("No fully-scored rows found -- nothing to summarize.")
        return

    summary = summarize(df)
    logger.info("Rubric summary (%d fully-scored narratives):\n%s", len(df), summary.to_string(index=False))

    auto_pass_rate = df["auto_faithfulness_passed"].astype(str).str.lower().eq("true").mean() * 100
    logger.info("Automated faithfulness check pass rate on this sample: %.1f%%", auto_pass_rate)

    low_dims = summary[summary["mean"] < 3.5]
    if not low_dims.empty:
        logger.warning(
            "Dimension(s) scoring below 3.5: %s -- consider revising llm/prompts.py "
            "and re-running the sample before treating narration as validated.",
            ", ".join(low_dims["dimension"]),
        )

    lines = [
        "# LLM Narrative Evaluation Summary",
        "",
        f"Based on {len(df)} manually-scored narratives "
        f"(stratified sample, see `llm/evaluation/sample_narratives.py`).",
        "",
        "| Dimension | Mean | Min | Max | N |",
        "|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row['dimension']} | {row['mean']} | {row['min']} | {row['max']} | {row['n_scored']} |")
    lines += [
        "",
        f"Automated faithfulness check pass rate on this sample: {auto_pass_rate:.1f}%",
        "",
        "See `reports/llm_evaluation_rubric.md` for scoring definitions.",
    ]

    OUTPUT_PATH.write_text("\n".join(lines))
    logger.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    run()
