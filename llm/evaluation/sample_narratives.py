"""llm/evaluation/sample_narratives.py"""

import logging
import time
from pathlib import Path

import pandas as pd

from api.data_store import store
from api.schemas.insights import CustomerInsights
from api.services.insight_service import build_customer_insights
from llm.narrator import generate_narrative

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "llm_evaluation"
OUTPUT_PATH = OUTPUT_DIR / "sample_narratives_for_review.csv"

# 7 per segment x 4 segments = 28, within the 20-30 sample the evaluation
# rubric calls for. NOTE: Gemini's free tier caps gemini-3.6-flash at 20
# requests per day per project -- a full 28-call run will not complete in
# a single calendar day on the free tier. Resume support below (skipping
# already-completed customers on rerun) is how this is meant to be spread
# across two days rather than solved by raising retries.
SAMPLE_PER_SEGMENT = 7
DELAY_BETWEEN_CALLS_SECONDS = 1.0
# After this many consecutive failures, stop the batch rather than grind
# through the rest of the sample retrying a wall that won't come down
# until the daily quota resets -- each failure here already cost 3 retries
# with backoff inside call_llm, so consecutive failures burn real time for
# zero chance of success once a daily cap is hit.
CONSECUTIVE_FAILURES_TO_ABORT = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("sample_narratives")


# Pure function, no I/O -- selects `per_segment` customers per segment,
# evenly spaced across that segment's churn_probability range rather than
# randomly, so the sample deliberately covers low/medium/high risk instead
# of clustering around whatever's most common in that segment.
def stratified_sample(df: pd.DataFrame, per_segment: int) -> pd.DataFrame:
    samples = []
    for segment_id, group in df.groupby("segment_id"):
        sorted_group = group.sort_values("churn_probability").reset_index(drop=True)
        n = len(sorted_group)
        k = min(per_segment, n)
        if k <= 1:
            indices = [0] if n else []
        else:
            indices = sorted(set(round(i * (n - 1) / (k - 1)) for i in range(k)))
        samples.append(sorted_group.iloc[indices])
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


# Rows already successfully narrated in a prior run -- these are skipped
# on rerun so a quota-interrupted batch can resume without re-spending
# quota on customers that already completed
def load_already_done() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=["customer_unique_id"])
    existing = pd.read_csv(OUTPUT_PATH)
    return existing[existing["summary"] != "GENERATION FAILED"]


def _row_for(customer_id: str, meta: pd.Series, result) -> dict:
    if result is None:
        return {
            "customer_unique_id": customer_id,
            "segment_label": meta["segment_label"],
            "churn_probability": meta["churn_probability"],
            "clv_ml": meta["clv_ml"],
            "summary": "GENERATION FAILED", "risk_explanation": "", "key_drivers": "",
            "recommended_actions": "", "limitations": "",
            "auto_faithfulness_passed": False,
            "auto_faithfulness_notes": "narration returned None -- see server log",
            "faithfulness_1_5": "", "numerical_accuracy_1_5": "", "clarity_1_5": "",
            "actionability_1_5": "", "non_causal_language_1_5": "", "reviewer_notes": "",
        }
    return {
        "customer_unique_id": customer_id,
        "segment_label": meta["segment_label"],
        "churn_probability": meta["churn_probability"],
        "clv_ml": meta["clv_ml"],
        "summary": result.content.summary,
        "risk_explanation": result.content.risk_explanation,
        "key_drivers": "; ".join(result.content.key_drivers),
        "recommended_actions": "; ".join(result.content.recommended_actions),
        "limitations": "; ".join(result.content.limitations),
        "auto_faithfulness_passed": result.faithfulness_passed,
        "auto_faithfulness_notes": "; ".join(result.faithfulness_notes),
        # Left blank for manual scoring -- see reports/llm_evaluation_rubric.md
        "faithfulness_1_5": "", "numerical_accuracy_1_5": "", "clarity_1_5": "",
        "actionability_1_5": "", "non_causal_language_1_5": "", "reviewer_notes": "",
    }


def run():
    store.load()
    logger.info("Loaded %d customers", len(store))

    sample_df = stratified_sample(store.all_customers(), SAMPLE_PER_SEGMENT)

    already_done = load_already_done()
    done_ids = set(already_done["customer_unique_id"])
    remaining = sample_df[~sample_df["customer_unique_id"].isin(done_ids)]

    logger.info(
        "%d of %d customers already completed in a prior run -- %d remaining this run",
        len(done_ids), len(sample_df), len(remaining),
    )

    new_rows = []
    consecutive_failures = 0
    aborted = False

    for i, meta in remaining.iterrows():
        customer_id = meta["customer_unique_id"]
        logger.info(
            "Generating narrative %d/%d remaining for %s...",
            len(new_rows) + 1, len(remaining), customer_id,
        )

        context = build_customer_insights(customer_id)
        validated_context = CustomerInsights(**context).model_dump(mode="json")
        result = generate_narrative(validated_context)

        if result is None:
            consecutive_failures += 1
            logger.warning(
                "Narration failed for %s (%d consecutive failure(s))",
                customer_id, consecutive_failures,
            )
        else:
            consecutive_failures = 0

        new_rows.append(_row_for(customer_id, meta, result))

        if consecutive_failures >= CONSECUTIVE_FAILURES_TO_ABORT:
            logger.error(
                "%d consecutive failures -- stopping early rather than retrying a wall "
                "that won't come down today. This is almost always a daily quota limit, "
                "not a transient issue: rerun this script tomorrow to resume -- already-"
                "completed customers will be skipped automatically.",
                consecutive_failures,
            )
            aborted = True
            break

        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    combined_df = pd.concat([already_done, pd.DataFrame(new_rows)], ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)

    n_done = len(combined_df[combined_df["summary"] != "GENERATION FAILED"])
    auto_pass_rate = combined_df["auto_faithfulness_passed"].mean() * 100
    logger.info(
        "Wrote %s -- %d/%d customers completed, automated faithfulness check passed for %.1f%%",
        OUTPUT_PATH, n_done, len(sample_df), auto_pass_rate,
    )

    if aborted or n_done < len(sample_df):
        logger.info("Sample incomplete -- rerun this script (later today or tomorrow) to fill in the rest.")
    else:
        logger.info(
            "Sample complete. Next step: open the CSV and manually score each row 1-5 "
            "on the five rubric dimensions -- see reports/llm_evaluation_rubric.md for definitions."
        )


if __name__ == "__main__":
    run()