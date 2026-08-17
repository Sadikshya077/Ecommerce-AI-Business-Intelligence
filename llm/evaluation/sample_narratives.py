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

# 7 per segment x 4 segments = 28, within the 20-30 sample the evaluation
# rubric calls for
SAMPLE_PER_SEGMENT = 7
# Delay between sequential API calls -- avoids tripping rate limits over
# a burst of ~28 requests. This script makes REAL, BILLED API calls.
DELAY_BETWEEN_CALLS_SECONDS = 1.0

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
    logger.info(
        "Selected %d customers for manual evaluation (stratified by segment and churn risk)",
        len(sample_df),
    )

    rows = []
    for i, meta in sample_df.iterrows():
        customer_id = meta["customer_unique_id"]
        logger.info("Generating narrative %d/%d for %s...", i + 1, len(sample_df), customer_id)

        context = build_customer_insights(customer_id)
        validated_context = CustomerInsights(**context).model_dump(mode="json")
        result = generate_narrative(validated_context)
        if result is None:
            logger.warning("Narration failed for %s -- recorded as failure", customer_id)
        rows.append(_row_for(customer_id, meta, result))

        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    output_df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sample_narratives_for_review.csv"
    output_df.to_csv(out_path, index=False)

    auto_pass_rate = output_df["auto_faithfulness_passed"].mean() * 100
    logger.info(
        "Wrote %s -- %d narratives, automated faithfulness check passed for %.1f%%",
        out_path, len(output_df), auto_pass_rate,
    )
    logger.info(
        "Next step: open the CSV and manually score each row 1-5 on the five "
        "rubric dimensions -- see reports/llm_evaluation_rubric.md for definitions."
    )


if __name__ == "__main__":
    run()
