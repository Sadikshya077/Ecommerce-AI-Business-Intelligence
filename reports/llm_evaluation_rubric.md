# LLM Narrative Manual Evaluation Rubric

Companion to `llm/evaluation/sample_narratives.py`, which generates a stratified sample of ~28 real customer narratives (7 per segment x 4 segments, spanning each segment's churn-risk range) for manual review, saved to `reports/llm_evaluation/sample_narratives_for_review.csv`.

## Why manual review, in addition to the automated check

`llm/faithfulness.py`'s automated check (the `auto_faithfulness_passed` / `auto_faithfulness_notes` columns already filled in on the CSV) is a lightweight, mechanical check: does a supplied SHAP feature name appear somewhere in the narrative, and does an obviously causal phrase appear. It cannot verify deeper faithfulness -- whether stated numbers are correct, whether the reasoning connecting claims is sound, or whether the tone suits a business reader. This is exactly why it's called a lightweight consistency check, not proof of complete faithfulness. Manual review covers what it can't.

## How to score

For each row, read `summary`, `risk_explanation`, `key_drivers`, `recommended_actions`, and `limitations` alongside that customer's actual data (`segment_label`, `churn_probability`, `clv_ml`), then fill in the five empty score columns using the scale below.

### Faithfulness (1-5)
Does the narrative only reference facts and features actually present in the underlying data for this customer?
- **5** -- every claim is traceable to supplied data, no unsupported assertions
- **3** -- mostly grounded, one minor unsupported inference
- **1** -- contains a fabricated number, feature, or fact not present in the context

### Numerical accuracy (1-5)
Where the narrative states or implies a number (churn probability, CLV, a SHAP value), does it match the supplied data?
- **5** -- all numeric claims correct
- **3** -- directionally correct but imprecise or oddly rounded
- **1** -- a stated number contradicts the supplied data

### Clarity (1-5)
Would a non-technical business reader (e.g. a marketing manager) understand this without further explanation?
- **5** -- clear, plain language, no jargon
- **3** -- understandable but includes some technical phrasing
- **1** -- confusing, overly technical, or ambiguous

### Actionability (1-5)
Are the recommended actions specific enough to actually act on, tailored to this customer?
- **5** -- specific, tailored to this customer's actual drivers
- **3** -- reasonable but generic (would apply to almost any customer)
- **1** -- vague filler ("monitor the customer", "reach out")

### Non-causal language (1-5)
Does the narrative correctly treat SHAP feature attributions as correlational, not causal?
- **5** -- consistently correlational phrasing ("was a factor contributing to")
- **3** -- mostly correct, one borderline phrase
- **1** -- repeatedly states a feature "caused" or "led to" the outcome

## After scoring

Compute the mean score per dimension across all rows. This becomes the headline result for your report's LLM narration evaluation section, alongside the automated faithfulness pass rate already in the CSV. A mean below roughly 3.5 on any dimension is worth investigating -- revise `llm/prompts.py` and re-run the sample -- before treating the narration layer as validated.
