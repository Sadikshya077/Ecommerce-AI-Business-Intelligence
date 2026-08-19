"""llm/faithfulness.py"""

# Lightweight consistency check ONLY: confirms the narrative's claims are
# traceable to the supplied context (known SHAP feature names appear
# somewhere, no obvious causal-language violations). This does NOT prove
# the narrative is fully faithful in a deeper sense -- it's string/keyword
# matching against known-good source data, not semantic verification.
# A fuller faithfulness claim requires manual rubric-based review of a
# sample of narratives (Phase 7 Part 3). Never cite this function alone as
# proof of complete faithfulness.
CAUSAL_PHRASES = ["caused", "causes", "causing", "because it led to", "resulted in the customer"]


def check_faithfulness(content, insight_context: dict) -> tuple:
    notes = []

    known_features = {f["feature"] for f in insight_context.get("churn_top_features", [])}
    known_features |= {f["feature"] for f in insight_context.get("clv_top_features", [])}

    # Search the full narrative, not just key_drivers. The prompt asks the
    # model to ground risk_explanation directly in the supplied SHAP
    # features, while key_drivers is often a paraphrased summary (e.g.
    # "avg_freight" becomes "average freight fees" -- good for
    # readability, bad for literal string matching). Checking key_drivers
    # alone produced false negatives on narratives that were, in fact,
    # correctly grounded: the literal feature name was just present
    # elsewhere in the same narrative instead.
    full_text = " ".join([content.summary, content.risk_explanation] + content.key_drivers).lower()

    if known_features:
        any_known = any(f.replace("_", " ") in full_text for f in known_features)
        if not any_known:
            notes.append("None of the supplied SHAP feature names appear anywhere in the narrative")

    for phrase in CAUSAL_PHRASES:
        if phrase in full_text:
            notes.append(f"Possible causal language detected: '{phrase}'")

    passed = len(notes) == 0
    return passed, notes