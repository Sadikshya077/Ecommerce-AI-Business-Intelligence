"""llm/narrator.py"""

import json
import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from llm.client import LLMClientError, call_llm
from llm.config import get_llm_settings
from llm.faithfulness import check_faithfulness
from llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from llm.schemas import NarrativeContent, NarrativeResult

logger = logging.getLogger("llm.narrator")


# Orchestrates prompt -> LLM call -> JSON validation -> faithfulness check.
# Never raises -- returns None on any failure (LLM unavailable, malformed
# output, schema violation) so callers can degrade gracefully rather than
# let a narration failure take down an endpoint that has perfectly good
# analytical data to return anyway.
def generate_narrative(insight_context: dict) -> NarrativeResult | None:
    settings = get_llm_settings()
    user_prompt = build_user_prompt(insight_context)

    try:
        raw_text, latency_ms, usage = call_llm(SYSTEM_PROMPT, user_prompt)
    except LLMClientError as exc:
        logger.error("Narrative generation failed (LLM unavailable): %s", exc.message)
        return None

    try:
        parsed = json.loads(raw_text)
        content = NarrativeContent(**parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("Narrative generation failed (invalid LLM output): %s", exc)
        return None

    passed, notes = check_faithfulness(content, insight_context)

    return NarrativeResult(
        content=content,
        model=settings.model,
        prompt_version=PROMPT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        latency_ms=round(latency_ms, 1),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        faithfulness_passed=passed,
        faithfulness_notes=notes,
    )
