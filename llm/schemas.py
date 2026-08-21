"""llm/schemas.py"""

from typing import List, Optional

from pydantic import BaseModel


# The LLM's own structured output -- nothing is trusted or returned to a
# caller until it validates against this
class NarrativeContent(BaseModel):
    summary: str
    risk_explanation: str
    key_drivers: List[str]
    recommended_actions: List[str]
    limitations: List[str]


# Wraps validated content with generation metadata (model, prompt version,
# timestamp, latency, tokens, faithfulness status) -- tracked per Phase 7
# requirements, never includes the API key or any secret
class NarrativeResult(BaseModel):
    content: NarrativeContent
    model: str
    prompt_version: str
    generated_at: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    faithfulness_passed: bool
    faithfulness_notes: List[str] = []
