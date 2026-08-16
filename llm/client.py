"""llm/client.py"""

import logging
import time

import anthropic

from llm.config import get_llm_settings

logger = logging.getLogger("llm.client")


# Raised on any failure -- never lets raw SDK exceptions leak upward.
# Callers (narrator.py) treat this as "the LLM is unavailable" and fall back.
class LLMClientError(Exception):
    def __init__(self, message: str):
        self.message = message


# Thin wrapper around the Anthropic API: bounded timeout, limited retries
# with exponential backoff (1s, 2s, 4s...). Never logs the API key.
def call_llm(system_prompt: str, user_prompt: str) -> tuple:
    settings = get_llm_settings()
    if not settings.api_key:
        raise LLMClientError("ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=settings.api_key, timeout=settings.timeout_seconds)

    last_error = None
    for attempt in range(settings.max_retries + 1):
        try:
            start = time.time()
            response = client.messages.create(
                model=settings.model,
                max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            latency_ms = (time.time() - start) * 1000
            text = "".join(block.text for block in response.content if block.type == "text")
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }
            return text, latency_ms, usage
        except Exception as exc:
            last_error = exc
            logger.warning(
                "LLM call failed (attempt %d/%d): %s", attempt + 1, settings.max_retries + 1, exc
            )
            if attempt < settings.max_retries:
                time.sleep(2 ** attempt)

    raise LLMClientError(f"LLM call failed after {settings.max_retries + 1} attempts: {last_error}")
