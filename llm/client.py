"""llm/client.py"""

import logging
import time

from google import genai
from google.genai import errors, types

from llm.config import get_llm_settings
from llm.schemas import NarrativeContent

logger = logging.getLogger("llm.client")


class LLMClientError(Exception):
    # Raised when the LLM cannot produce a usable response
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# Retry only failures where retrying is likely to help (rate limits,
# transient server errors, network/timeout). Don't retry permanent
# failures (bad request, auth, unsupported model) -- retrying those wastes
# time for no benefit, as seen when the earlier "credit balance too low"
# and "model not found" errors each got retried 3 times pointlessly.
def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    if isinstance(exc, errors.APIError):
        status_code = getattr(exc, "code", None)
        if status_code == 429:
            return True
        if status_code is not None and status_code >= 500:
            return True
        return False

    return False


# Calls Gemini with bounded timeout, selective retries, and
# schema-constrained JSON output. Returns (text, latency_ms, usage).
# Raises LLMClientError on configuration, provider, or response failure.
def call_llm(system_prompt: str, user_prompt: str) -> tuple:
    settings = get_llm_settings()
    if not settings.api_key:
        raise LLMClientError("GEMINI_API_KEY not configured")

    last_error = None
    for attempt in range(settings.max_retries + 1):
        try:
            # Gemini SDK timeout is specified in milliseconds.
            client = genai.Client(
                api_key=settings.api_key,
                http_options=types.HttpOptions(timeout=int(settings.timeout_seconds * 1000)),
            )

            start = time.time()
            response = client.models.generate_content(
                model=settings.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=NarrativeContent,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal"
                    ),
                    max_output_tokens=4096,
                ),
            )
            latency_ms = (time.time() - start) * 1000

            text = response.text
            if not text:
                raise LLMClientError("Gemini returned an empty response")

            usage_meta = getattr(response, "usage_metadata", None)
            usage = {
                "input_tokens": getattr(usage_meta, "prompt_token_count", None),
                "output_tokens": getattr(usage_meta, "candidates_token_count", None),
            }
            return text, latency_ms, usage

        except LLMClientError:
            raise

        except Exception as exc:
            last_error = exc
            retryable = _is_retryable(exc)
            logger.warning(
                "Gemini call failed (attempt %d/%d, retryable=%s): %s",
                attempt + 1, settings.max_retries + 1, retryable, exc,
            )

            if not retryable:
                raise LLMClientError(f"Gemini request failed: {exc}") from exc

            if attempt < settings.max_retries:
                time.sleep(2 ** attempt)

    raise LLMClientError(
        f"Gemini call failed after {settings.max_retries + 1} attempts: {last_error}"
    ) from last_error