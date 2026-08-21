"""api/exceptions.py"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.exceptions")


# Raised when a requested customer_unique_id doesn't exist in the store
class CustomerNotFoundError(Exception):
    def __init__(self, customer_id: str):
        self.customer_id = customer_id


# Raised when a required model artifact or data file is missing on disk
class ModelArtifactMissingError(Exception):
    def __init__(self, message: str):
        self.message = message


# Raised when a service call fails for a reason not caused by bad input
class PredictionServiceError(Exception):
    def __init__(self, message: str):
        self.message = message


# Raised when the LLM narration layer fails to produce a valid narrative
# (timeout, malformed output, schema violation). Kept distinct from
# ModelArtifactMissingError -- a live external API being unavailable is a
# different failure class than a missing local file, worth distinguishing
# in logs and in how a client might reasonably react (retry later vs. rerun
# a pipeline script).
class NarrationUnavailableError(Exception):
    def __init__(self, message: str):
        self.message = message


# Each handler returns a safe, minimal client-facing message; full detail
# goes to the server log only -- no stack traces, paths, or internals leak out
async def customer_not_found_handler(request: Request, exc: CustomerNotFoundError):
    return JSONResponse(status_code=404, content={"detail": f"Customer '{exc.customer_id}' not found"})


async def model_artifact_missing_handler(request: Request, exc: ModelArtifactMissingError):
    logger.error("Model artifact missing: %s", exc.message)
    return JSONResponse(status_code=503, content={"detail": "A required model artifact is unavailable"})


async def prediction_service_error_handler(request: Request, exc: PredictionServiceError):
    logger.error("Prediction service error: %s", exc.message)
    return JSONResponse(status_code=500, content={"detail": "Prediction service encountered an error"})


async def narration_unavailable_handler(request: Request, exc: NarrationUnavailableError):
    logger.error("Narration unavailable: %s", exc.message)
    return JSONResponse(status_code=503, content={"detail": "Narrative generation is temporarily unavailable"})


# Catch-all for anything unexpected
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def register_exception_handlers(app):
    app.add_exception_handler(CustomerNotFoundError, customer_not_found_handler)
    app.add_exception_handler(ModelArtifactMissingError, model_artifact_missing_handler)
    app.add_exception_handler(PredictionServiceError, prediction_service_error_handler)
    app.add_exception_handler(NarrationUnavailableError, narration_unavailable_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)