"""api/middleware.py"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("api.request")


# Add a request ID, measure request duration, and log the response details.
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate a short ID to trace the request through the logs.
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.time()

        response = await call_next(request)

        # Calculate request duration and attach the ID to the response.
        duration_ms = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "[%s] %s %s -> %d (%.1fms)",
            request_id, request.method, request.url.path, response.status_code, duration_ms,
        )
        return response