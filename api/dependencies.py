"""api/dependencies.py"""

from fastapi import Header, HTTPException

from api.config import get_settings


# Validate the shared API key from the request header.
def verify_api_key(x_api_key: str = Header(default=None)):
    settings = get_settings()

    # Reject requests if the server has no API key configured.
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured on the server")

    # Reject requests with a missing or incorrect API key.
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")