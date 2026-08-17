"""api/dependencies.py"""

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from api.config import get_settings

# Registering this as a proper OpenAPI security scheme (rather than a plain
# Header() parameter) is what makes Swagger UI show a global Authorize
# button with a padlock, instead of a separate text field on every endpoint.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Simple shared-secret header check -- appropriate for a single-tenant
# academic deployment; not a substitute for OAuth2/JWT in a real multi-user
# production system, which is a deliberate non-goal here (see Phase 6
# security notes: no OAuth2/JWT/rate limiting without a genuine requirement)
def verify_api_key(x_api_key: str = Depends(_api_key_header)):
    settings = get_settings()
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured on the server")
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")