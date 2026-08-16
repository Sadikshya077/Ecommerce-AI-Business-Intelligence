"""api/security.py"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Header, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("API_KEY")


# Validate the API key provided in the request header.
def verify_api_key(x_api_key: str = Header(default=None)):
    # Reject requests if the server has no API key configured.
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured on the server")

    # Reject requests with a missing or incorrect API key.
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")