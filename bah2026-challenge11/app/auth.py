"""JWT authentication middleware — toggled via ENABLE_AUTH env var."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"

_bearer_scheme = HTTPBearer(auto_error=False)


def _decode_token(token: str) -> dict:
    """Decode and verify a JWT token.

    Args:
        token: Raw JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException: 401 if invalid or expired.
    """
    try:
        import jwt

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


async def require_auth(request: Request) -> Optional[dict]:
    """FastAPI dependency: validate Bearer JWT when ENABLE_AUTH=true.

    Returns None immediately when auth is disabled (default for hackathon).

    Args:
        request: Incoming FastAPI request.

    Returns:
        Decoded JWT payload dict, or None if auth is disabled.

    Raises:
        HTTPException: 401 if auth is enabled and the token is missing/invalid.
    """
    if not ENABLE_AUTH:
        return None

    credentials: Optional[HTTPAuthorizationCredentials] = await _bearer_scheme(request)
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    payload = _decode_token(credentials.credentials)
    logger.debug(f"Authenticated user: {payload.get('sub', 'unknown')}")
    return payload
