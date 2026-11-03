import secrets

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = get_settings().api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
