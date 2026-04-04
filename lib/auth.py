from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import os
import secrets

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
VALID_API_KEY = os.getenv("MICROSERVICE_API_KEY")


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured",
        )
    if not secrets.compare_digest(api_key, VALID_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return api_key
