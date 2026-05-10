"""
Small role-based API key auth for protected operational endpoints.

This is intentionally simple: production deployments set ADMIN_API_KEY and
optionally RESEARCH_API_KEY. Local development can leave AUTH_REQUIRED=false
to keep read-only demo flows easy while still enforcing auth if a key is set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthContext:
    role: str
    subject: str


def _token_role(token: str) -> str | None:
    settings = get_settings()
    if settings.admin_api_key and token == settings.admin_api_key:
        return "admin"
    if settings.research_api_key and token == settings.research_api_key:
        return "research"
    return None


def _auth_is_enforced() -> bool:
    settings = get_settings()
    return settings.auth_required or bool(settings.admin_api_key or settings.research_api_key)


def require_role(*roles: str) -> Callable:
    """FastAPI dependency that requires a bearer token with one of `roles`."""
    allowed = set(roles)

    async def dependency(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    ) -> AuthContext:
        if not _auth_is_enforced():
            return AuthContext(role="admin", subject="development-bypass")

        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        role = _token_role(credentials.credentials)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {', '.join(sorted(allowed))}",
            )

        return AuthContext(role=role, subject=role)

    return dependency
