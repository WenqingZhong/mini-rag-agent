"""
TIER 3: API Key Authentication
================================
Production equivalent:
  Spring Security
  + allowedGroups check in AgentRegistryRsp
  + UserUtils.hasAnyRole(allowedGroups)

HOW IT WORKS:
  Every protected endpoint declares Depends(verify_api_key).
  FastAPI runs the dependency before the handler — if the key is wrong,
  it raises 401 and the handler never executes.

  Client sends:  X-API-Key: dev-secret-key-change-me
  Server checks: does it match config.API_KEY?

PRODUCTION DIFFERENCE:
  Tech-assistant-service validates JWT tokens from PingFederate (Walmart's IdP).
  The JWT contains group memberships checked against each agent's allowedGroups list.
  Here we use a simple shared API key — same pattern, less infrastructure.

USAGE (in main.py):
  from middleware.auth import verify_api_key

  @app.post("/chat", dependencies=[Depends(verify_api_key)])
  def chat(...):
      ...
"""

from fastapi import Header, HTTPException, status
from config import API_KEY


async def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency that validates the X-API-Key request header.
    Raises HTTP 401 if the key is missing or wrong.

    Using Header(None) instead of Header(...) so a missing header returns 401
    (our error) rather than 422 (FastAPI's default validation error).
    """
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid or missing API key.",
                "hint": (
                    "Set the X-API-Key header. "
                    "Default dev key: dev-secret-key-change-me"
                ),
            },
        )
    return x_api_key
