"""
TIER 1: Global Error Handlers
================================
Production equivalent:
  ObjectNotFoundException, ResponseStatusException,
  and Spring's @ControllerAdvice global handler.

WITHOUT THIS:
  An unhandled KeyError (agent not found) returns a raw Python traceback
  with HTTP 500 — not helpful for API consumers and leaks implementation details.

WITH THIS:
  Every exception type maps to a specific HTTP status + clean JSON body:

  KeyError        → 404  { "error": "...", "type": "NotFoundError"   }
  ValueError      → 422  { "error": "...", "type": "ValidationError" }
  PermissionError → 403  { "error": "...", "type": "PermissionError" }
  Exception       → 500  { "error": "Internal server error", "type": "..." }

USAGE (in main.py):
  from middleware.errors import register_error_handlers
  register_error_handlers(app)
"""

import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from utils.logger import get_logger

logger = get_logger("errors")


def register_error_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI app."""

    @app.exception_handler(KeyError)
    async def not_found_handler(request: Request, exc: KeyError) -> JSONResponse:
        # Production: ObjectNotFoundException → HTTP 404
        logger.warning("Not found: %s %s", request.url.path, str(exc))
        return JSONResponse(
            status_code=404,
            content={"error": f"Not found: {exc}", "type": "NotFoundError"},
        )

    @app.exception_handler(ValueError)
    async def validation_handler(request: Request, exc: ValueError) -> JSONResponse:
        # Production: @Valid / MethodArgumentNotValidException → HTTP 422
        return JSONResponse(
            status_code=422,
            content={"error": str(exc), "type": "ValidationError"},
        )

    @app.exception_handler(PermissionError)
    async def permission_handler(
        request: Request, exc: PermissionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"error": str(exc), "type": "PermissionError"},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log full traceback server-side, but never expose it to the client
        logger.error(
            "Unhandled exception on %s: %s\n%s",
            request.url.path,
            exc,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error. Check server logs.",
                "type": type(exc).__name__,
            },
        )
