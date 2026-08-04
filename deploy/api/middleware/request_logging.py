"""Middleware de logging estruturado de requisições HTTP."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("deploy.api.access")

_CORRELATION_HEADER = "X-Correlation-Id"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Loga método, path, status e latência de cada requisição.

    Também propaga um `X-Correlation-Id` (recebido do cliente ou gerado),
    devolvido no header da resposta e disponível em `request.state.correlation_id`
    para os routers incluírem em traces/logs próprios.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = request.headers.get(_CORRELATION_HEADER, uuid.uuid4().hex)
        request.state.correlation_id = correlation_id

        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "request_failed method=%s path=%s duration_ms=%.2f correlation_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                correlation_id,
            )
            raise

        duration_ms = (time.perf_counter() - started_at) * 1000
        response.headers[_CORRELATION_HEADER] = correlation_id
        logger.info(
            "request_completed method=%s path=%s status=%d duration_ms=%.2f correlation_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            correlation_id,
        )
        return response
