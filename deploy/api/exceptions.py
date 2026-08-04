"""Exception handlers: traduzem exceções de domínio em respostas HTTP padronizadas.

Nenhum router deve retornar um erro "cru" (stack trace, exceção genérica);
toda exceção de negócio é mapeada aqui para um :class:`ErrorResponse`
consistente, com um código de erro estável que clientes podem checar
programaticamente (em vez de fazer parsing de mensagem de erro).
"""

from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from deploy.api.schemas.responses import ErrorResponse
from deploy.api.services.feature_builder import UnknownZipcodeError
from deploy.api.services.model_registry import ModelArtifactError
from deploy.api.services.prediction_service import BatchSizeExceededError

logger = logging.getLogger(__name__)


def _json_error(status_code: int, error: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=error, message=message, details=details)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Erros de validação do Pydantic (schema do request) -> 422.

    `exc.errors()` pode conter objetos não serializáveis em `ctx` (ex.: a
    `ValueError` original de um `model_validator` customizado) —
    `jsonable_encoder` sabe reduzir esses casos a algo serializável (str),
    diferente de `model_dump()` puro.
    """
    return _json_error(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        error="validation_error",
        message="O corpo da requisição não passou na validação.",
        details={"errors": jsonable_encoder(exc.errors())},
    )


async def handle_unknown_zipcode(request: Request, exc: UnknownZipcodeError) -> JSONResponse:
    """CEP sem dados demográficos conhecidos -> 422 (erro do cliente, não do servidor)."""
    return _json_error(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        error="unknown_zipcode",
        message=str(exc),
        details={"zipcodes": exc.zipcodes},
    )


async def handle_batch_size_exceeded(request: Request, exc: BatchSizeExceededError) -> JSONResponse:
    """Lote maior que o permitido -> 413."""
    return _json_error(
        status.HTTP_413_CONTENT_TOO_LARGE,
        error="batch_size_exceeded",
        message=str(exc),
        details={"requested": exc.requested, "allowed": exc.allowed},
    )


async def handle_model_artifact_error(request: Request, exc: ModelArtifactError) -> JSONResponse:
    """Problema ao carregar/usar o modelo -> 503 (serviço indisponível, não erro do cliente)."""
    logger.error("Erro de artefato de modelo: %s", exc)
    return _json_error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        error="model_unavailable",
        message="O modelo de previsão está temporariamente indisponível.",
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Rede de segurança: qualquer exceção não mapeada vira 500 sem vazar detalhes internos."""
    logger.exception("Erro não tratado ao processar %s %s", request.method, request.url.path)
    return _json_error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        error="internal_error",
        message="Erro interno inesperado.",
    )


EXCEPTION_HANDLERS = {
    RequestValidationError: handle_validation_error,
    UnknownZipcodeError: handle_unknown_zipcode,
    BatchSizeExceededError: handle_batch_size_exceeded,
    ModelArtifactError: handle_model_artifact_error,
    Exception: handle_unexpected_error,
}
