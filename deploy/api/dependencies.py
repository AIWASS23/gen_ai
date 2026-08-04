"""Provedores de injeção de dependência do FastAPI.

Os serviços (model registry, cache, observabilidade, prediction service)
são instanciados uma única vez no `lifespan` da aplicação (`main.py`) e
guardados em `app.state`. As funções abaixo apenas os recuperam de lá —
mantê-las simples assim é o que permite sobrescrevê-las facilmente em
testes via `app.dependency_overrides`.
"""

from __future__ import annotations

from fastapi import Request

from deploy.api.config import get_settings
from deploy.api.services import ModelRegistry, ObservabilityService, PredictionCache, PredictionService

__all__ = [
    "get_settings",
    "get_model_registry",
    "get_prediction_service",
    "get_cache",
    "get_observability",
]


def get_model_registry(request: Request) -> ModelRegistry:
    """Retorna o :class:`ModelRegistry` inicializado no startup da aplicação."""
    return request.app.state.model_registry


def get_prediction_service(request: Request) -> PredictionService:
    """Retorna o :class:`PredictionService` inicializado no startup da aplicação."""
    return request.app.state.prediction_service


def get_cache(request: Request) -> PredictionCache | None:
    """Retorna o :class:`PredictionCache` ativo, ou `None` se o cache estiver desabilitado."""
    return request.app.state.cache


def get_observability(request: Request) -> ObservabilityService:
    """Retorna o :class:`ObservabilityService` inicializado no startup da aplicação."""
    return request.app.state.observability
