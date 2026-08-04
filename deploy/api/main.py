"""Ponto de entrada da API de previsão de preços de imóveis.

Uso local:
    uv run uvicorn deploy.api.main:app --reload --port 8000

Uso em produção: ver `deploy/Dockerfile` (roda `uvicorn` sem `--reload`,
atrás de um processo supervisor/orquestrador — Kubernetes, no caso deste
projeto, ver `deploy/k8s/`).
"""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deploy.api.config import Settings, get_settings
from deploy.api.exceptions import EXCEPTION_HANDLERS
from deploy.api.middleware import RequestLoggingMiddleware
from deploy.api.routers import admin, health, model_info, predictions
from deploy.api.services import (
    FeatureBuilder,
    ModelRegistry,
    ObservabilityService,
    PredictionCache,
    PredictionService,
)

logger = logging.getLogger(__name__)


def _configure_logging(settings: Settings) -> None:
    """Logging estruturado (JSON) — mais fácil de indexar em qualquer
    stack de observabilidade (ELK, Loki, CloudWatch Logs, etc.) do que
    texto livre."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                }
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "json"},
            },
            "root": {"level": settings.log_level, "handlers": ["console"]},
        }
    )


def _build_lifespan(settings: Settings):
    """Fábrica do context manager de lifespan, fechando sobre `settings`.

    Evita que o lifespan redescubra a configuração via `get_settings()`
    (o singleton cacheado por env) — ele deve sempre usar exatamente o
    `Settings` passado para `create_app`, o que importa para testes que
    injetam um `Settings` customizado (ex.: apontando para artefatos de
    modelo de teste).
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Inicializa (startup) e libera (shutdown) todos os recursos de longa
        duração da aplicação: modelo, Redis, cliente Langfuse.

        Tudo fica em `app.state`, de onde os provedores de dependência em
        `dependencies.py` o recuperam por requisição.
        """
        _configure_logging(settings)
        logger.info("Iniciando %s (env=%s)", settings.app_name, settings.environment)

        model_registry = ModelRegistry(
            model_path=settings.model_path,
            quantile_lower_path=settings.quantile_lower_path,
            quantile_upper_path=settings.quantile_upper_path,
            feature_columns_path=settings.feature_columns_path,
            metrics_path=settings.metrics_path,
        )
        model_registry.load()

        feature_builder = FeatureBuilder(demographics_path=settings.demographics_path)
        observability = ObservabilityService(settings)

        cache: PredictionCache | None = None
        redis_client: redis.Redis | None = None
        if settings.cache_enabled:
            redis_client = redis.from_url(
                settings.redis_url,
                socket_timeout=settings.redis_socket_timeout_seconds,
                decode_responses=True,
            )
            cache = PredictionCache(
                client=redis_client, key_prefix=settings.cache_key_prefix, ttl_seconds=settings.cache_ttl_seconds
            )

        prediction_service = PredictionService(
            model_registry=model_registry,
            feature_builder=feature_builder,
            observability=observability,
            cache=cache,
            max_batch_size=settings.max_batch_size,
        )

        app.state.settings = settings
        app.state.model_registry = model_registry
        app.state.feature_builder = feature_builder
        app.state.observability = observability
        app.state.cache = cache
        app.state.prediction_service = prediction_service

        logger.info("Inicialização concluída: modelo versão=%s", model_registry.version)
        try:
            yield
        finally:
            logger.info("Encerrando %s", settings.app_name)
            observability.shutdown()
            if redis_client is not None:
                await redis_client.aclose()

    return _lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory: constrói uma instância de :class:`FastAPI`
    totalmente configurada (rotas, middlewares, exception handlers).

    Aceita `settings` explícito para facilitar testes (ex.: apontar para um
    diretório de artefatos de teste) sem depender de variáveis de ambiente.
    """
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.api_version,
        description=(
            "API de previsão de preços de imóveis (King County, WA). "
            "Ver docs/model.md e docs/deploy_strategy.md no repositório para o desenho completo."
        ),
        docs_url=None if resolved_settings.is_production else "/docs",
        redoc_url=None if resolved_settings.is_production else "/redoc",
        lifespan=_build_lifespan(resolved_settings),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    for exc_type, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_type, handler)

    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(model_info.router)
    app.include_router(admin.router)

    # Qualquer `Depends(get_settings)` (ex.: a checagem de API key em
    # admin.py) deve enxergar exatamente `resolved_settings`, não o
    # singleton cacheado por variáveis de ambiente — essencial para testes
    # que constroem a app com um `Settings` customizado.
    app.dependency_overrides[get_settings] = lambda: resolved_settings

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": resolved_settings.app_name,
            "version": resolved_settings.api_version,
            "docs": "/docs" if not resolved_settings.is_production else "disabled",
        }

    return app


app = create_app()
