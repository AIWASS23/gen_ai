"""Endpoints de saúde: liveness (o processo está de pé?) e readiness (as
dependências estão OK para receber tráfego?) — distinção padrão para uso
com liveness/readiness probes do Kubernetes (ver `deploy/k8s/deployment.yaml`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from deploy.api.dependencies import get_cache, get_model_registry
from deploy.api.schemas.responses import ComponentHealth, HealthResponse
from deploy.api.services import PredictionCache
from deploy.api.services.model_registry import ModelArtifactError, ModelRegistry

router = APIRouter(tags=["health"])


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe: o processo está rodando?",
)
async def liveness() -> HealthResponse:
    """Não checa nenhuma dependência externa — só confirma que o processo
    da API está respondendo. Usado pelo Kubernetes para decidir se deve
    reiniciar o container (não se deve tirar tráfego dele)."""
    return HealthResponse(status="ok", components=[])


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe: a API está pronta para receber tráfego?",
)
async def readiness(
    registry: ModelRegistry = Depends(get_model_registry),
    cache: PredictionCache | None = Depends(get_cache),
) -> HealthResponse:
    """Checa as dependências críticas (modelo carregado) e opcionais (Redis).

    - Modelo indisponível -> `status="error"` (o pod deve sair do balanceamento).
    - Só o Redis indisponível -> `status="degraded"` (a API continua servindo
      previsões, só sem cache — não é motivo para tirar o pod de circulação).
    """
    components: list[ComponentHealth] = []

    model_healthy = True
    try:
        _ = registry.version
    except ModelArtifactError as exc:
        model_healthy = False
        components.append(ComponentHealth(name="model", healthy=False, detail=str(exc)))
    else:
        components.append(ComponentHealth(name="model", healthy=True))

    if cache is not None:
        redis_healthy = await cache.ping()
        components.append(
            ComponentHealth(name="redis", healthy=redis_healthy, detail=None if redis_healthy else "ping falhou")
        )
    else:
        redis_healthy = True  # cache desabilitado por configuração, não é uma falha

    if not model_healthy:
        status = "error"
    elif not redis_healthy:
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(status=status, components=components)
