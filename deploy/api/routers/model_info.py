"""Endpoint de metadados do modelo atualmente carregado (versionamento/auditoria)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from deploy.api.dependencies import get_model_registry
from deploy.api.schemas.responses import ModelInfoResponse
from deploy.api.services.model_registry import ModelRegistry

router = APIRouter(prefix="/v1", tags=["model"])


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Metadados do modelo em produção (versão, métricas, features)",
)
async def get_model_info(registry: ModelRegistry = Depends(get_model_registry)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=registry.version,
        best_model_name=registry.best_model_name,
        trained_at=registry.trained_at,
        holdout_metrics=registry.holdout_metrics,
        feature_columns=registry.feature_columns,
    )
