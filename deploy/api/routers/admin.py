"""Endpoints administrativos (fora do contrato público da API).

Em produção, este router deveria ficar atrás de uma rede interna/mTLS/IAM,
não só da checagem de API key abaixo — a checagem aqui é uma segunda camada
de defesa, não a única.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from deploy.api.config import Settings, get_settings
from deploy.api.dependencies import get_model_registry
from deploy.api.schemas.responses import ModelInfoResponse
from deploy.api.services.model_registry import ModelRegistry

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _require_admin_api_key(
    x_admin_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Dependência de autorização: exige `X-Admin-Api-Key` == `settings.admin_api_key`.

    Falha fechado: se `admin_api_key` não estiver configurado, o endpoint
    fica indisponível (503) em vez de aberto sem autenticação (400/401
    seriam enganosos aqui — o problema é configuração do servidor, não do
    cliente).
    """
    if settings.admin_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Endpoint administrativo desabilitado (ADMIN_API_KEY não configurada).",
        )
    if x_admin_api_key != settings.admin_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-Admin-Api-Key inválida.")


@router.post(
    "/reload-model",
    response_model=ModelInfoResponse,
    summary="Recarrega os artefatos do modelo do disco, sem reiniciar o processo",
    dependencies=[Depends(_require_admin_api_key)],
)
async def reload_model(registry: ModelRegistry = Depends(get_model_registry)) -> ModelInfoResponse:
    """Aciona `ModelRegistry.reload()` — útil após publicar um novo
    `model.joblib` em produção (ex.: volume compartilhado atualizado por um
    job de retreino), permitindo trocar de versão sem downtime."""
    registry.reload()
    return ModelInfoResponse(
        model_version=registry.version,
        best_model_name=registry.best_model_name,
        trained_at=registry.trained_at,
        holdout_metrics=registry.holdout_metrics,
        feature_columns=registry.feature_columns,
    )
