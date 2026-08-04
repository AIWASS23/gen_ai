"""Endpoint principal da API: previsão de preços de imóveis."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from deploy.api.dependencies import get_prediction_service
from deploy.api.schemas.requests import PredictionRequest
from deploy.api.schemas.responses import PredictionResponse
from deploy.api.services import PredictionService

router = APIRouter(prefix="/v1", tags=["predictions"])


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Prevê o preço de um lote de imóveis",
    description=(
        "Recebe de 1 a `max_batch_size` imóveis e retorna, para cada um, o "
        "preço estimado (ponto central) e um intervalo de previsão "
        "(~p10–p90). Previsões repetidas para o mesmo imóvel e a mesma "
        "versão de modelo são servidas do cache Redis quando habilitado."
    ),
)
async def create_predictions(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return await service.predict(request)
