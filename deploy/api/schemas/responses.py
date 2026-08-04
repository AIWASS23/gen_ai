"""Modelos de response da API (contratos de saída), fortemente tipados."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PricePrediction(BaseModel):
    """Previsão de preço para um único imóvel: ponto estimado + intervalo."""

    predicted_price: float = Field(..., description="Preço estimado (ponto central), em USD.")
    predicted_price_low: float = Field(..., description="Limite inferior do intervalo de previsão (~p10), em USD.")
    predicted_price_high: float = Field(..., description="Limite superior do intervalo de previsão (~p90), em USD.")
    currency: Literal["USD"] = "USD"
    cached: bool = Field(..., description="True se a previsão veio do cache Redis em vez de ser recomputada.")


class PredictionResponse(BaseModel):
    """Resposta de `POST /v1/predictions`: uma previsão por imóvel enviado, na mesma ordem."""

    request_id: str = Field(..., description="Eco do request_id enviado (ou gerado, se omitido).")
    model_version: str = Field(..., description="Identificador da versão do modelo que gerou as previsões.")
    predictions: list[PricePrediction] = Field(..., description="Previsões, na mesma ordem da lista `houses` enviada.")
    latency_ms: float = Field(..., description="Tempo total de processamento da requisição, em milissegundos.")


class ModelInfoResponse(BaseModel):
    """Resposta de `GET /v1/model`: metadados do modelo atualmente carregado."""

    model_version: str
    best_model_name: str = Field(..., description="Nome do algoritmo vencedor na comparação de treino (ex.: 'stacking').")
    trained_at: datetime | None = Field(None, description="Timestamp de modificação do artefato do modelo (proxy para data de treino).")
    holdout_metrics: dict[str, float] = Field(..., description="Métricas no conjunto de holdout (rmse, mae, mape, r2).")
    feature_columns: list[str] = Field(..., description="Colunas de feature esperadas pelo modelo, na ordem usada no treino.")


class ComponentHealth(BaseModel):
    """Status de saúde de uma dependência individual (ex.: Redis, modelo carregado)."""

    name: str
    healthy: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Resposta de `GET /health`: status agregado do serviço e de suas dependências."""

    status: Literal["ok", "degraded", "error"]
    components: list[ComponentHealth]


class ErrorResponse(BaseModel):
    """Formato padronizado de erro retornado pelos exception handlers."""

    error: str = Field(..., description="Código curto e estável do erro (ex.: 'validation_error').")
    message: str = Field(..., description="Mensagem legível descrevendo o problema.")
    details: dict[str, Any] | None = Field(None, description="Contexto adicional opcional (ex.: erros de validação por campo).")
