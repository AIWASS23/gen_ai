"""Contratos (Pydantic) de request/response da API."""

from deploy.api.schemas.requests import HouseFeatures, PredictionRequest
from deploy.api.schemas.responses import (
    ComponentHealth,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PricePrediction,
)

__all__ = [
    "HouseFeatures",
    "PredictionRequest",
    "ComponentHealth",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "PredictionResponse",
    "PricePrediction",
]
