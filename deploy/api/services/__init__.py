"""Camada de serviços: model registry, features, cache e observabilidade."""

from deploy.api.services.cache import PredictionCache
from deploy.api.services.feature_builder import FeatureBuilder, UnknownZipcodeError
from deploy.api.services.model_registry import ModelArtifactError, ModelRegistry
from deploy.api.services.observability import ObservabilityService
from deploy.api.services.prediction_service import BatchSizeExceededError, PredictionService

__all__ = [
    "PredictionCache",
    "FeatureBuilder",
    "UnknownZipcodeError",
    "ModelArtifactError",
    "ModelRegistry",
    "ObservabilityService",
    "PredictionService",
    "BatchSizeExceededError",
]
