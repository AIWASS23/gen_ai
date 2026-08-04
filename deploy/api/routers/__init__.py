"""Routers HTTP da API."""

from deploy.api.routers import admin, health, model_info, predictions

__all__ = ["admin", "health", "model_info", "predictions"]
