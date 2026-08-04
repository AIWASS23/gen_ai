"""Fixtures compartilhadas dos testes da API.

Estes são testes de integração real: sobem a aplicação completa (incluindo
o `ModelRegistry` carregando os artefatos verdadeiros de `models/`) e
fazem requisições via `TestClient`. Cache e observabilidade ficam
desabilitados por padrão para não exigir Redis/Langfuse rodando — os
testes que precisam deles ligam explicitamente via `Settings` customizado.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from deploy.api.config import Settings
from deploy.api.main import create_app

VALID_HOUSE: dict = {
    "bedrooms": 4,
    "bathrooms": 1.0,
    "sqft_living": 1680,
    "sqft_lot": 5043,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 6,
    "sqft_above": 1680,
    "sqft_basement": 0,
    "yr_built": 1911,
    "yr_renovated": 0,
    "zipcode": 98118,
    "lat": 47.5354,
    "long": -122.273,
    "sqft_living15": 1560,
    "sqft_lot15": 5765,
}


def make_settings(**overrides: object) -> Settings:
    """Constrói um `Settings` de teste: cache/observabilidade desligados por
    padrão, sobrescrevíveis via kwargs."""
    defaults: dict[str, object] = {"cache_enabled": False, "observability_enabled": False}
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Cliente de teste com a aplicação completa (modelo real, sem cache/observabilidade)."""
    app = create_app(settings=make_settings())
    with TestClient(app) as test_client:
        yield test_client
