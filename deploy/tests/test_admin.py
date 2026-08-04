"""Testes do endpoint administrativo de reload do modelo."""

from __future__ import annotations

from fastapi.testclient import TestClient

from deploy.api.main import create_app
from deploy.tests.conftest import make_settings


def test_reload_disabled_without_admin_api_key(client: TestClient) -> None:
    """Sem `admin_api_key` configurada, o endpoint fica indisponível (falha fechado)."""
    response = client.post("/v1/admin/reload-model")
    assert response.status_code == 503


def test_reload_rejects_wrong_api_key() -> None:
    app = create_app(settings=make_settings(admin_api_key="correct-key"))
    with TestClient(app) as client:
        response = client.post("/v1/admin/reload-model", headers={"X-Admin-Api-Key": "wrong-key"})
    assert response.status_code == 401


def test_reload_succeeds_with_correct_api_key() -> None:
    app = create_app(settings=make_settings(admin_api_key="correct-key"))
    with TestClient(app) as client:
        response = client.post("/v1/admin/reload-model", headers={"X-Admin-Api-Key": "correct-key"})
    assert response.status_code == 200
    assert response.json()["model_version"]
