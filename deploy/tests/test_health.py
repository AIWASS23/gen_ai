"""Testes dos endpoints de saúde."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_liveness_always_ok(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "components": []}


def test_readiness_reports_model_healthy(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"  # cache desabilitado -> só o componente "model"
    component_names = {c["name"] for c in body["components"]}
    assert component_names == {"model"}
    assert all(c["healthy"] for c in body["components"])
