"""Testes do endpoint de metadados do modelo."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_model_info_returns_expected_shape(client: TestClient) -> None:
    response = client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()

    assert body["model_version"]  # hash não vazio
    assert body["best_model_name"]
    assert set(body["holdout_metrics"]) >= {"rmse", "mae", "mape", "r2"}
    assert "grade" in body["feature_columns"]
    assert "sqft_living" in body["feature_columns"]
