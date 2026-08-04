"""Testes do endpoint de previsão: caminho feliz e validações de negócio."""

from __future__ import annotations

from fastapi.testclient import TestClient

from deploy.api.config import Settings
from deploy.api.main import create_app
from deploy.tests.conftest import VALID_HOUSE, make_settings


def test_predict_single_house_matches_offline_batch_prediction(client: TestClient) -> None:
    """Mesma casa usada em `outputs/future_predictions.csv` (ver src/predict.py) —
    o valor previsto pela API deve ser bit-a-bit igual ao gerado offline,
    provando que não há *skew* entre o pipeline de treino/batch e a API."""
    response = client.post("/v1/predictions", json={"houses": [VALID_HOUSE]})
    assert response.status_code == 200

    body = response.json()
    prediction = body["predictions"][0]
    assert prediction["predicted_price"] == 382704.878662418
    assert prediction["predicted_price_low"] == 274198.2092245995
    assert prediction["predicted_price_high"] == 480658.8292163427
    assert prediction["predicted_price_low"] < prediction["predicted_price"] < prediction["predicted_price_high"]
    assert prediction["cached"] is False
    assert body["model_version"]
    assert body["latency_ms"] > 0


def test_predict_batch_preserves_order(client: TestClient) -> None:
    cheap_house = {**VALID_HOUSE, "grade": 4, "sqft_living": 800, "sqft_above": 800}
    expensive_house = {**VALID_HOUSE, "grade": 11, "sqft_living": 5000, "sqft_above": 5000}

    response = client.post("/v1/predictions", json={"houses": [cheap_house, expensive_house]})
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["predicted_price"] < predictions[1]["predicted_price"]


def test_predict_rejects_known_bedroom_outlier(client: TestClient) -> None:
    """bedrooms=33 é o outlier de captura de dados removido no treino (ver
    notebooks/01_eda.ipynb) — o schema da API já rejeita esse valor na validação."""
    bad_house = {**VALID_HOUSE, "bedrooms": 33}
    response = client.post("/v1/predictions", json={"houses": [bad_house]})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_predict_rejects_structural_inconsistency(client: TestClient) -> None:
    """sqft_above + sqft_basement precisa ser igual a sqft_living."""
    bad_house = {**VALID_HOUSE, "sqft_above": 100, "sqft_basement": 0, "sqft_living": 999}
    response = client.post("/v1/predictions", json={"houses": [bad_house]})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_predict_rejects_unknown_zipcode(client: TestClient) -> None:
    """zipcode=10001 passa na validação de schema (5 dígitos, range genérico)
    e mantém lat/long dentro de King County, mas não existe em
    `zipcode_demographics.csv` — deve ser barrado na camada de features,
    não na de schema."""
    bad_house = {**VALID_HOUSE, "zipcode": 10001}
    response = client.post("/v1/predictions", json={"houses": [bad_house]})
    assert response.status_code == 422
    assert response.json()["error"] == "unknown_zipcode"


def test_predict_rejects_batch_larger_than_max_batch_size() -> None:
    app = create_app(settings=make_settings(max_batch_size=2))
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions", json={"houses": [VALID_HOUSE, VALID_HOUSE, VALID_HOUSE]}
        )
    assert response.status_code == 413
    assert response.json()["error"] == "batch_size_exceeded"


def test_predict_requires_at_least_one_house(client: TestClient) -> None:
    response = client.post("/v1/predictions", json={"houses": []})
    assert response.status_code == 422
