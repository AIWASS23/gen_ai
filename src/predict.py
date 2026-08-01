"""Gera previsões de preço para imóveis sem preço conhecido.

Uso:
    uv run python -m src.predict

Lê `data/raw/future_unseen_examples.csv`, aplica a mesma engenharia de
features e o mesmo merge demográfico usados no treino, carrega o modelo
final (ponto estimado) e os dois modelos de quantil (limites inferior/
superior) salvos em `models/`, e escreve as previsões em
`outputs/future_predictions.csv` com uma faixa de preço (não só um valor
único) — ver `docs/stakeholder_communication.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import (
    build_feature_frame,
    load_demographics,
    load_future_examples,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
QUANTILE_LOWER_PATH = ROOT / "models" / "quantile_lower.joblib"
QUANTILE_UPPER_PATH = ROOT / "models" / "quantile_upper.joblib"
FEATURE_COLUMNS_PATH = ROOT / "models" / "feature_columns.json"
OUTPUT_PATH = ROOT / "outputs" / "future_predictions.csv"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    lower_model = joblib.load(QUANTILE_LOWER_PATH)
    upper_model = joblib.load(QUANTILE_UPPER_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_cols = json.load(f)

    future = load_future_examples()
    demographics = load_demographics()

    features = build_feature_frame(future, demographics)
    missing = set(feature_cols) - set(features.columns)
    if missing:
        raise ValueError(f"Faltam colunas esperadas pelo modelo: {missing}")

    X_future = features[feature_cols]
    point_pred = model.predict(X_future)
    lower_pred = lower_model.predict(X_future)
    upper_pred = upper_model.predict(X_future)

    # Os modelos de quantil são treinados de forma independente (não há uma
    # garantia matemática de monotonicidade entre eles); por segurança,
    # ordenamos os limites para nunca reportar um intervalo invertido.
    lower_pred, upper_pred = np.minimum(lower_pred, upper_pred), np.maximum(lower_pred, upper_pred)

    result = future.copy()
    result["predicted_price"] = point_pred
    result["predicted_price_low"] = lower_pred
    result["predicted_price_high"] = upper_pred

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"{len(result)} previsões geradas -> {OUTPUT_PATH}")
    print(result[["predicted_price", "predicted_price_low", "predicted_price_high"]].describe())


if __name__ == "__main__":
    main()
