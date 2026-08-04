"""Constrói o DataFrame de features esperado pelo modelo a partir de requests da API.

Reaproveita deliberadamente a mesma lógica de engenharia de features e merge
demográfico usada no treino (`src/data.py`) — a única forma de garantir que
não existe *training-serving skew* (a API calculando uma feature de um jeito
sutilmente diferente do pipeline de treino) é usar literalmente o mesmo
código nos dois lugares, não uma reimplementação paralela.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import add_demographics, engineer_features, load_demographics

from deploy.api.schemas.requests import HouseFeatures


class FeatureBuilder:
    """Converte payloads de request (`HouseFeatures`) no DataFrame de
    features que os pipelines do modelo esperam como entrada."""

    def __init__(self, demographics_path: Path) -> None:
        """Carrega a tabela de demografia por CEP uma única vez (70 linhas,
        cabe inteira em memória) para reuso em todas as requisições."""
        self._demographics = load_demographics(demographics_path)

    def build(self, houses: list[HouseFeatures]) -> pd.DataFrame:
        """Monta o DataFrame de features para um lote de imóveis.

        Aplica, nesta ordem, exatamente os mesmos passos do treino:
        1. Engenharia de features físicas (`house_age`, `was_renovated`, ...).
        2. Merge com dados demográficos por `zipcode`.

        Levanta `KeyError` (propagada) se algum `zipcode` do request não
        existir em `zipcode_demographics.csv` — o router traduz isso em uma
        resposta HTTP 422 (ver `deploy/api/exceptions.py`).
        """
        raw = pd.DataFrame([house.model_dump() for house in houses])
        engineered = engineer_features(raw)
        merged = add_demographics(engineered, self._demographics)

        unknown_mask = merged["ppltn_qty"].isna()
        if unknown_mask.any():
            unknown_zipcodes = sorted(set(raw.loc[unknown_mask.to_numpy(), "zipcode"]))
            raise UnknownZipcodeError(unknown_zipcodes)
        return merged


class UnknownZipcodeError(ValueError):
    """Levantado quando um `zipcode` do request não tem dados demográficos conhecidos."""

    def __init__(self, zipcodes: list[int]) -> None:
        self.zipcodes = zipcodes
        super().__init__(f"CEP(s) sem dados demográficos conhecidos: {zipcodes}")
