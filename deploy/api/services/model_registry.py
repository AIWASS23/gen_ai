"""Registro leve de modelo: carrega, versiona e serve os artefatos treinados.

Esta é uma implementação de referência do papel de "Model Registry" descrito
em `docs/deploy_strategy.md`. Em uma stack de produção mais madura, isso
seria substituído por um registro externo (ex.: MLflow Model Registry,
SageMaker Model Registry) capaz de servir múltiplas versões simultaneamente
e promover/rebaixar modelos sem redeploy de código. Aqui, a "versão" é
derivada do hash de conteúdo do artefato — suficiente para detectar
mudanças e permitir rollback manual trocando o arquivo em `models/`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelArtifactError(RuntimeError):
    """Erro ao carregar ou usar um artefato de modelo (arquivo ausente/corrompido)."""


@dataclass(frozen=True, slots=True)
class _LoadedArtifacts:
    """Snapshot imutável de todos os artefatos carregados de uma vez.

    Trocar a referência a este objeto (em vez de mutar atributos individuais)
    torna `ModelRegistry.reload()` seguro sob concorrência: um request em
    andamento sempre vê um conjunto consistente de artefatos, nunca uma
    mistura de versão antiga e nova.
    """

    point_pipeline: Pipeline
    lower_pipeline: Pipeline
    upper_pipeline: Pipeline
    feature_columns: list[str]
    metrics: dict[str, Any]
    version: str
    trained_at: datetime | None


class ModelRegistry:
    """Carrega e serve os artefatos do modelo de previsão de preços.

    Responsabilidades:
        - Carregar os 3 pipelines (ponto, quantil inferior, quantil superior)
          e os metadados (`feature_columns.json`, `metrics.json`).
        - Calcular um identificador de versão determinístico a partir do
          conteúdo do artefato de ponto.
        - Expor `predict_point`/`predict_interval` sem vazar detalhes de
          scikit-learn para o resto da aplicação.
    """

    def __init__(
        self,
        model_path: Path,
        quantile_lower_path: Path,
        quantile_upper_path: Path,
        feature_columns_path: Path,
        metrics_path: Path,
    ) -> None:
        self._model_path = model_path
        self._quantile_lower_path = quantile_lower_path
        self._quantile_upper_path = quantile_upper_path
        self._feature_columns_path = feature_columns_path
        self._metrics_path = metrics_path
        self._lock = threading.Lock()
        self._artifacts: _LoadedArtifacts | None = None

    def load(self) -> None:
        """Carrega (ou recarrega) todos os artefatos do disco.

        Levanta :class:`ModelArtifactError` se algum arquivo estiver
        ausente ou não puder ser desserializado — falha rápido no startup
        em vez de deixar a API subir num estado inconsistente.
        """
        try:
            point_pipeline = joblib.load(self._model_path)
            lower_pipeline = joblib.load(self._quantile_lower_path)
            upper_pipeline = joblib.load(self._quantile_upper_path)
            feature_columns = json.loads(self._feature_columns_path.read_text())
            metrics = json.loads(self._metrics_path.read_text())
        except FileNotFoundError as exc:
            raise ModelArtifactError(f"Artefato de modelo não encontrado: {exc.filename}") from exc
        except Exception as exc:  # desserialização corrompida, versão incompatível de lib, etc.
            raise ModelArtifactError(f"Falha ao carregar artefatos do modelo: {exc}") from exc

        version = self._compute_version(self._model_path)
        trained_at = self._file_mtime(self._model_path)

        with self._lock:
            self._artifacts = _LoadedArtifacts(
                point_pipeline=point_pipeline,
                lower_pipeline=lower_pipeline,
                upper_pipeline=upper_pipeline,
                feature_columns=feature_columns,
                metrics=metrics,
                version=version,
                trained_at=trained_at,
            )
        logger.info("Modelo carregado: versão=%s treinado_em=%s", version, trained_at)

    def reload(self) -> None:
        """Recarrega os artefatos do disco (ex.: após um novo treino ser
        publicado em `models/`), sem reiniciar o processo da API."""
        self.load()

    @staticmethod
    def _compute_version(model_path: Path) -> str:
        """Hash sha256 (12 chars) do conteúdo do artefato de ponto — muda
        sempre que o modelo é retreinado e serializado novamente."""
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        return digest[:12]

    @staticmethod
    def _file_mtime(path: Path) -> datetime | None:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except FileNotFoundError:
            return None

    @property
    def _state(self) -> _LoadedArtifacts:
        if self._artifacts is None:
            raise ModelArtifactError("ModelRegistry.load() precisa ser chamado antes do uso")
        return self._artifacts

    @property
    def version(self) -> str:
        """Identificador de versão do modelo atualmente carregado."""
        return self._state.version

    @property
    def trained_at(self) -> datetime | None:
        return self._state.trained_at

    @property
    def feature_columns(self) -> list[str]:
        return list(self._state.feature_columns)

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._state.metrics)

    @property
    def best_model_name(self) -> str:
        return str(self._state.metrics.get("best_model", "unknown"))

    @property
    def holdout_metrics(self) -> dict[str, float]:
        return dict(self._state.metrics.get("holdout_test", {}))

    def predict_point(self, features: pd.DataFrame) -> np.ndarray:
        """Retorna o preço estimado (ponto central) para cada linha de `features`."""
        state = self._state
        return state.point_pipeline.predict(features[state.feature_columns])

    def predict_interval(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (limite_inferior, limite_superior) para cada linha de `features`.

        Os dois modelos de quantil são treinados de forma independente (ver
        `src/train.py`), então os limites são ordenados aqui para nunca
        retornar um intervalo invertido — mesma salvaguarda de
        `src/predict.py`.
        """
        state = self._state
        cols = state.feature_columns
        lower = state.lower_pipeline.predict(features[cols])
        upper = state.upper_pipeline.predict(features[cols])
        return np.minimum(lower, upper), np.maximum(lower, upper)
