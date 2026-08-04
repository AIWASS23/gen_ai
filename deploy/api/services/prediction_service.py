"""Orquestra features, cache, modelo e observabilidade para servir previsões.

Esta classe é o único ponto que o router `predictions.py` chama — todo o
"como" (construir features, checar cache, invocar o modelo, registrar a
trace) fica encapsulado aqui, mantendo o router fino (só HTTP: parsing,
status codes, injeção de dependências).
"""

from __future__ import annotations

import asyncio
import time
from typing import Sequence

from deploy.api.schemas.requests import HouseFeatures, PredictionRequest
from deploy.api.schemas.responses import PredictionResponse, PricePrediction
from deploy.api.services.cache import PredictionCache
from deploy.api.services.feature_builder import FeatureBuilder
from deploy.api.services.model_registry import ModelRegistry
from deploy.api.services.observability import ObservabilityService


class BatchSizeExceededError(ValueError):
    """Levantado quando o número de imóveis no request excede `max_batch_size`."""

    def __init__(self, requested: int, allowed: int) -> None:
        self.requested = requested
        self.allowed = allowed
        super().__init__(f"Lote de {requested} imóveis excede o máximo permitido de {allowed}")


class PredictionService:
    """Caso de uso "prever preços de um lote de imóveis", ponta a ponta."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        feature_builder: FeatureBuilder,
        observability: ObservabilityService,
        cache: PredictionCache | None,
        max_batch_size: int,
    ) -> None:
        self._model_registry = model_registry
        self._feature_builder = feature_builder
        self._observability = observability
        self._cache = cache
        self._max_batch_size = max_batch_size

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Retorna uma :class:`PredictionResponse` para o lote de imóveis do request.

        Fluxo:
            1. Valida o tamanho do lote.
            2. Consulta o cache (se habilitado) para cada imóvel, em paralelo.
            3. Para os que não estavam em cache, constrói as features e chama
               o modelo em lote (uma única chamada ao pipeline para todo o
               sub-lote, não uma por imóvel — muito mais eficiente).
            4. Grava os novos resultados no cache.
            5. Registra a operação inteira como um span no Langfuse.
        """
        if len(request.houses) > self._max_batch_size:
            raise BatchSizeExceededError(len(request.houses), self._max_batch_size)

        started_at = time.perf_counter()
        version = self._model_registry.version

        with self._observability.trace_prediction(
            name="predict_batch",
            input_payload={"request_id": request.request_id, "n_houses": len(request.houses)},
            metadata={"model_version": version},
        ) as span:
            predictions = await self._predict_with_cache(request.houses, version)
            latency_ms = (time.perf_counter() - started_at) * 1000

            response = PredictionResponse(
                request_id=request.request_id,
                model_version=version,
                predictions=predictions,
                latency_ms=latency_ms,
            )

            cache_hits = sum(1 for p in predictions if p.cached)
            span.update(output=response.model_dump(mode="json"))
            span.score(name="latency_ms", value=latency_ms)
            span.score(name="cache_hit_rate", value=cache_hits / len(predictions) if predictions else 0.0)

        return response

    async def _predict_with_cache(
        self, houses: Sequence[HouseFeatures], version: str
    ) -> list[PricePrediction]:
        results: list[PricePrediction | None] = [None] * len(houses)

        if self._cache is not None:
            cached_results = await asyncio.gather(*(self._cache.get(h, version) for h in houses))
            for i, cached in enumerate(cached_results):
                results[i] = cached

        miss_indices = [i for i, r in enumerate(results) if r is None]
        if miss_indices:
            missed_houses = [houses[i] for i in miss_indices]
            # A inferência é CPU-bound (pandas/scikit-learn, síncrona); roda
            # em thread separada para não bloquear o event loop enquanto
            # outras requisições estão em andamento.
            fresh = await asyncio.to_thread(self._predict_fresh, missed_houses)
            if self._cache is not None:
                await asyncio.gather(
                    *(self._cache.set(h, version, p) for h, p in zip(missed_houses, fresh))
                )
            for i, prediction in zip(miss_indices, fresh):
                results[i] = prediction

        return [r for r in results if r is not None]  # sempre populado nesse ponto; narrows o tipo

    def _predict_fresh(self, houses: Sequence[HouseFeatures]) -> list[PricePrediction]:
        """Roda o modelo (síncrono, CPU-bound) para imóveis fora do cache."""
        features = self._feature_builder.build(list(houses))
        point = self._model_registry.predict_point(features)
        low, high = self._model_registry.predict_interval(features)
        return [
            PricePrediction(
                predicted_price=float(point[i]),
                predicted_price_low=float(low[i]),
                predicted_price_high=float(high[i]),
                cached=False,
            )
            for i in range(len(houses))
        ]
