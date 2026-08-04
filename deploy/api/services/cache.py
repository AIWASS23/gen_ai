"""Cache de previsões em Redis.

Evita recomputar a mesma previsão quando o mesmo imóvel (mesmas 18
features) é consultado repetidamente — comum quando várias telas/serviços
consultam o preço estimado do mesmo anúncio. A chave de cache inclui a
versão do modelo: uma troca de modelo (`ModelRegistry.reload()`) invalida
implicitamente todo o cache anterior, sem precisar limpá-lo manualmente.
"""

from __future__ import annotations

import hashlib
import json
import logging

import redis.asyncio as redis

from deploy.api.schemas.requests import HouseFeatures
from deploy.api.schemas.responses import PricePrediction

logger = logging.getLogger(__name__)


class PredictionCache:
    """Cache assíncrono de previsões, com degradação graciosa.

    Se o Redis estiver indisponível, `get`/`set` logam um aviso e se
    comportam como cache miss/no-op em vez de propagar a exceção — a API
    deve continuar servindo previsões (só mais lentamente, sem cache) mesmo
    se o Redis cair.
    """

    def __init__(self, client: redis.Redis, key_prefix: str, ttl_seconds: int) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _hash_house(house: HouseFeatures) -> str:
        payload = json.dumps(house.model_dump(exclude={"request_id"}), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _make_key(self, house: HouseFeatures, model_version: str) -> str:
        return f"{self._key_prefix}:{model_version}:{self._hash_house(house)}"

    async def get(self, house: HouseFeatures, model_version: str) -> PricePrediction | None:
        """Retorna a previsão cacheada para `house`, ou `None` em cache miss/erro."""
        key = self._make_key(house, model_version)
        try:
            raw = await self._client.get(key)
        except redis.RedisError as exc:
            logger.warning("Falha ao ler do cache Redis (key=%s): %s", key, exc)
            return None
        if raw is None:
            return None
        prediction = PricePrediction.model_validate_json(raw)
        return prediction.model_copy(update={"cached": True})

    async def set(self, house: HouseFeatures, model_version: str, prediction: PricePrediction) -> None:
        """Grava a previsão no cache com o TTL configurado. Falhas são apenas logadas."""
        key = self._make_key(house, model_version)
        try:
            await self._client.set(key, prediction.model_dump_json(), ex=self._ttl_seconds)
        except redis.RedisError as exc:
            logger.warning("Falha ao gravar no cache Redis (key=%s): %s", key, exc)

    async def ping(self) -> bool:
        """Usado pelo endpoint de health check para reportar o status do Redis."""
        try:
            return bool(await self._client.ping())
        except redis.RedisError:
            return False

    async def close(self) -> None:
        await self._client.aclose()
