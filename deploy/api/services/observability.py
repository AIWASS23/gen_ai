"""Integração com Langfuse para observabilidade de previsões em produção.

Cada chamada de previsão é registrada como um *span* no Langfuse, contendo:
entrada (features do imóvel), saída (previsão + intervalo), latência,
versão do modelo e se a resposta veio do cache. Isso permite, em produção:

- Auditar decisões de precificação (qual versão do modelo gerou qual preço).
- Detectar drift observando a distribuição de inputs/outputs ao longo do tempo.
- Medir latência e cache hit rate por trace, sem instrumentação manual extra.

Ver `docs/deploy_strategy.md` para o desenho geral de monitoramento; este
módulo é a implementação de referência da camada de observabilidade.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Protocol

from langfuse import Langfuse

from deploy.api.config import Settings

logger = logging.getLogger(__name__)


class ObservationHandle(Protocol):
    """Interface mínima que um "span" de observação precisa oferecer.

    Implementada tanto pelo span real do Langfuse quanto por
    :class:`_NullObservationHandle`, permitindo que o código de negócio
    (`PredictionService`) chame `.update(...)`/`.score(...)` sem precisar
    checar se a observabilidade está habilitada (Null Object Pattern).
    """

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> Any: ...

    def score(self, *, name: str, value: float, data_type: str = "NUMERIC", comment: str | None = None) -> None: ...


class _NullObservationHandle:
    """Implementação no-op de :class:`ObservationHandle`, usada quando a
    observabilidade está desabilitada (`Settings.observability_enabled=False`)."""

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> "_NullObservationHandle":
        return self

    def score(self, *, name: str, value: float, data_type: str = "NUMERIC", comment: str | None = None) -> None:
        return None


class ObservabilityService:
    """Encapsula o ciclo de vida do cliente Langfuse e a criação de spans.

    Uma única instância é criada no startup da aplicação (ver
    `deploy/api/main.py`) e injetada via dependência nos routers que
    precisam rastrear operações.
    """

    def __init__(self, settings: Settings) -> None:
        """Inicializa o cliente Langfuse se `settings.observability_enabled`
        for verdadeiro e as credenciais estiverem presentes; caso contrário,
        opera em modo no-op (todas as chamadas viram `_NullObservationHandle`)."""
        self._enabled = bool(
            settings.observability_enabled
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        )
        self._client: Langfuse | None = None
        if self._enabled:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_host,
                flush_at=settings.langfuse_flush_at,
                flush_interval=settings.langfuse_flush_interval_seconds,
            )
            logger.info("Observabilidade Langfuse habilitada (host=%s)", settings.langfuse_host)
        else:
            logger.info("Observabilidade Langfuse desabilitada")

    @property
    def enabled(self) -> bool:
        """Indica se o rastreamento está ativo (credenciais presentes e habilitado)."""
        return self._enabled

    @contextmanager
    def trace_prediction(
        self, *, name: str, input_payload: Any, metadata: dict[str, Any] | None = None
    ) -> Iterator[ObservationHandle]:
        """Context manager que abre um span do tipo "span" no Langfuse.

        Uso:
            with observability.trace_prediction(
                name="predict_batch", input_payload=request.model_dump()
            ) as span:
                ... roda a previsão ...
                span.update(output=response.model_dump())
                span.score(name="latency_ms", value=latency_ms)

        Quando desabilitado, produz um :class:`_NullObservationHandle` — o
        código chamador não precisa saber a diferença.
        """
        if not self._enabled or self._client is None:
            yield _NullObservationHandle()
            return

        with self._client.start_as_current_observation(
            name=name, as_type="span", input=input_payload, metadata=metadata
        ) as span:
            try:
                yield span
            finally:
                self._client.flush()

    def shutdown(self) -> None:
        """Garante o envio de eventos pendentes ao encerrar a aplicação."""
        if self._client is not None:
            self._client.flush()
