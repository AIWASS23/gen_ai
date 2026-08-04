"""Configuração da aplicação (padrão 12-factor: tudo vem de variáveis de ambiente).

Nenhum outro módulo deve ler `os.environ` diretamente — todo acesso a
configuração passa por uma instância de :class:`Settings`, injetada via
:func:`get_settings` (ver `deploy/api/dependencies.py`). Isso torna a
configuração testável (basta injetar um `Settings` diferente em testes) e
documentada em um único lugar.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do repositório (deploy/api/config.py -> deploy/api -> deploy -> raiz).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Configuração tipada da API de previsão de preços.

    Todos os campos podem ser sobrescritos por variáveis de ambiente com o
    prefixo ``APP_`` (ex.: ``APP_REDIS_URL``), ou por um arquivo ``.env`` na
    raiz do projeto. Valores default assumem execução local fora de
    container, com o repositório completo disponível no filesystem.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Identidade do serviço -------------------------------------------------
    app_name: str = "house-price-prediction-api"
    api_version: str = "v1"
    environment: Literal["development", "staging", "production"] = "development"

    # --- Artefatos do modelo -----------------------------------------------
    model_path: Path = REPO_ROOT / "models" / "model.joblib"
    quantile_lower_path: Path = REPO_ROOT / "models" / "quantile_lower.joblib"
    quantile_upper_path: Path = REPO_ROOT / "models" / "quantile_upper.joblib"
    feature_columns_path: Path = REPO_ROOT / "models" / "feature_columns.json"
    metrics_path: Path = REPO_ROOT / "models" / "metrics.json"
    demographics_path: Path = REPO_ROOT / "data" / "raw" / "zipcode_demographics.csv"

    # --- Batch máximo aceito por requisição ---------------------------------
    max_batch_size: int = Field(default=500, gt=0, le=5000)

    # --- Redis (cache de previsões) -----------------------------------------
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    redis_socket_timeout_seconds: float = 2.0
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    cache_key_prefix: str = "house-price"

    # --- Langfuse (observabilidade) -----------------------------------------
    observability_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_flush_at: int = 15
    langfuse_flush_interval_seconds: float = 5.0

    # --- HTTP / CORS ---------------------------------------------------------
    cors_allow_origins: list[str] = ["*"]
    request_timeout_seconds: float = 10.0

    # --- Logging ---------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Administração (hot-reload de modelo) -------------------------------
    # Sem chave configurada, o endpoint de reload fica desabilitado (falha
    # fechado): mais seguro do que expor um endpoint administrativo sem
    # autenticação por omissão.
    admin_api_key: str | None = None

    @property
    def is_production(self) -> bool:
        """Atalho para checagens condicionais (ex.: docs habilitadas só fora de produção)."""
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância singleton de :class:`Settings`.

    Cacheada com ``lru_cache`` para que o arquivo `.env`/ambiente seja lido
    uma única vez por processo. Em testes, sobrescreva a dependência do
    FastAPI (`app.dependency_overrides[get_settings] = ...`) em vez de
    limpar este cache.
    """
    return Settings()
