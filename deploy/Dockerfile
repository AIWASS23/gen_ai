# syntax=docker/dockerfile:1
#
# Imagem de serving da API de previsão de preços de imóveis.
#
# Build (a partir da raiz do repositório, o contexto precisa incluir src/,
# models/ e data/raw/ além de deploy/):
#   docker build -f deploy/Dockerfile -t house-price-api:latest .
#
# Multi-stage: o estágio "builder" instala as dependências Python em
# /install; o estágio final ("runtime") copia só o resultado + o código e
# os artefatos necessários, sem toolchain de build nem cache do pip.

FROM python:3.12-slim AS builder

WORKDIR /build

# libgomp1: OpenMP runtime exigido por xgboost/lightgbm/catboost em tempo de
# execução (também instalado no estágio final, abaixo).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt ./requirements.txt

# xgboost é instalado à parte, com --no-deps: o pacote do PyPI traz
# nvidia-nccl-cu12 (~400MB) como dependência obrigatória no Linux, usada
# apenas para treino distribuído multi-GPU — não é exercitada em inferência
# CPU-only (validado: o pipeline treinado carrega e prevê normalmente sem
# essa lib instalada). numpy/scipy, os únicos deps de runtime que o xgboost
# realmente usa, já vêm via scikit-learn/pandas na linha de baixo.
RUN grep -v '^xgboost==' requirements.txt > requirements.no-xgboost.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements.no-xgboost.txt \
    && pip install --no-cache-dir --prefix=/install --no-deps \
        "$(grep '^xgboost==' requirements.txt)"


FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /install /usr/local

WORKDIR /app

# Só o necessário para servir — não o repositório inteiro (sem notebooks/,
# reports/, dados de treino brutos, etc.), para manter a imagem enxuta.
COPY src/__init__.py src/data.py ./src/
COPY deploy/__init__.py ./deploy/__init__.py
COPY deploy/api/ ./deploy/api/
COPY models/ ./models/
COPY data/raw/zipcode_demographics.csv ./data/raw/zipcode_demographics.csv

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENVIRONMENT=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8000/health/live || exit 1

# --workers 2: paralelismo de processos para aproveitar múltiplos cores em
# ambientes sem orquestrador de replicas; em Kubernetes (deploy/k8s/), o
# escalonamento horizontal via HPA é preferível a mais workers por pod.
CMD ["uvicorn", "deploy.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
