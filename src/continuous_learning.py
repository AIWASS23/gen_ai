"""Pipeline de retreino (aprendizado contínuo).

Implementação de referência do que `docs/continuous_learning.md` descreve:
retreina o modelo campeão com dados "até uma certa data", avalia o
resultado (o "desafiante") em uma janela **out-of-time** (dados futuros que
o desafiante nunca viu — não um split aleatório) e só promove a produção
se o desafiante não regredir além de uma margem de tolerância em relação
ao campeão atual. Cada execução fica registrada em
`models/retrain_history.jsonl`, e a promoção sempre faz backup do campeão
anterior primeiro, permitindo rollback imediato.

Uso:
    # Avalia um retreino sem escrever nada (seguro para explorar):
    uv run python -m src.continuous_learning --cutoff-date 2015-04-15 --dry-run

    # Retreina, avalia e promove se passar no gate:
    uv run python -m src.continuous_learning --cutoff-date 2015-04-15 --trigger scheduled

    # Notifica a API para trocar o modelo em produção sem reiniciar
    # (usa o endpoint já implementado em deploy/api/routers/admin.py):
    uv run python -m src.continuous_learning --cutoff-date 2015-04-15 \\
        --notify-url http://localhost:8000 --admin-api-key SEGREDO

    # Desfaz a última promoção:
    uv run python -m src.continuous_learning --rollback

Este dataset é estático (não há vendas "chegando" de verdade depois de
2015-05-27), então `--cutoff-date` simula "o dia do retreino": tudo até
essa data é tratado como conhecido; os `--holdout-days` seguintes simulam
as vendas mais recentes, usadas só para avaliar o desafiante antes de
decidir promovê-lo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data import get_feature_columns, load_labeled_frame
from src.train import (
    QUANTILE_HIGH,
    QUANTILE_LOW,
    MODELS_DIR,
    _regression_metrics,
    build_quantile_pipeline,
    ensemble_models,
)

# Arquitetura atualmente campeã (ver models/metrics.json:"best_model").
# Retreino de rotina reajusta os pesos dessa mesma arquitetura com dados
# novos; re-comparar as 16 famílias de modelo do zero (src/train.py) é uma
# tarefa mais rara e mais cara — "revisão de arquitetura", não retreino.
CHAMPION_ARCHITECTURE = "stacking"

# Desafiante pode ser até 2% pior em RMSE (out-of-time) que o campeão e
# ainda ser promovido — absorve ruído de amostragem sem travar todo
# retreino que não é estritamente melhor. Ajustável via --tolerance.
DEFAULT_RMSE_TOLERANCE = 0.02

DEFAULT_HOLDOUT_DAYS = 14

ARTIFACT_NAMES = [
    "model.joblib",
    "quantile_lower.joblib",
    "quantile_upper.joblib",
    "feature_columns.json",
    "metrics.json",
]


@dataclass
class RetrainDecision:
    """Resultado do gate de qualidade para um desafiante retreinado."""

    promoted: bool
    reason: str
    challenger_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None


def time_based_split(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, holdout_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split **out-of-time** (não aleatório): treino = tudo até `cutoff_date`
    (inclusive); holdout = os `holdout_days` dias seguintes.

    Diferente do split aleatório agrupado usado na seleção inicial de
    modelo (`src/train.py`), este split respeita a ordem cronológica —
    o desafiante nunca vê dados "do futuro" em relação ao seu próprio
    treino, replicando o cenário real de produção.
    """
    train_df = df[df["date"] <= cutoff_date]
    holdout_end = cutoff_date + pd.Timedelta(days=holdout_days)
    holdout_df = df[(df["date"] > cutoff_date) & (df["date"] <= holdout_end)]
    return train_df, holdout_df


def _split_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_cols], df["price"]


def train_challenger(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[Any, Any, Any]:
    """Treina a arquitetura campeã (ponto + quantis) nos dados fornecidos."""
    X_train, y_train = _split_xy(train_df, feature_cols)

    point_pipeline = ensemble_models()[CHAMPION_ARCHITECTURE]
    point_pipeline.fit(X_train, y_train)

    lower_pipeline = build_quantile_pipeline(QUANTILE_LOW).fit(X_train, y_train)
    upper_pipeline = build_quantile_pipeline(QUANTILE_HIGH).fit(X_train, y_train)
    return point_pipeline, lower_pipeline, upper_pipeline


def evaluate_gate(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    tolerance: float,
) -> RetrainDecision:
    """Decide se o desafiante deve ser promovido a campeão.

    Promove se não houver campeão anterior (bootstrap), ou se o RMSE do
    desafiante não ultrapassar `champion_rmse * (1 + tolerance)`.
    """
    if champion_metrics is None:
        return RetrainDecision(
            promoted=True,
            reason="Sem campeão anterior registrado — promovendo o desafiante por padrão.",
            challenger_metrics=challenger_metrics,
            champion_metrics=None,
        )

    champion_rmse = champion_metrics["rmse"]
    challenger_rmse = challenger_metrics["rmse"]
    threshold = champion_rmse * (1 + tolerance)
    promoted = challenger_rmse <= threshold

    verdict = "dentro da" if promoted else "acima da"
    reason = (
        f"RMSE do desafiante (${challenger_rmse:,.0f}) {verdict} tolerância de "
        f"{tolerance:.0%} sobre o campeão (${champion_rmse:,.0f}; limite ${threshold:,.0f})."
    )
    return RetrainDecision(
        promoted=promoted,
        reason=reason,
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
    )


def load_champion_metrics(models_dir: Path = MODELS_DIR) -> dict[str, float] | None:
    """Lê as métricas de holdout **originais** do campeão (do treino inicial,
    em `metrics.json`) — só para referência/exibição. **Não** usar para o
    gate de promoção: foram calculadas num split aleatório diferente do
    holdout out-of-time do desafiante, então não são comparáveis
    diretamente (ver `evaluate_champion_on_holdout`)."""
    metrics_path = models_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text())
    return payload.get("holdout_test")


def evaluate_champion_on_holdout(
    holdout_df: pd.DataFrame, feature_cols: list[str], models_dir: Path = MODELS_DIR
) -> dict[str, float] | None:
    """Avalia o campeão **atualmente em produção** na mesma janela
    out-of-time usada para o desafiante — essencial para uma comparação
    justa (maçãs com maçãs). Comparar contra o RMSE gravado em
    `metrics.json` seria enviesado: aquele número vem de um split aleatório
    diferente (tipicamente mais fácil que um recorte temporal curto e
    recente, que é menor e mais sujeito a sazonalidade/composição de
    imóveis específica do período). Retorna `None` se não houver campeão
    (`model.joblib`) para carregar.

    **Limitação conhecida, descoberta rodando este pipeline contra o
    campeão real deste repositório**: o `models/model.joblib` publicado
    aqui foi treinado com **100% do histórico disponível** (é o entregável
    final do desafio — `src/train.py` reaproveita todo o dado disponível
    de propósito). Isso significa que ele já "viu" qualquer janela
    out-of-time escolhida dentro do intervalo do dataset — avaliá-lo nessa
    janela mede desempenho *dentro* do treino (vazamento), não fora, e o
    deixa artificialmente ótimo. Em produção de verdade esse problema não
    existe: o campeão só teria visto dados até a *sua própria* data de
    treino, sempre anterior a qualquer janela de avaliação futura. Para uma
    demonstração honesta e sem vazamento do gate, rode o ciclo contra um
    `--models-dir` isolado (não `models/`) — o primeiro retreino ali cria
    um campeão inicial genuinamente limitado a um corte antigo. Ver
    `docs/continuous_learning.md`.
    """
    model_path = models_dir / "model.joblib"
    if not model_path.exists():
        return None
    champion_pipeline = joblib.load(model_path)
    X_holdout, y_holdout = _split_xy(holdout_df, feature_cols)
    y_pred = champion_pipeline.predict(X_holdout)
    return _regression_metrics(y_holdout, y_pred)


def backup_current_champion(models_dir: Path = MODELS_DIR) -> None:
    """Copia os artefatos atuais para `models_dir/previous/`, permitindo rollback."""
    backup_dir = models_dir / "previous"
    backup_dir.mkdir(exist_ok=True)
    for name in ARTIFACT_NAMES:
        source = models_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)


def rollback(models_dir: Path = MODELS_DIR) -> None:
    """Restaura `models_dir/previous/` para `models_dir/` (desfaz a última promoção)."""
    backup_dir = models_dir / "previous"
    if not backup_dir.exists():
        raise FileNotFoundError(f"Nenhum backup em {backup_dir} para restaurar.")
    for name in ARTIFACT_NAMES:
        backup_path = backup_dir / name
        if backup_path.exists():
            shutil.copy2(backup_path, models_dir / name)


def promote(
    point_pipeline: Any,
    lower_pipeline: Any,
    upper_pipeline: Any,
    feature_cols: list[str],
    challenger_metrics: dict[str, float],
    cutoff_date: pd.Timestamp,
    n_train: int,
    n_holdout: int,
    models_dir: Path = MODELS_DIR,
) -> None:
    """Substitui o campeão em produção pelo desafiante (com backup automático)."""
    backup_current_champion(models_dir)

    joblib.dump(point_pipeline, models_dir / "model.joblib", compress=3)
    joblib.dump(lower_pipeline, models_dir / "quantile_lower.joblib", compress=3)
    joblib.dump(upper_pipeline, models_dir / "quantile_upper.joblib", compress=3)
    (models_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    metrics_path = models_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    payload["best_model"] = CHAMPION_ARCHITECTURE
    payload["holdout_test"] = challenger_metrics
    payload["retrain"] = {
        "cutoff_date": str(cutoff_date.date()),
        "n_train": n_train,
        "n_holdout": n_holdout,
        "retrained_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))


def append_history(
    decision: RetrainDecision, cutoff_date: pd.Timestamp, trigger: str, models_dir: Path = MODELS_DIR
) -> None:
    """Acrescenta uma linha de auditoria a `models_dir/retrain_history.jsonl`."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
        "cutoff_date": str(cutoff_date.date()),
        "promoted": decision.promoted,
        "reason": decision.reason,
        "challenger_metrics": decision.challenger_metrics,
        "champion_metrics": decision.champion_metrics,
    }
    history_path = models_dir / "retrain_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def notify_api_reload(api_url: str, admin_api_key: str) -> None:
    """Aciona `POST /v1/admin/reload-model` para trocar o modelo em produção
    sem reiniciar o processo (endpoint implementado em
    `deploy/api/routers/admin.py`) — fecha o ciclo "promovido -> em produção"."""
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/v1/admin/reload-model",
        method="POST",
        headers={"X-Admin-Api-Key": admin_api_key},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        print(f"API notificada: HTTP {response.status} — {response.read().decode()}")


def run(
    cutoff_date: str,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    tolerance: float = DEFAULT_RMSE_TOLERANCE,
    trigger: str = "manual",
    dry_run: bool = False,
    notify_url: str | None = None,
    admin_api_key: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> RetrainDecision:
    """Executa um ciclo completo de retreino: split out-of-time, treino do
    desafiante, avaliação, gate de qualidade e (se aprovado) promoção."""
    cutoff = pd.Timestamp(cutoff_date)
    full = load_labeled_frame()
    feature_cols = get_feature_columns(full)

    train_df, holdout_df = time_based_split(full, cutoff, holdout_days)
    if len(holdout_df) == 0:
        raise ValueError(
            f"Nenhum dado entre {cutoff.date()} e +{holdout_days} dias — "
            "ajuste --cutoff-date/--holdout-days para uma janela com dados."
        )
    print(f"Treino (até {cutoff.date()}): {len(train_df)} linhas | Holdout out-of-time: {len(holdout_df)} linhas")

    point_pipeline, lower_pipeline, upper_pipeline = train_challenger(train_df, feature_cols)

    X_holdout, y_holdout = _split_xy(holdout_df, feature_cols)
    y_pred = point_pipeline.predict(X_holdout)
    challenger_metrics = _regression_metrics(y_holdout, y_pred)
    print("Métricas do desafiante (out-of-time):", challenger_metrics)

    # Comparação maçãs-com-maçãs: avalia o campeão atual na MESMA janela
    # out-of-time do desafiante — comparar contra o RMSE gravado em
    # metrics.json seria injusto (aquele número vem de um split aleatório
    # diferente, ver evaluate_champion_on_holdout).
    champion_metrics = evaluate_champion_on_holdout(holdout_df, feature_cols, models_dir)
    original_champion_metrics = load_champion_metrics(models_dir)
    if champion_metrics is not None:
        print("Métricas do campeão atual (mesma janela out-of-time):", champion_metrics)
        if original_champion_metrics is not None:
            print("(Para referência, RMSE original do campeão no split aleatório do treino:", f"${original_champion_metrics['rmse']:,.0f})")

    decision = evaluate_gate(challenger_metrics, champion_metrics, tolerance)
    print("Gate:", decision.reason)

    append_history(decision, cutoff, trigger, models_dir)

    if decision.promoted and dry_run:
        print("[--dry-run] Desafiante seria promovido, mas nada foi escrito em disco.")
    elif decision.promoted:
        promote(
            point_pipeline,
            lower_pipeline,
            upper_pipeline,
            feature_cols,
            challenger_metrics,
            cutoff,
            len(train_df),
            len(holdout_df),
            models_dir,
        )
        print(f"Promovido: {models_dir} atualizado, backup do campeão anterior em {models_dir / 'previous'}.")
        if notify_url:
            notify_api_reload(notify_url, admin_api_key or "")
    else:
        print("Rejeitado: campeão atual mantido, nada foi escrito em disco.")

    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff-date", help="Data (YYYY-MM-DD) até a qual os dados são tratados como conhecidos.")
    parser.add_argument(
        "--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS,
        help=f"Janela out-of-time (em dias) usada para avaliar o desafiante (padrão: {DEFAULT_HOLDOUT_DAYS}).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_RMSE_TOLERANCE,
        help=f"Margem de tolerância de RMSE para promoção, ex.: 0.02 = 2%% (padrão: {DEFAULT_RMSE_TOLERANCE}).",
    )
    parser.add_argument(
        "--trigger", default="manual", choices=["manual", "scheduled", "drift"],
        help="Motivo do disparo do retreino, só para o log de auditoria.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Avalia o desafiante mas não promove/escreve nada.")
    parser.add_argument("--notify-url", default=None, help="Se definido, chama POST {url}/v1/admin/reload-model após promover.")
    parser.add_argument("--admin-api-key", default=None, help="X-Admin-Api-Key para --notify-url.")
    parser.add_argument("--rollback", action="store_true", help="Desfaz a última promoção (restaura previous/).")
    parser.add_argument(
        "--models-dir", default=None,
        help="Diretório de artefatos a usar em vez de models/ (ex.: para rodar uma demonstração isolada, sem tocar no modelo real em produção).",
    )
    args = parser.parse_args()
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR

    if args.rollback:
        rollback(models_dir)
        print(f"Rollback concluído: {models_dir / 'previous'} restaurado para {models_dir}.")
        return

    if not args.cutoff_date:
        parser.error("--cutoff-date é obrigatório (exceto com --rollback).")

    run(
        cutoff_date=args.cutoff_date,
        holdout_days=args.holdout_days,
        tolerance=args.tolerance,
        trigger=args.trigger,
        dry_run=args.dry_run,
        notify_url=args.notify_url,
        admin_api_key=args.admin_api_key,
        models_dir=models_dir,
    )


if __name__ == "__main__":
    main()
