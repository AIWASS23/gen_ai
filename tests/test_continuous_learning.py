"""Testes do pipeline de retreino (`src/continuous_learning.py`).

Cobrem a lógica pura e rápida (split temporal, gate de qualidade,
backup/rollback de artefatos) com dados sintéticos — sem treinar nenhum
modelo de verdade. A validação com dados/modelo reais é feita manualmente
via `uv run python -m src.continuous_learning ...` (ver deploy/README.md e
docs/continuous_learning.md), não neste arquivo, para manter a suíte rápida.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.continuous_learning import (
    RetrainDecision,
    append_history,
    backup_current_champion,
    evaluate_gate,
    load_champion_metrics,
    rollback,
    time_based_split,
)


def _toy_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2015-01-01", "2015-01-05", "2015-01-10", "2015-01-16", "2015-01-25"]
    )
    return pd.DataFrame({"date": dates, "price": [100, 200, 300, 400, 500]})


def test_time_based_split_train_includes_cutoff_date() -> None:
    df = _toy_frame()
    # cutoff=01-10, holdout_days=10 -> janela (01-10, 01-20]: só 01-16 (400);
    # 01-25 fica fora da janela.
    train_df, holdout_df = time_based_split(df, pd.Timestamp("2015-01-10"), holdout_days=10)

    assert list(train_df["price"]) == [100, 200, 300]  # <= cutoff, inclusive
    assert list(holdout_df["price"]) == [400]


def test_time_based_split_respects_holdout_window() -> None:
    df = _toy_frame()
    # cutoff=01-10, holdout_days=5 -> janela (01-10, 01-15]: nenhuma data cai
    # aí (01-16 está 1 dia além do limite).
    train_df, holdout_df = time_based_split(df, pd.Timestamp("2015-01-10"), holdout_days=5)

    assert list(holdout_df["price"]) == []


def test_time_based_split_empty_holdout_when_no_future_data() -> None:
    df = _toy_frame()
    train_df, holdout_df = time_based_split(df, pd.Timestamp("2015-01-25"), holdout_days=10)

    assert len(train_df) == 5
    assert len(holdout_df) == 0


def test_evaluate_gate_promotes_without_previous_champion() -> None:
    decision = evaluate_gate(challenger_metrics={"rmse": 999_999.0}, champion_metrics=None, tolerance=0.02)
    assert decision.promoted is True
    assert "sem campeão" in decision.reason.lower()


def test_evaluate_gate_promotes_when_better_than_champion() -> None:
    decision = evaluate_gate(
        challenger_metrics={"rmse": 100_000.0}, champion_metrics={"rmse": 110_000.0}, tolerance=0.02
    )
    assert decision.promoted is True


def test_evaluate_gate_promotes_within_tolerance() -> None:
    # 2% pior que o campeão (110_000 * 1.02 = 112_200) ainda deve passar.
    decision = evaluate_gate(
        challenger_metrics={"rmse": 112_000.0}, champion_metrics={"rmse": 110_000.0}, tolerance=0.02
    )
    assert decision.promoted is True


def test_evaluate_gate_rejects_beyond_tolerance() -> None:
    decision = evaluate_gate(
        challenger_metrics={"rmse": 130_000.0}, champion_metrics={"rmse": 110_000.0}, tolerance=0.02
    )
    assert decision.promoted is False
    assert "acima da" in decision.reason.lower()


def test_evaluate_gate_is_deterministic_at_exact_threshold() -> None:
    # Exatamente no limite (110_000 * 1.02 = 112_200) deve promover (<=, não <).
    decision = evaluate_gate(
        challenger_metrics={"rmse": 112_200.0}, champion_metrics={"rmse": 110_000.0}, tolerance=0.02
    )
    assert decision.promoted is True


def _write_fake_artifacts(models_dir: Path, marker: str) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "model.joblib").write_text(f"model-{marker}")
    (models_dir / "quantile_lower.joblib").write_text(f"lower-{marker}")
    (models_dir / "quantile_upper.joblib").write_text(f"upper-{marker}")
    (models_dir / "feature_columns.json").write_text(json.dumps(["a", "b"]))
    (models_dir / "metrics.json").write_text(json.dumps({"holdout_test": {"rmse": 100.0}}))


def test_backup_and_rollback_roundtrip(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _write_fake_artifacts(models_dir, marker="v1")

    backup_current_champion(models_dir)
    assert (models_dir / "previous" / "model.joblib").read_text() == "model-v1"

    # simula uma promoção: sobrescreve os artefatos "em produção"
    _write_fake_artifacts(models_dir, marker="v2")
    assert (models_dir / "model.joblib").read_text() == "model-v2"

    rollback(models_dir)
    assert (models_dir / "model.joblib").read_text() == "model-v1"
    assert (models_dir / "quantile_lower.joblib").read_text() == "lower-v1"


def test_rollback_without_backup_raises(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        rollback(models_dir)


def test_load_champion_metrics_returns_none_without_file(tmp_path: Path) -> None:
    assert load_champion_metrics(tmp_path / "models") is None


def test_load_champion_metrics_reads_holdout_test(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _write_fake_artifacts(models_dir, marker="v1")
    assert load_champion_metrics(models_dir) == {"rmse": 100.0}


def test_append_history_writes_jsonl_line(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    decision = RetrainDecision(
        promoted=True, reason="ok", challenger_metrics={"rmse": 1.0}, champion_metrics=None
    )
    append_history(decision, pd.Timestamp("2015-01-01"), trigger="manual", models_dir=models_dir)

    lines = (models_dir / "retrain_history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["promoted"] is True
    assert entry["cutoff_date"] == "2015-01-01"
    assert entry["trigger"] == "manual"
