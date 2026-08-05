"""Carregamento, limpeza e engenharia de features dos dados de imóveis.

Este módulo é compartilhado pelo treino (`src/train.py`) e pela predição
(`src/predict.py`) para garantir que exatamente a mesma transformação seja
aplicada nos dois momentos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

HOUSE_DATA_PATH = RAW_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = RAW_DIR / "zipcode_demographics.csv"
FUTURE_EXAMPLES_PATH = RAW_DIR / "future_unseen_examples.csv"

# Ano de referência usado para calcular a idade do imóvel. `future_unseen_examples.csv`
# não traz uma data de venda, então usamos o último ano observado nos dados de
# treino como referência fixa, aplicada de forma idêntica em treino e inferência.
REFERENCE_YEAR = 2015

# Colunas de identificação/target que não entram como feature do modelo.
NON_FEATURE_COLUMNS = ["id", "date", "price"]

# Um único registro com bedrooms=33 e apenas 1.620 sqft de área construída é
# claramente um erro de digitação (identificado na EDA, ver notebooks/01_eda.ipynb).
BEDROOMS_OUTLIER_THRESHOLD = 15


def load_house_sales(path: Path = HOUSE_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    return df


def load_demographics(path: Path = DEMOGRAPHICS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_future_examples(path: Path = FUTURE_EXAMPLES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_house_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros com erro evidente de captura de dados."""
    cleaned = df[df["bedrooms"] < BEDROOMS_OUTLIER_THRESHOLD].copy()
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas a partir das colunas físicas do imóvel.

    Aplicável tanto aos dados de treino quanto aos dados futuros (sem `date`).
    """
    out = df.copy()
    out["house_age"] = REFERENCE_YEAR - out["yr_built"]
    out["was_renovated"] = (out["yr_renovated"] > 0).astype(int)
    out["years_since_renovation"] = out.apply(
        lambda r: REFERENCE_YEAR - r["yr_renovated"] if r["yr_renovated"] > 0 else r["house_age"],
        axis=1,
    )
    return out


def add_demographics(df: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(demographics, on="zipcode", how="left", validate="many_to_one")
    return merged


def build_feature_frame(
    house_df: pd.DataFrame, demographics: pd.DataFrame
) -> pd.DataFrame:
    """Pipeline completo de preparação: engenharia de features + merge demográfico."""
    df = engineer_features(house_df)
    df = add_demographics(df, demographics)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def load_labeled_frame() -> pd.DataFrame:
    """Retorna o dataframe completo (id, date, price, features) já limpo,
    com engenharia de features e merge demográfico aplicados — sem
    descartar `date`/`id`.

    Base tanto para `load_training_frame` (treino "de uma vez", ignora
    `date`) quanto para `src/continuous_learning.py` (retreino), que
    precisa de `date` para fazer splits temporais (out-of-time).
    """
    houses = load_house_sales()
    houses = clean_house_sales(houses)
    demographics = load_demographics()
    return build_feature_frame(houses, demographics)


def load_training_frame() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Retorna (X, y, groups) prontos para split/treino.

    `groups` é o `id` do imóvel, usado para evitar que o mesmo imóvel (vendido
    mais de uma vez na janela do dataset) apareça simultaneamente em treino e
    teste.
    """
    full = load_labeled_frame()
    feature_cols = get_feature_columns(full)

    X = full[feature_cols]
    y = full["price"]
    groups = full["id"]
    return X, y, groups
