"""Treino e seleção do modelo de previsão de preços de imóveis.

Uso:
    uv run python -m src.train

Os modelos candidatos são organizados em dois grupos (`non_ensemble_models` e
`ensemble_models`), comparados separadamente nos relatórios, mas competindo
juntos pela escolha do modelo final.

Saídas:
    models/model.joblib           -> pipeline (pré-processamento + modelo) treinado, ponto estimado
    models/quantile_lower.joblib  -> pipeline que estima o percentil 10 do preço (limite inferior)
    models/quantile_upper.joblib  -> pipeline que estima o percentil 90 do preço (limite superior)
    models/metrics.json           -> métricas de comparação e do modelo final
    models/feature_columns.json   -> lista de features esperadas pelo modelo
    reports/model_comparison_non_ensemble.png
    reports/model_comparison_ensemble.png
    reports/feature_importance.png
    reports/actual_vs_predicted.png
    reports/prediction_interval.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.data import load_training_frame

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42
N_SPLITS = 5
QUANTILE_LOW = 0.1
QUANTILE_HIGH = 0.9


def _cast_zipcode_to_int(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost exige que features categóricas sejam int/string, não float.

    A imputação anterior no pipeline converte todas as colunas para float64;
    aqui devolvemos `zipcode` ao tipo inteiro para que o CatBoost a trate
    como categoria (e não como uma variável contínua sem sentido ordinal).
    """
    X = X.copy()
    X["zipcode"] = X["zipcode"].astype(int)
    return X


def _regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        ]
    )


def non_ensemble_models() -> dict[str, Pipeline]:
    """Modelos de base: um único preditor, sem combinar vários modelos.

    - Regressão Linear: baseline simples e interpretável, precisa de
      padronização das variáveis (escalas muito diferentes: sqft_lot na casa
      dos milhares, waterfront é 0/1).
    - Ridge: mesma família da Regressão Linear, mas com regularização L2 —
      útil aqui porque há colinearidade forte entre variáveis de área
      (sqft_living, sqft_above, sqft_living15), o que deixa a Regressão
      Linear pura instável.
    - Lasso: regularização L1 em vez de L2 — além de encolher coeficientes,
      zera os menos relevantes (seleção de features embutida). Serve para
      checar se alguma das features atuais é dispensável.
    - KNN: não-linear e não-paramétrico; captura diretamente o efeito de
      "imóveis parecidos e próximos tendem a ter preço parecido" via
      lat/long, sem precisar de nenhuma suposição de forma funcional.
      Precisa de padronização (é baseado em distância).
    - Decision Tree: uma única árvore de decisão — não-linear, mas sem o
      "voto da maioria" que estabiliza os ensembles de árvore. Serve de
      referência para medir o quanto os ensembles (Random Forest, boosting)
      realmente ganham sobre uma única árvore.
    - MLP (rede neural pequena): outra família de modelo não-linear, que
      aprende a função por camadas densas + backpropagation em vez de
      regras de corte como as árvores. Precisa de padronização.
    """
    return {
        "linear_regression": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "lasso": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Lasso(alpha=1.0, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _ensemble_base_estimators() -> list[tuple[str, object]]:
    """Estimadores de base compartilhados por Voting e Stacking.

    Usar exatamente os mesmos 3 modelos nos dois torna a comparação justa:
    a única diferença entre eles passa a ser *como* as previsões são
    combinadas (média simples vs. meta-modelo aprendido), não quais modelos
    entram na conta.
    """
    return [
        (
            "xgboost",
            XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "lightgbm",
            LGBMRegressor(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def ensemble_models() -> dict[str, Pipeline]:
    """Modelos que combinam várias previsões (bagging, boosting ou blending).

    - Random Forest: bagging de árvores — cada árvore treina numa amostra
      bootstrap diferente e o resultado é a média; robusto a outliers/escala,
      não precisa de padronização.
    - Extra Trees: variante do Random Forest com ainda mais aleatoriedade
      (os pontos de corte de cada nó também são sorteados, não só a melhor
      divisão) — costuma treinar mais rápido e reduzir variância um pouco
      mais.
    - AdaBoost: boosting clássico, anterior ao gradient boosting — treina
      árvores rasas em sequência, reponderando as amostras para que cada
      nova árvore foque mais nos erros da anterior. Incluído como referência
      histórica da família de boosting.
    - Gradient Boosting (scikit-learn): gradient boosting "clássico", árvore
      por árvore, sem o truque de histograma. Permite comparar diretamente
      contra HistGradientBoosting/XGBoost/LightGBM, que usam histograma para
      acelerar o treino.
    - HistGradientBoosting: gradient boosting nativo do scikit-learn,
      histograma-based (mesma família de ideia do LightGBM), rápido mesmo
      com mais dados e sem dependência externa.
    - XGBoost: gradient boosting, geralmente um dos melhores desempenhos em
      dados tabulares deste tipo, também não precisa de padronização.
    - LightGBM: outra implementação de gradient boosting (histograma +
      crescimento leaf-wise), tende a ser mais rápida que o XGBoost e é um
      concorrente direto natural para confirmar se o ganho do XGBoost é
      robusto ou específico da implementação.
    - CatBoost: gradient boosting com suporte nativo a features categóricas.
      Diferente dos demais, trata `zipcode` como categoria (não como número
      contínuo), o que pode capturar efeitos de bairro que lat/long e as
      variáveis demográficas não cobrem sozinhas.
    - Voting: combina XGBoost + LightGBM + Random Forest por média simples
      das previsões, sem meta-modelo — comparação direta e mais barata
      contra o Stacking, para checar se vale a pena o custo extra de
      aprender os pesos da combinação.
    - Stacking: combina as mesmas 3 previsões (XGBoost, LightGBM, Random
      Forest) através de um meta-modelo linear (RidgeCV) treinado sobre
      elas — em vez de pesos fixos como no Voting, aprende a melhor
      combinação a partir dos dados.
    """
    return {
        "random_forest": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=150,
                        # max_depth=None (padrão do sklearn) deixa as árvores
                        # crescerem até cada folha ser pura, o que infla o
                        # artefato serializado para centenas de MB neste
                        # dataset sem ganho de RMSE mensurável — limitamos a
                        # profundidade por tamanho de artefato, não só por
                        # performance (testado: ~1000 USD de RMSE de
                        # diferença, mas ~14x menor em disco).
                        max_depth=12,
                        min_samples_leaf=4,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=4,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "adaboost": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    AdaBoostRegressor(
                        estimator=DecisionTreeRegressor(max_depth=4),
                        n_estimators=200,
                        learning_rate=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=500,
                        max_depth=6,
                        learning_rate=0.05,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=500,
                        max_depth=-1,
                        num_leaves=31,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
                ("cast_zipcode", FunctionTransformer(_cast_zipcode_to_int)),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=500,
                        depth=6,
                        learning_rate=0.05,
                        l2_leaf_reg=3.0,
                        cat_features=["zipcode"],
                        random_state=RANDOM_STATE,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        ),
        "voting": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    VotingRegressor(estimators=_ensemble_base_estimators(), n_jobs=-1),
                ),
            ]
        ),
        "stacking": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    # Nota: a validação cruzada interna do StackingRegressor
                    # (usada para gerar as previsões out-of-fold que treinam
                    # o meta-modelo) usa KFold padrão, não agrupado por id de
                    # imóvel. O possível vazamento fica restrito aos ~0,8%
                    # de imóveis vendidos duas vezes (ver EDA) e apenas
                    # dentro do treino do meta-modelo — não afeta a divisão
                    # treino/teste externa, que continua agrupada.
                    StackingRegressor(
                        estimators=_ensemble_base_estimators(),
                        final_estimator=RidgeCV(),
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def candidate_models() -> dict[str, Pipeline]:
    """União dos dois grupos, para funções que precisam olhar todos os modelos."""
    return {**non_ensemble_models(), **ensemble_models()}


def model_categories() -> dict[str, str]:
    """Mapa nome do modelo -> "non_ensemble" ou "ensemble", para relatórios separados."""
    categories = {name: "non_ensemble" for name in non_ensemble_models()}
    categories.update({name: "ensemble" for name in ensemble_models()})
    return categories


def cross_validate_models(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """Compara os modelos candidatos com GroupKFold (agrupado por id do imóvel)."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    categories = model_categories()
    rows = []
    for name in candidate_models():
        fold_metrics = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            model = candidate_models()[name]
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            fold_metrics.append(_regression_metrics(y.iloc[val_idx], preds))
        avg = pd.DataFrame(fold_metrics).mean().to_dict()
        avg["model"] = name
        avg["category"] = categories[name]
        rows.append(avg)
    return pd.DataFrame(rows).set_index("model")


def tune_best_model(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_name: str
) -> Pipeline:
    """Busca leve de hiperparâmetros para o modelo escolhido."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    base_pipeline = candidate_models()[model_name]

    if model_name == "xgboost":
        param_distributions = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        }
    elif model_name == "random_forest":
        param_distributions = {
            "model__n_estimators": [150, 200, 300, 500],
            # sem max_depth=None: profundidade ilimitada infla o artefato
            # serializado para centenas de MB sem ganho de RMSE mensurável.
            "model__max_depth": [8, 12, 16, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5, 1.0],
        }
    elif model_name == "lightgbm":
        param_distributions = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__num_leaves": [15, 31, 63, 127],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        }
    elif model_name == "hist_gradient_boosting":
        param_distributions = {
            "model__max_iter": [200, 400, 600, 800],
            "model__max_depth": [None, 4, 6, 8],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        }
    elif model_name == "ridge":
        param_distributions = {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
        }
    elif model_name == "knn":
        param_distributions = {
            "model__n_neighbors": [5, 10, 15, 25, 40],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    elif model_name == "catboost":
        param_distributions = {
            "model__iterations": [200, 400, 600, 800],
            "model__depth": [4, 6, 8, 10],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "model__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
        }
    elif model_name == "lasso":
        param_distributions = {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        }
    elif model_name == "decision_tree":
        param_distributions = {
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
        }
    elif model_name == "extra_trees":
        param_distributions = {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [8, 12, 16, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5, 1.0],
        }
    elif model_name == "adaboost":
        param_distributions = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5],
        }
    elif model_name == "gradient_boosting":
        param_distributions = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
        }
    else:
        # "mlp": custo/instabilidade de tunar uma rede neural neste
        # orçamento de tempo não compensa, já não é candidata a vencer.
        # "voting": não tem hiperparâmetro próprio além dos modelos de base
        # (já escolhidos em _ensemble_base_estimators).
        # "stacking": tunar um ensemble de 3 modelos é caro (cada fit
        # re-treina os 3 estimadores de base); usamos os hiperparâmetros de
        # base (já próximos dos ótimos individuais) sem busca adicional.
        return base_pipeline.fit(X, y)

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=list(gkf.split(X, y, groups)),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_


def plot_model_comparison(cv_results: pd.DataFrame, category: str, label: str) -> None:
    """Gera um gráfico de comparação só com os modelos de uma categoria
    ("non_ensemble" ou "ensemble"), para não misturar na mesma escala
    modelos com RMSE muito diferente (ex.: KNN vs. XGBoost)."""
    subset = cv_results[cv_results["category"] == category]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(subset) + 2))
    subset["rmse"].sort_values().plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"RMSE médio (validação cruzada) — {label}")
    axes[0].set_xlabel("RMSE (USD)")

    subset["r2"].sort_values().plot.barh(ax=axes[1], color="#55A868")
    axes[1].set_title(f"R² médio (validação cruzada) — {label}")
    axes[1].set_xlabel("R²")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"model_comparison_{category}.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> None:
    model = pipeline.named_steps["model"]
    title = "Top 15 features mais importantes"

    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif isinstance(model, (StackingRegressor, VotingRegressor)):
        # Stacking/Voting não têm feature_importances_ própria: usamos a
        # média das importâncias normalizadas dos estimadores de base que a
        # possuem (xgboost, lightgbm, random_forest).
        per_model = [
            est.feature_importances_ / est.feature_importances_.sum()
            for est in model.estimators_
            if hasattr(est, "feature_importances_")
        ]
        if not per_model:
            return
        raw_importances = np.mean(per_model, axis=0)
        kind = "stacking" if isinstance(model, StackingRegressor) else "voting"
        title = f"Top 15 features mais importantes (média dos modelos base do {kind})"
    else:
        return

    importances = pd.Series(raw_importances, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel("Importância (gain relativo)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_true, y_pred, s=8, alpha=0.4, color="#4C72B0")
    lims = [0, max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Preço real (USD)")
    axes[0].set_ylabel("Preço previsto (USD)")
    axes[0].set_title("Real vs. Previsto (conjunto de teste)")

    residuals = y_pred - y_true
    sns.histplot(residuals, bins=50, ax=axes[1], color="#C44E52")
    axes[1].set_title("Distribuição dos resíduos (previsto - real)")
    axes[1].set_xlabel("Resíduo (USD)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)


def build_quantile_pipeline(alpha: float) -> Pipeline:
    """Pipeline LightGBM treinado para estimar um percentil do preço (não a média).

    Usado para gerar um intervalo de previsão (ex.: p10–p90) em vez de um
    único ponto — ver `docs/stakeholder_communication.md`, que recomenda
    comunicar uma faixa de preço em vez de um valor exato.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=500,
                    num_leaves=31,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def plot_prediction_interval(
    y_true: pd.Series, lower: np.ndarray, point: np.ndarray, upper: np.ndarray
) -> None:
    order = np.argsort(y_true.values)
    y_sorted = y_true.values[order]
    lower_sorted, point_sorted, upper_sorted = lower[order], point[order], upper[order]
    x = np.arange(len(y_sorted))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(
        x, lower_sorted, upper_sorted, color="#4C72B0", alpha=0.25, label="Intervalo p10–p90"
    )
    ax.plot(x, point_sorted, color="#4C72B0", linewidth=1, label="Previsão (ponto)")
    ax.scatter(x, y_sorted, color="#C44E52", s=4, alpha=0.5, label="Preço real")
    ax.set_xlabel("Imóveis do conjunto de teste (ordenados pelo preço real)")
    ax.set_ylabel("Preço (USD)")
    ax.set_title("Intervalo de previsão (p10–p90) vs. preço real — conjunto de teste")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prediction_interval.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Carregando e preparando dados...")
    X, y, groups = load_training_frame()
    feature_cols = X.columns.tolist()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set(), (
        "Vazamento de dados: o mesmo imóvel aparece em treino e teste"
    )

    print(f"Treino: {len(X_train)} linhas | Teste (holdout): {len(X_test)} linhas")

    print("Comparando modelos candidatos (GroupKFold, 5 folds)...")
    t0 = time.time()
    cv_results = cross_validate_models(X_train, y_train, groups_train)

    print("\n-- Modelos não-ensemble --")
    print(
        cv_results[cv_results["category"] == "non_ensemble"]
        .drop(columns="category")
        .sort_values("rmse")
    )
    print("\n-- Modelos ensemble --")
    print(
        cv_results[cv_results["category"] == "ensemble"]
        .drop(columns="category")
        .sort_values("rmse")
    )

    plot_model_comparison(cv_results, "non_ensemble", "modelos não-ensemble")
    plot_model_comparison(cv_results, "ensemble", "modelos ensemble")

    best_model_name = cv_results["rmse"].idxmin()
    print(f"\nMelhor modelo geral na validação cruzada: {best_model_name} ({time.time()-t0:.1f}s)")

    print(f"Preparando modelo final '{best_model_name}' (busca de hiperparâmetros quando aplicável)...")
    best_pipeline = tune_best_model(X_train, y_train, groups_train, best_model_name)

    print("Avaliando no conjunto de teste (holdout, nunca visto no treino)...")
    y_pred_test = best_pipeline.predict(X_test)
    test_metrics = _regression_metrics(y_test, y_pred_test)
    print(test_metrics)

    plot_actual_vs_predicted(y_test.reset_index(drop=True), pd.Series(y_pred_test))
    plot_feature_importance(best_pipeline, feature_cols)

    print(f"Treinando modelos de quantil (p{int(QUANTILE_LOW*100)} / p{int(QUANTILE_HIGH*100)})...")
    lower_pipeline = build_quantile_pipeline(QUANTILE_LOW).fit(X_train, y_train)
    upper_pipeline = build_quantile_pipeline(QUANTILE_HIGH).fit(X_train, y_train)
    lower_pred_test = lower_pipeline.predict(X_test)
    upper_pred_test = upper_pipeline.predict(X_test)

    coverage = float(np.mean((y_test.values >= lower_pred_test) & (y_test.values <= upper_pred_test)))
    quantile_metrics = {
        "alpha_low": QUANTILE_LOW,
        "alpha_high": QUANTILE_HIGH,
        "pinball_loss_low": float(mean_pinball_loss(y_test, lower_pred_test, alpha=QUANTILE_LOW)),
        "pinball_loss_high": float(mean_pinball_loss(y_test, upper_pred_test, alpha=QUANTILE_HIGH)),
        "empirical_coverage": coverage,
        "expected_coverage": QUANTILE_HIGH - QUANTILE_LOW,
        "mean_interval_width": float(np.mean(upper_pred_test - lower_pred_test)),
    }
    print(quantile_metrics)

    plot_prediction_interval(
        y_test.reset_index(drop=True), lower_pred_test, y_pred_test, upper_pred_test
    )

    print("Retreinando no dataset completo (treino + teste) para o modelo final...")
    final_pipeline = candidate_models()[best_model_name]
    final_pipeline.set_params(**{
        k: v for k, v in best_pipeline.get_params().items() if k.startswith("model__")
    })
    final_pipeline.fit(X, y)

    final_lower_pipeline = build_quantile_pipeline(QUANTILE_LOW).fit(X, y)
    final_upper_pipeline = build_quantile_pipeline(QUANTILE_HIGH).fit(X, y)

    joblib.dump(final_pipeline, MODELS_DIR / "model.joblib", compress=3)
    joblib.dump(final_lower_pipeline, MODELS_DIR / "quantile_lower.joblib", compress=3)
    joblib.dump(final_upper_pipeline, MODELS_DIR / "quantile_upper.joblib", compress=3)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    metrics_payload = {
        "best_model": best_model_name,
        "cross_validation": cv_results.to_dict(orient="index"),
        "holdout_test": test_metrics,
        "prediction_interval": quantile_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X)),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Modelo, métricas e gráficos salvos em models/ e reports/.")


if __name__ == "__main__":
    main()
