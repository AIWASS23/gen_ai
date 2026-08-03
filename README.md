# Previsão de Preços de Casas — King County (Seattle)

Solução para o desafio de Data Science descrito em [`docs/CHALLENGE.md`](docs/CHALLENGE.md):
prever o preço de imóveis residenciais na região de Seattle combinando
características físicas do imóvel com dados demográficos do CEP.

**Resultado do modelo final** (stacking de XGBoost + LightGBM + Random
Forest, escolhido entre 16 modelos comparados, avaliado em dados nunca
vistos no treino): MAE ≈ US$ 64 mil · RMSE ≈ US$ 111 mil · R² ≈ 0,91. Cada
previsão também vem com uma faixa (p10–p90), não só um ponto. Detalhes e
justificativas em [`docs/model.md`](docs/model.md).

## Estrutura do repositório

```
requirements.txt             dependências do projeto, com versões fixas (instalar via uv/pip)
data/raw/                    dados originais fornecidos (não versionar dados sensíveis reais em projetos de cliente)
notebooks/01_eda.ipynb       análise exploratória (também disponível como script em 01_eda.py, formato jupytext)
src/data.py                  carregamento, limpeza, engenharia de features e merge com dados demográficos
src/train.py                 comparação de 16 modelos (não-ensemble + ensemble), tuning, modelos de quantil e serialização
src/predict.py                gera previsões (ponto + faixa p10-p90) para data/raw/future_unseen_examples.csv
models/                      modelo final (model.joblib), modelos de quantil (quantile_lower/upper.joblib) e métricas
reports/                     gráficos gerados pela EDA e pelo treino
outputs/future_predictions.csv previsões geradas para os imóveis sem preço (com faixa de preço)
docs/
  CHALLENGE.md                enunciado original do desafio
  model.md                    variáveis importantes, escolha do modelo e estratégia de generalização
  deploy_strategy.md          arquitetura proposta de deploy em produção (com diagrama)
  continuous_learning.md      como o modelo aprenderia com novos dados ao longo do tempo
  stakeholder_communication.md  como os resultados seriam comunicados a um público de negócio
```

## Como reproduzir

O ambiente é gerenciado com [`uv`](https://docs.astral.sh/uv/) no modo
compatível com `pip`, a partir de `requirements.txt` (sem `pyproject.toml`).

```bash
# cria o venv (usa a versão do Python fixada em .python-version)
uv venv

# instala as dependências
uv pip install -r requirements.txt

# ativa o venv (ou prefixe os comandos abaixo com `uv run --no-project`)
source .venv/bin/activate

# 1. análise exploratória (regenera os gráficos em reports/)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# ou abra notebooks/01_eda.ipynb normalmente no Jupyter/VS Code

# 2. treina, compara modelos e salva o modelo final em models/ (leva alguns minutos por causa do stacking)
python -m src.train

# 3. gera previsões para os imóveis sem preço (data/raw/future_unseen_examples.csv)
python -m src.predict
```

## Resumo das decisões

| Entregável do desafio | Onde está |
|---|---|
| Análise e entendimento dos dados | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| Variáveis importantes, escolha do modelo, generalização | [`docs/model.md`](docs/model.md) |
| Estratégia de deploy | [`docs/deploy_strategy.md`](docs/deploy_strategy.md) |
| Aprendizado contínuo | [`docs/continuous_learning.md`](docs/continuous_learning.md) |
| Comunicação com stakeholders | [`docs/stakeholder_communication.md`](docs/stakeholder_communication.md) |

## Principais decisões e por que

- **Merge com dados demográficos por `zipcode`**: correspondência 1:1 perfeita
  entre os 70 CEPs de `kc_house_data.csv` e `zipcode_demographics.csv`, sem
  perda de linhas.
- **Remoção de 1 outlier de captura de dados** (imóvel com `bedrooms = 33`,
  fisicamente incoerente com seus 1.620 sqft).
- **Split de treino/teste agrupado por `id` do imóvel** (não aleatório puro),
  porque 177 imóveis foram vendidos duas vezes na janela do dataset — um split
  aleatório vazaria informação do mesmo imóvel entre treino e teste.
- **16 modelos comparados**, organizados em não-ensemble (Regressão Linear,
  Ridge, Lasso, KNN, Decision Tree, MLP) e ensemble (Random Forest, Extra
  Trees, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost,
  LightGBM, CatBoost, Voting, Stacking) na mesma validação cruzada agrupada
  — **Stacking** (XGBoost + LightGBM + Random Forest com meta-modelo
  `RidgeCV`) venceu por uma margem pequena mas consistente sobre o XGBoost
  isolado; justificativa completa, incluindo o tradeoff de complexidade
  operacional, em [`docs/model.md`](docs/model.md).
- **Tamanho do artefato do modelo**: o Random Forest usado dentro do
  Stacking/Voting, sem limite de profundidade, gerava um `model.joblib` de
  259 MB — inviável para um repositório GitHub público. Limitamos a
  profundidade e comprimimos o artefato (`joblib.dump(..., compress=3)`);
  impacto no RMSE desprezível, arquivo final caiu para ~7,5 MB.
- **Intervalo de previsão**: dois modelos LightGBM de regressão quantílica
  (p10/p90) rodam junto com o modelo de ponto, entregando uma faixa de preço
  em vez de um valor único — com uma ressalva de calibração documentada e
  honesta em [`docs/model.md`](docs/model.md#d-intervalo-de-previsão-não-só-um-ponto).
- **`uv`** (modo pip-compatível) para criar o venv e instalar
  `requirements.txt` com versões fixas (`==`), garantindo que o ambiente seja
  reprodutível sem depender de um lockfile.

## Limitações e próximos passos (primeira versão)

Esta é a primeira versão da solução. Próximos passos naturais, não incluídos
aqui por escopo: testes automatizados do pipeline de dados/treino, uma API de
serving real (desenhada, não implementada, em `docs/deploy_strategy.md`),
recalibração dos intervalos de previsão (a cobertura empírica hoje é 72%,
abaixo do 80% nominal — ver `docs/model.md`) e uma segunda rodada de feature
engineering (ex.: distância a pontos de interesse, dados de mercado mais
recentes que os de 2014–2015).
