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
src/continuous_learning.py    retreino real: split out-of-time, gate de qualidade, promoção/rollback, hot-reload da API
tests/                        testes automatizados de src/ (pytest) — split temporal, gate, backup/rollback
models/                      modelo final (model.joblib), modelos de quantil (quantile_lower/upper.joblib) e métricas
reports/                     gráficos gerados pela EDA e pelo treino
outputs/future_predictions.csv previsões geradas para os imóveis sem preço (com faixa de preço)
presentation/                apresentação (.pptx) com os 5 entregáveis do desafio — ver presentation/desafio_data_science.pptx
deploy/                      implementação real da API de serving (não só o desenho) — ver deploy/README.md
  api/                          FastAPI, POO/tipado: routers, services (model registry, cache, observabilidade), schemas
  Dockerfile, docker-compose.yml  imagem de serving + Redis, testados de ponta a ponta
  k8s/                          manifests Kubernetes (Deployment, HPA, Ingress, Redis)
  tests/                         testes de integração da API (pytest)
docs/
  CHALLENGE.md                enunciado original do desafio
  model.md                    variáveis importantes, escolha do modelo e estratégia de generalização
  deploy_strategy.md          arquitetura de deploy em produção (com diagrama) — camada de serving/monitoramento já implementada em deploy/
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
| Estratégia de deploy (desenho) | [`docs/deploy_strategy.md`](docs/deploy_strategy.md) |
| Estratégia de deploy (implementação: API, Redis, Langfuse, Docker, K8s) | [`deploy/README.md`](deploy/README.md) |
| Aprendizado contínuo (desenho + implementação: retreino, gate, rollback) | [`docs/continuous_learning.md`](docs/continuous_learning.md) |
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

## API de serving (implementada)

Além do desenho em `docs/deploy_strategy.md`, este repositório inclui uma
**implementação real e testada** da camada de serving em [`deploy/`](deploy/):
API FastAPI orientada a objetos e tipada, cache Redis, observabilidade via
Langfuse, Dockerfile + docker-compose, e manifests Kubernetes. Ver
[`deploy/README.md`](deploy/README.md) para arquitetura, como rodar (local,
Docker, K8s) e a tabela honesta do que é implementação de referência vs. o
que uma stack de produção madura trocaria.

```bash
docker compose -f deploy/docker-compose.yml up --build
curl -X POST http://localhost:8000/v1/predictions -H "Content-Type: application/json" -d '{
  "houses": [{"bedrooms":4,"bathrooms":1.0,"sqft_living":1680,"sqft_lot":5043,"floors":1.5,
  "waterfront":0,"view":0,"condition":4,"grade":6,"sqft_above":1680,"sqft_basement":0,
  "yr_built":1911,"yr_renovated":0,"zipcode":98118,"lat":47.5354,"long":-122.273,
  "sqft_living15":1560,"sqft_lot15":5765}]
}'
```

## Aprendizado contínuo (implementado)

Além do desenho em `docs/continuous_learning.md`, o ciclo de retreino está
implementado em [`src/continuous_learning.py`](src/continuous_learning.py) e
validado com dados e modelo reais: split out-of-time (não aleatório), gate
de qualidade automático (promove só se o desafiante não piorar o RMSE além
de uma tolerância configurável), backup automático antes de promover, e
`--rollback`. Achado real ao validar: comparar o desafiante contra o RMSE
*gravado* do campeão é injusto (splits diferentes, e o campeão deste repo viu
100% do histórico) — corrigido avaliando o campeão na mesma janela
out-of-time do desafiante antes de decidir. Detalhes, comandos reprodutíveis
e os números da validação em [`docs/continuous_learning.md`](docs/continuous_learning.md#6-validação-real-não-só-desenho).

```bash
uv run python -m src.continuous_learning --cutoff-date 2015-04-15 --dry-run
```

## Limitações e próximos passos

Esta é a primeira versão da solução. Próximos passos naturais, não incluídos
aqui por escopo: shadow deployment/canário real para o retreino (o `--dry-run`
é o equivalente mais próximo hoje), recalibração dos intervalos de previsão
(a cobertura empírica hoje é 72%, abaixo do 80% nominal — ver `docs/model.md`),
e uma segunda rodada de feature engineering (ex.: distância a pontos de
interesse, dados de mercado mais recentes que os de 2014–2015).
