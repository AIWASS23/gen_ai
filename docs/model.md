# Desenvolvimento do Modelo de Machine Learning

Código completo em [`src/train.py`](../src/train.py) e [`src/data.py`](../src/data.py).
Resultados reproduzíveis com `uv run python -m src.train`.

## a. Variáveis importantes

O conjunto final de features combina as características físicas do imóvel
(`kc_house_data.csv`) com as variáveis demográficas do CEP
(`zipcode_demographics.csv`), mais 3 features derivadas: `house_age`,
`was_renovated` e `years_since_renovation`.

O modelo final é um ensemble (stacking, ver seção b.); a importância de
features abaixo é a **média das importâncias dos 3 modelos de árvore que o
compõem** (XGBoost, LightGBM, Random Forest) —
`reports/feature_importance.png`. As mais relevantes são:

1. **`grade`** (nota de qualidade de construção/acabamento) e **`sqft_living`**
   (área construída) — os dois pesos isolados mais altos. Fazem sentido:
   refletem diretamente o padrão construtivo e o tamanho do imóvel.
2. **`per_bchlr`** (% da população do CEP com diploma de bacharel) e
   **`hous_val_amt`** (valor mediano dos imóveis do CEP) — confirmam que o
   **perfil socioeconômico do bairro** é quase tão relevante quanto a casa em
   si.
3. **`per_prfsnl`** (% da população em ocupações profissionais) e **`lat`**
   (latitude) — reforçam o efeito de localização/bairro.
4. **`waterfront`** — amenidade rara (~0,75% dos imóveis) mas com forte prêmio
   de preço quando presente.

Isso confirma a hipótese levantada na EDA: **preço de imóvel = função do
imóvel + função do bairro**, e nenhuma das duas fontes de dados sozinha
captura todo o sinal.

## b. Escolha do modelo

Foram comparados **16 modelos** (ver `non_ensemble_models` e `ensemble_models`
em `src/train.py`), organizados em dois grupos e todos avaliados com a mesma
validação cruzada agrupada (`GroupKFold`, 5 folds, agrupando por `id` do
imóvel). Os dois grupos são comparados separadamente nos relatórios
(`reports/model_comparison_non_ensemble.png` e
`reports/model_comparison_ensemble.png`, porque suas escalas de erro são
muito diferentes), mas competem juntos pela escolha do modelo final.

### Grupo 1 — modelos não-ensemble (um único preditor)

| Modelo | RMSE (CV) | MAE (CV) | R² (CV) |
|---|---|---|---|
| MLP (rede neural) | US$ 141.291 | US$ 83.918 | 0,851 |
| KNN | US$ 156.306 | US$ 78.530 | 0,817 |
| Decision Tree | US$ 161.966 | US$ 90.881 | 0,803 |
| Regressão Linear | US$ 167.452 | US$ 101.914 | 0,790 |
| Lasso | US$ 167.607 | US$ 101.985 | 0,790 |
| Ridge | US$ 167.710 | US$ 102.091 | 0,790 |

- **Lineares (Regressão Linear, Ridge, Lasso)**: baseline de referência.
  Ridge/Lasso adicionam regularização L2/L1, relevante pela colinearidade
  forte entre variáveis de área (`sqft_living`, `sqft_above`,
  `sqft_living15`). Os três performam quase igual — sinal de que o problema é
  fundamentalmente **não-linear** e regularização sozinha não resolve isso
  (o Lasso nem chega a zerar nenhum coeficiente de forma relevante).
- **Decision Tree**: uma única árvore (RMSE US$ 161.966) fica pior até que o
  KNN e mal supera os modelos lineares — a comparação com Random Forest
  (US$ 128.908, grupo 2) isola exatamente o ganho que vem só de "combinar
  várias árvores" (bagging), sem trocar de família de modelo.
- **KNN**: não-paramétrico, captura diretamente "imóveis parecidos e
  próximos tendem a ter preço parecido" via lat/long, sem impor forma
  funcional. Supera os lineares e a árvore única, mas fica atrás dos
  ensembles de árvore.
- **MLP (rede neural pequena)**: o melhor modelo não-ensemble (R² = 0,851),
  mas ainda 20%+ pior em RMSE que o pior modelo do grupo ensemble — os logs
  de treino mostram `ConvergenceWarning` (não convergiu em 500 iterações);
  uma rede maior/mais bem tunada poderia melhorar isso, mas dificilmente
  ultrapassaria os boostings de árvore neste volume de dados (~17k linhas,
  pouco para redes neurais brilharem sobre árvores em dados tabulares).

### Grupo 2 — modelos ensemble (combinam várias previsões)

| Modelo | RMSE (CV) | MAE (CV) | R² (CV) |
|---|---|---|---|
| **Stacking (XGBoost + LightGBM + RF)** | **US$ 116.390** | **US$ 63.416** | **0,898** |
| XGBoost | US$ 117.054 | US$ 63.762 | 0,897 |
| Voting (XGBoost + LightGBM + RF) | US$ 117.228 | US$ 63.574 | 0,897 |
| CatBoost | US$ 117.356 | US$ 64.699 | 0,897 |
| Gradient Boosting (sklearn) | US$ 117.475 | US$ 67.079 | 0,897 |
| LightGBM | US$ 119.669 | US$ 63.822 | 0,892 |
| HistGradientBoosting | US$ 125.923 | US$ 67.983 | 0,881 |
| Extra Trees | US$ 127.258 | US$ 69.779 | 0,879 |
| Random Forest | US$ 128.908 | US$ 69.757 | 0,876 |
| AdaBoost | US$ 161.584 | US$ 107.318 | 0,805 |

- **Bagging (Random Forest, Extra Trees)**: no mesmo patamar entre si; Extra
  Trees leva vantagem por ter ainda mais aleatoriedade nos cortes dos nós.
  Ficam abaixo dos boostings — bagging reduz variância mas, sozinho, não
  corrige tão bem o viés quanto o boosting sequencial.
- **AdaBoost**: de longe o pior do grupo ensemble (RMSE US$ 161.584, no nível
  dos modelos lineares) — boosting clássico com árvores rasas e sem tuning
  agressivo não é competitivo aqui; incluído só como referência histórica da
  família de boosting, não como candidato real.
- **Boosting moderno (Gradient Boosting, HistGradientBoosting, XGBoost,
  LightGBM, CatBoost)**: dominam a comparação, todos com R² ≥ 0,88. Confirma
  que o preço depende de **interações não-lineares** entre variáveis (ex.: o
  efeito de `sqft_living` muda conforme `grade` e a renda do CEP). Dentro
  deste subgrupo, CatBoost (trata `zipcode` como categoria nativa) termina no
  mesmo patamar do XGBoost/LightGBM — a informação de bairro que `zipcode`
  adicionaria já está majoritariamente coberta por `lat`/`long` e pelas
  variáveis demográficas.
- **Voting vs. Stacking**: os dois combinam exatamente os mesmos 3 modelos
  (XGBoost + LightGBM + Random Forest — ver `_ensemble_base_estimators` em
  `src/train.py`), a única diferença é *como* combinam. Terminaram
  praticamente empatados (US$ 117.228 vs. US$ 116.390) — o meta-modelo do
  Stacking aprende pesos ligeiramente melhores que a média simples do
  Voting, mas o ganho é pequeno, o que é o resultado esperado quando os
  modelos de base já são fortes e correlacionados entre si.

**Stacking foi o escolhido como modelo final**, pelos seguintes motivos:

- **Melhor desempenho** entre os 16 candidatos na validação cruzada, e a
  vantagem se confirma no holdout final (RMSE US$ 111.319 / MAE US$ 63.654 /
  R² 0,911) — não é ruído de um único split.
- **Ganho vem de diversidade real, não de complexidade cosmética**: XGBoost,
  LightGBM e Random Forest cometem erros parcialmente diferentes entre si
  (crescimento de árvore diferente, amostragem diferente), então o
  meta-modelo consegue corrigir parte do erro sistemático de cada um — e o
  fato de Voting (média simples dos mesmos 3 modelos) também aparecer entre
  os 3 melhores confirma que a diversidade em si já ajuda, não é só ajuste
  fino do meta-modelo.
- **Resultado robusto, não específico de uma implementação**: XGBoost,
  CatBoost, LightGBM e Gradient Boosting (quatro implementações
  independentes de boosting em árvore) terminam muito próximas entre si —
  reforça que o ganho vem da abordagem, não de um detalhe específico de uma
  lib.
- Continua **interpretável o suficiente**: a importância de features é obtida
  como a média das importâncias dos 3 modelos de árvore que compõem o
  stacking (ver seção a.), então a explicabilidade não se perde.

**Custo a reconhecer**: o stacking é mais caro de treinar (treina 3 modelos +
o meta-modelo) e mais complexo de servir/depurar em produção que um único
XGBoost. Para uma equipe que priorize simplicidade operacional a um ganho de
~0,6% de RMSE sobre o XGBoost isolado, **XGBoost isolado é uma escolha
perfeitamente razoável** — é a alternativa de segunda colocação, com a mesma
robustez de resultado e bem menos complexidade para manter. Voting é uma
alternativa intermediária (mesmo ganho de diversidade do Stacking, sem o
custo de treinar um meta-modelo). CatBoost, Gradient Boosting e LightGBM
seguem como alternativas viáveis de terceira escolha. Regressão
Linear/Ridge/Lasso seguem como **baseline de referência**, e KNN/Decision
Tree/MLP como pontos de comparação de outras famílias de modelo (similaridade
geográfica, árvore única, rede neural).

**Nota sobre tamanho do artefato**: a primeira versão do Random Forest usado
aqui (`max_depth=None`, profundidade ilimitada) gerava um `model.joblib` de
**259 MB** dentro do Stacking/Voting — inviável para um repositório GitHub
público. Limitamos a profundidade (`max_depth=12`, `min_samples_leaf=4`) e
habilitamos compressão no `joblib.dump` (`compress=3`); o impacto no RMSE foi
de aproximadamente US$ 1.000 (desprezível), e o artefato final caiu para
**~7,5 MB**.

## c. Generalização

Cuidados tomados para que a métrica reportada reflita desempenho real em
dados novos, e não memorização:

1. **Split por grupo, não aleatório**: 177 imóveis aparecem duas vezes no
   dataset (vendidos 2x na janela coberta). Um split aleatório colocaria a
   mesma casa em treino e teste, vazando informação. Usamos
   `GroupShuffleSplit`/`GroupKFold` agrupando por `id` do imóvel, garantindo
   que um mesmo imóvel nunca apareça nos dois lados do split
   (`src/train.py`, asserção explícita disso no código).
2. **Holdout final nunca usado em nenhuma decisão de modelagem**: 20% dos
   dados (grupos únicos) foram reservados como conjunto de teste, usado
   **apenas uma vez**, no final, para reportar a métrica de generalização
   (RMSE US$ 111.319 / MAE US$ 63.654 / R² 0,911) — toda escolha de modelo e
   hiperparâmetro foi feita olhando somente para a validação cruzada no
   conjunto de treino.
3. **Validação cruzada (5-fold, agrupada)** tanto para comparar modelos
   quanto para a busca de hiperparâmetros (`RandomizedSearchCV`), reduzindo o
   risco de escolher hiperparâmetros que só funcionam bem em um split
   específico.
4. **Regularização via hiperparâmetros dos modelos de base**: `max_depth`
   limitado, `subsample`/`colsample_bytree` < 1 (bagging de linhas/colunas a
   cada árvore) e `learning_rate` baixo com mais árvores no XGBoost/LightGBM
   internos ao stacking — os mesmos hiperparâmetros validados isoladamente
   para cada um antes de compor o ensemble.
5. **Remoção do outlier de captura de dados** (`bedrooms = 33`) antes do
   treino, evitando que o modelo aprenda um padrão espúrio de um único
   registro corrompido.
6. Métrica final de holdout (R² = 0,911) muito próxima da métrica de
   validação cruzada (R² = 0,898) — sinal de que o modelo **não está
   sobreajustado** ao processo de seleção de modelo.
7. **Limitação assumida no meta-modelo do stacking**: a validação cruzada
   interna do `StackingRegressor` (usada para gerar as previsões
   out-of-fold que treinam o meta-modelo) usa `KFold` padrão, não agrupado
   por `id` do imóvel. O possível vazamento fica restrito aos ~0,8% de
   imóveis vendidos duas vezes e apenas dentro do treino do meta-modelo —
   não afeta a divisão treino/teste externa (holdout), que segue agrupada.
   Documentado no código (`src/train.py`) para transparência.

## d. Intervalo de previsão (não só um ponto)

Além do modelo de ponto (stacking), o pipeline treina dois modelos auxiliares
de **regressão quantílica** (LightGBM, `objective="quantile"`) para os
percentis 10 e 90 do preço — `build_quantile_pipeline` em `src/train.py`,
artefatos salvos em `models/quantile_lower.joblib` e
`models/quantile_upper.joblib`. O objetivo é entregar uma **faixa de preço**
(`predicted_price_low` – `predicted_price_high` em
`outputs/future_predictions.csv`), não só um valor único — conforme
recomendado em [`docs/stakeholder_communication.md`](stakeholder_communication.md).

Avaliação no holdout (`reports/prediction_interval.png`):

| Métrica | Valor |
|---|---|
| Cobertura empírica do intervalo p10–p90 | 72,4% |
| Cobertura esperada (nominal) | 80% |
| Largura média do intervalo | ≈ US$ 175.387 |
| Pinball loss (p10 / p90) | US$ 14.665 / US$ 17.210 |

**Leitura honesta**: a cobertura empírica (72,4%) ficou **abaixo** da
cobertura nominal esperada (80%) — os dois modelos de quantil são treinados
de forma independente (não é uma regressão quantílica conjunta/calibrada), o
que é uma simplificação razoável para uma primeira versão, mas explica a
subcobertura. Na prática isso significa que o intervalo reportado é um pouco
**otimista** (mais estreito do que deveria para atingir 80% de cobertura
real). Não houve, porém, nenhum caso de "quantile crossing" (limite inferior
maior que o superior) nos dados avaliados. Próxima iteração natural:
recalibrar os quantis (ex.: conformal prediction) ou usar `alpha` mais
extremos (ex.: p5/p95) para compensar a subcobertura observada.

### Limitações conhecidas (honestidade sobre o que o modelo não resolve)

- O modelo final é retreinado com **todos os dados** (treino + holdout) antes
  de ser salvo em `models/model.joblib`, para aproveitar o máximo de dado
  disponível em produção — as métricas reportadas vêm do modelo intermediário
  avaliado no holdout, não deste artefato final (prática padrão, mas vale
  registrar).
- Maior erro em imóveis de altíssimo padrão (poucos exemplos no treino).
- `future_unseen_examples.csv` não tem `date` de referência; a feature
  `house_age` usa um ano de referência fixo (2015, o último ano observado no
  treino) tanto no treino quanto na inferência, para manter consistência —
  ver `REFERENCE_YEAR` em `src/data.py`.
