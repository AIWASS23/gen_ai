# Comunicação com Stakeholders

Como eu apresentaria este projeto para um público de negócio (não técnico) —
foco em tradução de métricas técnicas em impacto prático, não em detalhes de
implementação do modelo.

## A pergunta de negócio, em uma frase

> "Dado um imóvel novo (características físicas + bairro), qual o preço de
> mercado esperado, e o quão confiável é essa estimativa?"

## 1. O que os dados mostram (EDA traduzida)

- **Tamanho e qualidade de construção vendem**: o tamanho da área construída
  (`sqft_living`) e a nota de qualidade/acabamento (`grade`) são, isoladamente,
  os fatores mais associados ao preço — o que já era esperado, mas agora está
  quantificado.
- **Bairro importa tanto quanto a casa em si**: ao cruzar com dados
  demográficos por CEP, vemos que renda média do bairro e escolaridade
  também se relacionam fortemente com o preço — reforçando o ditado do
  mercado imobiliário de que "localização" é parte do preço, não só metragem.
  (`reports/demographics_correlation.png`)
- **Qualidade dos dados**: encontramos e corrigimos 1 registro com erro de
  digitação óbvio (um imóvel cadastrado com 33 quartos) antes de treinar o
  modelo — importante para credibilidade: qualquer decisão de preço baseada
  em dado sujo teria efeito real em uma negociação.

**Gráfico-chave para negócio:** mapa de preço por localização
(`reports/price_by_location.png`) — mostra visualmente onde estão as regiões
de maior valorização, uma peça que times comerciais entendem imediatamente
sem precisar de contexto estatístico.

## 2. O que o modelo entrega

- Comparamos 16 abordagens (do baseline de Regressão Linear e uma rede
  neural simples a modelos de árvore como Random Forest, XGBoost, LightGBM e
  CatBoost) e escolhemos a combinação de melhor desempenho: um **ensemble
  (stacking)** que combina XGBoost, LightGBM e Random Forest.
- **Em imóveis que o modelo nunca viu**, o erro típico de previsão é de
  aproximadamente **US$ 64 mil**, o que equivale a cerca de **14% do preço
  mediano** de mercado (US$ 450 mil). O modelo explica a variação de preço
  com precisão consistente (R² ≈ 0,91).
- **Tradução prática**: para uma casa de US$ 450 mil, a estimativa do modelo
  tende a ficar numa faixa de aproximadamente ±US$ 60–65 mil do valor real —
  útil como **primeira estimativa rápida** (triagem, precificação inicial de
  anúncio, comparação entre imóveis), mas **não substitui uma avaliação
  formal** para decisões de alto valor (financiamento, litígio, seguro).
- **A previsão já vem com uma faixa, não só um número**: além do valor
  central, o pipeline gera um limite inferior (~p10) e superior (~p90) por
  imóvel (colunas `predicted_price_low`/`predicted_price_high` em
  `outputs/future_predictions.csv`, gráfico em
  `reports/prediction_interval.png`). Ressalva técnica registrada em
  [`docs/model.md`](model.md#d-intervalo-de-previsão-não-só-um-ponto): a
  cobertura real observada dessa faixa (72%) ficou abaixo do que ela promete
  nominalmente (80%), ou seja, hoje a faixa é um pouco mais estreita do que
  deveria — suficiente para comunicar incerteza de forma direcional em uma
  primeira versão, mas ainda não calibrada para uso em compromissos formais
  de preço.
- O gráfico "real vs. previsto" (`reports/actual_vs_predicted.png`) mostra
  que o modelo é preciso na maior parte da faixa de preços, com mais
  incerteza em imóveis de altíssimo padrão (acima de ~US$ 2–3 milhões) — o que
  faz sentido: imóveis de luxo têm menos exemplos no histórico e preços mais
  influenciados por características não capturadas nos dados (design
  exclusivo, negociação particular).

## 3. Como eu apresentaria isso (formato)

Para um público de negócio, eu evitaria termos como "RMSE" e "R²" na primeira
tela e usaria:

| Métrica técnica | Como comunicar para negócio |
|---|---|
| MAE = US$ 64k | "Em média, a estimativa erra por ~US$ 64 mil para mais ou para menos" |
| MAPE = 12% | "O erro típico é de ~12% do valor do imóvel" |
| R² = 0,91 | "O modelo captura ~91% dos fatores que explicam a variação de preço na região" |

E entregaria sempre uma **faixa de preço** em vez de um número único
(ex.: "US$ 420k – US$ 480k" em vez de só "US$ 450.000") — isso já está
implementado (`predicted_price_low`/`predicted_price_high`) e é mais honesto
sobre a incerteza, além de mais próximo de como o próprio mercado imobiliário
já comunica avaliações (laudos trazem faixas, não valores exatos).

## 4. Próximos passos sugeridos ao negócio

1. Validar o modelo com um piloto controlado (ex.: comparar a estimativa do
   modelo com o preço final de venda em um lote de imóveis já negociados
   recentemente, fora da amostra de treino).
2. Definir, junto ao time comercial, qual erro máximo é aceitável para cada
   caso de uso (ex.: triagem interna tolera mais erro do que uma proposta de
   preço enviada ao cliente).
3. Acompanhar o [plano de aprendizado contínuo](continuous_learning.md) para
   manter a precisão à medida que o mercado muda.
