# Aprendizado Contínuo

Como a solução poderia aprender com novos dados ao longo do tempo, sem
intervenção manual a cada atualização.

## 1. De onde vêm os novos dados

- **Novas transações de venda** (preço realizado) — o sinal mais valioso, pois
  é a variável alvo real chegando aos poucos.
- **Novas listagens sem preço** (equivalente a `future_unseen_examples.csv`) —
  úteis para monitorar drift de distribuição de entrada, mas não para retreino
  supervisionado (não têm rótulo).
- **Atualizações demográficas por CEP** — atualizadas com frequência muito
  mais baixa (ex.: anual, censo/pesquisas), então o pipeline deve suportar
  versões diferentes do arquivo demográfico ao longo do tempo.

## 2. Gatilhos de retreino

Dois gatilhos complementares, não mutuamente exclusivos:

1. **Agendado (calendário)**: retreino a cada ciclo fixo (ex.: mensal), já que
   o mercado imobiliário muda de forma relativamente lenta e contínua
   (sazonalidade, taxa de juros, oferta/demanda regional).
2. **Por gatilho de qualidade/drift**: se o monitoramento (ver
   [docs/deploy_strategy.md](deploy_strategy.md)) detectar:
   - degradação do erro real (RMSE/MAE em produção) acima de um limite, ou
   - drift significativo na distribuição das features de entrada (ex.: um
     novo padrão de imóveis, novos CEPs, mudança de mix de tipos de imóvel),

   um retreino é disparado fora do calendário normal.

## 3. Retreino

- Pipeline de treino é reexecutado com a base de dados acumulada (dados
  antigos + novas transações), não apenas com os dados novos — mantendo o
  histórico para o modelo continuar generalizando bem em cenários mais raros
  (ex.: imóveis de luxo, waterfront), que aparecem pouco em janelas curtas.
- Alternativas a considerar conforme o volume de dados novos crescer:
  - **Janela deslizante** (ex.: últimos 24 meses) para dar mais peso ao
    comportamento recente do mercado e evitar que dados muito antigos
    distorçam preços atuais.
  - **Ponderação por recência** (amostras mais recentes com peso maior no
    treino) como meio-termo entre usar tudo e usar só o mais novo.

## 4. Avaliação antes de substituir o modelo em produção

Um modelo novo **nunca substitui o modelo em produção automaticamente sem
validação**:

1. **Backtest out-of-time**: avaliar o candidato em vendas mais recentes que
   ficaram de fora do treino (não um split aleatório) — simula prever o
   futuro com dados do passado, cenário mais próximo do uso real.
2. **Comparação direta contra o modelo campeão** nas mesmas métricas (RMSE,
   MAE, MAPE) e, quando possível, em segmentos relevantes de negócio (por
   faixa de preço, por região) — um modelo pode melhorar a média e piorar
   sistematicamente um segmento específico (ex.: imóveis de alto padrão).
3. **Shadow deployment / canário**: antes de virar o modelo principal, o
   candidato roda em paralelo recebendo tráfego real (sem impactar o usuário
   final) por um período, ou atende a uma fração pequena do tráfego (ex.: 5%),
   permitindo comparar previsões com o campeão em condições reais.
4. **Aprovação**: se as métricas confirmarem ganho (ou não regressão) e não
   houver alerta do canário, o modelo é promovido a campeão no Model Registry.
   Uma revisão humana (data scientist/responsável do produto) confirma antes
   da promoção total nas primeiras iterações do processo, podendo ser
   progressivamente mais automatizada à medida que a confiança no processo
   aumenta.

## 5. Substituição e rollback

- A troca do modelo em produção é apenas uma troca de referência no Model
  Registry (o serving já foi desenhado para carregar o modelo por versão) —
  não exige redeploy de código.
- Se o novo modelo apresentar problemas depois de promovido (detectado pelo
  monitoramento contínuo), o rollback para a versão anterior é imediato, pelo
  mesmo mecanismo.
