# Aprendizado Contínuo

Como a solução poderia aprender com novos dados ao longo do tempo, sem
intervenção manual a cada atualização.

> **Atualização**: as seções 3, 4 e 5 (retreino, avaliação e substituição em
> produção) foram **efetivamente implementadas** em
> [`src/continuous_learning.py`](../src/continuous_learning.py), não só
> desenhadas — split out-of-time real, gate de qualidade, promoção com
> backup/rollback automático, log de auditoria, e integração com o
> hot-reload da API (`POST /v1/admin/reload-model`, já implementado em
> `deploy/api/`). Testado de ponta a ponta com dados e modelo reais deste
> repositório — resultados e comandos reprodutíveis na seção 6.

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

   O CLI implementado expõe isso via `--trigger {manual,scheduled,drift}`,
   gravado em cada linha do log de auditoria — o disparo em si (cron, job de
   drift) não está implementado, só o parâmetro que o identificaria.

## 3. Retreino — implementado

- `src/continuous_learning.py` retreina a **mesma arquitetura campeã**
  (stacking, ver `src/train.py`) com os dados disponíveis até uma data de
  corte (`--cutoff-date`) — reajustar pesos com dados novos é uma tarefa de
  rotina; recomparar as 16 famílias de modelo do zero é mais rara e cara
  ("revisão de arquitetura", uma execução manual de `src/train.py`, não
  deste pipeline).
- Reaproveita a base de dados acumulada (tudo até a data de corte, não só o
  que é "novo"), pelo mesmo motivo do desenho original: manter o modelo
  generalizando bem em cenários raros (imóveis de luxo, waterfront).
- Janela deslizante e ponderação por recência (mencionadas como alternativas
  no desenho original) **não** foram implementadas — o retreino atual sempre
  usa tudo até a data de corte.

## 4. Avaliação antes de substituir o modelo em produção — implementado (exceto canário)

1. **✅ Backtest out-of-time**: `time_based_split()` divide por `date` (não
   aleatório) — treino = tudo até a data de corte, avaliação = os
   `--holdout-days` dias seguintes, nunca vistos pelo desafiante.
2. **✅ Comparação direta contra o campeão**, mas com uma correção importante
   descoberta rodando isso de verdade (não só desenhando): comparar o RMSE
   do desafiante contra o RMSE *gravado* do campeão (`models/metrics.json`)
   é injusto, porque aquele número vem de um split aleatório diferente.
   Pior ainda: **o campeão publicado neste repositório foi treinado com
   100% do histórico** (é o entregável final do desafio), então ele já viu
   qualquer janela out-of-time que se escolha — reavaliá-lo nela mede
   desempenho *dentro* do treino, não fora. `evaluate_champion_on_holdout()`
   corrige isso reavaliando o campeão **na mesma janela out-of-time** do
   desafiante antes de comparar — ver a seção 6 para os números reais dessa
   armadilha e como contorná-la para uma demonstração honesta. Comparação
   por segmento de negócio (faixa de preço, região) **não** foi implementada.
3. **Shadow deployment / canário**: **não implementado** — exigiria tráfego
   de produção real duplicado, fora do escopo deste projeto. O `--dry-run`
   do CLI é o equivalente mais próximo disponível (avalia sem promover).
4. **✅ Aprovação**: `evaluate_gate()` decide automaticamente (promove se o
   desafiante não piorar o RMSE além de `--tolerance`, padrão 2%). Não há
   revisão humana no loop hoje — é 100% automático; adicionar uma etapa de
   aprovação manual (ex.: exigir `--force-promote` ou um `input()` de
   confirmação) seria trivial de acrescentar se o processo de negócio pedir.

## 5. Substituição e rollback — implementado

- A promoção (`promote()`) escreve os novos artefatos direto em `models/` e
  aciona (opcionalmente, via `--notify-url`) o mesmo endpoint
  `POST /v1/admin/reload-model` já implementado na API — o `ModelRegistry`
  troca de versão em memória sem reiniciar o processo, exatamente como
  desenhado.
- **Backup automático antes de qualquer promoção**: `backup_current_champion()`
  copia os artefatos atuais para `models/previous/` antes de sobrescrever.
  `--rollback` restaura de lá — testado restaurando os artefatos reais
  **byte-a-byte** (ver seção 6).
- Versionamento continua sendo o hash sha256 do artefato
  (`ModelRegistry.version`, já implementado em `deploy/api/`) — um Model
  Registry de verdade (MLflow/SageMaker) continua sendo a evolução natural
  para produção, guardando histórico completo de versões em vez de só
  "atual" + "anterior".

## 6. Validação real (não só desenho)

Comandos reprodutíveis e resultados obtidos rodando contra os dados e o
modelo reais deste repositório:

```bash
# Avalia sem promover (seguro para explorar) — usa o campeão real (models/):
uv run python -m src.continuous_learning --cutoff-date 2015-04-15 --dry-run

# Desfaz a última promoção, se necessário:
uv run python -m src.continuous_learning --rollback
```

**Demonstração honesta do gate, sem vazamento** — como o campeão real viu
100% do histórico, comparar contra ele em qualquer janela é injusto (ver
seção 4.2). Para provar o mecanismo sem essa distorção, ele foi rodado
contra um `--models-dir` isolado (nunca toca em `models/`), simulando um
campeão genuinamente mais antigo:

```bash
# Passo 1 — bootstrap: cria um campeão inicial só com dados até nov/2014
# (sem campeão anterior, o gate promove por padrão):
uv run python -m src.continuous_learning --models-dir /tmp/cl-demo/models \
  --cutoff-date 2014-11-01 --holdout-days 30

# Passo 2 — "6 meses depois": retreina com os dados que "chegaram" desde
# então (até mar/2015), avaliado numa janela de abril/2015 que NEM o
# campeão do passo 1 nem este desafiante viram:
uv run python -m src.continuous_learning --models-dir /tmp/cl-demo/models \
  --cutoff-date 2015-03-01 --holdout-days 30
```

Resultado real obtido:

| | Campeão (treinado até nov/2014) | Desafiante (treinado até mar/2015) |
|---|---|---|
| Linhas de treino | 11.755 | 16.866 |
| RMSE na mesma janela out-of-time (abr/2015) | \$116.705 | **\$112.452** |

O desafiante, com ~5 meses a mais de dados, teve RMSE **melhor** que o
campeão numa janela que nenhum dos dois viu — o gate promoveu corretamente.
Histórico de auditoria (`models/retrain_history.jsonl` do diretório de
demonstração) registrou as duas decisões, cada uma com métricas completas.

**Validação do ciclo completo contra os arquivos reais** (`models/`,
não uma cópia): promoção real (com `--tolerance` alto o bastante para
garantir aprovação, já que o objetivo aqui era testar a mecânica, não o
gate) → API notificada via `--notify-url` e `GET /v1/model` confirmando a
nova versão (`2c897d92d0bf`, diferente da original `4a11cf998ec9`) → `--rollback`
→ nova chamada a `/v1/admin/reload-model` → API de volta à versão original →
os 5 artefatos (`model.joblib`, `quantile_lower.joblib`,
`quantile_upper.joblib`, `feature_columns.json`, `metrics.json`) conferidos
**byte-a-byte** contra uma cópia de segurança feita antes do teste: idênticos.
O repositório terminou exatamente como começou.

Testes automatizados da lógica (split temporal, gate, backup/rollback) em
[`tests/test_continuous_learning.py`](../tests/test_continuous_learning.py)
— 13 testes, rápidos e determinísticos (sem treinar modelo de verdade).
