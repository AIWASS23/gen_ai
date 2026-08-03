# Estratégia de Deploy

Este documento descreve como o modelo de previsão de preços poderia ser colocado
em produção. **Nada aqui foi implementado** — é o desenho de arquitetura, como
pedido no desafio.

## Visão geral

```mermaid
flowchart TB
    subgraph SRC["Fontes de dados"]
        A1[(Transações de vendas\nMLS / cartório)]
        A2[(Dados demográficos\npor CEP)]
        A3[(Listagens novas\nsem preço)]
    end

    subgraph OFFLINE["Pipeline offline de treino (batch, agendado)"]
        B1[Ingestão e validação\nde dados]
        B2[Feature engineering\n+ merge demográfico]
        B3[Treino e tuning\nXGBoost / RF / Linear]
        B4{Gate de qualidade\nRMSE/MAE/R² vs.\nmodelo em produção}
        B5[Registro do modelo\nModel Registry]
    end

    subgraph SERVING["Serving online"]
        C1[API REST\nFastAPI + Docker]
        C2[Validação de schema\nde entrada]
        C3[(Cache de features\ndemográficas por CEP)]
        C4[Load Balancer]
    end

    subgraph CLIENTS["Consumidores"]
        D1[App / site interno\nde precificação]
        D2[Corretores / analistas]
    end

    subgraph MONITOR["Monitoramento e feedback"]
        E1[Logs de predições\n+ inputs]
        E2[Monitoramento de\ndrift de dados]
        E3[Monitoramento de\nperformance real\nquando preço de venda\nfica disponível]
        E4[Dashboards e alertas]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2 --> B3 --> B4
    B4 -- aprovado --> B5
    B4 -- reprovado --> B3

    B5 -- artefato versionado --> C1
    A3 --> D1
    D1 --> C4 --> C2 --> C1
    C3 --> C1
    C1 --> D1
    C1 --> D2

    C1 -- toda predição --> E1
    E1 --> E2
    E1 --> E3
    E2 --> E4
    E3 --> E4
    E4 -. dispara retreino .-> B1
```

## Camadas

### 1. Ingestão e preparação de dados
- Job batch (Airflow/Dagster/Prefect) roda periodicamente (ex.: diário ou
  semanal) para atualizar a base de transações e os dados demográficos por CEP.
- Validação de schema e de qualidade de dados na entrada (ex.: `great_expectations`
  ou `pandera`) para pegar problemas como o outlier de `bedrooms=33` identificado
  na EDA **antes** dele entrar no treino.
- O merge com dados demográficos é feito nesta camada (`src/data.py` já
  implementa essa lógica de forma testável e reutilizável).

### 2. Treino e Model Registry
- Pipeline de treino reexecuta `src/train.py` (ou equivalente produtizado),
  gerando um novo candidato a modelo.
- Um **gate automático de qualidade** compara o candidato contra o modelo
  campeão atual em um conjunto de teste/validação fixo (holdout out-of-time,
  não aleatório, para simular o cenário real de prever vendas futuras).
  Só é promovido se igualar ou superar o campeão em RMSE/MAE dentro de uma
  margem de tolerância.
- Modelos aprovados são versionados em um **Model Registry** (MLflow, SageMaker
  Model Registry, ou até um bucket S3/GCS versionado com metadados em JSON),
  guardando: hash dos dados de treino, hiperparâmetros, métricas, data de
  treino e artefato serializado (`joblib`/`onnx`).

### 3. Serving (API)
- API REST leve (FastAPI) que expõe `POST /predict`, recebendo as mesmas
  colunas de `future_unseen_examples.csv` e retornando `predicted_price` (e
  idealmente um intervalo de confiança, ver seção de comunicação).
- Empacotada em container Docker, versão da imagem atrelada à versão do
  modelo carregado — a API carrega o artefato do Model Registry no startup
  (não faz treino online).
- Validação de entrada com Pydantic (tipos, ranges plausíveis, zipcode
  conhecido) rejeitando requisições malformadas antes de chegar ao modelo.
- Dados demográficos por CEP são poucos (70 linhas) e mudam raramente — podem
  ficar em cache em memória/Redis na API em vez de round-trip a um banco a
  cada requisição.
- Deploy em Kubernetes (ou serverless tipo AWS Lambda/SageMaker Endpoint) atrás
  de um load balancer, com autoscaling horizontal e health checks.

### 4. Monitoramento
- **Logging de todas as predições** (input + output + versão do modelo) para
  auditoria e para formar a base de dados de retreino futuro.
- **Monitoramento de drift de dados**: comparar a distribuição das features de
  entrada em produção com a distribuição vista no treino (ex.: `evidently`,
  KS-test por feature). Um CEP novo ou uma mudança no perfil de imóveis
  avaliados é um sinal de alerta.
- **Monitoramento de performance real**: como o preço de venda real de um
  imóvel só é conhecido semanas/meses depois da predição (ex.: quando a venda
  se concretiza), o pipeline deve religar `id`/endereço da predição com o
  preço de venda realizado assim que disponível, para recalcular
  RMSE/MAE em produção — não apenas confiar nas métricas de treino.
- Dashboards (Grafana/Looker) e alertas (Slack/PagerDuty) quando drift ou erro
  real ultrapassam limites definidos.

### 5. Versionamento
- **Código**: Git + tags de release.
- **Dados**: versionamento leve (ex.: DVC ou snapshot com data no nome do
  arquivo/partição) para poder reproduzir exatamente o dataset usado em cada
  modelo treinado.
- **Modelo**: cada modelo em produção tem um ID único (registry), permitindo
  rollback imediato para a versão anterior se o novo modelo apresentar
  problemas.
