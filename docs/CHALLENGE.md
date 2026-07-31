# Desafio de Data Science — Previsão de Preços de Casas

## Contexto
Os dados disponibilizados correspondem a propriedades residenciais **anonimizadas** da região de **Seattle (EUA)**.  
O objetivo é **prever o preço das casas** a partir de suas características físicas e informações demográficas associadas ao CEP (zipcode).

Você receberá três arquivos:

- `kc_house_data.csv`: informações físicas dos imóveis **com preço**.  
- `zipcode_demographics.csv`: dados **demográficos** agregados por código postal.  
- `future_unseen_examples.csv`: informações de imóveis **sem preço**, que devem ser utilizadas para prever os valores após o treinamento.

---

## Objetivo do Desafio
Desenvolver uma solução completa de previsão de preços de imóveis, cobrindo desde a **análise exploratória dos dados até o desenho do deploy do modelo**.

---

## Entregáveis Esperados

### 1. Análise e entendimento dos dados
- Explique o que representam as principais variáveis.  
- Mostre correlações, outliers e padrões relevantes.  
- Descreva como combinou os dados físicos e demográficos.

### 2. Desenvolvimento do modelo de Machine Learning
Inclua:
- **a. Variáveis importantes:** quais features foram mais relevantes e por quê.  
- **b. Escolha do modelo:** qual modelo foi utilizado (ex: XGBoost, Random Forest, Regressão Linear, etc.) e o motivo da escolha.  
- **c. Generalização:** como você garante que o modelo generaliza bem para novos dados (ex: uso de *train/test split*, *cross-validation*, regularização, etc.).

### 3. Estratégia de Deploy
- Desenhe um **esquema ou diagrama** mostrando como o modelo poderia ser colocado em produção.  
- Explique as principais camadas (API, infraestrutura, monitoramento, versionamento de modelo, etc.).  
- O deploy **não precisa ser implementado**, apenas documentado.

### 4. Aprendizado Contínuo
- Explique como a solução poderia **aprender com novos dados** ao longo do tempo.  
- Descreva como o modelo seria reentreinado, avaliado e substituído em produção.

### 5. Comunicação com Stakeholders
- Mostre como apresentaria os resultados para um público de negócio:  
  - exemplos: gráficos interpretáveis, métricas traduzidas em impacto de negócio, storytelling, etc.

---

## Regras e Premissas
- É permitido o uso de **qualquer linguagem ou framework** (Python, AI, etc). 
- Utilize **boas práticas de ciência de dados**.  
- O dataset é **realista, mas anonimizado** - trate-o como se fosse um projeto de cliente real.
- Não é obrigatório usar todos os dados, mas justifique suas escolhas.
- O foco é **clareza, justificativa e comunicação** técnica e de negócio.

---

## Publicação do Projeto
Ao finalizar o desafio:

1. Crie um **repositório público no GitHub**.

2. Inclua todos os arquivos do projeto (notebooks, código-fonte, README, resultados e diagramas).  

3. Envie o **link do repositório público** para nossa equipe quando estiver tudo pronto.

---

## Dica
Este desafio **avalia o raciocínio, clareza e capacidade de comunicação técnica**. 
Mais importante do que o modelo em si é **como você estrutura, justifica e explica** suas decisões.

---

## Entrega
O prazo de entrega sugerido é de 7 dias a partir do recebimento deste desafio. No entanto, entendemos que imprevistos podem acontecer — caso precise de mais tempo, pedimos apenas que nos sinalize para alinharmos um novo prazo de entrega.
Envie o link do seu repositório quando finalizar a solução.