# 🏀 NBA Predictor: Análise Preditiva com Regressão

Este projeto implementa uma aplicação modular em Python, utilizando o framework **Streamlit** para a interface interativa e as bibliotecas **scikit-learn** e **statsmodels** para a análise preditiva de dados da NBA (temporada 2024-2025).

A aplicação permite que o usuário realize análises de **Regressão Linear Múltipla** e **Regressão Logística** para prever estatísticas de times da NBA.

## ⚙️ Tecnologias e Dependências

O projeto foi desenvolvido em **Python 3.10+** e requer as seguintes bibliotecas, listadas no arquivo `requirements.txt`:

*   `nba_api`: Para coleta de dados em tempo real da NBA.
*   `pandas`, `numpy`: Para manipulação e cálculo de dados.
*   `scikit-learn`: Para a implementação dos modelos de Regressão Linear e Logística.
*   `statsmodels`: Para o resumo estatístico avançado (OLS/Logit).
*   `streamlit`: Para a construção da interface web interativa.
*   `plotly`, `matplotlib`, `seaborn`: Para visualizações interativas e estáticas.

## 🧩 Estrutura do Projeto

O projeto segue uma arquitetura modular para facilitar a manutenção e o desenvolvimento:

```
nba_regression_app/
│
├── data/
│   ├── nba_data_loader.py          # Funções para baixar e preparar dados via nba_api (com cache @st.cache_data)
│
├── models/
│   ├── linear_regression_model.py  # Implementação da Regressão Linear Múltipla
│   └── logistic_regression_model.py # Implementação da Regressão Logística
│
├── utils/
│   ├── preprocessing.py            # Limpeza, normalização (StandardScaler) e divisão de variáveis
│   └── visualization.py            # Funções para gerar gráficos (Plotly e Matplotlib/Seaborn)
│
├── app.py                          # Interface Streamlit (menu principal e orquestração)
└── requirements.txt                # Lista de dependências
```

## ▶️ Como Rodar a Aplicação

### 1. Instalação das Dependências

Certifique-se de estar no diretório `nba_regression_app` e instale as dependências:

```bash
cd nba_regression_app
pip install -r requirements.txt
```

### 2. Execução

Execute a aplicação Streamlit a partir do diretório raiz do projeto:

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no seu navegador padrão (geralmente em `http://localhost:8501`).

## 🎯 Funcionalidades

A aplicação oferece as seguintes funcionalidades principais:

1.  **Seleção de Time e Tipo de Regressão:** O usuário escolhe o time e o tipo de análise (Linear ou Logística).
2.  **Seleção de Variáveis:** O usuário define a variável dependente (Y) e as variáveis independentes (X).
    *   *Nota:* Para a Regressão Logística, a variável Y é fixada em `WIN` (Vitória/Derrota).
3.  **Análise e Métricas:** Exibição da equação da regressão (Linear), coeficientes (β) e métricas de avaliação (R², RMSE, Acurácia, F1-Score, etc.).
4.  **Visualizações Interativas:**
    *   Diagrama de Dispersão com Linha de Regressão (Linear).
    *   Gráfico de Previsão vs. Realidade.
    *   Matriz de Confusão (Logística).
    *   Gráfico de Tendência com Média Móvel.
5.  **Modo Avançado:** Opção para visualizar o resumo estatístico completo gerado pelo `statsmodels`.

