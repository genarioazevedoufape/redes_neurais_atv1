# 🏀 NBA Predictor: Análise Preditiva com Regressão Linear e Logística

## 📋 Sobre o Projeto

Sistema de análise preditiva de dados da NBA utilizando técnicas de Regressão Linear e Logística. O projeto permite analisar estatísticas de times da NBA, prever resultados de jogos e visualizar tendências através de gráficos interativos.

### 🎓 Informações Acadêmicas

- **Disciplina:** Redes Neurais
- **Curso:** Ciências da Computação
- **Professor:** Ryan Azevedo
- **Discentes:**
  - Genário Azevedo
  - Matheus Henrique
  - João Victor Iane

---

## 🎯 Funcionalidades

### Regressão Linear
- Predição de variáveis numéricas (pontos, rebotes, assistências, etc.)
- Visualização de diagramas de dispersão com linhas de regressão
- Cálculo de métricas: R², MSE, RMSE
- Análise de coeficientes e equação da regressão
- Gráfico de importância de variáveis

### Regressão Logística
- Predição de probabilidade de vitória/derrota
- Curva ROC e cálculo de AUC
- Matriz de confusão
- Curvas sigmoides para visualização não-linear
- Métricas: Acurácia, Precisão, Recall, F1-Score

### Recursos Adicionais
- Gráfico de tendência com média móvel e intervalo de confiança
- Seleção de múltiplas variáveis independentes
- Configurações avançadas (tamanho do conjunto de teste, janela de média móvel, threshold)
- Visualizações interativas com Plotly
- Cache de dados para melhor performance

---

## 🛠️ Tecnologias Utilizadas

### Bibliotecas Principais
- **Streamlit** - Interface web interativa
- **NBA API** - Coleta de dados da NBA
- **Scikit-learn** - Modelos de machine learning
- **Statsmodels** - Análise estatística avançada
- **Plotly** - Visualizações interativas
- **Pandas/NumPy** - Manipulação de dados
- **Matplotlib/Seaborn** - Gráficos estatísticos

---

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/nba-predictor.git
cd nba-predictor
```

### 2. Crie um ambiente virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute a aplicação
```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`

---

## 📁 Estrutura do Projeto

```
nba-predictor/
│
├── app.py                          # Aplicação principal Streamlit
│
├── data/
│   └── nba_data_loader.py         # Carregamento de dados da NBA API
│
├── models/
│   ├── linear_regression_model.py  # Modelo de Regressão Linear
│   └── logistic_regression_model.py # Modelo de Regressão Logística
│
├── utils/
│   ├── preprocessing.py            # Pré-processamento de dados
│   └── visualization.py            # Funções de visualização
│
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação
```

---

## 🚀 Como Usar

### 1. Selecione o Tipo de Regressão
- **Linear:** Para prever valores numéricos (pontos, rebotes, etc.)
- **Logística:** Para prever probabilidade de vitória

### 2. Escolha a Equipe
Selecione um dos 30 times da NBA disponíveis na lista.

### 3. Configure as Variáveis

**Para Regressão Linear:**
- Escolha a variável dependente (Y): PTS, REB, AST, etc.
- Selecione múltiplas variáveis independentes (X)

**Para Regressão Logística:**
- Variável dependente fixada em WIN (Vitória/Derrota)
- Selecione as variáveis preditoras

### 4. Ajuste Configurações Avançadas
- **Tamanho do Conjunto de Teste:** 10% a 50% dos dados
- **Janela da Média Móvel:** 3 a 10 jogos
- **Threshold** (Logística): Probabilidade mínima para classificar como vitória

### 5. Execute a Análise
Clique em "▶️ Executar Análise" e visualize os resultados.

---

## 📊 Métricas e Interpretação

### Regressão Linear
- **R² (Coeficiente de Determinação):** Proporção da variância explicada (0 a 1, quanto maior melhor)
- **MSE (Mean Squared Error):** Erro quadrático médio (quanto menor melhor)
- **RMSE (Root Mean Squared Error):** Raiz do MSE, mesma unidade da variável

### Regressão Logística
- **Acurácia:** Percentual de predições corretas
- **Precisão:** Taxa de verdadeiros positivos entre os classificados como positivos
- **Recall:** Taxa de verdadeiros positivos identificados
- **F1-Score:** Média harmônica entre precisão e recall
- **AUC-ROC:** Área sob a curva ROC (0.5 a 1, quanto maior melhor)

---

## 🎨 Visualizações Disponíveis

### Regressão Linear
1. **Diagrama de Dispersão** - Relação entre variáveis com linha de regressão
2. **Previsão vs Realidade** - Comparação de valores preditos e reais
3. **Importância de Variáveis** - Impacto de cada variável no modelo
4. **Tendência Temporal** - Evolução ao longo da temporada

### Regressão Logística
1. **Gráfico de Probabilidades** - Probabilidades previstas por jogo
2. **Curva ROC** - Performance do classificador
3. **Matriz de Confusão** - Análise de erros e acertos
4. **Curvas Sigmoides** - Relação não-linear entre variáveis
5. **Diagrama de Dispersão Múltiplo** - Análise de várias variáveis

---

## ⚠️ Problemas Conhecidos e Soluções

### Erro de Matriz Singular
**Causa:** Multicolinearidade ou mais variáveis que observações

**Solução:**
- Remova variáveis altamente correlacionadas
- Reduza o número de variáveis independentes
- Tente diferentes combinações

### Curva Sigmoide Aparecendo Reta
**Causa:** Dados não escalonados no gráfico

**Solução:** O código já trata o escalonamento automaticamente

### Dados Insuficientes
**Causa:** NBA API pode não ter dados completos da temporada atual

**Solução:**
- Aguarde o início da temporada regular
- Tente com outras equipes que já tenham jogos registrados

---

## 🔧 Requisitos do Sistema

- **Python:** 3.8 ou superior
- **Memória RAM:** 4GB mínimo
- **Conexão Internet:** Necessária para carregar dados da NBA API
- **Navegador:** Chrome, Firefox, Safari ou Edge (versões recentes)

---

## 📚 Conceitos Implementados

### Regressão Linear Múltipla
Técnica estatística para modelar a relação entre múltiplas variáveis independentes e uma variável dependente contínua.

**Equação:** y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

### Regressão Logística
Modelo de classificação que estima a probabilidade de um evento binário usando a função sigmóide.

**Equação:** p = 1 / [1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ...))]

### Pré-processamento
- Normalização com StandardScaler
- Divisão treino/teste estratificada
- Tratamento de valores ausentes
- Remoção de variáveis de variância zero

---

## 📞 Infomações

UNIVERSIDADE FEDERAL DO AGRESTE DE PERNAMBUCO

- **Professor:** Ryan Azevedo
- **Alunos:** Genário Azevedo, Matheus Henrique, João Victor Iane

---

## 📖 Referências

- [Documentação Streamlit](https://docs.streamlit.io/)
- [NBA API Documentation](https://github.com/swar/nba_api)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Statsmodels Documentation](https://www.statsmodels.org/)

---

**Desenvolvido com 🏀 e 💻 para a disciplina de Redes Neurais**
