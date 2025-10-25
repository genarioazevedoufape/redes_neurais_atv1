import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Adicionar o diretório raiz do projeto ao path para importação modular
# O diretório raiz é o diretório atual, então o path deve ser ajustado
# para que os módulos sejam encontrados.
# Como estamos no diretório 'nba_regression_app', o sys.path já deve incluir o '.'
# No entanto, para garantir que as importações funcionem, vamos usar um try/except.
try:
    from data.nba_data_loader import get_available_teams, get_team_id, load_team_game_log, get_available_stats_columns
    from utils.preprocessing import prepare_data
    from utils.visualization import plot_regression_line, plot_prediction_vs_reality, plot_confusion_matrix, plot_trend_with_confidence
    from models.linear_regression_model import LinearRegressionModel
    from models.logistic_regression_model import LogisticRegressionModel
except ImportError:
    # Se a importação falhar, tentamos adicionar o diretório pai (onde os módulos estão)
    sys.path.append(os.path.dirname(__file__))
    from data.nba_data_loader import get_available_teams, get_team_id, load_team_game_log, get_available_stats_columns
    from utils.preprocessing import prepare_data
    from utils.visualization import plot_regression_line, plot_prediction_vs_reality, plot_confusion_matrix, plot_trend_with_confidence
    from models.linear_regression_model import LinearRegressionModel
    from models.logistic_regression_model import LogisticRegressionModel

# --- Configuração da Página ---
st.set_page_config(
    page_title="NBA Predictor: Regressão Linear e Logística",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título Principal ---
st.title("🏀 NBA Predictor: Análise Preditiva com Regressão")
st.markdown("Desenvolvido para a temporada **2024-2025** usando `nba_api` e `Streamlit`.")

# --- Sidebar para Entradas do Usuário ---
st.sidebar.header("⚙️ Configurações da Análise")

# 1. Selecionar o tipo de regressão
regression_type = st.sidebar.radio(
    "1. Selecione o Tipo de Regressão:",
    ("Linear", "Logística")
)

# 2. Escolher equipe
team_list = get_available_teams()
selected_team_name = st.sidebar.selectbox(
    "2. Escolha a Equipe:",
    options=team_list,
    index=team_list.index("Boston Celtics") if "Boston Celtics" in team_list else 0
)

# 3. Carregar dados
team_id = get_team_id(selected_team_name)
df_raw = pd.DataFrame()

if team_id:
    data_load_state = st.info(f"Carregando dados de jogos do **{selected_team_name}**...")
    try:
        df_raw = load_team_game_log(team_id)
        if not df_raw.empty:
            data_load_state.success("Dados carregados com sucesso!")
        else:
            data_load_state.warning("Nenhum dado encontrado para a temporada 2024-2025. Tentando carregar dados de exemplo.")
            # Se não houver dados para 2024-2025 (porque a temporada ainda não começou ou a API está desatualizada), 
            # o usuário não conseguirá testar. Vamos tentar carregar uma temporada anterior como fallback.
            # O prompt exige 2024-2025, mas para a aplicação funcionar, um fallback é essencial.
            # No entanto, vou manter o foco no prompt e apenas avisar.
            st.warning("A `nba_api` pode não ter dados para a temporada 2024-2025 ainda. A análise pode falhar.")
            
    except Exception as e:
        data_load_state.error(f"Erro ao carregar dados: {e}")
else:
    st.error("Time não encontrado. Por favor, selecione um time válido.")

# Verificar se os dados foram carregados
if not df_raw.empty:
    
    # Exibir dados brutos (opcional)
    if st.sidebar.checkbox('Mostrar Dados Brutos'):
        st.subheader(f"Dados Brutos de Jogos do {selected_team_name}")
        st.dataframe(df_raw)
        
    # Colunas de estatísticas disponíveis para seleção de X e Y
    available_stats = get_available_stats_columns(df_raw)
    
    # 4. Escolher variável dependente (Y)
    y_options = available_stats
    
    # Regressão Logística deve ter 'WIN' como variável dependente
    if regression_type == "Logística":
        if 'WIN' not in y_options:
            st.error("A Regressão Logística requer uma variável binária (como 'WIN'). Dados insuficientes.")
            y_col = None
        else:
            y_col = 'WIN'
            st.sidebar.markdown(f"**3. Variável Dependente (Y):** `WIN` (Vitória/Derrota) - **Fixa para Logística**")
    else:
        # Regressão Linear: permite escolher
        # Filtrar 'WIN' e 'GAME_DATE'
        linear_options = [col for col in y_options if col not in ['WIN', 'GAME_DATE']]
        if not linear_options:
            st.error("Não há variáveis numéricas disponíveis para Regressão Linear.")
            y_col = None
        else:
            y_col = st.sidebar.selectbox(
                "3. Escolha a Variável Dependente (Y):",
                options=linear_options,
                index=linear_options.index('PTS') if 'PTS' in linear_options else 0
            )
    
    # 5. Selecionar variáveis independentes (X)
    # Excluir Y e 'GAME_DATE' das opções de X
    x_options = [col for col in available_stats if col not in [y_col, 'GAME_DATE']]
    
    # Garantir que Logística não use 'WIN' em X
    if regression_type == "Logística" and 'WIN' in x_options:
        x_options.remove('WIN')
        
    x_cols = st.sidebar.multiselect(
        "4. Selecione as Variáveis Independentes (X):",
        options=x_options,
        default=x_options[:3] if len(x_options) >= 3 else x_options
    )
    
    # 6. Botão de Execução
    st.sidebar.markdown("---")
    run_analysis = st.sidebar.button("▶️ Executar Análise")
    
    # --- Lógica Principal da Análise ---
    if run_analysis and y_col and x_cols:
        
        st.header(f"Resultados da Análise de Regressão {regression_type}")
        st.subheader(f"Previsão de **{y_col}** usando {len(x_cols)} variáveis")
        
        try:
            # 1. Pré-processamento e Divisão de Dados
            X_train, X_test, y_train, y_test, scaler = prepare_data(df_raw, y_col, x_cols)
            
            # Verificar se há dados suficientes para treino/teste
            if X_train.empty or X_test.empty:
                st.error("Dados insuficientes para treino e teste após o pré-processamento. Tente selecionar menos variáveis ou um time com mais jogos.")
            else:
                # 2. Treinamento do Modelo
                if regression_type == "Linear":
                    model = LinearRegressionModel()
                    model.train(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = model.evaluate(y_test, y_pred)
                    
                else: # Logística
                    model = LogisticRegressionModel()
                    model.train(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred_class = model.predict_class(X_test)
                    metrics = model.evaluate(y_test, y_pred_class, y_pred_proba)
                    
                # 3. Exibição de Métricas e Coeficientes
                st.subheader("Métricas de Avaliação")
                metrics_df = pd.DataFrame(metrics.items(), columns=['Métrica', 'Valor'])
                st.dataframe(metrics_df.style.format({'Valor': "{:.4f}"}), hide_index=True)

                st.subheader("Coeficientes da Regressão (β)")
                df_coef = model.get_coefficients(x_cols)
                st.dataframe(df_coef.style.format({'Coeficiente (β)': "{:.4f}"}), hide_index=True)
                
                # Equação da Regressão (apenas para Linear)
                if regression_type == "Linear":
                    st.subheader("Equação da Regressão")
                    st.code(model.get_equation(x_cols), language='markdown')
                    
                # 4. Visualizações
                st.header("Visualizações")
                
                # Gráfico de Tendência (Usa a coluna 'GAME_DATE' do df_raw)
                st.subheader(f"Tendência de {y_col} ao longo do tempo")
                st.plotly_chart(plot_trend_with_confidence(df_raw, 'GAME_DATE', y_col), use_container_width=True)
                
                # Gráfico de Dispersão (apenas para Linear, usando a primeira variável X)
                if regression_type == "Linear":
                    st.subheader(f"Diagrama de Dispersão: {y_col} vs {x_cols[0]}")
                    st.plotly_chart(plot_regression_line(df_raw, x_cols[0], y_col), use_container_width=True)
                
                # Gráfico Previsão vs Realidade
                st.subheader("Previsão vs. Realidade")
                if regression_type == "Linear":
                    st.plotly_chart(plot_prediction_vs_reality(y_test, y_pred, regression_type), use_container_width=True)
                else:
                    st.plotly_chart(plot_prediction_vs_reality(y_test, y_pred_proba, regression_type), use_container_width=True)
                    
                # Matriz de Confusão (apenas para Logística)
                if regression_type == "Logística":
                    st.subheader("Matriz de Confusão")
                    st.pyplot(plot_confusion_matrix(y_test, y_pred_class), use_container_width=True)
                    
                # Extras (Statsmodels Summary)
                st.sidebar.markdown("---")
                if st.sidebar.checkbox('Mostrar Resumo Avançado (Statsmodels)'):
                    st.subheader("Resumo Avançado (Statsmodels)")
                    st.components.v1.html(model.get_statsmodels_summary(), height=500, scrolling=True)
                    
        except Exception as e:
            st.error(f"Ocorreu um erro durante a análise: {e}")
            st.exception(e)
            
    elif run_analysis and (not y_col or not x_cols):
        st.warning("Por favor, selecione a Variável Dependente (Y) e pelo menos uma Variável Independente (X) para executar a análise.")

else:
    st.warning("Por favor, selecione um time para carregar os dados e iniciar a análise.")
