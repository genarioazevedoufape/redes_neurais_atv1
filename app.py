from matplotlib.pylab import LinAlgError
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Adicionar o diretório raiz do projeto ao path para importação modular
try:
    from data.nba_data_loader import get_available_teams, get_team_id, load_team_game_log, get_available_stats_columns
    from utils.preprocessing import prepare_data
    from utils.visualization import (
        plot_regression_line, plot_prediction_vs_reality, plot_confusion_matrix, 
        plot_trend_with_confidence, plot_roc_curve, plot_calibration_curve,
        plot_feature_importance, plot_residuals  
    )
    from models.linear_regression_model import LinearRegressionModel
    from models.logistic_regression_model import LogisticRegressionModel
except ImportError:
    # Se a importação falhar, tentamos adicionar o diretório pai (onde os módulos estão)
    sys.path.append(os.path.dirname(__file__))
    from data.nba_data_loader import get_available_teams, get_team_id, load_team_game_log, get_available_stats_columns
    from utils.preprocessing import prepare_data
    from utils.visualization import (
        plot_regression_line, plot_prediction_vs_reality, plot_confusion_matrix, 
        plot_trend_with_confidence, plot_roc_curve, plot_calibration_curve,
        plot_feature_importance, plot_residuals  
    )
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
    x_options = [col for col in available_stats if col not in [y_col, 'GAME_DATE']]
    
    # Garantir que Logística não use 'WIN' em X
    if regression_type == "Logística" and 'WIN' in x_options:
        x_options.remove('WIN')
        
    x_cols = st.sidebar.multiselect(
        "4. Selecione as Variáveis Independentes (X):",
        options=x_options,
        default=x_options[:3] if len(x_options) >= 3 else x_options
    )
    
    # Configurações avançadas
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Configurações Avançadas")
    
    # Tamanho do conjunto de teste
    test_size = st.sidebar.slider(
        "Tamanho do Conjunto de Teste:",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Proporção dos dados que serão usados para teste"
    )
    
    # Janela para média móvel
    window_size = st.sidebar.slider(
        "Janela da Média Móvel:",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Número de jogos para calcular a média móvel"
    )
    
    # Threshold para classificação (apenas logística)
    if regression_type == "Logística":
        threshold = st.sidebar.slider(
            "Threshold de Classificação:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Probabilidade mínima para classificar como vitória"
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
            X_train, X_test, y_train, y_test, scaler = prepare_data(
                df_raw, y_col, x_cols, test_size=test_size
            )
            
            # Verificar se há dados suficientes
            if X_train.empty or X_test.empty:
                st.error("Dados insuficientes para treino e teste após o pré-processamento.")
            elif len(X_train) < len(x_cols) + 1:
                st.warning(f"Poucos dados ({len(X_train)}) para o número de variáveis ({len(x_cols)}). Tente reduzir as variáveis independentes.")
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
                    y_pred_class = model.predict_class(X_test, threshold=threshold)
                    metrics = model.evaluate(y_test, y_pred_class, y_pred_proba)
                    
                # 3. Exibição de Métricas e Coeficientes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Métricas de Avaliação")
                    metrics_df = pd.DataFrame(metrics.items(), columns=['Métrica', 'Valor'])
                    st.dataframe(metrics_df.style.format({'Valor': "{:.4f}"}), hide_index=True)
                
                with col2:
                    st.subheader("🔢 Coeficientes da Regressão (β)")
                    df_coef = model.get_coefficients(x_cols)
                    st.dataframe(df_coef.style.format({'Coeficiente (β)': "{:.4f}"}), hide_index=True)
                
                # Equação da Regressão (apenas para Linear)
                if regression_type == "Linear":
                    st.subheader("📐 Equação da Regressão")
                    st.code(model.get_equation(x_cols), language='markdown')
                
                # 4. Visualizações Principais
                st.header("📈 Visualizações")
                
                # Gráficos principais baseados no tipo de modelo
                if regression_type == "Linear":
                    st.subheader("🔍 Relações Individuais: Variáveis vs " + y_col)
                    st.plotly_chart(
                        plot_regression_line(df_raw, x_cols, y_col), 
                        use_container_width=True
                    )

                    # Importância das Features
                    st.subheader("🎯 Importância das Variáveis")
                    st.plotly_chart(plot_feature_importance(model.model.coef_, x_cols, regression_type), use_container_width=True)
                    
                    # Os gráficos existentes continuam abaixo...
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Previsão vs Realidade")
                        st.plotly_chart(plot_prediction_vs_reality(y_test, y_pred, regression_type), use_container_width=True)
                    
                    with col2:
                        st.subheader("📉 Gráfico de Resíduos")
                        st.plotly_chart(plot_residuals(y_test, y_pred, regression_type), use_container_width=True)
                    

                else:
                    # Gráficos para Regressão Logística
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Probabilidade de Vitória")
                        st.plotly_chart(plot_prediction_vs_reality(y_test, y_pred_proba, regression_type), use_container_width=True)
                        
                    with col2:
                        st.subheader("📈 Curva ROC")
                        st.plotly_chart(plot_roc_curve(y_test, y_pred_proba), use_container_width=True)
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.subheader("🎯 Matriz de Confusão")
                        st.pyplot(plot_confusion_matrix(y_test, y_pred_class), use_container_width=True)
                        
                    with col4:
                        st.subheader("⚖️ Curva de Calibração")
                        st.plotly_chart(plot_calibration_curve(y_test, y_pred_proba), use_container_width=True)
                    
                    # Importância das Features
                    st.subheader("📊 Importância das Variáveis")
                    st.plotly_chart(plot_feature_importance(model.model.coef_[0], x_cols, regression_type), use_container_width=True)
                
                # 6. Análise de Tendência
                st.subheader(f"📈 Tendência de {y_col} ao Longo do Tempo")
                st.plotly_chart(
                    plot_trend_with_confidence(df_raw, 'GAME_DATE', y_col, window=window_size), 
                    use_container_width=True
                )
       
        except LinAlgError as e:
            if "singular matrix" in str(e).lower():
                st.error("""
                **❌ Erro de Matriz Singular**: Isso geralmente ocorre quando:
                - Há multicolinearidade (variáveis muito correlacionadas)
                - Mais variáveis do que observações
                - Variáveis com variância zero
                
                **💡 Soluções**: 
                - Remova variáveis altamente correlacionadas
                - Reduza o número de variáveis independentes
                - Tente diferentes combinações de variáveis
                """)
            else:
                st.error(f"Erro de álgebra linear: {e}")
        except ValueError as e:
            st.error(f"Erro nos dados: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro durante a análise: {e}")
            st.exception(e)
            
    elif run_analysis and (not y_col or not x_cols):
        st.warning("⚠️ Por favor, selecione a Variável Dependente (Y) e pelo menos uma Variável Independente (X) para executar a análise.")

else:
    st.info("👆 Selecione uma equipe na sidebar para começar a análise.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🏀 Desenvolvido com NBA API e Streamlit • Análise Preditiva de Dados da NBA
    </div>
    """,
    unsafe_allow_html=True
)