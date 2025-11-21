import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

from data.nba_data_loader import load_player_game_log
from utils.visualization import plot_player_ranking, plot_rolling_performance, plot_team_vs_opponent

# ------------------------------------------------------------------
# IMPORTAÇÕES (com fallback caso rode fora da estrutura de pastas)
# ------------------------------------------------------------------
try:
    from data.nba_data_loader import get_available_teams, get_team_id, load_team_game_log, get_available_stats_columns
    from utils.preprocessing import prepare_data
    from utils.visualization import (
        plot_regression_line, plot_prediction_vs_reality, plot_confusion_matrix,
        plot_trend_with_confidence, plot_roc_curve, plot_feature_importance,
        plot_multiple_logistic_curves, plot_logistic_sigmoid_curve,
        # --- NOVOS GRÁFICOS DA MLP ---
        plot_mlp_prediction_vs_reality, plot_training_history_smoothed,
        plot_probability_histogram, plot_predicted_vs_actual_scatter,
        plot_bootstrap_confidence, plot_model_comparison_timeline
    )
    from models.linear_regression_model import LinearRegressionModel
    from models.logistic_regression_model import LogisticRegressionModel
    from models.mlp_model import MLPModel
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from data.nba_data_loader import get_available_teams, get_team_id, load_team_game_log, get_available_stats_columns
    from utils.preprocessing import prepare_data
    from utils.visualization import (
        plot_regression_line, plot_prediction_vs_reality, plot_confusion_matrix,
        plot_trend_with_confidence, plot_roc_curve, plot_feature_importance,
        plot_multiple_logistic_curves, plot_logistic_sigmoid_curve,
        plot_mlp_prediction_vs_reality, plot_training_history_smoothed,
        plot_probability_histogram, plot_predicted_vs_actual_scatter,
        plot_bootstrap_confidence, plot_model_comparison_timeline
    )
    from models.linear_regression_model import LinearRegressionModel
    from models.logistic_regression_model import LogisticRegressionModel
    from models.mlp_model import MLPModel

# ------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NBA Predictor Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("NBA Predictor Pro: Regressão Linear • Logística • Rede Neural (MLP)")

# ------------------------------------------------------------------
# INICIALIZAÇÃO DO SESSION STATE
# ------------------------------------------------------------------
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'player_data_loaded' not in st.session_state:
    st.session_state.player_data_loaded = False
if 'df_players_agg' not in st.session_state:
    st.session_state.df_players_agg = None
if 'df_players_games' not in st.session_state:
    st.session_state.df_players_games = None
if 'selected_metric' not in st.session_state:
    st.session_state.selected_metric = 'PTS_mean'
if 'top_n' not in st.session_state:
    st.session_state.top_n = 10

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("Configurações da Análise")

regression_type = st.sidebar.radio(
    "Tipo de Modelo:",
    ("Regressão Linear", "Regressão Logística", "MLP (Rede Neural)")
)

team_list = get_available_teams()
selected_team_name = st.sidebar.selectbox(
    "Escolha a Equipe:",
    options=team_list,
    index=team_list.index("Boston Celtics") if "Boston Celtics" in team_list else 0
)

team_id = get_team_id(selected_team_name)
df_raw = pd.DataFrame()

if team_id:
    with st.spinner(f"Carregando jogos do {selected_team_name}..."):
        try:
            df_raw = load_team_game_log(team_id)
            if not df_raw.empty:
                st.success("Dados carregados com sucesso!")
            else:
                st.warning("Sem dados recentes. Usando cache ou exemplo.")
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")

if df_raw.empty:
    st.info("Selecione uma equipe para começar.")
    st.stop()

# Mostrar dados brutos (opcional)
if st.sidebar.checkbox("Mostrar dados brutos"):
    st.subheader("Dados Brutos")
    st.dataframe(df_raw)

available_stats = get_available_stats_columns(df_raw)

# Variável dependente (Y)
if regression_type in ["Regressão Logística", "MLP (Rede Neural)"]:
    if 'WIN' not in available_stats:
        st.error("Coluna 'WIN' não encontrada. Não é possível treinar classificação.")
        st.stop()
    y_col = 'WIN'
    st.sidebar.markdown("**Variável Dependente (Y):** `WIN` (Vitória = 1)")
else:
    linear_options = [col for col in available_stats if col not in ['WIN', 'GAME_DATE']]
    y_col = st.sidebar.selectbox("Variável Dependente (Y):", linear_options, index=linear_options.index('PTS') if 'PTS' in linear_options else 0)

# Variáveis independentes (X)
x_options = [col for col in available_stats if col not in [y_col, 'GAME_DATE']]
x_cols = st.sidebar.multiselect(
    "Variáveis Independentes (X):",
    options=x_options,
    default=x_options[:5] if len(x_options) >= 5 else x_options
)

if not x_cols:
    st.warning("Selecione pelo menos uma variável independente.")
    st.stop()

# Configurações avançadas
st.sidebar.markdown("---")
test_size = st.sidebar.slider("Tamanho do conjunto de teste:", 0.1, 0.5, 0.2, 0.05)
window_size = st.sidebar.slider("Janela da média móvel:", 3, 10, 5)

if regression_type != "Regressão Linear":
    threshold = st.sidebar.slider("Threshold de classificação:", 0.1, 0.9, 0.5, 0.05)

run_analysis = st.sidebar.button("Executar Análise", type="primary")

# ------------------------------------------------------------------
# EXECUÇÃO DA ANÁLISE
# ------------------------------------------------------------------
if run_analysis or st.session_state.analysis_complete:
    
    # Se é uma nova análise, executa o processamento completo
    if run_analysis:
        st.session_state.analysis_complete = True
        st.session_state.player_data_loaded = False  # Reseta os dados de jogadores
        
        st.header(f"Análise: {regression_type}")
        st.write(f"**Previsão de:** `{y_col}` → usando {len(x_cols)} variáveis")

        try:
            X_train, X_test, y_train, y_test, scaler = prepare_data(
                df_raw, y_col, x_cols, test_size=test_size
            )

            if X_train.empty or len(X_train) < 10:
                st.error("Dados insuficientes após pré-processamento.")
                st.stop()

            # =============================================
            # TREINAMENTO DOS MODELOS
            # =============================================
            if regression_type == "Regressão Linear":
                model = LinearRegressionModel()
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = model.evaluate(y_test, y_pred)

            elif regression_type == "Regressão Logística":
                model = LogisticRegressionModel()
                model.train(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)
                y_pred_class = model.predict_class(X_test, threshold=threshold)
                metrics = model.evaluate(y_test, y_pred_class, y_pred_proba)

            else:  # MLP
                input_dim = X_train.shape[1]
                model = MLPModel(input_dim=input_dim)
                model.build_model(hidden_layers=2, neurons=64, activation='relu', optimizer_name='Adam')

                with st.spinner("Treinando Rede Neural (MLP)... Aguarde até 30s"):
                    model.train(X_train.values, y_train.values, epochs=200, batch_size=4, validation_split=0.2)
                st.success("Rede Neural treinada!")

                y_pred_proba = model.predict_proba(X_test.values)
                y_pred_class = model.predict_class(X_test.values, threshold=threshold)
                metrics = model.evaluate(y_test, y_pred_class, y_pred_proba)

            # =============================================
            # MÉTRICAS
            # =============================================
            st.subheader("Métricas de Desempenho")
            metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
            st.dataframe(metrics_df.style.format({"Valor": "{:.4f}"}))

            # =============================================
            # VISUALIZAÇÕES ESPECÍFICAS POR MODELO
            # =============================================
            st.markdown("---")
            st.header("Análise Visual")

            # ---------- REGRESSÃO LINEAR ----------
            if regression_type == "Regressão Linear":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Dispersão com Linha de Regressão")
                    st.plotly_chart(plot_regression_line(df_raw, x_cols, y_col), use_container_width=True)
                with col2:
                    st.subheader("Importância das Variáveis")
                    st.plotly_chart(plot_feature_importance(model.model.coef_, x_cols, "Linear"), use_container_width=True)

                st.subheader("Previsão vs Realidade")
                st.plotly_chart(plot_prediction_vs_reality(y_test, y_pred, "Linear"), use_container_width=True)

            # ---------- REGRESSÃO LOGÍSTICA ----------
            elif regression_type == "Regressão Logística":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Probabilidades Previstas")
                    st.plotly_chart(plot_prediction_vs_reality(y_test, y_pred_proba, "Logística"), use_container_width=True)
                with col2:
                    st.subheader("Curva ROC")
                    st.plotly_chart(plot_roc_curve(y_test, y_pred_proba), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Matriz de Confusão")
                    st.pyplot(plot_confusion_matrix(y_test, y_pred_class))
                with col2:
                    st.subheader("Importância das Variáveis")
                    st.plotly_chart(plot_feature_importance(model.model.coef_[0], x_cols, "Logística"), use_container_width=True)

                st.subheader("Curvas Sigmoides")
                st.plotly_chart(plot_multiple_logistic_curves(df_raw, x_cols, y_col, model=model), use_container_width=True)

            # ---------- MLP (REDE NEURAL) ----------
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Evolução do Erro (com suavização)")
                    history_df = model.get_history_df()
                    if not history_df.empty:
                        st.plotly_chart(plot_training_history_smoothed(history_df), use_container_width=True)
                with col2:
                    st.subheader("Previsão vs Realidade (MLP)")
                    st.plotly_chart(plot_mlp_prediction_vs_reality(y_test, y_pred_proba, y_pred_class), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Histograma das Probabilidades")
                    st.plotly_chart(plot_probability_histogram(y_test, y_pred_proba), use_container_width=True)
                with col2:
                    st.subheader("Classe Predita × Real")
                    st.plotly_chart(plot_predicted_vs_actual_scatter(y_test, y_pred_class), use_container_width=True)

                st.subheader("Matriz de Confusão")
                st.pyplot(plot_confusion_matrix(y_test, y_pred_class))

                st.subheader("Intervalo de Confiança 95% (Bootstrap)")
                st.plotly_chart(plot_bootstrap_confidence(y_pred_proba), use_container_width=True)

       
                st.subheader("Evolução Temporal: Resultado Real vs Previsão MLP")

                # Passo 1: Pegar os dados de teste com seus índices originais
                df_test_with_date = df_raw.loc[X_test.index].copy()

                # Passo 2: Adicionar as previsões e os valores reais
                df_test_with_date = df_test_with_date.assign(
                    Real=y_test.values,         # valores reais (na ordem do X_test)
                    Previsão_Probabilidade=y_pred_proba    # probabilidades previstas
                )

                # Passo 3: Ordenar por data
                df_plot = df_test_with_date.sort_values('GAME_DATE').reset_index(drop=True)

                # Passo 4: Plotar
                fig = go.Figure()

                # Linha do resultado real (0 ou 1)
                fig.add_trace(go.Scatter(
                    x=df_plot['GAME_DATE'],
                    y=df_plot['Real'],
                    mode='lines+markers',
                    name='Resultado Real (Vitória=1)',
                    line=dict(color='black', width=3),
                    marker=dict(size=8)
                            ))

                # Linha da probabilidade prevista pela MLP
                fig.add_trace(go.Scatter(
                    x=df_plot['GAME_DATE'],
                    y=df_plot['Previsão_Probabilidade'],
                    mode='lines',
                    name='Previsão MLP (probabilidade)',
                    line=dict(color='red', width=3)
                            ))

                # Linha do threshold
                fig.add_hline(y=threshold, line_dash="dash", line_color="orange",
                  annotation_text=f"Threshold = {threshold}", annotation_position="top left")

                fig.update_layout(
                    title="Comparação Temporal: Resultado Real vs Previsão da Rede Neural",
                    xaxis_title="Data do Jogo",
                    yaxis_title="Vitória (1) / Derrota (0) | Probabilidade",
                    yaxis=dict(range=[-0.1, 1.1], tickvals=[0, 0.5, 1]),
                    template="plotly_white",
                    height=550,
                    legend=dict(y=1.15, orientation='h')
                            )

                st.plotly_chart(fig, use_container_width=True)

                # Salvar dados no session state para uso posterior
                st.session_state.df_test_with_date = df_test_with_date
                st.session_state.y_test = y_test
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.y_pred_class = y_pred_class

            # =============================================
            # GRÁFICO COMUM: TENDÊNCIA TEMPORAL
            # =============================================
            st.markdown("---")
            st.subheader("Tendência Temporal da Performance")
            st.plotly_chart(plot_trend_with_confidence(df_raw, 'GAME_DATE', y_col, window=window_size),
                            use_container_width=True)

        except Exception as e:
            st.error(f"Erro durante a análise: {e}")
            st.exception(e)
    
    # Se a análise já foi completada, mostra apenas os resultados
    elif st.session_state.analysis_complete:
        st.header(f"Análise: {regression_type}")
        st.write(f"**Previsão de:** `{y_col}` → usando {len(x_cols)} variáveis")
        st.info("✅ Análise já executada. Modifique os parâmetros abaixo sem precisar reexecutar.")

    # =============================================
    # ANÁLISES ADICIONAIS (apenas para MLP)
    # =============================================
    if regression_type == "MLP (Rede Neural)" and st.session_state.analysis_complete:
        st.markdown("---")
        st.subheader("Análises Adicionais (Players e Time)")

        # 1) Ranking de Jogadores (Player-level)
        with st.expander("Ranking dos Jogadores (Player Game Log)", expanded=False):
            # Carregar dados de jogadores apenas uma vez
            if not st.session_state.player_data_loaded:
                st.write("Carregando estatísticas por jogador via NBA API...")
                
                try:
                    df_players_games, df_players_agg = load_player_game_log(team_id)
                    
                    if df_players_agg is None or df_players_agg.empty:
                        st.warning("""
                        Dados de jogadores indisponíveis no momento. Isso pode ocorrer devido a:
                        - Limitações da NBA API
                        - Temporada muito recente
                        - Problemas de conexão
                        
                        **Dica:** Tente times mais populares como 'Los Angeles Lakers' ou 'Golden State Warriors'
                        """)
                    else:
                        # Salvar no session state
                        st.session_state.df_players_agg = df_players_agg
                        st.session_state.df_players_games = df_players_games
                        st.session_state.player_data_loaded = True
                        st.success("Dados de jogadores carregados!")
                        
                except Exception as e:
                    st.error(f"Erro ao carregar dados de jogadores: {e}")
            
            # Se os dados estão carregados, mostrar interface interativa
            if st.session_state.player_data_loaded and st.session_state.df_players_agg is not None:
                # Escolher métrica para ranking
                metric_options = [c for c in st.session_state.df_players_agg.columns if c.endswith('_mean') and c != 'GAMES_PLAYED_mean']
                
                if metric_options:
                    # Usar session state para manter a seleção
                    selected_metric = st.selectbox(
                        "Escolha a métrica para ranking:", 
                        options=metric_options, 
                        index=metric_options.index(st.session_state.selected_metric) if st.session_state.selected_metric in metric_options else 0,
                        key="metric_selector"
                    )
                    
                    # Atualizar session state quando a métrica mudar
                    if selected_metric != st.session_state.selected_metric:
                        st.session_state.selected_metric = selected_metric
                    
                    # Slider para Top N
                    top_n = st.slider(
                        "Top N jogadores:", 
                        3, 20, 
                        value=st.session_state.top_n,
                        key="top_n_slider"
                    )
                    
                    # Atualizar session state quando o top_n mudar
                    if top_n != st.session_state.top_n:
                        st.session_state.top_n = top_n
                    
                    # Plotar o gráfico
                    st.plotly_chart(
                        plot_player_ranking(
                            st.session_state.df_players_agg, 
                            metric=st.session_state.selected_metric, 
                            top_n=st.session_state.top_n
                        ), 
                        use_container_width=True
                    )
                    
                    # Mostrar tabela resumo
                    st.subheader("Resumo dos Jogadores")
                    display_cols = ['PLAYER_NAME', 'GAMES_PLAYED', 'PTS_mean', 'REB_mean', 'AST_mean']
                    available_cols = [col for col in display_cols if col in st.session_state.df_players_agg.columns]
                    st.dataframe(st.session_state.df_players_agg[available_cols].head(st.session_state.top_n))
                else:
                    st.warning("Nenhuma métrica disponível para ranking.")

        # 2) Comparação Time x Adversário
        with st.expander("Comparação Time x Adversário"):
            try:
                st.plotly_chart(plot_team_vs_opponent(df_raw, stats=['PTS','REB','AST']), use_container_width=True)
            except Exception as e:
                st.warning(f"Erro ao gerar comparação Time x Adversário: {e}")

        # 3) Evolução Temporal da Performance (Rolling)
        with st.expander("Evolução Temporal da Performance (Rolling)"):
            if 'df_test_with_date' in st.session_state:
                # Usar dados salvos no session state
                df_test_with_date = st.session_state.df_test_with_date.copy()
                
                # Usar a coluna de probabilidade para o gráfico rolling
                if 'GAME_DATE' not in df_test_with_date.columns and isinstance(df_test_with_date.index, pd.DatetimeIndex):
                    df_test_with_date = df_test_with_date.reset_index().rename(columns={'index':'GAME_DATE'})
                
                st.plotly_chart(
                    plot_rolling_performance(
                        df_test_with_date, 
                        y_true_col='Real', 
                        y_pred_col='Previsão_Probabilidade',  # Usar probabilidades
                        window=window_size
                    ), 
                    use_container_width=True
                )
            else:
                st.warning("Dados de teste não disponíveis. Execute a análise novamente.")

# ------------------------------------------------------------------
# RODAPÉ
# ------------------------------------------------------------------
st.markdown("---")
st.caption("Desenvolvido para a disciplina de Redes Neurais e Deep Learning • NBA API + Streamlit + Scikit-learn + TensorFlow")

# Botão para resetar a análise
if st.session_state.analysis_complete:
    if st.sidebar.button("🔄 Resetar Análise"):
        st.session_state.analysis_complete = False
        st.session_state.player_data_loaded = False
        st.session_state.df_players_agg = None
        st.session_state.df_players_games = None
        st.session_state.selected_metric = 'PTS_mean'
        st.session_state.top_n = 10
        st.rerun()