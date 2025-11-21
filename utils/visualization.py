import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_regression_line(df: pd.DataFrame, X_cols: list, y_col: str):
    """
    Gera múltiplos Diagramas de Dispersão com Linhas de Regressão (Plotly).
    Um gráfico para cada variável independente selecionada.
    """
    if df.empty or not X_cols:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")

    # Se houver mais de 4 variáveis, seleciona as 4 primeiras para não sobrecarregar
    display_cols = X_cols[:4] if len(X_cols) > 4 else X_cols
    
    # Criar subplots
    rows = (len(display_cols) + 1) // 2  # Arredonda para cima
    cols = 2 if len(display_cols) > 1 else 1
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f"{y_col} vs {x_col}" for x_col in display_cols]
    )
    
    for i, x_col in enumerate(display_cols):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Filtrar dados sem NaN
        valid_data = df[[x_col, y_col]].dropna()
        
        if len(valid_data) > 1:
            # Scatter plot
            scatter = go.Scatter(
                x=valid_data[x_col],
                y=valid_data[y_col],
                mode='markers',
                name=f'{x_col}',
                showlegend=False,
                marker=dict(size=8, opacity=0.7)
            )
            fig.add_trace(scatter, row=row, col=col)
            
            # Linha de regressão
            try:
                # Calcular regressão linear simples
                coefficients = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
                trendline = np.poly1d(coefficients)
                x_range = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)
                
                line = go.Scatter(
                    x=x_range,
                    y=trendline(x_range),
                    mode='lines',
                    name=f'Trend {x_col}',
                    line=dict(color='red', width=3),
                    showlegend=False
                )
                fig.add_trace(line, row=row, col=col)
            except Exception:
                # Se não for possível calcular a regressão
                pass
            
            # Atualizar labels
            fig.update_xaxes(title_text=x_col, row=row, col=col)
            fig.update_yaxes(title_text=y_col, row=row, col=col)
        else:
            # Adicionar mensagem de dados insuficientes
            fig.add_annotation(
                text="Dados insuficientes",
                xref=f"x{i+1}", yref=f"y{i+1}",
                x=0.5, y=0.5, showarrow=False,
                row=row, col=col
            )
    
    fig.update_layout(
        height=400 * rows,
        title_text=f"Dispersão e Regressão: {y_col} vs Variáveis Independentes",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def plot_prediction_vs_reality(y_true: pd.Series, y_pred: np.ndarray, model_type: str):
    """
    Gera um Gráfico de Previsão vs. Realidade (Plotly).
    """
    df = pd.DataFrame({'Realidade': y_true, 'Previsão': y_pred}, index=y_true.index)
    
    if model_type == 'Linear':
        fig = px.scatter(
            df,
            x='Realidade',
            y='Previsão',
            title='Previsão vs. Realidade (Regressão Linear)',
            template="plotly_white"
        )
        # Adicionar linha de referência y=x (onde a previsão é perfeita)
        max_val = max(df['Realidade'].max(), df['Previsão'].max())
        min_val = min(df['Realidade'].min(), df['Previsão'].min())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', name='Perfeição', line=dict(dash='dash')))
    
    elif model_type == 'Logística':
        # Para Logística, y_true é binário (0 ou 1) e y_pred é a probabilidade (0 a 1)
        df['Índice'] = range(len(df))  # Índice sequencial para melhor visualização
        
        fig = px.scatter(
            df,
            x='Índice',
            y='Previsão',
            color=df['Realidade'].astype(str),
            color_discrete_map={'0': 'red', '1': 'green'},
            title='Gráfico de Probabilidades Previstas',
            labels={'Índice': 'Sequência de Jogos', 'Previsão': 'Probabilidade de Vitória', 'color': 'Resultado Real'},
            template="plotly_white"
        )
        
        # Configurações específicas para regressão logística
        fig.update_yaxes(range=[0, 1], title="Probabilidade de Vitória")
        fig.update_xaxes(title="Sequência de Jogos")
        
        # Adicionar linha de corte (threshold) em 0.5
        fig.add_hline(y=0.5, line_dash="dash", 
                     annotation_text="Threshold 0.5", 
                     annotation_position="top right")
        
        # Melhorar a legenda
        fig.update_layout(legend_title_text='Resultado Real')
        
    else:
        return go.Figure().add_annotation(text="Tipo de modelo desconhecido.")
    
    return fig

def plot_confusion_matrix(y_true: pd.Series, y_pred_class: np.ndarray):
    """
    Gera um Gráfico de Matriz de Confusão (Matplotlib/Seaborn).
    Retorna a figura do matplotlib para ser exibida no Streamlit.
    """
    cm = confusion_matrix(y_true, y_pred_class)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Previsão 0', 'Previsão 1'],
                yticklabels=['Realidade 0', 'Realidade 1'])
    ax.set_ylabel('Realidade')
    ax.set_xlabel('Previsão')
    ax.set_title('Matriz de Confusão (Regressão Logística)')
    
    return fig

def plot_trend_with_confidence(df: pd.DataFrame, date_col: str, y_col: str, window: int = 5):
    """
    Gera um Gráfico de Tendência com Média Móvel e Intervalo de Confiança.
    """
    if df.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")

    # Ordenar por data
    df_plot = df.sort_values(by=date_col).copy()
    
    # Calcular média móvel e desvio padrão
    df_plot['Média Móvel'] = df_plot[y_col].rolling(window=window, min_periods=1).mean()
    df_plot['Std'] = df_plot[y_col].rolling(window=window, min_periods=1).std().fillna(0)
    df_plot['Upper'] = df_plot['Média Móvel'] + 1.96 * df_plot['Std'] / np.sqrt(window)
    df_plot['Lower'] = df_plot['Média Móvel'] - 1.96 * df_plot['Std'] / np.sqrt(window)
    
    fig = go.Figure()
    
    # Dados originais
    fig.add_trace(go.Scatter(
        x=df_plot[date_col],
        y=df_plot[y_col],
        mode='markers',
        name=y_col,
        marker=dict(size=8, opacity=0.7, 
                   color=df_plot[y_col].map({0: 'red', 1: 'green'}) if y_col == 'WIN' else 'blue')
    ))
    
    # Média móvel
    fig.add_trace(go.Scatter(
        x=df_plot[date_col],
        y=df_plot['Média Móvel'],
        mode='lines',
        name=f'Média Móvel ({window} jogos)',
        line=dict(color='red', width=3)
    ))
    
    # Intervalo de confiança (apenas onde temos dados suficientes)
    valid_confidence = df_plot.dropna(subset=['Upper', 'Lower'])
    if len(valid_confidence) > 0:
        fig.add_trace(go.Scatter(
            x=valid_confidence[date_col].tolist() + valid_confidence[date_col].tolist()[::-1],
            y=valid_confidence['Upper'].tolist() + valid_confidence['Lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de Confiança 95%',
            showlegend=True
        ))
    
    # Configurações específicas para variável WIN
    if y_col == 'WIN':
        fig.update_yaxes(
            tickvals=[0, 1],
            ticktext=['Derrota', 'Vitória'],
            range=[-0.1, 1.1]
        )
    
    fig.update_layout(
        title=f"Gráfico de Tendência com Intervalo de Confiança",
        xaxis_title='Data do Jogo',
        yaxis_title=y_col,
        template="plotly_white",
        hovermode='x unified'
    )
    
    # Formatação do eixo X para datas
    fig.update_xaxes(
        tickformat="%b %Y",
        dtick="M1"
    )
    
    return fig

def plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray):
    """
    Gera a Curva ROC para avaliação do modelo de classificação.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # Área sob a curva
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.4f})',
        line=dict(color='darkorange', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,165,0,0.2)'
    ))
    
    # Linha de referência (classificador aleatório)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Classificador Aleatório',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Curva ROC (AUC = {roc_auc:.4f})',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        template="plotly_white",
        showlegend=True
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig

def plot_feature_importance(coef: np.ndarray, feature_names: list, model_type: str = 'Logística'):
    """
    Gera gráfico de importância das features baseado nos coeficientes.
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coef)
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Gráfico de Importância de Variáveis',
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title='Importância Absoluta',
        yaxis_title='Variáveis'
    )
    
    return fig

def plot_logistic_sigmoid_curve(df: pd.DataFrame, x_col: str, y_col: str, model=None):
    """
    Gráfico da curva sigmoide para regressão logística para UMA variável específica.
    Mostra a relação não-linear entre a variável e a probabilidade de vitória.
    """
    if df.empty or not x_col:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")

    # Filtrar dados válidos
    valid_data = df[[x_col, y_col]].dropna()
    
    if len(valid_data) < 2:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")
    
    fig = go.Figure()
    
    # Scatter plot dos dados reais (com jitter para melhor visualização)
    np.random.seed(42)
    jitter_y = valid_data[y_col] + np.random.normal(0, 0.02, len(valid_data))
    jitter_x = valid_data[x_col] + np.random.normal(0, 0.01, len(valid_data))
    
    scatter = go.Scatter(
        x=jitter_x,
        y=jitter_y,
        mode='markers',
        name='Dados Reais',
        marker=dict(
            size=8,
            opacity=0.6,
            color=valid_data[y_col].map({0: 'red', 1: 'green'}),
            line=dict(width=1, color='black')
        )
    )
    fig.add_trace(scatter)
    
    # CALCULAR E PLOTAR A CURVA SIGMOIDE
    if model is not None and hasattr(model, 'model'):
        try:
            # Obter coeficientes do modelo
            intercept = model.model.intercept_[0]
            coefficients = model.model.coef_[0]
            
            # Encontrar o índice da variável atual no modelo
            feature_names = list(model.model.feature_names_in_) if hasattr(model.model, 'feature_names_in_') else []
            if x_col in feature_names:
                coef_index = feature_names.index(x_col)
                coef_value = coefficients[coef_index]
                
                # Criar range de valores para a variável
                x_range = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)
                
                mean_x = valid_data[x_col].mean()
                std_x = valid_data[x_col].std()
                x_range_scaled = (x_range - mean_x) / std_x

                z = intercept + coef_value * x_range_scaled
                
                # Aplicar função sigmoide: p = 1 / (1 + e^(-z))
                probabilities = 1 / (1 + np.exp(-z))
                
                # Plotar curva sigmoide
                sigmoid_curve = go.Scatter(
                    x=x_range,
                    y=probabilities,
                    mode='lines',
                    name='Curva Sigmoide',
                    line=dict(color='blue', width=3),
                    hovertemplate='<b>%{x:.1f} ' + x_col + '</b><br>Probabilidade: %{y:.3f}<extra></extra>'
                )
                fig.add_trace(sigmoid_curve)
                
            else:
                # Fallback: calcular regressão logística simples com statsmodels
                import statsmodels.api as sm
                X_simple = valid_data[[x_col]]
                X_simple = sm.add_constant(X_simple)
                y_simple = valid_data[y_col]
                
                logit_model = sm.Logit(y_simple, X_simple)
                result = logit_model.fit(disp=False)
                
                # Calcular curva sigmoide
                x_range = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)
                X_pred = sm.add_constant(x_range)
                probabilities = result.predict(X_pred)
                
                sigmoid_curve = go.Scatter(
                    x=x_range,
                    y=probabilities,
                    mode='lines',
                    name='Curva Sigmoide (Simples)',
                    line=dict(color='blue', width=3),
                    hovertemplate='<b>%{x:.1f} ' + x_col + '</b><br>Probabilidade: %{y:.3f}<extra></extra>'
                )
                fig.add_trace(sigmoid_curve)
                
        except Exception as e:
            print(f"Erro ao calcular curva sigmoide: {e}")
            # Adicionar mensagem no gráfico
            fig.add_annotation(
                text=f"Erro na curva: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                bgcolor="white"
            )
    
    # Linha de threshold
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                 annotation_text="Threshold 0.5")
    
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text='Probabilidade de Vitória', range=[-0.1, 1.1])
    
    fig.update_layout(
        height=500,
        title_text=f"Curva Sigmoide: Probabilidade de Vitória vs {x_col}",
        template="plotly_white",
        showlegend=True
    )
    
    return fig

def plot_multiple_logistic_curves(df: pd.DataFrame, X_cols: list, y_col: str, model=None):
    """
    Gera múltiplos gráficos de curva sigmoide para diferentes variáveis.
    """
    if df.empty or not X_cols:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")

    # Selecionar até 4 variáveis para não sobrecarregar
    display_cols = X_cols[:4] if len(X_cols) > 4 else X_cols
    rows = (len(display_cols) + 1) // 2
    cols = 2 if len(display_cols) > 1 else 1
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f"Probabilidade vs {x_col}" for x_col in display_cols]
    )
    
    for i, x_col in enumerate(display_cols):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Filtrar dados válidos
        valid_data = df[[x_col, y_col]].dropna()
        
        if len(valid_data) > 1:
            # Scatter plot dos dados reais (com jitter)
            np.random.seed(42)
            jitter_y = valid_data[y_col] + np.random.normal(0, 0.02, len(valid_data))
            jitter_x = valid_data[x_col] + np.random.normal(0, 0.01, len(valid_data))
            
            scatter = go.Scatter(
                x=jitter_x,
                y=jitter_y,
                mode='markers',
                name=f'Dados {x_col}',
                marker=dict(
                    size=6,
                    opacity=0.6,
                    color=valid_data[y_col].map({0: 'red', 1: 'green'}),
                    line=dict(width=0.5, color='black')
                ),
                showlegend=False
            )
            fig.add_trace(scatter, row=row, col=col)
            
            # TENTAR ADICIONAR CURVA SIGMOIDE PARA CADA VARIÁVEL
            if model is not None and hasattr(model, 'model'):
                try:
                    # Obter coeficientes
                    intercept = model.model.intercept_[0]
                    coefficients = model.model.coef_[0]
                    
                    # Encontrar índice da variável
                    feature_names = list(model.model.feature_names_in_) if hasattr(model.model, 'feature_names_in_') else []
                    if x_col in feature_names:
                        coef_index = feature_names.index(x_col)
                        coef_value = coefficients[coef_index]
                        
                        # Calcular curva sigmoide
                        x_range = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)

                        mean_x = valid_data[x_col].mean()
                        std_x = valid_data[x_col].std()
                        x_range_scaled = (x_range - mean_x) / std_x

                        z = intercept + coef_value * x_range ** x_range_scaled
                        probabilities = 1 / (1 + np.exp(-z))
                        
                        # Adicionar curva ao subplot
                        curve = go.Scatter(
                            x=x_range,
                            y=probabilities,
                            mode='lines',
                            name=f'Curva {x_col}',
                            line=dict(color='blue', width=2),
                            showlegend=False
                        )
                        fig.add_trace(curve, row=row, col=col)
                except Exception as e:
                    print(f"Erro na curva para {x_col}: {e}")
            
            # Linha de threshold
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         row=row, col=col)
            
            fig.update_xaxes(title_text=x_col, row=row, col=col)
            fig.update_yaxes(title_text='Prob. Vitória', 
                           row=row, col=col, range=[-0.1, 1.1])
    
    fig.update_layout(
        height=400 * rows,
        title_text="Diagrama de Dispersão com Curvas Sigmoides - Regressão Logística",
        template="plotly_white",
        showlegend=False
    )
    
    return fig

# -------------------------------
# 1. Previsão vs Realidade (MLP)
# -------------------------------
def plot_mlp_prediction_vs_reality(y_true: pd.Series, y_pred_proba: np.ndarray, y_pred_class: np.ndarray):
    df = pd.DataFrame({
        'Índice': range(len(y_true)),
        'Probabilidade Prevista': y_pred_proba,
        'Classe Real': y_true.astype(str),
        'Classe Predita': y_pred_class
    })

    fig = go.Figure()

    # Pontos reais (vitória/derrota)
    fig.add_trace(go.Scatter(
        x=df['Índice'],
        y=df['Probabilidade Prevista'],
        mode='markers',
        marker=dict(
            color=df['Classe Real'].map({'0': 'red', '1': 'green'}),
            size=10,
            line=dict(width=1, color='black')
        ),
        name='Classe Real',
        text=[f"Real: {r}<br>Pred: {p:.3f}" for r, p in zip(y_true, y_pred_proba)],
        hovertemplate='<b>Jogo %{x}</b><br>%{text}<extra></extra>'
    ))

    # Linha de threshold
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Threshold 0.5")

    fig.update_layout(
        title="Previsão vs Realidade (MLP) - Probabilidade por Jogo",
        xaxis_title="Sequência de Jogos",
        yaxis_title="Probabilidade de Vitória",
        yaxis=dict(range=[-0.05, 1.05]),
        template="plotly_white",
        height=500,
        legend_title="Classe Real"
    )
    return fig


# -------------------------------
# 2. Evolução do Erro com Smoothing e Destaque de Overfitting
# -------------------------------
def plot_training_history_smoothed(history_df: pd.DataFrame):
    if history_df.empty:
        return go.Figure().add_annotation(text="Histórico de treino não disponível.")

    # Aplicar suavização exponencial (EWMA)
    loss_smoothed = history_df['loss'].ewm(alpha=0.2).mean()
    val_loss_smoothed = history_df['val_loss'].ewm(alpha=0.2).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=loss_smoothed, name='Loss (treino) - suavizado', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=val_loss_smoothed, name='Val Loss - suavizado', line=dict(color='red')))

    # Destacar overfitting
    overfitting_epochs = history_df[history_df['val_loss'] > history_df['loss']].index
    if len(overfitting_epochs) > 0:
        fig.add_vrect(
            x0=overfitting_epochs[0], x1=history_df.index[-1],
            fillcolor="red", opacity=0.1,
            annotation_text="Overfitting detectado", annotation_position="top left"
        )

    fig.update_layout(
        title="Evolução do Erro Durante o Treinamento (MLP) - Suavizado com EWMA",
        xaxis_title="Época",
        yaxis_title="Erro (Binary Crossentropy)",
        template="plotly_white",
        height=500
    )
    return fig


# -------------------------------
# 3. Três Gráficos da Matriz de Erros
# -------------------------------
def plot_probability_histogram(y_true: pd.Series, y_pred_proba: np.ndarray):
    df = pd.DataFrame({'Probabilidade': y_pred_proba, 'Resultado': y_true.map({0: 'Derrota', 1: 'Vitória'})})

    fig = px.histogram(
        df, x='Probabilidade', color='Resultado',
        nbins=20, marginal="box", opacity=0.7,
        color_discrete_map={'Derrota': 'red', 'Vitória': 'green'},
        title="Distribuição das Probabilidades Previstas (MLP)"
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="black")
    fig.update_layout(template="plotly_white", height=500)
    return fig


def plot_predicted_vs_actual_scatter(y_true: pd.Series, y_pred_class: np.ndarray):
    df = pd.DataFrame({'Real': y_true, 'Predito': y_pred_class})

    fig = go.Figure()
    correct = df['Real'] == df['Predito']
    fig.add_trace(go.Scatter(
        x=df[correct]['Real'] + np.random.normal(0, 0.05, sum(correct)),
        y=df[correct]['Predito'] + np.random.normal(0, 0.05, sum(correct)),
        mode='markers', name='Correto', marker=dict(color='green', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=df[~correct]['Real'] + np.random.normal(0, 0.05, sum(~correct)),
        y=df[~correct]['Predito'] + np.random.normal(0, 0.05, sum(~correct)),
        mode='markers', name='Errado', marker=dict(color='red', size=10, symbol='x')
    ))

    fig.update_xaxes(title="Classe Real", tickvals=[0, 1], ticktext=['Derrota', 'Vitória'])
    fig.update_yaxes(title="Classe Predita", tickvals=[0, 1], ticktext=['Derrota', 'Vitória'])
    fig.update_layout(title="Classe Predita × Classe Real (MLP)", template="plotly_white", height=500)
    return fig


# -------------------------------
# 7. Intervalo de Confiança por Bootstrap
# -------------------------------
def plot_bootstrap_confidence(y_pred_proba: np.ndarray, n_bootstraps: int = 1000, ci: float = 95):
    bootstraps = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(y_pred_proba, size=len(y_pred_proba), replace=True)
        bootstraps.append(np.mean(sample))

    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
    mean_pred = np.mean(y_pred_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[mean_pred, mean_pred], mode='lines', name=f'Média = {mean_pred:.3f}', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[lower, lower], mode='lines', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[upper, upper], mode='lines', line=dict(dash='dash', color='red')))
    fig.add_shape(type="rect", x0=0, x1=1, y0=lower, y1=upper, fillcolor="red", opacity=0.2, line_width=0)

    fig.update_layout(
        title=f"Intervalo de Confiança 95% das Previsões (Bootstrap, n={n_bootstraps})",
        xaxis=dict(showticklabels=False),
        yaxis=dict(range=[0, 1], title="Probabilidade Média Prevista"),
        template="plotly_white",
        height=400
    )
    fig.add_annotation(x=0.5, y=upper + 0.05, text=f"IC {ci}%: [{lower:.3f}, {upper:.3f}]", showarrow=False)
    return fig


# -------------------------------
# 8. Comparação Final Entre Modelos (Linha Temporal)
# -------------------------------
def plot_model_comparison_timeline(df_test: pd.DataFrame, y_test: pd.Series,
                                   pred_linear=None, pred_logistic=None, pred_mlp=None):
    df_plot = df_test.copy()
    df_plot['Real'] = y_test.values
    df_plot = df_plot.reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=df_plot['Real'], mode='lines+markers', name='Real (Vitória=1)', line=dict(color='black', width=3)))

    if pred_linear is not None:
        fig.add_trace(go.Scatter(y=(pred_linear > np.percentile(pred_linear, 50)).astype(int),
                                 mode='lines', name='Regressão Linear (binarizada)', line=dict(color='blue', dash='dot')))

    if pred_logistic is not None:
        fig.add_trace(go.Scatter(y=pred_logistic, mode='lines', name='Regressão Logística', line=dict(color='green')))

    if pred_mlp is not None:
        fig.add_trace(go.Scatter(y=pred_mlp, mode='lines', name='MLP (Rede Neural)', line=dict(color='red', width=2)))

    fig.update_layout(
        title="Comparação Temporal: Resultado Real vs Previsões dos Modelos",
        xaxis_title="Jogo (cronológico)",
        yaxis=dict(tickvals=[0, 1], ticktext=['Derrota', 'Vitória'], range=[-0.1, 1.1]),
        template="plotly_white",
        height=550,
        legend=dict(y=1.15, orientation='h')
    )
    return fig

# -------------------------------
# Player Ranking (Top N) - usa df_agg retornado por load_player_game_log
# -------------------------------
def plot_player_ranking(df_players_agg: pd.DataFrame, metric: str = 'PTS_mean', top_n: int = 10):
    """
    Recebe df_agg (cada linha = jogador, colunas agregadas como PTS_mean, AST_mean, etc).
    metric: nome da coluna agregada a ser ordenada (ex: 'PTS_mean', 'AST_mean').
    """
    if df_players_agg is None or df_players_agg.empty:
        return go.Figure().add_annotation(text="Dados de jogadores indisponíveis.")
    if metric not in df_players_agg.columns:
        # tentar variações comuns
        if metric.replace('_mean','') + '_mean' in df_players_agg.columns:
            metric = metric.replace('_mean','') + '_mean'
        else:
            return go.Figure().add_annotation(text=f"Métrica '{metric}' não encontrada nos dados de jogadores.")
    df_plot = df_players_agg[['PLAYER_NAME', metric, 'GAMES_PLAYED']].dropna().sort_values(by=metric, ascending=False).head(top_n)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot[metric][::-1],
        y=df_plot['PLAYER_NAME'][::-1],
        orientation='h',
        text=df_plot['GAMES_PLAYED'][::-1].astype(str),
        hovertemplate='<b>%{y}</b><br>' + metric + ': %{x:.2f}<br>Jogos: %{text}<extra></extra>'
    ))
    fig.update_layout(
        title=f"Top {top_n} Jogadores por {metric}",
        xaxis_title=metric,
        yaxis_title='Jogador',
        template='plotly_white',
        height=50*top_n + 150
    )
    return fig

# -------------------------------
# Comparação Time x Adversário por jogo (médias e scatter)
# -------------------------------
def plot_team_vs_opponent(df_games: pd.DataFrame, stats: list = ['PTS','REB','AST']):
    """
    df_games: DataFrame retornado por load_team_game_log que contém MATCHUP com string tipo 'BOS @ LAL' ou 'BOS vs LAL'.
    stats: lista de estatísticas a comparar (ex: ['PTS','REB']).
    Retorna subplots com barras de média Time vs Adversário e scatter PTS_time vs PTS_opp.
    """
    if df_games is None or df_games.empty:
        return go.Figure().add_annotation(text="Dados de jogos do time indisponíveis.")
    
    df = df_games.copy()
    
    # Verificar se temos a coluna MATCHUP
    if 'MATCHUP' not in df.columns:
        return go.Figure().add_annotation(text="Coluna MATCHUP ausente nos dados de jogos.")
    
    # Extrair adversário do MATCHUP
    def parse_matchup(m):
        try:
            parts = m.split()
            # exemplo 'BOS @ LAL' (fora) ou 'BOS vs. LAL' (casa)
            if '@' in parts:
                opp = parts[-1]  # Último elemento é o adversário quando joga fora
            else:
                opp = parts[2]   # Terceiro elemento é o adversário quando joga em casa (ex: "BOS vs. LAL")
            return opp
        except Exception:
            return "Unknown"
    
    df['OPPONENT'] = df['MATCHUP'].apply(parse_matchup)
    
    # Calcular estatísticas do adversário usando PLUS_MINUS
    if 'PLUS_MINUS' in df.columns and 'PTS' in df.columns:
        df['PTS_OPP'] = df['PTS'] - df['PLUS_MINUS']
    else:
        # Se não temos PLUS_MINUS, não podemos calcular estatísticas do adversário
        st.warning("Não é possível calcular estatísticas do adversário sem PLUS_MINUS")
        df['PTS_OPP'] = np.nan
    
    # Preparar figura combinada
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(
            "Médias: Time vs Adversário", 
            "Relação: PTS Time x PTS Adversário"
        ),
        column_widths=[0.5, 0.5]
    )
    
    # --- PRIMEIRO SUBPLOT: BARRAS DE MÉDIA ---
    team_means = []
    opp_means = []
    stat_names = []
    
    for stat in stats:
        if stat not in df.columns:
            continue
            
        # Média do time
        mean_team = df[stat].mean()
        team_means.append(mean_team)
        stat_names.append(stat)
        
        # Média do adversário (apenas para PTS, já que é a única que podemos calcular)
        if stat == 'PTS' and 'PTS_OPP' in df.columns:
            mean_opp = df['PTS_OPP'].mean()
            opp_means.append(mean_opp)
        else:
            # Para outras estatísticas, não temos dados do adversário
            opp_means.append(np.nan)
    
    # Adicionar barras para o Time
    fig.add_trace(
        go.Bar(
            name='Time',
            x=stat_names,
            y=team_means,
            marker_color='blue',
            text=[f'{x:.1f}' for x in team_means],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # Adicionar barras para o Adversário (apenas onde temos dados)
    valid_opp_stats = [(name, mean) for name, mean in zip(stat_names, opp_means) if not np.isnan(mean)]
    if valid_opp_stats:
        opp_stat_names, opp_stat_means = zip(*valid_opp_stats)
        fig.add_trace(
            go.Bar(
                name='Adversário',
                x=opp_stat_names,
                y=opp_stat_means,
                marker_color='red',
                text=[f'{x:.1f}' for x in opp_stat_means],
                textposition='auto',
            ),
            row=1, col=1
        )
    
    # --- SEGUNDO SUBPLOT: SCATTER PTS ---
    if 'PTS' in df.columns and 'PTS_OPP' in df.columns:
        valid_pts_data = df[['PTS', 'PTS_OPP']].dropna()
        if len(valid_pts_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_pts_data['PTS'],
                    y=valid_pts_data['PTS_OPP'],
                    mode='markers',
                    name='Jogos',
                    marker=dict(
                        size=8,
                        color='green',
                        opacity=0.6
                    ),
                    hovertemplate=(
                        '<b>Jogo</b><br>' +
                        'PTS Time: %{x}<br>' +
                        'PTS Adv: %{y}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=1, col=2
            )
            
            # Adicionar linha de referência y=x (onde seria empate)
            max_pts = max(valid_pts_data['PTS'].max(), valid_pts_data['PTS_OPP'].max())
            min_pts = min(valid_pts_data['PTS'].min(), valid_pts_data['PTS_OPP'].min())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_pts, max_pts],
                    y=[min_pts, max_pts],
                    mode='lines',
                    name='Linha de Empate',
                    line=dict(dash='dash', color='gray'),
                    showlegend=True
                ),
                row=1, col=2
            )
    
    # --- ATUALIZAR LAYOUT ---
    fig.update_xaxes(
        title_text="Estatística",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Valor Médio",
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="PTS do Time",
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="PTS do Adversário", 
        row=1, col=2
    )
    
    fig.update_layout(
        title="Comparação Time vs Adversário",
        template='plotly_white',
        height=500,
        showlegend=True,
        barmode='group'  # Para agrupar as barras
    )
    
    return fig

# -------------------------------
# Rolling Performance: Rolling Accuracy e Rolling MAE
# -------------------------------
def plot_rolling_performance(df_test_with_date: pd.DataFrame, y_true_col: str = 'Real', y_pred_col: str = 'Predito', window: int = 5):
    """
    df_test_with_date: DataFrame com coluna de data ('GAME_DATE') e colunas y_true_col (0/1) e y_pred_col (0/1 ou probabilidades).
    Gera duas linhas: rolling accuracy e rolling MAE (da probabilidade).
    """
    if df_test_with_date is None or df_test_with_date.empty:
        return go.Figure().add_annotation(text="Dados de teste para rolling performance indisponíveis.")
    
    df = df_test_with_date.copy()
    
    # Garantir que as colunas tenham o mesmo comprimento
    min_length = min(len(df), len(df[y_true_col]), len(df[y_pred_col]))
    df = df.head(min_length)
    
    # Garantir Series 1D e comprimento consistente
    y_true_values = np.array(df[y_true_col]).reshape(-1)[:min_length]
    y_pred_values = np.array(df[y_pred_col]).reshape(-1)[:min_length]
    
    # Atualizar o DataFrame com os valores corrigidos
    df = df.reset_index(drop=True).head(min_length)
    df[y_true_col] = y_true_values
    df[y_pred_col] = y_pred_values
    
    # Converter para numérico
    df[y_true_col] = pd.to_numeric(df[y_true_col], errors='coerce')
    df[y_pred_col] = pd.to_numeric(df[y_pred_col], errors='coerce')
    
    if 'GAME_DATE' not in df.columns:
        # tentar se índice é datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index':'GAME_DATE'})
        else:
            return go.Figure().add_annotation(text="GAME_DATE ausente para série temporal.")
    
    # Garantir que temos dados suficientes
    df = df.dropna(subset=[y_true_col, y_pred_col, 'GAME_DATE'])
    
    if len(df) == 0:
        return go.Figure().add_annotation(text="Dados insuficientes após limpeza.")
    
    # Se y_pred_col for probabilidade, criar pred_class a partir de 0.5
    if df[y_pred_col].max() <= 1.0 and df[y_pred_col].min() >= 0.0:
        df['Pred_Class'] = (df[y_pred_col] >= 0.5).astype(int)
    else:
        df['Pred_Class'] = df[y_pred_col].astype(int)
    
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    
    # Calcular métricas rolling
    df['Rolling_Accuracy'] = (df['Pred_Class'] == df[y_true_col]).rolling(window=window, min_periods=1).mean()
    
    # MAE: se y_pred_col é probabilidade, comparar com y_true
    df['Abs_Error'] = (df[y_pred_col] - df[y_true_col]).abs()
    df['Rolling_MAE'] = df['Abs_Error'].rolling(window=window, min_periods=1).mean()
    
    # Criar figura
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['GAME_DATE'], 
        y=df['Rolling_Accuracy'], 
        mode='lines+markers', 
        name=f'Rolling Accuracy (w={window})', 
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['GAME_DATE'], 
        y=df['Rolling_MAE'], 
        mode='lines+markers', 
        name=f'Rolling MAE (w={window})', 
        line=dict(color='red'), 
        yaxis='y2'
    ))
    
    # Configurar segundo eixo y
    fig.update_layout(
        title="Evolução Temporal da Performance (Rolling Accuracy & Rolling MAE)",
        xaxis_title="Data do Jogo",
        yaxis=dict(title='Rolling Accuracy', range=[0, 1]),
        yaxis2=dict(title='Rolling MAE', overlaying='y', side='right', range=[0, 1]),
        template='plotly_white',
        height=520
    )
    
    return fig