import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_regression_line(df: pd.DataFrame, X_col: str, y_col: str):
    """
    Gera um Diagrama de Dispersão com a Linha de Regressão (Plotly).
    Usa apenas a primeira variável independente para o gráfico 2D.
    """
    if df.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")

    fig = px.scatter(
        df,
        x=X_col,
        y=y_col,
        trendline="ols", # ols = Ordinary Least Squares (Regressão Linear)
        title=f"Dispersão e Regressão: {y_col} vs {X_col}",
        labels={X_col: X_col, y_col: y_col},
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
        # Vamos plotar a probabilidade de 1 (sucesso) contra o índice
        df['Índice'] = df.index
        fig = px.scatter(
            df,
            x=df.index,
            y='Previsão',
            color=df['Realidade'].astype(str), # Converter para string para cores discretas
            color_discrete_map={'0': 'red', '1': 'green'},
            title='Probabilidade de Sucesso (Regressão Logística)',
            labels={'x': 'Índice do Jogo', 'Previsão': 'Probabilidade de Sucesso (Y=1)', 'color': 'Realidade'},
            template="plotly_white"
        )
        # Adicionar linha de corte (threshold) em 0.5
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="Threshold 0.5", annotation_position="top right")
        
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

def plot_trend_with_confidence(df: pd.DataFrame, date_col: str, y_col: str):
    """
    Gera um Gráfico de Tendência com Média Móvel (Plotly).
    Assume que date_col é a data.
    """
    if df.empty:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")

    # Calcular a média móvel de 5 jogos (ajustável)
    df_plot = df.sort_values(by=date_col).copy()
    df_plot['Média Móvel'] = df_plot[y_col].rolling(window=5, min_periods=1).mean()
    
    fig = px.line(
        df_plot,
        x=date_col,
        y=y_col,
        title=f"Tendência de {y_col} ao longo do tempo (Média Móvel)",
        labels={date_col: 'Data do Jogo', y_col: y_col},
        template="plotly_white"
    )
    
    # Adicionar a linha de média móvel
    fig.add_trace(go.Scatter(
        x=df_plot[date_col],
        y=df_plot['Média Móvel'],
        mode='lines',
        name='Média Móvel (5 jogos)',
        line=dict(color='red', width=2)
    ))
    
    return fig

