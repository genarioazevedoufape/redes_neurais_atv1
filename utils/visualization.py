import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        marker=dict(size=6, opacity=0.6)
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
    
    fig.update_layout(
        title=f"Tendência de {y_col} ao longo do tempo com Intervalo de Confiança",
        xaxis_title='Data do Jogo',
        yaxis_title=y_col,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig
    