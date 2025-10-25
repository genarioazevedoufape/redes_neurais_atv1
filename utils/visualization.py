import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve

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

def plot_logistic_relationships(df: pd.DataFrame, X_cols: list, y_col: str):
    """
    Gráfico adequado para visualizar relações em regressão logística.
    Mostra a probabilidade estimada em função de cada variável independente.
    """
    if df.empty or not X_cols:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")

    display_cols = X_cols[:4] if len(X_cols) > 4 else X_cols
    rows = (len(display_cols) + 1) // 2
    cols = 2 if len(display_cols) > 1 else 1
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f"Probabilidade de Vitória vs {x_col}" for x_col in display_cols]
    )
    
    for i, x_col in enumerate(display_cols):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Filtrar dados válidos
        valid_data = df[[x_col, y_col]].dropna()
        
        if len(valid_data) > 1:
            # Usar suavização para mostrar a tendência logística
            try:
                # Agrupar por bins da variável X e calcular média de Y (probabilidade)
                valid_data['bin'] = pd.cut(valid_data[x_col], bins=8)
                bin_stats = valid_data.groupby('bin').agg({
                    x_col: 'mean', 
                    y_col: ['mean', 'count']
                }).reset_index()
                
                # Flatten column names
                bin_stats.columns = ['bin', f'{x_col}_mean', 'win_prob', 'count']
                bin_stats = bin_stats[bin_stats['count'] > 1]  # Remover bins com poucos dados
                
                if len(bin_stats) > 1:
                    # Linha de tendência logística (probabilidade média por bin)
                    line = go.Scatter(
                        x=bin_stats[f'{x_col}_mean'],
                        y=bin_stats['win_prob'],
                        mode='lines+markers',
                        name='Tendência',
                        line=dict(color='red', width=3),
                        marker=dict(size=8, color='red'),
                        showlegend=False
                    )
                    fig.add_trace(line, row=row, col=col)
                
                # Scatter plot dos dados originais (com jitter para evitar sobreposição)
                np.random.seed(42)  # Para reproducibilidade
                scatter = go.Scatter(
                    x=valid_data[x_col] + np.random.normal(0, 0.01, len(valid_data)),  # pequeno jitter
                    y=valid_data[y_col] + np.random.normal(0, 0.02, len(valid_data)),
                    mode='markers',
                    name='Dados',
                    showlegend=False,
                    marker=dict(
                        size=6, 
                        opacity=0.4, 
                        color=valid_data[y_col].map({0: 'red', 1: 'green'}),
                        line=dict(width=0.5, color='DarkSlateGrey')
                    )
                )
                fig.add_trace(scatter, row=row, col=col)
                
            except Exception as e:
                # Fallback: scatter plot simples
                scatter = go.Scatter(
                    x=valid_data[x_col],
                    y=valid_data[y_col],
                    mode='markers',
                    name=f'{x_col}',
                    showlegend=False,
                    marker=dict(
                        size=8, 
                        opacity=0.7,
                        color=valid_data[y_col].map({0: 'red', 1: 'green'})
                    )
                )
                fig.add_trace(scatter, row=row, col=col)
            
            fig.update_xaxes(title_text=x_col, row=row, col=col)
            fig.update_yaxes(
                title_text='Prob. Vitória', 
                row=row, col=col,
                tickvals=[0, 0.5, 1],
                range=[-0.1, 1.1]
            )
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
        title_text="Relações para Regressão Logística - Probabilidade de Vitória",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def plot_logistic_density(df: pd.DataFrame, X_cols: list, y_col: str):
    """
    Mostra a distribuição das variáveis contínuas separadas por resultado (vitória/derrota).
    """
    if df.empty or not X_cols:
        return go.Figure().add_annotation(text="Dados insuficientes para o gráfico.")
        
    display_cols = X_cols[:4] if len(X_cols) > 4 else X_cols
    
    fig = make_subplots(
        rows=len(display_cols), 
        cols=1,
        subplot_titles=[f"Distribuição de {x_col} por Resultado" for x_col in display_cols]
    )
    
    for i, x_col in enumerate(display_cols):
        row = i + 1
        
        # Dados para vitórias e derrotas
        wins = df[df[y_col] == 1][x_col].dropna()
        losses = df[df[y_col] == 0][x_col].dropna()
        
        if len(wins) > 0 and len(losses) > 0:
            # Box plot para comparar distribuições
            fig.add_trace(
                go.Box(
                    y=wins, 
                    name='Vitórias', 
                    marker_color='lightgreen',
                    boxpoints='outliers'
                ), 
                row=row, col=1
            )
            fig.add_trace(
                go.Box(
                    y=losses, 
                    name='Derrotas', 
                    marker_color='lightcoral',
                    boxpoints='outliers'
                ), 
                row=row, col=1
            )
    
    fig.update_layout(
        height=300 * len(display_cols),
        title_text="Distribuição das Variáveis por Resultado do Jogo",
        template="plotly_white",
        showlegend=False
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
            title='Probabilidade de Vitória (Regressão Logística)',
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
        title=f"Tendência de {y_col} ao longo do tempo com Intervalo de Confiança",
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

def plot_calibration_curve(y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10):
    """
    Gera gráfico de calibração para verificar se as probabilidades estão bem calibradas.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    fig = go.Figure()
    
    # Curva de calibração
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='lines+markers',
        name='Curva de Calibração',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Linha de referência (calibração perfeita)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Calibração Perfeita',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Curva de Calibração das Probabilidades',
        xaxis_title='Probabilidade Prevista Média',
        yaxis_title='Fração de Positivos',
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
        title=f'Importância das Variáveis (Regressão {model_type})',
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

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_type: str):
    """
    Gera gráfico de resíduos para análise de regressão linear.
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Resíduos',
        marker=dict(size=8, opacity=0.7, color='blue')
    ))
    
    # Linha de referência em y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f'Gráfico de Resíduos (Regressão {model_type})',
        xaxis_title='Valores Preditos',
        yaxis_title='Resíduos (Real - Previsto)',
        template="plotly_white"
    )
    
    return fig