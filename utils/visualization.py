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