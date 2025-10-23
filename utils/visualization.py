## `utils/visualization.py`

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def scatter_with_regression_line(df: pd.DataFrame, x: str, y: str):
    fig = px.scatter(df, x=x, y=y, trendline='ols', title=f'{y} vs {x} (scatter + regression)')
    return fig


def prediction_vs_actual(y_true, y_pred, title: str = 'Prediction vs Actual'):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).reset_index(drop=True)
    fig = px.scatter(df, x='y_true', y='y_pred', trendline='ols', title=title)
    fig.add_trace(go.Line(x=[df.y_true.min(), df.y_true.max()], y=[df.y_true.min(), df.y_true.max()], name='Ideal'))
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, title='Confusion Matrix')
    fig.update_xaxes(title='Predicted')
    fig.update_yaxes(title='Actual')
    return fig


def trend_with_ci(df: pd.DataFrame, x: str, y: str, ci: int = 95):
    # Usa seaborn para gerar linha com intervalo de confiança e salva em matplotlib; então converte para plotly
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x=x, y=y, ci=ci)
    plt.title(f'{y} trend with {ci}% CI')
    plt.tight_layout()
    fig = plt.gcf()
    return fig
