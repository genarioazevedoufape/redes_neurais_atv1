## `utils/preprocessing.py`

"""Funções para limpeza, transformação e preparação das features.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


def prepare_dataset(df: pd.DataFrame, y_col: str, x_cols: List[str], test_size: float = 0.2, scale: bool = True, random_state: int = 42) -> Tuple:
    """Seleciona X e y, trata NaNs, aplica split e escalonamento opcional.
    Retorna: X_train, X_test, y_train, y_test, scaler (ou None)
    """
    df_local = df.copy()
    # Seleciona colunas
    X = df_local[x_cols].copy()
    y = df_local[y_col].copy()

    # Tratamento simples de NaNs: remover linhas com NaN nas colunas selecionadas
    valid_mask = pd.concat([X, y], axis=1).dropna().index
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=x_cols, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=x_cols, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler


def binarize_target(y: pd.Series, threshold: float) -> pd.Series:
    """Converte um target numérico para binário usando threshold (>= True/1).
    útil para regressão logística.
    """
    return (y >= threshold).astype(int)