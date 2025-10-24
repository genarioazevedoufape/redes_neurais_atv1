import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df: pd.DataFrame, target_col: str, feature_cols: list, test_size: float = 0.2, random_state: int = 42):
    """
    Prepara os dados para o treinamento do modelo.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada com os dados da NBA.
        target_col (str): Nome da coluna da variável dependente (Y).
        feature_cols (list): Lista de nomes das colunas das variáveis independentes (X).
        test_size (float): Proporção do conjunto de dados a ser usado para o teste.
        random_state (int): Semente para reprodutibilidade.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    
    # 1. Seleção de Variáveis
    # Remove linhas com valores NaN nas colunas relevantes
    data = df.dropna(subset=[target_col] + feature_cols)
    
    X = data[feature_cols]
    y = data[target_col]
    
    # 2. Divisão em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 3. Escalonamento dos Features (X)
    # É importante escalar apenas X, e ajustar o scaler apenas nos dados de treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Retornar como DataFrame para manter os nomes das colunas, o que é útil para statsmodels
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    # O Streamlit lida melhor com Series para y
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler

