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
    # 1. Verificar se há colunas suficientes
    if not feature_cols:
        raise ValueError("Nenhuma variável independente selecionada.")
    
    # 2. Remover linhas com valores NaN
    data = df.dropna(subset=[target_col] + feature_cols)
    
    if data.empty:
        raise ValueError("Não há dados válidos após remover valores faltantes.")
    
    # 3. Verificar variância das features
    X = data[feature_cols]
    y = data[target_col]
    
    # Remover colunas com variância zero
    zero_variance_cols = X.columns[X.var() == 0]
    if len(zero_variance_cols) > 0:
        print(f"Removendo colunas com variância zero: {list(zero_variance_cols)}")
        X = X.drop(columns=zero_variance_cols)
        feature_cols = [col for col in feature_cols if col not in zero_variance_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("Todas as variáveis independentes foram removidas devido à variância zero.")
    
    # 4. Divisão em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) == 2 else None
    )
    
    # 5. Escalonamento dos Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Retornar como DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler

