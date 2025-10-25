import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from numpy.linalg import LinAlgError

class LinearRegressionModel:
    """
    Implementação da Regressão Linear Múltipla.
    """
    def __init__(self):
        self.model = LinearRegression()
        self.statsmodels_results = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Treina o modelo de Regressão Linear.
        """
        # Treinamento com scikit-learn
        self.model.fit(X_train, y_train)
        
        # Treinamento com statsmodels para estatísticas detalhadas (com tratamento de erros)
        if len(X_train) > len(X_train.columns) + 1:  # Garantir dados suficientes
            X_train_sm = sm.add_constant(X_train)
            try:
                sm_model = sm.OLS(y_train, X_train_sm)
                self.statsmodels_results = sm_model.fit()
            except (LinAlgError, ValueError) as e:
                print(f"Statsmodels falhou devido a: {e}. Continuando com scikit-learn.")
                self.statsmodels_results = None
        else:
            self.statsmodels_results = None
            print("Dados insuficientes para statsmodels.")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Faz previsões usando o modelo treinado.
        """
        return self.model.predict(X_test)

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """
        Avalia o modelo e retorna métricas.
        """
        metrics = {
            'R2 Score': r2_score(y_true, y_pred),
            'Mean Squared Error (MSE)': mean_squared_error(y_true, y_pred),
            'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        return metrics

    def get_equation(self, feature_cols: list) -> str:
        """
        Retorna a equação da regressão no formato y = β0 + β1x1 + ...
        """
        intercept = self.model.intercept_
        coefficients = self.model.coef_
        
        equation = f"y = {intercept:.4f}"
        
        for i, col in enumerate(feature_cols):
            coef = coefficients[i]
            sign = " + " if coef >= 0 else " - "
            equation += f"{sign} {abs(coef):.4f}×{col}"
            
        equation += " + ε"
        return equation

    def get_coefficients(self, feature_cols: list) -> pd.DataFrame:
        """
        Retorna os coeficientes em um DataFrame.
        """
        coefficients = self.model.coef_
        
        # Adicionar o intercepto (β0) ao início
        data = {
            'Variável': ['Intercepto (β0)'] + feature_cols,
            'Coeficiente (β)': [self.model.intercept_] + coefficients.tolist()
        }
        
        df_coef = pd.DataFrame(data)
        
        return df_coef

    def get_statsmodels_summary(self) -> str:
        """
        Retorna o resumo do statsmodels para o modo avançado.
        """
        if self.statsmodels_results:
            return self.statsmodels_results.summary().as_html()
        return "<p>Resumo do Statsmodels não disponível (possível problema de singularidade ou dados insuficientes).</p>"