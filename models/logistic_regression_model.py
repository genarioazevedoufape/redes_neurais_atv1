import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import statsmodels.api as sm
from numpy.linalg import LinAlgError

class LogisticRegressionModel:
    """
    Implementação da Regressão Logística.
    """
    def __init__(self):
        # Aumentar max_iter para evitar warnings de convergência
        self.model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        self.statsmodels_results = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Treina o modelo de Regressão Logística com tratamento de erros.
        """
        try:
            # Treinamento com scikit-learn
            self.model.fit(X_train, y_train)
            
            # Treinamento com statsmodels (com tratamento robusto de erros)
            if len(X_train) > len(X_train.columns) + 1:  # Garantir dados suficientes
                X_train_sm = sm.add_constant(X_train)
                try:
                    sm_model = sm.Logit(y_train, X_train_sm)
                    self.statsmodels_results = sm_model.fit(disp=False, maxiter=1000, method='bfgs')
                except (LinAlgError, ValueError) as e:
                    print(f"Statsmodels falhou devido a: {e}. Continuando com scikit-learn.")
                    self.statsmodels_results = None
            else:
                self.statsmodels_results = None
                print("Dados insuficientes para statsmodels.")
            
        except Exception as e:
            print(f"Erro no treinamento: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Faz previsões de probabilidade (0 a 1).
        """
        # Retorna a probabilidade da classe positiva (1)
        return self.model.predict_proba(X_test)[:, 1]

    def predict_class(self, X_test: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Faz previsões de classe (0 ou 1) usando um threshold.
        """
        probabilities = self.predict_proba(X_test)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, y_true: pd.Series, y_pred_class: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """
        Avalia o modelo e retorna métricas.
        """
        metrics = {
            'Acurácia': accuracy_score(y_true, y_pred_class),
            'Precisão': precision_score(y_true, y_pred_class, zero_division=0),
            'Recall': recall_score(y_true, y_pred_class, zero_division=0),
            'F1-score': f1_score(y_true, y_pred_class, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        }
        return metrics

    def get_coefficients(self, feature_cols: list) -> pd.DataFrame:
        """
        Retorna os coeficientes em um DataFrame.
        """
        coefficients = self.model.coef_[0]
        
        # Adicionar o intercepto (β0) ao início
        data = {
            'Variável': ['Intercepto (β0)'] + feature_cols,
            'Coeficiente (β)': [self.model.intercept_[0]] + coefficients.tolist()
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
    
    # Adicionar este método à classe LogisticRegressionModel

    def get_logistic_equation(self, feature_cols: list) -> str:
        """
        Retorna a equação da regressão logística no formato:
        p = 1 / [1 + e^(-(β0 + β1x1 + β2x2 + ...))]
        """
        if not hasattr(self.model, 'intercept_') or not hasattr(self.model, 'coef_'):
            return "Modelo não treinado ainda."
        
        intercept = self.model.intercept_[0]
        coefficients = self.model.coef_[0]
        
        # Construir a parte linear da equação
        linear_combination = f"{intercept:.4f}"
        
        for i, col in enumerate(feature_cols):
            coef = coefficients[i]
            if coef >= 0:
                linear_combination += f" + {coef:.4f}×{col}"
            else:
                linear_combination += f" - {abs(coef):.4f}×{col}"
        
        # Equação completa
        equation = f"p = 1 / [1 + e^(-({linear_combination}))]"
        
        return equation

    def get_odds_ratios(self, feature_cols: list) -> pd.DataFrame:
        """
        Retorna os odds ratios (e^β) para interpretação.
        """
        coefficients = self.model.coef_[0]
        odds_ratios = np.exp(coefficients)
        
        data = {
            'Variável': feature_cols,
            'Coeficiente (β)': coefficients,
            'Odds_Ratio (e^β)': odds_ratios,
            'Interpretação': odds_ratios.apply(
                lambda x: f"Aumenta {((x-1)*100):.1f}% a chance" if x > 1 
                else f"Reduz {((1-x)*100):.1f}% a chance" if x < 1 
                else "Sem efeito"
            )
        }
        
        return pd.DataFrame(data)
    
