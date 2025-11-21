import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error

class MLPModel:
    """
    Implementação de um Multi-Layer Perceptron (MLP) para classificação
    para a Atividade 2 de Redes Neurais.
    """
    def __init__(self, input_dim):
        self.model = None
        self.history = None
        self.input_dim = input_dim # Número de features de entrada
        self.metrics_to_calculate = ['accuracy'] # Métricas do Keras
        
    def build_model(self, optimizer_name='Adam', activation='relu', hidden_layers=1, neurons=8):
        """
        Constrói a arquitetura do modelo.
        """
        self.model = Sequential()
        
        # Camada de Entrada
        # Define a forma da entrada (número de features)
        self.model.add(InputLayer(input_shape=(self.input_dim,)))
        
        # Camadas Ocultas 
        current_neurons = neurons
        for i in range(hidden_layers):
            # Adiciona a camada Densa (totalmente conectada)
            self.model.add(Dense(current_neurons, activation=activation))
            
            # Adiciona BatchNormalization 
            # Ajuda a acelerar o treino e estabilizar a rede
            self.model.add(BatchNormalization()) 
            
            # Adiciona Dropout 
            # Técnica para evitar overfitting, "desligando" neurônios no treino
            self.model.add(Dropout(0.5)) # 30% de dropout
            
            if current_neurons > 16:
                current_neurons = current_neurons // 2 


        # Camada de Saída 
        # 1 neurônio com ativação 'sigmoid' para classificação binária (0 ou 1)
        self.model.add(Dense(1, activation='sigmoid'))

        # Selecionar o otimizador 
        if optimizer_name.lower() == 'adam':
            optimizer = Adam()
        elif optimizer_name.lower() == 'sgd':
            optimizer = SGD()
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = RMSprop()
        else:
            optimizer = Adam() # Padrão

        # Usamos 'binary_crossentropy' como 'loss' para classificação binária
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=self.metrics_to_calculate
        )
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, epochs=100, batch_size=2, validation_split=0.2):
        """
        Treina o modelo de MLP.
        """
        if self.model is None:
            self.build_model() # Constrói o modelo padrão se não foi construído
            
        # Parada Antecipada (Early Stopping) para evitar overfitting
        # Monitora a perda na validação ('val_loss')
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,          # Nº de épocas sem melhora antes de parar
            restore_best_weights=True # Restaura os melhores pesos do modelo
        )
        
        # Treinamento
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split, # Usa parte dos dados de treino para validação
            callbacks=[early_stopping_callback],
            verbose=0 # 0 = silencioso, 1 = barra de progresso
        )

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Faz previsões de probabilidade (0 a 1).
        """
        return self.model.predict(X_test).flatten()

    def predict_class(self, X_test: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Faz previsões de classe (0 ou 1) usando um threshold.
        """
        probabilities = self.predict_proba(X_test)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, y_true: pd.Series, y_pred_class: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """
        Avalia o modelo e retorna métricas (incluindo as pedidas na Atividade 2).
        """
        # Calcula o MSE separadamente para usar no cálculo do RMSE
        mse = mean_squared_error(y_true, y_pred_proba) 

        metrics = {
            'Acurácia': accuracy_score(y_true, y_pred_class),
            'Precisão': precision_score(y_true, y_pred_class, zero_division=0),
            'Recall': recall_score(y_true, y_pred_class, zero_division=0),
            'F1-score': f1_score(y_true, y_pred_class, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            
            # Métricas pedidas
            'MAE': mean_absolute_error(y_true, y_pred_proba),  # Calcula MAE corretamente
            'RMSE': np.sqrt(mse),                            # Calcula RMSE robustamente
            'R² (R-squared)': r2_score(y_true, y_pred_proba)
        }
        return metrics

    def get_summary(self):
        """ Retorna o resumo da arquitetura do modelo (para o relatório) """
        stringlist = []
        if self.model:
            self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)

    def get_history_df(self):
        """ 
        Retorna o histórico de treino como DataFrame.
        Usado para gerar o gráfico "Evolução do Erro"
        """
        if self.history:
            return pd.DataFrame(self.history.history)
        return pd.DataFrame()


    