import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class NeuralNetworkService:
    def __init__(self, hidden_layer_sizes: tuple = (100, 50)):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Обучает нейросеть и возвращает метрики качества.
        """
        # Разделение выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Масштабирование (важно для нейросетей)
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        X_test_scaled = self.scaler_x.transform(X_test)

        # Обучение
        self.model.fit(X_train_scaled, y_train)

        # Предсказание
        y_pred = self.model.predict(X_test_scaled)

        # Метрики
        metrics = {
            "R2_score": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
        }

        return metrics
