import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


class FactorAnalysisService:
    def __init__(self, n_components: int = 3, rotation: str = "varimax"):
        self.n_components = n_components
        self.model = FactorAnalysis(n_components=self.n_components, rotation=rotation)
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> dict:
        """
        Проводит факторный анализ и возвращает факторные нагрузки.
        """
        # Стандартизация данных перед анализом обязательна
        scaled_data = self.scaler.fit_transform(df)

        self.model.fit(scaled_data)

        # Получаем факторные нагрузки (вклад каждой переменной в фактор)
        loadings = pd.DataFrame(
            self.model.components_.T,
            columns=[f"Factor_{i+1}" for i in range(self.n_components)],
            index=df.columns,
        )

        return {"loadings": loadings, "noise_variance": self.model.noise_variance_}
