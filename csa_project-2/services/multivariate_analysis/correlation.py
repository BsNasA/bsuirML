import pandas as pd
import numpy as np


class CorrelationService:
    @staticmethod
    def calculate_correlation(
        df: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Вычисляет матрицу корреляций.
        Методы: 'pearson' (линейная), 'spearman' (ранговая, нелинейная).
        """
        # Оставляем только числовые колонки
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr(method=method)
        return corr_matrix

    @staticmethod
    def get_highly_correlated_pairs(
        corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> list:
        """
        Находит пары признаков с высокой корреляцией (выше порога threshold).
        Полезно для выявления мультиколлинеарности.
        """
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    pairs.append(
                        {
                            "feature_1": corr_matrix.columns[i],
                            "feature_2": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )
        return pairs
