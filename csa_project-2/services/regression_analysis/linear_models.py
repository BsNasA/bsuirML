"""
Построение линейных регрессионных моделей
"""

from sklearn.linear_model import LinearRegression


class LinearModelBuilder:
    """Класс построения линейной регрессии"""

    def build(self, X_train, y_train, feature_cols):
        model = LinearRegression()
        model.fit(X_train, y_train)
        model_info = {
            "model_type": "linear",
            "features": feature_cols,
        }
        return model, model_info
