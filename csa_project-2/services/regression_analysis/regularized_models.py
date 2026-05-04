"""
Построение регуляризованных регрессионных моделей
"""

from sklearn.linear_model import Ridge, Lasso


class RegularizedModelBuilder:
    """Класс построения ridge/lasso регрессии"""

    def build(self, X_train, y_train, feature_cols, method: str = "ridge", alpha: float = 1.0):
        if method == "ridge":
            model = Ridge(alpha=alpha)
        elif method == "lasso":
            model = Lasso(alpha=alpha)
        else:
            raise ValueError(f"Unknown regularized regression method: {method}")
        model.fit(X_train, y_train)
        model_info = {
            "model_type": method,
            "alpha": alpha,
            "features": feature_cols,
        }
        return model, model_info
