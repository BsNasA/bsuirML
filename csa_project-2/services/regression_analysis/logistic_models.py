"""
Построение логистических регрессионных моделей
"""

from sklearn.linear_model import LogisticRegression


class LogisticModelBuilder:
    """Класс построения логистической регрессии"""

    def build(self, X_train, y_train, feature_cols):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        model_info = {
            "model_type": "logistic",
            "features": feature_cols,
        }
        return model, model_info
