# services/regression_analysis/service.py
"""
Сервис регрессионного анализа
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..config import config
from ..preprocessor.service import PreprocessorService
from ..exceptions import AnalysisError, InsufficientDataError
from ..logging_config import LoggerMixin

from .linear_models import LinearModelBuilder
from .logistic_models import LogisticModelBuilder
from .regularized_models import RegularizedModelBuilder
from .diagnostics import RegressionDiagnostics


class RegressionAnalysisService(LoggerMixin):
    """
    Сервис регрессионного анализа
    """

    def __init__(self):
        self.preprocessor = PreprocessorService()
        self.linear_builder = LinearModelBuilder()
        self.logistic_builder = LogisticModelBuilder()
        self.regularized_builder = RegularizedModelBuilder()
        self.diagnostics = RegressionDiagnostics()

        self.results_path = Path(config.RESULTS_STORAGE_PATH) / "regression"
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("RegressionAnalysisService initialized")

    def analyze(self, dataset_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение регрессионного анализа

        Args:
            dataset_id: идентификатор датасета
            params: параметры анализа
                - target_column: целевая переменная
                - feature_columns: список признаков ('all' или список)
                - method: метод регрессии ('linear', 'ridge', 'lasso', 'logistic', 'stepwise')
                - test_size: размер тестовой выборки
                - cv_folds: количество фолдов для кросс-валидации
                - alpha: параметр регуляризации
        """
        self.logger.info(f"Starting regression analysis on dataset {dataset_id}")

        # Загрузка данных
        df = self.preprocessor.get_processed_data(dataset_id)
        if df is None:
            from ..data_loader.service import DataLoaderService
            loader = DataLoaderService()
            df = loader.get_dataset(dataset_id)

        if df is None:
            raise AnalysisError(f"Dataset {dataset_id} not found")

        # Проверка наличия целевой переменной
        target = params.get('target_column')
        if target not in df.columns:
            raise AnalysisError(f"Target column '{target}' not found in dataset")

        # Выбор признаков
        feature_cols = params.get('feature_columns', 'all')
        if feature_cols == 'all':
            # Все числовые колонки кроме целевой
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in feature_cols:
                feature_cols.remove(target)
        else:
            # Проверка наличия всех признаков
            missing = [f for f in feature_cols if f not in df.columns]
            if missing:
                raise AnalysisError(f"Features not found: {missing}")

        if not feature_cols:
            raise InsufficientDataError(1, 0)

        # Подготовка данных
        model_data = df[[target] + feature_cols].dropna()

        if len(model_data) < 10:
            raise InsufficientDataError(10, len(model_data))

        X = model_data[feature_cols].values
        y = model_data[target].values

        # Извлечение параметров
        method = params.get('method', 'linear')
        test_size = params.get('test_size', 0.2)
        cv_folds = params.get('cv_folds', 5)
        alpha = params.get('alpha', 1.0 if method == 'ridge' else 0.1)

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Выбор и обучение модели
        if method == 'linear':
            model, model_info = self.linear_builder.build(
                X_train_scaled, y_train, feature_cols
            )
        elif method in ['ridge', 'lasso']:
            model, model_info = self.regularized_builder.build(
                X_train_scaled, y_train, feature_cols,
                method=method, alpha=alpha
            )
        elif method == 'logistic':
            model, model_info = self.logistic_builder.build(
                X_train_scaled, y_train, feature_cols
            )
        else:
            raise AnalysisError(f"Unknown regression method: {method}")

        # Предсказания
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Метрики
        if method == 'logistic':
            metrics = self._get_classification_metrics(
                y_train, y_train_pred, y_test, y_test_pred, model, X_test_scaled
            )
        else:
            metrics = self._get_regression_metrics(
                y_train, y_train_pred, y_test, y_test_pred,
                model, X_train_scaled, y_train, cv_folds
            )

        # Диагностика
        diagnostics = self.diagnostics.check(
            model, X_train_scaled, y_train, X_test_scaled, y_test,
            y_train_pred, y_test_pred, feature_cols
        ) if method != 'logistic' else {}

        # Важность признаков
        feature_importance = self._get_feature_importance(
            model, feature_cols, method
        )

        # Формирование результата
        result = {
            'method': method,
            'method_name': self._get_method_name(method),
            'target_variable': target,
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'n_samples': len(model_data),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'model_info': model_info,
            'coefficients': self._get_coefficients(model, feature_cols, method),
            'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else 0,
            'feature_importance': feature_importance,
            'metrics': metrics,
            'diagnostics': diagnostics,
            'predictions': {
                'train': y_train_pred.tolist(),
                'test': y_test_pred.tolist()
            },
            'parameters': params,
            'dataset_id': dataset_id
        }

        # Интерпретация
        result['interpretation'] = self._interpret(result)

        # Сохранение результатов
        result_id = self._save_results(result, dataset_id, method)
        result['result_id'] = result_id

        self.logger.info(f"Regression analysis completed: {result_id}")

        return result

    def _get_method_name(self, method: str) -> str:
        """Получение названия метода"""
        names = {
            'linear': 'Множественная линейная регрессия',
            'ridge': 'Ридж-регрессия',
            'lasso': 'Лассо-регрессия',
            'logistic': 'Логистическая регрессия',
            'stepwise': 'Пошаговая регрессия'
        }
        return names.get(method, method)

    def _get_regression_metrics(self, y_train, y_train_pred, y_test, y_test_pred,
                               model, X_train, y_train_cv, cv_folds) -> Dict[str, float]:
        """Расчет метрик для регрессии"""

        metrics = {
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
            'test_mae': float(mean_absolute_error(y_test, y_test_pred))
        }

        # Кросс-валидация
        try:
            cv_scores = cross_val_score(model, X_train, y_train_cv,
                                       cv=cv_folds, scoring='r2')
            metrics['cv_mean_r2'] = float(np.mean(cv_scores))
            metrics['cv_std_r2'] = float(np.std(cv_scores))
        except:
            metrics['cv_mean_r2'] = None
            metrics['cv_std_r2'] = None

        return metrics

    def _get_classification_metrics(self, y_train, y_train_pred, y_test, y_test_pred,
                                   model, X_test) -> Dict[str, float]:
        """Расчет метрик для классификации"""

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
            'train_precision': float(precision_score(y_train, y_train_pred, average='weighted')),
            'test_precision': float(precision_score(y_test, y_test_pred, average='weighted')),
            'train_recall': float(recall_score(y_train, y_train_pred, average='weighted')),
            'test_recall': float(recall_score(y_test, y_test_pred, average='weighted')),
            'train_f1': float(f1_score(y_train, y_train_pred, average='weighted')),
            'test_f1': float(f1_score(y_test, y_test_pred, average='weighted'))
        }

        # ROC-AUC для бинарной классификации
        if len(np.unique(y_train)) == 2:
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_test_proba))
            except:
                metrics['roc_auc'] = None

        return metrics

    def _get_coefficients(self, model, feature_cols: List[str],
                         method: str) -> Dict[str, float]:
        """Получение коэффициентов модели"""

        if not hasattr(model, 'coef_'):
            return {}

        coef = model.coef_
        if len(coef.shape) > 1:
            coef = coef[0]  # для многоклассовой логистической регрессии

        return {
            col: float(coef[i]) for i, col in enumerate(feature_cols)
        }

    def _get_feature_importance(self, model, feature_cols: List[str],
                               method: str) -> List[Dict[str, Any]]:
        """Получение важности признаков"""

        coef_dict = self._get_coefficients(model, feature_cols, method)

        # Сортировка по абсолютному значению
        importance = [
            {'feature': col, 'importance': abs(coef_dict.get(col, 0))}
            for col in feature_cols
        ]
        importance.sort(key=lambda x: x['importance'], reverse=True)

        return importance

    def _interpret(self, result: Dict[str, Any]) -> str:
        """Интерпретация результатов"""

        method_name = result['method_name']
        target = result['target_variable']
        metrics = result['metrics']
        importance = result['feature_importance']

        lines = [
            f"Регрессионный анализ: {method_name}",
            f"Целевая переменная: {target}",
            f"Количество признаков: {result['n_features']}"
        ]

        if 'test_r2' in metrics:
            r2 = metrics['test_r2']
            if r2 > 0.8:
                quality = "отличное"
            elif r2 > 0.6:
                quality = "хорошее"
            elif r2 > 0.4:
                quality = "удовлетворительное"
            else:
                quality = "низкое"

            lines.append(f"R² на тестовой выборке: {r2:.3f} ({quality} качество)")
            lines.append(f"RMSE на тестовой выборке: {metrics['test_rmse']:.3f}")

        if 'test_accuracy' in metrics:
            acc = metrics['test_accuracy']
            lines.append(f"Точность на тестовой выборке: {acc:.1%}")

        # Наиболее важные признаки
        if importance:
            top = importance[:3]
            lines.append("\nНаиболее важные признаки:")
            for f in top:
                lines.append(f"  • {f['feature']}: {f['importance']:.3f}")

        return '\n'.join(lines)

    def _save_results(self, results: Dict, dataset_id: str, method: str) -> str:
        """Сохранение результатов анализа"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = f"reg_{method}_{dataset_id}_{timestamp}"

        file_path = self.results_path / f"{result_id}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return result_id

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Получение результата по ID"""
        file_path = self.results_path / f"{result_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
