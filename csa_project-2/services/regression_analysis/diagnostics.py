# services/regression_analysis/diagnostics.py
"""
Диагностика регрессионных моделей
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import scipy.stats as stats


class RegressionDiagnostics:
    """Класс для диагностики регрессионных моделей"""

    def check(self, model, X_train, y_train, X_test, y_test,
             y_train_pred, y_test_pred, feature_names: List[str]) -> Dict[str, Any]:
        """
        Проверка предпосылок регрессионного анализа
        """
        diagnostics = {}

        # Остатки
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred

        # 1. Проверка на нормальность остатков
        diagnostics['normality'] = self._check_normality(train_residuals)

        # 2. Проверка на гомоскедастичность
        diagnostics['homoscedasticity'] = self._check_homoscedasticity(
            y_train_pred, train_residuals
        )

        # 3. Проверка на автокорреляцию (Дарбин-Уотсон)
        diagnostics['autocorrelation'] = self._check_autocorrelation(train_residuals)

        # 4. Проверка на мультиколлинеарность (VIF)
        if hasattr(X_train, 'shape') and X_train.shape[1] > 1:
            diagnostics['multicollinearity'] = self._check_multicollinearity(
                X_train, feature_names
            )

        # 5. Влиятельные наблюдения
        diagnostics['influential_points'] = self._check_influential_points(
            X_train, train_residuals
        )

        # 6. Общая оценка
        diagnostics['overall_assessment'] = self._assess_overall(diagnostics)

        return diagnostics

    def _check_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Проверка нормальности остатков"""

        # Тест Шапиро-Уилка (для небольших выборок)
        if len(residuals) < 5000:
            statistic, p_value = stats.shapiro(residuals[:5000])
            test_name = "Shapiro-Wilk"
        else:
            # Критерий Колмогорова-Смирнова
            statistic, p_value = stats.kstest(residuals, 'norm',
                                             args=(np.mean(residuals), np.std(residuals)))
            test_name = "Kolmogorov-Smirnov"

        is_normal = p_value > 0.05

        return {
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_normal': bool(is_normal),
            'interpretation': 'Нормальные' if is_normal else 'Не нормальные'
        }

    def _check_homoscedasticity(self, fitted: np.ndarray,
                               residuals: np.ndarray) -> Dict[str, Any]:
        """Проверка гомоскедастичности (тест Бройша-Пагана)"""

        # Упрощенная проверка: корреляция между fitted и |residuals|
        abs_resid = np.abs(residuals)
        corr, p_value = stats.pearsonr(fitted, abs_resid)

        is_homoscedastic = p_value > 0.05 or abs(corr) < 0.1

        return {
            'correlation_with_fitted': float(corr),
            'p_value': float(p_value),
            'is_homoscedastic': bool(is_homoscedastic),
            'interpretation': 'Гомоскедастичность' if is_homoscedastic else 'Гетероскедастичность'
        }

    def _check_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Проверка автокорреляции (статистика Дарбина-Уотсона)"""

        # Расчет статистики Дарбина-Уотсона
        diff = np.diff(residuals)
        dw = np.sum(diff**2) / np.sum(residuals**2)

        # Интерпретация
        if dw < 1.5:
            interpretation = "Положительная автокорреляция"
            has_autocorrelation = True
        elif dw > 2.5:
            interpretation = "Отрицательная автокорреляция"
            has_autocorrelation = True
        else:
            interpretation = "Нет автокорреляции"
            has_autocorrelation = False

        return {
            'durbin_watson': float(dw),
            'has_autocorrelation': has_autocorrelation,
            'interpretation': interpretation
        }

    def _check_multicollinearity(self, X: np.ndarray,
                                feature_names: List[str]) -> Dict[str, Any]:
        """Проверка мультиколлинеарности (VIF)"""

        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_data = []
        for i in range(X.shape[1]):
            try:
                vif = variance_inflation_factor(X, i)
            except Exception:
                # На вырожденных матрицах/константных колонках VIF может не вычислиться.
                vif = np.inf
            vif_data.append({
                'feature': feature_names[i],
                'vif': float(vif)
            })

        # Определение максимального VIF
        max_vif = max(v['vif'] for v in vif_data)

        if max_vif > 10:
            severity = "критическая"
            has_multicollinearity = True
        elif max_vif > 5:
            severity = "умеренная"
            has_multicollinearity = True
        else:
            severity = "низкая"
            has_multicollinearity = False

        return {
            'vif_values': vif_data,
            'max_vif': float(max_vif),
            'has_multicollinearity': has_multicollinearity,
            'severity': severity
        }

    def _check_influential_points(self, X: np.ndarray,
                                 residuals: np.ndarray) -> Dict[str, Any]:
        """Проверка влиятельных наблюдений (Cook's distance)"""

        n, p = X.shape
        if n <= p + 1:
            return {'warning': 'Insufficient data for influential points analysis'}

        var_resid = np.var(residuals)
        if not np.isfinite(var_resid) or var_resid <= 1e-12:
            return {'warning': 'Residual variance is too small for stable Cook`s distance'}

        # Используем псевдообратную матрицу, чтобы избежать падений на сингулярных X^T X.
        xtx_pinv = np.linalg.pinv(X.T @ X)
        h = np.diag(X @ xtx_pinv @ X.T)
        h = np.clip(h, 0.0, 0.999999)
        cooks = (residuals**2 / (p * var_resid)) * (h / (1 - h)**2)
        cooks = np.nan_to_num(cooks, nan=0.0, posinf=0.0, neginf=0.0)

        influential = np.sum(cooks > 4/(n-p-1))

        return {
            'influential_count': int(influential),
            'influential_percentage': float(influential / n * 100),
            'max_cooks_distance': float(np.max(cooks))
        }

    def _assess_overall(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Общая оценка диагностики"""
        issues = []
        if not diagnostics.get('normality', {}).get('is_normal', True):
            issues.append('normality')
        if not diagnostics.get('homoscedasticity', {}).get('is_homoscedastic', True):
            issues.append('homoscedasticity')
        if diagnostics.get('autocorrelation', {}).get('has_autocorrelation', False):
            issues.append('autocorrelation')
        if diagnostics.get('multicollinearity', {}).get('has_multicollinearity', False):
            issues.append('multicollinearity')
        return {
            'quality': 'good' if not issues else 'requires_attention',
            'issues': issues
        }
