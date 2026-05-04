"""
Очистка данных для сервиса предобработки
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


class DataCleaner:
    """Класс очистки данных"""

    def handle_missing(self, df: pd.DataFrame, strategy: str = "drop",
                       columns: Optional[List[str]] = None, value: Any = None):
        result = df.copy()
        target_columns = columns or result.columns.tolist()
        before_missing = int(result[target_columns].isna().sum().sum())

        if strategy == "drop":
            result = result.dropna(subset=target_columns)
        elif strategy == "fill_mean":
            for col in target_columns:
                if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].mean())
        elif strategy == "fill_median":
            for col in target_columns:
                if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())
        elif strategy == "fill_mode":
            for col in target_columns:
                if col in result.columns:
                    mode = result[col].mode(dropna=True)
                    if not mode.empty:
                        result[col] = result[col].fillna(mode.iloc[0])
        elif strategy == "interpolate":
            result[target_columns] = result[target_columns].interpolate()
        elif strategy == "fill_value":
            result[target_columns] = result[target_columns].fillna(value)

        after_missing = int(result[target_columns].isna().sum().sum()) if len(result) else 0
        stats = {
            "strategy": strategy,
            "missing_before": before_missing,
            "missing_after": after_missing,
            "rows_before": int(len(df)),
            "rows_after": int(len(result)),
        }
        return result, stats

    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr",
                        columns: Optional[List[str]] = None, threshold: float = 3.0):
        result = df.copy()
        numeric_cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()
        mask = pd.Series(True, index=result.index)
        removed_by_column: Dict[str, int] = {}

        for col in numeric_cols:
            if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
                continue
            col_mask = pd.Series(True, index=result.index)
            if method == "iqr":
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                col_mask = (result[col] >= q1 - 1.5 * iqr) & (result[col] <= q3 + 1.5 * iqr)
            elif method == "zscore":
                std = result[col].std()
                if std and not pd.isna(std):
                    z = (result[col] - result[col].mean()) / std
                    col_mask = z.abs() <= threshold
            removed_by_column[col] = int((~col_mask).sum())
            mask &= col_mask.fillna(True)

        filtered = result[mask]
        stats = {
            "method": method,
            "rows_before": int(len(result)),
            "rows_after": int(len(filtered)),
            "removed_rows": int(len(result) - len(filtered)),
            "removed_by_column": removed_by_column,
        }
        return filtered, stats
