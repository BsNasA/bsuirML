"""
Валидация данных для сервиса предобработки
"""

import numpy as np
import pandas as pd


class DataValidator:
    """Класс валидации данных"""

    def validate(self, df: pd.DataFrame):
        duplicate_rows = int(df.duplicated().sum())
        missing_values = int(df.isna().sum().sum())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_rows": duplicate_rows,
            "missing_values": missing_values,
            "numeric_columns": int(len(numeric_cols)),
            "is_valid": duplicate_rows == 0 and len(df.columns) > 0,
        }
