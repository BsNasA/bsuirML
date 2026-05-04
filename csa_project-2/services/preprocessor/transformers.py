"""
Трансформация данных для сервиса предобработки
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


class DataTransformer:
    """Класс трансформации данных"""

    def normalize(self, df: pd.DataFrame, method: str = "standard",
                  columns: Optional[List[str]] = None):
        result = df.copy()
        numeric_cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()
        stats: Dict[str, Any] = {"method": method, "columns": []}

        for col in numeric_cols:
            if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
                continue
            if method == "minmax":
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val != min_val:
                    result[col] = (result[col] - min_val) / (max_val - min_val)
                stats["columns"].append({"column": col, "min": float(min_val), "max": float(max_val)})
            elif method == "standard":
                mean_val = result[col].mean()
                std_val = result[col].std()
                if std_val and not pd.isna(std_val):
                    result[col] = (result[col] - mean_val) / std_val
                stats["columns"].append({"column": col, "mean": float(mean_val), "std": float(std_val)})
            elif method == "robust":
                median_val = result[col].median()
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                if iqr:
                    result[col] = (result[col] - median_val) / iqr
                stats["columns"].append({"column": col, "median": float(median_val), "iqr": float(iqr)})
        return result, stats

    def encode_categorical(self, df: pd.DataFrame, method: str = "onehot",
                           columns: Optional[List[str]] = None):
        result = df.copy()
        cat_cols = columns or result.select_dtypes(include=["object", "category"]).columns.tolist()
        stats = {"method": method, "columns": cat_cols}

        if method == "onehot":
            result = pd.get_dummies(result, columns=cat_cols, drop_first=False)
        elif method == "label":
            mappings = {}
            for col in cat_cols:
                if col in result.columns:
                    result[col] = result[col].astype("category")
                    mappings[col] = {str(k): int(v) for v, k in enumerate(result[col].cat.categories)}
                    result[col] = result[col].cat.codes
            stats["mappings"] = mappings
        return result, stats

    def create_features(self, df: pd.DataFrame, features: Optional[List[Dict[str, Any]]] = None):
        result = df.copy()
        created = []
        for feature in features or []:
            name = feature.get("name")
            operation = feature.get("operation")
            columns = feature.get("columns", [])
            if not name or not columns:
                continue
            if operation == "sum":
                result[name] = result[columns].sum(axis=1)
            elif operation == "mean":
                result[name] = result[columns].mean(axis=1)
            elif operation == "difference" and len(columns) >= 2:
                result[name] = result[columns[0]] - result[columns[1]]
            elif operation == "ratio" and len(columns) >= 2:
                result[name] = result[columns[0]] / result[columns[1]].replace(0, np.nan)
            else:
                continue
            created.append(name)
        return result, {"created_features": created}

    def select_columns(self, df: pd.DataFrame, columns: List[str]):
        selected = [col for col in columns if col in df.columns]
        result = df[selected].copy()
        stats = {"columns_before": int(len(df.columns)), "columns_after": int(len(result.columns)), "selected_columns": selected}
        return result, stats

    def filter_rows(self, df: pd.DataFrame, conditions: Optional[List[Dict[str, Any]]] = None):
        result = df.copy()
        mask = pd.Series(True, index=result.index)
        for condition in conditions or []:
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")
            if column not in result.columns:
                continue
            if operator == "eq":
                mask &= result[column] == value
            elif operator == "ne":
                mask &= result[column] != value
            elif operator == "gt":
                mask &= result[column] > value
            elif operator == "gte":
                mask &= result[column] >= value
            elif operator == "lt":
                mask &= result[column] < value
            elif operator == "lte":
                mask &= result[column] <= value
            elif operator == "in":
                mask &= result[column].isin(value)
        filtered = result[mask]
        stats = {"rows_before": int(len(result)), "rows_after": int(len(filtered)), "removed_rows": int(len(result) - len(filtered))}
        return filtered, stats

    def aggregate(self, df: pd.DataFrame, group_by: List[str], aggregations: Dict[str, Any]):
        result = df.groupby(group_by).agg(aggregations).reset_index()
        stats = {"rows_before": int(len(df)), "rows_after": int(len(result)), "group_by": group_by, "aggregations": aggregations}
        return result, stats
