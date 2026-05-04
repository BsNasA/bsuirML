# services/preprocessor/service.py
"""
Сервис предобработки и очистки данных
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from pathlib import Path
import json

from ..config import config
from ..data_loader.service import DataLoaderService
from ..exceptions import PreprocessingError, MissingDataError
from ..logging_config import LoggerMixin

from .cleaners import DataCleaner
from .transformers import DataTransformer
from .validators import DataValidator


class PreprocessorService(LoggerMixin):
    """
    Сервис предобработки данных
    Выполняет очистку, трансформацию и валидацию данных
    """

    def __init__(self):
        self.data_loader = DataLoaderService()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.validator = DataValidator()

        self.storage_path = Path(config.DATA_STORAGE_PATH) / "processed"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("PreprocessorService initialized")

    def process(self, dataset_id: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Выполнение предобработки данных

        Args:
            dataset_id: идентификатор исходного датасета
            operations: список операций для выполнения

        Returns:
            Dict с результатами обработки
        """
        self.logger.info(f"Processing dataset {dataset_id} with {len(operations)} operations")

        # Загрузка исходных данных
        df = self.data_loader.get_dataset(dataset_id)
        if df is None:
            raise PreprocessingError(f"Dataset {dataset_id} not found")

        metadata = self.data_loader.get_metadata(dataset_id) or {}

        # Лог выполненных операций
        transformation_log = []

        # Применение операций
        for i, op in enumerate(operations):
            op_type = op.get('type')
            op_params = op.get('parameters', {})

            self.logger.debug(f"Applying operation {i+1}: {op_type}")

            try:
                if op_type == 'handle_missing':
                    df, stats = self.cleaner.handle_missing(df, **op_params)
                    transformation_log.append({
                        'operation': 'handle_missing',
                        'parameters': op_params,
                        'statistics': stats
                    })

                elif op_type == 'remove_outliers':
                    df, stats = self.cleaner.remove_outliers(df, **op_params)
                    transformation_log.append({
                        'operation': 'remove_outliers',
                        'parameters': op_params,
                        'statistics': stats
                    })

                elif op_type == 'normalize':
                    df, stats = self.transformer.normalize(df, **op_params)
                    transformation_log.append({
                        'operation': 'normalize',
                        'parameters': op_params,
                        'statistics': stats
                    })

                elif op_type == 'encode_categorical':
                    df, stats = self.transformer.encode_categorical(df, **op_params)
                    transformation_log.append({
                        'operation': 'encode_categorical',
                        'parameters': op_params,
                        'statistics': stats
                    })

                elif op_type == 'create_features':
                    df, stats = self.transformer.create_features(df, **op_params)
                    transformation_log.append({
                        'operation': 'create_features',
                        'parameters': op_params,
                        'statistics': stats
                    })

                elif op_type == 'select_columns':
                    df, stats = self.transformer.select_columns(df, **op_params)
                    transformation_log.append({
                        'operation': 'select_columns',
                        'parameters': op_params,
                        'statistics': stats
                    })

                elif op_type == 'filter_rows':
                    df, stats = self.transformer.filter_rows(df, **op_params)
                    transformation_log.append({
                        'operation': 'filter_rows',
                        'parameters': op_params,
                        'statistics': stats
                    })

                elif op_type == 'aggregate':
                    df, stats = self.transformer.aggregate(df, **op_params)
                    transformation_log.append({
                        'operation': 'aggregate',
                        'parameters': op_params,
                        'statistics': stats
                    })

                else:
                    self.logger.warning(f"Unknown operation type: {op_type}")

            except Exception as e:
                raise PreprocessingError(f"Error in operation {op_type}: {e}")

        # Валидация обработанных данных
        validation_report = self.validator.validate(df)

        # Сохранение обработанного датасета
        processed_id = self._save_processed_data(df, dataset_id, transformation_log)

        # Формирование отчета о качестве
        quality_report = self._generate_quality_report(
            df, metadata, transformation_log, validation_report
        )

        self.logger.info(f"Preprocessing completed: {processed_id}")

        return {
            'processed_dataset_id': processed_id,
            'transformations_log': transformation_log,
            'quality_report': quality_report,
            'validation_report': validation_report
        }

    def _save_processed_data(self, df: pd.DataFrame, original_id: str,
                            transformation_log: List[Dict]) -> str:
        """Сохранение обработанных данных"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_id = f"{original_id}_processed_{timestamp}"

        # Сохранение данных
        file_path = self.storage_path / f"{processed_id}.parquet"
        df.to_parquet(file_path, index=False)

        # Сохранение метаданных
        metadata = {
            'processed_id': processed_id,
            'original_id': original_id,
            'created_at': timestamp,
            'rows': len(df),
            'columns': len(df.columns),
            'size_bytes': file_path.stat().st_size,
            'transformations': transformation_log,
            'column_types': {col: str(df[col].dtype) for col in df.columns}
        }

        metadata_path = self.storage_path / f"{processed_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return processed_id

    def _generate_quality_report(self, df: pd.DataFrame, original_metadata: Dict,
                                transformation_log: List, validation_report: Dict) -> Dict:
        """Генерация отчета о качестве данных"""

        report = {
            'overall_quality': 'good',
            'issues': [],
            'warnings': [],
            'statistics': {}
        }

        # Проверка на пропущенные значения
        missing = df.isna().sum().sum()
        if missing > 0:
            missing_pct = missing / (df.shape[0] * df.shape[1]) * 100
            report['issues'].append({
                'type': 'missing_values',
                'count': int(missing),
                'percentage': float(missing_pct)
            })
            if missing_pct > 10:
                report['overall_quality'] = 'poor'
            elif missing_pct > 5:
                report['overall_quality'] = 'fair'

        # Проверка на выбросы
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            if outliers > 0:
                outlier_counts[col] = int(outliers)

        if outlier_counts:
            report['warnings'].append({
                'type': 'outliers',
                'counts': outlier_counts
            })

        # Статистика по типам данных
        report['statistics'] = {
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

        return report

    def get_processed_data(self, processed_id: str) -> Optional[pd.DataFrame]:
        """Получение обработанного датасета"""
        file_path = self.storage_path / f"{processed_id}.parquet"
        if not file_path.exists():
            return None
        return pd.read_parquet(file_path)

    def get_metadata(self, processed_id: str) -> Optional[Dict[str, Any]]:
        """Получение метаданных обработанного датасета"""
        metadata_path = self.storage_path / f"{processed_id}_metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
