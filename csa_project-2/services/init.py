"""
Модуль основных сервисов системы комплексного статистического анализа
Версия: 1.0.0
"""

from .config import ServicesConfig
from .dispatcher import ServiceDispatcher
from .exceptions import (
    CSAServiceError, DataLoadError, PreprocessingError,
    AnalysisError, VisualizationError, ReportError
)

__version__ = "1.0.0"

__all__ = [
    'ServicesConfig',
    'ServiceDispatcher',
    'CSAServiceError',
    'DataLoadError',
    'PreprocessingError',
    'AnalysisError',
    'VisualizationError',
    'ReportError'
]
