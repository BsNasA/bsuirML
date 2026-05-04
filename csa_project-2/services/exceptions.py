"""
Исключения сервисов системы комплексного статистического анализа
"""


class CSAServiceError(Exception):
    """Базовое исключение сервисов CSA"""


class DataLoadError(CSAServiceError):
    """Ошибка загрузки данных"""


class PreprocessingError(CSAServiceError):
    """Ошибка предобработки данных"""


class AnalysisError(CSAServiceError):
    """Ошибка анализа данных"""


class VisualizationError(CSAServiceError):
    """Ошибка визуализации"""


class ReportError(CSAServiceError):
    """Ошибка формирования отчета"""


class MissingDataError(PreprocessingError):
    """Ошибка отсутствующих данных"""


class InsufficientDataError(AnalysisError):
    """Ошибка недостаточного объема данных"""

    def __init__(self, required: int, actual: int):
        super().__init__(f"Insufficient data: required {required}, actual {actual}")
        self.required = required
        self.actual = actual
