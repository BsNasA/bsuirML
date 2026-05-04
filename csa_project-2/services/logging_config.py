"""
Конфигурация логирования сервисов
"""

import logging


class LoggerMixin:
    """Mixin для получения логгера класса"""

    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)
