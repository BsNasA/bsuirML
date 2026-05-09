# services/multivariate_analysis/service.py

import logging
from typing import Dict, Any
from .tasks import run_full_analysis_pipeline

logger = logging.getLogger(__name__)

class MultivariateAnalysisService:
    """
    Сервис для управления многомерным статистическим анализом.
    """
    def __init__(self):
        logger.info("MultivariateAnalysisService инициализирован")

    def start_analysis(self, file_path: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Запускает задачу в Celery и возвращает ID задачи.
        """
        try:
            # Отправляем в фоновую очередь
            task = run_full_analysis_pipeline.delay(file_path, user_id)
            
            return {
                "status": "accepted",
                "message": "Задача успешно добавлена в очередь на обработку.",
                "task_id": task.id,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Ошибка при запуске анализа: {str(e)}")
            return {
                "status": "error",
                "message": "Не удалось запустить анализ.",
                "details": str(e)
            }