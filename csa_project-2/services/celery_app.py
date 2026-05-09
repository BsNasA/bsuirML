import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from celery import Celery
from .config import config

# Инициализируем Celery
celery_app = Celery(
    "csa_tasks", broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND
)

# Настройки
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Если REDIS пока не запущен, поставьте тут True в .env (CSA_CELERY_EAGER=True),
    # чтобы код выполнился синхронно без ошибки
    task_always_eager=config.CELERY_TASK_ALWAYS_EAGER,
)

# Автоматический поиск задач
celery_app.autodiscover_tasks(["services.multivariate_analysis"])
