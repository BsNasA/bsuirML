# services/config.py
"""
Конфигурация сервисов системы комплексного статистического анализа
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class ServicesConfig(BaseSettings):
    """
    Конфигурация всех сервисов
    Загружается из переменных окружения или файла .env
    """

    # Общие настройки
    DEBUG: bool = Field(False, env="CSA_DEBUG")
    TESTING: bool = Field(False, env="CSA_TESTING")
    SECRET_KEY: str = Field("csa-secret-key-change-in-production", env="CSA_SECRET_KEY")
    ENVIRONMENT: str = Field("development", env="CSA_ENVIRONMENT")

    # Настройки API
    API_PREFIX: str = Field("/api/v1", env="CSA_API_PREFIX")
    API_TITLE: str = "CSA Core Services"
    API_VERSION: str = "1.0.0"
    API_HOST: str = Field("0.0.0.0", env="CSA_API_HOST")
    API_PORT: int = Field(8000, env="CSA_API_PORT")
    API_WORKERS: int = Field(4, env="CSA_API_WORKERS")

    # CORS
    CORS_ORIGINS: list = Field(["*"], env="CSA_CORS_ORIGINS")

    # Настройки Celery
    CELERY_BROKER_URL: str = Field("redis://localhost:6379/0", env="CSA_CELERY_BROKER")
    CELERY_RESULT_BACKEND: str = Field("redis://localhost:6379/0", env="CSA_CELERY_BACKEND")
    CELERY_TASK_ALWAYS_EAGER: bool = Field(False, env="CSA_CELERY_EAGER")

    # Настройки Redis
    REDIS_HOST: str = Field("localhost", env="CSA_REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="CSA_REDIS_PORT")
    REDIS_DB: int = Field(0, env="CSA_REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(None, env="CSA_REDIS_PASSWORD")

    # Настройки кэша
    CACHE_ENABLED: bool = Field(True, env="CSA_CACHE_ENABLED")
    CACHE_TTL: int = Field(3600, env="CSA_CACHE_TTL")  # 1 час
    CACHE_MAX_SIZE: int = Field(1024, env="CSA_CACHE_MAX_SIZE")  # МБ

    # Настройки базы данных результатов
    RESULTS_DB_URL: str = Field("sqlite:///./results.db", env="CSA_RESULTS_DB")

    # Настройки файлового хранилища
    DATA_STORAGE_PATH: str = Field(
        default_factory=lambda: os.getenv("CSA_DATA_PATH", "./data"),
        env="CSA_DATA_PATH"
    )
    RESULTS_STORAGE_PATH: str = Field(
        default_factory=lambda: os.getenv("CSA_RESULTS_PATH", "./results"),
        env="CSA_RESULTS_PATH"
    )
    REPORTS_STORAGE_PATH: str = Field(
        default_factory=lambda: os.getenv("CSA_REPORTS_PATH", "./reports"),
        env="CSA_REPORTS_PATH"
    )

    # Настройки визуализации
    PLOT_BACKEND: str = Field("plotly", env="CSA_PLOT_BACKEND")  # matplotlib, plotly, bokeh
    PLOT_DPI: int = Field(300, env="CSA_PLOT_DPI")
    PLOT_DEFAULT_WIDTH: int = Field(1200, env="CSA_PLOT_WIDTH")
    PLOT_DEFAULT_HEIGHT: int = Field(800, env="CSA_PLOT_HEIGHT")

    # Настройки отчетов
    REPORT_TEMPLATES_PATH: str = Field("./templates", env="CSA_TEMPLATES_PATH")
    PDF_ENGINE: str = Field("weasyprint", env="CSA_PDF_ENGINE")  # weasyprint, reportlab

    # Настройки нейросетей
    TENSORFLOW_ENABLED: bool = Field(True, env="CSA_TF_ENABLED")
    PYTORCH_ENABLED: bool = Field(True, env="CSA_PT_ENABLED")
    USE_GPU: bool = Field(False, env="CSA_USE_GPU")
    NEURAL_DEFAULT_EPOCHS: int = Field(50, env="CSA_NN_EPOCHS")
    NEURAL_DEFAULT_BATCH_SIZE: int = Field(32, env="CSA_NN_BATCH_SIZE")

    # Настройки предметных доменов
    DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
        "manufacturing": {
            "name": "Производство",
            "icon": "🏭",
            "default_metrics": ["oee", "downtime", "quality_rate", "throughput"],
            "visualization_templates": ["production_line", "equipment_status", "quality_control"],
            "report_sections": ["overview", "equipment", "quality", "costs"]
        },
        "education": {
            "name": "Образование",
            "icon": "🎓",
            "default_metrics": ["grades", "attendance_rate", "engagement_score", "retention"],
            "visualization_templates": ["learning_progress", "class_performance", "student_engagement"],
            "report_sections": ["overview", "performance", "attendance", "recommendations"]
        },
        "e_medicine": {
            "name": "Электронная медицина",
            "icon": "🏥",
            "default_metrics": ["diagnosis_accuracy", "treatment_efficacy", "patient_satisfaction"],
            "visualization_templates": ["patient_timeline", "health_indicators", "treatment_outcomes"],
            "report_sections": ["overview", "diagnostics", "treatment", "recommendations"]
        }
    }

    # Настройки безопасности
    JWT_SECRET: str = Field("jwt-secret-key", env="CSA_JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = Field(86400, env="CSA_JWT_EXPIRATION")  # 24 часа

    # Настройки логирования
    LOG_LEVEL: str = Field("INFO", env="CSA_LOG_LEVEL")
    LOG_FORMAT: str = "json"  # json или text
    LOG_FILE: Optional[str] = Field(None, env="CSA_LOG_FILE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Глобальный экземпляр конфигурации
config = ServicesConfig()
