# CSA Core Services

Python-проект собран по предоставленным методическим материалам системы комплексного статистического анализа (CSA).

## Структура

```text
services/
├── __init__.py
├── init.py
├── config.py
├── dispatcher.py
├── exceptions.py
├── logging_config.py
├── data_loader/
│   ├── __init__.py
│   └── service.py
├── preprocessor/
│   ├── __init__.py
│   ├── service.py
│   ├── cleaners.py
│   ├── transformers.py
│   └── validators.py
└── regression_analysis/
    ├── __init__.py
    ├── service.py
    ├── linear_models.py
    ├── logistic_models.py
    ├── regularized_models.py
    └── diagnostics.py
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Примечание

Реализованы только сервисы и операции, указанные в методических материалах: инициализация/конфигурация, предобработка данных и регрессионный анализ.
