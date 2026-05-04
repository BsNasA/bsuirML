"""
Сервис загрузки данных
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import pandas as pd

from ..config import config
from ..exceptions import DataLoadError
from ..logging_config import LoggerMixin


class DataLoaderService(LoggerMixin):
    """Сервис загрузки данных"""

    def __init__(self):
        self.storage_path = Path(config.DATA_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        for suffix, reader in [
            (".parquet", pd.read_parquet),
            (".csv", pd.read_csv),
            (".xlsx", pd.read_excel),
            (".json", pd.read_json),
        ]:
            file_path = self.storage_path / f"{dataset_id}{suffix}"
            if file_path.exists():
                return reader(file_path)
        return None

    def get_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        metadata_path = self.storage_path / f"{dataset_id}_metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
