"""
Сервис загрузки данных
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

from ..config import config
from ..exceptions import DataLoadError
from ..logging_config import LoggerMixin


class DataLoaderService(LoggerMixin):
    """Сервис загрузки данных"""

    WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    def __init__(self):
        self.storage_path = Path(config.DATA_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        for suffix, reader in [
            (".parquet", pd.read_parquet),
            (".csv", pd.read_csv),
            (".xlsx", pd.read_excel),
            (".json", pd.read_json),
            (".docx", self._read_docx_table),
        ]:
            file_path = self.storage_path / f"{dataset_id}{suffix}"
            if file_path.exists():
                try:
                    return reader(file_path)
                except Exception as exc:
                    raise DataLoadError(
                        f"Failed to load dataset '{dataset_id}' from {file_path.name}: {exc}"
                    ) from exc
        return None

    def get_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        metadata_path = self.storage_path / f"{dataset_id}_metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_docx_table(self, file_path: Path) -> pd.DataFrame:
        """Извлекает первую осмысленную таблицу из DOCX-файла."""

        try:
            with zipfile.ZipFile(file_path) as archive:
                document_xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise DataLoadError("DOCX does not contain word/document.xml") from exc
        except zipfile.BadZipFile as exc:
            raise DataLoadError("Invalid DOCX archive") from exc

        root = ET.fromstring(document_xml)
        tables = root.findall(".//w:tbl", self.WORD_NAMESPACE)

        best_rows = []
        for table in tables:
            rows = self._extract_table_rows(table)
            if len(rows) > len(best_rows):
                best_rows = rows

        if len(best_rows) < 2:
            raise DataLoadError("DOCX must contain a table with a header row and at least one data row")

        header = self._normalize_headers(best_rows[0])
        data_rows = [self._fit_row_to_header(row, len(header)) for row in best_rows[1:]]
        frame = pd.DataFrame(data_rows, columns=header)

        if frame.empty:
            raise DataLoadError("No data rows found in DOCX table")

        return self._coerce_column_types(frame)

    def _extract_table_rows(self, table: ET.Element) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in table.findall("./w:tr", self.WORD_NAMESPACE):
            cells = []
            for cell in row.findall("./w:tc", self.WORD_NAMESPACE):
                text_nodes = cell.findall(".//w:t", self.WORD_NAMESPACE)
                text = "".join(node.text or "" for node in text_nodes).strip()
                cells.append(text)
            if any(cell for cell in cells):
                rows.append(cells)
        return rows

    def _normalize_headers(self, header_row: list[str]) -> list[str]:
        headers = []
        seen: dict[str, int] = {}

        for index, raw_name in enumerate(header_row, start=1):
            base_name = raw_name.strip() or f"column_{index}"
            count = seen.get(base_name, 0)
            seen[base_name] = count + 1
            headers.append(base_name if count == 0 else f"{base_name}_{count + 1}")

        return headers

    def _fit_row_to_header(self, row: list[str], width: int) -> list[str]:
        padded = list(row[:width])
        if len(padded) < width:
            padded.extend([""] * (width - len(padded)))
        return padded

    def _coerce_column_types(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()

        for column in result.columns:
            series = result[column].astype(str).str.strip()
            non_empty = series.replace("", pd.NA)
            numeric = pd.to_numeric(non_empty, errors="coerce")

            if non_empty.notna().any() and numeric.notna().sum() == non_empty.notna().sum():
                result[column] = numeric
            else:
                result[column] = non_empty

        return result
