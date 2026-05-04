import os
import sys
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

GUI_PATH = Path(__file__).resolve().parent
CSA_PROJECT_PATH = (GUI_PATH / "../csa_project-2").resolve()

# Пути CSA
CSA_DATA_PATH = CSA_PROJECT_PATH / "data"
CSA_RESULTS_PATH = CSA_PROJECT_PATH / "results"
CSA_REPORTS_PATH = CSA_PROJECT_PATH / "reports"

CSA_DATA_PATH.mkdir(parents=True, exist_ok=True)
CSA_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
CSA_REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Важно: задаём переменные окружения ДО импорта services
os.environ["CSA_DATA_PATH"] = str(CSA_DATA_PATH)
os.environ["CSA_RESULTS_PATH"] = str(CSA_RESULTS_PATH)
os.environ["CSA_REPORTS_PATH"] = str(CSA_REPORTS_PATH)

sys.path.insert(0, str(CSA_PROJECT_PATH))

from services.regression_analysis.service import RegressionAnalysisService

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".docx"}

app = Flask(
    __name__,
    template_folder=str(GUI_PATH / "templates"),
    static_folder=str(GUI_PATH / "static")
)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    uploaded_dataset_id = None
    uploaded_filename = None
    saved_path = None
    target_column_value = "salary"
    feature_columns_value = "age,experience"
    method_value = "linear"

    if request.method == "POST":
        try:
            file = request.files.get("dataset_file")

            if file is None or file.filename == "":
                raise ValueError("Файл датасета не выбран")

            filename = secure_filename(file.filename)
            extension = Path(filename).suffix.lower()

            if extension not in ALLOWED_EXTENSIONS:
                raise ValueError("Поддерживаются только файлы CSV, XLSX и DOCX")

            dataset_id = Path(filename).stem
            save_path = CSA_DATA_PATH / filename
            file.save(save_path)

            uploaded_dataset_id = dataset_id
            uploaded_filename = filename
            saved_path = str(save_path)

            target_column = request.form.get("target_column")
            feature_columns_raw = request.form.get("feature_columns")
            method = request.form.get("method")
            target_column_value = target_column or ""
            feature_columns_value = feature_columns_raw or ""
            method_value = method or "linear"

            if not target_column:
                raise ValueError("Не указана целевая переменная")

            if not feature_columns_raw:
                raise ValueError("Не указаны признаки")

            feature_columns = [
                col.strip()
                for col in feature_columns_raw.split(",")
                if col.strip()
            ]

            service = RegressionAnalysisService()

            result = service.analyze(
                dataset_id=dataset_id,
                params={
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "method": method,
                    "test_size": 0.2,
                    "cv_folds": 3
                }
            )

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        uploaded_dataset_id=uploaded_dataset_id,
        uploaded_filename=uploaded_filename,
        saved_path=saved_path,
        target_column_value=target_column_value,
        feature_columns_value=feature_columns_value,
        method_value=method_value
    )


if __name__ == "__main__":
    print("GUI:", GUI_PATH)
    print("CSA:", CSA_PROJECT_PATH)
    print("DATA:", CSA_DATA_PATH)
    app.run(debug=True)
