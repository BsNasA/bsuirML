from ..celery_app import celery_app
from .correlation import CorrelationService
from .factor_analysis import FactorAnalysisService
from .neural_network import NeuralNetworkService
from ..preprocessor.service import PreprocessorService
from ..visualization.plots import VisualizationService
from ..report_generator.service import ReportGeneratorService


@celery_app.task(name="run_full_analysis_pipeline")
def run_full_analysis_pipeline(file_path: str, user_id: str):
    """
    Фоновая задача, которая объединяет все сервисы воедино.
    """
    # 1. Предобработка данных
    preprocessor = PreprocessorService()
    # Обрабатываем датасет (очищаем от пропусков и нормализуем)
    prep_data = preprocessor.process(
        file_path,
        operations=[
            {"type": "handle_missing", "parameters": {"strategy": "fill_median"}},
            {"type": "normalize", "parameters": {"method": "standard"}},
        ],
    )
    df = preprocessor.get_processed_data(prep_data["processed_dataset_id"])

    # 2. Анализ данных
    results = {}
    visualizer = VisualizationService()

    # Считаем корреляцию и сразу рисуем график
    corr_matrix = CorrelationService().calculate_correlation(df)
    results["correlation_matrix"] = corr_matrix.to_dict()
    results["corr_plot_path"] = visualizer.plot_correlation_heatmap(corr_matrix)

    # 3. Генерация PDF-отчета
    report_gen = ReportGeneratorService()
    report_path = report_gen.generate_comprehensive_report(
        analysis_results=results, filename_prefix=file_path
    )

    return {"status": "success", "report_path": report_path}
