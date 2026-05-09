from .dispatcher import ServiceDispatcher
from .data_loader.service import DataLoaderService
from .preprocessor.service import PreprocessorService
from .regression_analysis.service import RegressionAnalysisService
from .multivariate_analysis.service import MultivariateAnalysisService
from .report_generator.service import ReportGeneratorService

def initialize_services() -> ServiceDispatcher:
    dispatcher = ServiceDispatcher()
    
    # Регистрация всех ваших модулей
    dispatcher.register("data_loader", DataLoaderService())
    dispatcher.register("preprocessor", PreprocessorService())
    dispatcher.register("regression", RegressionAnalysisService())
    dispatcher.register("multivariate", MultivariateAnalysisService())
    dispatcher.register("reports", ReportGeneratorService())
    
    return dispatcher

# Создаем глобальный объект, который вы будете импортировать в API
csa_dispatcher = initialize_services()