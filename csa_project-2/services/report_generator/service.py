import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Импортируем ваш мощный конфиг
from services.config import config 

class ReportGeneratorService:
    def __init__(self):
        # Используем пути из конфига, создаем директорию, если её нет
        self.reports_dir = config.REPORTS_STORAGE_PATH
        os.makedirs(self.reports_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()

    def generate_comprehensive_report(self, analysis_results: dict, filename_prefix: str = "csa_report") -> str:
        """
        Генерирует PDF-отчет с результатами корреляционного, факторного и нейросетевого анализов.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pdf"
        filepath = os.path.join(self.reports_dir, filename)

        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []

        # Заголовок
        story.append(Paragraph("Отчет по комплексному статистическому анализу (CSA)", self.styles['Title']))
        story.append(Spacer(1, 12))

        # 1. Секция корреляционного анализа
        if "correlation_matrix" in analysis_results:
            story.append(Paragraph("1. Корреляционный анализ", self.styles['Heading1']))
            story.append(Paragraph("Ниже представлена тепловая карта значимых корреляций.", self.styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Вставляем график, если он был сгенерирован
            if "corr_plot_path" in analysis_results and os.path.exists(analysis_results["corr_plot_path"]):
                img = Image(analysis_results["corr_plot_path"], width=400, height=300)
                story.append(img)
            story.append(Spacer(1, 12))

        # 2. Секция факторного анализа
        if "factor_loadings" in analysis_results:
            story.append(Paragraph("2. Факторный анализ", self.styles['Heading1']))
            story.append(Paragraph("Результаты выделения скрытых (латентных) факторов.", self.styles['Normal']))
            story.append(Spacer(1, 12))
            
            if "factor_plot_path" in analysis_results and os.path.exists(analysis_results["factor_plot_path"]):
                img = Image(analysis_results["factor_plot_path"], width=400, height=300)
                story.append(img)
            story.append(Spacer(1, 12))

        # 3. Секция нейросетевого анализа
        if "nn_metrics" in analysis_results:
            story.append(Paragraph("3. Нейросетевой анализ", self.styles['Heading1']))
            story.append(Paragraph("Метрики качества обученной модели (Многослойный перцептрон):", self.styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Формируем таблицу с метриками
            metrics_data = [["Метрика", "Значение"]]
            for key, value in analysis_results["nn_metrics"].items():
                # Форматируем значения до 4 знаков после запятой
                formatted_val = f"{value:.4f}" if isinstance(value, float) else str(value)
                metrics_data.append([key, formatted_val])
                
            table = Table(metrics_data, colWidths=[200, 200])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)

        # Генерация документа
        doc.build(story)
        return filepath