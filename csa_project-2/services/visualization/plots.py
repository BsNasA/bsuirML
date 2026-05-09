import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


class VisualizationService:
    def __init__(self, output_dir: str = "reports/images"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_correlation_heatmap(
        self, corr_matrix: pd.DataFrame, filename: str = "corr_heatmap.png"
    ):
        """Рисует и сохраняет тепловую карту корреляций."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1
        )
        plt.title("Матрица корреляций")
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath

    def plot_factor_loadings(
        self, loadings: pd.DataFrame, filename: str = "factor_loadings.png"
    ):
        """Визуализация факторных нагрузок (тепловая карта)."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(loadings, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Факторные нагрузки")
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath
