"""Generate diagnostic plots for completed experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import EvaluationResult
from src.evaluation.reporter import ExperimentResult, ExperimentStatus


class ResultPlotter:
    """Save curves and comparisons instead of relying on manual screenshots."""

    def __init__(self, plots_dir: Path) -> None:
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-experiment plots
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        history: list[dict[str, float]],
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        epochs = [int(row["epoch"]) for row in history]
        loss_pairs = [
            ("total_loss", "Perdida total"),
            ("gender_loss", "Perdida de genero"),
            ("age_loss", "Perdida de edad"),
        ]

        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, (key, title) in zip(axes, loss_pairs):
            axis.plot(epochs, [row[f"train_{key}"] for row in history], label="train")
            axis.plot(epochs, [row[f"val_{key}"] for row in history], label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.legend()
            axis.grid(alpha=0.3)
        figure.suptitle(experiment_name, fontsize=10)
        figure.tight_layout()

        output_path = output_dir / "training_curves.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def plot_confusion_matrix(
        self,
        evaluation: EvaluationResult,
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        matrix = evaluation.confusion_matrix
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_xticks([0, 1], labels=["Masculino", "Femenino"])
        axis.set_yticks([0, 1], labels=["Masculino", "Femenino"])
        axis.set_xlabel("Genero predicho")
        axis.set_ylabel("Genero real")
        axis.set_title(f"Matriz de confusion\n{experiment_name}")

        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row][col]), ha="center", va="center",
                          fontsize=12, color="black")
        figure.tight_layout()

        output_path = output_dir / "gender_confusion_matrix.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def plot_age_predictions(
        self,
        evaluation: EvaluationResult,
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(
            evaluation.age_targets,
            evaluation.age_predictions,
            alpha=0.35,
            s=14,
        )
        all_ages = evaluation.age_targets + evaluation.age_predictions
        lower = min(all_ages)
        upper = max(all_ages)
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black",
                  label="prediccion perfecta")
        axis.set_xlabel("Edad real")
        axis.set_ylabel("Edad predicha")
        axis.set_title(f"Edad real vs. predicha\n{experiment_name}")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()

        output_path = output_dir / "age_real_vs_predicted.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    # ------------------------------------------------------------------
    # Strategy-level comparison plots
    # ------------------------------------------------------------------

    def plot_ablation_comparison(
        self,
        results: list[ExperimentResult],
        strategy_id: str,
    ) -> list[Path]:
        completed = [
            r for r in results
            if r.strategy_id == strategy_id
            and r.status == ExperimentStatus.COMPLETED
        ]
        if not completed:
            return []

        names = [r.experiment_name for r in completed]
        gender_accuracy = [float(r.metrics["gender_accuracy"]) for r in completed]
        gender_f1 = [float(r.metrics["gender_f1"]) for r in completed]
        age_mae = [float(r.metrics["age_mae"]) for r in completed]
        age_rmse = [float(r.metrics["age_rmse"]) for r in completed]

        output_paths = [
            self._bar_plot(
                names,
                [gender_accuracy, gender_f1],
                ["Accuracy", "F1"],
                f"{strategy_id}: metricas de genero",
                self.plots_dir / f"{strategy_id.lower()}_ablation_gender_metrics.png",
            ),
            self._bar_plot(
                names,
                [age_mae, age_rmse],
                ["MAE", "RMSE"],
                f"{strategy_id}: metricas de edad",
                self.plots_dir / f"{strategy_id.lower()}_ablation_age_metrics.png",
            ),
        ]
        return output_paths

    def plot_lambda_tradeoff(
        self,
        e6_results: list[ExperimentResult],
    ) -> Path:
        """Plot how gender accuracy and age MAE change with lambda_age (E6)."""
        import numpy as np

        completed = [
            r for r in e6_results
            if r.status == ExperimentStatus.COMPLETED
        ]
        if not completed:
            return self.plots_dir / "e6_lambda_tradeoff.png"

        # Extract lambda from changed_component field (e.g. "lambda_age=0.01 (referencia)")
        def _parse_lambda(result: ExperimentResult) -> float:
            for part in result.changed_component.replace("(referencia)", "").split():
                if part.startswith("lambda_age="):
                    try:
                        return float(part.split("=")[1])
                    except ValueError:
                        pass
            return float("nan")

        rows = sorted(
            [(r, _parse_lambda(r)) for r in completed],
            key=lambda x: x[1],
        )
        lambdas = [lam for _, lam in rows]
        accuracy = [float(r.metrics["gender_accuracy"]) for r, _ in rows]
        mae = [float(r.metrics["age_mae"]) for r, _ in rows]
        rmse = [float(r.metrics["age_rmse"]) for r, _ in rows]

        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(lambdas, accuracy, marker="o", color="steelblue")
        axes[0].set_xscale("log")
        axes[0].set_title("Accuracy de genero vs lambda")
        axes[0].set_xlabel("lambda_age (escala log)")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(alpha=0.3)

        axes[1].plot(lambdas, mae, marker="o", color="darkorange")
        axes[1].set_xscale("log")
        axes[1].set_title("MAE de edad vs lambda")
        axes[1].set_xlabel("lambda_age (escala log)")
        axes[1].set_ylabel("MAE (anios)")
        axes[1].grid(alpha=0.3)

        axes[2].plot(lambdas, rmse, marker="o", color="green")
        axes[2].set_xscale("log")
        axes[2].set_title("RMSE de edad vs lambda")
        axes[2].set_xlabel("lambda_age (escala log)")
        axes[2].set_ylabel("RMSE (anios)")
        axes[2].grid(alpha=0.3)

        figure.suptitle("E6: Trade-off de lambda (perdida combinada)", fontsize=12)
        figure.tight_layout()

        output_path = self.plots_dir / "e6_lambda_tradeoff.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def plot_final_summary(
        self,
        results: list[ExperimentResult],
    ) -> Path:
        """Bar chart comparing one base experiment per strategy."""
        BASE_NAMES = {
            "E1": "classical_base",
            "E2": "mlp_base",
            "E3": "cnn_base",
            "E4": "resnet_frozen_base",
            "E5": "resnet_finetuning_base",
        }
        rows = [
            r for r in results
            if r.experiment_name in BASE_NAMES.values()
            and r.status == ExperimentStatus.COMPLETED
        ]
        if not rows:
            return self.plots_dir / "final_summary.png"

        labels = [r.strategy_name for r in rows]
        accuracy = [float(r.metrics["gender_accuracy"]) for r in rows]
        f1 = [float(r.metrics["gender_f1"]) for r in rows]
        mae = [float(r.metrics["age_mae"]) for r in rows]
        rmse = [float(r.metrics["age_rmse"]) for r in rows]

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = range(len(labels))
        w = 0.35
        axes[0].bar([i - w / 2 for i in x], accuracy, w, label="Accuracy")
        axes[0].bar([i + w / 2 for i in x], f1, w, label="F1")
        axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
        axes[0].set_title("Metricas de genero (base experiments)")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar([i - w / 2 for i in x], mae, w, label="MAE")
        axes[1].bar([i + w / 2 for i in x], rmse, w, label="RMSE")
        axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
        axes[1].set_title("Metricas de edad (base experiments)")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        figure.suptitle("Comparacion final de estrategias", fontsize=12)
        figure.tight_layout()

        output_path = self.plots_dir / "final_summary.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _experiment_dir(self, experiment_name: str) -> Path:
        output_dir = self.plots_dir / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _bar_plot(
        names: list[str],
        series: list[list[float]],
        labels: list[str],
        title: str,
        output_path: Path,
    ) -> Path:
        import numpy as np

        x = np.arange(len(names))
        width = 0.8 / len(series)
        figure, axis = plt.subplots(figsize=(max(7, len(names) * 1.5), 5))
        for index, (values, label) in enumerate(zip(series, labels)):
            offset = (index - (len(series) - 1) / 2) * width
            axis.bar(x + offset, values, width=width, label=label)
        axis.set_xticks(x, labels=names, rotation=30, ha="right")
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path
