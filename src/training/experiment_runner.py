"""Experiment catalog and orchestration for the laboratory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn, optim

from src.baselines.classical_todo import (
    ClassicalBaseline,
    load_paths_from_manifest,
)
from src.config import AppConfig
from src.data.datamodule import UTKFaceDataModule
from src.evaluation.metrics import MultiTaskEvaluator, MultiTaskMetrics
from src.evaluation.plots import ResultPlotter
from src.evaluation.reporter import ExperimentResult, ExperimentStatus
from src.models.cnn import MultiTaskCNN
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet
from src.training.losses import MultiTaskLoss
from src.training.trainer import MultiTaskTrainer
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Experiment specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentSpec:
    """Configuration for one base experiment or one single-change ablation."""

    strategy_id: str
    strategy_name: str
    name: str
    variant: str
    changed_component: str
    implemented: bool
    model_kind: str
    use_augmentation: bool = True
    dropout: float = 0.4
    lambda_age: float = 0.01
    learning_rate: float = 1e-3
    # ResNet-specific: number of layer groups unfrozen from the deepest end.
    # 0 = all frozen (E4); 1 = layer4 open (E5 base); 2 = layer4+3 (E5 more).
    unfreeze_last_n: int = 0
    # Classical baseline: PCA components and RF estimators.
    n_pca_components: int = 100
    n_estimators: int = 100


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def build_experiment_catalog(config: AppConfig) -> dict[str, ExperimentSpec]:
    """Return all strategies and ablation studies.

    E1 – classical PCA baseline (GaussianNB + RandomForest).
    E2 – MLP multitask.
    E3 – CNN simple (delivered example + ablations).
    E4 – ResNet18 with frozen backbone.
    E5 – ResNet18 with partial / full fine-tuning.
    E6 – Lambda ablation: CNN trained with 5 different lambda_age values.
    """

    low_lambda = config.lambda_age / 10     # 0.001
    high_lambda = config.lambda_age * 10    # 0.1
    lr = config.learning_rate               # 1e-3
    ft_lr = lr / 10                        # 1e-4  (fine-tuning)

    specs: list[ExperimentSpec] = [
        # ------------------------------------------------------------------ E1
        ExperimentSpec("E1", "Baseline clasico", "classical_base",
                       "base", "ninguno", True, "classical",
                       n_pca_components=100, n_estimators=100),
        ExperimentSpec("E1", "Baseline clasico", "classical_pca_50",
                       "ablacion", "PCA=50 componentes", True, "classical",
                       n_pca_components=50, n_estimators=100),
        ExperimentSpec("E1", "Baseline clasico", "classical_pca_200",
                       "ablacion", "PCA=200 componentes", True, "classical",
                       n_pca_components=200, n_estimators=100),
        # ------------------------------------------------------------------ E2
        ExperimentSpec("E2", "MLP multitarea", "mlp_base",
                       "base", "ninguno", True, "mlp",
                       dropout=0.3, lambda_age=config.lambda_age, learning_rate=lr),
        ExperimentSpec("E2", "MLP multitarea", "mlp_no_dropout",
                       "ablacion", "dropout=0.0", True, "mlp",
                       dropout=0.0, lambda_age=config.lambda_age, learning_rate=lr),
        ExperimentSpec("E2", "MLP multitarea", "mlp_lambda_low",
                       "ablacion", f"lambda_age={low_lambda:g}", True, "mlp",
                       dropout=0.3, lambda_age=low_lambda, learning_rate=lr),
        ExperimentSpec("E2", "MLP multitarea", "mlp_lambda_high",
                       "ablacion", f"lambda_age={high_lambda:g}", True, "mlp",
                       dropout=0.3, lambda_age=high_lambda, learning_rate=lr),
        # ------------------------------------------------------------------ E3
        ExperimentSpec("E3", "CNN simple multitarea", "cnn_base",
                       "base", "ninguno", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=config.lambda_age, learning_rate=lr),
        ExperimentSpec("E3", "CNN simple multitarea", "cnn_no_augmentation",
                       "ablacion", "sin aumentacion", True, "cnn",
                       use_augmentation=False, dropout=0.4,
                       lambda_age=config.lambda_age, learning_rate=lr),
        ExperimentSpec("E3", "CNN simple multitarea", "cnn_no_dropout",
                       "ablacion", "dropout=0.0", True, "cnn",
                       use_augmentation=True, dropout=0.0,
                       lambda_age=config.lambda_age, learning_rate=lr),
        ExperimentSpec("E3", "CNN simple multitarea", "cnn_lambda_low",
                       "ablacion", f"lambda_age={low_lambda:g}", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=low_lambda, learning_rate=lr),
        ExperimentSpec("E3", "CNN simple multitarea", "cnn_lambda_high",
                       "ablacion", f"lambda_age={high_lambda:g}", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=high_lambda, learning_rate=lr),
        # ------------------------------------------------------------------ E4
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_base",
                       "base", "ninguno", True, "resnet_frozen",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=config.lambda_age, learning_rate=lr,
                       unfreeze_last_n=0),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_no_augmentation",
                       "ablacion", "sin aumentacion", True, "resnet_frozen",
                       use_augmentation=False, dropout=0.4,
                       lambda_age=config.lambda_age, learning_rate=lr,
                       unfreeze_last_n=0),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_lambda_low",
                       "ablacion", f"lambda_age={low_lambda:g}", True, "resnet_frozen",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=low_lambda, learning_rate=lr,
                       unfreeze_last_n=0),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_lambda_high",
                       "ablacion", f"lambda_age={high_lambda:g}", True, "resnet_frozen",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=high_lambda, learning_rate=lr,
                       unfreeze_last_n=0),
        # ------------------------------------------------------------------ E5
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_base",
                       "base", "layer4 descongelada", True, "resnet_finetuning",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=config.lambda_age, learning_rate=ft_lr,
                       unfreeze_last_n=1),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_unfreeze_more",
                       "ablacion", "layer4+layer3 descongeladas", True, "resnet_finetuning",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=config.lambda_age, learning_rate=ft_lr,
                       unfreeze_last_n=2),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_lr_low",
                       "ablacion", "learning rate menor (1e-5)", True, "resnet_finetuning",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=config.lambda_age, learning_rate=ft_lr / 10,
                       unfreeze_last_n=1),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_lambda_high",
                       "ablacion", f"lambda_age={high_lambda:g}", True, "resnet_finetuning",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=high_lambda, learning_rate=ft_lr,
                       unfreeze_last_n=1),
        # ------------------------------------------------------------------ E6
        # Lambda ablation: CNN trained with 5 different lambda_age values.
        # Losses are saved per task so the trade-off can be visualised.
        ExperimentSpec("E6", "Ablacion lambda (CNN)", "cnn_lambda_e6_0001",
                       "ablacion", "lambda_age=0.001", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=0.001, learning_rate=lr),
        ExperimentSpec("E6", "Ablacion lambda (CNN)", "cnn_lambda_e6_001",
                       "base", "lambda_age=0.01 (referencia)", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=0.01, learning_rate=lr),
        ExperimentSpec("E6", "Ablacion lambda (CNN)", "cnn_lambda_e6_01",
                       "ablacion", "lambda_age=0.1", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=0.1, learning_rate=lr),
        ExperimentSpec("E6", "Ablacion lambda (CNN)", "cnn_lambda_e6_1",
                       "ablacion", "lambda_age=1.0", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=1.0, learning_rate=lr),
        ExperimentSpec("E6", "Ablacion lambda (CNN)", "cnn_lambda_e6_10",
                       "ablacion", "lambda_age=10.0", True, "cnn",
                       use_augmentation=True, dropout=0.4,
                       lambda_age=10.0, learning_rate=lr),
    ]
    return {spec.name: spec for spec in specs}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run selected experiments and preserve report rows for every strategy."""

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        catalog: dict[str, ExperimentSpec],
    ) -> None:
        self.config = config
        self.device = device
        self.catalog = catalog
        self.plotter = ResultPlotter(config.plots_dir)

    def run(self, selected_names: set[str]) -> list[ExperimentResult]:
        unknown = selected_names.difference(self.catalog)
        if unknown:
            raise ValueError(f"Experimentos desconocidos: {', '.join(sorted(unknown))}")

        results: list[ExperimentResult] = []
        for spec in self.catalog.values():
            if not spec.implemented:
                results.append(self._not_implemented_result(spec))
            elif spec.name not in selected_names:
                results.append(self._not_executed_result(spec))
            else:
                results.append(self._run_spec(spec))

        for strategy_id in ("E1", "E2", "E3", "E4", "E5", "E6"):
            self.plotter.plot_ablation_comparison(results, strategy_id)

        # Extra plot: lambda trade-off chart across E6 experiments.
        e6_completed = [
            r for r in results
            if r.strategy_id == "E6" and r.status == ExperimentStatus.COMPLETED
        ]
        if e6_completed:
            self.plotter.plot_lambda_tradeoff(e6_completed)

        self.plotter.plot_final_summary(results)
        return results

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _run_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        print(f"\nEjecutando {spec.name}: {spec.changed_component or spec.variant}")
        try:
            set_seed(self.config.seed)
            if spec.model_kind == "classical":
                return self._run_classical_spec(spec)
            return self._run_neural_spec(spec)
        except Exception as error:
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    # ------------------------------------------------------------------
    # Neural training path
    # ------------------------------------------------------------------

    def _run_neural_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        data_module = UTKFaceDataModule(self.config, use_augmentation=spec.use_augmentation)
        data_module.setup()

        model, model_kwargs = self._build_model(spec)
        model = model.to(self.device)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=spec.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = MultiTaskLoss(lambda_age=spec.lambda_age)
        checkpoint_path = self.config.checkpoints_dir / spec.name / "best_model.pt"
        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            device=self.device,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata={
                "experiment_name": spec.name,
                "strategy_id": spec.strategy_id,
                "model_name": spec.model_kind,
                "model_kwargs": model_kwargs,
                "image_size": self.config.image_size,
                "lambda_age": spec.lambda_age,
            },
        )
        history, training_seconds = trainer.fit(
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            epochs=self.config.epochs,
        )
        trainer.load_best_checkpoint()

        evaluator = MultiTaskEvaluator(self.device)
        evaluation = evaluator.evaluate(model, data_module.test_dataloader())
        self.plotter.plot_training_history(history, spec.name)
        self.plotter.plot_confusion_matrix(evaluation, spec.name)
        self.plotter.plot_age_predictions(evaluation, spec.name)

        sizes = data_module.split_sizes()
        metrics = dict(evaluation.metrics)
        metrics.update({
            "train_samples": sizes["train"],
            "validation_samples": sizes["validation"],
            "test_samples": sizes["test"],
        })
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            trainable_parameters=self._count_trainable_parameters(model),
            training_seconds=training_seconds,
            checkpoint=str(checkpoint_path),
            message="",
        )

    # ------------------------------------------------------------------
    # Classical baseline path
    # ------------------------------------------------------------------

    def _run_classical_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        # Run the datamodule to ensure the split manifest is written.
        data_module = UTKFaceDataModule(self.config, use_augmentation=False)
        data_module.setup()
        sizes = data_module.split_sizes()

        manifest_path = self.config.splits_dir / "utkface_split.json"
        train_paths, _val_paths, test_paths = load_paths_from_manifest(
            manifest_path, self.config.dataset_dir
        )

        baseline = ClassicalBaseline(
            n_components=spec.n_pca_components,
            n_estimators=spec.n_estimators,
            random_state=self.config.seed,
        )

        t0 = time.perf_counter()
        baseline.fit(train_paths)
        training_seconds = time.perf_counter() - t0

        checkpoint_dir = self.config.checkpoints_dir / spec.name
        baseline.save(checkpoint_dir)

        gender_preds, age_preds, gender_targets, age_targets = baseline.predict(test_paths)

        evaluation = MultiTaskMetrics.calculate(
            gender_targets=torch.tensor(gender_targets),
            gender_predictions=torch.tensor(gender_preds),
            age_targets=torch.tensor(age_targets),
            age_predictions=torch.tensor(age_preds),
        )
        self.plotter.plot_confusion_matrix(evaluation, spec.name)
        self.plotter.plot_age_predictions(evaluation, spec.name)

        metrics = dict(evaluation.metrics)
        metrics.update({
            "train_samples": sizes["train"],
            "validation_samples": sizes["validation"],
            "test_samples": sizes["test"],
        })
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            trainable_parameters=None,
            training_seconds=training_seconds,
            checkpoint=str(checkpoint_dir),
            message="",
        )

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(spec: ExperimentSpec) -> tuple[nn.Module, dict]:
        if spec.model_kind == "cnn":
            kwargs = {"dropout": spec.dropout}
            return MultiTaskCNN(**kwargs), kwargs

        if spec.model_kind == "mlp":
            kwargs = {"dropout": spec.dropout}
            return MultiTaskMLP(**kwargs), kwargs

        if spec.model_kind == "resnet_frozen":
            kwargs = {"freeze_backbone": True, "unfreeze_last_n": 0}
            return MultiTaskResNet(**kwargs), kwargs

        if spec.model_kind == "resnet_finetuning":
            kwargs = {
                "freeze_backbone": True,
                "unfreeze_last_n": spec.unfreeze_last_n,
            }
            return MultiTaskResNet(**kwargs), kwargs

        raise NotImplementedError(
            f"No existe una fabrica para model_kind='{spec.model_kind}'."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_trainable_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _not_implemented_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_IMPLEMENTED,
            message="El experimento debe ser completado por los alumnos.",
        )

    @staticmethod
    def _not_executed_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_EXECUTED,
            message="Implementado, pero no fue seleccionado en esta ejecucion.",
        )
