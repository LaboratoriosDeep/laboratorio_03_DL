"""A transparent PyTorch training loop for educational use."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import MultiTaskLoss


class MultiTaskTrainer:
    """Train, validate and checkpoint a multitask PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: MultiTaskLoss,
        device: torch.device,
        checkpoint_path: Path,
        checkpoint_metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_metadata = checkpoint_metadata

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> tuple[list[dict[str, float]], float]:
        """Run all epochs and save the checkpoint with the lowest validation loss."""

        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        start_time = time.perf_counter()

        epoch_bar = tqdm(range(1, epochs + 1), desc="Entrenamiento", unit="epoch")
        for epoch in epoch_bar:
            epoch_start = time.perf_counter()
            train_losses = self._run_epoch(train_loader, training=True, desc="  Train")
            val_losses = self._run_epoch(val_loader, training=False, desc="  Val  ")
            epoch_seconds = time.perf_counter() - epoch_start

            row = {
                "epoch": float(epoch),
                "train_total_loss": train_losses["total_loss"],
                "train_gender_loss": train_losses["gender_loss"],
                "train_age_loss": train_losses["age_loss"],
                "val_total_loss": val_losses["total_loss"],
                "val_gender_loss": val_losses["gender_loss"],
                "val_age_loss": val_losses["age_loss"],
                "epoch_seconds": epoch_seconds,
            }
            history.append(row)

            epoch_bar.set_postfix(
                train=f"{row['train_total_loss']:.4f}",
                val=f"{row['val_total_loss']:.4f}",
            )

            if row["val_total_loss"] < best_val_loss:
                best_val_loss = row["val_total_loss"]
                self._save_checkpoint(epoch=epoch, val_loss=best_val_loss)

        training_seconds = time.perf_counter() - start_time
        return history, training_seconds

    def load_best_checkpoint(self) -> dict[str, Any]:
        """Restore the model selected using validation loss."""

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        desc: str = "",
    ) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        totals = {"total_loss": 0.0, "gender_loss": 0.0, "age_loss": 0.0}
        sample_count = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            batch_bar = tqdm(loader, desc=desc, leave=False, unit="batch")
            for images, gender_targets, age_targets in batch_bar:
                images = images.to(self.device)
                gender_targets = gender_targets.to(self.device)
                age_targets = age_targets.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                gender_logits, age_predictions = self.model(images)
                losses = self.loss_function(
                    gender_logits,
                    age_predictions,
                    gender_targets,
                    age_targets,
                )

                if training:
                    losses.total.backward()
                    self.optimizer.step()

                batch_size = images.size(0)
                sample_count += batch_size
                totals["total_loss"] += losses.total.item() * batch_size
                totals["gender_loss"] += losses.gender.item() * batch_size
                totals["age_loss"] += losses.age.item() * batch_size

                batch_bar.set_postfix(loss=f"{losses.total.item():.4f}")

        if sample_count == 0:
            raise RuntimeError("El DataLoader no contiene muestras.")
        return {name: value / sample_count for name, value in totals.items()}

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            **self.checkpoint_metadata,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
