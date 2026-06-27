"""Multitask MLP strategy (E2): flatten image, shared FC trunk, two heads."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskMLP(BaseMultiTaskModel):
    """Flatten the RGB image and learn a shared representation with two heads.

    Architecture:
        input (3 × H × W)  →  Flatten
        → Linear(input_dim, 512)  →  ReLU  →  Dropout
        → Linear(512, 128)        →  ReLU
        → gender_head: Linear(128, 2)   (logits)
        → age_head:    Linear(128, 1)   (scalar)

    The same trunk feeds both tasks, allowing shared visual features while
    each head specialises independently.
    """

    def __init__(self, dropout: float = 0.3, image_size: int = 224) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en el intervalo [0, 1).")

        self.dropout = dropout
        self.image_size = image_size
        input_features = 3 * image_size * image_size

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.gender_head = nn.Linear(128, 2)
        self.age_head = nn.Linear(128, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
