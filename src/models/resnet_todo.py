"""ResNet18 transfer-learning strategies (E4 frozen, E5 fine-tuning)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.models.base import BaseMultiTaskModel


class MultiTaskResNet(BaseMultiTaskModel):
    """Replace ResNet18's classifier with gender and age heads.

    Modes
    -----
    freeze_backbone=True, unfreeze_last_n=0  (E4 base)
        All convolutional weights are frozen; only the two linear heads are
        trained. Fast and suitable when the dataset is small.

    freeze_backbone=True, unfreeze_last_n=1  (E5 base – partial fine-tuning)
        The last residual block (layer4) is unfrozen in addition to the heads.
        Gives the model a chance to adapt high-level features to the task.

    freeze_backbone=True, unfreeze_last_n=2  (E5 ablation – more unfrozen)
        Unfreeze layer4 + layer3 for deeper adaptation.

    freeze_backbone=False  (full fine-tuning)
        All parameters receive gradients. Recommended with a lower learning
        rate to avoid destroying the pre-trained representations.

    Parameters
    ----------
    freeze_backbone:
        When True the backbone starts fully frozen; ``unfreeze_last_n``
        then controls how many residual blocks are selectively unfrozen.
        When False the entire network is trainable from the start.
    unfreeze_last_n:
        Number of ResNet layer groups (layer4, layer3, …) to unfreeze
        counting from the deepest. Only used when ``freeze_backbone=True``.
    """

    _LAYER_GROUPS = ["layer4", "layer3", "layer2", "layer1"]

    def __init__(
        self,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()
        if unfreeze_last_n < 0:
            raise ValueError("unfreeze_last_n no puede ser negativo.")

        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n = unfreeze_last_n

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base

        if not freeze_backbone:
            # Full fine-tuning: all backbone parameters get gradients.
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
        else:
            # Start with everything frozen.
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            # Selectively re-enable gradient flow for the last N layer groups.
            groups_to_unfreeze = self._LAYER_GROUPS[:unfreeze_last_n]
            for group_name in groups_to_unfreeze:
                layer = getattr(self.backbone, group_name)
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.backbone(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
