"""PyTorch models used in the laboratory."""

from src.models.cnn import MultiTaskCNN
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet

__all__ = ["MultiTaskCNN", "MultiTaskMLP", "MultiTaskResNet"]
