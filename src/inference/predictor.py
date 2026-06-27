"""Load any trained model checkpoint and run PyTorch or classical inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.baselines.classical_todo import ClassicalBaseline
from src.data.transforms import TransformFactory
from src.models.cnn import MultiTaskCNN
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet


@dataclass(frozen=True)
class Prediction:
    """Human-readable model output for one detected face."""

    gender_index: int
    gender_label: str
    gender_confidence: float
    estimated_age: float


GENDER_LABELS: dict[int, str] = {0: "Masculino", 1: "Femenino"}

# Map model_name strings stored in checkpoints to the model class.
_MODEL_REGISTRY: dict[str, type] = {
    "cnn": MultiTaskCNN,
    "mlp": MultiTaskMLP,
    "resnet_frozen": MultiTaskResNet,
    "resnet_finetuning": MultiTaskResNet,
}


# ---------------------------------------------------------------------------
# Neural predictor (works with any checkpoint saved by MultiTaskTrainer)
# ---------------------------------------------------------------------------

class NeuralPredictor:
    """Apply the same deterministic preprocessing used during testing."""

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.transform = TransformFactory.inference(image_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> "NeuralPredictor":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint {path}. "
                "Entrene el modelo antes de usar Streamlit."
            )
        checkpoint: dict[str, Any] = torch.load(
            path, map_location=device, weights_only=True
        )
        model_name: str = checkpoint.get("model_name", "")
        model_class = _MODEL_REGISTRY.get(model_name)
        if model_class is None:
            raise ValueError(
                f"model_name='{model_name}' no reconocido. "
                f"Valores validos: {list(_MODEL_REGISTRY)}."
            )
        model_kwargs: dict[str, Any] = checkpoint.get("model_kwargs", {})
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        image_size = int(checkpoint.get("image_size", 224))
        return cls(model=model, image_size=image_size, device=device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        gender_logits, age_pred = self.model(tensor)
        probs = torch.softmax(gender_logits, dim=1)
        confidence, gender_idx = probs.max(dim=1)
        idx = int(gender_idx.item())
        return Prediction(
            gender_index=idx,
            gender_label=GENDER_LABELS.get(idx, str(idx)),
            gender_confidence=float(confidence.item()),
            estimated_age=float(age_pred.item()),
        )


# ---------------------------------------------------------------------------
# Classical predictor (sklearn pipelines saved as .pkl)
# ---------------------------------------------------------------------------

class ClassicalPredictor:
    """Run inference using the PCA + GaussianNB / RandomForest pipeline."""

    def __init__(self, baseline: ClassicalBaseline) -> None:
        self.baseline = baseline

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> "ClassicalPredictor":
        return cls(ClassicalBaseline.from_checkpoint(Path(checkpoint_dir)))

    def predict(self, image: Image.Image) -> Prediction:
        gender_idx, estimated_age = self.baseline.predict_single(image)
        return Prediction(
            gender_index=gender_idx,
            gender_label=GENDER_LABELS.get(gender_idx, str(gender_idx)),
            gender_confidence=float("nan"),  # GaussianNB probability not surfaced
            estimated_age=estimated_age,
        )


# ---------------------------------------------------------------------------
# Convenience alias kept for backward compatibility with the original app
# ---------------------------------------------------------------------------

class CNNPredictor(NeuralPredictor):
    """Backward-compatible alias targeting only CNN checkpoints."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> "CNNPredictor":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint {path}. "
                "Entrene cnn_base antes de usar Streamlit."
            )
        checkpoint: dict[str, Any] = torch.load(
            path, map_location=device, weights_only=True
        )
        if checkpoint.get("model_name") != "cnn":
            raise ValueError("El checkpoint no corresponde a la CNN entregada.")
        model_kwargs = checkpoint.get("model_kwargs", {})
        model = MultiTaskCNN(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        image_size = int(checkpoint.get("image_size", 224))
        return cls(model=model, image_size=image_size, device=device)
