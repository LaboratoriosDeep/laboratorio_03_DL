"""Streamlit UI: compare five model strategies on any uploaded face image."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import AppConfig
from src.inference.face_detector import FaceDetector
from src.inference.predictor import (
    ClassicalPredictor,
    NeuralPredictor,
    Prediction,
)
from src.utils import resolve_device


# ---------------------------------------------------------------------------
# Configuration: which checkpoint goes with each strategy button.
# ---------------------------------------------------------------------------

_STRATEGIES: list[dict[str, Any]] = [
    {
        "label": "E1 – Baseline Clasico",
        "kind": "classical",
        "checkpoint": "classical_base",
        "description": "PCA (100 comp.) + GaussianNB (genero) + RandomForest (edad).",
    },
    {
        "label": "E2 – MLP multitarea",
        "kind": "neural",
        "checkpoint": "mlp_base/best_model.pt",
        "description": "Red neuronal densa, imagen aplanada, cabezales de genero y edad.",
    },
    {
        "label": "E3 – CNN simple",
        "kind": "neural",
        "checkpoint": "cnn_base/best_model.pt",
        "description": "CNN con tres bloques conv, entrenada desde cero.",
    },
    {
        "label": "E4 – ResNet18 congelada",
        "kind": "neural",
        "checkpoint": "resnet_frozen_base/best_model.pt",
        "description": "ResNet18 preentrenada en ImageNet; solo se entrenan los cabezales.",
    },
    {
        "label": "E5 – ResNet18 fine-tuning",
        "kind": "neural",
        "checkpoint": "resnet_finetuning_base/best_model.pt",
        "description": "ResNet18 con layer4 descongelada; ajuste fino de representaciones.",
    },
]


# ---------------------------------------------------------------------------
# Helper: load predictor (cached by Streamlit)
# ---------------------------------------------------------------------------

def _load_neural(checkpoint_path: Path, device) -> NeuralPredictor | None:
    """Return a NeuralPredictor or None if the checkpoint is missing."""
    import streamlit as st  # local import to avoid side-effects at module level

    if not checkpoint_path.exists():
        st.warning(
            f"Checkpoint no encontrado: `{checkpoint_path}`.\n"
            "Ejecute el experimento correspondiente con `python main.py --experiment <nombre>`."
        )
        return None
    try:
        return NeuralPredictor.from_checkpoint(checkpoint_path, device)
    except Exception as exc:
        st.error(f"Error al cargar el modelo: {exc}")
        return None


def _load_classical(checkpoint_dir: Path) -> ClassicalPredictor | None:
    """Return a ClassicalPredictor or None if the checkpoint is missing."""
    import streamlit as st

    if not (checkpoint_dir / "meta.json").exists():
        st.warning(
            f"Checkpoint clasico no encontrado en `{checkpoint_dir}`.\n"
            "Ejecute: `python main.py --experiment classical_base`."
        )
        return None
    try:
        return ClassicalPredictor.from_checkpoint(checkpoint_dir)
    except Exception as exc:
        st.error(f"Error al cargar el modelo clasico: {exc}")
        return None


# ---------------------------------------------------------------------------
# Result display helper
# ---------------------------------------------------------------------------

def _show_prediction(prediction: Prediction) -> None:
    import streamlit as st

    col_gender, col_age, col_conf = st.columns(3)
    col_gender.metric("Genero predicho", prediction.gender_label)
    col_age.metric("Edad estimada", f"{prediction.estimated_age:.1f} anos")
    if not math.isnan(prediction.gender_confidence):
        col_conf.metric("Confianza genero", f"{prediction.gender_confidence * 100:.1f}%")
    else:
        col_conf.metric("Confianza genero", "N/A (clasico)")


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------

def run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="UTKFace: genero y edad", layout="wide")
    st.title("Clasificacion de genero y regresion de edad – UTKFace")
    st.write(
        "Carga o captura una imagen facial y aplica cualquiera de las cinco "
        "estrategias entrenadas. La cara mas grande detectada se recorta y "
        "preprocesa igual que durante el entrenamiento."
    )

    config = AppConfig.from_env()
    device = resolve_device(config.device)
    checkpoints_dir = config.checkpoints_dir

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("Configuracion")
    st.sidebar.write(f"**Device:** `{device}`")
    st.sidebar.write(f"**Checkpoints:** `{checkpoints_dir}`")

    # ------------------------------------------------------------------
    # Image input
    # ------------------------------------------------------------------
    st.header("1. Cargar imagen")
    col_up, col_cam = st.columns(2)
    with col_up:
        uploaded = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])
    with col_cam:
        captured = st.camera_input("O capturar con camara")

    source = captured if captured is not None else uploaded
    if source is None:
        st.info("Sube o captura una imagen para comenzar.")
        return

    image = Image.open(source).convert("RGB")
    detector = FaceDetector()
    detection = detector.detect_largest(image)

    st.header("2. Deteccion facial")
    col_orig, col_crop = st.columns(2)
    with col_orig:
        if detection is None:
            st.image(image, caption="Imagen original (sin rostro detectado)",
                     use_container_width=True)
        else:
            st.image(
                detector.draw_box(image, detection),
                caption="Rostro seleccionado (rectangulo rojo)",
                use_container_width=True,
            )
    with col_crop:
        if detection is not None:
            st.image(detection.crop, caption="Recorte facial",
                     use_container_width=True)

    if detection is None:
        st.warning(
            "No se detecto un rostro frontal. Prueba una imagen con mejor "
            "iluminacion y cara mas visible."
        )
        return

    face_crop = detection.crop

    # ------------------------------------------------------------------
    # Strategy buttons
    # ------------------------------------------------------------------
    st.header("3. Seleccionar estrategia")
    st.write(
        "Haz clic en uno de los cinco botones para correr la inferencia con "
        "la estrategia correspondiente."
    )

    for strategy in _STRATEGIES:
        with st.expander(strategy["label"], expanded=False):
            st.caption(strategy["description"])
            if st.button(f"Ejecutar {strategy['label']}", key=strategy["label"]):
                with st.spinner("Ejecutando modelo..."):
                    if strategy["kind"] == "classical":
                        checkpoint_dir = checkpoints_dir / strategy["checkpoint"]
                        predictor = _load_classical(checkpoint_dir)
                    else:
                        checkpoint_path = checkpoints_dir / strategy["checkpoint"]
                        predictor = _load_neural(checkpoint_path, device)

                    if predictor is not None:
                        try:
                            prediction = predictor.predict(face_crop)
                            st.subheader("Resultado")
                            _show_prediction(prediction)
                        except Exception as exc:
                            st.error(f"Error durante la inferencia: {exc}")

    st.caption(
        "Las predicciones reflejan las etiquetas binarias y los sesgos del "
        "dataset UTKFace. No deben interpretarse como identidad de genero."
    )
