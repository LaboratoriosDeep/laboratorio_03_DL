"""Classical PCA baseline (E1): sklearn pipelines for gender and age."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.data.parser import UTKFaceFilenameParser

# Images are down-sampled to this size before PCA to keep memory manageable
# while still capturing the dominant facial structure.
_IMG_SIZE = 64


class ClassicalBaseline:
    """PCA + GaussianNB (gender) and PCA + RandomForestRegressor (age).

    Loss functions
    --------------
    Gender classification : CrossEntropy (equivalent, via GaussianNB log-likelihood).
    Age regression        : MSELoss (squared_error criterion inside RandomForest).

    The same 70/15/15 split used by the neural experiments is reused here
    to ensure a fair comparison. Images are converted to grayscale and
    resized to 64×64 before flattening.

    Parameters
    ----------
    n_components:
        Number of PCA components retained before the classifiers.
    n_estimators:
        Trees in the RandomForestRegressor.
    random_state:
        Seed for PCA whitening and RandomForest.
    """

    def __init__(
        self,
        n_components: int = 100,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.gender_pipeline: Pipeline | None = None
        self.age_pipeline: Pipeline | None = None
        self._parser = UTKFaceFilenameParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_paths: list[Path]) -> float:
        """Fit both pipelines on the training records. Returns training time."""
        X, y_gender, y_age = self._load_flat(train_paths)
        n_components = min(self.n_components, X.shape[0], X.shape[1])

        self.gender_pipeline = Pipeline([
            ("pca", PCA(n_components=n_components, whiten=True, random_state=self.random_state)),
            ("clf", GaussianNB()),
        ])
        self.age_pipeline = Pipeline([
            ("pca", PCA(n_components=n_components, whiten=True, random_state=self.random_state)),
            ("reg", RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion="squared_error",  # MSELoss equivalent
                random_state=self.random_state,
                n_jobs=-1,
            )),
        ])

        t0 = time.perf_counter()
        self.gender_pipeline.fit(X, y_gender)
        self.age_pipeline.fit(X, y_age)
        return time.perf_counter() - t0

    def predict(
        self,
        paths: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (gender_preds, age_preds, gender_targets, age_targets)."""
        if self.gender_pipeline is None or self.age_pipeline is None:
            raise RuntimeError("Llame a fit() antes de predict().")
        X, y_gender, y_age = self._load_flat(paths)
        gender_preds = self.gender_pipeline.predict(X)
        age_preds = self.age_pipeline.predict(X).astype(np.float32)
        return (
            gender_preds.astype(np.int64),
            age_preds,
            y_gender.astype(np.int64),
            y_age.astype(np.float32),
        )

    def predict_single(self, image: Image.Image) -> tuple[int, float]:
        """Run inference on one PIL image. Returns (gender_index, estimated_age)."""
        if self.gender_pipeline is None or self.age_pipeline is None:
            raise RuntimeError("Llame a fit() o from_checkpoint() antes de predict_single().")
        flat = _image_to_flat(image)
        X = flat.reshape(1, -1)
        gender = int(self.gender_pipeline.predict(X)[0])
        age = float(self.age_pipeline.predict(X)[0])
        return gender, age

    def save(self, checkpoint_dir: Path | str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _dump_pickle(self.gender_pipeline, checkpoint_dir / "gender_model.pkl")
        _dump_pickle(self.age_pipeline, checkpoint_dir / "age_model.pkl")
        meta: dict[str, Any] = {
            "n_components": self.n_components,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "img_size": _IMG_SIZE,
        }
        (checkpoint_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path | str) -> "ClassicalBaseline":
        checkpoint_dir = Path(checkpoint_dir)
        meta_path = checkpoint_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No se encontro el checkpoint clasico en {checkpoint_dir}. "
                "Ejecute classical_base antes de usar Streamlit."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instance = cls(
            n_components=meta["n_components"],
            n_estimators=meta["n_estimators"],
            random_state=meta["random_state"],
        )
        instance.gender_pipeline = _load_pickle(checkpoint_dir / "gender_model.pkl")
        instance.age_pipeline = _load_pickle(checkpoint_dir / "age_model.pkl")
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_flat(
        self,
        paths: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load images as flat grayscale arrays with their labels."""
        X_list, y_gender_list, y_age_list = [], [], []
        for path in paths:
            try:
                record = self._parser.parse(path)
                with Image.open(path) as img:
                    flat = _image_to_flat(img)
                X_list.append(flat)
                y_gender_list.append(record.gender)
                y_age_list.append(record.age)
            except Exception:
                continue
        if not X_list:
            raise RuntimeError("No se pudieron cargar imagenes validas para el baseline clasico.")
        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_gender_list, dtype=np.int64),
            np.array(y_age_list, dtype=np.float32),
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _image_to_flat(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalised grayscale 64×64 flat array."""
    gray = image.convert("L").resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32).flatten() / 255.0


def _dump_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_paths_from_manifest(
    manifest_path: Path,
    dataset_dir: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Read the split manifest saved by UTKFaceDataModule and return path lists."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    def to_paths(names: list[str]) -> list[Path]:
        return [dataset_dir / name for name in names]
    return (
        to_paths(manifest["train"]),
        to_paths(manifest["validation"]),
        to_paths(manifest["test"]),
    )
