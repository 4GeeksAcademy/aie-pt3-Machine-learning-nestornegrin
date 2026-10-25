"""
Clasificador perros vs gatos: uso en producción del modelo entrenado.

Carga el modelo guardado en models/ y clasifica una o varias imágenes.

Uso (desde la raíz del repositorio):
    python src/app.py foto1.jpg foto2.jpg     # clasifica las imágenes indicadas
    python src/app.py                         # demo con 6 fotos aleatorias del conjunto de test
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # oculta mensajes informativos de TensorFlow

import numpy as np  # noqa: E402
from tf_keras.models import load_model  # noqa: E402
from tf_keras.preprocessing.image import img_to_array, load_img  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "dogs_vs_cats_efficientnetb0.keras"
TEST_DIR = PROJECT_ROOT / "data" / "interim" / "split" / "test"

IMG_SIZE = (200, 200)
# Mismo orden que flow_from_directory (alfabético): cats -> 0, dogs -> 1
CLASS_NAMES = ("gato", "perro")


def load_image(path: Path) -> np.ndarray:
    """
    Prepara una imagen igual que en el entrenamiento: RGB, 200x200, píxeles en [0, 255].
    Sin rescale: EfficientNet normaliza internamente.
    """
    img = load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    return img_to_array(img)


def predict(model, paths: list[Path]) -> list[tuple[Path, str, float]]:
    """Clasifica todas las imágenes en un único lote con el modelo ya cargado."""
    batch = np.stack([load_image(p) for p in paths])
    probabilities = model.predict(batch, verbose=0)
    return [
        (path, CLASS_NAMES[int(probs.argmax())], float(probs.max()))
        for path, probs in zip(paths, probabilities)
    ]


def demo_paths(n: int = 6) -> list[Path]:
    """Elige n fotos aleatorias del conjunto de test (requiere haber ejecutado el notebook)."""
    candidates = sorted(TEST_DIR.rglob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(
            f"No hay imágenes en {TEST_DIR}. Ejecuta antes src/explore.ipynb o pasa rutas de imágenes."
        )
    return random.Random(42).sample(candidates, min(n, len(candidates)))


def main(args: list[str]) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el modelo en {MODEL_PATH}. Ejecuta antes src/explore.ipynb.")

    paths = [Path(a) for a in args] if args else demo_paths()
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No existen: {', '.join(map(str, missing))}")

    model = load_model(MODEL_PATH)  # se carga una sola vez
    print(f"Modelo: {MODEL_PATH.relative_to(PROJECT_ROOT)}\n")

    for path, label, confidence in predict(model, paths):
        real = path.name.split(".")[0]
        real_note = f" (real: {'gato' if real == 'cat' else 'perro'})" if real in {"cat", "dog"} else ""
        print(f"{path.name:<16} -> {label:<6} {confidence:6.1%}{real_note}")


if __name__ == "__main__":
    main(sys.argv[1:])
