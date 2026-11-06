"""Dogs-vs-cats load path, split, and training helpers.

The step-by-step narrative lives in ``explore.ipynb``. This module is the
single implementation of the loader so the notebook and ``tests/`` share it.

Low-RAM path from the assignment: ``ImageDataGenerator`` and
``flow_from_directory``, images resized to 200×200. The sample architecture
block uses ``input_shape=(224, 224, 3)``. The prose requires 200×200, and
that is the size used here. ``weights=None`` as in the assignment.

``solution.ipynb`` and ``solution.es.ipynb`` were not downloaded or read.

Keras 3 removed ``Model.fit_generator``. ``ensure_fit_generator`` restores
that call as a wrapper around ``Model.fit``, which is the supported way to
train on a generator and to receive callbacks.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

SEED = 42
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 16
# Full passes over the training split. Updated only if a timed probe on
# this CPU shows that a documented smaller budget is required.
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2
MAX_STEPS_PER_EPOCH = None  # None means every batch of the training split.
MAX_EPOCHS_REASON = (
    "El enunciado no fija el número de épocas. No hay GPU en esta VM. "
    "Una sonda de 8 pasos (batch 16) el 2026-11-06 midió 19,97 s en el primer "
    "paso y 0,406 s de media en los siete siguientes. Una época son 1250 pasos "
    "de entrenamiento (unas 8,5 min) más 157 de validación. El cupo es 8 épocas "
    "completas, con EarlyStopping(patience=2, monitor=val_loss). No se recorta "
    "el conjunto de imágenes."
)
SUBSET_NOTE = (
    "No se usa un subconjunto: steps_per_epoch cubre todas las imágenes "
    "del split de entrenamiento y validation_steps cubre toda la validación."
)
LEARNING_RATE = 1e-4
TRAIN_PER_CLASS = 10000
VAL_PER_CLASS = 1250
TEST_PER_CLASS = 1250

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "dogs-vs-cats.zip"
EXTRACTED_TRAIN = RAW_DIR / "dogs-vs-cats" / "train"
PROCESSED = ROOT / "data" / "processed" / "dogs-vs-cats"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "dogs_vs_cats_best.keras"
REPORTS = ROOT / "reports"

DATASET_URL = "https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip"
EXPECTED_ZIP_BYTES = 581507449
EXPECTED_ZIP_SHA256 = (
    "8aabb6755161f0ec2a37f4fa9cff733c304824fe0a5cee652cf021e979d0937b"
)

TEMPLATE_REPO = "https://github.com/4GeeksAcademy/machine-learning-python-template"
TEMPLATE_COMMIT = "b49eff6673b76ba5b0003103e32f62a2eb9904a4"
STATEMENT_REPO = "4GeeksAcademy/image-classifier-project-tutorial"
STATEMENT_COMMIT = "aa69493bc3a3930e1656f089498ba70bb9786734"


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def label_from_filename(name: str) -> str:
    base = Path(name).name
    if base.startswith(".") or base.startswith("._"):
        raise ValueError(f"not a labeled photo: {name}")
    kind = Path(base).stem.split(".", 1)[0].lower()
    if kind not in {"cat", "dog"}:
        raise ValueError(f"filename is not cat.* or dog.*: {name}")
    return kind


def numeric_key(path: Path) -> int:
    return int(path.stem.split(".")[-1])


def verify_zip(path: Path = ZIP_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download it from {DATASET_URL} "
            "(see README). The zip is not stored in git."
        )
    size = path.stat().st_size
    digest = file_sha256(path)
    if size != EXPECTED_ZIP_BYTES or digest != EXPECTED_ZIP_SHA256:
        raise ValueError(
            f"Unexpected dataset file: bytes={size} sha256={digest}. "
            f"Expected bytes={EXPECTED_ZIP_BYTES} sha256={EXPECTED_ZIP_SHA256}."
        )
    return {"path": str(path), "bytes": size, "sha256": digest, "url": DATASET_URL}


def extract_dataset(zip_path: Path = ZIP_PATH, dest_train: Path = EXTRACTED_TRAIN) -> dict:
    verify_zip(zip_path)
    dest_train.mkdir(parents=True, exist_ok=True)
    cats = [p for p in dest_train.glob("cat.*.jpg") if not p.name.startswith(".")]
    dogs = [p for p in dest_train.glob("dog.*.jpg") if not p.name.startswith(".")]
    if len(cats) == 12500 and len(dogs) == 12500:
        return {"train_dir": str(dest_train), "cats": 12500, "dogs": 12500, "extracted_now": False}

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename
            if not name.startswith("dogs-vs-cats/train/"):
                continue
            base = Path(name).name
            if not base.lower().endswith(".jpg") or base.startswith("."):
                continue
            target = dest_train / base
            if target.exists() and target.stat().st_size > 0:
                continue
            with archive.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    cats = list(dest_train.glob("cat.*.jpg"))
    dogs = list(dest_train.glob("dog.*.jpg"))
    if len(cats) != 12500 or len(dogs) != 12500:
        raise RuntimeError(f"Expected 12500 cats and 12500 dogs, got {len(cats)} and {len(dogs)}.")
    return {"train_dir": str(dest_train), "cats": len(cats), "dogs": len(dogs), "extracted_now": True}


def assign_splits(n_items: int, seed: int = SEED) -> dict:
    """Split one class into train/val/test index lists.

    10000 / 1250 / 1250 for a class of 12500. The same rule (80/10/10,
    integer counts) is applied for any other count so tests can use a
    smaller folder: test = n // 10, val = n // 10, train = the rest.
    """
    if n_items < 10:
        raise ValueError("need at least 10 images per class to split 80/10/10")
    n_test = n_items // 10
    n_val = n_items // 10
    n_train = n_items - n_val - n_test
    order = np.random.RandomState(seed).permutation(n_items)
    return {
        "train": order[:n_train].tolist(),
        "val": order[n_train : n_train + n_val].tolist(),
        "test": order[n_train + n_val :].tolist(),
    }


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        os.symlink(os.path.relpath(src, dest.parent), dest)


def materialize_split(
    src_train: Path,
    dest_root: Path,
    seed: int = SEED,
) -> dict:
    cats = sorted(
        [p for p in src_train.glob("cat.*.jpg") if not p.name.startswith(".")],
        key=numeric_key,
    )
    dogs = sorted(
        [p for p in src_train.glob("dog.*.jpg") if not p.name.startswith(".")],
        key=numeric_key,
    )
    if not cats or not dogs:
        raise RuntimeError(f"No labeled jpg files in {src_train}")

    # Independent streams so the dog permutation does not depend on n_cats.
    cat_splits = assign_splits(len(cats), seed)
    dog_splits = assign_splits(len(dogs), seed + 1)
    manifest = {
        "seed": seed,
        "dog_seed": seed + 1,
        "method": "hardlink-or-symlink",
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "counts": {
            "cat": {name: len(idxs) for name, idxs in cat_splits.items()},
            "dog": {name: len(idxs) for name, idxs in dog_splits.items()},
        },
        "src_train": str(src_train),
    }
    manifest_path = dest_root / "split_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        previous.pop("src_train", None)
        current = dict(manifest)
        current.pop("src_train", None)
        if previous == current and (dest_root / "train" / "cat").is_dir():
            manifest["reused"] = True
            return manifest

    if dest_root.exists():
        shutil.rmtree(dest_root)
    groups = {"cat": (cats, cat_splits), "dog": (dogs, dog_splits)}
    for label, (files, splits) in groups.items():
        for bucket, indexes in splits.items():
            for index in indexes:
                src = files[index]
                _link(src, dest_root / bucket / label / src.name)
    dest_root.mkdir(parents=True, exist_ok=True)
    manifest["reused"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def survey_sizes(src_train: Path, limit: int | None = None) -> dict:
    from PIL import Image

    paths = sorted(
        [p for p in src_train.glob("*.jpg") if not p.name.startswith(".")],
        key=lambda p: (label_from_filename(p.name), numeric_key(p)),
    )
    if limit is not None:
        paths = paths[:limit]
    widths = []
    heights = []
    modes: dict[str, int] = {}
    unique = set()
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        widths.append(width)
        heights.append(height)
        modes[mode] = modes.get(mode, 0) + 1
        unique.add((width, height))
    arr_w = np.asarray(widths)
    arr_h = np.asarray(heights)
    return {
        "n": len(paths),
        "modes": modes,
        "unique_sizes": len(unique),
        "width": {"min": int(arr_w.min()), "max": int(arr_w.max()), "median": float(np.median(arr_w))},
        "height": {"min": int(arr_h.min()), "max": int(arr_h.max()), "median": float(np.median(arr_h))},
        "not_200x200": int(np.sum((arr_w != 200) | (arr_h != 200))),
    }


def first_n_images(src_train: Path, label: str, n: int = 9) -> list[Path]:
    files = sorted(
        [p for p in src_train.glob(f"{label}.*.jpg") if not p.name.startswith(".")],
        key=numeric_key,
    )
    if len(files) < n:
        raise RuntimeError(f"Need {n} {label} images, found {len(files)}")
    return files[:n]


def prepare_all(run_survey: bool = True) -> dict:
    zip_info = verify_zip()
    extracted = extract_dataset()
    split = materialize_split(EXTRACTED_TRAIN, PROCESSED, SEED)
    summary = {
        "zip": zip_info,
        "extracted": extracted,
        "split": split,
    }
    if run_survey:
        cache = REPORTS / "image_size_survey.json"
        if cache.is_file():
            summary["size_survey"] = json.loads(cache.read_text())
        else:
            REPORTS.mkdir(parents=True, exist_ok=True)
            summary["size_survey"] = survey_sizes(EXTRACTED_TRAIN)
            cache.write_text(json.dumps(summary["size_survey"], indent=2))
    REPORTS.mkdir(parents=True, exist_ok=True)
    public = {
        "zip_bytes": zip_info["bytes"],
        "zip_sha256": zip_info["sha256"],
        "zip_url": DATASET_URL,
        "cats": extracted["cats"],
        "dogs": extracted["dogs"],
        "split_counts": split["counts"],
        "seed": SEED,
        "size_survey": summary.get("size_survey"),
    }
    (REPORTS / "split_summary.json").write_text(json.dumps(public, indent=2))
    return summary


def flow_directory(directory: Path, image_size=IMAGE_SIZE, batch_size: int = BATCH_SIZE, seed: int = SEED, shuffle: bool = False):
    """Directory iterator. Pixels stay in 0–255.

    Keras 3 EfficientNetB0 rescales by 1/255 inside the network. A second
    rescale here would shrink the inputs again.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator()
    return datagen.flow_from_directory(
        str(directory),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=["cat", "dog"],
        color_mode="rgb",
        shuffle=shuffle,
        seed=seed,
        interpolation="bilinear",
    )


def make_generators(processed_root: Path = PROCESSED, image_size=IMAGE_SIZE, batch_size: int = BATCH_SIZE, seed: int = SEED):
    trdata = flow_directory(processed_root / "train", image_size, batch_size, seed, shuffle=True)
    tsdata = flow_directory(processed_root / "test", image_size, batch_size, seed, shuffle=False)
    valdata = flow_directory(processed_root / "val", image_size, batch_size, seed, shuffle=False)
    return trdata, tsdata, valdata


def build_model(input_shape=(200, 200, 3)):
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Sequential

    model = Sequential(name="dogs_vs_cats")
    model.add(
        EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    return model


def compile_model(model, learning_rate: float = LEARNING_RATE):
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


def ensure_fit_generator() -> str:
    """Return which implementation ``fit_generator`` will call."""
    import inspect

    import keras

    if getattr(keras.Model, "_dogs_cats_fit_generator", False):
        return "wrapper-already-installed"

    original = getattr(keras.Model, "fit_generator", None)

    def fit_generator(
        self,
        generator,
        steps_per_epoch=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        class_weight=None,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0,
    ):
        # Keras 3 Model.fit has no workers/use_multiprocessing arguments.
        # They are accepted so the assignment's fit_generator call still
        # type-checks, and dropped when the installed fit() does not take them.
        candidate = {
            "x": generator,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "verbose": verbose,
            "callbacks": callbacks,
            "validation_data": validation_data,
            "validation_steps": validation_steps,
            "class_weight": class_weight,
            "shuffle": shuffle,
            "initial_epoch": initial_epoch,
            "workers": workers,
            "use_multiprocessing": use_multiprocessing,
        }
        accepted = inspect.signature(self.fit).parameters
        kwargs = {key: value for key, value in candidate.items() if key in accepted}
        if "x" not in accepted:
            raise TypeError("Model.fit does not accept x")
        return self.fit(**kwargs)

    # Prefer a real historical method when the installed Keras still has one.
    if original is not None and not getattr(original, "_dogs_cats_wrapper", False):
        return "native"
    fit_generator._dogs_cats_wrapper = True
    keras.Model.fit_generator = fit_generator
    keras.Model._dogs_cats_fit_generator = True
    return "wrapper-around-fit"


def build_callbacks(model_path: Path = MODEL_PATH, patience: int = EARLY_STOPPING_PATIENCE, csv_path: Path | None = None):
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )
    logger = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger = CSVLogger(str(csv_path), append=False)
    return checkpoint, early_stopping, logger


class EpochTimer:
    """Keras-compatible callback implemented locally to record epoch wall time."""

    def __init__(self):
        import tensorflow as tf

        class _Timer(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.epoch_seconds: list[float] = []
                self._started = 0.0

            def on_epoch_begin(self, epoch, logs=None):
                self._started = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_seconds.append(round(time.perf_counter() - self._started, 3))

        self.callback = _Timer()


def steps_for(generator) -> int:
    if MAX_STEPS_PER_EPOCH is None:
        return len(generator)
    return min(len(generator), int(MAX_STEPS_PER_EPOCH))


def hardware_note() -> dict:
    mem_total_kb = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
                break
    except OSError:
        mem_total_kb = None
    return {
        "gpu": False,
        "cpu_count": os.cpu_count(),
        "mem_total_kib": mem_total_kb,
        "float32_tensor_gib_25000x200x200x3": round((25000 * 200 * 200 * 3 * 4) / (1024**3), 3),
    }
