"""Loader checks that do not need the 25,000-photo training run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline import (
    EXPECTED_ZIP_BYTES,
    EXPECTED_ZIP_SHA256,
    IMAGE_SIZE,
    ZIP_PATH,
    assign_splits,
    build_callbacks,
    build_model,
    compile_model,
    ensure_fit_generator,
    flow_directory,
    label_from_filename,
    set_seed,
)
from plots import COLORS, apply_style, contrast_ratio, resolved_font_name


def _write_class_tree(root: Path, per_class: int = 10, size=(36, 48)) -> None:
    colors = {"cat": (180, 40, 30), "dog": (30, 70, 160)}
    for label, color in colors.items():
        folder = root / label
        folder.mkdir(parents=True)
        for index in range(per_class):
            # Uneven source sizes: the generator must still emit 200×200.
            width = size[0] + index
            height = size[1] + index * 2
            Image.new("RGB", (width, height), color).save(folder / f"{index}.jpg", "JPEG")


def test_label_from_filename():
    assert label_from_filename("cat.10.jpg") == "cat"
    assert label_from_filename("dog.0.jpg") == "dog"
    with pytest.raises(ValueError):
        label_from_filename("bird.1.jpg")
    with pytest.raises(ValueError):
        label_from_filename("._cat.1.jpg")


def test_assign_splits_is_deterministic_and_exhaustive():
    first = assign_splits(12500, seed=42)
    second = assign_splits(12500, seed=42)
    assert first == second
    assert {len(first[name]) for name in first} == {10000, 1250, 1250}
    joined = first["train"] + first["val"] + first["test"]
    assert sorted(joined) == list(range(12500))
    assert assign_splits(12500, seed=7) != first


def test_flow_directory_batch_shape_and_seed(tmp_path: Path):
    set_seed(42)
    _write_class_tree(tmp_path / "train", per_class=8)
    first = flow_directory(tmp_path / "train", batch_size=4, seed=42, shuffle=True)
    second = flow_directory(tmp_path / "train", batch_size=4, seed=42, shuffle=True)
    batch_x, batch_y = next(first)
    other_x, other_y = next(second)
    assert batch_x.shape == (4, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    assert batch_y.shape == (4, 2)
    assert batch_x.dtype == np.float32
    assert batch_x.min() >= 0.0
    assert batch_x.max() <= 255.0
    assert first.class_indices == {"cat": 0, "dog": 1}
    assert np.allclose(batch_x, other_x)
    assert np.allclose(batch_y, other_y)
    # One-hot rows, automatic directory labels.
    assert set(np.argmax(batch_y, axis=1)).issubset({0, 1})
    assert np.allclose(batch_y.sum(axis=1), 1.0)


def test_fit_generator_callbacks_write_checkpoint(tmp_path: Path):
    set_seed(42)
    _write_class_tree(tmp_path / "train", per_class=4)
    _write_class_tree(tmp_path / "val", per_class=2)
    train = flow_directory(tmp_path / "train", batch_size=4, seed=42, shuffle=True)
    val = flow_directory(tmp_path / "val", batch_size=2, seed=42, shuffle=False)
    model = compile_model(
        __import__("tensorflow").keras.Sequential(
            [
                __import__("tensorflow").keras.layers.Input(shape=(200, 200, 3)),
                __import__("tensorflow").keras.layers.GlobalAveragePooling2D(),
                __import__("tensorflow").keras.layers.Dense(2, activation="softmax"),
            ]
        )
    )
    kind = ensure_fit_generator()
    assert kind in {"native", "wrapper-around-fit", "wrapper-already-installed"}
    checkpoint_path = tmp_path / "tiny.keras"
    csv_path = tmp_path / "log.csv"
    checkpoint, early_stopping, logger = build_callbacks(
        checkpoint_path, patience=3, csv_path=csv_path
    )
    history = model.fit_generator(
        train,
        steps_per_epoch=1,
        epochs=1,
        validation_data=val,
        validation_steps=1,
        callbacks=[checkpoint, early_stopping, logger],
        verbose=0,
        shuffle=False,
    )
    assert "loss" in history.history
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 0
    assert csv_path.is_file()


def test_efficientnet_head_and_input_shape():
    set_seed(42)
    model = build_model((200, 200, 3))
    class_names = [layer.__class__.__name__ for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]
    assert "efficientnetb0" in layer_names
    assert "Functional" in class_names
    assert "GlobalAveragePooling2D" in class_names
    assert model.layers[-1].units == 2
    assert model.layers[-1].activation.__name__ == "softmax"
    assert model.layers[-2].units == 128
    assert model.layers[-2].activation.__name__ == "relu"
    assert tuple(model.input_shape[1:]) == (200, 200, 3)
    assert tuple(model.output_shape[1:]) == (2,)


def test_plot_contrast_and_font_stack():
    font = apply_style()
    assert font in {"Arial", "Helvetica", "Arimo", "Liberation Sans", "DejaVu Sans"}
    assert contrast_ratio(COLORS["ink"], COLORS["canvas"]) >= 4.5
    assert contrast_ratio(COLORS["ink"], COLORS["surface"]) >= 4.5
    assert contrast_ratio(COLORS["muted"], COLORS["canvas"]) >= 4.5
    assert contrast_ratio(COLORS["on_action"], COLORS["action"]) >= 4.5
    # Graphical lines against the surface need at least 3:1.
    assert contrast_ratio(COLORS["info"], COLORS["surface"]) >= 3.0
    assert contrast_ratio(COLORS["action"], COLORS["surface"]) >= 3.0
    name = resolved_font_name()
    assert name


@pytest.mark.skipif(not ZIP_PATH.is_file(), reason="official zip is not on disk")
def test_official_zip_when_present():
    assert ZIP_PATH.stat().st_size == EXPECTED_ZIP_BYTES
    import hashlib

    digest = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_ZIP_SHA256


def test_notebook_calls_shared_loader_and_fit_generator():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "src" / "explore.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for token in (
        "make_generators",
        "trdata",
        "tsdata",
        "fit_generator",
        "ModelCheckpoint",
        "EarlyStopping",
        "build_model",
        "load_model",
        "SEED",
    ):
        assert token in source, token
