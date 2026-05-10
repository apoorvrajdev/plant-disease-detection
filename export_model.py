"""
export_model.py — One-shot export of the trained model to a portable .keras
file plus class_names.json, ready for Hugging Face Spaces deployment.

Run locally on a machine that has the trained checkpoints from the notebook:

    python export_model.py

Outputs at repo root (both gitignored — uploaded to a GitHub Release later):
    plant_disease_model.keras
    class_names.json

Also written for the Space to bundle:
    app/class_names.json     (committed — the canonical 38-class order)
    examples/<5 sample images copied from valid/>

Environment overrides (defaults match the in-repo layout from CLAUDE.md):
    TRAIN_DIR        train/ folder containing 38 class subdirs
    VALID_DIR        valid/ folder (used to source example images)
    CHECKPOINT_DIR   TF checkpoint directory written by ModelCheckpoint
    SOURCE_KERAS     (optional) if set, copy this pre-saved .keras file
                     directly instead of rebuilding + loading from checkpoint.
                     Use when the original training run produced a single
                     .keras / SavedModel artifact and the raw checkpoint
                     shards aren't available (or aren't loadable).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent

TRAIN_DIR = Path(os.environ.get(
    "TRAIN_DIR",
    ROOT / "New Plant Diseases Dataset" / "New Plant Diseases Dataset(Augmented)" / "train",
))
VALID_DIR = Path(os.environ.get(
    "VALID_DIR",
    ROOT / "New Plant Diseases Dataset" / "New Plant Diseases Dataset(Augmented)" / "valid",
))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", ROOT / "fine_tune_checkpoints"))
SOURCE_KERAS = os.environ.get("SOURCE_KERAS")  # optional fast path

KERAS_OUT = ROOT / "plant_disease_model.keras"
CLASS_NAMES_OUT_ROOT = ROOT / "class_names.json"
CLASS_NAMES_OUT_APP = ROOT / "app" / "class_names.json"
EXAMPLES_DIR = ROOT / "examples"

EXAMPLE_CLASSES = [
    "Tomato___Early_blight",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___Black_rot",
    "Potato___Late_blight",
]


def build_model() -> tf.keras.Model:
    """Mirror the notebook's architecture exactly so load_weights matches."""
    image_shape = (224, 224, 3)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = False

    inputs = layers.Input(shape=image_shape, name="input_layer")
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D_layer")(x)
    outputs = layers.Dense(38, activation="softmax", name="output_layer")(x)
    model = tf.keras.Model(inputs, outputs, name="plant_disease_model")

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def discover_class_names(train_dir: Path) -> list[str]:
    if not train_dir.is_dir():
        raise SystemExit(f"TRAIN_DIR not found: {train_dir}")
    return sorted(d.name for d in train_dir.iterdir() if d.is_dir())


def load_trained_weights(model: tf.keras.Model, ckpt_dir: Path) -> None:
    if not ckpt_dir.is_dir():
        raise SystemExit(f"CHECKPOINT_DIR not found: {ckpt_dir}")
    latest = tf.train.latest_checkpoint(str(ckpt_dir))
    if latest is None:
        raise SystemExit(
            f"No TensorFlow checkpoint found in {ckpt_dir}. "
            "Re-run the fit() cell in the notebook first."
        )
    print(f"Loading weights from: {latest}")
    model.load_weights(latest).expect_partial()


def copy_examples(valid_dir: Path, dest: Path, classes: list[str]) -> None:
    if not valid_dir.is_dir():
        print(f"VALID_DIR not found ({valid_dir}); skipping examples copy.")
        return
    dest.mkdir(exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png"}
    for cls in classes:
        src_dir = valid_dir / cls
        if not src_dir.is_dir():
            print(f"  skip example (missing class): {cls}")
            continue
        candidates = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in image_exts)
        if not candidates:
            print(f"  skip example (no images): {cls}")
            continue
        out_name = f"{cls}.jpg"
        shutil.copyfile(str(candidates[0]), str(dest / out_name))
        print(f"  example: {out_name}")


def main() -> None:
    print(f"TRAIN_DIR      = {TRAIN_DIR}")
    print(f"VALID_DIR      = {VALID_DIR}")
    if SOURCE_KERAS:
        print(f"SOURCE_KERAS   = {SOURCE_KERAS}  (skipping checkpoint load)")
    else:
        print(f"CHECKPOINT_DIR = {CHECKPOINT_DIR}")
    print()

    class_names = discover_class_names(TRAIN_DIR)
    if len(class_names) != 38:
        raise SystemExit(
            f"Expected 38 class folders in {TRAIN_DIR}, found {len(class_names)}."
        )

    if SOURCE_KERAS:
        src = Path(SOURCE_KERAS)
        if not src.is_file():
            raise SystemExit(f"SOURCE_KERAS not found: {src}")
        print(f"Loading pre-saved model {src} (will re-emit in current TF format)")
        # compile=False skips the optimizer-state load, which is the part most
        # likely to break across TF versions. We don't need the optimizer for
        # inference. Accept .h5 or .keras input.
        try:
            model = tf.keras.models.load_model(str(src), compile=False)
        except Exception as e:
            raise SystemExit(
                f"Failed to load {src} in this TF version ({tf.__version__}). "
                f"If the source was saved with TF 2.13+ try the .h5 sibling instead. "
                f"Underlying error: {type(e).__name__}: {e}"
            )
        if model.output_shape[-1] != 38:
            raise SystemExit(
                f"Loaded model has output dim {model.output_shape[-1]}, expected 38."
            )
        print(f"Re-saving in TF {tf.__version__} format -> {KERAS_OUT}")
        model.save(str(KERAS_OUT))
    else:
        model = build_model()
        load_trained_weights(model, CHECKPOINT_DIR)
        print(f"\nSaving model -> {KERAS_OUT}")
        model.save(str(KERAS_OUT))

    payload = json.dumps(class_names, indent=2) + "\n"
    CLASS_NAMES_OUT_ROOT.write_text(payload, encoding="utf-8")
    print(f"Saved class names -> {CLASS_NAMES_OUT_ROOT}")

    CLASS_NAMES_OUT_APP.parent.mkdir(exist_ok=True)
    CLASS_NAMES_OUT_APP.write_text(payload, encoding="utf-8")
    print(f"Saved class names -> {CLASS_NAMES_OUT_APP}  (committed copy)")

    print(f"\nCopying example images -> {EXAMPLES_DIR}/")
    copy_examples(VALID_DIR, EXAMPLES_DIR, EXAMPLE_CLASSES)

    print("\n=== sanity check ===")
    print(f"Total params:    {model.count_params():,}")
    print(f"First 3 classes: {class_names[:3]}")
    print(f"Last 3 classes:  {class_names[-3:]}")


if __name__ == "__main__":
    main()
