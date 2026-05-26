"""Inference module for the plant disease classifier.

Public API:
    load_model() -> (tf.keras.Model, list[str])  # cached, idempotent
    predict(pil_image: PIL.Image.Image) -> dict   # see docstring
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from .model_utils import ensure_model_present

# Upper bound on the longest edge we let through before resizing to 224×224.
# Prevents a multi-hundred-MB float32 allocation on accidental (or hostile)
# huge uploads. 2048 is well above any realistic phone photo's useful detail
# at 224×224 and keeps the float32 intermediate under ~50 MB.
_MAX_INPUT_EDGE = 2048
_INPUT_SIZE = (224, 224)
_NUM_CLASSES = 38

_PKG = Path(__file__).resolve().parent
_REPO_ROOT = _PKG.parent

CLASS_NAMES_PATH = _PKG / "class_names.json"
MODEL_PATH = _REPO_ROOT / "plant_disease_model.keras"

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Optional[tuple[tf.keras.Model, list[str]]] = None

_PARENS_RE = re.compile(r"_*\([^)]*\)")


def load_model() -> tuple[tf.keras.Model, list[str]]:
    """Load and cache the .keras model and class names. Idempotent."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    with _MODEL_LOCK:
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE
        ensure_model_present(MODEL_PATH)
        if not CLASS_NAMES_PATH.is_file():
            raise FileNotFoundError(
                f"class_names.json not found at {CLASS_NAMES_PATH}. "
                "Run export_model.py first to generate it."
            )
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        if len(class_names) != _NUM_CLASSES:
            raise ValueError(
                f"Expected {_NUM_CLASSES} class names in {CLASS_NAMES_PATH}, "
                f"got {len(class_names)}."
            )
        model = tf.keras.models.load_model(str(MODEL_PATH))
        out_dim = int(model.output_shape[-1])
        if out_dim != _NUM_CLASSES:
            raise ValueError(
                f"Loaded model at {MODEL_PATH} has output dim {out_dim}, "
                f"expected {_NUM_CLASSES}. The class_names.json and the model "
                f"weights are out of sync — re-run export_model.py."
            )
        _MODEL_CACHE = (model, class_names)
        return _MODEL_CACHE


def _clean_crop(raw_crop: str) -> str:
    # Drop parenthetical qualifiers ("Cherry_(including_sour)" -> "Cherry").
    s = _PARENS_RE.sub("", raw_crop)
    # Comma + underscore -> space; "Pepper,_bell" -> "Pepper bell".
    s = s.replace(",", " ").replace("_", " ")
    s = " ".join(s.split())
    # Title-case so "Pepper bell" -> "Pepper Bell". Safe across this dataset
    # (no apostrophes / digits in any class name).
    return s.title()


def _clean_condition(raw_condition: str) -> str:
    parts = []
    # Some labels join two synonym names with a literal space, e.g.
    # "Cercospora_leaf_spot Gray_leaf_spot". Render them as "A / B".
    for piece in raw_condition.split(" "):
        piece = piece.rstrip("_").replace("_", " ")
        piece = " ".join(piece.split())
        if piece:
            parts.append(piece)
    if not parts:
        return raw_condition
    return " / ".join(p.title() for p in parts)


def _parse_label(raw_label: str) -> tuple[str, str, bool]:
    crop_raw, _, cond_raw = raw_label.partition("___")
    crop = _clean_crop(crop_raw)
    is_healthy = cond_raw.lower() == "healthy"
    condition = "Healthy" if is_healthy else _clean_condition(cond_raw)
    return crop, condition, is_healthy


def predict(pil_image: Image.Image) -> dict:
    """Run inference on a PIL image and return a structured prediction dict.

    Returns:
        {
          "crop": "Tomato",
          "condition": "Early blight" or "Healthy",
          "is_healthy": bool,
          "confidence": float,                # softmax prob of top-1
          "top_3": [
            {"crop": str, "condition": str, "prob": float},
            ...
          ],
          "raw_label": "Tomato___Early_blight"  # untouched dataset label
        }
    """
    if not isinstance(pil_image, Image.Image):
        raise TypeError(
            f"predict() expects a PIL.Image.Image, got {type(pil_image).__name__}."
        )

    model, class_names = load_model()

    # 1) Honour EXIF orientation. Phone uploads frequently carry an orientation
    #    tag rather than physically rotated pixels; without this the model sees
    #    sideways/upside-down leaves and silently regresses on field photos.
    # 2) Bound the working size *before* the float32 cast to keep memory
    #    deterministic on oversized inputs.
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    if max(img.size) > _MAX_INPUT_EDGE:
        img.thumbnail((_MAX_INPUT_EDGE, _MAX_INPUT_EDGE), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)
    tensor = tf.image.resize(arr, _INPUT_SIZE)
    tensor = tf.expand_dims(tensor, axis=0)
    # model(...) is the low-latency single-image path; model.predict() adds
    # callback / dispatch overhead that dominates the per-request cost at
    # batch size 1.
    probs = np.asarray(model(tensor, training=False)[0], dtype=np.float64)

    top3_idx = np.argsort(probs)[-3:][::-1]
    top1 = int(top3_idx[0])
    raw_label = class_names[top1]
    crop, condition, is_healthy = _parse_label(raw_label)

    top_3 = []
    for idx in top3_idx:
        i = int(idx)
        c, cond, _ = _parse_label(class_names[i])
        top_3.append({"crop": c, "condition": cond, "prob": float(probs[i])})

    return {
        "crop": crop,
        "condition": condition,
        "is_healthy": is_healthy,
        "confidence": float(probs[top1]),
        "top_3": top_3,
        "raw_label": raw_label,
    }
