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
from PIL import Image

from .model_utils import ensure_model_present

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
        if len(class_names) != 38:
            raise ValueError(
                f"Expected 38 class names in {CLASS_NAMES_PATH}, got {len(class_names)}."
            )
        model = tf.keras.models.load_model(str(MODEL_PATH))
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
    model, class_names = load_model()

    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    img = tf.image.resize(arr, (224, 224))
    img = tf.expand_dims(img, axis=0)
    probs = np.asarray(model.predict(img, verbose=0)[0], dtype=np.float64)

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
