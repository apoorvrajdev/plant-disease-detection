"""Lazy download of the trained .keras model from a GitHub Release.

The Space's image is built without the trained weights (~30 MB) — they live
in a GitHub Release. On first import, ``ensure_model_present`` checks the
local cache and downloads the file if missing. Subsequent boots are no-ops.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL_RELEASE_URL = (
    "https://github.com/apoorvrajdev/plant-disease-detection"
    "/releases/download/v1.0-model-weights/plant_disease_model.keras"
)

_CHUNK = 1 << 20  # 1 MiB


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError):
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def ensure_model_present(model_path: Path) -> None:
    """Ensure the .keras file is on disk, downloading from a GH Release if not."""
    if model_path.is_file():
        return
    url = os.environ.get("MODEL_RELEASE_URL", DEFAULT_MODEL_RELEASE_URL)
    if not url:
        raise FileNotFoundError(
            f"Model not found at {model_path} and MODEL_RELEASE_URL is unset. "
            "Either run export_model.py locally, or set MODEL_RELEASE_URL to "
            "the GitHub Release asset URL."
        )
    print(f"Downloading model from {url} -> {model_path}", flush=True)
    _download(url, model_path)
    print(f"Downloaded model to {model_path}", flush=True)
