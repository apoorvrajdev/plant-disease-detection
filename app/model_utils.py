"""Lazy download of the trained .keras model from a GitHub Release.

The Space's image is built without the trained weights (~30 MB) — they live
in a GitHub Release. On first import, ``ensure_model_present`` checks the
local cache and downloads the file if missing. Subsequent boots are no-ops.

Environment overrides:
    MODEL_RELEASE_URL  Source URL for the .keras asset.
    MODEL_SHA256       Optional hex SHA-256 of the asset. When set, the
                       downloaded file is verified and rejected on mismatch.
                       Strongly recommended in production: a corrupted or
                       tampered .keras would otherwise be loaded blindly by
                       Keras, which can execute arbitrary code via Lambda
                       layers.
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL_RELEASE_URL = (
    "https://github.com/apoorvrajdev/plant-disease-detection"
    "/releases/download/v1.0-model-weights/plant_disease_model.keras"
)

_CHUNK = 1 << 20  # 1 MiB
_CONNECT_TIMEOUT_S = 30
_MAX_ATTEMPTS = 3
# Floor on a plausible model size. The real artifact is ~30 MB; anything
# under 1 MB is almost certainly a redirect/error page that slipped through
# as a 200 — refusing it gives a useful error instead of a cryptic
# "bad .keras file" from the Keras loader.
_MIN_BYTES = 1 << 20


def _download_once(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Some CDNs reject the default ``Python-urllib/x.y`` UA on redirects.
    req = urllib.request.Request(
        url, headers={"User-Agent": "plant-disease-detection/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_CONNECT_TIMEOUT_S) as resp, \
                open(tmp, "wb") as out:
            written = 0
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        if written < _MIN_BYTES:
            raise OSError(
                f"Downloaded artifact is suspiciously small ({written} bytes); "
                f"refusing to install. Check MODEL_RELEASE_URL."
            )
        tmp.replace(dest)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _verify_sha256(path: Path, expected_hex: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected_hex.lower():
        try:
            path.unlink()
        except OSError:
            pass
        raise OSError(
            "SHA-256 mismatch for downloaded model.\n"
            f"  expected: {expected_hex.lower()}\n"
            f"  actual:   {actual}\n"
            "File deleted. Refusing to load an unverified model."
        )


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

    expected_sha = os.environ.get("MODEL_SHA256", "").strip()

    last_err: BaseException | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        print(
            f"Downloading model (attempt {attempt}/{_MAX_ATTEMPTS}) "
            f"from {url} -> {model_path}",
            flush=True,
        )
        try:
            _download_once(url, model_path)
            last_err = None
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            print(f"  download failed: {type(e).__name__}: {e}", flush=True)
            if attempt < _MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
    if last_err is not None:
        raise RuntimeError(
            f"Failed to download model after {_MAX_ATTEMPTS} attempts: {last_err}"
        ) from last_err

    if expected_sha:
        _verify_sha256(model_path, expected_sha)
        print("  sha256 verified", flush=True)

    print(f"Downloaded model to {model_path}", flush=True)
