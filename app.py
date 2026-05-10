"""Gradio entry point for the plant disease classifier.

Local:        python app.py
HF Space:     uploaded as the Space's app_file (see deploy_to_hf.py).

The model weights live in a GitHub Release, not in this Space's image.
``ensure_model_present`` downloads them on cold start.
"""
from __future__ import annotations

from pathlib import Path


def _patch_gradio_client_for_bool_schemas() -> None:
    """Workaround for a Gradio 4.44.x bundled-gradio_client bug.

    Two functions in ``gradio_client.utils`` mishandle bool JSON schemas
    (``additionalProperties: True`` etc.):

    * ``get_type`` does ``if "const" in schema`` without a type check,
      so it raises ``TypeError: argument of type 'bool' is not iterable``.
    * ``_json_schema_to_python_type`` falls through to a final
      ``raise APIInfoParseError(f"Cannot parse schema {schema}")`` when
      none of its branches match a bool input.

    Either failure short-circuits Gradio's launch self-check (the homepage
    handler invokes ``api_info`` even with ``show_api=False``) and crashes
    the app with a misleading "localhost not accessible" ValueError.

    We wrap both functions to short-circuit bool inputs to ``"Any"``,
    which is the correct JSON-Schema-to-Python rendering for a bool
    schema (``True`` = any value matches; ``False`` = nothing matches —
    "Any" is a safe Python type representation for either at this layer).

    Must run before ``import gradio`` so subsequent imports observe the
    patched module attributes.
    """
    try:
        from gradio_client import utils as _gcu
    except Exception:
        return

    _orig_get_type = _gcu.get_type
    _orig_walker = _gcu._json_schema_to_python_type

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)

    def _safe_walker(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_walker(schema, defs)

    _gcu.get_type = _safe_get_type
    _gcu._json_schema_to_python_type = _safe_walker


_patch_gradio_client_for_bool_schemas()

import gradio as gr  # noqa: E402  (must follow the patch above)

from app.model_utils import ensure_model_present
from app.predict import MODEL_PATH, predict

# Trigger the weights download (or no-op if already on disk) before the UI
# accepts traffic. On HF Spaces this happens once per cold boot.
ensure_model_present(MODEL_PATH)

REPO_URL = "https://github.com/apoorvrajdev/plant-disease-detection"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def _list_examples() -> list[list[str]] | None:
    if not EXAMPLES_DIR.is_dir():
        return None
    files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(EXAMPLES_DIR.glob(ext)))
    return [[str(f)] for f in files] if files else None


def _format_result(result: dict) -> str:
    header = (
        result["condition"]
        if result["is_healthy"]
        else f"{result['crop']} — {result['condition']}"
    )

    top3_lines = []
    for i, item in enumerate(result["top_3"], start=1):
        pct = item["prob"] * 100
        bar_len = max(1, int(round(pct / 5)))
        bar = "█" * bar_len
        top3_lines.append(
            f"{i}. **{item['crop']} — {item['condition']}** — "
            f"{pct:.2f}% `{bar}`"
        )

    return (
        f"### {header}\n\n"
        f"- **Crop:** {result['crop']}\n"
        f"- **Condition:** {result['condition']}\n"
        f"- **Healthy:** {'yes' if result['is_healthy'] else 'no'}\n"
        f"- **Confidence:** {result['confidence'] * 100:.2f}%\n\n"
        f"**Top 3 predictions**\n\n"
        f"{chr(10).join(top3_lines)}\n\n"
        f"<sub>Raw label: <code>{result['raw_label']}</code></sub>"
    )


def classify(pil_image) -> str:
    if pil_image is None:
        return "_Upload a leaf photo to get a prediction._"
    return _format_result(predict(pil_image))


with gr.Blocks(title="Plant Disease Detection") as demo:
    gr.Markdown(
        "# Plant Disease Detection\n\n"
        "Upload a leaf photo. The model identifies the crop and classifies it "
        "across 38 disease classes (including healthy) for 14 plant species. "
        "Built on EfficientNetB0 fine-tuned with TensorFlow / Keras."
    )

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Leaf photo")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_md = gr.Markdown(value="_Upload a leaf photo to get a prediction._")

    btn.click(fn=classify, inputs=inp, outputs=out_md)
    inp.change(fn=classify, inputs=inp, outputs=out_md)

    _examples = _list_examples()
    if _examples:
        gr.Examples(
            examples=_examples,
            inputs=inp,
            outputs=out_md,
            fn=classify,
            cache_examples=False,
            label="Examples",
        )

    gr.Markdown(
        f"---\n\nSource: [{REPO_URL}]({REPO_URL})  •  "
        "Model: EfficientNetB0 transfer-learned on the New Plant Diseases Dataset (38 classes)."
    )


if __name__ == "__main__":
    # show_api=False sidesteps a known Gradio 4.44.x bug where the schema
    # walker chokes on bool additionalProperties. With dict-shaped outputs
    # eliminated (gr.Label removed), we'd no longer hit it anyway, but keeping
    # the flag is belt-and-suspenders.
    demo.launch(show_api=False)
