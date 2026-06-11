#!/usr/bin/env python3
"""
Clothing Lay-Up Generator — Web UI
Run:  python3 app.py
Then open http://localhost:7860 in your browser.
"""

import sys
import gradio as gr
from PIL import Image
import numpy as np

from clothing_layup import (
    TYPE_TO_LABEL_IDS,
    rembg_mask,
    refine_mask,
    feather_mask,
    compose_layup,
)

GARMENT_CHOICES = [
    "top", "pants", "dress", "skirt", "outfit",
    "jacket", "coat", "blouse", "shirt",
    "jeans", "trousers", "shorts", "jumpsuit",
]

BG_PRESETS = {
    "White":      (255, 255, 255),
    "Light grey": (240, 240, 240),
    "Cream":      (255, 253, 240),
    "Black":      (0,   0,   0),
    "Soft pink":  (255, 240, 245),
}

# ── Load rembg model in background so server starts immediately ───────────────
import threading
_REMBG_SESSION = None
_rembg_ready = threading.Event()

def _load_rembg():
    global _REMBG_SESSION
    try:
        from rembg import new_session
        _REMBG_SESSION = new_session("u2net")
        print("Background-removal model ready.")
    except Exception as e:
        print(f"rembg warning: {e}")
    _rembg_ready.set()

threading.Thread(target=_load_rembg, daemon=True).start()

# ── SegFormer loaded lazily on first use ──────────────────────────────────────
_SEGFORMER_LOADED = False
_SF_PROCESSOR = None
_SF_MODEL = None


def _segformer_mask_cached(image, label_ids):
    """Use pre-loaded model if available, otherwise load (and cache) from HuggingFace."""
    global _SEGFORMER_LOADED, _SF_PROCESSOR, _SF_MODEL
    if not _SEGFORMER_LOADED:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        from clothing_layup import SEGFORMER_MODEL
        _SF_PROCESSOR = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
        _SF_MODEL = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
        _SF_MODEL.eval()
        _SEGFORMER_LOADED = True

    import torch
    inputs = _SF_PROCESSOR(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = _SF_MODEL(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    return np.isin(seg_map, label_ids).astype(np.uint8)


def _rembg_mask_fast(image, garment_type):
    """rembg using the pre-loaded session (waits for it if still loading)."""
    from rembg import remove
    from clothing_layup import TYPE_TO_VBAND

    _rembg_ready.wait()   # blocks only if model hasn't finished loading yet
    kwargs = {"session": _REMBG_SESSION} if _REMBG_SESSION else {}
    rgba = remove(image, **kwargs)
    alpha = np.array(rgba.split()[3])

    h = alpha.shape[0]
    v_start, v_end = TYPE_TO_VBAND[garment_type]
    y0, y1 = int(h * v_start), int(h * v_end)

    band_mask = np.zeros_like(alpha, dtype=np.float32)
    band_mask[y0:y1, :] = 1.0

    feather_px = max(1, int(h * 0.06))
    for dy in range(feather_px):
        blend = dy / feather_px
        if y0 + dy < h:
            band_mask[y0 + dy, :] = np.minimum(band_mask[y0 + dy, :], blend)
        if y1 - 1 - dy >= 0:
            band_mask[y1 - 1 - dy, :] = np.minimum(band_mask[y1 - 1 - dy, :], blend)

    combined = (alpha / 255.0) * band_mask
    return (combined > 0.3).astype(np.uint8)


def run_layup(image, garment_type, engine, bg_preset, output_size, feather, shadow):
    if image is None:
        raise gr.Error("Please upload a photo first.")

    image = image.convert("RGB")
    label_ids = TYPE_TO_LABEL_IDS[garment_type]
    bg_color = BG_PRESETS[bg_preset]
    raw_mask = None

    if engine in ("SegFormer + rembg fallback", "SegFormer only"):
        try:
            raw_mask = _segformer_mask_cached(image, label_ids)
        except Exception as exc:
            if engine == "SegFormer only":
                raise gr.Error(
                    f"SegFormer failed: {exc}\n"
                    "It needs to download ~400 MB on first use — check your internet connection."
                )

    if raw_mask is None:
        raw_mask = _rembg_mask_fast(image, garment_type)

    clean_mask = refine_mask(raw_mask)
    soft_mask  = feather_mask(clean_mask, radius=feather)

    return compose_layup(
        image, soft_mask,
        bg_color=bg_color,
        output_size=output_size,
        shadow=shadow,
    )


with gr.Blocks(title="Clothing Lay-Up Generator") as demo:
    gr.Markdown("# Clothing Lay-Up Generator\nUpload a model photo → get a clean flat-lay product shot.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload model photo", type="pil", height=400)
            garment_dd  = gr.Dropdown(choices=GARMENT_CHOICES, value="top", label="Garment type")

            with gr.Accordion("Options", open=False):
                engine_dd = gr.Dropdown(
                    choices=["rembg (fast)", "SegFormer + rembg fallback", "SegFormer only"],
                    value="rembg (fast)",
                    label="Segmentation engine",
                    info="rembg is fast and works offline. SegFormer is more accurate but downloads ~400 MB on first use.",
                )
                bg_dd      = gr.Dropdown(choices=list(BG_PRESETS.keys()), value="White", label="Background")
                size_sl    = gr.Slider(500, 3000, step=100, value=1500, label="Output size (px)")
                feather_sl = gr.Slider(0, 20, step=1, value=6, label="Edge feather (px)")
                shadow_cb  = gr.Checkbox(value=True, label="Drop shadow")

            run_btn = gr.Button("Generate Lay-Up", variant="primary", size="lg")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Flat-lay output", type="pil", height=400)

    run_btn.click(
        fn=run_layup,
        inputs=[image_input, garment_dd, engine_dd, bg_dd, size_sl, feather_sl, shadow_cb],
        outputs=image_output,
    )

    gr.Markdown(
        "**Tips:** For two-piece outfits run twice — once for `top`, once for `pants`/`skirt`. "
        "Increase *Edge feather* if edges look jagged. "
        "Switch to *SegFormer* for more precise cuts around complex shapes."
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
