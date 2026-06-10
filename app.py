#!/usr/bin/env python3
"""
Clothing Lay-Up Generator — Web UI
Run:  python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from PIL import Image
import numpy as np

from clothing_layup import (
    TYPE_TO_LABEL_IDS,
    rembg_mask,
    segformer_mask,
    refine_mask,
    feather_mask,
    compose_layup,
)

GARMENT_CHOICES = [
    "top",
    "pants",
    "dress",
    "skirt",
    "outfit",
    "jacket",
    "coat",
    "blouse",
    "shirt",
    "jeans",
    "trousers",
    "shorts",
    "jumpsuit",
]

BG_PRESETS = {
    "White":       (255, 255, 255),
    "Light grey":  (240, 240, 240),
    "Cream":       (255, 253, 240),
    "Black":       (0,   0,   0),
    "Soft pink":   (255, 240, 245),
}


def run_layup(
    image: Image.Image,
    garment_type: str,
    engine: str,
    bg_preset: str,
    output_size: int,
    feather: int,
    shadow: bool,
) -> Image.Image:
    if image is None:
        raise gr.Error("Please upload a photo first.")

    image = image.convert("RGB")
    label_ids = TYPE_TO_LABEL_IDS[garment_type]
    bg_color = BG_PRESETS[bg_preset]

    raw_mask: np.ndarray | None = None

    if engine in ("Auto (SegFormer → rembg)", "SegFormer only"):
        try:
            raw_mask = segformer_mask(image, label_ids)
        except Exception as exc:
            if engine == "SegFormer only":
                raise gr.Error(
                    f"SegFormer failed: {exc}. "
                    "Make sure you have internet access for the first run."
                )

    if raw_mask is None:
        raw_mask = rembg_mask(image, garment_type)

    clean_mask = refine_mask(raw_mask)
    soft_mask  = feather_mask(clean_mask, radius=feather)

    result = compose_layup(
        image, soft_mask,
        bg_color=bg_color,
        output_size=output_size,
        shadow=shadow,
    )
    return result


with gr.Blocks(title="Clothing Lay-Up Generator") as demo:
    gr.Markdown(
        """
        # 👗 Clothing Lay-Up Generator
        Upload a model photo, choose the garment, and get a clean flat-lay product shot.
        """
    )

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload model photo",
                type="pil",
                height=420,
            )
            garment_dd = gr.Dropdown(
                choices=GARMENT_CHOICES,
                value="top",
                label="Garment type",
            )

            with gr.Accordion("Options", open=False):
                engine_dd = gr.Dropdown(
                    choices=[
                        "Auto (SegFormer → rembg)",
                        "SegFormer only",
                        "rembg only",
                    ],
                    value="Auto (SegFormer → rembg)",
                    label="Segmentation engine",
                    info=(
                        "SegFormer is clothing-aware and more accurate. "
                        "Downloads ~400 MB on first run. "
                        "rembg works offline."
                    ),
                )
                bg_dd = gr.Dropdown(
                    choices=list(BG_PRESETS.keys()),
                    value="White",
                    label="Background colour",
                )
                size_sl = gr.Slider(
                    minimum=500, maximum=3000, step=100, value=1500,
                    label="Output canvas size (px)",
                )
                feather_sl = gr.Slider(
                    minimum=0, maximum=20, step=1, value=6,
                    label="Edge feather radius (px)",
                )
                shadow_cb = gr.Checkbox(value=True, label="Add drop shadow")

            run_btn = gr.Button("Generate Lay-Up", variant="primary")

        # ── Right column: output ──────────────────────────────────────────────
        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Flat-lay output",
                type="pil",
                height=420,
            )

    run_btn.click(
        fn=run_layup,
        inputs=[
            image_input,
            garment_dd,
            engine_dd,
            bg_dd,
            size_sl,
            feather_sl,
            shadow_cb,
        ],
        outputs=image_output,
    )

    gr.Markdown(
        """
        ---
        **Tips**
        - *SegFormer* gives the cleanest cut because it understands clothing categories.
          It downloads ~400 MB on first run, then works offline.
        - *rembg* is the fast offline fallback — it removes the background and uses a
          positional split to isolate the garment.
        - For a two-piece, run it **twice**: once for `top` and once for `pants`/`skirt`.
        - Increase *Edge feather* if you see hard jagged edges.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
