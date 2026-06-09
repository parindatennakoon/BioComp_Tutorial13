#!/usr/bin/env python3
"""
Clothing Lay-Up Generator
Extracts a specific garment from a model photo and creates a clean flat-lay output.

Usage:
  python clothing_layup.py photo.jpg --type top
  python clothing_layup.py photo.jpg --type pants
  python clothing_layup.py photo.jpg --type dress
  python clothing_layup.py photo.jpg --type skirt
  python clothing_layup.py photo.jpg --type outfit   # full look

Supported --type values:
  top / shirt / blouse / jacket / coat  -> upper body garment
  pants / trousers / jeans / shorts     -> lower body trousers
  skirt                                 -> skirt
  dress / jumpsuit                      -> dress
  outfit / all                          -> every garment at once

Engines (--engine):
  auto       – try SegFormer first, fall back to rembg if unavailable [default]
  segformer  – mattmdjaga/segformer_b2_clothes (best quality, needs internet once)
  rembg      – background removal + positional split (no internet required)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

# ── Clothing type → SegFormer label IDs ──────────────────────────────────────
# mattmdjaga/segformer_b2_clothes label set:
#  0 background  1 hat      2 hair       3 sunglasses  4 upper-clothes
#  5 skirt       6 pants    7 dress      8 belt        9 left-shoe
# 10 right-shoe 11 face    12 left-leg  13 right-leg  14 left-arm
# 15 right-arm  16 bag     17 scarf
SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"

TYPE_TO_LABEL_IDS: dict[str, list[int]] = {
    "top":       [4],
    "shirt":     [4],
    "blouse":    [4],
    "jacket":    [4],
    "coat":      [4],
    "pants":     [6],
    "trousers":  [6],
    "jeans":     [6],
    "shorts":    [6],
    "skirt":     [5],
    "dress":     [7],
    "jumpsuit":  [7],
    "outfit":    [4, 5, 6, 7],
    "all":       [4, 5, 6, 7],
}

# For the rembg fallback: where the garment sits vertically in the person crop
# (0.0 = very top, 1.0 = very bottom).  Each entry is (start_frac, end_frac).
TYPE_TO_VBAND: dict[str, tuple[float, float]] = {
    "top":       (0.10, 0.58),
    "shirt":     (0.10, 0.58),
    "blouse":    (0.10, 0.58),
    "jacket":    (0.08, 0.62),
    "coat":      (0.08, 0.75),
    "pants":     (0.45, 1.00),
    "trousers":  (0.45, 1.00),
    "jeans":     (0.45, 1.00),
    "shorts":    (0.45, 0.75),
    "skirt":     (0.42, 0.90),
    "dress":     (0.08, 1.00),
    "jumpsuit":  (0.08, 1.00),
    "outfit":    (0.08, 1.00),
    "all":       (0.08, 1.00),
}


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation engines
# ─────────────────────────────────────────────────────────────────────────────

def segformer_mask(image: Image.Image, label_ids: list[int]) -> np.ndarray:
    """Return uint8 mask (H×W) using SegFormer clothing segmentation."""
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, n_labels, H/4, W/4)

    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    return np.isin(seg_map, label_ids).astype(np.uint8)


def rembg_mask(image: Image.Image, clothing_type: str) -> np.ndarray:
    """
    Fallback: use rembg to remove background, then isolate the garment
    using a vertical band heuristic based on clothing_type.
    """
    from rembg import remove

    rgba = remove(image)                        # removes bg, person becomes RGBA
    alpha = np.array(rgba.split()[3])           # 0-255 alpha of the full person

    h = alpha.shape[0]
    v_start, v_end = TYPE_TO_VBAND[clothing_type]
    y0 = int(h * v_start)
    y1 = int(h * v_end)

    # Build a gradient vertical band mask blended with the person alpha
    band_mask = np.zeros_like(alpha, dtype=np.float32)
    band_mask[y0:y1, :] = 1.0

    # Feather the top and bottom edges of the band
    feather_px = max(1, int(h * 0.06))
    for dy in range(feather_px):
        blend = dy / feather_px
        if y0 + dy < h:
            band_mask[y0 + dy, :] = np.minimum(band_mask[y0 + dy, :], blend)
        if y1 - 1 - dy >= 0:
            band_mask[y1 - 1 - dy, :] = np.minimum(band_mask[y1 - 1 - dy, :], blend)

    combined = (alpha / 255.0) * band_mask
    return (combined > 0.3).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Mask refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Close holes, remove specks, keep largest component."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask


def feather_mask(mask: np.ndarray, radius: int = 6) -> np.ndarray:
    """Gaussian-blur mask edges for anti-aliased cut-outs."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=radius)
    return np.clip(blurred, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Composition helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_to_content(rgba: Image.Image, padding: int = 40) -> Image.Image:
    alpha = np.array(rgba.split()[3])
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any():
        return rgba
    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    rmin = max(0, rmin - padding)
    rmax = min(rgba.height - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(rgba.width  - 1, cmax + padding)
    return rgba.crop((cmin, rmin, cmax + 1, rmax + 1))


def add_shadow(
    rgba: Image.Image,
    offset: int = 20,
    blur: int = 24,
    opacity: int = 75,
) -> Image.Image:
    w, h = rgba.size
    pad = abs(offset) + blur + 10
    canvas = (w + pad * 2, h + pad * 2)

    # Shadow
    shadow = Image.new("RGBA", canvas, (0, 0, 0, 0))
    shadow_src = rgba.copy()
    r, g, b, a = shadow_src.split()
    shadow_src = Image.merge("RGBA", (
        Image.new("L", rgba.size, 25),
        Image.new("L", rgba.size, 25),
        Image.new("L", rgba.size, 25),
        a,
    ))
    shadow.paste(shadow_src, (pad + offset, pad + offset))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))
    sr, sg, sb, sa = shadow.split()
    sa = sa.point(lambda x: int(x * opacity / 255))
    shadow = Image.merge("RGBA", (sr, sg, sb, sa))

    # Garment on top
    result = Image.new("RGBA", canvas, (0, 0, 0, 0))
    result = Image.alpha_composite(result, shadow)
    fg = Image.new("RGBA", canvas, (0, 0, 0, 0))
    fg.paste(rgba, (pad, pad))
    return Image.alpha_composite(result, fg)


def compose_layup(
    original: Image.Image,
    soft_mask: np.ndarray,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    output_size: int = 1500,
    shadow: bool = True,
) -> Image.Image:
    orig_np = np.array(original.convert("RGBA"))
    orig_np[:, :, 3] = (soft_mask * 255).astype(np.uint8)
    rgba = Image.fromarray(orig_np, "RGBA")

    rgba = crop_to_content(rgba, padding=30)
    if shadow:
        rgba = add_shadow(rgba)

    # Scale to fit within output_size
    ratio = output_size / max(rgba.size)
    new_w = max(1, int(rgba.width  * ratio))
    new_h = max(1, int(rgba.height * ratio))
    rgba = rgba.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (output_size, output_size), (*bg_color, 255))
    px = (output_size - new_w) // 2
    py = (output_size - new_h) // 2
    canvas.paste(rgba, (px, py), rgba)
    return canvas.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract a garment from a model photo and produce a flat-lay image."
    )
    p.add_argument("image", help="Path to the input model photo")
    p.add_argument(
        "--type", "-t", required=True,
        choices=sorted(TYPE_TO_LABEL_IDS.keys()),
        metavar="TYPE",
        help="Garment to extract. Choices: " + ", ".join(sorted(TYPE_TO_LABEL_IDS.keys())),
    )
    p.add_argument("--output", "-o", default=None,
                   help="Output path (default: <input>_layup_<type>.png)")
    p.add_argument("--engine", choices=["auto", "segformer", "rembg"], default="auto",
                   help="Segmentation engine (default: auto)")
    p.add_argument("--size", type=int, default=1500,
                   help="Output canvas size in px (default: 1500)")
    p.add_argument("--bg", default="255,255,255",
                   help="Background colour as R,G,B (default: white)")
    p.add_argument("--no-shadow", action="store_true", help="Disable drop shadow")
    p.add_argument("--feather", type=int, default=6,
                   help="Edge feather radius in px (default: 6)")
    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.exists():
        sys.exit(f"Error: file not found – {input_path}")

    try:
        bg_color: tuple[int, int, int] = tuple(int(v) for v in args.bg.split(","))  # type: ignore[assignment]
        assert len(bg_color) == 3
    except Exception:
        sys.exit("Error: --bg must be R,G,B e.g. 240,240,240")

    output_path = (
        Path(args.output) if args.output
        else input_path.with_name(f"{input_path.stem}_layup_{args.type}.png")
    )

    print(f"Reading {input_path} …")
    image = Image.open(input_path).convert("RGB")
    print(f"  Size: {image.width}×{image.height}")

    # ── Choose engine ──────────────────────────────────────────────────────────
    label_ids = TYPE_TO_LABEL_IDS[args.type]
    engine = args.engine
    raw_mask: np.ndarray | None = None

    if engine in ("auto", "segformer"):
        try:
            print("Engine: SegFormer (clothing segmentation) …")
            raw_mask = segformer_mask(image, label_ids)
            print("  SegFormer OK")
        except Exception as exc:
            if engine == "segformer":
                sys.exit(f"SegFormer failed: {exc}")
            print(f"  SegFormer unavailable ({exc.__class__.__name__}), falling back to rembg …")

    if raw_mask is None:
        print("Engine: rembg + positional split …")
        raw_mask = rembg_mask(image, args.type)
        print("  rembg OK")

    coverage = float(raw_mask.mean()) * 100
    if coverage < 0.5:
        print(
            f"Warning: only {coverage:.1f}% of pixels matched '{args.type}'. "
            "Verify the clothing type matches the photo."
        )

    print("Refining mask …")
    clean_mask  = refine_mask(raw_mask)
    soft_mask   = feather_mask(clean_mask, radius=args.feather)

    print("Composing flat-lay …")
    result = compose_layup(
        image, soft_mask,
        bg_color=bg_color,
        output_size=args.size,
        shadow=not args.no_shadow,
    )

    result.save(output_path, "PNG")
    print(f"\nSaved → {output_path}  ({result.width}×{result.height} px)")


if __name__ == "__main__":
    main()
