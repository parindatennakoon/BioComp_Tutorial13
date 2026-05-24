import io
import os
import textwrap
import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    ColorClip,
)

WIDTH = 1080
HEIGHT = 1920  # vertical / portrait (Instagram Reels ratio)
SLIDE_DURATION = 3.5  # seconds per product
TRANSITION_DURATION = 0.4
FPS = 30

# Colour palette
BG_COLOR = (15, 15, 15)
OVERLAY_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
PRICE_COLOR = (220, 185, 120)  # warm gold


def _load_font(size: int):
    """Try to load a bold system font, fall back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay-Bold.otf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _download_image(url: str) -> Image.Image | None:
    """Download and return a PIL image, or None on failure."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/*,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        return None


def _fit_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale + center-crop the image to fill the target dimensions."""
    img_w, img_h = img.size
    scale = max(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    color: tuple[int, int, int],
    shadow_offset: int = 3,
):
    x, y = xy
    # Shadow
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 180))
    # Main text
    draw.text((x, y), text, font=font, fill=color)


def _make_product_frame(product: dict, img: Image.Image) -> Image.Image:
    """Compose one 1080×1920 frame for a single product."""
    # Fit product image to canvas
    canvas = _fit_image(img, WIDTH, HEIGHT)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Gradient overlay at the bottom (covers ~45% of height)
    overlay_h = int(HEIGHT * 0.45)
    overlay_top = HEIGHT - overlay_h
    for i in range(overlay_h):
        alpha = int(210 * (i / overlay_h) ** 0.7)
        draw.rectangle(
            [(0, overlay_top + i), (WIDTH, overlay_top + i + 1)],
            fill=(0, 0, 0, alpha),
        )

    name_font = _load_font(52)
    price_font = _load_font(46)
    label_font = _load_font(28)

    # Price badge
    price = product.get("price", "")
    if price:
        price_text = price
        bbox = draw.textbbox((0, 0), price_text, font=price_font)
        pw, ph = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 18
        px = WIDTH - pw - pad * 2 - 40
        py = HEIGHT - 220
        draw.rounded_rectangle(
            [(px - pad, py - pad), (px + pw + pad, py + ph + pad)],
            radius=14,
            fill=(0, 0, 0, 190),
        )
        draw.text((px, py), price_text, font=price_font, fill=PRICE_COLOR)

    # Product name (wrapped)
    name = product.get("name", "")
    wrapped = textwrap.fill(name, width=26)
    lines = wrapped.split("\n")
    line_height = 62
    total_text_h = len(lines) * line_height
    name_y = HEIGHT - 230 - total_text_h

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=name_font)
        lw = bbox[2] - bbox[0]
        name_x = (WIDTH - lw) // 2
        _draw_text_with_shadow(draw, (name_x, name_y), line, name_font, TEXT_COLOR)
        name_y += line_height

    return canvas


def _make_title_frame(shop_name: str) -> Image.Image:
    """Create an opening title card."""
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(80)
    sub_font = _load_font(38)

    title = "SHOP COLLECTION"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 80), title, font=title_font, fill=TEXT_COLOR)

    if shop_name:
        short = shop_name[:40]
        bbox2 = draw.textbbox((0, 0), short, font=sub_font)
        sw = bbox2[2] - bbox2[0]
        draw.text(((WIDTH - sw) // 2, HEIGHT // 2 + 30), short, font=sub_font, fill=PRICE_COLOR)

    return canvas


def _pil_to_moviepy(pil_img: Image.Image, duration: float) -> ImageClip:
    import numpy as np
    arr = np.array(pil_img)
    clip = ImageClip(arr).set_duration(duration)
    return clip


def create_reel(
    products: list[dict],
    output_path: str,
    shop_name: str = "",
    progress_callback=None,
) -> str:
    """Generate an MP4 reel from a list of products.

    Returns the output path on success, raises on failure.
    """
    clips = []

    # Title slide (2 seconds)
    title_img = _make_title_frame(shop_name)
    clips.append(_pil_to_moviepy(title_img, 2.0))

    downloaded = 0
    for i, product in enumerate(products):
        if progress_callback:
            progress_callback(i, len(products))

        local_path = product.get("local_image_path")
        if local_path and os.path.exists(local_path):
            try:
                img = Image.open(local_path).convert("RGB")
            except Exception:
                img = None
        else:
            img = _download_image(product.get("image_url", ""))
        if img is None:
            continue

        frame = _make_product_frame(product, img)
        clip = _pil_to_moviepy(frame, SLIDE_DURATION)
        # Simple crossfade: each clip fades in and out
        clip = clip.crossfadein(TRANSITION_DURATION).crossfadeout(TRANSITION_DURATION)
        clips.append(clip)
        downloaded += 1

    if downloaded == 0:
        raise ValueError("No product images could be downloaded.")

    # Concatenate with crossfade
    final = concatenate_videoclips(clips, method="compose", padding=-TRANSITION_DURATION)
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio=False,
        preset="ultrafast",
        logger=None,
    )
    return output_path
