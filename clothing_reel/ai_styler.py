import io
import base64
import requests
from PIL import Image

VIBE_PROMPTS = {
    "luxury": "luxury high-end fashion editorial photography, professional studio lighting, elegant dark marble backdrop, Vogue magazine quality, sophisticated and glamorous",
    "retro": "retro 1970s fashion editorial, warm analog film grain, vintage faded color palette, nostalgic 70s photography aesthetic, funky",
    "vintage": "vintage 1950s old Hollywood fashion photography, classic timeless glamour, muted warm tones, elegant cinematic quality",
    "streetwear": "urban streetwear fashion editorial, gritty city street background, dynamic natural lighting, bold modern youth culture, edgy",
    "minimalist": "minimalist fashion editorial, clean white seamless background, simple elegant composition, airy contemporary modern style",
}


def restyle_image(image_path: str, vibe: str) -> str:
    """Restyle a product image with the given vibe using Replicate Flux.
    Returns the path to the restyled image (falls back to original on error).
    """
    import replicate

    vibe_desc = VIBE_PROMPTS.get(vibe.lower(), VIBE_PROMPTS["luxury"])
    prompt = (
        f"fashion model wearing stylish clothing, {vibe_desc}, "
        "photorealistic, high resolution, full body or three-quarter shot"
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    img = Image.open(io.BytesIO(image_bytes))
    fmt = (img.format or "JPEG").lower()
    if fmt == "jpg":
        fmt = "jpeg"

    image_b64 = base64.b64encode(image_bytes).decode()

    output = replicate.run(
        "black-forest-labs/flux-dev",
        input={
            "prompt": prompt,
            "image": f"data:image/{fmt};base64,{image_b64}",
            "prompt_strength": 0.65,
            "num_outputs": 1,
            "aspect_ratio": "9:16",
            "output_format": "jpg",
            "output_quality": 90,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
        },
    )

    result = output[0] if isinstance(output, list) else output
    url = str(result)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    from pathlib import Path

    orig = Path(image_path)
    out_path = str(orig.parent / f"restyled_{orig.stem}.jpg")
    with open(out_path, "wb") as f:
        f.write(resp.content)

    return out_path
