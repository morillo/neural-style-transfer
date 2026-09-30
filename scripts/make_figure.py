"""Build the README results grid: style image, then originals above stylised outputs.

nst -s samples/style/starry_night.jpg samples/content -o outputs --size 768
python scripts/make_figure.py outputs docs/images/results.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ROW_HEIGHT = 300
GAP = 12
LABEL = 34


def fit(image: Image.Image, height: int) -> Image.Image:
    """Resize ``image`` to ``height`` pixels tall, keeping its aspect ratio."""
    return image.convert("RGB").resize((round(image.width * height / image.height), height), Image.Resampling.LANCZOS)


def font(size: int) -> ImageFont.ImageFont:
    """Load a common system font, falling back to Pillow's built-in one."""
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(stylized_dir: str, output: str) -> None:
    """Lay out the style image, the sample photos and their stylised versions in one image."""
    contents = sorted((ROOT / "samples" / "content").glob("*.jpg"))
    pairs = [
        (fit(Image.open(c), ROW_HEIGHT), fit(Image.open(Path(stylized_dir) / f"{c.stem}_stylized.png"), ROW_HEIGHT))
        for c in contents
    ]
    style = fit(Image.open(ROOT / "samples" / "style" / "starry_night.jpg"), 2 * ROW_HEIGHT + GAP)

    width = style.width + GAP + sum(p[0].width for p in pairs) + GAP * len(pairs) + GAP
    height = LABEL + 2 * ROW_HEIGHT + GAP + GAP
    canvas = Image.new("RGB", (width + GAP, height), "white")
    draw = ImageDraw.Draw(canvas)
    text = font(22)

    x = GAP
    draw.text((x, 6), "Style", fill="black", font=text)
    canvas.paste(style, (x, LABEL))
    x += style.width + 2 * GAP
    draw.text((x, 6), "Content (top) and stylised output (bottom)", fill="black", font=text)
    for original, result in pairs:
        canvas.paste(original, (x, LABEL))
        canvas.paste(result, (x, LABEL + ROW_HEIGHT + GAP))
        x += original.width + GAP

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88)
    print(f"Wrote {output} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
