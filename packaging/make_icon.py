"""Generate a 1024x1024 app icon PNG (rounded square, purple gradient, "V").

Run ephemerally with Pillow, no permanent project dependency:
    uv run --with pillow python packaging/make_icon.py packaging/build/icon.png
"""

from __future__ import annotations

import sys

from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
TOP = (124, 92, 255)     # #7c5cff
BOTTOM = (74, 42, 200)   # deeper violet
RADIUS = 224


def _gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    base = Image.new("RGB", (size, size), top)
    px = base.load()
    for y in range(size):
        t = y / (size - 1)
        r = round(top[0] + (bottom[0] - top[0]) * t)
        g = round(top[1] + (bottom[1] - top[1]) * t)
        b = round(top[2] + (bottom[2] - top[2]) * t)
        for x in range(size):
            px[x, y] = (r, g, b)
    return base


def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "icon.png"

    icon = _gradient(SIZE, TOP, BOTTOM).convert("RGBA")
    icon.putalpha(_rounded_mask(SIZE, RADIUS))

    draw = ImageDraw.Draw(icon)
    font = _load_font(560)
    box = draw.textbbox((0, 0), "V", font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    pos = ((SIZE - w) / 2 - box[0], (SIZE - h) / 2 - box[1] - 20)
    draw.text(pos, "V", font=font, fill=(255, 255, 255, 235))

    icon.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
