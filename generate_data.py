"""
Synthetic data generator for transformers-ocr
==============================================
Generates 20,000 labeled images of numeric meter readings.

Styles:
  - LCD/meter  : dark background, bright green/amber/white digits
  - Printed    : white/off-white background, dark digits

Number formats:
  - Integer    : e.g. 12345
  - Decimal    : e.g. 123.45

Output layout:
  data/
  ├── train/          16000 images
  ├── val/             2000 images
  ├── test/            2000 images
  ├── train_labels.txt
  ├── val_labels.txt
  └── test_labels.txt

Usage (PowerShell):
  python generate_data.py
  python generate_data.py --total 20000 --out data
  python generate_data.py --total 5000  --out data --seed 99
"""

import argparse
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  Font download  (free, no licence issues)
# ──────────────────────────────────────────────────────────────────────────────
FONT_URLS = {
    "dseg7":    "https://github.com/keshikan/DSEG/releases/download/v0.46/fonts-DSEG_v046.zip",
    "cousine":  "https://github.com/google/fonts/raw/main/apache/cousine/Cousine-Regular.ttf",
    "roboto":   "https://github.com/google/fonts/raw/main/apache/robotomono/static/RobotoMono-Regular.ttf",
}

FONT_CACHE = Path("assets/fonts")


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    print("done")


def _get_fonts(sizes: list[int]) -> list[ImageFont.FreeTypeFont]:
    """Download fonts if needed and return a list of ImageFont objects."""
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    fonts_found: list[Path] = []

    # Cousine (monospace, clean)
    cousine = FONT_CACHE / "Cousine-Regular.ttf"
    if not cousine.exists():
        _download_file(FONT_URLS["cousine"], cousine)
    fonts_found.append(cousine)

    # RobotoMono
    roboto = FONT_CACHE / "RobotoMono-Regular.ttf"
    if not roboto.exists():
        _download_file(FONT_URLS["roboto"], roboto)
    fonts_found.append(roboto)

    # DSEG7 (7-segment LCD look) — packaged as zip
    dseg = FONT_CACHE / "DSEG7Classic-Regular.ttf"
    if not dseg.exists():
        zip_path = FONT_CACHE / "dseg.zip"
        _download_file(FONT_URLS["dseg7"], zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if "DSEG7Classic-Regular.ttf" in name:
                    data = zf.read(name)
                    dseg.write_bytes(data)
                    break
        zip_path.unlink(missing_ok=True)
    if dseg.exists():
        fonts_found.append(dseg)

    loaded: list[ImageFont.FreeTypeFont] = []
    for path in fonts_found:
        for size in sizes:
            try:
                loaded.append(ImageFont.truetype(str(path), size))
            except Exception:
                pass

    if not loaded:
        # Last resort: PIL default bitmap font (ugly but functional)
        loaded = [ImageFont.load_default()]

    return loaded


# ──────────────────────────────────────────────────────────────────────────────
#  Number generators
# ──────────────────────────────────────────────────────────────────────────────
def _random_number(rng: random.Random) -> str:
    use_decimal = rng.random() < 0.5
    int_digits  = rng.randint(4, 8)
    int_part    = rng.randint(10 ** (int_digits - 1), 10 ** int_digits - 1)
    if use_decimal:
        dec_digits = rng.randint(1, 2)
        dec_part   = rng.randint(0, 10 ** dec_digits - 1)
        return f"{int_part}.{dec_part:0{dec_digits}d}"
    return str(int_part)


# ──────────────────────────────────────────────────────────────────────────────
#  Image generators
# ──────────────────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 48, 320


def _lcd_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    rng: random.Random,
) -> Image.Image:
    """Dark background with bright digit colour — simulates LCD/meter display."""

    # Background: very dark green, dark grey, or near-black
    bg_choices = [
        (10, 20, 10),   # dark green tint
        (15, 15, 15),   # near-black
        (5,  15, 25),   # dark blue tint
        (20, 10, 5),    # dark amber tint
    ]
    bg = rng.choice(bg_choices)
    # Add slight noise to bg
    bg = tuple(max(0, min(255, c + rng.randint(-5, 5))) for c in bg)

    img  = Image.new("RGB", (IMG_W, IMG_H), bg)
    draw = ImageDraw.Draw(img)

    # Digit colour: bright green, amber, white, cyan
    fg_choices = [
        (0,   255, 80),   # green
        (255, 180, 0),    # amber
        (220, 220, 220),  # cool white
        (0,   230, 230),  # cyan
    ]
    fg = rng.choice(fg_choices)
    # Slight per-sample jitter
    fg = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in fg)

    # Centre text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(2, (IMG_W - tw) // 2 + rng.randint(-10, 10))
    y = max(0, (IMG_H - th) // 2 + rng.randint(-3, 3))
    draw.text((x, y), text, font=font, fill=fg)

    # Optional: faint scanlines
    if rng.random() < 0.4:
        for row in range(0, IMG_H, 3):
            for col in range(IMG_W):
                px = img.getpixel((col, row))
                img.putpixel((col, row), tuple(max(0, c - 20) for c in px))

    return img


def _printed_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    rng: random.Random,
) -> Image.Image:
    """Light background with dark digits — simulates printed label."""

    bg_value = rng.randint(230, 255)
    bg_tint  = (
        bg_value + rng.randint(-5, 5),
        bg_value + rng.randint(-5, 5),
        bg_value + rng.randint(-5, 5),
    )
    bg_tint = tuple(max(200, min(255, c)) for c in bg_tint)

    img  = Image.new("RGB", (IMG_W, IMG_H), bg_tint)
    draw = ImageDraw.Draw(img)

    fg_value = rng.randint(0, 60)
    fg = (fg_value, fg_value, fg_value)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(2, (IMG_W - tw) // 2 + rng.randint(-10, 10))
    y = max(0, (IMG_H - th) // 2 + rng.randint(-3, 3))
    draw.text((x, y), text, font=font, fill=fg)

    return img


def generate_image(
    text: str,
    fonts: list[ImageFont.FreeTypeFont],
    rng: random.Random,
    style: str,  # "lcd" | "printed"
) -> Image.Image:
    font = rng.choice(fonts)
    if style == "lcd":
        return _lcd_image(text, font, rng)
    return _printed_image(text, font, rng)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic meter OCR data")
    parser.add_argument("--total",  type=int,   default=20_000, help="Total images to generate")
    parser.add_argument("--out",    type=str,   default="data", help="Output root directory")
    parser.add_argument("--seed",   type=int,   default=42,     help="Random seed")
    parser.add_argument("--val",    type=float, default=0.10,   help="Val fraction")
    parser.add_argument("--test",   type=float, default=0.10,   help="Test fraction")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── Font sizes that fit in 48px height ────────────────────────────────────
    print("Setting up fonts...")
    fonts = _get_fonts(sizes=[28, 30, 32, 34])
    print(f"  {len(fonts)} font variants loaded")

    # ── Split sizes ───────────────────────────────────────────────────────────
    n_val   = int(args.total * args.val)
    n_test  = int(args.total * args.test)
    n_train = args.total - n_val - n_test
    splits  = [("train", n_train), ("val", n_val), ("test", n_test)]

    out = Path(args.out)
    for split, _ in splits:
        (out / split).mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.total} images  "
          f"(train={n_train}, val={n_val}, test={n_test})\n")

    idx = 0
    for split, count in splits:
        label_lines: list[str] = []
        for i in tqdm(range(count), desc=f"{split:>5}", unit="img"):
            text  = _random_number(rng)
            style = "lcd" if (idx % 2 == 0) else "printed"   # strict 50/50
            img   = generate_image(text, fonts, rng, style)

            fname = f"{split}_{idx:06d}.jpg"
            img.save(out / split / fname, "JPEG", quality=92)
            label_lines.append(f"{fname} {text}")
            idx += 1

        label_path = out / f"{split}_labels.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        print(f"  Saved {count} images + labels → {label_path}")

    print(f"\nDone. Total images: {idx}")
    print(f"Output root: {out.resolve()}")


if __name__ == "__main__":
    main()
