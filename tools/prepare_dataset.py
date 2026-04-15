"""
tools/prepare_dataset.py
------------------------
Prepare a raw image + label dump into the folder structure expected by
configs/config.py and engine/dataset.py.

Expected input layout (two accepted formats):

  Format A – labels.txt beside images
  ─────────────────────────────────────
  source_dir/
      0001.jpg        # image
      0002.png
      labels.txt      # one line per image:  <filename> <label>
                      # e.g.  0001.jpg  12345.6

  Format B – label embedded in filename  (label_filename.jpg  OR  filename_label.jpg)
  ────────────────────────────────────────────────────────────────────────────────────
  source_dir/
      12345.6_0001.jpg
      12345.6_0002.jpg

Output layout (mirrors config.py):
  output_dir/
      train/  val/  test/          ← copied images
      train_labels.txt
      val_labels.txt
      test_labels.txt

Usage examples
--------------
# Format A  (labels.txt present in source folder)
python tools/prepare_dataset.py --src data/raw --out data

# Format B  (label is the part before the first underscore in filename)
python tools/prepare_dataset.py --src data/raw --out data --label-from-name prefix

# Custom split ratios
python tools/prepare_dataset.py --src data/raw --out data --train 0.8 --val 0.1 --test 0.1

# Dry run – print stats without copying anything
python tools/prepare_dataset.py --src data/raw --out data --dry-run

# Validate that every label matches the lexicon defined in config.py
python tools/prepare_dataset.py --src data/raw --out data --validate-lexicon
"""

import argparse
import os
import re
import shutil
import sys
import random
from collections import Counter
from pathlib import Path

# ── image extensions considered valid ────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── default lexicon (matches config.py BASE_CFG) ─────────────────────────────
DEFAULT_LEXICON = r"^\d{4,8}(\.\d{1,2})?$"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def find_images(src: Path) -> list[Path]:
    """Return all image files directly inside src (non-recursive by default)."""
    return sorted(p for p in src.iterdir() if p.is_file() and is_image(p))


def load_labels_from_file(src: Path, label_file: str) -> dict[str, str]:
    """Parse a labels.txt → {filename: label}."""
    lf = src / label_file
    if not lf.exists():
        sys.exit(f"[ERROR] Label file not found: {lf}")

    mapping: dict[str, str] = {}
    bad_lines = []
    with open(lf) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                bad_lines.append((i, line))
                continue
            filename, label = parts[0], parts[1]
            mapping[filename] = label

    if bad_lines:
        print(f"[WARN] {len(bad_lines)} malformed line(s) in {lf.name} – skipped:")
        for ln, txt in bad_lines[:5]:
            print(f"       line {ln}: {txt!r}")
        if len(bad_lines) > 5:
            print(f"       … and {len(bad_lines) - 5} more")

    return mapping


def label_from_filename(path: Path, strategy: str) -> str | None:
    """
    Extract a label from the image filename.

    strategy='prefix'  →  <label>_<rest>.ext   e.g. 12345.6_001.jpg  → 12345.6
    strategy='suffix'  →  <rest>_<label>.ext   e.g. 001_12345.6.jpg  → 12345.6
    strategy='stem'    →  entire stem is the label  e.g. 12345.6.jpg  → 12345.6
    """
    stem = path.stem
    if strategy == "prefix":
        return stem.split("_")[0] if "_" in stem else None
    if strategy == "suffix":
        return stem.rsplit("_", 1)[-1] if "_" in stem else None
    if strategy == "stem":
        return stem
    return None


def split_samples(
    samples: list[tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]
    return train, val, test


def write_split(
    samples: list[tuple[str, str]],
    img_dir: Path,
    src: Path,
    label_path: Path,
    dry_run: bool,
) -> None:
    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for fname, label in samples:
        lines.append(f"{fname} {label}\n")
        if not dry_run:
            shutil.copy2(src / fname, img_dir / fname)

    if not dry_run:
        label_path.write_text("".join(lines))


def validate_labels(
    samples: list[tuple[str, str]],
    lexicon: str,
) -> tuple[list, list]:
    pattern = re.compile(lexicon)
    ok, bad = [], []
    for fname, label in samples:
        (ok if pattern.match(label) else bad).append((fname, label))
    return ok, bad


def print_split_stats(name: str, samples: list[tuple[str, str]]) -> None:
    labels = [s[1] for s in samples]
    label_counts = Counter(labels)
    print(f"  {name:6s}  {len(samples):>6} samples  "
          f"  unique labels: {len(label_counts)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare dataset for transformers-ocr training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── input / output ────────────────────────────────────────────────────────
    p.add_argument(
        "--src", required=True, metavar="DIR",
        help="Source folder containing images (and optionally a labels.txt).",
    )
    p.add_argument(
        "--out", default="data", metavar="DIR",
        help="Output root directory (default: data). "
             "Will contain train/, val/, test/ and the label .txt files.",
    )

    # ── label sourcing ────────────────────────────────────────────────────────
    p.add_argument(
        "--label-file", default="labels.txt", metavar="FILE",
        help="Name of the label file inside --src (default: labels.txt). "
             "Ignored when --label-from-name is set.",
    )
    p.add_argument(
        "--label-from-name", metavar="STRATEGY",
        choices=["prefix", "suffix", "stem"],
        help="Derive labels from image filenames instead of a label file. "
             "  prefix → <label>_<rest>.ext  "
             "  suffix → <rest>_<label>.ext  "
             "  stem   → entire filename stem",
    )

    # ── split ratios ──────────────────────────────────────────────────────────
    p.add_argument("--train", type=float, default=0.8,
                   metavar="RATIO", help="Train split ratio (default: 0.8)")
    p.add_argument("--val",   type=float, default=0.1,
                   metavar="RATIO", help="Val split ratio   (default: 0.1)")
    p.add_argument("--test",  type=float, default=0.1,
                   metavar="RATIO", help="Test split ratio  (default: 0.1)")

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for shuffling (default: 42).")
    p.add_argument("--copy-mode", choices=["copy", "move", "symlink"],
                   default="copy",
                   help="How to transfer images to output folders (default: copy).")
    p.add_argument("--validate-lexicon", action="store_true",
                   help="Check every label against the lexicon regex and "
                        "print / discard non-matching ones.")
    p.add_argument("--lexicon", default=DEFAULT_LEXICON, metavar="REGEX",
                   help=f"Lexicon regex used with --validate-lexicon "
                        f"(default: {DEFAULT_LEXICON!r}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print statistics without writing any files.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()

    # ── validate ratios ───────────────────────────────────────────────────────
    total = args.train + args.val + args.test
    if not (0.99 < total < 1.01):
        sys.exit(
            f"[ERROR] --train + --val + --test must sum to 1.0 "
            f"(got {total:.4f})"
        )

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if not src.exists():
        sys.exit(f"[ERROR] Source directory does not exist: {src}")

    # ── discover images ───────────────────────────────────────────────────────
    images = find_images(src)
    if not images:
        sys.exit(f"[ERROR] No image files found in {src}")

    print(f"\n  transformers-ocr  |  prepare_dataset")
    print(f"  source : {src}")
    print(f"  output : {out}")
    print(f"  images found : {len(images)}\n")

    # ── build (filename, label) pairs ─────────────────────────────────────────
    samples: list[tuple[str, str]] = []

    if args.label_from_name:
        # ── Format B: label from filename ─────────────────────────────────────
        missing = []
        for img in images:
            label = label_from_filename(img, args.label_from_name)
            if label is None:
                missing.append(img.name)
            else:
                samples.append((img.name, label))

        if missing:
            print(f"[WARN] Could not extract label from {len(missing)} filename(s) – skipped:")
            for m in missing[:10]:
                print(f"       {m}")
            if len(missing) > 10:
                print(f"       … and {len(missing) - 10} more")

    else:
        # ── Format A: labels.txt ──────────────────────────────────────────────
        label_map = load_labels_from_file(src, args.label_file)

        no_label = []
        for img in images:
            if img.name in label_map:
                samples.append((img.name, label_map[img.name]))
            else:
                no_label.append(img.name)

        if no_label:
            print(
                f"[WARN] {len(no_label)} image(s) have no entry in "
                f"{args.label_file} – skipped:"
            )
            for m in no_label[:10]:
                print(f"       {m}")
            if len(no_label) > 10:
                print(f"       … and {len(no_label) - 10} more")

        # warn about label entries with no corresponding image
        img_names = {img.name for img in images}
        phantom = [fn for fn in label_map if fn not in img_names]
        if phantom:
            print(
                f"[WARN] {len(phantom)} label entry/entries reference "
                f"non-existent image(s) – ignored."
            )

    if not samples:
        sys.exit("[ERROR] No valid (image, label) pairs found. Aborting.")

    # ── optional lexicon validation ───────────────────────────────────────────
    if args.validate_lexicon:
        ok, bad = validate_labels(samples, args.lexicon)
        if bad:
            print(
                f"\n[VALIDATE] {len(bad)} label(s) do NOT match lexicon "
                f"{args.lexicon!r} – these will be excluded:\n"
            )
            for fname, label in bad[:20]:
                print(f"           {fname}  →  {label!r}")
            if len(bad) > 20:
                print(f"           … and {len(bad) - 20} more")
            samples = ok
            print(f"\n[VALIDATE] Keeping {len(samples)} valid samples.\n")
        else:
            print(
                f"[VALIDATE] All {len(samples)} labels match lexicon {args.lexicon!r}  ✓\n"
            )

    # ── split ─────────────────────────────────────────────────────────────────
    train_s, val_s, test_s = split_samples(
        samples, args.train, args.val, args.seed
    )

    print("  Split summary")
    print("  " + "─" * 50)
    print_split_stats("train", train_s)
    print_split_stats("val",   val_s)
    print_split_stats("test",  test_s)
    print(f"  {'total':6s}  {len(samples):>6} samples")
    print("  " + "─" * 50)

    if args.dry_run:
        print("\n  [DRY RUN] No files written.\n")
        return

    # ── write output ──────────────────────────────────────────────────────────
    # Override shutil.copy2 if move/symlink requested
    transfer = shutil.copy2
    if args.copy_mode == "move":
        transfer = shutil.move
    elif args.copy_mode == "symlink":
        def transfer(src_path, dst_path):  # type: ignore[misc]
            os.symlink(src_path, dst_path)

    splits = [
        ("train", train_s, out / "train",  out / "train_labels.txt"),
        ("val",   val_s,   out / "val",    out / "val_labels.txt"),
        ("test",  test_s,  out / "test",   out / "test_labels.txt"),
    ]

    for name, split_samples_, img_dir, label_path in splits:
        img_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for fname, label in split_samples_:
            dst = img_dir / fname
            if not dst.exists():           # avoid re-copying on re-runs
                transfer(str(src / fname), str(dst))
            lines.append(f"{fname} {label}\n")
        label_path.write_text("".join(lines))
        print(f"  wrote  {label_path.relative_to(out.parent)}"
              f"  +  {img_dir.relative_to(out.parent)}/  ({len(split_samples_)} files)")

    print(f"\n  Done! Dataset ready at: {out}\n")

    # ── print the config snippet ──────────────────────────────────────────────
    rel = out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out
    print("  ── Paste into configs/config.py BASE_CFG ────────────────────")
    print(f'    "train_dir":    "{rel}/train",')
    print(f'    "val_dir":      "{rel}/val",')
    print(f'    "test_dir":     "{rel}/test",')
    print(f'    "train_labels": "{rel}/train_labels.txt",')
    print(f'    "val_labels":   "{rel}/val_labels.txt",')
    print(f'    "test_labels":  "{rel}/test_labels.txt",')
    print("  ─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
