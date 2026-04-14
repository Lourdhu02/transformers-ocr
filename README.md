<div align="center">

<img src="https://img.shields.io/badge/transformers--ocr-v1.0.0-0d1117?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" alt="version"/>

# Transformer OCR for Digit Recognition using SVTR

**A high-accuracy Transformer-based OCR framework for digit recognition using SVTR, PyTorch, CTC loss, beam search decoding, and advanced augmentation techniques.**

Scene Text Vision Transformer · CTC · Beam Search · ONNX · AMP · TTA

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Lourdhu02/transformers-ocr/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/Lourdhu02/transformers-ocr/actions)
[![ONNX](https://img.shields.io/badge/ONNX-export-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

</div>

---

## Overview

**transformers-ocr** is a clean, fully-featured OCR pipeline specialised for reading numeric values (e.g. `12345.67`) from numbers images in challenging real-world conditions — motion blur, overexposure, grime, and low contrast.

The backbone is [SVTR (Scene Text Vision Transformer)](https://arxiv.org/abs/2205.00159), extended with:

- **Feature Rearrangement Module (FRM)** — global context pooling before the CTC head
- **Semantic Guidance Module (SGM)** — auxiliary CTC loss for richer supervision
- **FocalCTCLoss** — focal weighting + label smoothing for hard-sample mining
- **CutMix** — width-axis image mixing augmentation
- **Beam Search** with lexicon regex filter and optional C++ `ctcdecode` fast path
- **5× TTA** — brightness/contrast sweep with majority vote at inference
- **ONNX export** + optional INT8 dynamic quantization

Supports `tiny / small / base` variants (~5M / ~12M / ~25M paranumberss).

---

## Architecture

```
Input image  (B, 3, H, W)
      │
      ▼
Patch Embed   Conv2d ×2  stride-4  →  (B, C₁, H/4, W/4)
      │
      ▼
Stage 1       SVTRBlock ×d₁          mixing attention + depthwise conv
      │
Merge 1       Conv2d stride-(2,1)    height halved  →  C₂ channels
      │
      ▼
Stage 2       SVTRBlock ×d₂
      │
Merge 2       Conv2d stride-(2,1)    height halved  →  C₃ channels
      │
      ▼
Stage 3       SVTRBlock ×d₃
      │
LayerNorm + FRM (optional)
      │
AdaptiveAvgPool2d((1, W))            collapse height  →  (B, C₃, W)
      │
      ├──► FC → log_softmax          main CTC head  →  (B, W, num_classes)
      │
      └──► SGM (optional)            auxiliary CTC head (same T=W)
```

Each `SVTRBlock` fuses **global self-attention** (scaled dot-product) with a **depthwise local Conv1d** and a standard MLP with GELU activation.

---

## Variants

| Variant | Dims        | Depths    | Heads     | Params  | Batch | LR    |
|---------|-------------|-----------|-----------|---------|-------|-------|
| `tiny`  | 64/128/256  | 3/6/3     | 2/4/8     | ~5 M    | 128   | 3e-4  |
| `small` | 96/192/384  | 3/6/6     | 3/6/12    | ~12 M   | 96    | 2e-4  |
| `base`  | 128/256/512 | 3/6/9     | 4/8/16    | ~25 M   | 32    | 1e-4  |

---

## Requirements

```bash
pip install -r requirements.txt
```

| Package        | Version  |
|----------------|----------|
| torch          | ≥ 2.2.0  |
| torchvision    | ≥ 0.17.0 |
| albumentations | ≥ 1.4.0  |
| opencv-python  | ≥ 4.9.0  |
| numpy          | ≥ 1.26.0 |
| editdistance   | ≥ 0.8.1  |
| timm           | ≥ 0.9.16 |
| onnx           | ≥ 1.16.0 |
| onnxruntime    | ≥ 1.18.0 |
| tqdm           | ≥ 4.66.0 |

Optional — ~50× faster beam search:

```bash
pip install ctcdecode
```

Tested with Python 3.10+, PyTorch 2.2+, CUDA 12.1.

---

## Project Structure

```
transformers-ocr/
├── configs/
│   └── config.py               # tiny / small / base variant configs
├── engine/
│   ├── augment.py              # Albumentations pipelines + CutMix
│   ├── codec.py                # CTC encoder / greedy & beam decoder
│   ├── dataset.py              # numbersDataset + DataLoader factory
│   ├── loss.py                 # FocalCTCLoss
│   └── preprocess.py           # CLAHE → bilateral → deskew → unsharp
├── models/
│   └── svtr.py                 # SVTRNet + FRM + SGM
├── tools/
│   └── export_onnx.py          # FP32 + INT8 ONNX export
├── tests/
│   └── test_core.py            # Pytest unit tests (CPU)
├── data/                       # Dataset root (gitignored)
├── weights/                    # Saved checkpoints (gitignored)
├── exports/                    # ONNX exports (gitignored)
├── logs/                       # CSV logs + hard negatives (gitignored)
├── train.py
├── predict.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Dataset Format

```
data/
├── train/              image files
├── val/                image files
├── test/               image files
├── train_labels.txt
├── val_labels.txt
└── test_labels.txt
```

Each label file uses whitespace-separated `filename value` pairs, one per line:

```
numbers_00001.jpg 12345.6
numbers_00002.jpg 00987.3
```

The default character set is `0123456789.`. Change `CHARS` in `configs/config.py` to adapt to your use case.

---

## Training

```bash
# Tiny (~5M params) — good starting point
python train.py --variant tiny --data data --device cuda

# Small (~12M params) — higher accuracy
python train.py --variant small --data data --device cuda

# Base (~25M params) — maximum accuracy
python train.py --variant base  --data data --device cuda
```

Resume from a checkpoint:

```bash
python train.py --variant tiny --weights weights/tiny_best.pth --device cuda
```

Override epoch count:

```bash
python train.py --variant tiny --epochs 50
```

Training outputs:

| Output | Description |
|--------|-------------|
| `weights/<variant>_best.pth` | Best checkpoint by val exact-match accuracy |
| `logs/<variant>_train.csv` | Epoch-level metrics |
| `logs/<variant>_hard_negs_ep<N>.txt` | Mismatched samples at each best epoch |

The progress bar reports batch loss, running average, current exact-match accuracy, and character error rate in real time.

---

## Inference

Single image:

```bash
python predict.py --weights weights/tiny_best.pth --img path/to/numbers.jpg
```

Folder of images:

```bash
python predict.py --weights weights/tiny_best.pth --img path/to/folder/
```

With 5× TTA (brightness/contrast sweep + majority vote):

```bash
python predict.py --weights weights/tiny_best.pth --img path/to/folder/ --tta
```

Example output:

```
Image                                    Pred           Conf    Flag
--------------------------------------------------------------------
numbers_0001.jpg                           12345.6        0.9872
numbers_0002.jpg                           00987.3        0.9541
numbers_0003.jpg                           84201.0        0.7612  LOW_CONF
```

`LOW_CONF` is printed when confidence is below `conf_threshold` (default `0.85`).

---

## ONNX Export

```bash
# FP32 export
python tools/export_onnx.py --weights weights/tiny_best.pth --variant tiny

# FP32 + INT8 dynamic quantization
python tools/export_onnx.py --weights weights/tiny_best.pth --variant tiny --quantize
```

Output: `exports/transformers_ocr_tiny.onnx` (and `_int8.onnx` with `--quantize`).

Typical size reduction: **~3.5×** FP32 → INT8.

---

## Configuration

Edit `configs/config.py` to change global defaults or per-variant overrides.

| Key | Default | Description |
|-----|---------|-------------|
| `CHARS` | `0123456789.` | Character set |
| `img_h` / `img_w` | `48 / 320` | Input resolution |
| `epochs` | `100` | Max training epochs |
| `patience` | `25` | Early stopping patience |
| `lr` | `3e-4` | Peak learning rate |
| `warmup_epochs` | `5` | Linear LR warmup length |
| `grad_clip` | `5.0` | Gradient norm clip |
| `focal_gamma` | `2.0` | Focal loss exponent |
| `label_smooth` | `0.1` | CTC label smoothing |
| `cutmix_alpha` | `0.5` | CutMix Beta distribution alpha |
| `beam_width` | `10` | Beam search width |
| `lexicon_re` | `^\d{4,8}(\.\d{1,2})?$` | Beam lexicon filter regex |
| `conf_threshold` | `0.85` | Low-confidence flag threshold |
| `use_frm` | `True` | Enable Feature Rearrangement Module |
| `use_sgm` | `True` | Enable Semantic Guidance Module |

---

## Preprocessing Pipeline

Applied per image at load time (training and inference):

```
Raw BGR image
      │
      ▼
CLAHE on HSV V-channel         contrast normalisation
      │
      ▼
Bilateral filter               edge-preserving noise reduction
      │
      ▼
Deskew (Hough lines)           rotation correction up to ±15°
      │
      ▼
Unsharp mask (if blur < 80)    sharpening for out-of-focus images
      │
      ▼
Reconstructed BGR
```

---

## Loss Function

`FocalCTCLoss` combines three techniques:

1. **CTC** — `nn.CTCLoss` with `zero_infinity=True`
2. **Label Smoothing** — uniform smoothing at weight `label_smooth`
3. **Focal Weighting** — per-sample weight `(1 − pₜ)^γ` to down-weight easy samples

When `use_sgm=True`, the auxiliary SGM head adds `0.1 × FocalCTCLoss(sgm_out)` to the total loss during training only.

---

## Decoder

| Mode | Function | Speed | Notes |
|------|----------|-------|-------|
| Greedy | `encoder.decode_greedy(seq)` | Fastest | For quick internal checks |
| Beam (Python) | `encoder.decode_beam(log_probs, beam_width, lexicon_pattern)` | Moderate | Pure Python fallback |
| Beam (C++) | — auto-selected — | ~50× faster | Requires `pip install ctcdecode` |

The lexicon filter discards beam hypotheses not matching `lexicon_re` and falls back to the top beam if nothing matches.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests run on CPU and cover: encoder round-trips, greedy/beam decoding, FocalCTCLoss, forward passes for `nano` and `tiny` variants, SGM T-dimension regression, preprocess smoke test, and config key validation.

---

## Contributing

Pull requests are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
git clone https://github.com/Lourdhu02/transformers-ocr.git
cd transformers-ocr
pip install -r requirements-dev.txt
pre-commit install
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{lourdhu02_transformers_ocr_2026,
  author    = {Lourdhu02},
  title     = {transformers-ocr: SVTR-based OCR for numbers Reading},
  year      = {2026},
  url       = {https://github.com/Lourdhu02/transformers-ocr},
  license   = {MIT}
}
```

The SVTR architecture is based on:

```bibtex
@inproceedings{du2022svtr,
  title     = {SVTR: Scene Text Recognition with a Single Visual Model},
  author    = {Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Zheng, Tianlun and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  booktitle = {IJCAI},
  year      = {2022}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with precision for real-world numbers OCR.

[![Stars](https://img.shields.io/github/stars/Lourdhu02/transformers-ocr?style=social)](https://github.com/Lourdhu02/transformers-ocr/stargazers)
[![Forks](https://img.shields.io/github/forks/Lourdhu02/transformers-ocr?style=social)](https://github.com/Lourdhu02/transformers-ocr/network/members)

</div>
