# transformers-ocr

SVTR-based OCR for meter reading. Reads numeric values (e.g. `12345.67`) from meter images using a Scene Text Vision Transformer with CTC decoding.

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.2+, CUDA 12.1.

## Project structure

```
transformers-ocr/
├── configs/
│   └── config.py          # tiny / small / base variants
├── engine/
│   ├── augment.py
│   ├── codec.py
│   ├── dataset.py
│   ├── loss.py
│   └── preprocess.py
├── models/
│   └── svtr.py
├── tools/
│   └── export_onnx.py
├── notebooks/
│   └── colab_train.ipynb
├── data/                  # put your dataset here (gitignored)
├── weights/               # saved checkpoints (gitignored)
├── exports/               # onnx exports (gitignored)
├── logs/                  # csv logs + hard negatives (gitignored)
├── train.py
├── predict.py
└── requirements.txt
```

## Dataset format

```
data/
├── train/         image files
├── val/           image files
├── test/          image files
├── train_labels.txt
├── val_labels.txt
└── test_labels.txt
```

Each label file is whitespace-separated: `filename.jpg 12345.6`

## Training

```bash
python train.py --variant tiny --data data --device cuda
python train.py --variant small --data data --device cuda
python train.py --variant base  --data data --device cuda
```

Resume from checkpoint:

```bash
python train.py --variant tiny --weights weights/tiny_best.pth --device cuda
```

Override epoch count:

```bash
python train.py --variant tiny --epochs 50
```

Variants:

| Variant | Batch | LR    | Params (approx) |
|---------|-------|-------|-----------------|
| tiny    | 128   | 3e-4  | ~5M             |
| small   | 96    | 2e-4  | ~12M            |
| base    | 64    | 1e-4  | ~25M            |

Training writes:
- `weights/<variant>_best.pth` — best checkpoint by val exact-match accuracy
- `logs/<variant>_train.csv` — epoch-level metrics
- `logs/<variant>_hard_negs_ep<N>.txt` — mismatched samples at each best epoch

## Inference

Single image:

```bash
python predict.py --weights weights/tiny_best.pth --img path/to/meter.jpg
```

Folder of images:

```bash
python predict.py --weights weights/tiny_best.pth --img path/to/folder/
```

With 5× TTA (brightness/contrast sweep + majority vote):

```bash
python predict.py --weights weights/tiny_best.pth --img path/to/folder/ --tta
```

Output columns: `Image | Pred | Conf | Flag`

`LOW_CONF` is printed when confidence is below `conf_threshold` (default 0.85).

## Export to ONNX

```bash
python tools/export_onnx.py --weights weights/tiny_best.pth --variant tiny
python tools/export_onnx.py --weights weights/tiny_best.pth --variant tiny --quantize
```

`--quantize` produces an additional `_int8.onnx` file and prints FP32 vs INT8 size.

Output: `exports/transformers_ocr_<variant>.onnx`

## Colab training

Open `notebooks/colab_train.ipynb`. Set `VARIANT`, `REPO_URL`, and `DATA_ZIP` in the first cell, then run all. Weights and the ONNX export are saved back to your Google Drive automatically.

## Config

Edit `configs/config.py` to change:

- `CHARS` — character set (default `0123456789.`)
- `img_h`, `img_w` — input resolution (default `48×320`)
- `lexicon_re` — regex for beam search lexicon filter (default: 4–8 digits, optional `.XX`)
- `conf_threshold` — low-confidence flag threshold
- `epochs`, `patience`, `lr`, `weight_decay`, `grad_clip`

## Decoder

The codec supports two decoding modes:

- **Greedy** — `encoder.decode_greedy(seq)` — fastest, used internally for quick checks
- **Beam search** — `encoder.decode_beam(log_probs, beam_width, lexicon_pattern)` — used at eval/inference; lexicon filter keeps only results matching `lexicon_re`

## Preprocessing pipeline

Per image at load time: CLAHE on HSV V-channel → bilateral filter → deskew (Hough lines) → unsharp mask if blur score is below threshold.

## Loss

`FocalCTCLoss` — CTC with label smoothing and focal weighting `(1-p)^γ` per sample. When `use_sgm=True` an auxiliary semantic guidance branch adds `0.1 × CTC(sgm_out)` to the loss.
