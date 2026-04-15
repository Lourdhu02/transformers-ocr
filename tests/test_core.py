"""
Unit tests for transformers-ocr core components.
Run with: pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
#  Codec
# ──────────────────────────────────────────────────────────────
from engine.codec import Encoder

CHARS = "0123456789."


def test_encoder_encode_decode():
    enc = Encoder(CHARS)
    text = "12345.67"
    ids = enc.encode(text)
    assert len(ids) == len(text)
    decoded = enc.decode_greedy(ids)
    assert decoded == text


def test_encoder_unknown_char_warns():
    enc = Encoder(CHARS)
    with pytest.warns(UserWarning, match="unknown chars"):
        ids = enc.encode("12A3")
    assert len(ids) == 3  # 'A' is skipped


def test_encoder_blank():
    enc = Encoder(CHARS)
    assert enc.blank == 0
    assert enc.num_classes == len(CHARS) + 1


def test_decode_greedy_with_blanks():
    enc = Encoder(CHARS)
    # CTC raw sequence: 1, 1, 0, 2 → "01" (blank=0 collapses repeats; index 1='0', index 2='1')
    seq = [1, 1, 0, 2]
    result = enc.decode_greedy(seq)
    assert result == "01"


def test_decode_beam_pure_python():
    enc = Encoder(CHARS)
    T, V = 20, enc.num_classes
    # Peaked log-probs: mostly predict '1' (index 1)
    log_probs = np.full((T, V), -10.0)
    log_probs[:, 1] = 0.0  # log(1) ≈ 0 → strong signal for char '0'
    pred, score = enc.decode_beam(log_probs, beam_width=5, lexicon_pattern=None)
    assert isinstance(pred, str)
    assert isinstance(score, float)


# ──────────────────────────────────────────────────────────────
#  Loss
# ──────────────────────────────────────────────────────────────
from engine.loss import FocalCTCLoss


def test_focal_ctc_loss_forward():
    blank = 0
    num_classes = 12
    T, B = 80, 4
    loss_fn = FocalCTCLoss(blank=blank, gamma=2.0, label_smoothing=0.1)

    log_probs = torch.randn(T, B, num_classes).log_softmax(dim=-1)
    targets = torch.tensor([1, 2, 3, 1, 2, 1, 2, 3], dtype=torch.long)
    input_lengths = torch.full((B,), T, dtype=torch.long)
    target_lengths = torch.tensor([2, 2, 2, 2], dtype=torch.long)

    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


# ──────────────────────────────────────────────────────────────
#  Model (CPU, forward pass only)
# ──────────────────────────────────────────────────────────────
from models.svtr import build_model


@pytest.mark.parametrize("variant", ["nano", "tiny"])
def test_model_forward(variant):
    model = build_model(variant, img_h=48, img_w=320, num_classes=12,
                        use_frm=True, use_sgm=True)
    model.eval()
    x = torch.randn(2, 3, 48, 320)
    with torch.no_grad():
        out = model(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 12


def test_model_return_sgm():
    model = build_model("nano", img_h=48, img_w=320, num_classes=12,
                        use_frm=True, use_sgm=True)
    model.eval()
    x = torch.randn(2, 3, 48, 320)
    with torch.no_grad():
        logits, sgm = model(x, return_sgm=True)
    # Both heads must share the same T dimension
    assert logits.shape == sgm.shape


def test_model_sgm_t_matches_main():
    """Regression: SGM T must equal main-head T (was H*W before fix)."""
    model = build_model("nano", img_h=48, img_w=320, num_classes=12,
                        use_frm=True, use_sgm=True)
    model.eval()
    x = torch.randn(1, 3, 48, 320)
    with torch.no_grad():
        logits, sgm = model(x, return_sgm=True)
    assert logits.shape[1] == sgm.shape[1], (
        f"T mismatch: main={logits.shape[1]}, sgm={sgm.shape[1]}"
    )


def test_param_count():
    model = build_model("tiny", img_h=48, img_w=320, num_classes=12)
    assert model.param_count() > 0


# ──────────────────────────────────────────────────────────────
#  Preprocess (smoke test — no image file needed)
# ──────────────────────────────────────────────────────────────
from engine.preprocess import preprocess


def test_preprocess_returns_bgr():
    fake_img = np.random.randint(0, 255, (64, 200, 3), dtype=np.uint8)
    out = preprocess(fake_img, apply_deskew=False)
    assert out.shape == fake_img.shape
    assert out.dtype == np.uint8


# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────
from configs.config import CONFIGS


def test_configs_have_required_keys():
    required = ["chars", "img_h", "img_w", "epochs", "lr", "batch_train",
                "save_path", "lexicon_re"]
    for variant, cfg in CONFIGS.items():
        for key in required:
            assert key in cfg, f"CONFIGS[{variant}] missing key '{key}'"