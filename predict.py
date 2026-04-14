import os
import sys
import argparse
import cv2
import torch
import numpy as np
# FIX: use non-deprecated torch.amp API
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.svtr import build_model
from engine.codec import Encoder
from engine.preprocess import preprocess
from engine.augment import get_val_transforms
from configs.config import CONFIGS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--img",     required=True, help="Image path or folder")
    p.add_argument("--variant", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--device",  default=None)
    p.add_argument("--tta",     action="store_true", help="5x TTA majority vote")
    return p.parse_args()


def setup_device(override=None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights_path: str, variant: str, device: torch.device):
    ckpt    = torch.load(weights_path, map_location=device)
    cfg     = ckpt.get("cfg", CONFIGS[variant])
    encoder = Encoder(cfg["chars"])
    model   = build_model(
        variant, cfg["img_h"], cfg["img_w"],
        encoder.num_classes,
        use_frm=cfg.get("use_frm", True),
        use_sgm=cfg.get("use_sgm", True),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, encoder, cfg


def predict_single(model, encoder, img_bgr, cfg, device, tf, use_tta=False):
    img = preprocess(img_bgr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    amp_enabled = device.type != "cpu"

    if not use_tta:
        tensor = tf(image=img)["image"].unsqueeze(0).to(device)
        with torch.no_grad(), autocast(device_type=device.type, enabled=amp_enabled):
            log_probs = model(tensor)
        lp_np      = log_probs[0].cpu().numpy()
        pred, conf = encoder.decode_beam(
            lp_np,
            beam_width=cfg.get("beam_width", 10),
            lexicon_pattern=cfg.get("lexicon_re"),
        )
        return pred, conf

    import albumentations as A
    from collections import Counter
    votes = []
    bcs   = [(-0.3, -0.3), (-0.15, 0.0), (0.0, 0.0), (0.15, 0.15), (0.3, 0.3)]
    for b, c in bcs:
        aug     = A.Compose([A.RandomBrightnessContrast(
                                brightness_limit=(b, b),
                                contrast_limit=(c, c), p=1.0)])
        img_aug = aug(image=img)["image"]
        tensor  = tf(image=img_aug)["image"].unsqueeze(0).to(device)
        with torch.no_grad(), autocast(device_type=device.type, enabled=amp_enabled):
            log_probs = model(tensor)
        lp_np    = log_probs[0].cpu().numpy()
        pred, sc = encoder.decode_beam(
            lp_np,
            beam_width=cfg.get("beam_width", 10),
            lexicon_pattern=cfg.get("lexicon_re"),
        )
        votes.append((pred, sc))

    texts  = [v[0] for v in votes]
    winner = Counter(texts).most_common(1)[0][0]
    best_sc = max(sc for t, sc in votes if t == winner)
    return winner, best_sc


def main():
    args   = get_args()
    device = setup_device(args.device)
    model, encoder, cfg = load_model(args.weights, args.variant, device)
    tf     = get_val_transforms(cfg["img_h"], cfg["img_w"])

    paths = []
    if os.path.isdir(args.img):
        for fn in sorted(os.listdir(args.img)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(args.img, fn))
    else:
        paths = [args.img]

    print(f"\n{'Image':<40} {'Pred':<14} {'Conf':>6}  {'Flag'}")
    print("-" * 68)
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"{os.path.basename(path):<40} READ_ERROR")
            continue
        pred, conf = predict_single(model, encoder, img, cfg, device, tf,
                                    use_tta=args.tta)
        flag = "LOW_CONF" if conf < cfg.get("conf_threshold", 0.85) else ""
        print(f"{os.path.basename(path):<40} {pred:<14} {conf:>6.4f}  {flag}")


if __name__ == "__main__":
    main()