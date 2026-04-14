import re as _re

CHARS = "0123456789."

BASE_CFG = {
    "chars": CHARS,
    "img_h": 48,
    "img_w": 320,
    "epochs": 100,
    "batch_val": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "grad_clip": 5.0,
    "patience": 25,
    "workers": 4,
    "prefetch_factor": 2,
    "seed": 42,
    "use_frm": True,
    "use_sgm": True,
    "beam_width": 10,
    "conf_threshold": 0.85,
    "focal_gamma": 2.0,
    "label_smooth": 0.1,
    "cutmix_alpha": 0.5,
    "lexicon_re": r"^\d{4,8}(\.\d{1,2})?$",
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "train_labels": "data/train_labels.txt",
    "val_labels": "data/val_labels.txt",
    "test_labels": "data/test_labels.txt",
}

CONFIGS = {
    "tiny": {
        **BASE_CFG,
        "variant": "tiny",
        "batch_train": 128,
        "save_path": "weights/tiny_best.pth",
    },
    "small": {
        **BASE_CFG,
        "variant": "small",
        "batch_train": 96,
        "lr": 2e-4,
        "save_path": "weights/small_best.pth",
    },
    "base": {
        **BASE_CFG,
        "variant": "base",
        "batch_train": 32,
        "lr": 1e-4,
        "weight_decay": 2e-4,
        "patience": 20,
        "save_path": "weights/base_best.pth",
    },
}

# Validate regexes at import time — crash loudly rather than silently at eval
for _k, _v in CONFIGS.items():
    try:
        _re.compile(_v["lexicon_re"])
    except _re.error as e:
        raise ValueError(f"CONFIGS[{_k}]['lexicon_re'] is invalid regex: {e}")