import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from engine.preprocess import preprocess
from engine.codec import Encoder


class MeterDataset(Dataset):
    def __init__(self, img_dir: str, label_file: str, transform,
                 encoder: Encoder, apply_preprocess: bool = True):
        self.img_dir           = img_dir
        self.transform         = transform
        self.encoder           = encoder
        self.apply_preprocess  = apply_preprocess
        self.samples           = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        if img is None:
            raise FileNotFoundError(f"Image not found: {name}")
        if self.apply_preprocess:
            img = preprocess(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"], label


def collate_fn(batch, encoder: Encoder):
    imgs, labels = zip(*batch)
    imgs    = torch.stack(imgs)
    targets, lengths = [], []
    for label in labels:
        enc = encoder.encode(label)
        targets.extend(enc)
        lengths.append(len(enc))
    return (imgs,
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            labels)


def build_loaders(cfg: dict, encoder: Encoder,
                  train_tf, val_tf) -> tuple:
    from functools import partial

    train_ds = MeterDataset(cfg["train_dir"], cfg["train_labels"],
                             train_tf, encoder, apply_preprocess=True)
    val_ds   = MeterDataset(cfg["val_dir"],   cfg["val_labels"],
                             val_tf,   encoder, apply_preprocess=True)
    test_ds  = MeterDataset(cfg["test_dir"],  cfg["test_labels"],
                             val_tf,   encoder, apply_preprocess=True)

    fn = partial(collate_fn, encoder=encoder)
    train_loader = DataLoader(train_ds, cfg["batch_train"], shuffle=True,
                               num_workers=cfg["workers"], collate_fn=fn,
                               pin_memory=True, persistent_workers=cfg["workers"] > 0)
    val_loader   = DataLoader(val_ds,   cfg["batch_val"],   shuffle=False,
                               num_workers=cfg["workers"], collate_fn=fn,
                               pin_memory=True, persistent_workers=cfg["workers"] > 0)
    test_loader  = DataLoader(test_ds,  cfg["batch_val"],   shuffle=False,
                               num_workers=cfg["workers"], collate_fn=fn,
                               pin_memory=True, persistent_workers=cfg["workers"] > 0)
    return train_loader, val_loader, test_loader

