import random
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(h: int, w: int) -> A.Compose:
    return A.Compose([
        A.Resize(h, w),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.6),
                                       contrast_limit=(-0.4, 0.5), p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.8, hue=0.5, p=1.0),
        ], p=0.85),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=60,
                              val_shift_limit=40, p=0.7),
        A.ToGray(p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=9, p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(1, 5), p=1.0),
            A.ZoomBlur(max_factor=1.08, p=1.0),
        ], p=0.55),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 60), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.5), p=1.0),
        ], p=0.45),
        A.Affine(scale=(0.85, 1.1), rotate=(-5, 5), shear=(-4, 4), p=0.45),
        A.RandomShadow(p=0.3),
        A.ImageCompression(quality_lower=35, quality_upper=95, p=0.35),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(h: int, w: int) -> A.Compose:
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def cutmix_batch(imgs, targets, lengths, alpha: float = 0.5):
    if random.random() > 0.5:
        return imgs, targets, lengths
    B, C, H, W = imgs.shape
    lam   = np.random.beta(alpha, alpha)
    rand_idx = np.random.permutation(B)
    cut_w = int(W * (1 - lam))
    cx    = random.randint(0, W - cut_w)
    imgs_mix = imgs.clone()
    imgs_mix[:, :, :, cx:cx + cut_w] = imgs[rand_idx, :, :, cx:cx + cut_w]
    return imgs_mix, targets, lengths


