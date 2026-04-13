import cv2
import numpy as np


def _blur_score(img_gray: np.ndarray) -> float:
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def _deskew(img_gray: np.ndarray) -> np.ndarray:
    edges  = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    if lines is None:
        return img_gray
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 15:
            angles.append(angle)
    if not angles:
        return img_gray
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img_gray
    h, w    = img_gray.shape
    M       = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _clahe_hsv(img_bgr: np.ndarray) -> np.ndarray:
    img_hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v    = cv2.split(img_hsv)
    clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    v_eq       = clahe.apply(v)
    img_hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2BGR)


def _unsharp(img_gray: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_gray, (0, 0), sigma)
    return cv2.addWeighted(img_gray, 1 + strength, blurred, -strength, 0)


def preprocess(img_bgr: np.ndarray,
               blur_threshold: float = 80.0,
               apply_deskew: bool = True) -> np.ndarray:
    img      = _clahe_hsv(img_bgr)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)
    if apply_deskew:
        img_gray = _deskew(img_gray)
    if _blur_score(img_gray) < blur_threshold:
        img_gray = _unsharp(img_gray)
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
