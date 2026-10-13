"""E6 hand-crafted features (78-d, D9): uniform LBP (P=8, R=1,2), GLCM (4 props x 4 angles),
Gabor bank (4 orientations x 3 frequencies; mean + energy) and colour moments (HSV, CIELAB)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

SIZE = 224
LBP_RADII = (1, 2)
LBP_P = 8
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")
ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GABOR_FREQS = (0.1, 0.2, 0.3)

FEATURE_NAMES = (
    [f"lbp_r{r}_b{b}" for r in LBP_RADII for b in range(LBP_P + 2)]
    + [f"glcm_{p}_a{int(np.degrees(a))}" for p in GLCM_PROPS for a in ANGLES]
    + [f"gabor_f{f}_a{int(np.degrees(a))}_{s}" for f in GABOR_FREQS for a in ANGLES for s in ("mean", "energy")]
    + [f"{space}_{ch}_{stat}" for space, chans in (("hsv", "hsv"), ("lab", "lab")) for ch in chans
       for stat in ("mean", "std", "skew")]
)


def gabor_kernels(freq: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Real and imaginary Gabor kernels; sigma matches skimage's default bandwidth of 1."""
    sigma = 0.56 / freq
    ksize = int(6 * sigma) | 1
    args = dict(ksize=(ksize, ksize), sigma=sigma, theta=theta, lambd=1.0 / freq, gamma=1.0)
    real = cv2.getGaborKernel(**args, psi=0, ktype=cv2.CV_32F)
    imag = cv2.getGaborKernel(**args, psi=np.pi / 2, ktype=cv2.CV_32F)
    return real - real.mean(), imag


def texture_features(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    feats = []
    for r in LBP_RADII:
        lbp = local_binary_pattern(gray, LBP_P, r, method="uniform")
        hist, _ = np.histogram(lbp, bins=LBP_P + 2, range=(0, LBP_P + 2), density=True)
        feats.extend(hist)
    q = (gray // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(q, distances=[1], angles=ANGLES, levels=32, symmetric=True, normed=True)
    for p in GLCM_PROPS:
        feats.extend(graycoprops(glcm, p)[0])
    g = gray.astype(np.float32) / 255.0
    for f in GABOR_FREQS:
        for a in ANGLES:
            real, imag = (cv2.filter2D(g, cv2.CV_32F, k) for k in gabor_kernels(f, a))
            mag = np.hypot(real, imag)
            feats.extend([mag.mean(), (mag ** 2).mean()])
    for conv in (cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LAB):
        x = cv2.cvtColor(img, conv).reshape(-1, 3).astype(np.float32)
        for c in range(3):
            v = x[:, c]
            feats.extend([v.mean(), v.std(), float(skew(v)) if v.std() > 0 else 0.0])
    out = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
