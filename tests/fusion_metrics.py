"""
Fusion quality metrics implemented on numpy/OpenCV only (no scikit-image).

Two families are provided:

* Full-reference (need the sharp ground truth): psnr, ssim.
* No-reference (only need the fused image and the source stack):
  entropy, spatial_frequency, std_dev, qabf.

The no-reference set is what applies to real focus stacks, where no all-in-focus
ground truth exists; the full-reference set is used against the synthetic stacks
in tests/synthetic_stack.py.
"""

import cv2
import numpy as np

_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2


def _gray32(img):
    """Convert a BGR or single-channel uint8 image to float32 grayscale."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def psnr(fused, reference):
    """Peak signal-to-noise ratio in dB; higher is better, inf on exact match."""
    a = fused.astype(np.float64)
    b = reference.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(fused, reference, sigma=1.5):
    """Mean structural similarity in [-1, 1] on the Gaussian-weighted window."""
    a = _gray32(fused)
    b = _gray32(reference)

    mu_a = cv2.GaussianBlur(a, (0, 0), sigma)
    mu_b = cv2.GaussianBlur(b, (0, 0), sigma)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    var_a = cv2.GaussianBlur(a * a, (0, 0), sigma) - mu_aa
    var_b = cv2.GaussianBlur(b * b, (0, 0), sigma) - mu_bb
    cov_ab = cv2.GaussianBlur(a * b, (0, 0), sigma) - mu_ab

    num = (2 * mu_ab + _SSIM_C1) * (2 * cov_ab + _SSIM_C2)
    den = (mu_aa + mu_bb + _SSIM_C1) * (var_a + var_b + _SSIM_C2)
    return float(np.mean(num / den))


def entropy(img):
    """Shannon entropy of the grayscale histogram, in bits; higher = more information."""
    gray = _gray32(img).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def spatial_frequency(img):
    """
    Spatial frequency (Eskicioglu & Fisher): RMS of the row and column
    gradients. Higher means more detail retained; blurry results score low.
    """
    gray = _gray32(img)
    rf = np.mean(np.diff(gray, axis=1) ** 2)
    cf = np.mean(np.diff(gray, axis=0) ** 2)
    return float(np.sqrt(rf + cf))


def std_dev(img):
    """Grayscale standard deviation: a coarse proxy for contrast."""
    return float(np.std(_gray32(img)))


def _sobel_gradient(gray):
    """Return (magnitude, orientation) of the Sobel gradient."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx + np.finfo(np.float32).eps)
    return mag, ang


def _edge_preservation(src_gray, fused_gray):
    """Xydeas-Petrovic per-pixel edge preservation Q^{AB/F} for one source."""
    # Constants from the original paper
    gamma_g, kappa_g, sigma_g = 0.9994, -15.0, 0.5
    gamma_a, kappa_a, sigma_a = 0.9879, -22.0, 0.8

    g_src, a_src = _sobel_gradient(src_gray)
    g_fus, a_fus = _sobel_gradient(fused_gray)

    eps = np.finfo(np.float32).eps
    # Relative strength: ratio folded so it always lands in (0, 1]
    ratio = np.where(g_src > g_fus, g_fus / (g_src + eps), g_src / (g_fus + eps))
    ratio = np.clip(np.nan_to_num(ratio), 0.0, 1.0)

    # Orientation agreement, 1 when the edges point the same way
    da = 1.0 - np.abs(a_src - a_fus) / (np.pi / 2)

    q_g = gamma_g / (1.0 + np.exp(kappa_g * (ratio - sigma_g)))
    q_a = gamma_a / (1.0 + np.exp(kappa_a * (da - sigma_a)))
    return q_g * q_a, g_src


def qabf(fused, sources):
    """
    Q^{AB/F} gradient-based fusion quality (Xydeas & Petrovic, 2000), in [0, 1].

    Measures how much of the edge information present in the source stack
    survives in the fused image, weighted by source edge strength. No ground
    truth needed, so this is the headline metric for real stacks.
    """
    fused_gray = _gray32(fused)
    num = 0.0
    den = 0.0
    for src in sources:
        q, weight = _edge_preservation(_gray32(src), fused_gray)
        num += float(np.sum(q * weight))
        den += float(np.sum(weight))
    if den == 0:
        return 0.0
    return num / den


def evaluate(fused, sources, reference=None):
    """Collect every applicable metric into one dict."""
    scores = {
        "qabf": qabf(fused, sources),
        "entropy": entropy(fused),
        "spatial_frequency": spatial_frequency(fused),
        "std_dev": std_dev(fused),
    }
    if reference is not None:
        scores["psnr"] = psnr(fused, reference)
        scores["ssim"] = ssim(fused, reference)
    return scores
