"""Optional post-fusion contrast enhancement.

Fusion is faithful to its inputs, so a stack shot flat - a pale subject on a
bright, even background - fuses to a pale result. This module adds contrast back
as an *output-time* step: it runs on the finished fusion result, not inside any
method, so the stored result stays pristine and the strength can be changed
without re-rendering.

Two methods, both driven by one 0..1 strength:

- ``auto``  a global tone curve: a gentle black/white-point stretch followed by
            a midtone S-curve. Predictable, and it does not touch hue.
- ``clahe`` local adaptive equalisation (CLAHE). Reveals fine texture, at the
            cost of making background noise and dust more visible.

Both are **colour-safe** and **depth-agnostic**. Colour-safe because the curve
is applied to luminance alone and re-applied to the colour channels as a per-
pixel gain, so channel ratios - hue and saturation - are preserved and only
brightness/contrast changes. Depth-agnostic because the work is done in float
[0, 1] and written back at the input's own dtype, so an 8- or 16-bit frame comes
back at the same depth (see utils/bitdepth.py).

Strength blends the luminance between its original and fully-transformed values,
so strength 0 is exactly the identity - the same image, byte for byte.
"""

from typing import Optional

import cv2
import numpy as np

from utils import bitdepth

METHOD_OFF = "off"
METHOD_AUTO = "auto"
METHOD_CLAHE = "clahe"
VALID_METHODS = (METHOD_OFF, METHOD_AUTO, METHOD_CLAHE)

# Luminance weights matching cv2.COLOR_BGR2GRAY on a BGR image.
_BGR_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)

# Below this luminance a pixel is treated as black and left alone, so the gain
# ratio never divides by ~0 and near-black regions do not get colour noise
# amplified into them.
_BLACK_EPS = 1e-4


def _luminance(bgr: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a float BGR image, shape (H, W)."""
    return bgr @ _BGR_LUMA


def _sigmoid_scurve(x: np.ndarray, pivot: float, k: float) -> np.ndarray:
    """Logistic S-curve through (0,0) and (1,1), steepening around `pivot`.

    Increasing `k` increases midtone contrast. Normalising by the endpoints
    keeps black black and white white, so only the distribution between them is
    reshaped.
    """
    def s(t):
        return 1.0 / (1.0 + np.exp(-k * (t - pivot)))
    s0 = s(0.0)
    s1 = s(1.0)
    return ((s(x) - s0) / (s1 - s0 + 1e-8)).astype(np.float32)


def _auto_curve(y: np.ndarray, strength: float) -> np.ndarray:
    """Full-strength 'auto' luminance transform.

    A robust percentile stretch sets the black and white points, then a midtone
    S-curve adds punch around the median. The S-curve does most of the work on
    images whose black point is already pinned (a dark border, say), which a bare
    min/max stretch cannot help.
    """
    lo = float(np.percentile(y, 0.5))
    hi = float(np.percentile(y, 99.5))
    if hi - lo > 1e-3:
        stretched = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    else:
        stretched = y

    pivot = float(np.clip(np.median(stretched), 0.15, 0.85))
    # k scales with strength so a partial blend and a low strength agree in
    # direction; the blend in apply_contrast does the final interpolation.
    return _sigmoid_scurve(stretched, pivot, k=6.0)


def _clahe_curve(y: np.ndarray, strength: float) -> np.ndarray:
    """Full-strength CLAHE luminance transform.

    CLAHE only accepts integer input, so luminance is taken to 16-bit for the
    equalisation and normalised back. clipLimit is fixed at full strength; the
    strength blend in apply_contrast scales the effect.
    """
    y16 = np.rint(np.clip(y, 0.0, 1.0) * 65535.0).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return (clahe.apply(y16).astype(np.float32) / 65535.0)


def apply_contrast(image: Optional[np.ndarray],
                   method: str = METHOD_OFF,
                   strength: float = 0.0) -> Optional[np.ndarray]:
    """Return a contrast-enhanced copy of `image`, at its own dtype.

    Args:
        image: BGR uint8 or uint16 image, or None.
        method: 'off', 'auto', or 'clahe'.
        strength: 0..1. 0 (or method 'off') returns the input unchanged.

    The input is never modified in place.
    """
    if image is None:
        return image
    strength = float(np.clip(strength, 0.0, 1.0))
    if method == METHOD_OFF or method not in VALID_METHODS or strength <= 0.0:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        # Only BGR colour frames are handled; anything else is passed through
        # rather than risk mangling it.
        return image

    out_dtype = image.dtype
    bgr = bitdepth.to_float01(image)
    y = _luminance(bgr)

    if method == METHOD_AUTO:
        y_full = _auto_curve(y, strength)
    else:  # METHOD_CLAHE
        y_full = _clahe_curve(y, strength)

    # Blend luminance between original and transformed, so strength 0 is the
    # identity and strength 1 is the full effect.
    y_final = y * (1.0 - strength) + y_full * strength

    # Re-apply the luminance change to the colour channels as a per-pixel gain.
    # This preserves channel ratios - hue and saturation - so only brightness
    # and contrast move. Near-black pixels keep a gain of 1 to avoid amplifying
    # colour noise up from nothing.
    gain = np.where(y > _BLACK_EPS, y_final / np.maximum(y, _BLACK_EPS), 1.0)
    out = bgr * gain[:, :, np.newaxis]
    return bitdepth.from_float01(out, out_dtype)


def describe(method: str, strength: float) -> str:
    """Short tag for output filenames, or '' when contrast is off."""
    if method == METHOD_OFF or method not in VALID_METHODS or strength <= 0:
        return ""
    label = "Auto" if method == METHOD_AUTO else "CLAHE"
    return f"Contrast{label}{int(round(strength * 100))}"
