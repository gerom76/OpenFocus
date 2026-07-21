"""
Synthetic focus-stack generator shared by the fusion tests and report tools.

Builds a sharp reference image and derives a stack from it by blurring every
region except the one that slice is "focused" on. A perfect fusion of that
stack reproduces the reference exactly, which gives the quality metrics a
ground truth to compare against.

Two reference styles are available, and the choice matters:

* ``"texture"`` (default) is dense broadband noise. It stresses edge-preserving
  spatial methods hard, but it is pathological for focus measures that read
  variance as sharpness - noise is high-variance whether or not it is in focus,
  so DCT scores near a single unfused slice on it.
* ``"photographic"`` is smooth gradients with solid shapes and thin lines,
  which is what the app's methods are actually aimed at. Use this one whenever
  several methods are being compared against each other.
"""

import cv2
import numpy as np


def make_reference(height=256, width=256, seed=0, style="texture"):
    """
    Render a sharp reference image.

    style="texture"      - broadband noise plus a checkerboard band
    style="photographic" - smooth gradients, solid shapes and thin lines
    """
    if style == "photographic":
        return _photographic_reference(height, width, seed)
    if style != "texture":
        raise ValueError(f"Unknown reference style {style!r}")

    rng = np.random.default_rng(seed)

    # Low-frequency colour gradient so the base layer is not flat
    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    img = np.stack([xs.repeat(height, 0), ys.repeat(width, 1),
                    0.5 * (xs + ys)], axis=2) * 180.0

    # High-frequency texture, so focus measures have something to latch onto
    img += rng.normal(0.0, 28.0, size=(height, width, 3)).astype(np.float32)

    # Hard edges: a checkerboard band and a few shapes
    tile = max(8, height // 16)
    checker = (((np.arange(height)[:, None] // tile) +
                (np.arange(width)[None, :] // tile)) % 2).astype(np.float32)
    band = slice(height // 3, 2 * height // 3)
    img[band] = img[band] * 0.4 + checker[band, :, None] * 255.0 * 0.6

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.circle(img, (width // 4, height // 4), max(8, height // 10), (255, 255, 255), 2)
    cv2.rectangle(img, (width // 2, height // 2),
                  (width - 10, height - 10), (0, 0, 255), 2)
    return img


def _photographic_reference(height, width, seed):
    """
    A scene closer to what the fusion methods target: smooth colour gradients,
    solid shapes with clean boundaries, and thin diagonal lines. Detail lives in
    edges rather than in per-pixel noise, so variance-based focus measures can
    tell blur from sharpness.
    """
    rng = np.random.default_rng(seed)

    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    img = np.stack([xs.repeat(height, 0) * 140 + 60,
                    ys.repeat(width, 1) * 120 + 70,
                    (1.0 - xs).repeat(height, 0) * 130 + 50], axis=2)

    # A trace of sensor noise, not enough to swamp the edges
    img += rng.normal(0.0, 4.0, size=(height, width, 3)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    scale = min(height, width) / 256.0
    for i in range(14):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        colour = tuple(int(v) for v in rng.integers(30, 225, 3))
        if i % 2:
            cv2.circle(img, centre, int(rng.integers(10, 34) * scale), colour, -1)
        else:
            corner = (centre[0] + int(rng.integers(14, 44) * scale),
                      centre[1] + int(rng.integers(14, 44) * scale))
            cv2.rectangle(img, centre, corner, colour, -1)

    # Thin diagonals give the high-frequency content a defocus blur destroys
    for x in range(0, width, max(8, int(26 * scale))):
        cv2.line(img, (x, 0), (x + int(40 * scale), height), (245, 245, 245), 1)

    return img


def make_stack(reference=None, num_slices=3, blur_ksize=15, feather=0, **ref_kwargs):
    """
    Slice `reference` into `num_slices` horizontal bands and produce one image
    per band that is sharp inside its band and blurred everywhere else.

    Returns (stack, reference, masks) where masks[k] is the float mask of the
    region slice k is in focus on.
    """
    if reference is None:
        reference = make_reference(**ref_kwargs)

    height, width = reference.shape[:2]
    blurred = cv2.GaussianBlur(reference, (blur_ksize, blur_ksize), 0)

    edges = np.linspace(0, height, num_slices + 1).round().astype(int)
    stack, masks = [], []
    for k in range(num_slices):
        mask = np.zeros((height, width), dtype=np.float32)
        mask[edges[k]:edges[k + 1]] = 1.0
        if feather:
            ksize = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        m = mask[:, :, None]
        img = reference.astype(np.float32) * m + blurred.astype(np.float32) * (1.0 - m)
        stack.append(np.clip(img, 0, 255).astype(np.uint8))
        masks.append(mask)

    return stack, reference, masks
