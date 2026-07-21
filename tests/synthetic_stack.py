"""
Synthetic focus-stack generator shared by the Guided Filter tests and benchmark.

Builds a sharp reference image and derives a stack from it by blurring every
region except the one that slice is "focused" on. A perfect fusion of that
stack reproduces the reference exactly, which gives the quality metrics a
ground truth to compare against.
"""

import cv2
import numpy as np


def make_reference(height=256, width=256, seed=0):
    """Render a detail-rich sharp image: noise texture plus geometric shapes."""
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
