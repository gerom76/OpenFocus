"""
GPU implementation of DCT/variance multi-focus fusion (Haghighat et al., 2011).

Mirrors fusion_methods/dct.py: by Parseval's theorem the DCT AC energy of a
block equals its pixel variance, so the per-block sharpness measure reduces to
block means of x and x^2 (computed here with avg_pool2d on the GPU). The tiny
consistency-verification step (median filtering of the index map) stays on the
CPU via cv2 for exact behavioral parity with the CPU implementation.

Reference:
Haghighat M B A, Aghagolzadeh A, Seyedarabi H. Multi-focus image fusion for
visual sensor networks in DCT domain[J]. Computers & Electrical Engineering,
2011, 37(5): 789-797.
"""

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from fusion_methods.dct import (
    _collect_images_from_folder,
    _median_filter_index_map,
    _normalize_image_stack,
)
from fusion_methods import torch_depth
from utils import bitdepth

# Number of stack images processed per GPU batch (bounds peak memory)
CHUNK_SIZE = 4


def dct_torch_impl(
    source: Union[str, Sequence[np.ndarray]],
    block_size: int = 8,
    kernel_size: int = 7,
    device: str = None,
) -> np.ndarray:
    """
    DCT/variance fusion on a torch device.

    Args:
        source: Directory path or list of BGR uint8 arrays
        block_size: Block size for the variance measure
        kernel_size: Median filter kernel size for consistency verification (odd)
        device: 'cuda', 'mps', or None to auto-select

    Returns:
        Fused BGR uint8 image
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            raise RuntimeError("No GPU device (CUDA/MPS) available for DCT fusion")

    if kernel_size % 2 == 0:
        kernel_size += 1
    block_size = max(2, int(block_size))

    if isinstance(source, str):
        images, _ = _collect_images_from_folder(source)
    elif isinstance(source, (list, tuple)):
        images = [img for img in source if img is not None]
    else:
        raise TypeError("Source must be folder path or image list.")

    if len(images) < 2:
        raise ValueError("At least 2 images are required for fusion.")

    normalized_images, (h, w) = _normalize_image_stack(images)

    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    map_h = h_trim // block_size
    map_w = w_trim // block_size

    if map_h == 0 or map_w == 0:
        raise ValueError("Block size is too large for image size.")

    n = len(normalized_images)
    dev = torch.device(device)

    # Output pixels are copied verbatim from the source frames, so the result
    # keeps the depth the stack arrived in. torch has no uint16, so 16-bit
    # levels are carried through the composition pass as int32 and narrowed back
    # on the host at the end.
    out_dtype = bitdepth.stack_dtype(normalized_images)
    compose_dtype = torch.uint8 if out_dtype == bitdepth.UINT8 else torch.int32

    with torch.no_grad():
        def to_device_levels(chunk, trim=True):
            """Raw pixel levels on the device, for verbatim copying."""
            if trim:
                chunk = [img[:h_trim, :w_trim] for img in chunk]
            arr = np.stack([np.ascontiguousarray(img) for img in chunk])
            if arr.dtype != np.uint8:
                arr = arr.astype(np.int32)
            return torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2)  # (B, 3, H, W) BGR

        def to_device_float01(chunk, trim=True):
            """Normalised [0, 1] copy on the device, for the variance measure."""
            if trim:
                chunk = [img[:h_trim, :w_trim] for img in chunk]
            return torch_depth.stack_to_float01(chunk, dev)

        # Pass 1: per-block variance via Var(X) = E[X^2] - E[X]^2, running max
        max_variance = torch.full((map_h, map_w), -1.0, device=dev)
        best_index = torch.zeros((map_h, map_w), dtype=torch.long, device=dev)

        for start in range(0, n, CHUNK_SIZE):
            chunk = normalized_images[start:start + CHUNK_SIZE]
            # Normalised before squaring: at 16 bits, squaring raw levels lands
            # near float32's precision limit and the E[X^2] - E[X]^2 difference
            # cancels away small variances (see the note in dct.py).
            t = to_device_float01(chunk)
            gray = (0.114 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.299 * t[:, 2:3])
            mean_sq = F.avg_pool2d(gray * gray, block_size)
            sq_mean = F.avg_pool2d(gray, block_size) ** 2
            var = (mean_sq - sq_mean)[:, 0]  # (B, map_h, map_w)

            # Sequential update preserves first-max-wins tie behavior of the CPU path
            for j in range(var.shape[0]):
                mask = var[j] > max_variance
                max_variance = torch.where(mask, var[j], max_variance)
                best_index = torch.where(mask, torch.tensor(start + j, device=dev), best_index)

        # Consistency verification: double median filter on the tiny index map
        # (CPU/cv2). uint16 end to end so indices never wrap at 256 frames.
        index_np = best_index.cpu().numpy().astype(np.uint16)
        filtered = _median_filter_index_map(index_np, kernel_size)
        filtered = _median_filter_index_map(filtered, kernel_size)
        final_index = torch.from_numpy(filtered.astype(np.int64)).to(dev)

        # Pass 2: reconstruction — nearest-neighbor upscale of the index map, then
        # copy each source image into its selected blocks
        full_index = final_index.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

        # The block grid only covers a multiple of block_size. Extend the last
        # row/column of decisions over the remaining strip so the output keeps
        # the geometry it was handed (matches the CPU path in dct.py).
        if (h_trim, w_trim) != (h, w):
            full_index = torch.cat(
                [full_index, full_index[-1:, :].expand(h - h_trim, -1)], dim=0)
            full_index = torch.cat(
                [full_index, full_index[:, -1:].expand(-1, w - w_trim)], dim=1)

        fused = torch.zeros((3, h, w), dtype=compose_dtype, device=dev)

        used = torch.unique(final_index).tolist()
        for start in range(0, n, CHUNK_SIZE):
            chunk_ids = [k for k in range(start, min(start + CHUNK_SIZE, n)) if k in used]
            if not chunk_ids:
                continue
            t = to_device_levels([normalized_images[k] for k in chunk_ids], trim=False)
            for j, k in enumerate(chunk_ids):
                fused = torch.where(full_index == k, t[j], fused)

        # (H, W, 3) BGR at the stack's own depth. The int32 carrier holds exact
        # source levels, so narrowing back to uint16 here is lossless.
        result = fused.permute(1, 2, 0).cpu().numpy()
        return result if result.dtype == out_dtype else result.astype(out_dtype)
