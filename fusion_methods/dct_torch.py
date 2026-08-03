"""
GPU implementation of DCT/variance multi-focus fusion (Haghighat et al., 2011).

Mirrors fusion_methods/dct.py: by Parseval's theorem the DCT AC energy of a
block equals the variance of its pixels, so the per-block sharpness measure
reduces to block means of a high-passed frame's squared detail (computed here
with avg_pool2d on the GPU - a box blur, a block mean and the two pooling
windows are all average pools). See dct.py for why the measure is band-limited
from below and how undecidable blocks are settled. The tiny
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
    _BLEND,
    _HIGHPASS_SCALE,
    _NOISE_FLOOR,
    _NOISE_PERCENTILE,
    _PLATEAU,
    _POOL_WINDOW,
    _RUN_EXIT,
    _SUBFRAME,
    _collect_images_from_folder,
    _compose,
    _median_filter_map,
    _normalize_image_stack,
    _odd,
)
from fusion_methods import torch_depth
from utils import bitdepth

# Number of stack images processed per GPU batch (bounds peak memory)
CHUNK_SIZE = 4


def _box_blur(x: torch.Tensor, window: int) -> torch.Tensor:
    """Box blur with reflected borders - the twin of cv2.blur/cv2.boxFilter.

    cv2's default border is BORDER_REFLECT_101, which is what torch calls
    'reflect', so padding then average-pooling with stride 1 reproduces the CPU
    filter. The window is already clamped to the input by _odd, so the pad stays
    inside torch's limit of one reflection.
    """
    if window <= 1:
        return x
    pad = window // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                        window, stride=1)


def dct_torch_impl(
    source: Union[str, Sequence[np.ndarray]],
    block_size: int = 8,
    kernel_size: int = 7,
    device: str = None,
    plateau: float = None,
    blend: bool = None,
) -> np.ndarray:
    """
    DCT/variance fusion on a torch device.

    Args:
        source: Directory path or list of BGR uint8 arrays
        block_size: Block size for the variance measure
        kernel_size: Median filter kernel size for consistency verification (odd)
        device: 'cuda', 'mps', or None to auto-select
        plateau: How far below the peak a frame still counts as in focus;
            None uses dct._PLATEAU. Clamped as on the CPU path.
        blend: Weight neighbouring frames together, or copy blocks verbatim;
            None uses dct._BLEND.

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
    plateau = _PLATEAU if plateau is None else min(max(float(plateau), 0.1), 0.85)
    blend = _BLEND if blend is None else bool(blend)

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

    # Output keeps the depth the stack arrived in. torch has no uint16, so the
    # fallback composition pass carries 16-bit levels as int32 and narrows back
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

        hp_window = _odd(round(block_size * _HIGHPASS_SCALE), min(h_trim, w_trim))
        pool_window = _odd(_POOL_WINDOW, min(map_h, map_w))

        def chunk_energy(start):
            """Normalised per-block detail energy for one chunk of frames."""
            chunk = normalized_images[start:start + CHUNK_SIZE]
            # Normalised before squaring: at 16 bits, squaring raw levels lands
            # near float32's precision limit (see the note in dct.py).
            t = to_device_float01(chunk)
            gray = (0.114 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.299 * t[:, 2:3])
            # High-pass, then block-mean the squared detail: DCT AC energy with
            # the sub-block band removed, so a defocused wash scores near zero.
            detail = gray - _box_blur(gray, hp_window)
            energy = F.avg_pool2d(detail * detail, block_size)
            pooled = _box_blur(energy, pool_window)[:, 0]      # (B, map_h, map_w)

            # Each frame in units of its own noise, so grain in a bright veil
            # cannot outbid detail in a dark frame (see the note in dct.py).
            for j in range(pooled.shape[0]):
                noise = torch.quantile(pooled[j].flatten(),
                                       _NOISE_PERCENTILE / 100.0)
                pooled[j] = pooled[j] / torch.clamp(noise, min=_NOISE_FLOOR)
            return pooled

        # Pass 1: the bar for being in focus on each block.
        peak = torch.full((map_h, map_w), -1.0, device=dev)
        for start in range(0, n, CHUNK_SIZE):
            peak = torch.maximum(peak, chunk_energy(start).amax(dim=0))

        # Pass 2: the middle of the in-focus run containing the peak - the focal
        # plane. Mirrors dct._focal_plane, which carries the reasoning; the
        # frames arrive here in chunks but are still folded in one at a time and
        # in order, which is what the run tracking needs.
        threshold = peak * plateau
        exit_bar = threshold * _RUN_EXIT
        lo = torch.zeros((map_h, map_w), device=dev)
        hi = torch.zeros((map_h, map_w), device=dev)
        best = torch.full((map_h, map_w), -float('inf'), device=dev)
        run_start = torch.zeros((map_h, map_w), device=dev)
        inside = torch.zeros((map_h, map_w), dtype=torch.bool, device=dev)
        alive = torch.zeros((map_h, map_w), dtype=torch.bool, device=dev)
        for start in range(0, n, CHUNK_SIZE):
            pooled = chunk_energy(start)
            for j in range(pooled.shape[0]):
                energy, index = pooled[j], torch.full_like(lo, float(start + j))
                in_focus = energy >= threshold
                was_inside = inside
                inside = (inside & (energy >= exit_bar)) | in_focus
                run_start = torch.where(inside & ~was_inside, index, run_start)
                alive = alive & inside

                peak_here = energy > best
                best = torch.where(peak_here, energy, best)
                lo = torch.where(peak_here, run_start, lo)
                hi = torch.where(peak_here | (alive & in_focus), index, hi)
                alive = alive | peak_here
        field = (lo + hi) * 0.5

        # Consistency verification: double median filter on the tiny focal-plane
        # map (CPU/cv2). Carried in units of 1/_SUBFRAME of a frame so a plane
        # between two frames survives the filter, and uint16 so it never wraps
        # at 256 frames.
        plane_np = torch.clamp(torch.round(field * _SUBFRAME), 0,
                               (n - 1) * _SUBFRAME).cpu().numpy().astype(np.uint16)
        plane_np = _median_filter_map(plane_np, kernel_size)
        plane_np = _median_filter_map(plane_np, kernel_size)
        field_np = plane_np.astype(np.float32) / _SUBFRAME

        # Reconstruction. Compositing is a weighted sum over a handful of full
        # frames, which is memory-bound rather than arithmetic-bound, so it runs
        # on the host through the CPU path's own helper: that keeps the two
        # implementations from drifting and keeps peak device memory to the
        # measurement pass.
        if blend:
            return _compose(normalized_images, field_np, block_size, (h, w),
                            out_dtype)

        final_index = torch.from_numpy(np.rint(field_np).astype(np.int64)).to(dev)
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
