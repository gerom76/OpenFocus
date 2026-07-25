"""
GPU implementation of Laplacian-pyramid multi-focus fusion.

Mirrors fusion_methods/pyramid.py on a torch device (CUDA/MPS): choose-max on
pooled band energy for the detail levels, activity-weighted average for the
base, strict '>' tie-break in frame order. Frames stream through one at a
time, so device memory is bounded by the fused accumulators plus a single
frame's pyramid regardless of stack size.

pyrDown/pyrUp are reimplemented with the same 5-tap Burt-Adelson kernel and
REFLECT_101 borders OpenCV uses, so the two variants agree to rounding noise
away from the outermost border pixels.

Reference:
Burt P J, Adelson E H. The Laplacian pyramid as a compact image code[J].
IEEE Transactions on Communications, 1983, 31(4): 532-540.
"""

import cv2
import torch
import torch.nn.functional as F

from fusion_methods import torch_depth
from fusion_methods.gff_torch import _load_stack
from fusion_methods.pyramid import (
    _resolve_levels, _ENERGY_WINDOW, _BASE_ACTIVITY_EPS,
)
from utils import bitdepth

# The 1-D half of OpenCV's fixed pyramid kernel: [1, 4, 6, 4, 1] / 16.
_PYR_TAPS = (1.0, 4.0, 6.0, 4.0, 1.0)


def _pyr_kernel(channels, device):
    """(C, 1, 5, 5) depthwise Burt-Adelson kernel, normalised to sum 1."""
    taps = torch.tensor(_PYR_TAPS, dtype=torch.float32, device=device) / 16.0
    k2d = torch.outer(taps, taps)
    return k2d.expand(channels, 1, 5, 5)


def _pyr_down(x, kernel):
    """cv2.pyrDown: blur with the pyramid kernel, keep even coordinates."""
    x = F.pad(x, (2, 2, 2, 2), mode='reflect')
    return F.conv2d(x, kernel, stride=2, groups=x.shape[1])


def _pyr_up(x, size, kernel):
    """cv2.pyrUp with an explicit destination size.

    Zero-insertion upsampling followed by the pyramid kernel scaled by 4, so
    the interpolated samples carry the same energy as the sources.
    """
    height, width = size
    up = torch.zeros((x.shape[0], x.shape[1], height, width),
                     dtype=x.dtype, device=x.device)
    up[:, :, ::2, ::2] = x[:, :, :(height + 1) // 2, :(width + 1) // 2]
    up = F.pad(up, (2, 2, 2, 2), mode='reflect')
    return F.conv2d(up, kernel * 4.0, groups=x.shape[1])


def _box_energy(band, box_kernel):
    """Pooled squared response of one detail band, on a single grey map."""
    squared = (band * band).sum(dim=1, keepdim=True)
    squared = F.pad(squared, (_ENERGY_WINDOW // 2,) * 4, mode='reflect')
    return F.conv2d(squared, box_kernel)


def _laplacian_pyramid(img, levels, kernel):
    """Return (detail levels 0..levels-1, coarsest Gaussian base)."""
    gaussian = [img]
    for _ in range(levels):
        gaussian.append(_pyr_down(gaussian[-1], kernel))
    detail = [gaussian[i] - _pyr_up(gaussian[i + 1], gaussian[i].shape[2:], kernel)
              for i in range(levels)]
    return detail, gaussian[levels]


def pyramid_torch_impl(input_source, img_resize=None, levels=None, device=None):
    """
    Laplacian-pyramid fusion on a torch device.

    Args:
        input_source: Directory path or list of BGR uint8/uint16 arrays
        img_resize: Optional (width, height) target size
        levels: Number of pyramid decomposition levels (None -> method default)
        device: 'cuda', 'mps', or None to auto-select

    Returns:
        Fused BGR image at the stack's own depth
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            raise RuntimeError("No GPU device (CUDA/MPS) available for pyramid fusion")

    stack_ori = _load_stack(input_source)
    if not stack_ori:
        raise ValueError("No image data was loaded")

    out_dtype = bitdepth.stack_dtype(stack_ori)

    if img_resize:
        cols, rows = int(img_resize[0]), int(img_resize[1])
    else:
        rows, cols = stack_ori[0].shape[:2]
    if min(rows, cols) < 3:
        raise ValueError(
            f"Image {cols}x{rows} too small for GPU pyramid fusion (needs >= 3 px per side)"
        )

    levels = _resolve_levels(rows, cols, levels)
    dev = torch.device(device)

    with torch.no_grad():
        kernel = _pyr_kernel(3, dev)
        box_kernel = torch.full((1, 1, _ENERGY_WINDOW, _ENERGY_WINDOW),
                                1.0 / _ENERGY_WINDOW ** 2, device=dev)

        fused_detail = None
        best_energy = None
        base_accumulator = None
        base_weight = None

        # Frames stream through in index order; the strict '>' below keeps the
        # first-frame-wins tie-break of the CPU implementation.
        for frame in stack_ori:
            if img_resize and (frame.shape[1], frame.shape[0]) != (cols, rows):
                frame = cv2.resize(frame, (cols, rows))
            img = torch_depth.image_to_float01(frame, dev)

            detail, base = _laplacian_pyramid(img, levels, kernel)
            del img

            if fused_detail is None:
                fused_detail = [torch.zeros_like(band) for band in detail]
                best_energy = [torch.full(band.shape[:1] + (1,) + band.shape[2:],
                                          -torch.inf, device=dev)
                               for band in detail]
                base_accumulator = torch.zeros_like(base)
                base_weight = torch.zeros((1, 1) + base.shape[2:], device=dev)

            # Detail energies cascade down to the base grid and become the
            # frame's per-pixel weight in the coarse band (see pyramid.py).
            activity = None
            for i, band in enumerate(detail):
                energy = _box_energy(band, box_kernel)
                better = energy > best_energy[i]
                best_energy[i] = torch.where(better, energy, best_energy[i])
                fused_detail[i] = torch.where(better, band, fused_detail[i])
                activity = energy if activity is None else activity + energy
                # Next band's grid is the same (h+1)//2 halving _pyr_down
                # produces, so no explicit dstsize is needed here.
                activity = _pyr_down(activity, kernel[:1])
            weight = activity + _BASE_ACTIVITY_EPS
            base_accumulator += base * weight
            base_weight += weight
            del detail, base, activity

        fused = base_accumulator / base_weight
        del base_accumulator, base_weight, best_energy
        for i in range(levels - 1, -1, -1):
            fused = fused_detail[i] + _pyr_up(fused, fused_detail[i].shape[2:], kernel)
            fused_detail[i] = None

    return torch_depth.from_float01(fused.squeeze(0).permute(1, 2, 0), out_dtype)
