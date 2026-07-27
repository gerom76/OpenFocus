"""
GPU implementation of Laplacian-pyramid multi-focus fusion.

Mirrors fusion_methods/pyramid.py on a torch device (CUDA/MPS): the same
pooled band energy, the same noise gate, the same peak-normalised weighting and
the same envelope clamp, with every dial carrying the same meaning. Frames
stream through one at a time, so device memory is bounded by the fused
accumulators plus a single frame's pyramid regardless of stack size.

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
    NOISE_FLOOR, NOISE_FLOOR_RATIO, NOISE_PERCENTILE,
    _resolve_coherence, _resolve_exponent, _resolve_levels, _resolve_window,
    BASE_SELECTIVITY, ENVELOPE_CLIP, NOISE_GATE, SELECTIVITY,
)
from utils import bitdepth

# The 1-D half of OpenCV's fixed pyramid kernel: [1, 4, 6, 4, 1] / 16.
_PYR_TAPS = (1.0, 4.0, 6.0, 4.0, 1.0)


def _pyr_kernel(channels, device):
    """(C, 1, 5, 5) depthwise Burt-Adelson kernel, normalised to sum 1."""
    taps = torch.tensor(_PYR_TAPS, dtype=torch.float32, device=device) / 16.0
    k2d = torch.outer(taps, taps)
    return k2d.expand(channels, 1, 5, 5)


def _pyr_down(x, kernel, size=None):
    """cv2.pyrDown: blur with the pyramid kernel, keep even coordinates.

    `size` names the destination explicitly, for the one caller that has to
    land on a grid it did not derive itself.
    """
    x = F.pad(x, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(x, kernel, stride=2, groups=x.shape[1])
    if size is not None and out.shape[2:] != tuple(size):
        out = out[:, :, :size[0], :size[1]]
    return out


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


def _box_energy(band, box_kernel, window):
    """Pooled squared response of one detail band, on a single grey map."""
    squared = (band * band).sum(dim=1, keepdim=True)
    if window <= 1:
        return squared
    squared = F.pad(squared, (window // 2,) * 4, mode='reflect')
    return F.conv2d(squared, box_kernel)


def _noise_level(energy):
    """The level below which this band of this frame resolves nothing.

    The same subsampled low percentile as the CPU path, taken with the same
    stride so the two agree on the number: torch.quantile sorts, and sorting a
    4K band per frame per level would cost more than the pyramid it measures.
    """
    step = max(1, int((energy.numel() / 65536.0) ** 0.5))
    sample = energy[:, :, ::step, ::step].reshape(-1)
    live = sample[sample > 0.0]
    if live.numel() == 0:
        return NOISE_FLOOR
    cuts = torch.tensor([NOISE_PERCENTILE / 100.0, 0.5], device=live.device)
    level, middle = torch.quantile(live, cuts)
    return torch.clamp(torch.maximum(level, middle * NOISE_FLOOR_RATIO),
                       min=NOISE_FLOOR)


def _mix_parent(energy, parent, coherence):
    """Weighted geometric mean of a band's energy and its parent's salience."""
    if coherence >= 1.0:
        return parent
    if coherence == 0.5:
        return torch.sqrt(energy * parent)
    return energy.pow(1.0 - coherence) * parent.pow(coherence)


def _laplacian_pyramid(img, levels, kernel):
    """Return (detail levels 0..levels-1, coarsest Gaussian base)."""
    gaussian = [img]
    for _ in range(levels):
        gaussian.append(_pyr_down(gaussian[-1], kernel))
    detail = [gaussian[i] - _pyr_up(gaussian[i + 1], gaussian[i].shape[2:], kernel)
              for i in range(levels)]
    return detail, gaussian[levels]


class _Accumulator:
    """Running weighted sum of one band across the stack.

    The device twin of pyramid._Accumulator, including the rescale that lets a
    peak-relative weighting be computed in one streaming pass - see the comment
    there for why that is exact rather than an approximation.
    """

    __slots__ = ("total", "weight", "peak", "exponent")

    def __init__(self, band, exponent):
        self.exponent = exponent
        self.total = torch.zeros_like(band)
        shape = (band.shape[0], 1) + tuple(band.shape[2:])
        self.weight = torch.zeros(shape, device=band.device)
        self.peak = torch.full(shape, -torch.inf if exponent == float('inf') else 0.0,
                               device=band.device)

    def add(self, band, salience):
        if self.exponent == float('inf'):
            better = salience > self.peak
            self.peak = torch.where(better, salience, self.peak)
            self.total = torch.where(better, band, self.total)
            self.weight = torch.where(better, torch.ones_like(self.weight), self.weight)
            return

        peak = torch.maximum(self.peak, salience)
        safe = peak.clamp(min=NOISE_FLOOR)
        if self.exponent > 0:
            rescale = (self.peak / safe).pow(self.exponent)
            self.total *= rescale
            self.weight *= rescale
            weight = (salience / safe).pow(self.exponent)
        else:
            weight = torch.ones_like(salience)
        self.total += band * weight
        self.weight += weight
        self.peak = peak

    def result(self):
        return self.total / torch.where(self.weight > 0.0, self.weight,
                                        torch.ones_like(self.weight))


def pyramid_torch_impl(input_source, img_resize=None, levels=None, device=None,
                       energy_window=None, selectivity=None, coherence=None,
                       noise_gate=None, base_selectivity=None, envelope=None):
    """
    Laplacian-pyramid fusion on a torch device.

    Args:
        input_source: Directory path or list of BGR uint8/uint16 arrays
        img_resize: Optional (width, height) target size
        levels: Number of pyramid decomposition levels (None -> method default)
        device: 'cuda', 'mps', or None to auto-select
        energy_window: Window the band energy is pooled over before comparing
        selectivity: How sharply the weights favour the sharpest frame
        coherence: How much of a band's decision comes from the coarser bands
        noise_gate: Compare frames in units of their own noise
        base_selectivity: The weighting exponent for the coarse base band
        envelope: Clamp the result to the range its own frames span

    Every dial means what it means on the CPU path; see fusion_methods/pyramid.py
    for what each one is for.

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

    window = _resolve_window(energy_window)
    selectivity = _resolve_exponent(selectivity, SELECTIVITY)
    base_selectivity = _resolve_exponent(base_selectivity, BASE_SELECTIVITY)
    coherence = _resolve_coherence(coherence)
    noise_gate = NOISE_GATE if noise_gate is None else bool(noise_gate)
    envelope = ENVELOPE_CLIP if envelope is None else bool(envelope)

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
        grey_kernel = kernel[:1]
        box_kernel = torch.full((1, 1, window, window),
                                1.0 / (window * window), device=dev)

        detail_acc = None
        base_acc = None
        lowest = highest = None

        # Frames stream through in index order; the strict '>' in the
        # winner-take-all path keeps the first-frame-wins tie-break of the CPU
        # implementation.
        for frame in stack_ori:
            if img_resize and (frame.shape[1], frame.shape[0]) != (cols, rows):
                frame = cv2.resize(frame, (cols, rows))
            img = torch_depth.image_to_float01(frame, dev)

            detail, base = _laplacian_pyramid(img, levels, kernel)
            if envelope:
                if lowest is None:
                    lowest, highest = img, img.clone()
                else:
                    lowest = torch.minimum(lowest, img)
                    highest = torch.maximum(highest, img)
            del img

            if detail_acc is None:
                detail_acc = [_Accumulator(band, selectivity) for band in detail]
                base_acc = _Accumulator(base, base_selectivity)

            energies = [_box_energy(band, box_kernel, window) for band in detail]
            if noise_gate:
                energies = [e / _noise_level(e) for e in energies]

            # Detail energies cascade down to the base grid and become the
            # frame's per-pixel weight in the coarse band (see pyramid.py).
            activity = None
            for i, energy in enumerate(energies):
                activity = energy if activity is None else activity + energy
                shape = (energies[i + 1].shape[2:] if i + 1 < levels
                         else base.shape[2:])
                activity = _pyr_down(activity, grey_kernel, shape)

            if coherence > 0.0:
                for i in range(levels - 2, -1, -1):
                    parent = _pyr_up(energies[i + 1], energies[i].shape[2:],
                                     grey_kernel)
                    energies[i] = _mix_parent(energies[i], parent, coherence)

            for i, band in enumerate(detail):
                detail_acc[i].add(band, energies[i])
            base_acc.add(base, activity)

            del detail, base, energies, activity

        fused = base_acc.result()
        base_acc = None
        for i in range(levels - 1, -1, -1):
            fused = detail_acc[i].result() + _pyr_up(fused,
                                                     detail_acc[i].total.shape[2:],
                                                     kernel)
            detail_acc[i] = None

        if envelope and lowest is not None:
            # Clamped in [0, 1] rather than in levels as the CPU path does;
            # rounding is monotonic, so the quantised result still lands inside
            # the source levels.
            fused = torch.minimum(torch.maximum(fused, lowest), highest)
            del lowest, highest

    return torch_depth.from_float01(fused.squeeze(0).permute(1, 2, 0), out_dtype)
