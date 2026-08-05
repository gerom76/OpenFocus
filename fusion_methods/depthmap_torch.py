"""
GPU implementation of depth-map multi-focus fusion.

Mirrors fusion_methods/depthmap.py on a torch device (CUDA/MPS): the same
squared-Laplacian focus energy, taken through the same measurement prefilter
and pooled over the same two windows into the same geometric mean, the same
order-independent per-pixel argmax (MODE_MAX), the same contrast-weighted blend
with its baseline weight (MODE_AVERAGE), and halo suppression through the same
elliptical structuring element - reproduced span by span rather than
approximated with a square window, so a radius means on the GPU what it means
on the CPU.

Frames are uploaded and measured in chunks, so device memory is bounded by the
accumulators plus one chunk however deep the stack is. The chunk is sized from
the VRAM the card actually reports free rather than pinned to a constant, and
halves itself and retries if the allocator still runs out.

Two deliberate departures from the CPU path, both confined to the frame border:
cv2 pools the energy with BORDER_REFLECT while torch's reflect padding is
BORDER_REFLECT_101, and the chunk's contributions in MODE_AVERAGE are summed
as a batch rather than one frame at a time. Neither is visible away from the
outermost `window // 2` pixels, and both are deterministic run to run.
"""

import functools
import math

import cv2
import torch
import torch.nn.functional as F

from fusion_methods import torch_depth
from fusion_methods.depthmap import (
    MEASURE_BLUR_KSIZE,
    MEASURE_SIGMA,
    MODE_AVERAGE,
    MODE_MAX,
    _BASELINE_FRACTION,
    _WEIGHT_FLOOR,
    _near_window,
    _resolve_halo_radius,
    _resolve_kernel,
)
from fusion_methods.gff_torch import _load_stack
from utils import bitdepth

# Frames uploaded and measured per device batch. The batch is what makes the
# GPU worth the transfer at all - a 3x3 convolution over one frame does not
# fill a modern card - but the chunk is also the bulk of the run's device
# memory, so it is derived from free VRAM instead of pinned to a constant.
CHUNK_MIN = 1
CHUNK_MAX = 16

# What to use when the device cannot be asked how much memory it has free
# (MPS has no equivalent of cuda.mem_get_info). Matches gff_torch's constant.
CHUNK_FALLBACK = 4

# Share of the free VRAM the in-flight chunk may claim. The rest covers the
# accumulators, the workspace cuDNN picks for the convolutions, and whatever
# the caching allocator is still holding from an earlier render.
VRAM_SHARE = 0.5


def _frame_bytes(rows, cols, channels, radius):
    """Device bytes one frame of a chunk costs while it is being measured.

    Conservative and deliberately rough - it decides a batch size, not a
    correctness bound, and the allocator retry below covers an underestimate.
    The normalised frame, the prefilter's two passes over it and its Laplacian
    response carry `channels` planes each and the padded copy each convolution
    makes carries a little more; the energy map and the two pooled outputs the
    measure combines are single-channel, as are the row maxima the halo
    dilation builds.
    """
    planes = 4.5 * channels + 4.0
    if radius > 0:
        planes += 3.0
    return max(1, int(4 * rows * cols * planes))


def _chunk_size(rows, cols, channels, radius, dev):
    """Frames per device batch, from what this card reports free."""
    if dev.type != "cuda":
        return CHUNK_FALLBACK
    try:
        free, _total = torch.cuda.mem_get_info(dev)
    except Exception:
        # Older torch, or a driver that will not answer; the retry loop in
        # depthmap_torch_impl is what makes guessing safe.
        return CHUNK_FALLBACK
    affordable = int(free * VRAM_SHARE) // _frame_bytes(rows, cols, channels, radius)
    return int(max(CHUNK_MIN, min(CHUNK_MAX, affordable)))


# The filter taps below are constant per device and per window, and a render
# rebuilds them once per tile otherwise. They are kilobytes at most, so holding
# them past the render costs nothing measurable next to the image buffers
# core.memory.release_render_memory() reclaims.
@functools.lru_cache(maxsize=8)
def _laplacian_kernel(channels, device):
    """cv2.Laplacian's ksize=3 taps as a depthwise (C, 1, 3, 3) filter.

    Not the four-neighbour [[0,1,0],[1,-4,1],[0,1,0]] stencil: for ksize=3
    OpenCV uses the Sobel-derived [[2,0,2],[0,-8,0],[2,0,2]], and the CPU path
    asks for ksize=3. The kernel is symmetric, so correlating with it (what
    conv2d does) and convolving agree.
    """
    taps = torch.tensor([[2.0, 0.0, 2.0],
                         [0.0, -8.0, 0.0],
                         [2.0, 0.0, 2.0]], dtype=torch.float32, device=device)
    return taps.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()


@functools.lru_cache(maxsize=8)
def _box_kernel(window, device):
    """(1, 1, w, w) normalised box filter, the twin of cv2.boxFilter."""
    return torch.full((1, 1, window, window), 1.0 / (window * window),
                      dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=8)
def _blur_kernel(channels, device):
    """The measurement prefilter's taps as a depthwise separable (1, k) filter.

    Built from cv2.getGaussianKernel rather than from a formula of our own, so
    the GPU convolves with bit-for-bit the coefficients cv2.GaussianBlur uses
    on the CPU path.
    """
    taps = cv2.getGaussianKernel(MEASURE_BLUR_KSIZE, MEASURE_SIGMA).ravel()
    row = torch.tensor(taps, dtype=torch.float32, device=device)
    return row.view(1, 1, 1, -1).expand(channels, 1, 1, MEASURE_BLUR_KSIZE).contiguous()


def _measure_blur(batch, blur_kernel):
    """Separable Gaussian prefilter over a (B, C, H, W) chunk.

    Separated into two passes for the same reason cv2 does it: a k-tap square
    kernel costs k times as much as two k-tap passes for the same result.
    Reflect padding matches cv2.GaussianBlur's default border.
    """
    channels = batch.shape[1]
    pad = MEASURE_BLUR_KSIZE // 2
    out = F.conv2d(F.pad(batch, (pad, pad, 0, 0), mode="reflect"),
                   blur_kernel, groups=channels)
    return F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"),
                    blur_kernel.transpose(2, 3), groups=channels)


@functools.lru_cache(maxsize=16)
def _ellipse_spans(radius):
    """cv2's MORPH_ELLIPSE element as (half-width, rows sharing it) pairs.

    getStructuringElement fills row `dy` of a (2r+1)-square ellipse out to
    round(sqrt(r*r - dy*dy)) either side of the centre. Reproducing that
    exactly is what keeps the GPU dilation identical to cv2.dilate rather than
    merely similar - a square window would over-claim the corners and pull the
    halo band out to r*sqrt(2). Rows that share a half-width can share one
    horizontal maximum, so they are grouped.
    """
    spans = {}
    for dy in range(-radius, radius + 1):
        half = min(radius, int(round(math.sqrt(radius * radius - dy * dy))))
        spans.setdefault(half, []).append(dy)
    return tuple((half, tuple(rows)) for half, rows in spans.items())


def _dilate_ellipse(energy, radius):
    """Grey-dilate a (B, 1, H, W) energy map by cv2's elliptical element.

    Decomposed into one horizontal maximum per distinct row half-width, each
    then read at the row offsets that share it: the 2r+1 rows of the ellipse
    cost r+1 pooling passes rather than one (2r+1)-square window. Padded with
    -inf so pixels outside the frame can never win, which is what cv2.dilate's
    default border value does.
    """
    if radius <= 0:
        return energy

    height, width = energy.shape[-2:]
    padded = F.pad(energy, (radius,) * 4, mode="constant", value=float("-inf"))
    out = torch.full_like(energy, float("-inf"))

    for half, offsets in _ellipse_spans(radius):
        row_max = F.max_pool2d(padded, (1, 2 * half + 1), stride=1)
        left = radius - half
        for dy in offsets:
            top = radius + dy
            out = torch.maximum(out, row_max[:, :, top:top + height,
                                             left:left + width])
        del row_max

    return out


def _pool(energy, window, kernel):
    """Box-pool a (B, 1, H, W) energy map, the twin of depthmap._pool."""
    if window <= 1:
        return energy
    pad = window // 2
    return F.conv2d(F.pad(energy, (pad,) * 4, mode="reflect"), kernel)


def _focus_energy(batch, window, near, lap_kernel, box_kernel, near_kernel,
                  blur_kernel):
    """(B, 1, H, W) focus measure of a (B, C, H, W) chunk.

    Channels are squared and summed before the pooling, exactly as on the CPU,
    so all three channels of a pixel stay tied to one frame and colour can
    never split across sources, and the same two poolings are combined as the
    same geometric mean; see depthmap._focus_energy for what the narrow window
    is for.
    """
    channels = batch.shape[1]
    if blur_kernel is not None:
        batch = _measure_blur(batch, blur_kernel)
    lap = F.conv2d(F.pad(batch, (1, 1, 1, 1), mode="reflect"),
                   lap_kernel, groups=channels)
    energy = (lap * lap).sum(dim=1, keepdim=True)
    del lap

    wide = _pool(energy, window, box_kernel)
    if not near:
        return wide
    return torch.sqrt(wide * _pool(energy, near, near_kernel))


def _resolve_device(device):
    """The device to run on, auto-selecting CUDA then MPS when none is named."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No GPU device (CUDA/MPS) available for depth-map fusion")


def _fuse_on_device(stack, mode, window, radius, dev, chunk, rows, cols,
                    channels, img_resize):
    """One full pass over the stack on `dev`, returning a (C, H, W) tensor."""
    lap_kernel = _laplacian_kernel(channels, dev)
    box_kernel = _box_kernel(window, dev) if window > 1 else None
    near = _near_window(window)
    near_kernel = _box_kernel(near, dev) if near else None
    blur_kernel = _blur_kernel(channels, dev) if MEASURE_SIGMA > 0 else None
    count = len(stack)

    def upload(frames):
        if img_resize:
            frames = [cv2.resize(f, (cols, rows))
                      if (f.shape[1], f.shape[0]) != (cols, rows) else f
                      for f in frames]
        return torch_depth.stack_to_float01(frames, dev)

    def measure(batch):
        energy = _focus_energy(batch, window, near, lap_kernel, box_kernel,
                               near_kernel, blur_kernel)
        return _dilate_ellipse(energy, radius)

    if mode == MODE_MAX:
        best = torch.full((rows, cols), float("-inf"), device=dev)
        fused = torch.zeros((channels, rows, cols), device=dev)

        for start in range(0, count, chunk):
            batch = upload(stack[start:start + chunk])
            energy = measure(batch)
            # Reduced one frame at a time even though the measurement is
            # batched: the strict '>' walking the stack in index order is what
            # gives the CPU path its first-frame-wins tie-break, and a batched
            # argmax would quietly resolve ties the other way. Written back
            # through `out=` so the two accumulators are updated in place -
            # allocating a fresh copy of the fused frame per slice is what
            # would make a deep chunk expensive.
            for i in range(batch.shape[0]):
                better = energy[i, 0] > best
                torch.where(better, energy[i, 0], best, out=best)
                torch.where(better.unsqueeze(0), batch[i], fused, out=fused)
            del batch, energy

        return fused

    # MODE_AVERAGE: the contrast-weighted sum, the weight sum and a plain sum,
    # blended once the stack's mean energy - and with it the baseline weight
    # that makes flat regions average - is known.
    weighted = torch.zeros((channels, rows, cols), device=dev)
    plain = torch.zeros((channels, rows, cols), device=dev)
    weight_sum = torch.zeros((rows, cols), device=dev)

    for start in range(0, count, chunk):
        batch = upload(stack[start:start + chunk])
        energy = measure(batch)
        plain += batch.sum(dim=0)
        weight_sum += energy.sum(dim=(0, 1))
        # The chunk is this iteration's own upload and is dead after the
        # weighting, so it doubles as the scratch - `(batch * energy).sum(0)`
        # would allocate a second copy of the whole chunk at the peak.
        batch *= energy
        weighted += batch.sum(dim=0)
        del batch, energy

    mean_energy = float(weight_sum.sum()) / max(1, count * rows * cols)
    baseline = max(mean_energy * _BASELINE_FRACTION, _WEIGHT_FLOOR)

    plain *= baseline
    weighted += plain
    del plain
    weight_sum += baseline * count
    weighted /= weight_sum.unsqueeze(0)
    return weighted


def depthmap_torch_impl(input_source, img_resize=None, mode=MODE_MAX,
                        kernel_size=None, halo_radius=None, device=None,
                        chunk_size=None):
    """
    Depth-map fusion on a torch device.

    Args:
        input_source: Directory path or list of BGR uint8/uint16 arrays
        img_resize: Optional (width, height) target size
        mode: 'max' for the hard per-pixel select, 'average' for the
              contrast-weighted blend
        kernel_size: Side of the focus-measure pooling window (odd)
        halo_radius: Halo-suppression radius in pixels; 0/None disables it
        device: 'cuda', 'mps', 'cpu', or None to auto-select a GPU
        chunk_size: Frames per device batch; None sizes it from free VRAM

    Every dial means what it means on the CPU path; see
    fusion_methods/depthmap.py for what each one is for.

    Returns:
        Fused BGR image at the stack's own depth
    """
    if mode not in (MODE_MAX, MODE_AVERAGE):
        raise ValueError(f"Unknown depth-map mode {mode!r}; "
                         f"expected {MODE_MAX!r} or {MODE_AVERAGE!r}")

    dev = _resolve_device(device)
    window = _resolve_kernel(kernel_size)
    radius = _resolve_halo_radius(halo_radius)

    stack_ori = _load_stack(input_source)
    if not stack_ori:
        raise ValueError("No image data was loaded")

    first = stack_ori[0]
    if first.ndim != 3 or first.shape[2] != 3:
        # The upload path is BGR-only. Raising hands the stack back to the CPU
        # implementation, which handles single-channel frames.
        raise ValueError("GPU depth-map fusion requires 3-channel BGR frames")

    out_dtype = bitdepth.stack_dtype(stack_ori)
    channels = first.shape[2]

    if img_resize:
        cols, rows = int(img_resize[0]), int(img_resize[1])
    else:
        rows, cols = first.shape[:2]

    # Reflect padding requires pad < dim; below this size the CPU path (whose
    # cv2 borders have no such limit) is the one that can do the work. The
    # measurement prefilter pads too, so it sets the floor for small windows.
    max_pad = max(1, window // 2, MEASURE_BLUR_KSIZE // 2 if MEASURE_SIGMA > 0 else 1)
    if min(rows, cols) <= max_pad:
        raise ValueError(
            f"Image {cols}x{rows} too small for GPU depth-map fusion with a "
            f"{window} px window (needs > {max_pad} px per side)"
        )

    chunk = (max(1, int(chunk_size)) if chunk_size
             else _chunk_size(rows, cols, channels, radius, dev))

    while True:
        try:
            with torch.no_grad():
                fused = _fuse_on_device(stack_ori, mode, window, radius, dev,
                                        chunk, rows, cols, channels, img_resize)
            break
        except torch.cuda.OutOfMemoryError:
            # Sizing from free VRAM can still be beaten by another process
            # taking the card mid-render, so the chunk gives ground rather than
            # failing the fusion outright.
            torch.cuda.empty_cache()
            if chunk <= CHUNK_MIN:
                raise
            chunk = max(CHUNK_MIN, chunk // 2)
            print(f"Warning: CUDA out of memory in depth-map fusion; "
                  f"retrying with chunk_size={chunk}.")

    return torch_depth.from_float01(fused.permute(1, 2, 0), out_dtype)
