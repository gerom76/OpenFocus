"""
GPU implementation of GFG-FGF multi-focus fusion (Fu et al., 2024).

Mirrors fusion_methods/gfg_fgf.py but runs batched on a torch device
(CUDA/MPS). The stack is processed in fixed-size chunks to bound GPU memory
usage. Three passes over the stack are needed because each stage depends on a
global reduction of the previous one:

    1. Focus measure (Scharr energy) -> per-image scores, active/skip mask.
    2. Activity focus maps (AFMs) -> per-pixel initial decision map (argmax).
    3. Guided-filter-refined weights -> weighted accumulation and normalize.

Reference:
Fu Hongyu, Gong Yan, Wang Luhan, et al. Multi-focus microscopy image fusion
algorithm[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022.
"""

import os
import re
import cv2
import torch
import torch.nn.functional as F

from fusion_methods import torch_depth
from utils import bitdepth
from utils.image_utils import read_image_any_depth

# Parameters matching the CPU implementation in gfg_fgf.py
SCALE = 1e-6          # skip only frames with essentially no gradient energy
G_GSZ = 5             # guided-filter radius for weight refinement
G_EPS = 0.3           # guided-filter regularization
THRESHOLD = 0.005     # TOZERO threshold on the local-contrast map

# Number of stack images processed per GPU batch (bounds peak memory)
CHUNK_SIZE = 4


def _load_stack(input_source):
    """Load a stack from a directory path or pass through a list of arrays."""
    if not isinstance(input_source, str):
        return input_source

    if not os.path.exists(input_source):
        raise ValueError(f"Path not found: {input_source}")

    filenames = os.listdir(input_source)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    img_paths = [
        os.path.join(input_source, f) for f in filenames
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    if not img_paths:
        raise ValueError("No images found in folder")

    try:
        img_paths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]))
    except Exception:
        img_paths.sort()

    return [img for img in (read_image_any_depth(p) for p in img_paths) if img is not None]


def _box_filter(x, radius):
    """Normalized box (mean) filter via separable convolution with reflect padding.

    x: (B, C, H, W) float32 tensor. Window size is 2*radius+1, matching
    cv2.blur/cv2.boxFilter with an odd ksize.
    """
    k = 2 * radius + 1
    c = x.shape[1]
    kernel = torch.full((c, 1, 1, k), 1.0 / k, dtype=x.dtype, device=x.device)
    x = F.pad(x, (radius, radius, 0, 0), mode='reflect')
    x = F.conv2d(x, kernel, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode='reflect')
    x = F.conv2d(x, kernel.transpose(2, 3), groups=c)
    return x


def _guided_filter(I, p, radius, eps):
    """Guided filter for single-channel guide I and input p, batched. (B, 1, H, W)."""
    mean_I = _box_filter(I, radius)
    mean_p = _box_filter(p, radius)
    mean_Ip = _box_filter(I * p, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(I * I, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    return mean_a * I + mean_b


def gfgfgf_torch_impl(input_source, img_resize=None, kernel_size=7, device=None):
    """
    GFG-FGF fusion on a torch device.

    Args:
        input_source: Directory path or list of BGR uint8 arrays (H, W, 3)
        img_resize: Optional (width, height) target size
        kernel_size: Mean-filter kernel size for the local-contrast map (odd)
        device: 'cuda', 'mps', or None to auto-select

    Returns:
        Fused BGR uint8 image (H, W, 3)
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            raise RuntimeError("No GPU device (CUDA/MPS) available for GFG-FGF fusion")

    blur_size = kernel_size if kernel_size is not None and kernel_size > 0 else 7
    if blur_size % 2 == 0:
        blur_size += 1
    blur_radius = blur_size // 2

    stack_ori = _load_stack(input_source)
    if not stack_ori:
        raise ValueError("Input stack is empty")
    if img_resize is not None:
        if (stack_ori[0].shape[1], stack_ori[0].shape[0]) != img_resize:
            stack_ori = [cv2.resize(img, img_resize) for img in stack_ori]

    # The result is written back at the depth the stack arrived in.
    out_dtype = bitdepth.stack_dtype(stack_ori)

    n = len(stack_ori)
    h, w = stack_ori[0].shape[:2]

    # Reflect padding requires pad < dim; below this size use the CPU path instead
    max_pad = max(G_GSZ, blur_radius, 1)
    if min(h, w) <= max_pad:
        raise ValueError(
            f"Image {w}x{h} too small for GPU GFG-FGF fusion (needs > {max_pad} px per side)"
        )

    dev = torch.device(device)

    with torch.no_grad():
        # Scharr-like focus kernels (match gfg_fgf.py kx/ky exactly)
        kx = torch.tensor(
            [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], device=dev
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], device=dev
        ).view(1, 1, 3, 3)

        def to_device(chunk):
            # (B, 3, H, W) BGR in [0, 1], from uint8 or uint16 alike
            return torch_depth.stack_to_float01(chunk, dev)

        # Guide image for the guided filter: luminance, not a single colour
        # channel. Detail is measured across all channels below, so structure
        # living in one channel only is still picked up (matches CPU).
        bgr_to_gray = torch.tensor([0.114, 0.587, 0.299], device=dev).view(1, 3, 1, 1)

        def guide_of(t):
            return (t * bgr_to_gray).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # -- Pass 1: focus measure (mean Scharr energy, 1px border ignored) ----
        # Run per channel and keep the strongest response at each pixel.
        kx3 = kx.expand(3, 1, 3, 3).contiguous()
        ky3 = ky.expand(3, 1, 3, 3).contiguous()
        focus_vals = torch.empty(n, device=dev)
        for start in range(0, n, CHUNK_SIZE):
            chunk = stack_ori[start:start + CHUNK_SIZE]
            t = to_device(chunk)
            tp = F.pad(t, (1, 1, 1, 1), mode='reflect')
            gx = F.conv2d(tp, kx3, groups=3)
            gy = F.conv2d(tp, ky3, groups=3)
            energy = (gx * gx + gy * gy)[:, :, 1:-1, 1:-1]
            focus_vals[start:start + len(chunk)] = energy.amax(dim=1).mean(dim=(1, 2))

        max_focus = float(focus_vals.max()) if n > 0 else 1.0
        if max_focus == 0:
            max_focus = 1.0
        # Only blank/dead frames are dropped; the per-pixel argmax decides the
        # rest, so a mostly-defocused frame can still own its one sharp region.
        active = focus_vals > (SCALE * max_focus)  # (n,) bool

        # -- Pass 2: activity focus maps -> initial decision map ----------------
        afms = torch.zeros((n, h, w), device=dev)
        for start in range(0, n, CHUNK_SIZE):
            chunk = stack_ori[start:start + CHUNK_SIZE]
            b = len(chunk)
            t = to_device(chunk)
            g = guide_of(t)

            src_blur = _box_filter(t, blur_radius)
            src_diff = (t - src_blur).abs()
            # Collapse the per-channel local contrast by taking the strongest channel
            activity = src_diff.amax(dim=1, keepdim=True)
            gfg_map = activity * (activity > THRESHOLD)  # THRESH_TOZERO
            afm = _guided_filter(g, gfg_map, G_GSZ, G_EPS)

            # Low-focus images stay zero so they never win the argmax
            afm = afm * active[start:start + b].view(b, 1, 1, 1)
            afms[start:start + b] = afm[:, 0]

        idm = afms.argmax(dim=0)  # (H, W) index of the sharpest image per pixel
        del afms

        # -- Pass 3: refine per-image weights and accumulate -------------------
        num = torch.zeros((3, h, w), device=dev)
        den = torch.zeros((h, w), device=dev)
        for start in range(0, n, CHUNK_SIZE):
            chunk = stack_ori[start:start + CHUNK_SIZE]
            b = len(chunk)
            t = to_device(chunk)
            g = guide_of(t)

            ks = torch.arange(start, start + b, device=dev).view(b, 1, 1)
            masks = (idm.unsqueeze(0) == ks).float().unsqueeze(1)  # (B, 1, H, W)
            masks = masks * active[start:start + b].view(b, 1, 1, 1)

            fdm = _guided_filter(g, masks, G_GSZ, G_EPS)

            num += (t * fdm).sum(dim=0)       # (3, H, W)
            den += fdm.sum(dim=(0, 1))        # (H, W)

        den.clamp_(min=1e-6)
        fused = num / den.unsqueeze(0)

        # (H, W, 3) BGR, back at the depth the stack arrived in
        return torch_depth.from_float01(fused.permute(1, 2, 0), out_dtype)
