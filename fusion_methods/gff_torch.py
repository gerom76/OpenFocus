"""
GPU implementation of guided-filter multi-focus fusion (Li et al., 2013).

Mirrors fusion_methods/gff.py but runs batched on a torch device (CUDA/MPS).
The stack is processed in fixed-size chunks to bound GPU memory usage.

Reference:
Li S, Kang X, Hu J. Image fusion with guided filtering[J].
IEEE Transactions on Image Processing, 2013, 22(7): 2864-2875.
"""

import os
import glob
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Parameters matching the CPU implementation in gff.py
DEFAULT_R1 = 45
DEFAULT_R2 = 7
DEFAULT_EPS1 = 0.3
DEFAULT_EPS2 = 10e-6
DEFAULT_SIGMA_R = 5

# cv2.GaussianBlur with ksize=(0,0) derives the kernel size from sigma:
# for CV_32F it is round(sigma*4*2+1)|1 = 41 for sigma=5
GAUSS_KSIZE = 41

# Number of stack images processed per GPU batch (bounds peak memory)
CHUNK_SIZE = 4


def _load_stack(input_source):
    """Load a stack from a directory path or pass through a list of arrays."""
    if not isinstance(input_source, str):
        return input_source

    filenames = os.listdir(input_source)
    if len(filenames) == 0:
        raise ValueError("Input folder is empty or no images were found")
    img_ext = os.path.splitext(filenames[0])[1]

    img_paths = glob.glob(os.path.join(input_source, '*' + img_ext))
    try:
        img_paths.sort(key=lambda x: int(str(re.findall(r"\d+", x.split(os.sep)[-1])[-1])))
    except IndexError:
        img_paths.sort()

    return [cv2.imread(p) for p in img_paths]


def _box_filter(x, radius):
    """Normalized box (mean) filter via separable convolution with reflect padding.

    x: (B, C, H, W) float32 tensor
    """
    k = 2 * radius + 1
    c = x.shape[1]
    kernel = torch.full((c, 1, 1, k), 1.0 / k, dtype=x.dtype, device=x.device)
    x = F.pad(x, (radius, radius, 0, 0), mode='reflect')
    x = F.conv2d(x, kernel, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode='reflect')
    x = F.conv2d(x, kernel.transpose(2, 3), groups=c)
    return x


def _gaussian_kernel1d(sigma, ksize, device):
    half = ksize // 2
    xs = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur(x, kernel1d):
    """Separable Gaussian blur with reflect padding. x: (B, 1, H, W)."""
    k = kernel1d.numel()
    half = k // 2
    kh = kernel1d.view(1, 1, 1, k)
    x = F.pad(x, (half, half, 0, 0), mode='reflect')
    x = F.conv2d(x, kh)
    x = F.pad(x, (0, 0, half, half), mode='reflect')
    x = F.conv2d(x, kh.transpose(2, 3))
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


def gff_torch_impl(input_source, img_resize=None, kernel_size=31, device=None):
    """
    Guided-filter fusion on a torch device.

    Args:
        input_source: Directory path or list of BGR uint8 arrays (H, W, 3)
        img_resize: Optional (width, height) target size
        kernel_size: Mean filter kernel size for base/detail decomposition (odd)
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
            raise RuntimeError("No GPU device (CUDA/MPS) available for guided-filter fusion")

    average_filter_size = kernel_size if kernel_size is not None and kernel_size > 0 else 31
    if average_filter_size % 2 == 0:
        average_filter_size += 1
    blur_radius = average_filter_size // 2

    stack_ori = _load_stack(input_source)
    if not stack_ori:
        raise ValueError("No image data was loaded")
    if img_resize:
        stack_ori = [cv2.resize(img, img_resize) for img in stack_ori]

    n = len(stack_ori)
    h, w = stack_ori[0].shape[:2]

    # Reflect padding requires pad < dim; below this size use the CPU path instead
    max_pad = max(DEFAULT_R1, blur_radius, GAUSS_KSIZE // 2)
    if min(h, w) <= max_pad:
        raise ValueError(
            f"Image {w}x{h} too small for GPU guided-filter fusion (needs > {max_pad} px per side)"
        )

    dev = torch.device(device)

    with torch.no_grad():
        gauss_k = _gaussian_kernel1d(DEFAULT_SIGMA_R, GAUSS_KSIZE, dev)
        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=dev
        ).view(1, 1, 3, 3)

        def to_device(chunk):
            arr = np.stack([np.ascontiguousarray(img) for img in chunk])
            t = torch.from_numpy(arr).to(dev)
            return t.permute(0, 3, 1, 2).float() / 255.0  # (B, 3, H, W) BGR

        # Pass 1: saliency = Gaussian(abs(Laplacian(sum of channels)))
        saliency = torch.empty((n, h, w), device=dev)
        for start in range(0, n, CHUNK_SIZE):
            chunk = stack_ori[start:start + CHUNK_SIZE]
            t = to_device(chunk)
            img_sum = t.sum(dim=1, keepdim=True)
            lap = F.conv2d(F.pad(img_sum, (1, 1, 1, 1), mode='reflect'), lap_kernel).abs()
            sal = _gaussian_blur(lap, gauss_k)
            saliency[start:start + len(chunk)] = sal[:, 0]

        # Initial decision map: index of the most salient image per pixel
        max_indices = saliency.argmax(dim=0)  # (H, W)
        del saliency

        # Pass 2: decompose, refine weights with the guided filter, accumulate
        base_num = torch.zeros((3, h, w), device=dev)
        base_den = torch.zeros((h, w), device=dev)
        detail_num = torch.zeros((3, h, w), device=dev)
        detail_den = torch.zeros((h, w), device=dev)

        for start in range(0, n, CHUNK_SIZE):
            chunk = stack_ori[start:start + CHUNK_SIZE]
            b = len(chunk)
            t = to_device(chunk)

            base = _box_filter(t, blur_radius)
            detail = t - base

            # BGR to gray guide (matches cv2.cvtColor coefficients)
            gray = 0.114 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.299 * t[:, 2:3]

            ks = torch.arange(start, start + b, device=dev).view(b, 1, 1)
            masks = (max_indices.unsqueeze(0) == ks).float().unsqueeze(1)  # (B, 1, H, W)

            weight_base = _guided_filter(gray, masks, DEFAULT_R1, DEFAULT_EPS1)
            weight_detail = _guided_filter(gray, masks, DEFAULT_R2, DEFAULT_EPS2)

            base_num += (base * weight_base).sum(dim=0)
            base_den += weight_base.sum(dim=(0, 1))
            detail_num += (detail * weight_detail).sum(dim=0)
            detail_den += weight_detail.sum(dim=(0, 1))

        base_den.clamp_(min=1e-6)
        detail_den.clamp_(min=1e-6)

        fused = base_num / base_den.unsqueeze(0) + detail_num / detail_den.unsqueeze(0)
        fused = (fused * 255.0).clamp_(0, 255).to(torch.uint8)

        return fused.permute(1, 2, 0).cpu().numpy()  # (H, W, 3) BGR
