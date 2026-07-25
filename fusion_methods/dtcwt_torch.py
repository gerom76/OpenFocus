"""
GPU implementation of DTCWT multi-focus fusion (Lewis et al., 2007).

Mirrors fusion_methods/dtcwt.py but runs on a torch device (CUDA/MPS) using
pytorch_wavelets, which implements the same Kingsbury dual-tree complex wavelet
transform with the same default filters ('near_sym_a' / 'qshift_a') as the CPU
dtcwt package.

The stack is processed in streaming fashion — one image is transformed and
fused into the running result at a time — which matches the CPU version's
sequential pairwise fusion order exactly and bounds GPU memory to ~2 images'
worth of coefficients regardless of stack size.

Reference:
Lewis J J, O'Callaghan R J, Nikolov S G, et al. Pixel- and region-based image
fusion with complex wavelets[J]. Information Fusion, 2007, 8(2): 119-130.
"""

import sys

import torch
import torch.nn.functional as F

from fusion_methods.dtcwt import _load_images
from fusion_methods import torch_depth
from utils import bitdepth

# pytorch_wavelets 1.3.0 imports pkg_resources, which setuptools >= 81 no longer
# provides. Install a minimal in-process shim exposing the one function it uses.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import types
    import importlib.resources

    def _resource_stream(package, resource_name):
        try:
            return importlib.resources.files(package).joinpath(resource_name).open('rb')
        except (ImportError, FileNotFoundError, NotADirectoryError):
            # pytorch_wavelets names its data package by string only
            # ('pytorch_wavelets.dtcwt.data'), so a frozen build ships the .npz
            # files without the package that owns them and importlib.resources
            # cannot anchor on it. Anchor on the nearest packaged ancestor and
            # treat the rest of the name as directories instead.
            parts = package.split('.')
            for cut in range(len(parts) - 1, 0, -1):
                try:
                    anchor = importlib.resources.files('.'.join(parts[:cut]))
                except (ImportError, FileNotFoundError, NotADirectoryError):
                    continue
                candidate = anchor.joinpath(*parts[cut:], resource_name)
                if candidate.is_file():
                    return candidate.open('rb')
            raise

    _shim = types.ModuleType('pkg_resources')
    _shim.resource_stream = _resource_stream
    sys.modules['pkg_resources'] = _shim

WINDOW_SIZE = 3

# Mirrors _LOWPASS_ACTIVITY_EPS in fusion_methods/dtcwt.py: keeps the
# activity-weighted lowpass average defined on flat stacks, where it degrades
# to the plain mean.
LOWPASS_ACTIVITY_EPS = 1e-6


def _lowpass_activity(yh, lowpass_shape):
    """Aggregate detail activity of one frame on the lowpass grid.

    Torch mirror of _lowpass_activity in fusion_methods/dtcwt.py: complex
    magnitudes summed over the six orientations per level, area-resampled to
    the lowpass resolution and summed across levels. yh: list of
    (C, 6, h, w, 2) tensors; returns (C, h', w').
    """
    acc = None
    for level in yh:
        mag = torch.sqrt(level[..., 0] ** 2 + level[..., 1] ** 2).sum(dim=1)
        mag = F.interpolate(mag.unsqueeze(0), size=lowpass_shape, mode='area')[0]
        acc = mag if acc is None else acc + mag
    return acc


def _activity(mag):
    """3x3 spatial maximum filter per direction. mag: (C, 6, h, w)."""
    c, d, h, w = mag.shape
    x = mag.reshape(c * d, 1, h, w)
    x = F.pad(x, (1, 1, 1, 1), mode='reflect')
    x = F.max_pool2d(x, kernel_size=WINDOW_SIZE, stride=1)
    return x.reshape(c, d, h, w)


def _fuse_pair(c1, c2):
    """Fuse two coefficient tensors of shape (C, 6, h, w, 2) (real/imag pairs).

    A single decision mask - built from the strongest channel's activity at
    each coefficient - selects all C channels together, so a pixel's colour
    cannot split across source frames (item 12 in
    docs/ALGORITHM_IMPROVEMENTS.md). Mirrors fuse_highfreq_vectorized in
    fusion_methods/dtcwt.py.
    """
    _, d, h, w, _ = c1.shape

    # Activity level: 3x3 max filter of the strongest channel's magnitude
    mag1 = torch.sqrt(c1[..., 0] ** 2 + c1[..., 1] ** 2).amax(dim=0, keepdim=True)
    mag2 = torch.sqrt(c2[..., 0] ** 2 + c2[..., 1] ** 2).amax(dim=0, keepdim=True)
    a1 = _activity(mag1)
    a2 = _activity(mag2)

    initial_mask = (a1 > a2).float()

    # Consistency verification: majority vote in a 3x3 window (zero-padded count)
    kernel = torch.ones(1, 1, WINDOW_SIZE, WINDOW_SIZE, device=c1.device)
    count = F.conv2d(initial_mask.reshape(d, 1, h, w), kernel, padding=WINDOW_SIZE // 2)
    decision = (count > (WINDOW_SIZE * WINDOW_SIZE) / 2.0).reshape(1, d, h, w)

    return torch.where(decision.unsqueeze(-1), c1, c2)


def dtcwt_torch_impl(input_source, img_resize=None, N=4, device=None):
    """
    DTCWT fusion on a torch device.

    Args:
        input_source: Directory path or list of BGR uint8 arrays
        img_resize: Optional (width, height) target size
        N: Number of wavelet decomposition levels
        device: 'cuda', 'mps', or None to auto-select

    Returns:
        Fused BGR uint8 image
    """
    try:
        from pytorch_wavelets import DTCWTForward, DTCWTInverse
    except ImportError as exc:
        raise RuntimeError(
            "GPU DTCWT fusion requires pytorch_wavelets. Install with: pip install pytorch_wavelets"
        ) from exc

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            raise RuntimeError("No GPU device (CUDA/MPS) available for DTCWT fusion")

    images = _load_images(input_source, img_resize)
    if len(images) < 2:
        raise ValueError("At least two images are required for fusion.")

    h_orig, w_orig = images[0].shape[:2]
    dev = torch.device(device)

    # The result is written back at the depth the stack arrived in.
    out_dtype = bitdepth.stack_dtype(images)

    # Same filter banks as the CPU dtcwt package defaults
    xfm = DTCWTForward(J=N, biort='near_sym_a', qshift='qshift_a').to(dev)
    ifm = DTCWTInverse(biort='near_sym_a', qshift='qshift_a').to(dev)

    with torch.no_grad():
        lowpass_sum = None
        lowpass_weight = None
        fused_highpass = None

        # The shared decision mask reduces over channels with a max, which is
        # order-invariant, so BGR order can be kept as-is (the CPU path works
        # in RGB and lands on the same mask)
        for img in images:
            # (1, 3, H, W) in [0, 1], from uint8 or uint16 alike
            t = torch_depth.image_to_float01(img, dev)

            yl, yh = xfm(t)  # yl: (1, 3, h', w'); yh: list of (1, 3, 6, h, w, 2)
            yl = yl[0]
            yh = [level[0] for level in yh]

            # Lowpass is averaged with each frame weighted by its aggregate
            # highpass activity (item 16 in docs/ALGORITHM_IMPROVEMENTS.md);
            # the weighted sum streams just like the plain sum did.
            weight = _lowpass_activity(yh, yl.shape[-2:]) + LOWPASS_ACTIVITY_EPS

            if lowpass_sum is None:
                lowpass_sum = yl * weight
                lowpass_weight = weight
                fused_highpass = yh
            else:
                lowpass_sum = lowpass_sum + yl * weight
                lowpass_weight = lowpass_weight + weight
                fused_highpass = [_fuse_pair(f, y) for f, y in zip(fused_highpass, yh)]

        fused_lowpass = (lowpass_sum / lowpass_weight).unsqueeze(0)
        fused_highpass = [level.unsqueeze(0) for level in fused_highpass]

        out = ifm((fused_lowpass, fused_highpass))[0]  # (3, H, W)
        # The inverse transform may return a padded size for odd dimensions
        out = out[:, :h_orig, :w_orig]

        # (H, W, 3) BGR, back at the depth the stack arrived in
        return torch_depth.from_float01(out.permute(1, 2, 0), out_dtype)
