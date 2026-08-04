"""
GPU implementation of DTCWT multi-focus fusion (Lewis et al., 2007).

Mirrors fusion_methods/dtcwt.py but runs on a torch device (CUDA/MPS) using
pytorch_wavelets, which implements the same Kingsbury dual-tree complex wavelet
transform with the same default filters ('near_sym_a' / 'qshift_a') as the CPU
dtcwt package.

The stack is processed in streaming fashion — one image is transformed and
folded into the running result at a time — which mirrors the CPU version's
selection exactly and bounds GPU memory to ~2 images' worth of coefficients
regardless of stack size. Each coefficient goes to the frame carrying the most
activity around it, so the fold is a running maximum rather than a chain of
pairwise fusions and the answer does not depend on the order of the stack
(item 7 in docs/ALGORITHM_IMPROVEMENTS.md).

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


def _window(x, pool):
    """Slide `pool` over a 3x3 window per direction. x: (1, 6, h, w)."""
    c, d, h, w = x.shape
    padded = F.pad(x.reshape(c * d, 1, h, w), (1, 1, 1, 1), mode='reflect')
    return pool(padded, kernel_size=WINDOW_SIZE, stride=1).reshape(c, d, h, w)


def _coefficient_activity(coeffs):
    """How much detail one frame carries at each coefficient of one level.

    Torch mirror of _coefficient_activity in fusion_methods/dtcwt.py. coeffs:
    (C, 6, h, w, 2) real/imag pairs; returns (score, magnitude), both
    (1, 6, h, w). The magnitude is reduced over the colour channels with a
    maximum, so one decision serves all of them (item 12 in
    docs/ALGORITHM_IMPROVEMENTS.md); the score is that magnitude max-filtered
    over the window - Lewis's activity - and pooled over the same window, so
    frames are compared regionally (item 7).
    """
    magnitude = torch.sqrt(coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2).amax(
        dim=0, keepdim=True)
    score = _window(_window(magnitude, F.max_pool2d), F.avg_pool2d)
    return score, magnitude


def _keep_stronger(fused, best_score, best_magnitude, coeffs, score, magnitude):
    """Fold one frame's coefficients into the running per-coefficient winner.

    Mirrors _keep_stronger in fusion_methods/dtcwt.py, returning the updated
    (fused, best_score, best_magnitude) rather than writing in place. A frame
    wins a coefficient by carrying more regional activity there; an exact tie
    goes to the stronger coefficient rather than to whichever frame arrived
    first, so the result does not depend on the order of the stack.
    """
    takes = (score > best_score) | ((score == best_score) &
                                    (magnitude > best_magnitude))
    return (torch.where(takes.unsqueeze(-1), coeffs, fused),
            torch.where(takes, score, best_score),
            torch.where(takes, magnitude, best_magnitude))


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

    # The lowpass sum is the one running quantity whose value depends on the
    # order it is accumulated in, so it is carried at double precision to keep
    # a reordered stack rendering the identical picture. MPS has no float64.
    accumulate = torch.float32 if dev.type == 'mps' else torch.float64

    with torch.no_grad():
        lowpass_sum = None
        lowpass_weight = None
        fused_highpass = None
        best_score = None
        best_magnitude = None

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
            activity = [_coefficient_activity(level) for level in yh]

            if lowpass_sum is None:
                lowpass_sum = (yl * weight).to(accumulate)
                lowpass_weight = weight.to(accumulate)
                fused_highpass = yh
                best_score = [score for score, _ in activity]
                best_magnitude = [magnitude for _, magnitude in activity]
            else:
                lowpass_sum = lowpass_sum + (yl * weight).to(accumulate)
                lowpass_weight = lowpass_weight + weight.to(accumulate)
                # Every frame is judged against every other on the same
                # measure, in one streaming pass, rather than folded together
                # two at a time (item 7 in docs/ALGORITHM_IMPROVEMENTS.md).
                for level, (score, magnitude) in enumerate(activity):
                    fused_highpass[level], best_score[level], best_magnitude[level] = \
                        _keep_stronger(fused_highpass[level], best_score[level],
                                       best_magnitude[level], yh[level],
                                       score, magnitude)

        fused_lowpass = (lowpass_sum / lowpass_weight).to(yl.dtype).unsqueeze(0)
        fused_highpass = [level.unsqueeze(0) for level in fused_highpass]

        out = ifm((fused_lowpass, fused_highpass))[0]  # (3, H, W)
        # The inverse transform may return a padded size for odd dimensions
        out = out[:, :h_orig, :w_orig]

        # (H, W, 3) BGR, back at the depth the stack arrived in
        return torch_depth.from_float01(out.permute(1, 2, 0), out_dtype)
