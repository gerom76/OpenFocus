import os
import glob
import re
import cv2
import numpy as np

from utils import bitdepth
from utils.image_utils import read_image_any_depth
from fusion_methods.gff import _map_in_order

# Reference:
# The depth-map family of focus stacking - as shipped by Shine Stacker's
# DepthMapStack, Zerene's DMap and Helicon's Methods A/B - decides focus per
# pixel from a local contrast measure rather than in a transform domain. Two
# selection rules are offered:
#   MODE_MAX      hard per-pixel select: the pixel is taken whole from the frame
#                 whose focus measure is highest there (an explicit depth map).
#   MODE_AVERAGE  contrast-weighted average: frames are blended in proportion to
#                 their focus measure, so a flat region - where every frame is
#                 equally (de)focused - collapses to the plain mean and recovers
#                 the stack's multi-frame SNR (a free sqrt(N) noise reduction),
#                 while sharp detail is still dominated by the frame that holds
#                 it.

MODE_MAX = "max"
MODE_AVERAGE = "average"

# Side of the window the per-pixel Laplacian energy is pooled over before the
# decision. Pooling turns a bare |Laplacian| - noisy, and zero on flat-but-in-
# focus areas - into a region measure, so selection follows genuinely sharp
# regions instead of speckling. Exposed to the caller as the kernel-size dial;
# this is only the fallback when none is supplied.
DEFAULT_KERNEL_SIZE = 9

# Halo suppression radius, in pixels; 0 disables it. A defocused foreground
# edge spills a glow of roughly its blur radius over the background in the
# frames where the background is sharp, and the argmax happily takes those
# contaminated pixels because the veiled background still out-measures the
# defocused background of the foreground frame. Grey-dilating every frame's
# pooled energy by this radius lets a strongly focused region claim that band
# outright: within `halo_radius` of a sharp edge the frame holding the edge
# wins, so the ring comes out as that frame's (glow-free) defocused background
# instead of the glow. The price is the usual one the competitors document for
# their Radius dials - genuinely sharp detail of another frame within the band
# is rounded off - which is why it defaults to off.
DEFAULT_HALO_RADIUS = 0

# In MODE_AVERAGE every frame is given a small baseline weight on top of its
# focus measure, set to this fraction of the stack's mean energy. It is what
# makes a flat region average rather than chase the noisiest frame: where the
# real focus energy is far above the baseline (detail) the sharp frame still
# dominates, but where it is at or below the baseline (flat) the weights are
# near-equal and the result is the mean. Scale-free because it tracks the
# image's own energy.
_BASELINE_FRACTION = 0.1

# Absolute floor for that baseline, in normalised [0, 1] energy units, so a
# perfectly flat stack (mean energy 0) still averages instead of dividing by 0.
_WEIGHT_FLOOR = 1e-8


def _resolve_kernel(kernel_size):
    """Coerce the pooling window to a positive odd integer."""
    if kernel_size is None:
        return DEFAULT_KERNEL_SIZE
    try:
        k = max(1, int(kernel_size))
    except (TypeError, ValueError):
        return DEFAULT_KERNEL_SIZE
    if k % 2 == 0:
        k = max(1, k - 1)
    return k


def _resolve_halo_radius(halo_radius):
    """Coerce the halo-suppression radius to a non-negative integer."""
    if halo_radius is None:
        return DEFAULT_HALO_RADIUS
    try:
        return max(0, int(halo_radius))
    except (TypeError, ValueError):
        return DEFAULT_HALO_RADIUS


def _halo_element(radius):
    """Elliptical structuring element covering `radius` pixels, or None for 0."""
    if radius <= 0:
        return None
    side = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def _focus_energy(img, window):
    """Local focus energy of one frame: pooled squared Laplacian response.

    The Laplacian is taken per colour channel and the squared responses are
    summed across channels first, so the decision rides on a single grey
    activity map. That keeps all three channels of a pixel tied to the same
    frame (MODE_MAX) or the same blend weight (MODE_AVERAGE); colour can never
    split across sources.
    """
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    if lap.ndim == 3:
        energy = np.sum(lap * lap, axis=2)
    else:
        energy = lap * lap
    return cv2.boxFilter(energy, cv2.CV_32F, (window, window),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def depthmap_impl(input_source, img_resize=None, mode=MODE_MAX,
                  kernel_size=None, thread_count=None, halo_radius=None):
    """Depth-map multi-focus fusion (per-pixel select or contrast-weighted avg).

    A local Laplacian-energy focus measure is computed for every frame. In
    MODE_MAX the pixel is taken whole from the frame whose measure is highest -
    an order-independent argmax that is the classic hard depth map. In
    MODE_AVERAGE the frames are blended in proportion to that measure, so flat
    regions average (recovering multi-frame SNR) and sharp regions still follow
    the frame that holds the detail.

    `halo_radius` > 0 turns on halo suppression: each frame's energy map is
    grey-dilated by that many pixels before the decision, so a sharply focused
    region also claims the surrounding band its defocused image contaminates
    in the other frames. See DEFAULT_HALO_RADIUS for the mechanism.

    The stack's own depth is preserved end to end: an 8-bit stack returns
    uint8, a 16-bit stack returns uint16.
    """
    if mode not in (MODE_MAX, MODE_AVERAGE):
        raise ValueError(f"Unknown depth-map mode {mode!r}; "
                         f"expected {MODE_MAX!r} or {MODE_AVERAGE!r}")

    if thread_count is None:
        max_workers = min(8, os.cpu_count() or 4)
    else:
        try:
            max_workers = max(1, int(thread_count))
        except (TypeError, ValueError):
            max_workers = min(8, os.cpu_count() or 4)

    window = _resolve_kernel(kernel_size)
    halo_element = _halo_element(_resolve_halo_radius(halo_radius))

    # ---------- Data loading ----------
    if isinstance(input_source, str):
        filenames = os.listdir(input_source)
        if not filenames:
            raise ValueError("Input folder is empty or no images were found")
        suffixes = [os.path.splitext(f)[1] for f in filenames if os.path.splitext(f)[1]]
        img_ext = suffixes[0] if suffixes else ''
        img_paths = glob.glob(os.path.join(input_source, '*' + img_ext))

        def sort_key(path):
            nums = re.findall(r"\d+", os.path.basename(path))
            return int(nums[-1]) if nums else path
        img_paths.sort(key=sort_key)

        stack_ori = [read_image_any_depth(p) for p in img_paths]
        stack_ori = [img for img in stack_ori if img is not None]
    else:
        stack_ori = list(input_source)

    if not stack_ori:
        raise ValueError("No image data was loaded")

    out_dtype = bitdepth.stack_dtype(stack_ori)
    num_images = len(stack_ori)

    if img_resize:
        cols, rows = int(img_resize[0]), int(img_resize[1])
    else:
        rows, cols = stack_ori[0].shape[:2]

    def load_float(k):
        # Per-frame resize and float conversion, so only the handful of frames
        # in flight are ever held as float32 rather than the whole stack.
        img = stack_ori[k]
        if img_resize and (img.shape[1], img.shape[0]) != (cols, rows):
            img = cv2.resize(img, (cols, rows))
        return bitdepth.to_float01(img)

    def measure(k):
        img = load_float(k)
        energy = _focus_energy(img, window)
        if halo_element is not None:
            energy = cv2.dilate(energy, halo_element)
        return img, energy

    # Frames are measured on a thread pool but reduced in index order, so both
    # the strict-'>' tie-break (MODE_MAX) and the float accumulation
    # (MODE_AVERAGE) are bitwise-stable run to run.
    if mode == MODE_MAX:
        fused = None
        best_energy = np.full((rows, cols), -np.inf, dtype=np.float32)
        for _, (img, energy) in _map_in_order(measure, num_images, max_workers):
            if fused is None:
                fused = np.zeros_like(img)
            better = energy > best_energy
            np.copyto(best_energy, energy, where=better)
            if img.ndim == 3:
                np.copyto(fused, img, where=better[:, :, np.newaxis])
            else:
                np.copyto(fused, img, where=better)
            del img, energy
        return bitdepth.from_float01(fused, out_dtype)

    # MODE_AVERAGE: accumulate the contrast-weighted sum, the weight sum, and a
    # plain sum. The plain sum lets flat regions fall back to the mean via a
    # baseline weight added after the total energy is known.
    weighted = None
    plain = None
    weight_sum = np.zeros((rows, cols), dtype=np.float32)
    for _, (img, energy) in _map_in_order(measure, num_images, max_workers):
        if weighted is None:
            weighted = np.zeros_like(img)
            plain = np.zeros_like(img)
        if img.ndim == 3:
            weighted += img * energy[:, :, np.newaxis]
        else:
            weighted += img * energy
        plain += img
        weight_sum += energy
        del img, energy

    mean_energy = float(weight_sum.sum()) / max(1, num_images * rows * cols)
    baseline = max(mean_energy * _BASELINE_FRACTION, _WEIGHT_FLOOR)

    denom = weight_sum + baseline * num_images
    if weighted.ndim == 3:
        fused = (weighted + baseline * plain) / denom[:, :, np.newaxis]
    else:
        fused = (weighted + baseline * plain) / denom
    return bitdepth.from_float01(fused, out_dtype)
