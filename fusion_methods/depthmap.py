import functools
import os
import glob
import re
import cv2
import numpy as np

from utils import bitdepth
from utils.image_utils import read_image_any_depth
from fusion_methods.gff import _map_in_order

try:
    import psutil
except ImportError:  # psutil is a hard requirement, but never fail a load over it
    psutil = None

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

# Default cap on the frame-measurement pool when the caller names no thread
# count. The OpenCV filters underneath are already internally parallel, so past
# a handful of frames the outer threads buy contention rather than throughput;
# the cap is also what bounds how many frames are in flight as float32 at once.
# Same reasoning, and the same number, as gff.py.
DEFAULT_MAX_WORKERS = 8

# Share of the free physical memory those in-flight buffers may occupy. The
# pool size alone does not bound memory: one 60 MP 16-bit frame is 720 MB once
# it is float32, so eight of them in flight would reserve more than a 16 GB
# machine has free and the run would swap - which costs far more than the
# threads were worth. Conservative because the caller is holding the whole
# source stack at the same time, and, under tiled fusion, several tiles of it.
WORKER_MEMORY_SHARE = 0.25


def _available_bytes():
    """Physical memory that can be handed out without swapping, or None.

    Deliberately a local copy of ``core.memory.available_bytes`` rather than an
    import of it: ``core/__init__`` pulls in the Qt workers and imports this
    module in turn, so reaching into ``core`` from a fusion method would make
    the package circular and drag the UI into a headless run.
    """
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


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


@functools.lru_cache(maxsize=16)
def _halo_element(radius):
    """Elliptical structuring element covering `radius` pixels, or None for 0.

    Cached because tiled fusion calls this once per tile with the same radius,
    and the elements are both tiny and read-only to cv2.dilate. Callers must
    not write to the array they get back.
    """
    if radius <= 0:
        return None
    side = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def _worker_bytes(rows, cols, channels):
    """float32 bytes one in-flight measurement task holds at its peak.

    A task normalises its frame (C channels), takes the Laplacian of it (C
    again) and pools that into a single-channel energy map which outlives both;
    cv2.boxFilter allocates its output rather than working in place, so the
    energy map is counted twice.
    """
    return 4 * rows * cols * (2 * channels + 2)


def _resolve_workers(thread_count, rows, cols, channels, num_images):
    """Pool size for the measurement pass: what was asked for, within memory.

    Both bounds matter. The thread count is what the caller set (or the default
    cap when they left it alone), and the memory bound is what the machine can
    actually hold in flight. A stack with fewer frames than threads never needs
    the extra ones.
    """
    if thread_count is None:
        workers = min(DEFAULT_MAX_WORKERS, os.cpu_count() or 4)
    else:
        try:
            workers = max(1, int(thread_count))
        except (TypeError, ValueError):
            workers = min(DEFAULT_MAX_WORKERS, os.cpu_count() or 4)

    workers = max(1, min(workers, num_images))

    available = _available_bytes()
    if available is None:
        return workers

    affordable = int(available * WORKER_MEMORY_SHARE) // _worker_bytes(rows, cols, channels)
    capped = max(1, min(workers, int(affordable)))
    if capped < workers:
        print(f"Note: depth-map fusion is using {capped} measurement thread(s) "
              f"instead of {workers} to stay inside available memory.")
    return capped


def _index_dtype(num_images):
    """Narrowest unsigned type that can name every frame of the stack."""
    if num_images <= 256:
        return np.uint8
    return np.uint16 if num_images <= 65536 else np.uint32


def _focus_energy(img, window):
    """Local focus energy of one frame: pooled squared Laplacian response.

    The Laplacian is taken per colour channel and the squared responses are
    summed across channels first, so the decision rides on a single grey
    activity map. That keeps all three channels of a pixel tied to the same
    frame (MODE_MAX) or the same blend weight (MODE_AVERAGE); colour can never
    split across sources.
    """
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    # Squared in place: the Laplacian response is this function's own buffer and
    # is dead after the sum, so the temporary a `lap * lap` would allocate is a
    # full frame of float32 nobody needs.
    np.square(lap, out=lap)
    energy = lap.sum(axis=2) if lap.ndim == 3 else lap
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
    channels = stack_ori[0].shape[2] if stack_ori[0].ndim == 3 else 1

    if img_resize:
        cols, rows = int(img_resize[0]), int(img_resize[1])
    else:
        rows, cols = stack_ori[0].shape[:2]

    # Resolved here rather than up front because both bounds need the frame
    # geometry: a pool that fits a 2 MP stack does not fit a 60 MP one.
    max_workers = _resolve_workers(thread_count, rows, cols, channels, num_images)

    def load_native(k):
        """Frame k at the stack's own depth, resized if the caller asked."""
        img = stack_ori[k]
        if img_resize and (img.shape[1], img.shape[0]) != (cols, rows):
            img = cv2.resize(img, (cols, rows))
        # A no-op returning the same array when the frame already carries the
        # stack's depth; a mixed 8/16-bit stack is rescaled to the common one.
        return bitdepth.convert(img, out_dtype)

    def load_float(k):
        # Per-frame resize and float conversion, so only the handful of frames
        # in flight are ever held as float32 rather than the whole stack.
        img = stack_ori[k]
        if img_resize and (img.shape[1], img.shape[0]) != (cols, rows):
            img = cv2.resize(img, (cols, rows))
        norm = bitdepth.to_float01(img)
        # An already-normalised float stack is handed back unchanged by
        # to_float01, and MODE_AVERAGE writes into this buffer; copy so the
        # accumulation can never reach through into the caller's frames.
        return norm if norm is not img else norm.copy()

    def measure_energy(k):
        """Focus energy of frame k. The frame itself is not held on to."""
        energy = _focus_energy(load_float(k), window)
        if halo_element is not None:
            energy = cv2.dilate(energy, halo_element)
        return energy

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
        # The decision pass records *which* frame won each pixel, not the pixel
        # itself: an index map costs one byte per pixel against twelve for a
        # float32 BGR accumulator, the measurement tasks no longer have to keep
        # their normalised frame alive until the reduction consumes it, and the
        # result can then be gathered at the stack's own depth without a float
        # round trip. Identical output - the round trip was already exact.
        best_energy = np.full((rows, cols), -np.inf, dtype=np.float32)
        best_index = np.zeros((rows, cols), dtype=_index_dtype(num_images))
        for k, energy in _map_in_order(measure_energy, num_images, max_workers):
            better = energy > best_energy
            np.copyto(best_energy, energy, where=better)
            np.copyto(best_index, k, where=better)
            del energy, better
        del best_energy

        # Every pixel names exactly one frame - frame 0 already beats the -inf
        # the map starts at - so every pixel is written exactly once below and
        # an uninitialised buffer is safe to gather into.
        fused = None
        for k in range(num_images):
            won = best_index == k
            if not won.any():
                continue  # a frame that won nowhere need not be loaded at all
            frame = load_native(k)
            if fused is None:
                fused = np.empty(frame.shape, dtype=out_dtype)
            np.copyto(fused, frame,
                      where=won[:, :, np.newaxis] if frame.ndim == 3 else won)
        return fused

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
        plain += img
        # The frame is this iteration's own buffer and is dead after the two
        # accumulations, so it doubles as the scratch the weighting needs -
        # `weighted += img * energy` would allocate a whole extra frame.
        img *= energy[:, :, np.newaxis] if img.ndim == 3 else energy
        weighted += img
        weight_sum += energy
        del img, energy

    mean_energy = float(weight_sum.sum()) / max(1, num_images * rows * cols)
    baseline = max(mean_energy * _BASELINE_FRACTION, _WEIGHT_FLOOR)

    # Folded in place for the same reason: the closed form allocates two more
    # full frames at the one moment the run is already at its peak.
    plain *= baseline
    weighted += plain
    del plain
    weight_sum += baseline * num_images
    weighted /= weight_sum[:, :, np.newaxis] if weighted.ndim == 3 else weight_sum
    return bitdepth.from_float01(weighted, out_dtype)
