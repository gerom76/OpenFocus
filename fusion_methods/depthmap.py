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
# Both rules read the same focus measure, and that measure is pooled at two
# scales rather than one so it cannot leak across an occlusion boundary; see
# _focus_energy and NEAR_WINDOW_DIVISOR for what that fixes.

MODE_MAX = "max"
MODE_AVERAGE = "average"

# Side of the window the per-pixel Laplacian energy is pooled over before the
# decision. Pooling turns a bare |Laplacian| - noisy, and zero on flat-but-in-
# focus areas - into a region measure, so selection follows genuinely sharp
# regions instead of speckling. Exposed to the caller as the kernel-size dial;
# this is only the fallback when none is supplied.
DEFAULT_KERNEL_SIZE = 9

# Side of the second, narrow pooling window, as a divisor of the first. Pooling
# alone is what produced this method's worst artefact: a box window is edge-
# blind, so the enormous energy of a sharply focused high-contrast contour is
# smeared `window // 2` pixels in *every* direction, including out across the
# occlusion boundary onto background that belongs to a different slice. Inside
# that band the foreground's frame wins the argmax and the result takes the
# background from a frame where it is defocused - a flat, washed-out ring
# hugging every contour, its width set by the kernel dial (25 px at k=51).
#
# The narrow window barely leaks, so the ratio between the two poolings says
# how much of a frame's regional score really belongs to this pixel's own
# neighbourhood. Combining them as a geometric mean (below) makes a frame carry
# both to win, which is what collapses the ring. A divisor rather than a fixed
# size so the correction is self-similar: it stays the same fraction of the
# window whatever the caller dials in.
NEAR_WINDOW_DIVISOR = 3

# Below this the narrow term is dropped and the measure is the plain pooled
# energy it always was. A window this small hardly leaks - there is no ring to
# undo - while a 3 px pooling of a squared Laplacian is mostly sensor noise,
# and letting noise into the decision costs far more than the leak it would
# save.
MIN_NEAR_WINDOW = 5

# Gaussian the frame is smoothed by before its Laplacian is taken, in pixels of
# sigma; 0 disables it. This is a *measurement* prefilter only - the output
# pixels are still gathered from the untouched frames, so it costs nothing in
# output resolution. A bare ksize=3 Laplacian is the most noise-sensitive
# high-pass there is, and on flat, low-signal regions the argmax was ranking
# frames by their noise floor rather than their focus, which reads as mottling
# and picks whichever frame is most veiled. Smoothing first removes most of
# that without touching real detail at the scales the pooling works over.
#
# 0.6 is where the ground-truth scenarios in tests/fusion_scenarios.py settle:
# it carries almost all of the gain on a deep stack whose background is never
# sharp - the regime where noise-ranking hurt worst - while a wider one starts
# costing measurably on the noisy and fine-texture scenes.
MEASURE_SIGMA = 0.6

# Taps for that Gaussian, fixed rather than left to OpenCV's sigma-to-ksize
# rule, so the CPU and GPU paths convolve with exactly the same kernel. This is
# the value that rule picks for a float32 image at MEASURE_SIGMA, so the two
# agree bit for bit.
MEASURE_BLUR_KSIZE = 7

# Halo suppression radius, in pixels; 0 disables it. A defocused foreground
# edge spills a glow of roughly its blur radius over the background in the
# frames where the background is sharp, and the argmax happily takes those
# contaminated pixels because the veiled background still out-measures the
# defocused background of the foreground frame. Grey-dilating every frame's
# focus measure by this radius lets a strongly focused region claim that band
# outright: within `halo_radius` of a sharp edge the frame holding the edge
# wins, so the ring comes out as that frame's (glow-free) defocused background
# instead of the glow. The price is the usual one the competitors document for
# their Radius dials - genuinely sharp detail of another frame within the band
# is rounded off - which is why it defaults to off.
#
# It is now needed far less often than it was. The dilation is a blunt trade -
# it does not remove a ring, it fills one with defocused pixels, and it does so
# whether or not there was a glow to fight - and the two-scale measure below
# already discounts glow on its own merits: a veil is low-frequency, so it
# scores badly on the narrow window even where it scores well on the wide one.
# Reach for this only if a glow survives that, and keep the radius well under
# the kernel size.
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

    A task normalises its frame (C channels), smooths it for measurement (C
    again) and takes the Laplacian of that (C again), then pools the result
    into single-channel energy maps which outlive all three; cv2.boxFilter
    allocates its output rather than working in place, and the measure pools
    twice, so the energy map is counted three times over.
    """
    return 4 * rows * cols * (3 * channels + 3)


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


def _near_window(window):
    """Side of the narrow pooling window, or 0 when it is not worth taking.

    See NEAR_WINDOW_DIVISOR. Kept odd so it stays centred, and dropped entirely
    once it would be smaller than MIN_NEAR_WINDOW or would not actually be
    narrower than the window it is meant to check.
    """
    near = max(3, window // NEAR_WINDOW_DIVISOR)
    if near % 2 == 0:
        near -= 1
    return 0 if near < MIN_NEAR_WINDOW or near >= window else near


def _pool(energy, window):
    return cv2.boxFilter(energy, cv2.CV_32F, (window, window),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def _focus_energy(img, window):
    """Local focus energy of one frame: pooled squared Laplacian response.

    The Laplacian is taken per colour channel and the squared responses are
    summed across channels first, so the decision rides on a single grey
    activity map. That keeps all three channels of a pixel tied to the same
    frame (MODE_MAX) or the same blend weight (MODE_AVERAGE); colour can never
    split across sources.

    That activity map is then pooled twice - once over the window the caller
    dialled in, once over a narrow one - and the two are combined as their
    geometric mean. The wide pooling is the region measure the dial asks for;
    the narrow one is local evidence that barely leaks past an occlusion
    boundary. Their product is what a frame has to win on, so a sharply focused
    contour can no longer claim the flat background beside it on the strength
    of energy that lives 20 pixels away - which is the contour ring this method
    used to draw around every subject. Where the energy really is uniform over
    the window the two poolings agree and the mean is the plain pooled energy,
    so flat regions decide exactly as before.
    """
    # Measurement-only prefilter; see MEASURE_SIGMA. The blurred copy is scratch
    # for the Laplacian below and never reaches the output.
    if MEASURE_SIGMA > 0:
        img = cv2.GaussianBlur(img, (MEASURE_BLUR_KSIZE, MEASURE_BLUR_KSIZE),
                               MEASURE_SIGMA)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    # Squared in place: the Laplacian response is this function's own buffer and
    # is dead after the sum, so the temporary a `lap * lap` would allocate is a
    # full frame of float32 nobody needs.
    np.square(lap, out=lap)
    energy = lap.sum(axis=2) if lap.ndim == 3 else lap

    wide = _pool(energy, window)
    near = _near_window(window)
    if not near:
        return wide
    # Folded in place: `wide` is this function's own buffer, so the geometric
    # mean costs no allocation beyond the narrow pooling itself.
    np.multiply(wide, _pool(energy, near), out=wide)
    return np.sqrt(wide, out=wide)


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
