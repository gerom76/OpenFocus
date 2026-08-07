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
#   MODE_AVERAGE  contrast-weighted average: frames are blended by a power of
#                 their focus measure, so a flat region - where every frame is
#                 equally (de)focused - collapses to the plain mean and recovers
#                 the stack's multi-frame SNR (a free sqrt(N) noise reduction),
#                 while sharp detail is still dominated by the frame that holds
#                 it. The power is what makes that second half true on a stack
#                 of any depth; see DEFAULT_SELECTIVITY.
# Both rules read the same focus measure, and that measure is pooled at two
# scales rather than one so it cannot leak across an occlusion boundary; see
# _focus_energy and NEAR_WINDOW_DIVISOR for what that fixes. MODE_MAX then
# despeckles the depth map it arrived at before gathering any pixels, so a
# region no frame ever resolves comes out coherent instead of torn; see
# INDEX_MEDIAN_PASSES.

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

# Passes of a 5x5 median over the finished index map, before the pixels are
# gathered; 0 disables it. MODE_MAX only - MODE_AVERAGE has no index map, and
# its blend is coherent by construction.
#
# The argmax has no spatial prior whatsoever, and over a region no frame ever
# resolves - a background well behind the stack's focus sweep, or any dark,
# flat patch - every frame's energy sits at the same defocused level and the
# winner is decided by noise. Neighbouring pixels then take frames from
# opposite ends of the stack, which render that background at visibly different
# blur and brightness, and the result is a torn mosaic of hard-edged patches.
# On the reference ant stack 22% of pixels in such a region chose a frame more
# than 3 slices from their neighbourhood's choice, against 2% over the subject.
#
# A depth map is piecewise-smooth - it varies continuously across a surface and
# steps only at an occlusion - so the incoherent pixels are outliers against
# their own neighbourhood, which is exactly what a median removes while leaving
# genuine steps standing. Where the measure has a real peak the argmax already
# agrees with its neighbours and the median is a no-op, so sharp detail is not
# what pays for this.
INDEX_MEDIAN_PASSES = 3

# Side of that median. Deliberately the smallest window that can outvote a
# speckle rather than one wide enough to despeckle in a single pass: iterating a
# narrow median converges towards its root signal, removing what is smaller than
# the window without eroding what is larger, whereas one wide median rounds off
# real depth features of its own size too. Three 5x5 passes match a single 9x9
# on incoherence while keeping measurably more detail, and 5 is also the widest
# window cv2.medianBlur takes for anything larger than uint8 - the exact wide
# median is a scipy call costing of order a second per megapixel and growing
# with the window, which a 60 MP stack cannot wear.
INDEX_MEDIAN_KSIZE = 5

# ---------------------------------------------------------------------------
# Coherent depth rendering
# ---------------------------------------------------------------------------
# What the median above cannot fix, and what Helicon's Method B demonstrably
# does fix. Over a region no frame ever resolves - the background behind a deep
# stack, a dark flat patch - every frame's energy sits at the same defocused
# level, so the argmax is decided by noise and neighbouring pixels take frames
# from opposite ends of the stack. On the reference 333-frame ant stack the
# chosen index varies by 12.9 slices inside a 9x9 window there, against 3.8 over
# the subject, and because those frames render that background at visibly
# different blur and brightness the result is a torn mosaic of hard-edged
# patches. Three passes of a 5x5 median take the speckle off that mosaic but
# leave the patches: the incoherence is far wider than the window.
#
# Helicon renders the same region smoothly even at its lowest smoothing
# setting, which is the tell that it is not despeckling a hard selection at all.
# It is declining to make a hard selection where the measurement does not
# support one - and, crucially, it is not *averaging* there either, because the
# background it produces keeps the contrast and brightness of its surroundings.
#
# So: work out which pixels the focus measure can actually speak for, throw away
# the depth of the ones it cannot, and fill the holes by interpolating the
# depths of the ones it can. One dial, `depth_smoothing`, sets how readily a
# pixel is declared unresolvable; 0 is the plain hard select, byte for byte.
#
# What this deliberately does *not* do is vary the rendering rule from pixel to
# pixel. An earlier version widened the slice blend where trust was low, and it
# was wrong twice over. The trust map is bimodal on real stacks - on the
# reference ant stack only 9-14% of pixels land between 0.05 and 0.95, the rest
# saturate - so "blend width proportional to (1 - trust)" is a step, not a ramp,
# and it drew its own hard-edged boundary. And averaging tens of defocused,
# drifting frames converges on something flatter and paler than any one of them,
# so the regions it covered came out as washed-out patches that had lost the
# texture the hard select still had. The depth field is the thing that must be
# continuous; the rule that renders it stays the same everywhere.

DEFAULT_DEPTH_SMOOTHING = 50

# How trustworthy a pixel's focus decision is, from the shape of its focus
# curve across the stack. The measure is the participation ratio of the energy,
# eff = (sum E)^2 / sum(E^2) - the effective number of frames sharing the
# energy, 1 for a single spike and N for a perfectly flat curve - mapped onto
# sharpness = 1 - (eff - 1) / (N - 1), which is 1 for a spike and 0 for a flat
# curve whatever N is.
#
# That N-invariance is the whole reason for this form rather than the obvious
# (peak - mean) / peak: the latter drifts with stack depth (0.69 -> 0.60 on the
# same background as N falls from 333 to 15), so any fixed threshold on it means
# something different on a short stack than on a long one. The participation
# ratio holds the same background at 0.38-0.40 and the same subject at
# 0.58-0.63 across that whole range, so the anchors below can be constants.
#
# It also has to be an absolute measure rather than a percentile of this
# image's own distribution, which would separate the two regimes just as well:
# tiled fusion runs each tile through here independently, and a threshold taken
# from the tile's own histogram would mean something different in every tile
# and lay a visible seam along the tile lattice.
#
# The dial slides the pair: at 1 only the most hopeless pixels are given up on,
# at 100 anything short of a clean spike is. Measured on the ant stack, the
# never-resolved background scores 0.05-0.20 and the subject 0.85-0.95, so the
# midpoint of this range separates them with room to spare either side.
TRUST_LO_MIN, TRUST_LO_MAX = 0.02, 0.50
TRUST_SPAN_MIN, TRUST_SPAN_MAX = 0.10, 0.30

# Levels the depth fill runs to at full dial. Each level roughly doubles the
# reach, so 5 covers a couple of hundred pixels; the default lands on 3, which
# is where both the scenarios and the reference ant stack put the best result.
# Past 5 the low-pass starts flattening depth structure that is real - the
# background stops being a surface and becomes one plane.
DEPTH_FILL_LEVELS_MAX = 5

# Weight every pixel carries into that low-pass on top of its trust. See
# _fill_depth: this is what makes the fill an average of what an unresolvable
# region measured rather than an extrapolation of what its neighbours did, and
# it is worth 7 dB on deep_stack.
DEPTH_FILL_WEIGHT_FLOOR = 0.1

# Stop taking the pyramid down once a side reaches this, so a small tile cannot
# reduce to a single pixel and average its whole depth map to one value.
PYRAMID_MIN_SIDE = 4

# Narrowest blend the renderer will use, in slices. Under 1 so that a depth
# sitting on a whole slice - which is what every trusted pixel's depth still is
# - puts the entire tent on that one frame and none on its neighbours: a
# confident pixel is selected, not blended, exactly as it was before any of
# this. Above 0.5 so a depth that lands midway between two slices still has
# both of them inside the tent rather than neither, which would leave the pixel
# with no frame at all.
#
# Interpolating between slices is deliberately *not* done here, and this floor
# is what prevents it. It reads like an improvement - a focus peak that falls
# between two frames could be rendered from both, in proportion - and it
# measurably is not: the two frames either side of a peak are the two that
# resolve the pixel *worst* among those that resolve it at all, so mixing them
# veils detail that taking the winner outright keeps. Fitting a sub-slice peak
# and rendering through it cost 8 dB of PSNR on the long_stack scenario, where
# every pixel is genuinely resolved by exactly one frame. Sub-slice accuracy
# belongs to a depth map that is being exported as depth; it does not belong to
# one that is being used to gather pixels.
BLEND_WIDTH_FLOOR = 0.75

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

# ---------------------------------------------------------------------------
# Average selectivity
# ---------------------------------------------------------------------------
# What stops MODE_AVERAGE from collapsing into a plain mean of the whole stack,
# which is what it did on any stack deeper than a handful of frames.
#
# Weighting the frames by their focus energy *linearly* is only selective while
# the stack is short. A defocused frame does not measure zero at a detailed
# pixel - it measures a small fraction of the peak - and the blend adds that
# fraction up N times. Measured on the reference 333-frame ant stack at k=21:
# the winning frame's energy is a fortieth of what the other 332 sum to, so
# after the baseline the sharpest frame contributed 1.8% of the output at the
# median pixel and the blend mixed an effective 233 frames. That is not a
# contrast-weighted average, it is the arithmetic mean of 333 differently
# defocused frames - and the mean of many defocus kernels is one enormous
# defocus kernel, which is exactly the veiled, low-contrast, black-lifting haze
# the mode was producing. It scaled with stack depth, so the short scenarios in
# tests/fusion_scenarios.py never showed it.
#
# The fix is to make the weight super-linear in the energy, w = (E + b)^p, so
# the defocused tail cannot outvote the peak by sheer count. Raising a *ratio*
# of energies to a power is what makes this work at any stack depth: where one
# frame genuinely stands out its weight pulls away from the rest as the p-th
# power of how far it stands out, while in a region every frame measures the
# same the ratios are all 1, every weight is equal whatever p is, and the blend
# is still the plain mean that buys the mode its multi-frame SNR. The dial
# therefore costs nothing where averaging is the right answer and everything
# where it was not.
#
# Measured on a controlled 120-frame stack (flat noisy half, single-frame-sharp
# half), against the plain mean's 1x and a single frame's noise:
#     p=1   35% of the sharp half's contrast kept, 118 frames averaged flat
#     p=3   94%                                    103
#     p=4.5 99%                                     85
#     p=8  101%                                     50
# So detail saturates around p=4-6 while the flat-region averaging keeps
# eroding, which is where the default below sits.
DEFAULT_SELECTIVITY = 50

# Exponent the dial reaches at 100. Past this the sharp half has nothing left to
# recover - it is already at 101% of the reference's own contrast - and the only
# thing still moving is the noise in the regions that should be averaging.
SELECTIVITY_EXPONENT_MAX = 8.0

# Frames the energy scale is estimated from, evenly spaced through the stack.
#
# The power above has to be taken on a dimensionless ratio, and the baseline has
# to be a fraction of the stack's own energy, so both need the scale of that
# energy *before* the blend starts rather than after it - which is when the old
# linear fold could recover it for free. Only the baseline survives the
# normalisation, the scale itself cancels, and the baseline is a soft floor that
# tolerates being wrong by a factor of two (dropping it entirely costs 7% of the
# flat-region averaging), so an estimate is ample and it does not need to be a
# whole extra pass. Eight frames land within 3% of the true mean on the ant
# stack and, being a fixed count rather than a fraction, cost under 2% of a deep
# run - which is the regime this exists for.
ENERGY_PROBE_FRAMES = 8

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
    twice, so the energy map is counted three times over. Two more planes for
    the MODE_AVERAGE weight, which takes its exponent by squaring on the pool:
    one running product and one square root. MODE_MAX does not allocate those,
    and is charged for them anyway rather than splitting the estimate in two
    over a tenth of a task.
    """
    return 4 * rows * cols * (3 * channels + 5)


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


def _regularise_index(index):
    """Median-despeckle the index map so the depth it encodes is coherent.

    See INDEX_MEDIAN_PASSES for what this is undoing. Returns the map unchanged
    when the filtering is switched off or the window cannot fit.
    """
    if INDEX_MEDIAN_PASSES <= 0 or min(index.shape) < INDEX_MEDIAN_KSIZE:
        return index
    # cv2.medianBlur takes uint8 and uint16 directly; the uint32 map a stack of
    # more than 65536 frames needs goes through float32, which is exact for
    # every index that can address such a stack.
    native = index.dtype in (np.uint8, np.uint16)
    work = index if native else index.astype(np.float32)
    for _ in range(INDEX_MEDIAN_PASSES):
        work = cv2.medianBlur(work, INDEX_MEDIAN_KSIZE)
    return work if native else work.astype(index.dtype)


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
    # Both poolings average a non-negative quantity, so the product is
    # non-negative in exact arithmetic - but boxFilter accumulates a running sum,
    # and over a window that spans a sharp edge the cancellation can leave a tiny
    # negative behind. Unclamped that reaches sqrt as a NaN, and a NaN loses
    # every '>' it is compared with, so the pixel would silently keep whichever
    # frame happened to be there rather than the one that won.
    np.maximum(wide, 0.0, out=wide)
    return np.sqrt(wide, out=wide)


def _selectivity_exponent(strength):
    """Power the focus energy is raised to in MODE_AVERAGE, from the 0-100 dial.

    Linear in the exponent rather than in anything perceptual: the dial's whole
    range is 1 to SELECTIVITY_EXPONENT_MAX, and 0 is exactly the linear weighting
    the mode always used, so the old behaviour stays reachable.

    Quantised to half steps, which is what lets _powered take the exponent by
    squaring instead of by pow(). Fifteen positions over a curve that saturates
    by two thirds of the way along is finer than the result can distinguish, and
    the default lands on a half step exactly.
    """
    t = max(0.0, min(100.0, float(strength))) / 100.0
    return round((1.0 + (SELECTIVITY_EXPONENT_MAX - 1.0) * t) * 2.0) / 2.0


def _powered(energy, exponent):
    """`energy ** exponent`, for exponents that are multiples of 0.5.

    Square-and-multiply with at most one square root, rather than np.power.
    This runs once per frame over a full-resolution plane and the dial only ever
    asks for a half-integer, where pow() would still take the general
    transcendental route: measured 2-4x faster on a 1.7 MP plane, which is a
    fifth of everything MODE_AVERAGE spends on a deep stack.

    `energy` is consumed - it is squared in place along the way - so the caller
    must use the value returned rather than the array it passed in.
    """
    whole = int(exponent)
    # The half step, taken before `energy` is squared out from under it.
    tail = np.sqrt(energy) if exponent != whole else None

    result = None
    while whole:
        if whole & 1:
            # A copy the first time round rather than an alias: `energy` is
            # still needed as the running square below, so the accumulating
            # product cannot be the same buffer.
            result = (energy.copy() if result is None
                      else np.multiply(result, energy, out=result))
        whole >>= 1
        if whole:
            np.multiply(energy, energy, out=energy)

    if tail is not None:
        np.multiply(result, tail, out=result)
    return result


def _probe_indices(count, probes):
    """Evenly spaced frame indices the energy scale is estimated from."""
    if count <= probes:
        return list(range(count))
    return sorted({int(round(i)) for i in np.linspace(0, count - 1, probes)})


def _resolve_percent(value, default):
    """Coerce a 0-100 slider reading to an int, falling back to `default`."""
    if value is None:
        return default
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def _smoothstep(edge0, edge1, values):
    """Hermite ramp from 0 below `edge0` to 1 above `edge1`, in place.

    A ramp rather than a threshold because this decides how much of two
    renderings a pixel gets, and a hard cut would draw exactly the kind of
    contour into the result that the whole exercise is removing.
    """
    span = max(float(edge1) - float(edge0), 1e-6)
    t = np.subtract(values, edge0, out=values)
    t /= span
    np.clip(t, 0.0, 1.0, out=t)
    # t*t*(3 - 2t), folded so the ramp costs one temporary rather than three.
    scratch = np.multiply(t, -2.0)
    scratch += 3.0
    t *= t
    t *= scratch
    return t


def _trust_anchors(strength):
    """The (low, high) sharpness the trust ramp spans at this dial setting."""
    t = max(0.0, min(1.0, (strength - 1) / 99.0))
    low = TRUST_LO_MIN + (TRUST_LO_MAX - TRUST_LO_MIN) * t
    span = TRUST_SPAN_MIN + (TRUST_SPAN_MAX - TRUST_SPAN_MIN) * t
    return low, low + span


def _trust_map(energy_sum, energy_sq_sum, count, strength):
    """How far each pixel's focus curve is from flat, on a 0-1 scale.

    See TRUST_LO_MIN for what the statistic is and why it is this one. The
    result says which pixels the hard selection can be believed for: 1 where a
    frame genuinely stood out, 0 where every frame measured the same and the
    argmax was reading its own noise.
    """
    if count < 2:
        return np.ones_like(energy_sum)
    trust_lo, trust_hi = _trust_anchors(strength)
    # eff = (sum E)^2 / sum(E^2), the effective number of frames sharing the
    # energy. A pixel with no energy at all in any frame divides 0 by 0; it has
    # no focus information by definition, so it is handed the flat-curve answer.
    eff = np.square(energy_sum)
    np.divide(eff, np.maximum(energy_sq_sum, 1e-20), out=eff)
    np.clip(eff, 1.0, float(count), out=eff)
    # -> sharpness = 1 - (eff - 1)/(count - 1), folded in place.
    eff -= 1.0
    eff /= float(count - 1)
    np.subtract(1.0, eff, out=eff)
    return _smoothstep(trust_lo, trust_hi, eff)


def _fill_levels(strength):
    """Pyramid depth the fill runs to, from the dial. 0 is handled by caller."""
    levels = 1 + int(round(DEPTH_FILL_LEVELS_MAX * (strength / 100.0)))
    return max(1, min(DEPTH_FILL_LEVELS_MAX, levels))


def _fill_depth(depth, weight, levels):
    """Trust-weighted low-pass of the depth map, at a scale set by `levels`.

    A normalised convolution carried on a Gaussian pyramid: the weighted depth
    and its weight are taken down `levels` times and brought straight back up,
    so the cost is the same whatever the scale and the support grows by roughly
    an order of magnitude for three levels rather than by the width of a window.
    That reach is the point - the incoherence being removed is tens of pixels
    across, which no affordable box filter covers.

    `weight` carries a floor as well as the trust map, and the floor is what
    stops this from being extrapolation. With trust alone, a region nothing
    resolves takes its depth entirely from the nearest pixels that *were*
    resolved - which on deep_stack means the background is rendered at the
    subject's depth and loses 7 dB, because a never-sharp background still has
    a real, weak preference of its own. The floor lets such a region average its
    own measurement instead: the scatter goes, the regional level stays.
    """
    num = (depth * weight).astype(np.float32)
    den = np.asarray(weight, dtype=np.float32)
    shapes = []
    for _ in range(levels):
        if min(num.shape[:2]) <= PYRAMID_MIN_SIDE:
            break
        shapes.append((num.shape[1], num.shape[0]))
        num = cv2.pyrDown(num)
        den = cv2.pyrDown(den)
    for size in reversed(shapes):
        num = cv2.pyrUp(num, dstsize=size)
        den = cv2.pyrUp(den, dstsize=size)
    return num / np.maximum(den, 1e-8)


def _coherent_depth(despeckled, trust, strength):
    """The depth the renderer follows: measured where trusted, smoothed where not.

    `trust` gates the crossover rather than thresholding it, so the pixels that
    genuinely sit between the two regimes move over gradually. Everywhere else
    trust has saturated - it is close to bimodal on real stacks - and this is
    simply a choice between the frame the argmax named and what the low-pass
    says its neighbourhood meant.

    The two agree wherever the argmax was already coherent, which is why this
    leaves resolved detail alone: the low-pass of a locally constant depth is
    that same constant, so the crossover has nothing to change there.
    """
    if strength <= 0:
        return despeckled.astype(np.float32)
    measured = despeckled.astype(np.float32)
    smoothed = _fill_depth(measured, trust + DEPTH_FILL_WEIGHT_FLOOR,
                           _fill_levels(strength))
    return (trust * measured + (1.0 - trust) * smoothed).astype(np.float32)


def _gather_blended(load_float, count, depth, out_dtype, rows, cols):
    """Render the stack at a continuous depth, one narrow tent per pixel.

    The tent is BLEND_WIDTH_FLOOR wide for every pixel in the frame - the rule
    does not vary, only the depth it is applied at. A depth that landed on a
    whole slice takes that slice outright; one that landed between two takes
    both, weighted towards the nearer. Since the depth is only ever fractional
    where the measurement had nothing to say, the frames that carry real detail
    are still selected rather than mixed.

    Frames are visited in index order and only loaded when some pixel is
    actually asking for them, so a stack whose depths cluster does not pay to
    read the slices nobody wants.
    """
    accum = None
    weight_sum = np.zeros((rows, cols), dtype=np.float32)
    for k in range(count):
        # max(0, width - |depth - k|). The textbook tent is 1 - |depth-k|/width,
        # which is this divided by a constant; the normalisation at the end
        # cancels it, so the division is never taken.
        tent = np.abs(depth - k)
        np.subtract(BLEND_WIDTH_FLOOR, tent, out=tent)
        np.maximum(tent, 0.0, out=tent)
        if not tent.any():
            continue
        img = load_float(k)
        if accum is None:
            accum = np.zeros_like(img)
        # load_float hands back a buffer of its own, so it doubles as the
        # scratch the weighting needs rather than allocating a second frame.
        img *= tent[:, :, np.newaxis] if img.ndim == 3 else tent
        accum += img
        weight_sum += tent
        del img, tent

    # Every pixel's band holds at least the slice nearest its depth - the tent
    # is wider than half a slice and the depth is inside the stack - so the
    # accumulator is written and the weights are positive everywhere.
    np.maximum(weight_sum, 1e-8, out=weight_sum)
    accum /= weight_sum[:, :, np.newaxis] if accum.ndim == 3 else weight_sum
    return bitdepth.from_float01(accum, out_dtype)


def depthmap_impl(input_source, img_resize=None, mode=MODE_MAX,
                  kernel_size=None, thread_count=None, halo_radius=None,
                  depth_smoothing=None, selectivity=None):
    """Depth-map multi-focus fusion (per-pixel select or contrast-weighted avg).

    A local Laplacian-energy focus measure is computed for every frame. In
    MODE_MAX the pixel is taken whole from the frame whose measure is highest -
    an order-independent argmax that is the classic hard depth map, despeckled
    by INDEX_MEDIAN_PASSES before the pixels are gathered. In
    MODE_AVERAGE the frames are blended in proportion to that measure, so flat
    regions average (recovering multi-frame SNR) and sharp regions still follow
    the frame that holds the detail.

    `halo_radius` > 0 turns on halo suppression: each frame's energy map is
    grey-dilated by that many pixels before the decision, so a sharply focused
    region also claims the surrounding band its defocused image contaminates
    in the other frames. See DEFAULT_HALO_RADIUS for the mechanism.

    `depth_smoothing` (0-100, MODE_MAX only) is what stops a region no frame
    resolves from tearing into a mosaic of hard-edged patches: it sets how
    readily a pixel's focus decision is given up as unfounded, and the depth of
    every pixel given up on is interpolated from the ones that were not. 0
    reproduces the plain hard select exactly. See the block above
    DEFAULT_DEPTH_SMOOTHING for what it is undoing.

    `selectivity` (0-100, MODE_AVERAGE only) is what stops a deep stack's blend
    from collapsing into the plain mean of every frame - the veiled, washed-out
    haze the mode used to produce - by raising the focus weight to a power. 0 is
    the linear weighting it always used. See the block above
    DEFAULT_SELECTIVITY.

    The stack's own depth is preserved end to end: an 8-bit stack returns
    uint8, a 16-bit stack returns uint16.
    """
    if mode not in (MODE_MAX, MODE_AVERAGE):
        raise ValueError(f"Unknown depth-map mode {mode!r}; "
                         f"expected {MODE_MAX!r} or {MODE_AVERAGE!r}")

    window = _resolve_kernel(kernel_size)
    halo_element = _halo_element(_resolve_halo_radius(halo_radius))
    smoothing = _resolve_percent(depth_smoothing, DEFAULT_DEPTH_SMOOTHING)
    selectivity = _resolve_percent(selectivity, DEFAULT_SELECTIVITY)
    # MODE_AVERAGE is already a blend of every frame by construction; there is
    # no index map to fill, so the dial has nothing to act on.
    coherent = mode == MODE_MAX and smoothing > 0

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

    def measure_weight(k, baseline, scale, exponent):
        """Frame k and the MODE_AVERAGE weight it carries, w = ((E+b)/s)**p.

        The weight is built here rather than in the reduction below so that the
        exponent is taken on the pool alongside the measurement it belongs to;
        the reduction is then a multiply-accumulate that cannot be parallelised
        anyway, because its order is what makes the run reproducible.
        """
        img = load_float(k)
        energy = _focus_energy(img, window)
        if halo_element is not None:
            energy = cv2.dilate(energy, halo_element)
        # Folded in place: the energy map is this task's own buffer, so the
        # whole weight is built where the measurement already sits.
        energy += baseline
        energy /= scale
        if exponent != 1.0:
            energy = _powered(energy, exponent)
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
        # The coherent path needs the shape of each pixel's focus curve, not
        # just where its peak is. Two running sums are the whole of it - see
        # _trust_map - and they are allocated only when it is switched on, so
        # the plain hard select still costs one float32 plane and one index
        # plane exactly as it always did.
        energy_sum = np.zeros((rows, cols), dtype=np.float32) if coherent else None
        energy_sq_sum = np.zeros((rows, cols), dtype=np.float32) if coherent else None

        for k, energy in _map_in_order(measure_energy, num_images, max_workers):
            if coherent:
                energy_sum += energy
                energy_sq_sum += np.square(energy)
            better = energy > best_energy
            np.copyto(best_energy, energy, where=better)
            np.copyto(best_index, k, where=better)
            del energy, better

        # The depth map is despeckled before anything is gathered, so a pixel
        # whose winner was decided by noise takes its neighbourhood's frame
        # instead of tearing away from it.
        despeckled = _regularise_index(best_index)

        if coherent:
            del best_energy, best_index
            trust = _trust_map(energy_sum, energy_sq_sum, num_images, smoothing)
            del energy_sum, energy_sq_sum
            depth = _coherent_depth(despeckled, trust, smoothing)
            del trust, despeckled
            return _gather_blended(load_float, num_images, depth,
                                   out_dtype, rows, cols)

        del best_energy
        best_index = despeckled

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

    # MODE_AVERAGE: blend the frames by w = ((E + baseline) / scale) ** exponent.
    #
    # The scale cancels in the normalisation and is carried only to keep the
    # power on numbers of order one: an energy ratio spans some three decades on
    # a real stack, and at the top of the dial its eighth power would otherwise
    # be asked of float32 at both ends at once. The baseline is the one part
    # that does not cancel, and it is what a flat region rides on - there every
    # energy is far below it, every weight comes out equal, and the blend is the
    # plain mean whatever the exponent is.
    probes = _probe_indices(num_images, ENERGY_PROBE_FRAMES)
    # Reduced to a scalar as each probe arrives rather than collected: the
    # energy planes are frame-sized, and holding all eight would reserve more at
    # this one moment than the whole measurement pool is allowed.
    probe_mean = 0.0
    for _, mean_e in _map_in_order(
            lambda i: float(measure_energy(probes[i]).mean()),
            len(probes), max_workers):
        probe_mean += mean_e / len(probes)
    scale = max(probe_mean, _WEIGHT_FLOOR)
    baseline = scale * _BASELINE_FRACTION
    exponent = _selectivity_exponent(selectivity)

    weighted = None
    weight_sum = np.zeros((rows, cols), dtype=np.float32)
    weigh = functools.partial(measure_weight, baseline=baseline, scale=scale,
                              exponent=exponent)
    for _, (img, weight) in _map_in_order(weigh, num_images, max_workers):
        if weighted is None:
            weighted = np.zeros_like(img)
        # The frame is this iteration's own buffer and is dead after the
        # accumulation, so it doubles as the scratch the weighting needs -
        # `weighted += img * weight` would allocate a whole extra frame.
        img *= weight[:, :, np.newaxis] if img.ndim == 3 else weight
        weighted += img
        weight_sum += weight
        del img, weight

    # Every weight is at least (baseline / scale) ** exponent, so the sum is
    # positive everywhere and needs no floor of its own.
    weighted /= weight_sum[:, :, np.newaxis] if weighted.ndim == 3 else weight_sum
    return bitdepth.from_float01(weighted, out_dtype)
