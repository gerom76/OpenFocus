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
# Two things get us there, and both are switchable dials because both trade
# something:
#
#   depth_smoothing  the depth map is regularised by how much each pixel can be
#                    trusted, so unresolvable pixels inherit depth from
#                    confident neighbours instead of voting with their noise.
#   slice_blending   the pixel is no longer taken whole from one frame but
#                    blended across a band of slices whose width is set per
#                    pixel, so the mosaic's hard edges cannot form and a region
#                    nobody resolves converges on the local mean of the stack -
#                    the same multi-frame SNR win MODE_AVERAGE gets for free.
#
# Both off reproduces the plain hard select exactly, byte for byte.

DEFAULT_DEPTH_SMOOTHING = 50
DEFAULT_SLICE_BLENDING = 50

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
TRUST_LO = 0.35
TRUST_HI = 0.60

# Exponent on the trust map when it is used as a vote weight in the depth
# regularisation. Cubed rather than linear because this decides who gets to
# speak for a neighbourhood, not how much of a blend they contribute: a pixel
# half as trustworthy as its neighbour should not carry half its weight, it
# should be most of the way to silent.
TRUST_EXPONENT = 3

# Windows the depth map is propagated over, coarse-to-fine, at full smoothing.
# Three scales rather than one wide pass because the incoherent regions vary in
# size - a thin band beside a contour and a whole quadrant of dead background
# both have to be filled - and a single window wide enough for the largest
# would drag depth across every occlusion in the frame on its way there.
DEPTH_SMOOTH_SCALES = (9, 21, 45)

# Side of the window the depth map's local spread is measured over, which is
# what sets the blend width below.
DISAGREE_WINDOW = 9

# Cap on the blend width at full slider, as a share of the stack depth. The
# width is otherwise self-scaling - it is read straight off the neighbourhood's
# own disagreement in slices, so a neighbourhood that cannot agree within 20
# slices blends 20 - and this only stops a pathological stack from averaging
# itself flat.
BLEND_MAX_SHARE = 0.25

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


def _trust_map(energy_sum, energy_sq_sum, count):
    """How far each pixel's focus curve is from flat, on a 0-1 scale.

    See TRUST_LO for what the statistic is and why it is this one. The result
    is what both the depth regularisation and the blend width read: 1 where a
    frame genuinely stood out and the hard selection can be believed, 0 where
    every frame measured the same and it cannot.
    """
    if count < 2:
        return np.ones_like(energy_sum)
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
    return _smoothstep(TRUST_LO, TRUST_HI, eff)


def _local_spread(depth, window):
    """Standard deviation of the depth map over `window`, per pixel.

    How far the neighbourhood is from agreeing on a depth, in slices. Read
    directly as a blend width by _blend_width, which is why it is worth having
    in the units it is already in.
    """
    mean = cv2.boxFilter(depth, cv2.CV_32F, (window, window),
                         normalize=True, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(np.square(depth), cv2.CV_32F, (window, window),
                       normalize=True, borderType=cv2.BORDER_REFLECT)
    var = np.subtract(sq, np.square(mean), out=sq)
    np.maximum(var, 0.0, out=var)
    return np.sqrt(var, out=var)


def _regularise_depth(depth, trust, strength):
    """Let confident pixels dictate the depth of the ones around them.

    A normalised convolution - each window's depth averaged with the trust map
    as the weight - so a pixel that measured nothing takes the depth of
    whichever neighbours did measure something, rather than the depth its own
    noise picked. Run coarse-to-fine over DEPTH_SMOOTH_SCALES, and the result
    is mixed back in by trust at every scale, so a pixel that was confident to
    begin with keeps its own depth and only the unresolvable ones actually move.
    """
    if strength <= 0:
        return depth

    scale = strength / 100.0
    weight = np.power(trust, TRUST_EXPONENT)
    # A floor under the weights so a window in which nothing at all is trusted
    # still averages its members instead of dividing by zero.
    weight += 1e-4

    smoothed = depth
    for base in DEPTH_SMOOTH_SCALES:
        window = max(3, int(round(base * scale)))
        if window % 2 == 0:
            window += 1
        if window < 3 or min(depth.shape) < window:
            continue
        num = cv2.boxFilter(smoothed * weight, cv2.CV_32F, (window, window),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
        den = cv2.boxFilter(weight, cv2.CV_32F, (window, window),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
        num /= np.maximum(den, 1e-8)
        # trust*own + (1-trust)*propagated, folded into the buffer boxFilter
        # already allocated.
        num *= (1.0 - trust)
        smoothed = num
        smoothed += trust * depth
    return smoothed


def _blend_width(despeckled, trust, strength, count):
    """Half-width of the slice blend, per pixel, in slices.

    The product of two things, and it needs both. The neighbourhood's own
    disagreement says how wide a blend would have to be to cover what the
    pixels around here wanted - it is already in slices, so it needs no scaling
    - and the trust map says whether that disagreement is real depth or noise.
    A genuine occlusion edge disagrees violently and is trusted completely, so
    it stays hard; unresolvable background disagrees just as violently and is
    trusted not at all, so it blends wide and comes out as the local mean of
    the stack. Disagreement alone would soften every occlusion in the frame.
    """
    if strength <= 0 or count < 2:
        return None
    spread = _local_spread(despeckled.astype(np.float32), DISAGREE_WINDOW)
    ceiling = (strength / 100.0) * BLEND_MAX_SHARE * count
    np.clip(spread, 0.0, max(ceiling, 0.0), out=spread)
    spread *= (1.0 - trust)
    np.maximum(spread, BLEND_WIDTH_FLOOR, out=spread)
    return spread


def _gather_blended(load_float, count, depth, width, out_dtype, rows, cols):
    """Render the stack through a per-pixel band of slices centred on `depth`.

    One tent per pixel, `width` slices to either side, which is the step that
    stops the depth map's remaining disagreement from reaching the output as an
    edge: where the band is at its floor the pixel still resolves to a single
    frame, and where it is wide the frames inside it are averaged, so a region
    nobody resolved comes out as the local mean of the stack rather than as
    whichever slice its noise happened to name.

    Frames are visited in index order and only loaded when some pixel is
    actually asking for them, so a band that never reaches the far end of a
    deep stack does not pay to read it.
    """
    if width is None:
        # Smoothing on, blending off: the regularised depth is still followed,
        # but every band stays at its floor, so each pixel resolves to the one
        # slice nearest the depth it settled on.
        width = np.full((rows, cols), BLEND_WIDTH_FLOOR, dtype=np.float32)

    accum = None
    weight_sum = np.zeros((rows, cols), dtype=np.float32)
    for k in range(count):
        # max(0, width - |depth - k|). The textbook tent is 1 - |depth-k|/width,
        # which is this divided by a per-pixel constant; the normalisation at
        # the end cancels it, so the division is never taken.
        tent = np.abs(depth - k)
        np.subtract(width, tent, out=tent)
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

    # Every pixel's band holds at least the slice nearest its depth - the floor
    # is a full slice wide and the depth is clamped inside the stack - so the
    # accumulator is written and the weights are positive everywhere.
    np.maximum(weight_sum, 1e-8, out=weight_sum)
    accum /= weight_sum[:, :, np.newaxis] if accum.ndim == 3 else weight_sum
    return bitdepth.from_float01(accum, out_dtype)


def depthmap_impl(input_source, img_resize=None, mode=MODE_MAX,
                  kernel_size=None, thread_count=None, halo_radius=None,
                  depth_smoothing=None, slice_blending=None):
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

    `depth_smoothing` and `slice_blending` (0-100, MODE_MAX only) are what stop
    a region no frame resolves from tearing into a mosaic of hard-edged
    patches: the first regularises the depth map by how far each pixel can be
    trusted, the second renders through a per-pixel band of slices rather than
    one. Both at 0 reproduces the plain hard select exactly. See the block
    above DEFAULT_DEPTH_SMOOTHING for what each one is undoing.

    The stack's own depth is preserved end to end: an 8-bit stack returns
    uint8, a 16-bit stack returns uint16.
    """
    if mode not in (MODE_MAX, MODE_AVERAGE):
        raise ValueError(f"Unknown depth-map mode {mode!r}; "
                         f"expected {MODE_MAX!r} or {MODE_AVERAGE!r}")

    window = _resolve_kernel(kernel_size)
    halo_element = _halo_element(_resolve_halo_radius(halo_radius))
    smoothing = _resolve_percent(depth_smoothing, DEFAULT_DEPTH_SMOOTHING)
    blending = _resolve_percent(slice_blending, DEFAULT_SLICE_BLENDING)
    # MODE_AVERAGE is already a blend of every frame by construction; there is
    # no index map to regularise and no band to widen, so neither dial applies.
    coherent = mode == MODE_MAX and (smoothing > 0 or blending > 0)

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
            trust = _trust_map(energy_sum, energy_sq_sum, num_images)
            del energy_sum, energy_sq_sum
            # Both the depth the renderer follows and the disagreement that
            # sets its band are read off the despeckled map, so the two agree
            # about what the neighbourhood wanted.
            depth = _regularise_depth(despeckled.astype(np.float32), trust,
                                      smoothing)
            width = _blend_width(despeckled, trust, blending, num_images)
            del trust, despeckled
            return _gather_blended(load_float, num_images, depth, width,
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
