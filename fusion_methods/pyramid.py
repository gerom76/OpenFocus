import os
import glob
import re
import cv2
import numpy as np

from utils import bitdepth
from utils.image_utils import read_image_any_depth
from fusion_methods.gff import _map_in_order

# Reference:
# Burt P J, Adelson E H. The Laplacian pyramid as a compact image code[J].
# IEEE Transactions on Communications, 1983, 31(4): 532-540.
# Multi-focus fusion selects, band by band, the coefficient carrying the most
# local energy - the classic "choose-max" rule on a Laplacian decomposition.
#
# Choose-max is where the classic method fails on a real stack, and every dial
# below exists because of one of those failures. See item 19 in
# docs/ALGORITHM_IMPROVEMENTS.md for the measurements.

# Number of pyramid levels (pyrDown steps) when the caller does not ask for a
# specific depth. Five bands is enough to separate coarse structure from fine
# texture on the image sizes this app renders, while staying cheap.
DEFAULT_LEVELS = 5

# Side of the window the per-band focus energy is pooled over before the
# selection. Pooling turns a per-pixel |Laplacian| - which is noisy and zero on
# flat-but-in-focus areas - into a region measure, so the selection follows
# genuinely sharp regions instead of speckling between slices.
ENERGY_WINDOW = 5

# How sharply the per-band weights favour the frame with the most energy.
#
# The published rule is choose-max: the winning coefficient is copied and the
# rest are discarded. That is the right answer where one frame is plainly
# sharper, and the wrong one everywhere else - and "everywhere else" is most of
# a macro frame. In a defocused background no frame resolves anything, the
# energies differ only by grain, and the winner map becomes a speckle field;
# the fused band is then stitched from coefficients belonging to frames that
# disagree about what is there, which reconstructs as thin dark filaments over
# smooth backgrounds - structure present in no source frame.
#
# Weighting by (energy / best energy) ** selectivity keeps choose-max where it
# was right: a frame 2x behind the winner contributes 2**-8 = 0.4% at the
# default, so a genuine focus decision is still effectively a decision. Where
# frames tie, they average instead, which is the correct answer for "nothing is
# in focus here" and removes the filaments with it.
#
# 0 averages every frame equally, inf restores the published choose-max.
SELECTIVITY = 8.0

# How much of a band's decision comes from the coarser bands above it.
#
# Each band choosing independently lets one pixel take its fine detail from
# frame 3 and its coarse detail from frame 40. Neither is wrong on its own
# measure, and the sum is a chimera - an edge at one scale sitting on a base
# that never had one. The salience of band i is therefore mixed with its
# parent's, cascaded from the top:
#
#     S[last] = e[last];   S[i] = e[i] ** (1 - c) * up(S[i + 1]) ** c
#
# so band i+k contributes with weight c**k and the bands of one frame decide
# together. 0 is per-band independence (the published rule) and 1 hands every
# band the decision made on the coarsest one.
#
# Off by default, because on the test fixtures it costs more than it buys: the
# envelope clamp below already removes the reconstructions that disagreeing
# bands produce, while a coarse-guided decision blurs the choice across a depth
# boundary the coarse band cannot see - 2.4 dB on fine_texture and 5.3 dB on
# long_stack at 0.5, for no measurable gain in flat-field noise. It is here for
# the stack where the bands visibly disagree anyway; 1.0 is not a setting to
# render with, it is the end of the range (17.3 dB on fine_texture).
COHERENCE = 0.0

# Divide each frame's band energy by that frame's own noise level before
# frames are compared.
#
# Without it the comparison is absolute, and grain is not distributed evenly
# across a stack: photon noise grows with brightness, so a frame that is a
# bright defocused veil carries several times the band energy of a dark sharp
# one in grain alone. It then wins every region where nothing is in focus and
# stamps its flat tone across the render - the same defect item 17 fixed for
# DCT. Dividing by the frame's own noise puts every frame at about 1.0 where it
# resolves nothing, so those regions tie and the selectivity rule above
# averages them.
NOISE_GATE = True

# The noise level is read off each band as this percentile of its pooled
# energies - low enough to land in whatever that band's quietest region is,
# which in a focus stack is always somewhere out of focus.
NOISE_PERCENTILE = 10.0

# Pixels carrying exactly no energy are left out of that percentile. A frame
# with a clipped-black region - a velvet backdrop, a synthetic fixture - is
# exactly zero over all of it, and reading the level off those pixels returns
# zero however much grain the rest of the frame has. Dividing by that would
# hand the frame a hundredfold advantage everywhere it *does* have energy, and
# it would take the render. A region at exactly zero needs no noise estimate:
# it cannot win anything anyway.
#
# The ratio guards what is left: a band whose quietest live tenth still sits a
# thousand times below its own median has no measurable noise level, and the
# median stands in for one. Measured against the median rather than the mean
# because the mean is set by the sharp regions - floor a frame against its own
# mean and the sharpest frame in the stack gets the largest divisor, which is
# backwards, and cost 1.8 dB on deep_stack and depth_edge alike. The absolute
# floor catches a band that is zero everywhere, where there is nothing to scale
# to at all.
NOISE_FLOOR_RATIO = 1e-3
NOISE_FLOOR = 1e-12

# How sharply the coarse base band favours the frames that won the detail
# bands. The base carries the low-frequency picture, where a hard selection
# shows as blotches rather than as detail, so this stays a weighting by
# aggregate activity rather than a selection. 0 is the plain mean the method
# used before item 16, and 1 the activity-proportional weighting that replaced
# it - which is still far too gentle when most of the stack is defocused: on
# the veil fixture 24 frames of bright haze outvote the few that resolve the
# subject, and the base comes back hazed. Raising the exponent lets the frames
# that carry detail carry the base with it: 18.5 dB at the plain mean, 30.4 at
# 1, 34.0 at 2, 35.1 at 3, and flat past 4. The gain elsewhere is small but the
# cost is too - 0.09 dB on saturated_colour, nothing measurable on the rest.
BASE_SELECTIVITY = 3.0

# Hold the result inside the range its own frames span at each pixel.
#
# Collapsing a fused pyramid is a sum of bands taken from different frames, and
# nothing in that sum keeps it near any of them: where the winner changes from
# one band to the next, the bands reconstruct a value no frame ever had. Over a
# smooth defocused background that reads as a thin dark filament - the artefact
# that prompted item 19 - and it is measurable as a pixel darker than every
# frame in the stack: 24 levels below the darkest source at worst on the
# deep_stack fixture, 2.05% of the frame more than 8 levels below it.
#
# The true all-in-focus value at a pixel comes from whichever frame resolves it,
# so it is one of the sources by construction, and any blend of them lies
# between the darkest and the brightest. Clamping there therefore removes only
# values that no frame supports. It costs one running min and max over the
# stack, which is two frames of memory, and it is what the block methods get
# for free by compositing source pixels rather than coefficients.
ENVELOPE_CLIP = True


def _resolve_levels(height, width, requested):
    """Clamp the decomposition depth so the coarsest band stays workable.

    Each level halves both sides; decomposing past a handful of pixels buys no
    focus information and risks a degenerate 1-pixel band. The depth is capped
    so the smallest Gaussian level keeps both sides >= 2.
    """
    requested = DEFAULT_LEVELS if requested is None else max(1, int(requested))
    depth = 0
    smallest = min(height, width)
    while depth < requested and (smallest + 1) // 2 >= 2:
        smallest = (smallest + 1) // 2
        depth += 1
    return max(1, depth)


def _resolve_window(requested):
    """Coerce the energy pooling window to a positive odd integer."""
    if requested is None:
        return ENERGY_WINDOW
    window = max(1, int(requested))
    return window if window % 2 else window + 1


def _resolve_exponent(requested, default):
    """Coerce a weighting exponent to a non-negative float; None keeps default."""
    if requested is None:
        return default
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return default
    if value != value:            # NaN, from a config file that lost its type
        return default
    return max(0.0, value)


def _resolve_coherence(requested):
    """Coerce the cross-scale mix to [0, 1]; None keeps the default."""
    if requested is None:
        return COHERENCE
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return COHERENCE
    if value != value:
        return COHERENCE
    return min(max(value, 0.0), 1.0)


def _laplacian_pyramid(img, levels):
    """Return (detail levels 0..levels-1, coarsest Gaussian base).

    pyrUp is given an explicit destination size at every step so an odd-sized
    level reconstructs to exactly the size it came from - the pipeline hands
    back the geometry it was given, and the tiled renderer accumulates fused
    tiles at fixed offsets.
    """
    gaussian = [img]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

    detail = []
    for i in range(levels):
        target = (gaussian[i].shape[1], gaussian[i].shape[0])
        up = cv2.pyrUp(gaussian[i + 1], dstsize=target)
        detail.append(gaussian[i] - up)
    return detail, gaussian[levels]


def _band_energy(detail, window):
    """Local focus energy of one detail band: pooled squared response.

    Summing the squared coefficients across colour channels first keeps the
    decision on a single grey activity map, so all three channels of a pixel are
    taken from the same slice and colour cannot split across sources.
    """
    if detail.ndim == 3:
        squared = np.sum(detail * detail, axis=2)
    else:
        squared = detail * detail
    if window <= 1:
        return squared
    return cv2.boxFilter(squared, cv2.CV_32F, (window, window),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def _noise_level(energy):
    """The level below which this band of this frame resolves nothing.

    Read as a low percentile of the pooled energies, on a subsample - a
    percentile is a partition of the whole array, and at full resolution that
    costs more than the pyramid it is measuring. Every fourth row and column of
    a 4K band still leaves a quarter of a million samples behind one number.
    """
    step = max(1, int(np.sqrt(energy.size / 65536.0)))
    sample = energy[::step, ::step]
    live = sample[sample > 0.0]
    if live.size == 0:
        return NOISE_FLOOR
    level, middle = np.percentile(live, [NOISE_PERCENTILE, 50.0])
    return max(float(level), float(middle) * NOISE_FLOOR_RATIO, NOISE_FLOOR)


def _mix_parent(energy, parent, coherence):
    """Weighted geometric mean of a band's energy and its parent's salience.

    Geometric rather than arithmetic because only the ratios between frames
    decide anything: multiplying one band's energies by a constant - which is
    all the difference in scale between two pyramid levels amounts to - leaves
    every frame's share of that band untouched. An arithmetic mix would let the
    band with the larger numbers speak for both.
    """
    if coherence >= 1.0:
        return parent
    if coherence == 0.5:          # much cheaper than two calls into pow
        return np.sqrt(energy * parent, dtype=np.float32)
    return (np.power(energy, 1.0 - coherence, dtype=np.float32)
            * np.power(parent, coherence, dtype=np.float32))


def _ratio_power(ratio, exponent):
    """`ratio ** exponent` for ratios in [0, 1], consuming `ratio`.

    The exponents in play are small and usually integral, and repeated squaring
    costs one multiply per set bit against roughly ten for a call into pow -
    per pixel, per band, per frame, which is where this method spends its time.
    Powers of two, which is what the presets offer, square in place and
    allocate nothing at all: at 2048x1364 each avoided temporary is 11 MB
    through the memory bus.

    The argument is a freshly divided ratio at both call sites, so overwriting
    it is safe and saves the copy.
    """
    if exponent == 1.0:
        return ratio
    whole = int(exponent)
    if whole != exponent or not 0 < whole <= 64:
        return np.power(ratio, exponent, dtype=np.float32)
    if whole & (whole - 1) == 0:                  # 2, 4, 8, 16, 32, 64
        while whole > 1:
            ratio *= ratio
            whole >>= 1
        return ratio
    result = None
    base = ratio
    while whole:
        if whole & 1:
            result = base if result is None else result * base
        whole >>= 1
        if whole:
            base = base * base
    return result


class _Accumulator:
    """Running weighted sum of one band across the stack.

    Each frame contributes with weight (salience / best salience so far) **
    exponent. The best salience is not known until the last frame has been
    seen, and holding every frame's bands to find it out would put the whole
    stack in memory - so the peak is tracked as the frames arrive and the
    accumulated sums are rescaled whenever it rises. Rescaling by
    (old peak / new peak) ** exponent leaves the result identical to what a
    second pass would have produced, because every term ends up divided by the
    same final peak, and that factor cancels against the weight sum.

    `add` consumes the band it is given: it is one frame's, and the caller
    drops it immediately afterwards.
    """

    __slots__ = ("total", "weight", "peak", "exponent")

    def __init__(self, band, exponent):
        self.exponent = exponent
        self.total = np.zeros_like(band)
        self.weight = np.zeros(band.shape[:2], dtype=np.float32)
        # -inf for the winner-take-all path, so the first frame wins outright
        # even where its energy is zero. The weighted path starts at the noise
        # floor rather than at zero: every ratio below divides by the running
        # peak, and starting from a positive number means no division needs
        # guarding. It cannot distort a weight, since the peak is never below
        # the salience being divided by it.
        self.peak = np.full(band.shape[:2],
                            -np.inf if exponent == np.inf else NOISE_FLOOR,
                            dtype=np.float32)

    def add(self, band, salience):
        if self.exponent == np.inf:
            # The published choose-max, kept bit-for-bit: strictly greater, so
            # ties go to the earlier frame and the result does not depend on
            # the order the thread pool happened to finish in.
            better = salience > self.peak
            np.copyto(self.peak, salience, where=better)
            np.copyto(self.total, band,
                      where=better[:, :, None] if band.ndim == 3 else better)
            np.copyto(self.weight, 1.0, where=better)
            return

        peak = np.maximum(self.peak, salience)
        if self.exponent > 0:
            # One scratch array covers both ratios: the rescale has been spent
            # by the time the weight is wanted. Where no frame has had any
            # energy yet both come out at ~0 and the pixel ends with no
            # contributors, which the reconstruction reads as "no detail here"
            # - exactly what a band of zeros means.
            scratch = np.divide(self.peak, peak)
            rescale = _ratio_power(scratch, self.exponent)
            self.total *= rescale[:, :, None] if self.total.ndim == 3 else rescale
            self.weight *= rescale
            weight = _ratio_power(np.divide(salience, peak, out=scratch),
                                  self.exponent)
        else:
            # Exponent 0: every frame counts the same, and the peak is only
            # tracked so the zero-energy case stays consistent.
            weight = np.ones_like(salience)
        # Weighted in place. The band belongs to the frame being folded in and
        # is dropped as soon as this returns, so scaling it saves a full
        # three-channel temporary per band per frame - 33 MB of traffic on a
        # 2048x1364 stack, on the hottest loop the method has.
        np.multiply(band, weight[:, :, None] if band.ndim == 3 else weight,
                    out=band)
        self.total += band
        self.weight += weight
        self.peak = peak

    def result(self):
        """The weighted mean, with the untouched pixels left at zero."""
        weight = np.where(self.weight > 0.0, self.weight, 1.0)
        return self.total / (weight[:, :, None] if self.total.ndim == 3
                             else weight)


def _frame_salience(detail, base_shape, window, coherence, noise_gate):
    """One frame's per-band selection salience, plus its base-band activity.

    Returns (salience per detail band, activity on the base grid). Both are
    derived from the same pooled band energies: the salience decides the detail
    bands, the activity - those energies cascaded down to the coarsest grid -
    weights the base, so the frames that win the detail also carry the coarse
    band instead of a plain mean ghosting the blurred frames into it.
    """
    energies = [_band_energy(band, window) for band in detail]
    if noise_gate:
        for energy in energies:
            energy /= _noise_level(energy)

    activity = None
    for i, energy in enumerate(energies):
        activity = energy if activity is None else activity + energy
        shape = energies[i + 1].shape[:2] if i + 1 < len(energies) else base_shape
        activity = cv2.pyrDown(activity, dstsize=(shape[1], shape[0]))

    if coherence > 0.0:
        for i in range(len(energies) - 2, -1, -1):
            height, width = energies[i].shape[:2]
            parent = cv2.pyrUp(energies[i + 1], dstsize=(width, height))
            energies[i] = _mix_parent(energies[i], parent, coherence)

    return energies, activity


def pyramid_impl(input_source, img_resize=None, levels=None, thread_count=None,
                 energy_window=None, selectivity=None, coherence=None,
                 noise_gate=None, base_selectivity=None, envelope=None):
    """Laplacian-pyramid multi-focus fusion.

    Each frame is split into band-pass detail levels plus a low-frequency base.
    Every detail band is then a weighted mean over the stack, weighted by how
    far each frame's pooled band energy falls behind the best on offer, and the
    base is weighted by aggregate activity. Collapsing the fused pyramid gives
    the all-in-focus image.

    The published rule takes the winner outright at every band, which is
    `selectivity=inf` here. Everything else this method exposes exists because
    that rule has no answer for the parts of a frame where nothing is in focus,
    which on a macro stack is most of it - see the notes at the top of the
    module.

    The stack's own depth is preserved end to end: an 8-bit stack returns
    uint8, a 16-bit stack returns uint16.

    Args:
        input_source: Directory path, or a list of BGR uint8/uint16 arrays.
        img_resize: Optional (width, height) the frames are resized to first.
        levels: Decomposition depth. None uses DEFAULT_LEVELS. Self-limiting,
            so the coarsest band keeps both sides >= 2 px.
        thread_count: Frames decomposed concurrently. None picks a default.
        energy_window: Side of the window each band's energy is pooled over
            before frames are compared. Small follows fine detail and speckles
            on grain; large decides regionally and rounds off narrow in-focus
            structures. Coerced to odd.
        selectivity: How sharply the weights favour the sharpest frame.
            0 averages the stack, inf is the published choose-max, and the
            default sits high enough that a real focus decision is still a
            decision.
        coherence: How much of each band's decision comes from the coarser
            bands, in [0, 1]. 0 lets every band choose alone.
        noise_gate: Compare frames in units of their own noise rather than
            absolutely, so a bright grainy frame cannot win regions that hold
            no detail. None uses NOISE_GATE.
        base_selectivity: The same weighting exponent for the coarse base band,
            where a hard selection shows as blotches rather than as detail.
            0 restores the plain mean.
        envelope: Clamp every pixel to the range its own frames span, so a
            collapsed pyramid cannot reconstruct a value no frame had. None
            uses ENVELOPE_CLIP.

    Returns:
        Fused BGR image at the stack's own depth.
    """
    if thread_count is None:
        max_workers = min(8, os.cpu_count() or 4)
    else:
        try:
            max_workers = max(1, int(thread_count))
        except (TypeError, ValueError):
            max_workers = min(8, os.cpu_count() or 4)

    window = _resolve_window(energy_window)
    selectivity = _resolve_exponent(selectivity, SELECTIVITY)
    base_selectivity = _resolve_exponent(base_selectivity, BASE_SELECTIVITY)
    coherence = _resolve_coherence(coherence)
    noise_gate = NOISE_GATE if noise_gate is None else bool(noise_gate)
    envelope = ENVELOPE_CLIP if envelope is None else bool(envelope)

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

    levels = _resolve_levels(rows, cols, levels)

    def load_frame(k):
        # Per-frame resize, so only the handful of frames in flight are ever
        # held as float32 rather than the whole stack. The integer frame is
        # returned alongside for the envelope clamp: without a resize it is the
        # caller's own array, so it costs a reference rather than a copy.
        img = stack_ori[k]
        if img_resize and (img.shape[1], img.shape[0]) != (cols, rows):
            img = cv2.resize(img, (cols, rows))
        if envelope and img.dtype != out_dtype:
            img = bitdepth.convert(img, out_dtype)
        return img, bitdepth.to_float01(img)

    # A single frame decides every band's shape; all frames share it because
    # they share (rows, cols) after the resize step above.
    template_detail, template_base = _laplacian_pyramid(load_frame(0)[1], levels)
    detail_acc = [_Accumulator(band, selectivity) for band in template_detail]
    base_acc = _Accumulator(template_base, base_selectivity)
    base_shape = template_base.shape[:2]
    del template_detail, template_base

    def decompose(k):
        # The energy measurement runs on the pool with the decomposition it
        # belongs to; only the accumulation below has to be serialised.
        frame, image = load_frame(k)
        detail, base = _laplacian_pyramid(image, levels)
        salience, activity = _frame_salience(detail, base_shape, window,
                                             coherence, noise_gate)
        return detail, base, salience, activity, frame

    lowest = highest = None

    # Frames are decomposed on a thread pool but reduced in index order, so the
    # running sums and the strict-'>' tie-break are stable run to run.
    for _, (detail, base, salience, activity, frame) in _map_in_order(
            decompose, num_images, max_workers):
        for i, band in enumerate(detail):
            detail_acc[i].add(band, salience[i])
        base_acc.add(base, activity)
        if envelope:
            if lowest is None:
                lowest, highest = frame.copy(), frame.copy()
            else:
                np.minimum(lowest, frame, out=lowest)
                np.maximum(highest, frame, out=highest)
        del detail, base, salience, activity, frame

    # ---------- Reconstruction ----------
    fused = base_acc.result()
    base_acc = None
    for i in range(levels - 1, -1, -1):
        target = (detail_acc[i].total.shape[1], detail_acc[i].total.shape[0])
        fused = detail_acc[i].result() + cv2.pyrUp(fused, dstsize=target)
        detail_acc[i] = None

    result = bitdepth.from_float01(fused, out_dtype)
    if envelope and lowest is not None:
        # Clamped after quantisation, so the bound is exactly the source levels
        # rather than a float that rounds either side of them.
        np.clip(result, lowest, highest, out=result)
    return result
