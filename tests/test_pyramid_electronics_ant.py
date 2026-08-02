"""
The Laplacian-pyramid method against a real capture: samples/electronics_ant.

Every other quality suite for this method fuses something that was rendered -
tests/synthetic_stack.py blurs outside a mask, samples/ composites layers at
known distances - and both are built to have an answer. That is what makes them
testable, and it is also the whole of their weakness: the defocus is the one the
generator applied and the frames register perfectly because nothing moved.

electronics_ant moved. It is 333 frames off a focus rail over an ant lying on a
circuit board, and it brings the four things a rendered stack cannot:

* real defocus, from a real iris, so out-of-focus edges double up instead of
  falling off like a Gaussian
* real focus breathing - the magnification creeps frame to frame, so the stack
  is not registered and the method has to produce something usable anyway
* real grain, over a subject that is mostly dark glossy plastic holding no
  detail at all, which is where a choose-max rule has nothing to choose on
* genuine occlusion: legs and antennae standing clear of the board behind them

What it does not bring is an answer. It ships eleven renders of the same stack
by two other programs, and not one of them is a ground truth: each aligned the
stack before fusing, so its output matches the geometry of no raw frame, and
read pixel by pixel every one of them ranks a single defocused frame five
decibels above a correct fusion. samples/captures.py sets out why in full. So
nothing here compares pixels. The claims split three ways:

* against the stack itself, which needs no reference at all - edge transfer,
  sharpness, and the envelope the collapsed pyramid has to stay inside
* against the eleven renders at tile resolution, which is coarse enough to
  survive the misalignment - did the method find detail where these programs
  found it, and how much of it
* against the method's own published settings, which is where the departures
  from Burt-Adelson choose-max are shown to still be worth making on a stack
  nobody generated

The most useful of the renders are Helicon Focus method C - its own Laplacian
pyramid - at four smoothing settings, because four settings of the same
algorithm bracket rather than score. Recovering less local contrast than
Helicon's pyramid does at its lightest smoothing means soft; more than at its
heaviest means crunchy; between them is inside the range a mature implementation
of this algorithm considers reasonable, and that is where the default sits.

Thresholds are set well clear of the measured values; they catch a method that
got materially worse, not run-to-run jitter. Measured figures are quoted next to
each one.

Run with:  python -m pytest tests/test_pyramid_electronics_ant.py -v
"""

import os
import re
import shutil
import sys
import warnings

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_methods.pyramid as pyramid_module
from fusion_methods.pyramid import pyramid_impl
from samples import captures
from tests import fusion_metrics as fm

# Read off the module rather than restated, so this cannot drift from what the
# method actually does when a default is retuned.
MODULE_DEFAULTS = {
    "levels": pyramid_module.DEFAULT_LEVELS,
    "energy_window": pyramid_module.ENERGY_WINDOW,
    "selectivity": pyramid_module.SELECTIVITY,
    "coherence": pyramid_module.COHERENCE,
    "base_selectivity": pyramid_module.BASE_SELECTIVITY,
    "noise_gate": pyramid_module.NOISE_GATE,
    "envelope": pyramid_module.ENVELOPE_CLIP,
}

CAPTURE = "electronics_ant"

if not captures.is_available(CAPTURE):
    pytest.skip(f"samples/{CAPTURE} is not present (it is photographed, not "
                f"generated, so there is nothing to run to produce it)",
                allow_module_level=True)

# Every 12th frame: 28 of the 333, at full capture resolution.
#
# The frame count is what is cut, never the resolution - a capture is here to
# say what the lens did at the pixel level, and resizing would throw exactly
# that away. 28 frames still covers the whole throw, at a focus step wide enough
# that neighbouring frames plainly differ, and fuses in under two seconds
# against twenty for all 333 at 2.2 GB of stack.
#
# Density costs the suite little: measured over 28, 84 and 333 frames the
# agreement with the Affinity render lands at 0.649, 0.708 and 0.710, so the
# subsampled stack answers the same question. What it does move is how much
# contrast is recovered, which is why the full stack is checked separately.
STEP = 12

# Tile side for the comparisons against the other programs' renders. Wide enough
# to absorb the five to ten pixels the renders sit away from the raw frames,
# narrow enough to still leave ~500 tiles to correlate over.
BLOCK = 64

# The settings the method shipped with before item 19 in
# docs/ALGORITHM_IMPROVEMENTS.md: choose-max, no noise gate, activity
# proportional base, no clamp. Fixed here rather than read from the module, so
# these tests compare against the published rule itself and not against whatever
# the defaults have since become - the same dictionary as
# tests/test_pyramid_flat_field.py.
PUBLISHED = dict(selectivity=float("inf"), coherence=0.0, noise_gate=False,
                 base_selectivity=1.0, envelope=False)

MIN_QABF_GAIN = 0.025           # measured 0.1618 against a best frame of 0.1135
MIN_SHARPNESS_RATIO = 1.25      # measured 1.63 (11.26 against 6.91)

# Q^AB/F scores one image against every source, so scoring every frame as a
# candidate as well is quadratic - 28 candidates over 28 sources at 2.2
# megapixels is three minutes, and the whole rest of this module is one. The
# baseline is therefore the best of every fourth frame, which is a spread wide
# enough to find the peak: over the full 28 the best is 0.1135 and over these
# seven it is 0.1134.
QABF_CANDIDATE_STEP = 4

# Tile agreement, over all eleven renders. The floor is what the worst of them
# measured (0.587-0.649); the margin is over the mean of the stack rather than
# over the best single frame, because agreement is the half of the comparison a
# blurred image can satisfy - see test_finds_detail_where_the_other_programs_did.
MIN_AGREEMENT = 0.50            # measured 0.587 - 0.649
MIN_AGREEMENT_OVER_MEAN = 0.05  # measured 0.096 - 0.148

# Recovered contrast, as a share of each render's. Against a single frame this
# is where the separation is, and it is startlingly consistent: the ratio sits
# between 1.63 and 1.68 against every one of the eleven renders.
MIN_DETAIL_SHARE_RATIO = 1.35   # measured 1.63 - 1.68

# Where the default has to sit inside Helicon's own pyramid smoothing sweep.
# Measured 0.916 of its lightest setting and 1.168 of its heaviest, so the
# default is bracketed; these leave room either side of that.
LIGHTEST_HELICON_PYRAMID = "HF-C-1"
HEAVIEST_HELICON_PYRAMID = "HF-C-10"
MAX_SHARE_AGAINST_LIGHTEST = 1.15    # measured 0.916
MIN_SHARE_AGAINST_HEAVIEST = 0.85    # measured 1.168

MAX_GRAIN_AGAINST_PUBLISHED = 0.90   # measured 0.74 (1.580 against 2.136)
MIN_REGISTERED_QABF_GAIN = 0.02      # measured 0.053 (0.1618 -> 0.2145)

# Fusing all 333 frames costs ~50 s and 2.2 GB. Off by default, on for a release
# check:  OPENFOCUS_FULL_CAPTURE=1 python -m pytest tests/test_pyramid_electronics_ant.py
FULL_CAPTURE = os.environ.get("OPENFOCUS_FULL_CAPTURE", "").strip() not in ("", "0")

# The renders by other software. Our own output is excluded on purpose: it is a
# baseline for "did this change what we ship", not evidence that the method
# found the right detail, and folding it in would let the method grade its own
# paper. It gets its own section at the end.
ALL_RENDERS = sorted(captures.third_party_renders(CAPTURE))
COUNTERPART = captures.capture_meta(CAPTURE)["counterpart"]
OUR_RENDERS = sorted(captures.our_renders(CAPTURE))

# Reproducing the shipped render end to end: register, fuse, contrast. Measured
# 49.4 dB against the file and 0.36 of a level of mean difference - the residual
# is the CUDA fusion path it was rendered on against this CPU one, and the
# feature matching in registration, neither of which is bitwise stable.
MIN_REPRODUCTION_PSNR = 40.0

# The registration the shipped render used, in the order its metadata records.
SHIPPED_REGISTRATION = ("scale", "homography")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stack():
    frames, _meta = captures.load_capture(CAPTURE, step=STEP)
    return frames


@pytest.fixture(scope="module")
def fused(stack):
    """The stack at the method's defaults, fused once for the whole module."""
    return pyramid_impl(list(stack))


@pytest.fixture(scope="module")
def published(stack):
    """The same stack under Burt-Adelson choose-max, for the contrasts below."""
    return pyramid_impl(list(stack), **PUBLISHED)


@pytest.fixture(scope="module")
def rendered():
    """
    Every other program's render, decoded once and kept at its native size.

    A dict rather than a parametrised fixture because several tests want to
    compare across renders in one assertion rather than once per render.
    """
    loaded = {}
    for key in ALL_RENDERS:
        image = captures.load_render(CAPTURE, key)
        if image is not None:
            loaded[key] = image
    if not loaded:
        pytest.skip("no render could be decoded (JPEG XL needs imagecodecs)")
    return loaded


def _pyramid_settings(options):
    """
    Turn a shipped render's recorded options into pyramid_impl keyword arguments.

    The metadata is written for a person to read - "Auto (5)", "Balanced (8)",
    "Strong (0.75)", "On" - because it is what the render dialog showed. The
    number in the brackets is the value that was used, so that is what comes
    back out; a setting recorded without one is left to the method's default.
    """
    def number(key):
        if key not in options:
            return None
        found = re.findall(r"-?\d+(?:\.\d+)?", options[key])
        return float(found[-1]) if found else None

    def flag(key):
        return options[key].strip().lower() == "on" if key in options else None

    settings = {
        "levels": number("PyramidLevels"),
        "energy_window": number("KernelSize"),
        "selectivity": number("PyramidSelectivity"),
        "coherence": number("PyramidCoherence"),
        "base_selectivity": number("PyramidBaseWeighting"),
        "noise_gate": flag("PyramidNoiseGate"),
        "envelope": flag("PyramidEnvelopeClip"),
    }
    for key in ("levels", "energy_window"):
        if settings[key] is not None:
            settings[key] = int(settings[key])
    return {key: value for key, value in settings.items() if value is not None}


def _against(image, render, block=BLOCK):
    """(agreement, share) for one image against one render, on a common grid."""
    ours, theirs = captures.fit_render(image, render)
    return fm.detail_agreement(ours, theirs, block)


def _quiet_mask(image, percentile=35.0):
    """
    The parts of the frame holding no detail, measured on the image itself.

    Most of this capture is dark glossy board that no frame resolves, and that
    is exactly where a selection rule has nothing to select on and grain decides
    instead. Taken from the fused result rather than from the stack so it needs
    no reference and no depth.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3)),
                              (0, 0), 8.0)
    return detail <= np.percentile(detail, percentile)


def _grain(image, quiet):
    """Standard deviation of what a 2px high-pass leaves, over the quiet mask."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((grey - cv2.GaussianBlur(grey, (0, 0), 2.0))[quiet].std())


def _envelope_excursion(fused, stack):
    """
    (under, over): how far outside the range its own frames span the result got.

    A collapsed pyramid is a sum of bands taken from different frames, and
    nothing in the sum keeps it near any of them, so it can reconstruct a value
    no frame had. In 8-bit levels; 0 means every pixel is inside the stack.
    """
    darkest = np.min(np.stack(stack), axis=0).astype(np.int32)
    brightest = np.max(np.stack(stack), axis=0).astype(np.int32)
    result = fused.astype(np.int32)
    return (int(np.maximum(darkest - result, 0).max()),
            int(np.maximum(result - brightest, 0).max()))


# ---------------------------------------------------------------------------
# The capture itself
# ---------------------------------------------------------------------------

def test_the_capture_is_intact(stack):
    """
    Guard the fixture before anything is concluded from it.

    A half-copied capture would otherwise surface as a puzzling quality failure
    rather than as the missing frames it is - and unlike the rendered scenes,
    there is no generator to re-run, so the distinction matters. The count is
    read off disk rather than off the fixture, which only holds every 12th.
    """
    meta = captures.capture_meta(CAPTURE)
    paths = captures.frame_paths(CAPTURE)
    assert len(paths) == meta["frame_count"], (
        f"{len(paths)} frames on disk, {meta['frame_count']} expected")

    width, height = meta["frame_size"]
    assert len(stack) == len(paths[::STEP])
    for frame in stack:
        assert frame.shape == (height, width, 3), "frames disagree on geometry"
        assert frame.dtype == np.uint8


def test_both_programs_and_the_whole_helicon_sweep_are_present():
    """
    The comparisons below are only worth their thresholds if the set they run
    over is the set the thresholds were measured on. In particular the bracket
    test needs Helicon's pyramid at both ends of its smoothing range, and would
    otherwise skip silently into meaninglessness.
    """
    found = captures.third_party_renders(CAPTURE)
    assert {entry["program"] for entry in found.values()} == \
        {"Affinity Photo", "Helicon Focus"}

    pyramid_renders = {key for key, entry in found.items() if entry["method"] == "C"}
    assert {LIGHTEST_HELICON_PYRAMID, HEAVIEST_HELICON_PYRAMID, COUNTERPART} \
        <= pyramid_renders, f"Helicon's pyramid sweep is incomplete: {pyramid_renders}"
    assert {entry["method"] for entry in found.values()} == {"A", "B", "C", None}

    # And our own output is kept out of that set rather than merely absent.
    assert all(not entry["ours"] for entry in found.values())
    assert captures.our_renders(CAPTURE), "no OpenFocus baseline render on disk"


@pytest.mark.parametrize("key", ALL_RENDERS)
def test_pixelwise_scores_are_meaningless_here(key, stack, fused, rendered):
    """
    Guard on the guards: the reason nothing in this module compares pixels.

    Every one of these renders came out of a program that aligned the stack
    first, so none of them shares a pixel grid with the raw frames, and PSNR
    against them ranks a single defocused frame *above* a correct fusion -
    measured about 20 dB for the best frame against 15 for the fused result, for
    all eleven, even after regrading each candidate onto the render's tone.

    Parametrised over the whole set on purpose. One program doing this could be
    a quirk of that program; two, across three different fusion algorithms and
    ten parameter settings, is a property of the capture.

    If this ever fails, that render has become comparable pixel by pixel and the
    far stronger reconstruction tests the rendered samples get should be brought
    over for it. Until then it is documentation that stops someone adding a PSNR
    assertion here and tuning the method to satisfy it.
    """
    ours, theirs = captures.fit_render(fused, rendered[key])
    fused_psnr = fm.psnr(ours, fm.match_tone(theirs, ours))

    best_frame = 0.0
    for frame in stack:
        one, theirs_one = captures.fit_render(frame, rendered[key])
        best_frame = max(best_frame, fm.psnr(one, fm.match_tone(theirs_one, one)))

    assert best_frame > fused_psnr, (
        f"a single frame no longer beats the fusion on PSNR against "
        f"{captures.render_meta(CAPTURE, key)['label']} ({best_frame:.2f} "
        f"against {fused_psnr:.2f} dB) - it may now be usable pixel by pixel")


# ---------------------------------------------------------------------------
# Against the stack, with no reference at all
# ---------------------------------------------------------------------------

def test_transfers_more_edge_information_than_any_frame(stack, fused):
    """
    Q^AB/F against the stack it was fused from: how much of the edge information
    the frames carry between them survives into the result.

    The headline claim on a real stack, because it needs no answer key. A method
    that merely averaged would score below the sharpest frame; one that picked a
    single frame would score exactly it.

    Every frame is a source; only a spread of them is a candidate, for the
    reason QABF_CANDIDATE_STEP gives.
    """
    fused_qabf = fm.qabf(fused, stack)
    best_frame = max(fm.qabf(frame, stack) for frame in stack[::QABF_CANDIDATE_STEP])

    assert fused_qabf > best_frame + MIN_QABF_GAIN, (
        f"Q_ABF {fused_qabf:.4f} against a best single frame of "
        f"{best_frame:.4f} - fusing 28 real frames gained almost nothing")


def test_is_sharper_than_every_frame(stack, fused):
    """
    No frame in this capture is sharp across more than a band of depth, so the
    result has to carry more spatial frequency than the sharpest of them.

    Weaker than the Q^AB/F test and kept because it fails differently: doubled
    edges from a misregistered stack raise Q^AB/F and this together, but a
    result that has quietly gone soft loses this one first.
    """
    fused_sf = fm.spatial_frequency(fused)
    best_frame = max(fm.spatial_frequency(frame) for frame in stack)

    assert fused_sf > best_frame * MIN_SHARPNESS_RATIO, (
        f"spatial frequency {fused_sf:.2f} against a best single frame of "
        f"{best_frame:.2f} ({fused_sf / best_frame:.2f}x)")


def test_no_pixel_falls_outside_the_stack(stack, fused):
    """
    Every pixel has to sit within the range its own frames span.

    The clamp that guarantees this was added against a rendered fixture (item 19
    in docs/ALGORITHM_IMPROVEMENTS.md); this is the same claim on a capture,
    where the smooth dark board is precisely the ground the defect appears on -
    a value below every frame is a thin dark filament over a background that had
    none, and it is the artefact a user files a bug about.
    """
    under, over = _envelope_excursion(fused, stack)
    assert (under, over) == (0, 0), (
        f"the result runs {under} levels below and {over} above the frames it "
        f"was made from - the collapsed pyramid is reconstructing structure no "
        f"frame had")


def test_the_published_rule_is_what_needed_the_clamp(stack, published):
    """
    Guard on the guard above: if choose-max stopped overshooting on this capture
    the envelope test would be passing for the wrong reason and watching
    nothing. Measured 65 levels under and 79 over.
    """
    under, over = _envelope_excursion(published, stack)
    assert max(under, over) > 8, (
        "the unclamped published rule no longer leaves the envelope on this "
        "capture, so the envelope test above has nothing left to catch")


def test_the_departures_from_choose_max_calm_the_dead_regions(stack, fused, published):
    """
    Most of this frame is dark glossy board that no frame resolves. There the
    band energies differ only by grain, choose-max stitches the result out of
    frames that disagree about what is there, and the region comes back
    speckled. The defaults have to be materially calmer over exactly those
    pixels - measured 1.58 against 2.14, a quarter less.

    Note the fused result is grainier than any single frame (1.58 against
    1.19-1.53) and that is correct: selecting the sharpest band selects its
    grain with it. The claim is about the rule, not about beating a frame.
    """
    quiet = _quiet_mask(fused)
    ours = _grain(fused, quiet)
    theirs = _grain(published, quiet)

    assert ours < theirs * MAX_GRAIN_AGAINST_PUBLISHED, (
        f"{ours:.3f} levels of grain over the detail-free regions against "
        f"{theirs:.3f} for the published rule ({ours / theirs:.2f}x) - the "
        f"selectivity and noise-gate departures are no longer paying for "
        f"themselves on a real stack")


# ---------------------------------------------------------------------------
# Against the other programs, at tile resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ALL_RENDERS)
def test_finds_detail_where_the_other_programs_did(key, stack, fused, rendered):
    """
    Tile-level agreement with one render, against what averaging the stack
    manages.

    Where in the picture detail ended up is not something a tone curve or a few
    pixels of misalignment can move, so correlating the two detail maps over
    64px tiles asks a fair question: did the method resolve the same parts of
    the scene these programs resolved, from the same capture?

    The baseline is the plain mean of the stack rather than the best single
    frame, and that is deliberate. Agreement is the half of this comparison a
    soft image can satisfy - the mean reaches 0.48-0.52 here simply by being
    dimly right everywhere - so it is the naive fusion that has to be beaten on
    it, while the sharpness half of the claim is carried by the share test below
    and by Q^AB/F above. Measured 0.59-0.65 against 0.48-0.52.
    """
    render = rendered[key]
    agreement, _share = _against(fused, render)
    mean = np.mean(np.stack(stack).astype(np.float32), axis=0).astype(np.uint8)
    naive, _ = _against(mean, render)

    label = captures.render_meta(CAPTURE, key)["label"]
    assert agreement > MIN_AGREEMENT, (
        f"tile agreement with {label} is {agreement:.3f}")
    assert agreement > naive + MIN_AGREEMENT_OVER_MEAN, (
        f"tile agreement with {label} is {agreement:.3f} against {naive:.3f} "
        f"for the plain mean of the stack - fusing barely improved on averaging")


@pytest.mark.parametrize("key", ALL_RENDERS)
def test_recovers_far_more_contrast_than_any_single_frame(key, stack, fused, rendered):
    """
    How much local contrast came back, as a share of one render's.

    Agreement says the detail is in the right places and says nothing about how
    much of it there is; a heavily blurred copy of a render correlates well and
    resolves nothing. The share against a single frame is where the separation
    lives, and it is the steadiest number in this module: measured between 1.63
    and 1.68 against all eleven renders, whichever program and whichever
    settings produced them.

    Deliberately a ratio and not a floor. The absolute share depends entirely on
    what the other program did - 0.44 against Affinity, which renders at the
    capture's full size and sharpens, against roughly 1.0 for Helicon's pyramid
    at the stack's own resolution - and none of that spread is a fact about this
    method.
    """
    render = rendered[key]
    _agreement, share = _against(fused, render)
    best_frame = max(_against(frame, render)[1] for frame in stack)

    assert share > best_frame * MIN_DETAIL_SHARE_RATIO, (
        f"recovered {share:.3f} of the tile contrast of "
        f"{captures.render_meta(CAPTURE, key)['label']} against {best_frame:.3f} "
        f"for the best single frame ({share / best_frame:.2f}x)")


def test_sits_inside_helicon_pyramids_own_smoothing_range(fused, rendered):
    """
    The one like-for-like comparison in the set.

    Helicon Focus method C is a Laplacian pyramid, the same algorithm as this
    one, and it is here at four smoothing settings. Four settings of the same
    algorithm bracket where a single render can only score: recovering less
    contrast than Helicon's pyramid does at its lightest smoothing means this
    method has gone soft, more than at its heaviest means it has gone crunchy,
    and between the two is inside the range a mature implementation of this
    algorithm considers reasonable.

    Measured 0.916 of the lightest and 1.168 of the heaviest, so the default
    sits inside the bracket with room at both ends.

    The two halves do not bite equally, and the numbers say which. The soft half
    is the sharp one: a method that stops selecting lands at 0.15-0.40 of the
    heaviest, nowhere near the bound. The crunchy half is a wide guard - the
    published choose-max rule reaches 1.03 of the lightest and passes here, and
    is caught by the envelope and grain tests instead. Tightening it to catch
    that would leave the default barely a tenth of margin, which is a threshold
    fitted to one variant rather than a claim about the method.
    """
    light = _against(fused, rendered[LIGHTEST_HELICON_PYRAMID])[1]
    heavy = _against(fused, rendered[HEAVIEST_HELICON_PYRAMID])[1]

    assert light < MAX_SHARE_AGAINST_LIGHTEST, (
        f"recovered {light:.3f} of the tile contrast of Helicon's pyramid at "
        f"its lightest smoothing - this method is now the crunchier of the two "
        f"at every setting Helicon offers")
    assert heavy > MIN_SHARE_AGAINST_HEAVIEST, (
        f"recovered {heavy:.3f} of the tile contrast of Helicon's pyramid at "
        f"its heaviest smoothing - this method is now the softer of the two at "
        f"every setting Helicon offers")


# ---------------------------------------------------------------------------
# The stack is not registered, and that is the point
# ---------------------------------------------------------------------------

def test_registration_lifts_the_result(stack, fused):
    """
    Focus breathing is in this capture whether or not anything corrects it - it
    is why every render here sits a few pixels off the raw frames - so aligning
    the stack first must transfer more edge information than not.

    Q^AB/F rather than sharpness, for the reason the rendered drift scene gives:
    misaligned frames fuse into doubled edges, and doubling an edge *raises*
    spatial frequency. A ghosted result looks busier, not softer. Each result is
    scored against the stack it was fused from, because registration crops to
    the region every frame covers and so returns a different geometry.
    """
    from core.registration import ImageRegistration

    aligned = ImageRegistration(method="scale").process(list(stack))
    raw = fm.qabf(fused, stack)
    lifted = fm.qabf(pyramid_impl(list(aligned)), aligned)

    assert lifted > raw + MIN_REGISTERED_QABF_GAIN, (
        f"Q_ABF {raw:.4f} unregistered against {lifted:.4f} registered - "
        f"correcting the rail's focus breathing gained nothing")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_the_result_keeps_the_capture_geometry(stack, fused):
    assert fused.shape == stack[0].shape
    assert fused.dtype == np.uint8


def test_repeat_runs_are_identical(stack, fused):
    """The method is declared deterministic; on a 2.2 megapixel real stack it
    has to be bitwise so, or the thresholds above are measuring luck."""
    assert np.array_equal(pyramid_impl(list(stack)), fused)


def test_thread_count_does_not_change_the_result(stack, fused):
    """Frames are decomposed concurrently and accumulated; the accumulation has
    to be order-independent, which floating-point addition is not for free."""
    assert np.array_equal(pyramid_impl(list(stack), thread_count=1), fused)


def test_the_folder_path_reads_the_webp_stack(tmp_path):
    """
    The capture is stored as WebP, and a folder of them is what a user actually
    points the app at. Exercised on a handful of frames copied out, so this
    tests the loading path rather than re-fusing the stack.
    """
    folder = tmp_path / "stack"
    folder.mkdir()
    for path in captures.frame_paths(CAPTURE, step=48)[:6]:
        shutil.copy(path, folder / os.path.basename(path))

    result = pyramid_impl(str(folder))
    width, height = captures.capture_meta(CAPTURE)["frame_size"]
    assert result.shape == (height, width, 3)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Our own shipped render, as a baseline rather than a reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", OUR_RENDERS)
def test_the_shipped_render_says_what_made_it(key):
    """
    A render of ours carries its settings, and they have to still parse.

    This is what turns the file from a picture into a baseline: the recipe is
    inside it, so it can be run again. If the metadata group is renamed or
    dropped, every render already written stops being reproducible and this is
    the only thing that would say so.
    """
    options = captures.read_render_options(CAPTURE, key)
    assert options, f"{key} carries no OpenFocus metadata group"
    assert options.get("FusionMethod") == "Pyramid"
    assert "333" in options.get("SourceImages", ""), \
        "the shipped render was not made from the whole capture"

    settings = _pyramid_settings(options)
    assert set(settings) >= {"levels", "selectivity", "coherence",
                             "base_selectivity", "noise_gate", "envelope"}, \
        f"only {sorted(settings)} could be read back out of {key}"


@pytest.mark.parametrize("key", OUR_RENDERS)
def test_every_setting_the_render_recorded_is_still_a_control(key, stack):
    """
    Each knob named in a shipped render must still be one this method accepts
    and acts on.

    A rename would not fail anything else here - the defaults would quietly take
    over and the render would still come out, just not the render that was
    asked for - and the first sign would be an old file that cannot be
    reproduced. Run on a short stack, because what is being checked is that the
    argument reaches something, not what it does.
    """
    settings = _pyramid_settings(captures.read_render_options(CAPTURE, key))
    short = stack[::4]

    at_settings = pyramid_impl(list(short), **settings)
    assert at_settings.shape == short[0].shape

    default = pyramid_impl(list(short))
    for name, value in settings.items():
        alone = pyramid_impl(list(short), **{name: value})
        moved = not np.array_equal(alone, default)
        # The claim is exact in both directions, which also checks that the
        # human-readable metadata was parsed into the value it names: a setting
        # the render recorded at something other than the method's default has
        # to change the picture, and one recorded at the default has to leave it
        # alone. Two of the seven differ here - coherence and base weighting.
        assert moved == (value != MODULE_DEFAULTS[name]), (
            f"{name}={value} against a default of {MODULE_DEFAULTS[name]} "
            f"{'changed' if moved else 'changed nothing in'} the result")


@pytest.mark.skipif(not FULL_CAPTURE,
                    reason="set OPENFOCUS_FULL_CAPTURE=1 (~60 s, 2.2 GB)")
@pytest.mark.parametrize("key", OUR_RENDERS)
def test_the_shipped_render_reproduces(key):
    """
    Run the shipped render's own recipe again and land back on the file.

    The one place in this module where pixels are comparable, and the reason is
    worth being precise about: every third-party render was aligned by the
    program that made it, so none of them shares a grid with anything here. This
    one came out of this pipeline, so registering the same way puts it back on
    the same grid - measured to the pixel, 949x1458 both times.

    That makes this the anchor for improving the method. The reference-free
    metrics say the result is good; the third-party renders say it found what
    others found; this says what changed against what we last shipped, and by
    how much. When a deliberate change moves it, re-render and replace the file.

    Measured 49.4 dB. The threshold sits well below because neither side is
    bitwise stable: the file was rendered on CUDA and this runs the CPU path,
    and the feature matching in registration is not deterministic.
    """
    from core.contrast import apply_contrast
    from core.registration import ImageRegistration

    shipped = captures.load_render(CAPTURE, key)
    if shipped is None:
        pytest.skip("the JPEG XL render needs imagecodecs to decode")
    options = captures.read_render_options(CAPTURE, key)

    frames, _meta = captures.load_capture(CAPTURE)
    for method in SHIPPED_REGISTRATION:
        frames = ImageRegistration(method=method, downscale_width=1024,
                                   reference_index=0).process(list(frames))

    result = pyramid_impl(list(frames), **_pyramid_settings(options))
    if options.get("Contrast", "").lower().startswith("auto"):
        strength = float(re.findall(r"(\d+)%", options["Contrast"])[0]) / 100.0
        result = apply_contrast(result, "auto", strength)

    assert result.shape == shipped.shape, (
        f"reproducing {key} gave {result.shape}, the file is {shipped.shape} - "
        f"the registration stages no longer crop the same way")

    reproduction = fm.psnr(result, shipped)
    assert reproduction > MIN_REPRODUCTION_PSNR, (
        f"reproducing {key} from its own recorded settings reached only "
        f"{reproduction:.2f} dB against the file - something in the pipeline "
        f"has moved since it was rendered")


def test_pooled_band_energy_is_never_negative(monkeypatch):
    """
    Pooled band energy must not reach _mix_parent negative, whatever the box
    filter hands back.

    Found by reproducing the shipped render, which uses coherence 0.75. Pooling
    is a box filter, which slides a running column sum and subtracts the column
    leaving the window, so over an exactly-zero region the subtraction can
    cancel to float32 residue of either sign - around -1e-19. Harmless until the
    geometric mix in _mix_parent raises it to a fractional power, which returns
    NaN, which survives to an undefined cast into the output. Any coherence
    above 0 reaches it, which is every preset above "Off".

    This was originally driven through the real pipeline, by a stack registered
    with "both" that came back with an irregular all-zero border. That border
    was itself a defect - the GPU ECC warp applied the inverse of the transform
    it had measured, pushing content off a canvas cropped for the forward one
    (docs/REGISTRATION_IMPROVEMENTS.md item 1). With the warp fixed, registration
    crops to the region its own transforms make valid and leaves no such border,
    on either device; and masking one in by hand does not reproduce the residue,
    because the cancellation depended on the pixels the bad warp produced.

    So the trigger is gone and the guard is not: any future pooling that returns
    a negative would still reach the fractional power. What is checked here is
    therefore the invariant _band_energy exists to enforce, with the filter made
    to return the residue that the real one once did.
    """
    from fusion_methods.pyramid import _band_energy

    real_box_filter = cv2.boxFilter

    def leaky_box_filter(src, ddepth, ksize, **kwargs):
        pooled = real_box_filter(src, ddepth, ksize, **kwargs)
        # What the running column sum left over the zero border, to scale.
        pooled[0, :4] = -1e-19
        return pooled

    monkeypatch.setattr(cv2, "boxFilter", leaky_box_filter)

    detail = np.zeros((64, 64, 3), dtype=np.float32)
    detail[16:48, 16:48] = 12.5
    pooled = _band_energy(detail, pyramid_module.ENERGY_WINDOW)

    assert pooled.min() >= 0.0, (
        f"pooled energy came back as low as {pooled.min():.3e}; a fractional "
        f"power of that is NaN, and the NaN reaches the output cast")
    # The clamp must not cost the signal it is protecting.
    assert pooled.max() > 0.0


def test_a_registered_stack_fuses_finite(stack):
    """
    The end the guard above protects: a real stack, really registered, fused at
    the coherence the shipped render used, raises nothing and stays in envelope.
    """
    from core.registration import ImageRegistration

    aligned = ImageRegistration(method="both", downscale_width=1024,
                                reference_index=0).process(list(stack))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pyramid_impl(list(aligned), coherence=0.75, base_selectivity=0.0)

    complaints = sorted({str(item.message) for item in caught
                         if "invalid value" in str(item.message)})
    assert not complaints, (
        f"fusing a registered stack at coherence 0.75 raised {complaints} - "
        f"negative pooled energy is reaching a fractional power again")
    assert result.shape == aligned[0].shape
    # A NaN cast to uint8 is undefined, so it shows as an impossible pixel
    # rather than as a NaN. The envelope is what would catch that.
    assert _envelope_excursion(result, aligned) == (0, 0)


# ---------------------------------------------------------------------------
# All 333 frames
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FULL_CAPTURE,
                    reason="set OPENFOCUS_FULL_CAPTURE=1 (~50 s, 2.2 GB)")
def test_the_whole_capture_fuses(rendered):
    """
    The subsampled stack is a stand-in, and this is the thing itself: all 333
    frames, which is what the other programs' renders were made from and the
    only fully like-for-like comparison available.

    Two claims the 28-frame fixture cannot make. The envelope has to hold with
    twelve times as many bands competing for every pixel, and the density has to
    buy detail rather than average it away, so the agreement and the bracket
    both still have to hold on the full stack.
    """
    frames, _meta = captures.load_capture(CAPTURE)
    assert len(frames) == captures.capture_meta(CAPTURE)["frame_count"]

    result = pyramid_impl(list(frames))
    assert _envelope_excursion(result, frames) == (0, 0)

    agreement, _share = _against(result, rendered[COUNTERPART])
    assert agreement > MIN_AGREEMENT, (
        f"tile agreement with {COUNTERPART} is {agreement:.3f}")

    light = _against(result, rendered[LIGHTEST_HELICON_PYRAMID])[1]
    heavy = _against(result, rendered[HEAVIEST_HELICON_PYRAMID])[1]
    assert light < MAX_SHARE_AGAINST_LIGHTEST, (
        f"recovered {light:.3f} of the tile contrast of "
        f"{LIGHTEST_HELICON_PYRAMID} on the full stack")
    assert heavy > MIN_SHARE_AGAINST_HEAVIEST, (
        f"recovered {heavy:.3f} of the tile contrast of "
        f"{HEAVIEST_HELICON_PYRAMID} on the full stack")
