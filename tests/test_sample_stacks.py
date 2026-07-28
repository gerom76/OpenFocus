"""
The fusion methods measured against the depth-rendered samples in samples/.

The other suites score the methods on stacks built by blurring everything
outside a mask. That is the right fixture for "is this method broken", but the
sharp and blurred regions there come from the same image, so nothing ever
spreads across the boundary between them - and a stack where nothing spreads
cannot show what fusion does at an occlusion edge, which is where the visible
failures are.

The samples are rendered from depth instead: layers at known distances, each
convolved with its circle of confusion before being composited. That buys
claims the mask fixtures cannot support, and this module is the set of them:

* a defocused foreground genuinely spreads over the background, so the halo at
  an occlusion edge can be measured against exactly where the depth breaks
* every pixel has a label saying which frame is focused on it, so "the fused
  pixel came from the right frame" is checkable rather than assumed
* one scene is recorded at 16 bits, so the deep path can be compared against
  the 8-bit one on identical content
* one scene carries focus breathing and drift, with the transform that was
  applied to each frame recorded, so registration has something to undo and
  the undo can be checked

Thresholds are set well below (or above) the measured values - they catch a
method that got materially worse, not run-to-run jitter.

The samples are generated, not committed by hand. If they are absent:

    python samples/generate_samples.py

Run with:  python -m pytest tests/test_sample_stacks.py -v
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_metrics as fm
from tests import fusion_registry as reg

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "samples")

if not os.path.exists(os.path.join(SAMPLES_DIR, "manifest.json")):
    pytest.skip("samples/ not generated; run `python samples/generate_samples.py`",
                allow_module_level=True)

from samples.load import list_scenes, load_stack     # noqa: E402  (after the skip)

# The six distinct CPU algorithms, matching the ratchet in
# test_fusion_regression.py. The GPU twins duplicate them and are covered by the
# parity test there.
METHODS = ["guided_filter", "gfgfgf", "dct", "dtcwt", "pyramid", "depthmap_max"]

# Scenes where every method clearly beats a single frame. pale_specimen and
# handheld_drift are excluded on purpose and get their own weaker contracts
# below - see the docstrings there.
MARGIN_SCENES = ["macro_dome", "tilted_print", "circuit_steps", "fibre_thicket"]
MIN_MARGIN_DB = 1.5             # measured range at time of writing: 2.75 - 11.9

# pale_specimen is low contrast and noisy by design, and the methods barely
# gain on it - one of them loses slightly. The contract there is only that
# fusion does not make things materially worse than simply picking a frame.
MAX_REGRESSION_DB = 0.75        # measured worst at time of writing: -0.24

# The per-pixel focus label is exact by construction but only meaningful where
# the depth of a pixel is a sensible thing to state at all. Each scene records
# how far a plain focus measure agrees with its own labels; below this the
# label is ambiguous and nothing should be asserted against it.
MIN_LABEL_AGREEMENT = 0.85
MAX_LABEL_RATIO = 0.90          # measured worst at time of writing: 0.77

# Error right at a depth discontinuity, both as a multiple of the error well
# away from one and in absolute levels. Two bounds because neither alone is
# enough: the ratio isolates a halo but divides out overall quality, so a
# uniformly bad result sails through it at about 1.0, while the absolute bound
# catches that and would itself drift up unnoticed if a method traded edge
# quality for background quality.
MAX_EDGE_ERROR_RATIO = 15.0     # measured range at time of writing: 5.6 - 10.3
MAX_EDGE_ERROR_LEVELS = 20.0    # real methods 10.2 - 14.8; broken ones 22.7 - 29.2

MIN_QABF_GAIN = 0.005           # measured range at time of writing: 0.012 - 0.048
MIN_UNDRIFT_GAIN_DB = 3.0       # measured range at time of writing: 5.8 - 6.4
MAX_BIT_DEPTH_DELTA_DB = 0.5    # measured worst at time of writing: 0.02


def _params():
    """One pytest param per method, pre-marked with its skip reason."""
    out = []
    for key in METHODS:
        method = reg.get(key)
        ok, reason = method.available()
        marks = [] if ok else [pytest.mark.skip(reason=f"{method.label}: {reason}")]
        out.append(pytest.param(method, id=key, marks=marks))
    return out


ALL_METHODS = _params()


def _label_scenes():
    """Scenes whose focus labels the generator judged reliable enough to test."""
    with open(os.path.join(SAMPLES_DIR, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)["scenes"]
    return [scene["name"] for scene in manifest
            if scene["focus_index_agreement"] >= MIN_LABEL_AGREEMENT]


LABEL_SCENES = _label_scenes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Scenes(dict):
    """
    The samples, loaded on first use and kept.

    Reading a stack per test would dominate the run, but so would reading every
    stack for a test that wants one: the set now includes a scene at the
    photograph's full 2560x1430, and decoding its fourteen frames costs more
    than all six synthetic scenes together. Only the integrity test touches
    everything, so everything is only paid for when it is asked for.
    """

    def __missing__(self, name):
        self[name] = load_stack(name)
        return self[name]


@pytest.fixture(scope="module")
def scenes():
    return _Scenes()


@pytest.fixture(scope="module")
def fused_cache():
    """Fuse each (method, scene) pair at most once across the whole module."""
    return {}


def _fused(method, scenes, cache, name):
    key = (method.key, name)
    if key not in cache:
        cache[key] = method.run(scenes[name][0])
    return cache[key]


def _texture_mask(reference, percentile=70):
    """
    The pixels where the frames can actually be told apart.

    Over a smooth patch every frame in the stack looks alike, so asking which
    one the fused result resembles has no answer, and averaging that non-answer
    into a score only dilutes it. Detail is where the question means something.
    """
    grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3)), (0, 0), 3.0)
    return energy > np.percentile(energy, percentile)


# ---------------------------------------------------------------------------
# The sample set itself
# ---------------------------------------------------------------------------

def test_sample_set_is_intact(scenes):
    """
    Guard the dataset before anything is concluded from it.

    A half-generated samples/ would otherwise surface as a puzzling quality
    failure in an unrelated test rather than as the missing frames it is.
    """
    names = list_scenes()
    assert names, "manifest.json lists no scenes"

    # Driven from the manifest, not from whatever the fixture happens to hold:
    # it loads lazily, so iterating it would check only what earlier tests
    # already asked for - and in a fresh run, nothing at all.
    for name in names:
        stack, truth, meta = scenes[name]
        assert len(stack) == meta["frame_count"], f"{name}: frames missing"
        assert all(frame.shape == stack[0].shape for frame in stack), \
            f"{name}: frames disagree on geometry"
        assert truth["all_in_focus"].shape == stack[0].shape, f"{name}: truth mismatch"
        assert truth["focus_index"].max() < len(stack), \
            f"{name}: focus label names a frame that does not exist"

        near, far = meta["depth_range"]
        assert near < far
        assert truth["depth"].min() >= near - 1e-3
        assert truth["depth"].max() <= far + 1e-3


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("name", MARGIN_SCENES)
def test_beats_every_input_frame(method, name, scenes, fused_cache):
    """
    Fusion must clearly beat the best single frame it was given.

    These stacks are 12 to 18 frames over a scene where no frame is sharp
    across more than a band, so the margin is the whole point of running the
    method at all.
    """
    stack, truth, _ = scenes[name]
    fused = _fused(method, scenes, fused_cache, name)

    reference = truth["all_in_focus"]
    fused_psnr = fm.psnr(fused, reference)
    best_frame = max(fm.psnr(frame, reference) for frame in stack)

    assert fused_psnr > best_frame + MIN_MARGIN_DB, (
        f"{method.label} on {name}: {fused_psnr:.2f} dB against a best single "
        f"frame of {best_frame:.2f} dB")


@pytest.mark.parametrize("method", ALL_METHODS)
def test_low_contrast_scene_does_not_fall_apart(method, scenes, fused_cache):
    """
    pale_specimen is the scene the methods are worst on, and that is by design:
    grain is a large fraction of the local variation, so a focus measure has
    little to go on. Some methods gain under a decibel here and one loses a
    little, so demanding a margin would be encoding a wish rather than a fact.

    What must hold is that fusing does not do materially worse than picking one
    frame and keeping it. A method that fails this is not struggling with a
    hard scene, it is actively destroying detail it was handed.
    """
    stack, truth, _ = scenes["pale_specimen"]
    fused = _fused(method, scenes, fused_cache, "pale_specimen")

    reference = truth["all_in_focus"]
    fused_psnr = fm.psnr(fused, reference)
    best_frame = max(fm.psnr(frame, reference) for frame in stack)

    assert fused_psnr > best_frame - MAX_REGRESSION_DB, (
        f"{method.label} on pale_specimen: {fused_psnr:.2f} dB, worse than the "
        f"best single frame at {best_frame:.2f} dB")


# ---------------------------------------------------------------------------
# Did it take the pixels from the right frame?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("name", LABEL_SCENES)
def test_tracks_the_labelled_focus_frame(method, name, scenes, fused_cache):
    """
    Where the truth says frame k is focused, the fused pixels must resemble
    frame k more than they resemble a frame from elsewhere in the stack.

    PSNR against the all-in-focus render says the answer is close on average;
    it does not say the detail was taken from the frame that actually had it.
    A method can score respectably while sourcing a region from a neighbour
    that happens to be similar, and this is the check that separates the two.

    Frames within three of the label are excluded from the comparison: they
    differ from it by less than the blur step, so a method blending across the
    transition is being sensible rather than wrong.
    """
    stack, truth, _ = scenes[name]
    fused = _fused(method, scenes, fused_cache, name).astype(np.float32)

    labels = truth["focus_index"]
    textured = _texture_mask(truth["all_in_focus"])
    frames = [frame.astype(np.float32) for frame in stack]

    worst, worst_label = 0.0, None
    tested = 0
    for label in range(len(stack)):
        region = (labels == label).astype(np.uint8)
        if region.sum() < 2000:
            continue
        # Erode: pixels straddling two labels belong to neither cleanly
        region = cv2.erode(region, np.ones((9, 9), np.uint8)).astype(bool) & textured
        if region.sum() < 600:
            continue

        own = np.abs(fused[region] - frames[label][region]).mean()
        distant = [np.abs(fused[region] - frames[other][region]).mean()
                   for other in range(len(stack)) if abs(other - label) >= 4]
        if not distant:
            continue

        tested += 1
        ratio = own / min(distant)
        if ratio > worst:
            worst, worst_label = ratio, label

    if tested < 3:
        pytest.skip(f"{name}: too few labelled regions carry testable detail")

    assert worst < MAX_LABEL_RATIO, (
        f"{method.label} on {name}: in the region labelled frame {worst_label} the "
        f"result is {worst:.2f}x closer to a frame at least 4 away than to the "
        f"frame focused there")


@pytest.mark.parametrize("method", ALL_METHODS)
def test_occlusion_edges_do_not_halo(method, scenes, fused_cache):
    """
    Error along a depth discontinuity, against error well away from one.

    circuit_steps is four flat planes with hard silhouettes, so the exact depth
    map gives the discontinuities to the pixel. Every method smears an edge
    where a defocused foreground overlaps a sharp background - some error there
    is unavoidable and it runs several times the background level for all of
    them. This catches the case where it has grown from a soft fringe into the
    halo a user would file a bug about.
    """
    stack, truth, _ = scenes["circuit_steps"]
    fused = _fused(method, scenes, fused_cache, "circuit_steps").astype(np.float32)

    depth = truth["depth"]
    span = float(depth.max() - depth.min())
    steps = (cv2.morphologyEx(depth, cv2.MORPH_GRADIENT,
                              np.ones((3, 3), np.uint8)) > 0.05 * span).astype(np.uint8)
    at_edge = cv2.dilate(steps, np.ones((13, 13), np.uint8)).astype(bool)
    away = ~cv2.dilate(steps, np.ones((41, 41), np.uint8)).astype(bool)

    error = np.abs(fused - truth["all_in_focus"].astype(np.float32)).mean(axis=2)
    edge_error = float(error[at_edge].mean())
    ratio = edge_error / max(float(error[away].mean()), 1e-6)

    assert edge_error < MAX_EDGE_ERROR_LEVELS, (
        f"{method.label}: {edge_error:.2f} levels of error along depth edges")
    assert ratio < MAX_EDGE_ERROR_RATIO, (
        f"{method.label}: {edge_error:.2f} levels of error at depth edges "
        f"against {error[away].mean():.2f} away from them ({ratio:.1f}x)")


# ---------------------------------------------------------------------------
# Bit depth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_sixteen_bit_matches_eight_bit_on_the_same_scene(method, scenes, fused_cache):
    """
    tilted_print is stored at 16 bits, so the same content can be run through
    both paths and compared. The deep path must return 16-bit data and land in
    the same place - a precision path that quietly degrades the result is worse
    than not having one, because the cost is paid in file size either way.
    """
    deep_stack, deep_truth, _ = load_stack("tilted_print", as_uint8=False)
    deep = method.run(deep_stack)

    assert deep.dtype == np.uint16, (
        f"{method.label} returned {deep.dtype} for a 16-bit stack")

    # Compare in 8 bits, which is the only scale the metrics are defined on
    def to_eight(img):
        return (img.astype(np.float32) / 257.0).round().clip(0, 255).astype(np.uint8)

    deep_psnr = fm.psnr(to_eight(deep), to_eight(deep_truth["all_in_focus"]))

    _, truth, _ = scenes["tilted_print"]
    shallow_psnr = fm.psnr(_fused(method, scenes, fused_cache, "tilted_print"),
                           truth["all_in_focus"])

    assert abs(deep_psnr - shallow_psnr) < MAX_BIT_DEPTH_DELTA_DB, (
        f"{method.label} on tilted_print: {deep_psnr:.2f} dB at 16 bits against "
        f"{shallow_psnr:.2f} dB at 8 bits")


# ---------------------------------------------------------------------------
# Movement between frames
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registered_drift(scenes):
    """
    handheld_drift put through the scale/focus-breathing registration.

    Module scoped because registering sixteen frames is the slowest thing here
    and every method wants the same result.
    """
    from core.registration import ImageRegistration

    stack = scenes["handheld_drift"][0]
    return ImageRegistration(method="scale").process(list(stack))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_registration_improves_a_drifting_stack(method, scenes, registered_drift):
    """
    Registering handheld_drift must transfer more edge information than not.

    The comparison is Q^AB/F, each result against the stack it was fused from,
    because registration crops to the region every frame covers and so returns
    a different geometry - scoring both against the all-in-focus render would
    compare a cropped result to an uncropped reference and measure the crop.

    Sharpness is the wrong metric here for a subtler reason: misaligned frames
    fuse into doubled edges, and doubling an edge *raises* spatial frequency.
    A ghosted result looks busier, not softer.
    """
    stack = scenes["handheld_drift"][0]
    raw = fm.qabf(method.run(stack), stack)
    aligned = fm.qabf(method.run(registered_drift), registered_drift)

    assert aligned > raw + MIN_QABF_GAIN, (
        f"{method.label}: Q_ABF {raw:.4f} unregistered against {aligned:.4f} "
        f"registered - alignment gained nothing")


def test_recorded_drift_is_what_costs_the_quality(scenes):
    """
    Undo each frame's recorded transform and the loss must come back.

    This is the sample set holding up its own end of the bargain. If fusing the
    un-warped stack did not recover most of the gap to macro_dome, the drifted
    scene would be penalising methods for something other than the movement
    scene.json claims is in it, and every registration result measured against
    it would be misattributed.
    """
    stack, truth, meta = scenes["handheld_drift"]
    affines = meta["per_frame_affine"]
    assert affines, "handheld_drift did not record its per-frame transforms"

    height, width = stack[0].shape[:2]
    undrifted = [cv2.warpAffine(frame, cv2.invertAffineTransform(np.array(matrix, np.float32)),
                                (width, height), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REFLECT)
                 for frame, matrix in zip(stack, affines)]

    method = reg.get("pyramid")
    reference = truth["all_in_focus"]
    drifted_psnr = fm.psnr(method.run(stack), reference)
    undrifted_psnr = fm.psnr(method.run(undrifted), reference)

    assert undrifted_psnr > drifted_psnr + MIN_UNDRIFT_GAIN_DB, (
        f"undoing the recorded drift moved quality {drifted_psnr:.2f} -> "
        f"{undrifted_psnr:.2f} dB; the scene's difficulty is not the drift it "
        f"documents")
