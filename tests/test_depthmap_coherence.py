"""
Weight-coherence tests for the Depth Map (Average) fusion method (depthmap.py).

At any selectivity high enough to keep the haze off a deep stack the blend is a
hard select in all but name - on the reference 333-frame capture the winning
frame takes 96% of the weight at the median resolved pixel - and it inherits the
hard select's failure without inheriting MODE_MAX's cure for it. Over a region
no frame resolves, neighbouring pixels are handed frames from opposite ends of
the stack, which do not carry the same local brightness, and the region breaks
into blotches.

`coherence_radius` passes each frame's share of the blend through a guided
filter before the pixels are gathered. These tests pin what that has to mean:
off is off, a featureless region gets averaged, a region the picture explains
does not, and the shares still add up to a blend.

`slice_radius` pools along the other axis and both modes read it - the average
over each frame's share, MODE_MAX over the tent it renders its depth map
through - so its tests cover both. On MODE_MAX it is the only dial that touches
the rendering rule at all, which is why it is also the only one whose quality
claim can be made on a fixture rather than on a capture; see the frame-axis
section below.

Thresholds are loose regression guards, not a leaderboard.

Run with:  python -m pytest tests/test_depthmap_coherence.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.depthmap import (  # noqa: E402
    COHERENCE_EPS,
    DEFAULT_COHERENCE_RADIUS,
    DEFAULT_SLICE_RADIUS,
    MODE_AVERAGE,
    MODE_MAX,
    _guided_filter,
    _resolve_coherence_radius,
    _resolve_slice_radius,
    depthmap_impl,
)

SIZE = 160
KERNEL = 9


def _incoherent_stack(slices=40, seed=7, noise_std=6.0):
    """A stack whose left half no frame ever resolves, and whose right half is
    resolved by one frame per band.

    Enough to exercise the plumbing - two passes, the filter, the normalisation
    - on something small and fast. It is deliberately *not* used to make a
    quality claim: a rendered stack's flat region is flat, so every frame
    measures the same noise there (peak energy 1.25x the mean across 40 frames)
    and the unfiltered blend already averages it correctly. The artefact this
    stage exists for needs a region that no frame resolves and every frame gets
    wrong *differently*, which is a property of real defocus over a real
    surface. That claim is made against the capture below.
    """
    rng = np.random.default_rng(seed)
    ref = np.full((SIZE, SIZE, 3), 110.0, dtype=np.float32)
    texture = rng.random((SIZE, SIZE // 2, 3)).astype(np.float32) * 150 + 50
    ref[:, SIZE // 2:] = texture

    bands = np.linspace(0, SIZE, 6).round().astype(int)
    stack = []
    for k in range(slices):
        img = cv2.GaussianBlur(ref, (0, 0), 4.0)
        band = bands[k % 5], bands[k % 5 + 1]
        img[band[0]:band[1]] = ref[band[0]:band[1]]
        img += rng.normal(0.0, noise_std, img.shape).astype(np.float32)
        stack.append(np.clip(img, 0, 255).astype(np.uint8))
    return stack, np.clip(ref, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def incoherent():
    return _incoherent_stack()


# ---------------------------------------------------------------------------
# The dial
# ---------------------------------------------------------------------------

def test_default_is_off_so_the_blend_is_unchanged_until_it_is_asked_for():
    """It costs a second pass over the stack; nobody pays that by accident."""
    assert DEFAULT_COHERENCE_RADIUS == 0
    assert _resolve_coherence_radius(None) == 0


def test_out_of_range_and_nonsense_settings_fall_back_to_the_default():
    for bad in ("nonsense", object(), None):
        assert _resolve_coherence_radius(bad) == DEFAULT_COHERENCE_RADIUS
    assert _resolve_coherence_radius(-5) == 0
    assert _resolve_coherence_radius(7.9) == 7


def test_zero_reproduces_the_unfiltered_blend_exactly(incoherent):
    """0 has to stay the single-pass path, byte for byte, not a small radius."""
    stack, _ = incoherent
    plain = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL)
    off = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                        coherence_radius=0)
    assert np.array_equal(plain, off)


def test_zero_reproduces_the_unfiltered_hard_select_exactly(incoherent):
    """MODE_MAX reads the same dial over its depth field; 0 still has to be off.

    Both of the mode's rendering paths, for the reason the slice tests check
    both: with the smoothing dial off it copies pixels, with it on it gathers a
    tent, and a radius of 0 must leave each exactly where it was.
    """
    stack, _ = incoherent
    for smoothing in (0, 50):
        plain = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                              depth_smoothing=smoothing)
        off = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                            depth_smoothing=smoothing, coherence_radius=0)
        assert np.array_equal(plain, off), f"smoothing {smoothing}"


def test_the_depth_filter_still_renders_from_the_stack(incoherent):
    """A filtered depth is still a depth: no frame invented, none reached past."""
    stack, _ = incoherent
    filtered = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                             coherence_radius=8)
    lo = np.min(np.stack(stack), axis=0)
    hi = np.max(np.stack(stack), axis=0)
    assert np.all(filtered >= lo) and np.all(filtered <= hi)


def test_the_depth_filter_follows_the_picture_rather_than_reaching_over_it():
    """The property that makes it a guided filter and not a blur.

    Two frames, one sharp on the left half and one on the right, so the correct
    depth is a step down the middle of the picture and the picture has an edge in
    exactly that place. A filter of radius 12 - wide enough for an isotropic
    low-pass to smear that step over 24 px - has to leave it standing, because
    the guide it is fitted against steps there too.

    Read as the width of the transition in the *rendered* result rather than of
    the depth map, since the depth map is internal and the render is what the
    claim is about.
    """
    rng = np.random.default_rng(23)
    texture = (cv2.GaussianBlur(rng.random((SIZE, SIZE, 3), dtype=np.float32),
                                (0, 0), 0.8) * 255)
    # Each half is a different tone as well as differently focused, so the seam
    # between the two sources is visible in the output at all.
    texture[:, :SIZE // 2] *= 0.45
    left, right = texture.copy(), texture.copy()
    left[:, SIZE // 2:] = cv2.GaussianBlur(texture, (0, 0), 4.0)[:, SIZE // 2:]
    right[:, :SIZE // 2] = cv2.GaussianBlur(texture, (0, 0), 4.0)[:, :SIZE // 2]
    stack = [np.clip(f, 0, 255).astype(np.uint8) for f in (left, right)]

    fused = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                          depth_smoothing=0, coherence_radius=12)
    # How far the render sits from the nearer of the two sources, column by
    # column: a depth that has been smeared across the step renders a band that
    # matches neither frame.
    fused32 = fused.astype(np.float32)
    away = np.minimum(np.abs(fused32 - stack[0]), np.abs(fused32 - stack[1]))
    per_column = away.mean(axis=(0, 2))
    smeared = int((per_column > 2.0).sum())
    assert smeared <= 8, f"{smeared} columns sit between the two sources"


def test_the_result_stays_inside_the_range_of_the_frames_it_blended(incoherent):
    """A filtered share is still a share: no negative weights, no overshoot.

    The guided filter fits a local linear model rather than constraining one, so
    a share can come back negative - and a negative weight would subtract a
    frame, which would show up as a value no frame in the stack ever held.
    """
    stack, _ = incoherent
    filtered = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                             coherence_radius=8)
    lo = np.min(np.stack(stack), axis=0)
    hi = np.max(np.stack(stack), axis=0)
    assert np.all(filtered >= lo) and np.all(filtered <= hi)


# ---------------------------------------------------------------------------
# The filter itself
# ---------------------------------------------------------------------------

def test_guided_filter_leaves_a_constant_share_untouched():
    """The no-op case, which is what makes the stage free where depth is smooth."""
    rng = np.random.default_rng(1)
    guide = rng.random((64, 64)).astype(np.float32)
    share = np.full((64, 64), 0.25, dtype=np.float32)
    out = _guided_filter(share.copy(), guide, 8, COHERENCE_EPS)
    assert np.allclose(out, 0.25, atol=1e-5)


def test_guided_filter_smooths_a_featureless_guide_but_follows_a_structured_one():
    """Edge-aware, in the one comparison that says so.

    The same noisy share under two guides: one flat, one carrying the share's own
    structure. Under the flat guide the fit has nothing to follow and the result
    is the windowed mean; under the matching guide it keeps the step.
    """
    share = np.zeros((64, 64), dtype=np.float32)
    share[:, 32:] = 1.0
    rng = np.random.default_rng(2)
    noisy = share + rng.normal(0, 0.05, share.shape).astype(np.float32)

    flat_guide = np.full((64, 64), 0.5, dtype=np.float32)
    edge_guide = share.copy()

    blurred = _guided_filter(noisy.copy(), flat_guide, 8, COHERENCE_EPS)
    followed = _guided_filter(noisy.copy(), edge_guide, 8, COHERENCE_EPS)

    def step(plane):
        return float(plane[:, 40:].mean() - plane[:, :24].mean())

    assert step(followed) > 0.95
    assert step(blurred) > 0.95        # a wide step survives either way
    # but only the flat guide smears the transition across the window
    transition = slice(28, 36)
    assert followed[:, transition].std() > 2.0 * blurred[:, transition].std()


# ---------------------------------------------------------------------------
# The frame axis
# ---------------------------------------------------------------------------
# `slice_radius` is testable on a rendered stack in a way `coherence_radius` was
# not, because what it exploits is a property of the *sampling* rather than of
# the scene: when several consecutive frames resolve a pixel equally well, they
# carry the same detail and independent grain, and averaging them is free. A
# stack can be built that way exactly.

# For MODE_AVERAGE the quality claim is still not made on a rendered stack, and
# the reason is sharper than it was for `coherence_radius`. Two things have to be
# true at once before the dial has anything to do there: the blend must be
# concentrating on a single frame, and that frame's neighbours must carry nearly
# - but not exactly - the same detail. Every fixture that can be built here fails
# one of them. A stack whose frames are identical within a focus band measures
# the same energy across it, so the weights are already equal there and the blend
# already averages (measured: 36.7 dB at radius 0 against 34.4 at radius 1, all
# of it boundary spill). A stack with a smooth focus ramp instead has too few
# frames for the exponent to concentrate at all. What produces the regime is a
# deep sweep of real defocus, and that is a capture.
#
# MODE_MAX is the exception, and it is the same fixture that makes it one. The
# first of those two conditions is not something the fixture has to arrange: a
# hard select renders from exactly one frame by definition, whatever the energies
# across the band look like. So the stack that was useless for the average -
# frames identical within a band, differing only in their grain - is precisely
# the one that isolates this dial for the hard select, because there the blend
# the average was already getting for free is the whole of what is being added.

OVERSAMPLE = 8      # frames per depth band, i.e. how far the stack oversamples


def _oversampled_stack(bands=5, seed=11, noise_std=7.0):
    """A stack that samples its depth of field `OVERSAMPLE` times over.

    Enough frames for the ring buffer to wrap several times and for a window to
    sit wholly inside a band, which is what the contract tests below need.
    """
    rng = np.random.default_rng(seed)
    ref = np.full((SIZE, SIZE, 3), 110.0, dtype=np.float32)
    ref[:, SIZE // 2:] = rng.random((SIZE, SIZE // 2, 3)).astype(np.float32) * 150 + 50

    edges = np.linspace(0, SIZE, bands + 1).round().astype(int)
    stack = []
    for band in range(bands):
        sharp = cv2.GaussianBlur(ref, (0, 0), 4.0)
        sharp[edges[band]:edges[band + 1]] = ref[edges[band]:edges[band + 1]]
        for _ in range(OVERSAMPLE):
            noisy = sharp + rng.normal(0.0, noise_std, sharp.shape).astype(np.float32)
            stack.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return stack, np.clip(ref, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def oversampled():
    return _oversampled_stack()


def test_slice_default_is_off():
    """It shares the second pass but not the assumption; nobody pays by accident."""
    assert DEFAULT_SLICE_RADIUS == 0
    assert _resolve_slice_radius(None, 100) == 0


def test_slice_radius_is_clamped_inside_the_stack():
    """A window reaching past both ends is the plain mean, which is not this."""
    assert _resolve_slice_radius(50, 11) == 5
    assert _resolve_slice_radius(3, 100) == 3
    assert _resolve_slice_radius(-2, 100) == 0
    assert _resolve_slice_radius("nonsense", 100) == DEFAULT_SLICE_RADIUS


def test_slice_zero_reproduces_the_unpooled_blend_exactly(oversampled):
    stack, _ = oversampled
    plain = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL)
    off = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                        slice_radius=0)
    assert np.array_equal(plain, off)


def test_slice_zero_reproduces_the_hard_select_exactly(oversampled):
    """The dial that changes MODE_MAX's rendering rule must not change it at 0.

    Both ways round the mode can render: the plain copy, and the tent gather the
    smoothing dial already sends it through. Setting the radius to 0 has to leave
    each of them byte for byte where it was, or every stack rendered before this
    existed silently moved.
    """
    stack, _ = oversampled
    for smoothing in (0, 50):
        plain = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                              depth_smoothing=smoothing)
        off = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                            depth_smoothing=smoothing, slice_radius=0)
        assert np.array_equal(plain, off), f"smoothing {smoothing}"


def _band_scores(img, reference):
    """(grain, detail, error) inside the first focus band of the fixture.

    Read away from the band's edges so what is measured is the rendering rule and
    not a boundary: `grain` is the deviation from the truth over the flat half,
    `detail` the local contrast surviving in the textured half, and `error` how
    far that half sits from the truth - which is the one that sees both at once.
    """
    band = SIZE // 5
    rows = slice(4, band - 4)
    flat = img[rows, :SIZE // 2].astype(np.float32)
    sharp = img[rows, SIZE // 2:].astype(np.float32)
    grey = cv2.cvtColor(img[rows, SIZE // 2:], cv2.COLOR_BGR2GRAY)
    return (
        float((flat - reference[rows, :SIZE // 2]).std()),
        float(np.abs(cv2.Laplacian(grey.astype(np.float32),
                                   cv2.CV_32F, ksize=3)).mean()),
        float(np.sqrt(((sharp - reference[rows, SIZE // 2:]) ** 2).mean())),
    )


def test_the_slice_dial_divides_the_hard_select_s_grain(oversampled):
    """The one thing no other MODE_MAX dial can do, on a stack built to allow it.

    Every frame inside a band of this fixture carries the same picture and its
    own grain, which is what a capture that oversamples its depth of field looks
    like - so a pixel rendered from the band instead of from one frame keeps its
    detail and averages its noise away.

    All three readings are taken, because two of them alone would be satisfied by
    simply softening the picture: the grain has to fall, the textured half's local
    contrast has to survive, and that half has to end up *closer* to the truth
    rather than merely smoother.
    """
    stack, reference = oversampled
    hard = _band_scores(depthmap_impl(list(stack), mode=MODE_MAX,
                                      kernel_size=KERNEL, depth_smoothing=0),
                        reference)
    pooled = _band_scores(depthmap_impl(list(stack), mode=MODE_MAX,
                                        kernel_size=KERNEL, depth_smoothing=0,
                                        slice_radius=1), reference)

    assert pooled[0] < 0.75 * hard[0]       # measured 0.62
    assert pooled[1] > 0.93 * hard[1]       # measured 0.96
    assert pooled[2] < 0.9 * hard[2]        # measured 0.79


def test_the_slice_dial_overshoots_past_the_band_it_was_given(oversampled):
    """Turning it past the sampling costs, and the fixture has to show that too.

    A band here is OVERSAMPLE frames wide with a cliff at each end rather than a
    focus curve that tapers, so its usable radius is smaller than the half-of-the-
    half-maximum rule of thumb would suggest - which makes it a sharp test of the
    failure rather than a calibration of the dial. At radius 4 the window reaches
    outside the band for most pixels and pulls in frames that do not resolve them
    at all: the grain keeps falling and the picture gets further from the truth,
    which is exactly the trade BLEND_WIDTH_FLOOR refuses by default.
    """
    stack, reference = oversampled
    hard = _band_scores(depthmap_impl(list(stack), mode=MODE_MAX,
                                      kernel_size=KERNEL, depth_smoothing=0),
                        reference)
    far = _band_scores(depthmap_impl(list(stack), mode=MODE_MAX,
                                     kernel_size=KERNEL, depth_smoothing=0,
                                     slice_radius=4), reference)

    assert far[0] < hard[0]                 # grain still falling: measured 0.36
    assert far[2] > hard[2]                 # and the detail paying for it: 1.17


def test_the_slice_dial_still_renders_from_the_frames_it_pooled(oversampled):
    """A wider tent is still a blend of source pixels, not an extrapolation."""
    stack, _ = oversampled
    pooled = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                           depth_smoothing=50, slice_radius=4)
    lo = np.min(np.stack(stack), axis=0)
    hi = np.max(np.stack(stack), axis=0)
    assert np.all(pooled >= lo) and np.all(pooled <= hi)


def test_it_pools_across_the_whole_stack_including_both_ends(oversampled):
    """The ring has to cover the first and last frames too, on a shrinking window.

    The head and tail are where a sliding window is got wrong, and getting them
    wrong is invisible in an average - it shows up as the ends of the stack
    quietly contributing less than the middle. Rendered against a stack whose
    first and last bands are the only place their own detail exists, so losing
    either would show as that band going soft.
    """
    stack, reference = oversampled
    pooled = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                           selectivity=100, slice_radius=3)

    def sharpness(img, rows):
        grey = cv2.cvtColor(img[rows, SIZE // 2:], cv2.COLOR_BGR2GRAY)
        return float(np.abs(cv2.Laplacian(grey.astype(np.float32),
                                          cv2.CV_32F, ksize=3)).mean())

    band = SIZE // 5
    first = sharpness(pooled, slice(4, band - 4))
    last = sharpness(pooled, slice(SIZE - band + 4, SIZE - 4))
    middle = sharpness(pooled, slice(2 * band + 4, 3 * band - 4))
    truth = sharpness(reference, slice(2 * band + 4, 3 * band - 4))
    assert first > 0.7 * middle
    assert last > 0.7 * middle
    assert middle > 0.5 * truth


def test_the_two_stages_compose(oversampled):
    """Both on has to cost one extra pass, not two, and still be a blend."""
    stack, _ = oversampled
    both = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                         coherence_radius=8, slice_radius=2)
    lo = np.min(np.stack(stack), axis=0)
    hi = np.max(np.stack(stack), axis=0)
    assert np.all(both >= lo) and np.all(both <= hi)


# ---------------------------------------------------------------------------
# What the dial is for, against a capture nobody generated
# ---------------------------------------------------------------------------
# The claim cannot be made on a rendered stack; see _incoherent_stack. It needs
# a region no frame resolves and every frame gets wrong differently, which is
# what real defocus over dark glossy plastic produces and what a mask-and-blur
# fixture does not. samples/electronics_ant is that stack, and it ships renders
# of itself by another program to be read against - see samples/captures.py for
# why none of them is a ground truth and why the comparison is made over tiles.
#
# The reference is Helicon Focus method A, its own weighted average, at radius
# 30 - the like-for-like counterpart of this mode, and the smoothest of its
# three Method A settings. Two numbers are read off it:
#
#   agreement - correlation of the two detail maps: did we find detail where it
#               found detail. Absolute values here are low (~0.5) because these
#               frames are unregistered on purpose, so the comparison is between
#               candidates and not against 1.0.
#   share     - our tile contrast over its tile contrast, where 1.0 is "as much
#               local contrast recovered". Unfiltered this mode carries half as
#               much again as Helicon at the same nominal job, which is the
#               excess the stage exists to remove.

CAPTURE = "electronics_ant"
REFERENCE = "HF-A-30-1"

# Every 12th frame: 28 of the 333, at full capture resolution, matching
# tests/test_pyramid_electronics_ant.py. The frame count is what is cut and
# never the resolution - a capture is here to say what the lens did at the pixel
# level. The effect is on the same trend at step 8 (42 frames), so the
# subsample is not what produces it.
CAPTURE_STEP = 12
BLOCK = 32


@pytest.fixture(scope="module")
def capture():
    from samples import captures
    if not captures.is_available(CAPTURE):
        pytest.skip(f"samples/{CAPTURE} is not present (it is photographed, "
                    f"not generated)")
    frames, _ = captures.load_capture(CAPTURE, step=CAPTURE_STEP)
    reference = captures.load_render(CAPTURE, REFERENCE)
    if reference is None:
        pytest.skip(f"{REFERENCE} could not be decoded")
    return frames, reference


def _against_reference(fused, reference):
    from samples import captures
    from tests import fusion_metrics as fm
    ours, theirs = captures.fit_render(fused, reference)
    return fm.detail_agreement(ours, theirs, block=BLOCK)


def test_it_brings_the_recovered_contrast_into_line_with_helicons(capture):
    """The headline: the excess local contrast is grain, and it goes.

    Measured at step 12 - share 1.50 unfiltered, 1.16 at radius 8, 1.05 at 16 -
    so the filtered blend recovers what an independent implementation of the
    same idea recovers, where the unfiltered one carries half as much again.
    """
    frames, reference = capture
    _, plain = _against_reference(
        depthmap_impl(list(frames), mode=MODE_AVERAGE, kernel_size=KERNEL),
        reference)
    _, filtered = _against_reference(
        depthmap_impl(list(frames), mode=MODE_AVERAGE, kernel_size=KERNEL,
                      coherence_radius=16),
        reference)
    assert plain > 1.3
    assert abs(filtered - 1.0) < abs(plain - 1.0) / 2.0


def test_slice_pooling_is_not_free_on_a_stack_that_is_not_registered(capture):
    """The caveat that decides when the frame axis can be used at all.

    The spatial filter re-mixes decisions inside one frame and does not care how
    the stack is aligned. Slice pooling averages *different frames* into a pixel,
    so it is only free while they agree on where that pixel is - and these frames
    are the raw capture, which nothing has aligned. On the registered stack the
    same dial takes the reference gap from 0.160 to 0.050 at 111 frames; here it
    collapses the recovered contrast instead, because it is averaging a subject
    that moved.

    Pinned as a guard rather than as a wish: if this ever stops holding it means
    the dial has quietly become insensitive to registration, and the guidance in
    DEFAULT_SLICE_RADIUS would be wrong.
    """
    frames, reference = capture
    _, plain = _against_reference(
        depthmap_impl(list(frames), mode=MODE_AVERAGE, kernel_size=KERNEL,
                      selectivity=100, coherence_radius=16),
        reference)
    _, pooled = _against_reference(
        depthmap_impl(list(frames), mode=MODE_AVERAGE, kernel_size=KERNEL,
                      selectivity=100, coherence_radius=16, slice_radius=2),
        reference)
    # Measured at step 12: 1.16 unpooled against 0.49 at radius 2.
    assert pooled < 0.8 * plain


def test_it_finds_detail_more_nearly_where_helicon_found_it(capture):
    """And it does not buy that by smoothing detail away.

    Recovering less contrast is easy on its own - blurring does it - so the
    agreement has to move the right way at the same time. Measured 0.4905
    unfiltered against 0.5175 at radius 16.
    """
    frames, reference = capture
    plain, _ = _against_reference(
        depthmap_impl(list(frames), mode=MODE_AVERAGE, kernel_size=KERNEL),
        reference)
    filtered, _ = _against_reference(
        depthmap_impl(list(frames), mode=MODE_AVERAGE, kernel_size=KERNEL,
                      coherence_radius=16),
        reference)
    assert filtered > plain + 0.01
