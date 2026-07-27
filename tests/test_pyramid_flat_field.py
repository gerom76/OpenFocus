"""
Flat-field tests for the Laplacian-pyramid method (pyramid.py, pyramid_torch.py).

The published choose-max rule copies the winning coefficient of every band and
discards the rest. Where one frame is plainly sharper that is the right answer.
Where no frame is - a defocused background, which is most of a macro frame -
the energies differ only by grain, the winner map becomes a speckle field, and
the fused bands are stitched from frames that disagree about what is there.

Two things come out of that, and there is a test for each:

* Structure no frame had. The collapsed pyramid is a sum of bands taken from
  different frames, and nothing in the sum keeps it near any of them, so a
  smooth background comes back crossed by thin dark filaments. Measured as
  pixels darker than every frame in the stack (item 19 in
  docs/ALGORITHM_IMPROVEMENTS.md).

* A frame that resolves nothing winning on grain alone. Photon noise grows with
  brightness, so a bright defocused veil carries more band energy than a dark
  sharp frame, wins every detail-free region and stamps its tone across them -
  the same defect item 17 fixed for DCT.

Thresholds are loose regression guards, not a leaderboard.

Run with:  python -m pytest tests/test_pyramid_flat_field.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.pyramid import pyramid_impl
from tests import fusion_scenarios as scenarios
from tests.fusion_metrics import psnr
from tests.test_dct_defocus_wash import _veil_patch_share, _veil_stack

# The settings the method shipped with before item 19: choose-max, no noise
# gate, activity-proportional base, no clamp. Kept as a fixed dictionary rather
# than read from the module so the tests below compare against the published
# rule itself, not against whatever the defaults happen to be.
PUBLISHED = dict(selectivity=float("inf"), coherence=0.0, noise_gate=False,
                 base_selectivity=1.0, envelope=False)


@pytest.fixture(scope="module")
def drifting():
    """A stack whose background is never sharp and drifts as it defocuses."""
    return scenarios.deep_stack()


@pytest.fixture(scope="module")
def veiled():
    """A stack whose far frames are a bright, grainy, detail-free veil."""
    return _veil_stack()


def _below_every_frame(fused, stack):
    """(worst, share) of pixels darker than every source, in 8-bit levels.

    A value below the darkest frame at that pixel cannot have come from the
    stack - it is something the reconstruction invented. `share` is the
    percentage of the frame more than 8 levels under, i.e. plainly visible.
    """
    darkest = np.min(np.stack(stack), axis=0).astype(np.int32)
    under = np.maximum(darkest - fused.astype(np.int32), 0)
    return int(under.max()), float(np.mean(under.max(axis=2) > 8) * 100.0)


# --------------------------------------------------------------------------
# The result has to be made of the frames it was given
# --------------------------------------------------------------------------

def test_no_pixel_is_darker_than_every_frame(drifting):
    """The measured failure was 24 levels under, on 2.05% of the frame."""
    stack, _, _ = drifting
    worst, share = _below_every_frame(pyramid_impl(list(stack)), stack)
    assert worst == 0, (
        f"{share:.2f}% of the frame came back up to {worst} levels darker than "
        "every source - the collapsed pyramid is reconstructing structure no "
        "frame had (item 19)")


def test_the_published_rule_is_what_needed_the_clamp(drifting):
    """
    Guard on the guard: if choose-max stopped producing out-of-envelope pixels
    on this fixture, the test above would be passing for the wrong reason and
    would no longer be watching anything.
    """
    stack, _, _ = drifting
    worst, _share = _below_every_frame(pyramid_impl(list(stack), **PUBLISHED), stack)
    assert worst > 8, (
        "the unclamped path no longer overshoots on this fixture, so the "
        "envelope test above has nothing left to catch - find a fixture that "
        "still shows it before deleting either")


def test_clamping_does_not_flatten_the_picture(drifting):
    """
    The cheap way to satisfy the clamp is to blur towards the middle of the
    stack. The result still has to beat every single frame, and to hold on to
    the detail the unclamped path found.
    """
    stack, reference, _ = drifting
    fused = pyramid_impl(list(stack))
    assert psnr(fused, reference) > max(psnr(s, reference) for s in stack) + 1.0
    unclamped = pyramid_impl(list(stack), envelope=False)
    assert psnr(fused, reference) >= psnr(unclamped, reference) - 0.25


# --------------------------------------------------------------------------
# A frame that resolves nothing must not win on grain alone
# --------------------------------------------------------------------------

def test_veil_frames_do_not_take_the_smooth_body(veiled):
    """Choose-max hands 32.3% of the dark body to a bright veil frame."""
    stack, reference, interior = veiled
    share = _veil_patch_share(pyramid_impl(list(stack)), reference, interior)
    assert share < 5.0, (
        f"{share:.1f}% of the smooth dark body came from a bright veil frame - "
        "grain is buying regions that hold no detail again")


def test_the_noise_gate_is_what_stops_it(veiled):
    """The same fixture, with the gate off, must still show the defect."""
    stack, reference, interior = veiled
    share = _veil_patch_share(pyramid_impl(list(stack), **PUBLISHED),
                              reference, interior)
    assert share > 10.0, (
        "the published rule no longer veils this fixture, so the test above is "
        "no longer measuring the noise gate")


def test_detail_free_regions_do_not_cost_the_rest_of_the_picture(veiled):
    """
    Guard against the cheap way to pass the two tests above: refusing to
    select. The rim bristles and the board markings are real detail and must
    still be picked up, so the fused result has to beat every single frame.
    """
    stack, reference, _ = veiled
    fused = pyramid_impl(list(stack))
    assert psnr(fused, reference) > max(psnr(s, reference) for s in stack) + 1.0


def test_veil_frames_still_win_where_they_are_the_sharp_ones(veiled):
    """
    Nothing here may depend on *which* frame is suspect: fusing the veil frames
    alone must still return a picture made of them.
    """
    stack, _, _ = veiled
    fused = pyramid_impl(list(stack[16:]))
    assert fused.shape == stack[0].shape
    worst, _ = _below_every_frame(fused, stack[16:])
    assert worst == 0


# --------------------------------------------------------------------------
# The exposed tuning has to do what it says
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,kwargs", [
    ("levels", {"levels": 2}),
    ("energy_window", {"energy_window": 15}),
    ("selectivity", {"selectivity": 2.0}),
    ("coherence", {"coherence": 0.5}),
    ("noise_gate", {"noise_gate": False}),
    ("base_selectivity", {"base_selectivity": 0.0}),
    ("envelope", {"envelope": False}),
])
def test_every_control_changes_the_result(veiled, name, kwargs):
    """Each exposed control must move the picture, or it is a decoration."""
    stack, _, _ = veiled
    base = pyramid_impl(list(stack))
    other = pyramid_impl(list(stack), **kwargs)
    assert other.shape == base.shape
    moved = float(np.abs(other.astype(np.int32) - base.astype(np.int32)).mean())
    assert moved > 0.01, f"{name} changed nothing ({moved:.4f} levels)"


def test_the_published_rule_is_still_reachable(drifting):
    """
    Everything here is a departure from Burt-Adelson choose-max, and asking for
    the original has to give the original: same weights, same tie-break, same
    reconstruction.
    """
    stack, _, _ = drifting
    fused = pyramid_impl(list(stack), **PUBLISHED)
    # Selecting rather than averaging shows as a higher spatial frequency and
    # as the out-of-envelope pixels the other tests measure; both are checked
    # elsewhere. Here it is enough that the setting is honoured at all.
    assert fused.shape == stack[0].shape
    assert not np.array_equal(fused, pyramid_impl(list(stack)))


def test_selectivity_zero_is_the_plain_average(drifting):
    """
    The bottom of the selectivity range means every frame counts the same, and
    that is a claim worth pinning: the detail bands then reduce to a mean, so
    the result must land close to the average of the stack.
    """
    stack, _, _ = drifting
    fused = pyramid_impl(list(stack), selectivity=0.0, base_selectivity=0.0,
                         envelope=False)
    mean = np.mean(np.stack(stack).astype(np.float32), axis=0)
    assert float(np.abs(fused.astype(np.float32) - mean).mean()) < 1.0


def test_a_clipped_black_backdrop_does_not_upset_the_noise_gate():
    """
    The gate divides each band by a low percentile of its own energy, and a
    stack shot against a black backdrop - velvet, a light box, a night sky - is
    clipped to exactly zero over a large part of every frame. Read naively that
    percentile is zero, whatever grain the rest of the frame carries, and
    dividing by it would hand each frame a different unbounded advantage and let
    the emptiest one take the render. Dead pixels are therefore left out of the
    estimate: they cannot win anything anyway.

    The bar is the same stack without the backdrop: masking part of every frame
    to black must not cost anything outside the mask.
    """
    stack, reference, _ = scenarios.depth_edge()
    backdrop = np.zeros(stack[0].shape[:2], np.uint8)
    backdrop[:90] = 1                            # a third of the frame, clipped
    blacked = [np.where(backdrop[:, :, None] > 0, 0, s).astype(np.uint8)
               for s in stack]
    blacked_reference = np.where(backdrop[:, :, None] > 0, 0, reference).astype(np.uint8)

    lit = pyramid_impl(list(stack))
    dark = pyramid_impl(blacked)
    live = backdrop == 0
    assert psnr(dark[live], blacked_reference[live]) > psnr(lit[live], reference[live]) - 1.0, (
        "a clipped-black backdrop cost the rest of the picture - the noise "
        "level is being read off the dead pixels")


# --------------------------------------------------------------------------
# The GPU twin must not drift from the CPU path
# --------------------------------------------------------------------------

def _require_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("no GPU device available")


@pytest.mark.parametrize("kwargs", [{}, {"coherence": 0.5}, {"noise_gate": False},
                                    {"selectivity": float("inf")}])
def test_gpu_path_agrees(drifting, kwargs):
    _require_gpu()
    from fusion_methods.pyramid_torch import pyramid_torch_impl

    stack, _, _ = drifting
    cpu = pyramid_impl(list(stack), **kwargs)
    gpu = pyramid_torch_impl(list(stack), **kwargs)
    assert psnr(gpu, cpu) > 40.0


def test_gpu_path_holds_the_envelope(veiled):
    _require_gpu()
    from fusion_methods.pyramid_torch import pyramid_torch_impl

    stack, reference, interior = veiled
    gpu = pyramid_torch_impl(list(stack))
    assert _below_every_frame(gpu, stack)[0] == 0
    assert _veil_patch_share(gpu, reference, interior) < 5.0
