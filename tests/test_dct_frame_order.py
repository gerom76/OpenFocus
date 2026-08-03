"""
Frame-order tests for the DCT fusion method (dct.py, dct_torch.py).

DCT reads a stack as a focus sweep: it estimates where along that sweep each
block comes into focus, so unlike the other methods its answer is a position in
the input order and depends on it. Item 4 in docs/ALGORITHM_IMPROVEMENTS.md is
about how much of that dependence is the model and how much was a defect.

The defect was taking the plane as the mean over *every* frame that cleared the
in-focus bar. Two frames at opposite ends of a stack both clearing it - one
because it resolves the block, one on grain - put the plane halfway between
them, on a frame that resolves nothing. Reading the in-focus set as a run
containing the peak removes that, and the tests below are in two groups:

* `TestFocalPlane` drives `_focal_plane` directly with hand-built energies, one
  per situation it has to get right. These are the reason the end-to-end
  numbers move.

* The order tests measure whole renders. Reversing a stack is still a focus
  sweep, so it has to give back exactly the same picture - that one is an
  equality, not a threshold. Shuffling it is not a sweep, so agreement there is
  a floor that the old rule fell through rather than a property.

Run with:  python -m pytest tests/test_dct_frame_order.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods import dct as D
from fusion_methods.dct import dct_focus_stack_fusion
from tests import fusion_scenarios as sc
from tests.test_dct_defocus_wash import _veil_stack


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def _plane(profile, plateau=D._PLATEAU):
    """The focal plane for a single block with this per-frame energy profile."""
    energies = [np.array([[float(v)]], np.float32) for v in profile]
    threshold = np.max(np.stack(energies), axis=0) * plateau
    field = D._focal_plane(lambda i: energies[i], len(energies), threshold)
    return float(field[0, 0])


class TestFocalPlane:
    """The plane a block is given, situation by situation."""

    def test_a_sharp_run_gives_its_middle(self):
        """Frames 3-5 in focus, the rest not: the plane is frame 4."""
        assert _plane([1, 1, 1, 9, 10, 9, 1, 1, 1]) == 4.0

    def test_an_even_run_gives_a_half_frame(self):
        """Sub-frame accuracy is the point of carrying halves through."""
        assert _plane([1, 1, 10, 10, 1, 1]) == 2.5

    def test_a_distant_noise_spike_does_not_move_the_plane(self):
        """
        The defect this file exists for.

        Frame 40 clears the bar on grain alone, far from where the block is
        actually in focus. Averaging the whole in-focus set puts the plane at
        frame 15 - blurred in every sense - while the run containing the peak
        stays where the detail is.
        """
        profile = [1] * 41
        profile[3], profile[4], profile[5] = 9, 10, 9
        profile[40] = 8.1                     # clears 0.8 * 10 on noise
        assert _plane(profile) == 4.0

    def test_a_dip_inside_a_plateau_does_not_split_the_run(self):
        """Grain drops one frame below the bar; the plateau is still one run."""
        assert _plane([1, 10, 10, 7.5, 10, 10, 1]) == 3.0

    def test_a_real_gap_does_split_the_run(self):
        """
        Two surfaces at different depths both resolve this block. The plane has
        to pick one - the peak's, frames 1-2, so 1.5 - not sit in the defocused
        middle, which is where the mean over all four in-focus frames lands
        (frame 4, resolving nothing).
        """
        assert _plane([1, 9, 10, 1, 1, 1, 8.5, 8.5, 1]) == 1.5

    def test_nothing_in_focus_gives_mid_stack(self):
        """
        Every frame equally poor: the run is the whole stack and the plane is
        its middle, which is the same answer for every such block and so leaves
        a defocused region uniform instead of patchwork (item 18).
        """
        assert _plane([5.0] * 9) == 4.0

    def test_a_flat_stack_with_grain_still_gives_mid_stack(self):
        """The same, with the few percent of grain that made runs gappy."""
        rng = np.random.default_rng(4)
        profile = 5.0 + rng.normal(0, 0.12, 41)
        assert 15.0 < _plane(list(profile)) < 25.0


# ---------------------------------------------------------------------------
# Reversing a stack is still a sweep, so it must render identically
# ---------------------------------------------------------------------------

ORDER_STACKS = ["depth_edge", "long_stack", "deep_stack"]


@pytest.mark.parametrize("scenario", ORDER_STACKS)
def test_reversing_the_stack_renders_the_same_picture(scenario):
    """
    Front-to-back and back-to-front are the same sweep, so this is an equality.

    It holds because the plane is carried as halves of a frame all the way to
    compositing. Rounding it to a whole frame first cannot keep it: reversal
    maps k + 0.5 to (n - 1 - k) - 0.5, and any rule for rounding a half sends
    those two to frames that are not each other's mirror.
    """
    stack, _, _ = sc.build(scenario)
    forward = dct_focus_stack_fusion(list(stack))
    backward = dct_focus_stack_fusion(list(stack)[::-1])
    worst = int(np.abs(forward.astype(np.int32) - backward.astype(np.int32)).max())
    assert worst == 0, f"reversing the stack moved a pixel by {worst} levels"


# ---------------------------------------------------------------------------
# Shuffling one is not a sweep, so agreement is a floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario,floor", [("long_stack", 29.0), ("deep_stack", 32.0)])
def test_shuffling_the_stack_stays_close(scenario, floor):
    """
    A shuffled stack has no sweep to read, so the plane cannot mean what it
    normally does and the picture is not required to be identical. What it must
    not do is fall apart: before the run rule these scored 25.4 and 25.1 dB.
    """
    stack, _, _ = sc.build(scenario)
    base = dct_focus_stack_fusion(list(stack))
    rng = np.random.default_rng(0)
    for _ in range(3):
        order = rng.permutation(len(stack))
        other = dct_focus_stack_fusion([stack[i] for i in order])
        score = _psnr(base, other)
        assert score > floor, (
            f"{scenario} rendered {score:.1f} dB apart under a shuffle")


def test_a_veil_at_the_far_end_does_not_travel(veiled=None):
    """
    The stacks most at risk are the ones whose ends hold no detail, because
    that is where a frame can clear the in-focus bar on grain alone. Moving the
    veil frames to the front must not change what the body is rendered from.
    """
    stack, reference, interior = _veil_stack()
    front = dct_focus_stack_fusion(stack[16:] + stack[:16])
    normal = dct_focus_stack_fusion(list(stack))
    from tests.test_dct_defocus_wash import _veil_patch_share
    assert _veil_patch_share(front, reference, interior) < 5.0
    assert _psnr(front, normal) > 24.0
