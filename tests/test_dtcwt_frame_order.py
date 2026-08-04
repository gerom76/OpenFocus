"""
Frame-order tests for the DTCWT fusion method (dtcwt.py, dtcwt_torch.py).

Item 7 in docs/ALGORITHM_IMPROVEMENTS.md: the method used to fold a stack
together two frames at a time, so `(((1,2),3),4)` decided the picture and the
early frames passed through more rounds of the consistency filter than the late
ones. The answer therefore depended on the order the stack was handed over -
46 dB of agreement across orderings, on a method whose input is a set of frames
with no meaning attached to their order at all.

The frames are now compared against each other in one pass: each coefficient
goes to the frame carrying the most activity around it. Three groups of tests:

* `TestActivity` and `TestFold` drive the two helpers the pass is built from,
  `_coefficient_activity` and `_keep_stronger`, with hand-built arrays. They
  are where the rule itself is pinned down, including what happens on a tie -
  which is the only thing left that could make the order matter.

* The render tests measure whole pictures. A reordered stack is the same stack,
  so those are equalities rather than tolerances wherever the fixture allows
  one; `long_stack` and `depth_edge` carry coefficients that tie in both the
  regional activity and their own magnitude, so they get a floor instead. The
  floor is one the pairwise rule fell well through.

* Both paths run every render test, since the GPU twin mirrors the CPU rule and
  the two have to land together.

Run with:  python -m pytest tests/test_dtcwt_frame_order.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_registry as reg
from tests import fusion_scenarios as sc
from tests.synthetic_stack import make_stack

dtcwt_module = pytest.importorskip("fusion_methods.dtcwt")
D = dtcwt_module

SHAPE = (9, 11, 6, 3)          # (H, W, orientations, channels), as one level


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def _spike(magnitude=1.0, at=(4, 5), orientation=0, channel=0):
    """Coefficients that are zero everywhere but one."""
    coeffs = np.zeros(SHAPE, dtype=np.complex64)
    coeffs[at[0], at[1], orientation, channel] = magnitude
    return coeffs


class TestActivity:
    """What a frame's claim on a coefficient is measured as."""

    def test_the_score_is_the_activity_pooled_over_the_window(self):
        """
        A lone coefficient raises the maximum filter over its 3x3 window, and
        pooling that over the same window is what makes the comparison
        regional: the frame is credited over a 5x5 patch, most strongly at the
        coefficient itself.
        """
        score, _ = D._coefficient_activity(_spike())

        assert score[4, 5, 0] == pytest.approx(1.0)      # window filled by the max
        assert score[4, 3, 0] == pytest.approx(1 / 3)    # a third of it in reach
        assert score[4, 2, 0] == pytest.approx(0.0)      # outside both windows
        assert score[4, 5, 1] == pytest.approx(0.0)      # orientations stay apart

    def test_the_strongest_channel_speaks_for_the_coefficient(self):
        """
        Detail living in one colour channel scores at full strength rather than
        being diluted by the two flat ones, and the map it produces is what
        selects all three channels together (item 12).
        """
        blue_only = D._coefficient_activity(_spike(channel=2))[0]
        red_only = D._coefficient_activity(_spike(channel=0))[0]

        assert blue_only[4, 5, 0] == pytest.approx(1.0)
        np.testing.assert_array_equal(blue_only, red_only)

    def test_a_frame_with_nothing_in_it_scores_nothing(self):
        score, magnitude = D._coefficient_activity(np.zeros(SHAPE, np.complex64))
        assert not score.any() and not magnitude.any()


class TestFold:
    """The running winner: one frame at a time, but every frame judged alike."""

    @staticmethod
    def _fold(frames, scores, magnitudes):
        """Run the module's own fold over hand-built arrays."""
        fused = frames[0].copy()
        best_score, best_magnitude = scores[0].copy(), magnitudes[0].copy()
        for coeffs, score, magnitude in zip(frames[1:], scores[1:], magnitudes[1:]):
            D._keep_stronger(fused, best_score, best_magnitude, coeffs,
                             score, magnitude)
        return fused

    @staticmethod
    def _random_stack(seed, count=5, tie_scores=False):
        """Frames whose (score, magnitude) pairs order them without ambiguity."""
        rng = np.random.default_rng(seed)
        frames, scores, magnitudes = [], [], []
        for index in range(count):
            frames.append((rng.standard_normal(SHAPE)
                           + 1j * rng.standard_normal(SHAPE)).astype(np.complex64))
            if tie_scores:
                # Deliberately coarse, so frames tie on the score and the
                # coefficient's own magnitude has to settle it - which is the
                # situation a strong shared structure produces in a real stack.
                scores.append(rng.integers(0, 3, SHAPE[:3]).astype(np.float32))
                magnitudes.append(np.full(SHAPE[:3], index + 1, np.float32))
            else:
                scores.append(rng.random(SHAPE[:3]).astype(np.float32))
                magnitudes.append(rng.random(SHAPE[:3]).astype(np.float32))
        return frames, scores, magnitudes

    def test_each_coefficient_comes_from_the_frame_with_the_most_activity(self):
        """
        The whole of item 7: the winner is the best of the stack, not the
        survivor of a chain of pairwise fusions.
        """
        frames, scores, magnitudes = self._random_stack(seed=1)
        fused = self._fold(frames, scores, magnitudes)

        winner = np.stack(scores).argmax(axis=0)
        expected = np.take_along_axis(np.stack(frames), winner[None, ..., None],
                                      axis=0)[0]
        np.testing.assert_array_equal(fused, expected)

    def test_all_channels_of_a_coefficient_follow_one_decision(self):
        """A pixel's colour cannot split across source frames (item 12)."""
        frames, scores, magnitudes = self._random_stack(seed=2)
        fused = self._fold(frames, scores, magnitudes)

        stack = np.stack(frames)
        for channel in range(1, SHAPE[-1]):
            first = (stack[:, :, :, :, 0] == fused[None, :, :, :, 0])
            this = (stack[:, :, :, :, channel] == fused[None, :, :, :, channel])
            np.testing.assert_array_equal(first, this)

    def test_a_tie_goes_to_the_stronger_coefficient_not_the_earlier_frame(self):
        """
        Scores do tie - a strong structure two frames share fills the maximum
        filter around it - and settling that on arrival order is exactly the
        order dependence this file exists to remove.
        """
        held = _spike(magnitude=1.0)
        arriving = _spike(magnitude=2.0)
        score = np.ones(SHAPE[:3], np.float32)

        fused = held.copy()
        D._keep_stronger(fused, score.copy(), np.ones(SHAPE[:3], np.float32),
                         arriving, score, np.full(SHAPE[:3], 2.0, np.float32))
        np.testing.assert_array_equal(fused, arriving)

        fused = arriving.copy()
        D._keep_stronger(fused, score.copy(), np.full(SHAPE[:3], 2.0, np.float32),
                         held, score, np.ones(SHAPE[:3], np.float32))
        np.testing.assert_array_equal(fused, arriving)

    def test_a_better_scoring_frame_wins_even_with_a_weaker_coefficient(self):
        """The regional activity decides; the magnitude only breaks its ties."""
        held = _spike(magnitude=5.0)
        arriving = _spike(magnitude=1.0)

        fused = held.copy()
        D._keep_stronger(fused, np.ones(SHAPE[:3], np.float32),
                         np.full(SHAPE[:3], 5.0, np.float32), arriving,
                         np.full(SHAPE[:3], 2.0, np.float32),
                         np.ones(SHAPE[:3], np.float32))
        np.testing.assert_array_equal(fused, arriving)

    @pytest.mark.parametrize("tie_scores", [False, True])
    def test_the_fold_does_not_depend_on_the_order_it_runs_in(self, tie_scores):
        frames, scores, magnitudes = self._random_stack(seed=3, tie_scores=tie_scores)
        expected = self._fold(frames, scores, magnitudes)

        rng = np.random.default_rng(9)
        for _ in range(6):
            order = rng.permutation(len(frames))
            fused = self._fold([frames[i] for i in order],
                               [scores[i] for i in order],
                               [magnitudes[i] for i in order])
            np.testing.assert_array_equal(fused, expected)


# ---------------------------------------------------------------------------
# Whole renders, on both paths
# ---------------------------------------------------------------------------

PATHS = ["dtcwt", "dtcwt_gpu"]

# Fixtures whose coefficients never tie in both the activity and the magnitude
# at once, so a reordered stack has to come back byte for byte.
EXACT = ["fine_texture", "sensor_noise", "low_contrast"]

# saturated_colour does tie - 571 coefficients across its four levels are won on
# a tie in both the activity and the magnitude, by frames that differ only in
# phase, which is the residual item 7 in docs/ALGORITHM_IMPROVEMENTS.md records
# and declines to break with an arbitrary rule about complex numbers. It
# rendered byte for byte regardless until item 12 moved its coarse band by 1e-7
# in 1.30.14, and one pixel of its 102,400 turned out to be sitting on exactly
# 21.5. So the assertion here is the one the fixture can actually carry: the
# tie is allowed to round a pixel either way, and nothing else may move.
TIED = "saturated_colour"
TIED_PIXELS = 4        # measured: 1 on the CPU path, 0 on the GPU path
TIED_LEVELS = 1

# The rest tie somewhere and land within a level of themselves. The pairwise
# rule managed 42-53 dB on most of these fixtures, so the floor is a long way
# below what the fix achieves and a long way above what it replaced.
FLOOR_DB = 60.0


@pytest.fixture(scope="module")
def photographic():
    stack, _, _ = make_stack(num_slices=5, height=192, width=192, seed=4,
                             style="photographic")
    return stack


def _fuse(path_key):
    method = reg.get(path_key)
    ok, why = method.available()
    if not ok:
        pytest.skip(f"{method.label}: {why}")
    return method.run


@pytest.mark.parametrize("path_key", PATHS)
def test_a_reversed_stack_renders_the_identical_picture(path_key, photographic):
    """Back to front is the same set of frames, so it is the same picture."""
    fuse = _fuse(path_key)
    fused = fuse(photographic)
    np.testing.assert_array_equal(fuse(list(photographic)[::-1]), fused)


@pytest.mark.parametrize("path_key", PATHS)
@pytest.mark.parametrize("scenario", EXACT)
def test_reordering_a_scenario_renders_the_identical_picture(path_key, scenario):
    fuse = _fuse(path_key)
    stack, _, _ = sc.build(scenario)
    fused = fuse(stack)

    np.testing.assert_array_equal(fuse(list(stack)[::-1]), fused)
    order = np.random.default_rng(5).permutation(len(stack))
    np.testing.assert_array_equal(fuse([stack[i] for i in order]), fused)


@pytest.mark.parametrize("path_key", PATHS)
def test_a_tied_scenario_moves_at_most_a_pixel(path_key):
    """
    Where two frames are tied on every measure the method has, which of them a
    coefficient comes from is decided by phase alone, and the picture may round
    a pixel either way. That is the whole of what a reordering is allowed to
    change - not a tolerance on the frame, a count of pixels.
    """
    fuse = _fuse(path_key)
    stack, _, _ = sc.build(TIED)
    fused = fuse(stack)

    orderings = [list(stack)[::-1]]
    for seed in (5, 11, 12):
        order = np.random.default_rng(seed).permutation(len(stack))
        orderings.append([stack[i] for i in order])

    for reordered in orderings:
        moved = np.abs(fuse(reordered).astype(np.int32) - fused.astype(np.int32))
        assert moved.max() <= TIED_LEVELS, (
            f"{TIED} on {path_key}: reordering moved a pixel by "
            f"{int(moved.max())} levels")
        assert (moved > 0).sum() <= TIED_PIXELS, (
            f"{TIED} on {path_key}: reordering moved {int((moved > 0).sum())} "
            f"pixels, which is more than the tie can account for")


@pytest.mark.parametrize("path_key", PATHS)
@pytest.mark.parametrize("scenario", [key for key, _, _, _ in sc.SCENARIOS])
def test_every_scenario_agrees_with_itself_across_orderings(path_key, scenario):
    fuse = _fuse(path_key)
    stack, _, _ = sc.build(scenario)
    fused = fuse(stack)

    for seed in (11, 12):
        order = np.random.default_rng(seed).permutation(len(stack))
        agreement = _psnr(fuse([stack[i] for i in order]), fused)
        assert agreement > FLOOR_DB, (
            f"{scenario} on {path_key}: reordering the stack moved the picture "
            f"by more than a level ({agreement:.1f} dB)")
