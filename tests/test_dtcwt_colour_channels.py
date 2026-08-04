"""Colour-channel tests for the DTCWT fusion method (dtcwt.py, dtcwt_torch.py).

Item 12 in docs/ALGORITHM_IMPROVEMENTS.md: the three colour channels are
transformed and fused separately, so nothing stops a pixel's channels being
taken from different frames and rendering a colour no frame carried.

The method decides two things, and each has to be settled once for all three
channels:

* **Which frame owns a detail coefficient.** Fixed in 1.14.0 and kept by item
  7's rewrite: `_coefficient_activity` reduces the channels with a maximum, so
  one decision selects all three. That half is pinned by
  `test_dtcwt_frame_order.py::TestFold::test_all_channels_of_a_coefficient_follow_one_decision`.

* **In what proportion the frames are mixed in the coarse band.** Item 16's
  activity-weighted lowpass average (1.11.3) built one weight map per channel,
  which put the defect straight back at large scale: each channel mixed the
  stack in its own proportions, so the coarse band could hold a colour that is
  on no line between the frames it came from. Fixed in 1.30.14 by summing the
  channels into one weight map, and that is what this file is about.

The unit tests pin `_lowpass_weight` itself - one map, blind to which channel
carries the detail. The render test then measures the property that matters on
a stack built so the coarse band is all there is to see over a wide region: the
fused colour there has to lie on the line joining the two frames' colours,
because a shared mix cannot leave it. Both paths run it.

Run with:  python -m pytest tests/test_dtcwt_colour_channels.py -v
"""

import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_registry as reg

dtcwt_module = pytest.importorskip("fusion_methods.dtcwt")
D = dtcwt_module

LEVEL_SHAPE = (8, 8, 6)        # (H, W, orientations) of one highpass level
LOWPASS_SHAPE = (8, 8)


def _pyramids(channels):
    """Three stand-in transforms, one per channel, each with a single level.

    `_lowpass_weight` reads nothing but `.highpasses`, so the real transform is
    not needed to pin what it does with the channels.
    """
    return [SimpleNamespace(highpasses=(coeffs,)) for coeffs in channels]


def _detail(magnitude=1.0, at=(3, 4)):
    """One level's coefficients: zero everywhere but one orientation of one pixel."""
    coeffs = np.zeros(LEVEL_SHAPE, dtype=np.complex64)
    coeffs[at[0], at[1], 0] = magnitude
    return coeffs


def _flat():
    return np.zeros(LEVEL_SHAPE, dtype=np.complex64)


class TestLowpassWeight:
    """One weight per pixel, and it does not care which channel it came from."""

    def test_the_weight_is_one_map_for_all_three_channels(self):
        """
        A weight per channel is the defect: it lets each channel mix the stack
        in its own proportions. The map is per pixel, and the channels are
        already gone by the time it is returned.
        """
        weight = D._lowpass_weight(_pyramids([_detail(), _flat(), _flat()]),
                                   LOWPASS_SHAPE)

        assert weight.shape == LOWPASS_SHAPE
        assert weight.dtype == np.float32

    def test_detail_in_any_one_channel_weighs_the_same(self):
        """
        Which channel carries a structure is a property of the subject, not of
        how much detail the frame has, so it cannot move the weight.
        """
        maps = [D._lowpass_weight(_pyramids(
            [_detail() if ch == carrier else _flat() for ch in range(3)]),
            LOWPASS_SHAPE) for carrier in range(3)]

        np.testing.assert_array_equal(maps[0], maps[1])
        np.testing.assert_array_equal(maps[0], maps[2])
        assert maps[0].max() > 0.0

    def test_permuting_the_channels_leaves_the_weight_alone(self):
        """The same, with all three channels carrying different detail."""
        channels = [_detail(magnitude=3.0), _detail(magnitude=1.0, at=(5, 2)),
                    _detail(magnitude=2.0)]
        weight = D._lowpass_weight(_pyramids(channels), LOWPASS_SHAPE)

        for order in ((2, 0, 1), (1, 2, 0), (2, 1, 0)):
            permuted = D._lowpass_weight(
                _pyramids([channels[i] for i in order]), LOWPASS_SHAPE)
            np.testing.assert_array_equal(permuted, weight)

    def test_every_channel_contributes_its_own_detail(self):
        """
        Blind to *which* channel, not to how many: a frame with detail in all
        three is more active than one carrying the same detail in one, and the
        weight says so.
        """
        one = D._lowpass_weight(_pyramids([_detail(), _flat(), _flat()]),
                                LOWPASS_SHAPE)
        all_three = D._lowpass_weight(_pyramids([_detail()] * 3), LOWPASS_SHAPE)

        assert all_three.sum() == pytest.approx(3.0 * one.sum(), rel=1e-5)

    def test_a_frame_with_no_detail_weighs_nothing(self):
        """The epsilon added at the call site is what keeps the mean defined."""
        weight = D._lowpass_weight(_pyramids([_flat()] * 3), LOWPASS_SHAPE)
        assert not weight.any()


# ---------------------------------------------------------------------------
# Whole renders, on both paths
# ---------------------------------------------------------------------------

PATHS = ["dtcwt", "dtcwt_gpu"]

SIZE = 256
BAND = SIZE // 5          # depth of the textured strip at each edge
MARGIN = 24               # kept clear of the strips, so the middle really is flat

# What "the colour came from somewhere" is allowed to cost. The mix is shared,
# so the only thing that can move a flat pixel off the line between the two
# frames is the faint detail the strips' defocus spreads into it. Per-channel
# weights miss by 2.76 levels on average and 8.09 at worst here.
MEAN_TOLERANCE = 1.0
WORST_TOLERANCE = 2.0


def _chromatic_pair(seed=7, gain=0.78):
    """Two frames whose detail lives in different channels, over a flat field.

    The top strip carries blue texture and the bottom strip red; each frame is
    sharp in one strip and defocused in the other, so the two channels'
    aggregate activities disagree about which frame to favour - which is the
    disagreement a per-channel weight acts on. The middle of the frame is a
    smooth colour ramp with no detail in it at all, so nothing there is taken
    from either frame's detail bands and what is rendered is the coarse band
    alone. Frame 1 is also 22% darker, which is what makes the frames disagree
    at coarse scale in the first place - item 16's own failure case.

    Returns the stack and the mask of the flat middle.
    """
    rng = np.random.default_rng(seed)
    scene = np.zeros((SIZE, SIZE, 3), np.float32)
    scene[:, :, 0] = 70 + 100 * np.linspace(0, 1, SIZE, np.float32)[None, :]
    scene[:, :, 1] = 100 + 40 * np.linspace(0, 1, SIZE, np.float32)[:, None]
    scene[:, :, 2] = 190 - 90 * np.linspace(0, 1, SIZE, np.float32)[:, None]

    for _ in range(500):
        y, x = int(rng.integers(0, BAND)), int(rng.integers(0, SIZE))
        cv2.circle(scene, (x, y), 1, (245.0, 100.0, 30.0), -1)
    for _ in range(500):
        y, x = int(rng.integers(SIZE - BAND, SIZE)), int(rng.integers(0, SIZE))
        cv2.circle(scene, (x, y), 1, (30.0, 100.0, 245.0), -1)
    scene = np.clip(scene, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(scene, (15, 15), 0)
    near, far = scene.copy(), scene.copy()
    near[BAND:] = blurred[BAND:]
    far[:SIZE - BAND] = blurred[:SIZE - BAND]
    far = np.clip(far.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    flat = np.zeros((SIZE, SIZE), bool)
    flat[BAND + MARGIN:SIZE - BAND - MARGIN] = True
    return [near, far], flat


def _off_segment(stack, fused, region):
    """Distance, in levels, from each fused colour to the two frames' own line.

    Mixing two colours in one proportion traces the straight line between them
    in RGB. Mixing each channel in its own proportion leaves that line, and how
    far it leaves it is how much colour the fusion invented.
    """
    a = stack[0].astype(np.float32)[region]
    b = stack[1].astype(np.float32)[region]
    f = fused.astype(np.float32)[region]
    ab = b - a
    t = np.clip(((f - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9),
                0.0, 1.0)
    return np.linalg.norm(f - (a + t[..., None] * ab), axis=-1)


def _fuse(path_key):
    method = reg.get(path_key)
    ok, why = method.available()
    if not ok:
        pytest.skip(f"{method.label}: {why}")
    return method.run


@pytest.mark.parametrize("path_key", PATHS)
def test_the_coarse_band_mixes_every_channel_alike(path_key):
    """
    Where there is no detail to select, the picture is the coarse band, and a
    coarse band mixed in one shared proportion cannot hold a colour that is not
    between the frames it was mixed from.
    """
    stack, flat = _chromatic_pair()
    distance = _off_segment(stack, _fuse(path_key)(stack), flat)

    assert distance.mean() < MEAN_TOLERANCE, (
        f"{path_key}: the flat field averages {distance.mean():.2f} levels off "
        f"the colours its two frames carry")
    assert distance.max() < WORST_TOLERANCE, (
        f"{path_key}: worst flat pixel is {distance.max():.2f} levels off the "
        f"colours its two frames carry")
