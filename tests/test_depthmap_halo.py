"""
Halo-suppression tests for the Depth Map fusion method (depthmap.py).

The stack is built the way a camera actually records an occlusion: the
background-focused slice carries the foreground's defocused glow spilling over
sharp background texture. That veiled texture still out-measures the other
slice's defocused background, so the plain argmax drags the glow into the
result - the classic focus-stacking halo. The halo reads as a positive
luminance bias in the spill zone, and `halo_radius` must remove it.

Thresholds are loose regression guards, not a leaderboard.

Run with:  python -m pytest tests/test_depthmap_halo.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.depthmap import depthmap_impl, MODE_MAX, MODE_AVERAGE

BLUR = 31   # strong defocus, so the glow reaches well past the pooling window
KERNEL = 9  # the method's default analysis window


def _occlusion_stack(size=320, seed=5, noise_std=20.0):
    """Bright textured disc over a dark textured background, two slices.

    Slice A focuses the disc (background defocused); slice Z focuses the
    background, with the disc's premultiplied blur composited on top so its
    glow contaminates the background near the silhouette.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.2)
    noise = noise_std * noise / max(noise.std(), 1e-6)
    bg = np.stack([np.full((size, size), 70, np.float32) + noise,
                   np.full((size, size), 80, np.float32) + noise,
                   np.full((size, size), 75, np.float32) + noise], axis=2)
    bg = np.clip(bg, 0, 255)

    fg = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size), np.float32)
    centre = (size // 2, size // 2)
    cv2.circle(alpha, centre, size // 4, 1.0, -1)
    cv2.circle(fg, centre, size // 4, (235, 240, 245), -1)
    for r in range(10, size // 4, 8):
        cv2.circle(fg, centre, r, (40, 60, 160), 1)

    a3 = alpha[:, :, None]
    reference = fg * a3 + bg * (1.0 - a3)

    bg_blur = cv2.GaussianBlur(bg, (BLUR, BLUR), 0)
    slice_a = fg * a3 + bg_blur * (1.0 - a3)

    fg_premult = cv2.GaussianBlur(fg * a3, (BLUR, BLUR), 0)
    alpha_blur = cv2.GaussianBlur(alpha, (BLUR, BLUR), 0)[:, :, None]
    slice_z = fg_premult + bg * (1.0 - alpha_blur)

    def to8(x):
        return np.clip(x, 0, 255).astype(np.uint8)

    return [to8(slice_a), to8(slice_z)], to8(reference), alpha


def _spill_zone(alpha, inner=4, outer=12):
    """Background pixels inside the glow ring, past the pooling window's reach."""
    dist = cv2.distanceTransform((alpha < 0.5).astype(np.uint8),
                                 cv2.DIST_L2, 5)
    return (dist > inner) & (dist <= outer)


def _zone_bias(fused, reference, zone):
    """Mean luminance deviation in the zone; the halo is a positive bias."""
    diff = (cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY).astype(np.float64)
            - cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float64))
    return float(diff[zone].mean())


@pytest.fixture(scope="module")
def scene():
    stack, reference, alpha = _occlusion_stack()
    return stack, reference, _spill_zone(alpha)


# --------------------------------------------------------------------------
# The halo exists, and the radius removes it
# --------------------------------------------------------------------------

def test_plain_argmax_halos(scene):
    """Without suppression the glow must be visible, or this file tests nothing."""
    stack, reference, zone = scene
    fused = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL)
    assert _zone_bias(fused, reference, zone) > 3.0


def test_radius_suppresses_the_halo(scene):
    stack, reference, zone = scene
    fused = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                          halo_radius=8)
    assert abs(_zone_bias(fused, reference, zone)) < 1.0


def test_average_mode_also_improves(scene):
    """The blend weights spread too, so MODE_AVERAGE must halo less with it."""
    stack, reference, zone = scene
    plain = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL)
    treated = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                            halo_radius=8)
    assert (_zone_bias(treated, reference, zone)
            < _zone_bias(plain, reference, zone))


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

def test_zero_radius_is_the_old_behaviour(scene):
    """halo_radius=0 (and None) must be bit-identical to not passing it."""
    stack, _, _ = scene
    base = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL)
    zero = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                         halo_radius=0)
    none = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                         halo_radius=None)
    assert np.array_equal(base, zero)
    assert np.array_equal(base, none)


@pytest.mark.parametrize("bad", ["wide", -3, 2.7])
def test_invalid_radius_never_crashes(scene, bad):
    """Junk values fall back to off (negative clamps, floats truncate)."""
    stack, _, _ = scene
    out = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                        halo_radius=bad)
    assert out.shape == stack[0].shape


def test_output_shape_and_dtype(scene):
    stack, _, _ = scene
    fused = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                          halo_radius=6)
    assert fused.shape == stack[0].shape
    assert fused.dtype == np.uint8


# --------------------------------------------------------------------------
# The radius the UI is willing to offer
# --------------------------------------------------------------------------
# The dial's cost scales with the radius and nothing else: it fills a band
# around every contour in the frame with defocused pixels, and the focus measure
# cannot tell a contour that needed it from one that did not. On the reference
# ant stack at kernel 5 that moved 8.5% of the frame at r=2 and 29.9% at r=15.
# So the slider is bounded by the pooling window, which is the only thing that
# says how far a claim can be justified - see constants.halo_radius_ceiling.
# The engine stays permissive; only the UI is bounded.

def test_ceiling_follows_the_pooling_window():
    from constants import HALO_RADIUS_MAX, halo_radius_ceiling
    assert halo_radius_ceiling(5) == 5
    assert halo_radius_ceiling(9) == 9
    # Never past the slider's own ceiling, however wide the window gets.
    assert halo_radius_ceiling(51) == HALO_RADIUS_MAX
    # A window that does not pool at all can justify no claim.
    assert halo_radius_ceiling(1) == 1
    assert halo_radius_ceiling(0) == 0


def test_the_engine_is_not_bounded_by_it():
    """A scripted caller with a genuinely wide glow must still be able to ask.

    This file's own fixture is exactly that case: blurred by 31 px, and it takes
    r=8 at k=9 to cover the spill - a pair the UI ceiling happens to allow, but
    the point is that the engine never consults it.
    """
    from fusion_methods.depthmap import _resolve_halo_radius
    assert _resolve_halo_radius(30) == 30


# The slider wiring that applies this ceiling - `OpenFocus._set_halo_range`,
# called from the kernel handler and from update_slider_availability - is not
# covered here. Constructing the real main window in-process takes Qt down hard
# enough at interpreter exit to swallow pytest's own summary and everything
# scheduled after it, and a window is the only thing that exercises the wiring.
# What is testable without one is the rule it applies, above.
