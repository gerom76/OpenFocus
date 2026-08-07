"""
Selectivity tests for the Depth Map (Average) fusion method (depthmap.py).

The mode weights every frame by its focus energy and adds the results up. With
a linear weight that only selects while the stack is short: a defocused frame
still measures a fraction of the peak, so a stack of N frames adds that fraction
up N times until it buries the one frame that actually resolved the pixel. Past
a few dozen frames the blend is the arithmetic mean of the whole stack, and the
mean of many defocus kernels is one enormous defocus kernel - a veiled,
low-contrast, black-lifting result that got worse the deeper the stack went.

`selectivity` raises the weight to a power, which is what makes the blend
scale-free in stack depth. These tests pin the two halves of that bargain: it
must recover the detail on a deep stack, and it must not give up the averaging
in the regions that genuinely have nothing to choose between - which is what
the mode exists for in the first place.

Thresholds are loose regression guards, not a leaderboard.

Run with:  python -m pytest tests/test_depthmap_selectivity.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.depthmap import (
    DEFAULT_SELECTIVITY,
    MODE_AVERAGE,
    SELECTIVITY_EXPONENT_MAX,
    _powered,
    _selectivity_exponent,
    depthmap_impl,
)

SIZE = 192
KERNEL = 9


def _deep_stack(slices=48, seed=5, noise_std=8.0):
    """One sharp frame per depth band, the rest of the stack defocused.

    Deliberately deep: this is the regime the linear weighting fails in, and a
    three-frame fixture would not show it at all. The left third is flat grey
    that no frame ever resolves - the region the mode is supposed to average -
    and the rest is fine texture that exactly one frame resolves.

    The noise is well above what uint8 quantisation contributes on purpose. At a
    couple of levels the averaged result lands on the quantisation floor and the
    flat-region measurements below stop saying anything about the weighting.
    """
    rng = np.random.default_rng(seed)
    ref = np.full((SIZE, SIZE, 3), 90.0, dtype=np.float32)
    texture = rng.random((SIZE, SIZE - SIZE // 3, 3)).astype(np.float32) * 150 + 50
    ref[:, SIZE // 3:] = texture

    bands = np.linspace(0, SIZE, slices + 1).round().astype(int)
    stack = []
    for k in range(slices):
        # Every frame is defocused everywhere except its own horizontal band.
        img = cv2.GaussianBlur(ref, (0, 0), 4.0)
        img[bands[k]:bands[k + 1]] = ref[bands[k]:bands[k + 1]]
        img = img + rng.normal(0.0, noise_std, img.shape).astype(np.float32)
        stack.append(np.clip(img, 0, 255).astype(np.uint8))
    return stack, np.clip(ref, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def deep():
    return _deep_stack()


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def _flat_noise(img):
    """Noise std of the region no frame resolves: what averaging is bought for."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat = grey[:, :SIZE // 3]
    return float((flat - cv2.medianBlur(flat, 5)).std())


# ---------------------------------------------------------------------------
# The exponent helpers
# ---------------------------------------------------------------------------

def test_dial_spans_linear_to_the_documented_maximum():
    assert _selectivity_exponent(0) == 1.0
    assert _selectivity_exponent(100) == SELECTIVITY_EXPONENT_MAX
    # Monotone, so the dial never reverses on the user part way along.
    values = [_selectivity_exponent(s) for s in range(0, 101)]
    assert values == sorted(values)


def test_dial_only_ever_asks_for_half_steps():
    """_powered takes the exponent by squaring, which needs multiples of 0.5."""
    for strength in range(0, 101):
        exponent = _selectivity_exponent(strength)
        assert exponent * 2 == int(exponent * 2), exponent


def test_powered_matches_pow_for_every_exponent_the_dial_produces():
    rng = np.random.default_rng(3)
    for exponent in sorted({_selectivity_exponent(s) for s in range(0, 101)}):
        values = rng.random((32, 48)).astype(np.float32) * 3.0 + 0.05
        got = _powered(values.copy(), exponent)
        want = np.power(values, exponent)
        assert np.allclose(got, want, rtol=2e-5, atol=1e-6), exponent


def test_powered_consumes_its_input_but_not_the_frame_it_came_from():
    """It squares in place, so the caller must use the value returned."""
    values = np.full((8, 8), 2.0, dtype=np.float32)
    result = _powered(values, 3.0)
    assert np.allclose(result, 8.0)
    assert result is not values


# ---------------------------------------------------------------------------
# What the dial is for
# ---------------------------------------------------------------------------

def test_default_beats_the_linear_weighting_on_a_deep_stack(deep):
    stack, reference = deep
    linear = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                           selectivity=0)
    default = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL)
    assert _psnr(default, reference) > _psnr(linear, reference) + 3.0


def test_selectivity_recovers_the_contrast_the_mean_flattened(deep):
    """The haze is lost local contrast; the dial must put it back."""
    stack, reference = deep

    def detail(img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3)).mean())

    truth = detail(reference)
    linear = detail(depthmap_impl(list(stack), mode=MODE_AVERAGE,
                                  kernel_size=KERNEL, selectivity=0))
    default = detail(depthmap_impl(list(stack), mode=MODE_AVERAGE,
                                   kernel_size=KERNEL))
    # Measured: the linear blend keeps 19% of the reference's own detail on this
    # 48-frame fixture - it has all but stopped selecting - and the default 67%.
    assert linear < 0.35 * truth
    assert default > 0.55 * truth


def test_more_selectivity_never_costs_detail(deep):
    """Monotone in the thing the dial is sold on, so the slider reads honestly."""
    stack, _ = deep

    def detail(strength):
        img = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                            selectivity=strength)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3)).mean())

    scores = [detail(s) for s in (0, 25, 50, 75, 100)]
    assert scores == sorted(scores)


def test_the_region_no_frame_resolves_still_averages(deep):
    """The half of the bargain the mode exists for: multi-frame noise reduction.

    Where every frame measures the same the weights come out equal whatever the
    exponent is, so the blend is still the plain mean there. The dial erodes
    that a little - the energies are not exactly equal, they are noise - but
    the flat region must stay far quieter than any single frame.
    """
    from fusion_methods.depthmap import MODE_MAX
    stack, _ = deep
    one_frame = _flat_noise(stack[len(stack) // 2])
    default = _flat_noise(depthmap_impl(list(stack), mode=MODE_AVERAGE,
                                        kernel_size=KERNEL))
    assert default < one_frame / 3.0

    # And still far quieter there than the hard select, which takes the region
    # from whichever single frame's noise happened to measure highest. That gap
    # is the whole reason to reach for the average rather than Max.
    hard = _flat_noise(depthmap_impl(list(stack), mode=MODE_MAX,
                                     kernel_size=KERNEL))
    assert default < hard / 2.0


def test_zero_is_the_linear_weighting_the_mode_always_used(deep):
    """0 has to stay a real setting, not merely a small exponent."""
    stack, _ = deep
    out = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                        selectivity=0)

    # Rebuilt from the definition: sum(w*I)/sum(w) with w linear in the energy.
    from fusion_methods.depthmap import _BASELINE_FRACTION, _focus_energy
    from utils import bitdepth
    frames = [bitdepth.to_float01(f) for f in stack]
    energies = [_focus_energy(f.copy(), KERNEL) for f in frames]
    baseline = _BASELINE_FRACTION * float(np.mean([e.mean() for e in energies]))
    num = sum(f * (e + baseline)[:, :, None] for f, e in zip(frames, energies))
    den = sum(e + baseline for e in energies)
    want = bitdepth.from_float01(num / den[:, :, None], out.dtype)

    # The baseline is estimated from probe frames rather than the whole stack,
    # so the two agree to within that estimate rather than exactly - and the
    # baseline is a soft floor, which is why an estimate is enough.
    assert _psnr(out, want) > 45.0


def test_the_dial_does_nothing_to_the_hard_select(deep):
    """MODE_MAX takes its pixel from one frame whatever the weights look like."""
    from fusion_methods.depthmap import MODE_MAX
    stack, _ = deep
    low = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                        selectivity=0)
    high = depthmap_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                         selectivity=100)
    assert np.array_equal(low, high)


def test_out_of_range_and_nonsense_settings_fall_back_to_the_default(deep):
    stack, _ = deep
    default = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                            selectivity=DEFAULT_SELECTIVITY)
    for bad in (None, "nonsense", object()):
        assert np.array_equal(
            depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                          selectivity=bad),
            default)
    # A number outside 0-100 clamps rather than throwing.
    assert np.array_equal(
        depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                      selectivity=500),
        depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                      selectivity=100))
