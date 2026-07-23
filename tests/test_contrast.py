"""The post-fusion contrast enhancement contract.

Contrast is an optional output-time step, so the bar it has to clear is:

1. Off / strength 0 is exactly the identity - the stored result is never
   altered, so the control can be toggled freely (`TestIdentity`).
2. It actually raises contrast, and more strength raises it more
   (`TestEffect`).
3. It is depth-preserving and uses the full range at 16-bit, like every other
   stage (`TestDepth`).
4. It is colour-safe: hue and saturation ratios are preserved, only brightness
   and contrast move (`TestColourSafety`).

Run with:  python -m pytest tests/test_contrast.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import contrast
from tests.synthetic_stack import make_reference
from utils import bitdepth

METHODS = (contrast.METHOD_AUTO, contrast.METHOD_CLAHE)


def _pale_image(dtype=np.uint8):
    """A deliberately flat, bright image - the case contrast exists to fix."""
    ref = make_reference(height=128, width=192, seed=7, style="photographic").astype(np.float32) / 255.0
    # Compress into a narrow bright band, like the reported crane-fly render.
    pale = 0.55 + (ref - ref.mean()) * 0.25
    return bitdepth.from_float01(np.clip(pale, 0, 1), dtype)


def _std(img):
    return cv2.cvtColor(bitdepth.to_display8(img), cv2.COLOR_BGR2GRAY).std()


class TestIdentity:
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    def test_strength_zero_is_byte_identical(self, method, dtype):
        img = _pale_image(dtype)
        out = contrast.apply_contrast(img, method, 0.0)
        assert np.array_equal(out, img)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    def test_method_off_is_byte_identical(self, dtype):
        img = _pale_image(dtype)
        assert np.array_equal(contrast.apply_contrast(img, contrast.METHOD_OFF, 1.0), img)

    def test_none_passthrough(self):
        assert contrast.apply_contrast(None, contrast.METHOD_AUTO, 1.0) is None

    def test_does_not_mutate_input(self):
        img = _pale_image(np.uint16)
        before = img.copy()
        contrast.apply_contrast(img, contrast.METHOD_AUTO, 1.0)
        assert np.array_equal(img, before)

    def test_grayscale_is_passed_through(self):
        gray = np.full((16, 16), 128, np.uint8)
        assert np.array_equal(contrast.apply_contrast(gray, contrast.METHOD_AUTO, 1.0), gray)


class TestEffect:
    @pytest.mark.parametrize("method", METHODS)
    def test_raises_contrast(self, method):
        img = _pale_image(np.uint8)
        base = _std(img)
        assert _std(contrast.apply_contrast(img, method, 1.0)) > base + 2.0

    @pytest.mark.parametrize("method", METHODS)
    def test_strength_is_monotonic(self, method):
        img = _pale_image(np.uint8)
        stds = [_std(contrast.apply_contrast(img, method, s)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
        # Non-decreasing, and the endpoints genuinely differ.
        assert all(b >= a - 0.5 for a, b in zip(stds, stds[1:]))
        assert stds[-1] > stds[0] + 1.0


class TestDepth:
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    def test_preserves_dtype(self, method, dtype):
        out = contrast.apply_contrast(_pale_image(dtype), method, 0.7)
        assert out.dtype == dtype

    @pytest.mark.parametrize("method", METHODS)
    def test_16bit_uses_full_range(self, method):
        """The result must be real 16-bit data, not 8-bit values in a wide type."""
        out = contrast.apply_contrast(_pale_image(np.uint16), method, 0.8)
        assert out.max() > 255

    @pytest.mark.parametrize("method", METHODS)
    def test_8bit_and_16bit_agree(self, method):
        """Same picture at either depth, within rounding."""
        img8 = _pale_image(np.uint8)
        img16 = bitdepth.convert(img8, np.uint16)
        out8 = contrast.apply_contrast(img8, method, 0.7)
        out16 = bitdepth.convert(contrast.apply_contrast(img16, method, 0.7), np.uint8)
        assert np.abs(out16.astype(np.int16) - out8.astype(np.int16)).mean() < 1.5


class TestColourSafety:
    def test_auto_preserves_channel_ratios(self):
        """Auto changes brightness via a luminance gain, so hue is untouched.

        Checked where no channel clips (clipping is the one place ratios can
        legitimately move).
        """
        img = _pale_image(np.uint8).astype(np.float32)
        out = contrast.apply_contrast(_pale_image(np.uint8), contrast.METHOD_AUTO, 0.8).astype(np.float32)
        unclipped = (out.max(axis=2) < 254) & (img.min(axis=2) > 2)
        rin = img[..., 2] / np.maximum(img[..., 0], 1e-3)
        rout = out[..., 2] / np.maximum(out[..., 0], 1e-3)
        assert np.abs(rout[unclipped] - rin[unclipped]).mean() < 0.02


class TestDescribe:
    def test_tag_for_each_method(self):
        assert contrast.describe(contrast.METHOD_AUTO, 0.6) == "ContrastAuto60"
        assert contrast.describe(contrast.METHOD_CLAHE, 1.0) == "ContrastCLAHE100"

    def test_empty_when_off(self):
        assert contrast.describe(contrast.METHOD_OFF, 1.0) == ""
        assert contrast.describe(contrast.METHOD_AUTO, 0.0) == ""
