"""ECC registers a single-channel stack instead of crashing on it.

Item 9 of docs/REGISTRATION_IMPROVEMENTS.md. The two feature stages hand their
downscaled frame straight to SIFT, which takes one channel or three, so a
grayscale stack has always registered under 'scale' and 'homography'. ECC
converted unconditionally with `COLOR_BGR2GRAY`, so the same stack died in
preprocessing with "Bad number of channels" before a single pair was measured -
a stage that is unreachable in that state from the GUI, whose loader normalises
to BGR, but reachable from the CLI and from library use.

The bar here:

1. `_to_gray` answers for every channel count a loader produces, and refuses
   the ones it does not, rather than letting cv2 raise from three frames deeper
   (`TestToGray`).
2. It is bit-identical to the `COLOR_BGR2GRAY` call it replaces on BGR input,
   which is what keeps the colour path - i.e. every in-app render - exactly
   where it was (`TestToGray`).
3. Every stage and every composed pipeline registers a single-channel stack,
   and *registers* it: the drift is removed rather than the crash merely being
   swallowed (`TestGrayscaleStackRegisters`).
4. The grayscale result is the result the same content gives as BGR, to the
   pixel, so grayscale is not a second-class path (`TestGrayscaleMatchesColour`).

Reinstating the unconditional conversion fails 3 and 4 outright.

Run with:  python -m pytest tests/test_registration_grayscale.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registration import ImageRegistration, _to_gray
from tests.synthetic_stack import make_reference


SIZE = 384
STAGES = ["scale", "homography", "ecc", "both", "scale+homography+ecc"]


# --- shared synthetic input (mirrors the other registration tests) ---

def _base_scene(size=SIZE, seed=5):
    """A feature-rich BGR frame both SIFT and ECC can lock onto."""
    img = make_reference(height=size, width=size, seed=seed, style="texture")
    for (cy, cx) in [(96, 96), (96, 288), (288, 96), (288, 288), (192, 192)]:
        cv2.circle(img, (cx, cy), 16, (30, 220, 30), -1)
    return img


def _drifting_stack(frame, n=6, step=3.0):
    """A stack of `frame` that translates by a known, constant amount per frame."""
    h, w = frame.shape[:2]
    return [
        cv2.warpAffine(frame, np.float32([[1, 0, i * step], [0, 1, i * step * 0.6]]),
                       (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        for i in range(n)
    ]


def _gray_scene():
    """The scene as a single-channel frame."""
    return cv2.cvtColor(_base_scene(), cv2.COLOR_BGR2GRAY)


def _residuals(images):
    """RMS difference of each frame against the first - misalignment, in levels."""
    ref = images[0].astype(np.float64)
    return [float(np.sqrt(np.mean((img.astype(np.float64) - ref) ** 2)))
            for img in images[1:]]


class TestToGray:
    """The helper answers for what a loader produces and refuses what it does not."""

    def test_two_dimensional_input_is_returned_unchanged(self):
        gray = _gray_scene()
        out = _to_gray(gray)
        assert out.ndim == 2
        assert np.array_equal(out, gray)

    def test_trailing_length_one_axis_is_dropped(self):
        gray = _gray_scene()
        out = _to_gray(gray[:, :, np.newaxis])
        assert out.ndim == 2
        assert np.array_equal(out, gray)

    def test_bgr_conversion_is_the_call_it_replaces(self):
        """The colour path must be bit-identical to the unconditional cvtColor."""
        bgr = _base_scene()
        assert np.array_equal(_to_gray(bgr), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

    def test_bgra_input_drops_the_alpha(self):
        bgr = _base_scene()
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        assert np.array_equal(_to_gray(bgra), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

    def test_sixteen_bit_input_keeps_its_depth(self):
        """The conversion is not where the stack is narrowed to 8 bits."""
        gray16 = (_gray_scene().astype(np.uint16) * 257)
        bgr16 = cv2.merge([gray16, gray16, gray16])
        assert _to_gray(bgr16).dtype == np.uint16

    @pytest.mark.parametrize("channels", [2, 5])
    def test_an_impossible_channel_count_is_refused_here(self, channels):
        img = np.zeros((16, 16, channels), dtype=np.uint8)
        with pytest.raises(ValueError, match="grayscale"):
            _to_gray(img)


class TestGrayscaleStackRegisters:
    """Every stage takes a single-channel stack, and removes its drift."""

    @pytest.mark.parametrize("method", STAGES)
    def test_stage_does_not_crash_on_single_channel_input(self, method):
        stack = _drifting_stack(_gray_scene())
        aligned = ImageRegistration(method=method).process(stack)
        assert len(aligned) == len(stack)
        assert all(img.ndim == 2 for img in aligned)

    @pytest.mark.parametrize("method", STAGES)
    def test_stage_actually_aligns_a_single_channel_stack(self, method):
        """Not crashing is half of it; the drift has to come out too."""
        stack = _drifting_stack(_gray_scene())
        before = _residuals(stack)
        after = _residuals(ImageRegistration(method=method).process(stack))
        assert max(after) < 0.35 * max(before), (
            f"{method} left {max(after):.1f} of {max(before):.1f} levels of drift"
        )

    def test_a_frame_with_a_trailing_length_one_axis_registers_too(self):
        """(h, w, 1) is what some loaders hand back for a grayscale file."""
        stack = [img[:, :, np.newaxis] for img in _drifting_stack(_gray_scene())]
        aligned = ImageRegistration(method="ecc").process(stack)
        assert len(aligned) == len(stack)

    def test_a_four_channel_stack_registers_and_keeps_its_alpha(self):
        bgra = cv2.cvtColor(_base_scene(), cv2.COLOR_BGR2BGRA)
        aligned = ImageRegistration(method="ecc").process(_drifting_stack(bgra))
        assert all(img.shape[2] == 4 for img in aligned)


class TestGrayscaleMatchesColour:
    """A grayscale stack registers to the same place its BGR twin does."""

    @pytest.mark.parametrize("method", ["ecc", "both"])
    def test_the_measured_alignment_does_not_depend_on_the_channel_count(self, method):
        gray = _gray_scene()
        # The same content as three identical channels: BGR2GRAY inverts this
        # exactly, so ECC is being handed the same pixels either way and any
        # difference in the result is the channel count alone.
        colour = cv2.merge([gray, gray, gray])

        aligned_gray = ImageRegistration(method=method).process(_drifting_stack(gray))
        aligned_colour = ImageRegistration(method=method).process(_drifting_stack(colour))

        assert aligned_gray[0].shape == aligned_colour[0].shape[:2], (
            "the crop differs, so the two stacks were not registered the same way"
        )
        for g, c in zip(aligned_gray, aligned_colour):
            assert np.array_equal(g, cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
