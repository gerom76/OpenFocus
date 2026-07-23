"""The scale / focus-breathing registration contract.

Focus stacking changes the optical magnification slightly from frame to frame as
the focus plane moves - "focus breathing". The 'scale' registration method models
that as a similarity (uniform scale + rotation + translation) and warps every
frame back onto the first, so the bar it has to clear is:

1. It is exposed as a first-class method (`TestRegistry`).
2. It removes the magnification drift a breathing stack has, bringing the first
   and last frames back into register (`TestBreathingRemoved`).
3. Every corrected frame comes out at one common size - the shared valid region -
   like the homography and ECC methods (`TestCommonCrop`).
4. It leaves already-aligned and degenerate inputs alone (`TestNoOp`).

Run with:  python -m pytest tests/test_registration_scale.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registration import ImageRegistration, register_images
from tests.synthetic_stack import make_reference


def _base_scene(size=512, seed=3):
    """A feature-rich BGR frame SIFT can lock onto reliably."""
    img = make_reference(height=size, width=size, seed=seed, style="texture")
    # A few solid blobs guarantee strong, well-separated keypoints for RANSAC.
    for (cy, cx) in [(120, 120), (120, 392), (392, 120), (392, 392), (256, 256)]:
        cv2.circle(img, (cx, cy), 20, (30, 220, 30), -1)
    return img


def _breathing_stack(size=512, n=7, max_zoom=1.06, seed=3):
    """A stack whose magnification grows uniformly from frame 0 to frame n-1."""
    base = _base_scene(size=size, seed=seed)
    center = (size / 2.0, size / 2.0)
    stack = []
    for s in np.linspace(1.0, max_zoom, n):
        M = cv2.getRotationMatrix2D(center, 0, float(s))
        frame = cv2.warpAffine(base, M, (size, size),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        stack.append(frame)
    return stack


def _center_crop(img, margin=90):
    return img[margin:-margin, margin:-margin]


def _mean_abs_diff(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(ga - gb)))


class TestRegistry:
    def test_scale_is_a_supported_method(self):
        assert "scale" in ImageRegistration.SUPPORTED_METHODS

    def test_convenience_function_accepts_scale(self):
        stack = _breathing_stack(n=4)
        out = register_images(stack, method="scale")
        assert len(out) == len(stack)


class TestBreathingRemoved:
    def test_first_and_last_frames_come_back_into_register(self):
        stack = _breathing_stack()

        # The breathing is real: uncorrected, the frames disagree at the edges.
        before = _mean_abs_diff(_center_crop(stack[0]), _center_crop(stack[-1]))
        assert before > 5.0, "synthetic stack should start with visible breathing"

        aligned = ImageRegistration(method="scale", downscale_width=512).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)

        after = _mean_abs_diff(_center_crop(aligned[0]), _center_crop(aligned[-1]))

        # Correcting the magnification should cut the first/last disagreement hard.
        assert after < before * 0.35, f"scale correction did not converge (before={before:.2f}, after={after:.2f})"

    def test_every_frame_tracks_the_reference_after_correction(self):
        stack = _breathing_stack(n=6)
        aligned = ImageRegistration(method="scale", downscale_width=512).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)

        ref = _center_crop(aligned[0])
        # No frame should drift far from the reference once breathing is removed.
        worst = max(_mean_abs_diff(ref, _center_crop(a)) for a in aligned[1:])
        assert worst < 6.0, f"a corrected frame still drifts from the reference (worst={worst:.2f})"


class TestCommonCrop:
    def test_all_outputs_share_one_size(self):
        stack = _breathing_stack()
        aligned = ImageRegistration(method="scale", downscale_width=512).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)

        sizes = {a.shape for a in aligned}
        assert len(sizes) == 1, f"expected one common output size, got {sizes}"

    def test_common_crop_is_no_larger_than_the_input(self):
        stack = _breathing_stack()
        h, w = stack[0].shape[:2]
        aligned = ImageRegistration(method="scale", downscale_width=512).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)
        ah, aw = aligned[0].shape[:2]
        assert ah <= h and aw <= w


class TestNoOp:
    def test_single_image_stack_is_returned_untouched(self):
        one = [_base_scene()]
        out = ImageRegistration(method="scale").process(one, output_path=None, thread_count=2)
        assert len(out) == 1
        assert out[0].shape == one[0].shape

    def test_identical_frames_survive_intact(self):
        base = _base_scene()
        stack = [base.copy() for _ in range(5)]
        aligned = ImageRegistration(method="scale", downscale_width=512).process(
            stack, output_path=None, thread_count=2)

        # Nothing to correct: frames stay a common size and near-identical content.
        assert len({a.shape for a in aligned}) == 1
        drift = _mean_abs_diff(_center_crop(aligned[0]), _center_crop(aligned[-1]))
        assert drift < 1.0, f"identity input should not be distorted (drift={drift:.2f})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
