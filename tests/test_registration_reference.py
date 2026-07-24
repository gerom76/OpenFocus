"""The reference-frame contract for image-sequence registration.

Registration chains each frame to its neighbour, so error grows with distance
from whichever frame is held fixed. Historically that anchor was always frame 0.
The reference-frame option lets the caller anchor elsewhere - the middle frame -
so the longest chain is halved and residual drift is spread symmetrically.

The bar the feature has to clear:

1. The mode -> index mapping is well defined and defensive (`TestResolveIndex`).
2. Re-referencing is exact: a no-op at frame 0, it collapses the chosen frame to
   identity and rewrites every other transform relative to it (`TestRereference`).
3. End to end, anchoring on the middle frame still warps the whole stack into one
   common, fully-registered frame (`TestReferenceIntegration`).

Run with:  python -m pytest tests/test_registration_reference.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registration import (
    ImageRegistration,
    resolve_reference_index,
    _rereference_transforms,
)
from tests.synthetic_stack import make_reference


# --- shared synthetic stack (mirrors test_registration_scale.py) ---

def _base_scene(size=512, seed=3):
    """A feature-rich BGR frame SIFT can lock onto reliably."""
    img = make_reference(height=size, width=size, seed=seed, style="texture")
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


def _mean_abs_diff(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(ga - gb)))


def _translation(dx, dy):
    T = np.eye(3, dtype=np.float32)
    T[0, 2] = dx
    T[1, 2] = dy
    return T


class TestResolveIndex:
    def test_first_mode_is_frame_zero(self):
        assert resolve_reference_index("first", 7) == 0

    def test_unknown_mode_falls_back_to_first(self):
        assert resolve_reference_index("nonsense", 7) == 0
        assert resolve_reference_index(None, 7) == 0

    @pytest.mark.parametrize("n,expected", [(2, 1), (3, 1), (6, 3), (7, 3), (8, 4)])
    def test_middle_mode_is_the_centre_frame(self, n, expected):
        assert resolve_reference_index("middle", n) == expected

    @pytest.mark.parametrize("n,expected", [(2, 1), (3, 2), (7, 6)])
    def test_last_mode_is_the_final_frame(self, n, expected):
        assert resolve_reference_index("last", n) == expected

    def test_degenerate_counts_are_safe(self):
        assert resolve_reference_index("middle", 0) == 0
        assert resolve_reference_index("middle", 1) == 0
        assert resolve_reference_index("last", 0) == 0
        assert resolve_reference_index("last", 1) == 0


class TestRereference:
    def test_frame_zero_is_an_identity_no_op(self):
        # H[0] is always identity, so re-referencing to 0 must change nothing.
        chain = [_translation(0, 0), _translation(3, 0), _translation(6, 0)]
        out = _rereference_transforms([m.copy() for m in chain], 0)
        for original, result in zip(chain, out):
            assert np.allclose(original, result)

    def test_reference_frame_collapses_to_identity(self):
        chain = [_translation(i * 3.0, i * -2.0) for i in range(5)]
        out = _rereference_transforms([m.copy() for m in chain], 2)
        assert np.allclose(out[2], np.eye(3))

    def test_every_transform_is_rewritten_relative_to_the_reference(self):
        # Forward maps frame i -> frame 0; after re-referencing to k they must
        # read frame i -> frame k, i.e. inv(H[k]) @ H[i].
        chain = [_translation(i * 3.0, i * -2.0) for i in range(5)]
        k = 2
        out = _rereference_transforms([m.copy() for m in chain], k)
        ref_inv = np.linalg.inv(chain[k])
        for i, result in enumerate(out):
            assert np.allclose(result, ref_inv @ chain[i])
        # Pure translations: frame i ends at offset (i-k) * step.
        assert np.allclose(out[0], _translation(-6.0, 4.0))
        assert np.allclose(out[4], _translation(6.0, -4.0))

    def test_out_of_range_index_is_a_no_op(self):
        chain = [_translation(0, 0), _translation(3, 0)]
        out = _rereference_transforms([m.copy() for m in chain], 9)
        for original, result in zip(chain, out):
            assert np.allclose(original, result)

    def test_singular_reference_falls_back_without_raising(self):
        singular = np.zeros((3, 3), dtype=np.float32)
        chain = [np.eye(3, dtype=np.float32), singular]
        out = _rereference_transforms(chain, 1)
        # Returned unchanged rather than raising on the un-invertible matrix.
        assert out is chain


class TestReferenceIntegration:
    def test_middle_reference_registers_the_whole_stack(self):
        stack = _breathing_stack(n=7)
        mid = len(stack) // 2

        aligned = ImageRegistration(method="scale", downscale_width=512,
                                    reference_index=mid).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)

        # One common valid region, exactly like the frame-0 path.
        assert len({a.shape for a in aligned}) == 1

        # With the anchor in the middle, both ends must register onto it.
        worst = max(_mean_abs_diff(aligned[mid], a)
                    for i, a in enumerate(aligned) if i != mid)
        assert worst < 8.0, f"a frame still drifts from the middle reference (worst={worst:.2f})"

    def test_last_reference_registers_the_whole_stack(self):
        stack = _breathing_stack(n=7)
        last = len(stack) - 1

        aligned = ImageRegistration(method="scale", downscale_width=512,
                                    reference_index=last).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)

        assert len({a.shape for a in aligned}) == 1
        worst = max(_mean_abs_diff(aligned[last], a) for a in aligned[:last])
        assert worst < 8.0, f"a frame still drifts from the last reference (worst={worst:.2f})"

    def test_first_and_middle_reference_both_produce_valid_registration(self):
        stack = _breathing_stack(n=6)
        mid = len(stack) // 2

        aligned_first = ImageRegistration(method="scale", downscale_width=512,
                                          reference_index=0).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)
        aligned_mid = ImageRegistration(method="scale", downscale_width=512,
                                        reference_index=mid).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)

        # Each anchoring leaves its own reference frame the best-registered one.
        worst_first = max(_mean_abs_diff(aligned_first[0], a) for a in aligned_first[1:])
        worst_mid = max(_mean_abs_diff(aligned_mid[mid], a)
                        for i, a in enumerate(aligned_mid) if i != mid)
        assert worst_first < 8.0
        assert worst_mid < 8.0

    def test_reference_index_zero_matches_the_default(self):
        # An explicit reference_index of 0 must reproduce the historical output.
        stack = _breathing_stack(n=5)
        default = ImageRegistration(method="scale", downscale_width=512).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)
        explicit = ImageRegistration(method="scale", downscale_width=512,
                                     reference_index=0).process(
            [f.copy() for f in stack], output_path=None, thread_count=2)
        for a, b in zip(default, explicit):
            assert a.shape == b.shape
            assert _mean_abs_diff(a, b) < 0.01


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
