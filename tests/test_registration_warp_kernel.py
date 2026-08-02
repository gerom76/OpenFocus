"""Registration resamples with Lanczos4, on one path, on every machine.

This file used to guard a second warp implementation: a CuPy one that sampled
with `map_coordinates`. It is the guard two defects were caught by and the third
was measured against, so what it asserts now is what is left standing after all
three.

* Item 1 was that the CuPy warp applied the *inverse* of the transform ECC had
  just measured, doubling the misalignment it was asked to remove.
* Item 3 was that the reference frame's transform came out an integer
  translation - a copy - so one frame escaped resampling entirely and the focus
  measure read the difference as sharpness. That bias was 187% on the CuPy path
  against 25% on the CPU one, because bilinear loses so much more per warp.
* Item 4 is why there is no second path any more. `map_coordinates` has no
  Lanczos kernel, so the CuPy warp ran at order=1, and bilinear keeps 72% of the
  high-frequency energy Lanczos4 keeps on a real frame. In a focus stacker that
  is the wrong thing to trade: every frame is resampled *before* the focus
  measure reads it, so the detail the kernel discards is the signal the whole
  pipeline exists to find. The GPU was not buying anything for it either - at the
  spline order that matches Lanczos4's sharpness it is slower than the CPU warp.

So the bar now:

1. The one warp is a wide-kernel one, and a stage's output carries the
   high-frequency energy that implies rather than bilinear's (`TestWarpKernel`).
   These fail if `WARP_INTERPOLATION` is lowered or a second path is added back.
2. Registering a stack with known drift *removes* it rather than doubling it
   (`TestDriftIsRemoved`) - item 1's assertion, kept, since a sign error in a
   warp is not a device-specific mistake.
3. No stage reaches for a GPU array library while warping
   (`TestNoSecondWarpPath`), which is what makes 1 and 2 hold on every machine
   rather than only on the one running the suite.

Run with:  python -m pytest tests/test_registration_warp_kernel.py -v
"""

import builtins
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registration import (
    ImageRegistration,
    WARP_INTERPOLATION,
    _warp_frame,
)
from tests.synthetic_stack import make_reference


_real_import = builtins.__import__


# --- shared synthetic input (mirrors the other registration tests) ---

def _base_scene(size=384, seed=5):
    """A feature-rich BGR frame both SIFT and ECC can lock onto."""
    img = make_reference(height=size, width=size, seed=seed, style="texture")
    for (cy, cx) in [(96, 96), (96, 288), (288, 96), (288, 288), (192, 192)]:
        cv2.circle(img, (cx, cy), 16, (30, 220, 30), -1)
    return img


def _drifting_stack(size=384, n=6, step=3.0, seed=5):
    """A stack that translates by a known, constant amount per frame."""
    base = _base_scene(size=size, seed=seed)
    stack = []
    for i in range(n):
        M = np.float32([[1, 0, i * step], [0, 1, i * step * 0.6]])
        stack.append(cv2.warpAffine(base, M, (size, size),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_REFLECT))
    return stack


def _sharpness(img):
    """High-frequency energy: the variance of the Laplacian on the luma.

    The same measure the audit states item 4 in, and the one a selection-based
    fusion effectively asks each frame for.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


def _interior(img, margin=24):
    """Drop the border, where the warp pads rather than resamples."""
    return img[margin:-margin, margin:-margin]


def _residual_shift(a, b):
    """Sub-pixel displacement between two frames, measured by ECC.

    Phase correlation is the obvious tool and the wrong one here: the reference
    scene carries a checkerboard band, whose periodicity leaves the correlation
    peak ambiguous - it reports half a pixel between two byte-identical frames.
    ECC has no such failure mode and returns exactly zero on identical input.
    """
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_TRANSLATION,
                                   criteria, None, 1)
    return float(np.hypot(warp[0, 2], warp[1, 2]))


def _register(frames, method):
    return ImageRegistration(method=method, downscale_width=1024,
                             reference_index=0).process(
                                 [f.copy() for f in frames],
                                 output_path=None, thread_count=2)


# The sub-pixel transform the assertions below bracket against: a fraction of a
# pixel of translation plus the slight magnification a focus stack breathes by,
# i.e. the kind of matrix the stages actually produce.
_TYPICAL_WARP = np.array([[1.0008, 0.0007, 0.43],
                          [-0.0007, 1.0008, -0.61],
                          [0.0, 0.0, 1.0]], dtype=np.float64)


def _kernel_bracket(img, M):
    """(bilinear, lanczos) sharpness of one frame warped by M.

    The two ends of the trade item 4 is about, measured on the input at hand so
    the assertions do not depend on an absolute number.
    """
    h, w = img.shape[:2]
    linear = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
    lanczos = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT)
    return _sharpness(_interior(linear)), _sharpness(_interior(lanczos))


class TestWarpKernel:
    """The warp keeps the detail a wide kernel keeps, not what bilinear keeps."""

    def test_the_kernel_is_lanczos4(self):
        assert WARP_INTERPOLATION == cv2.INTER_LANCZOS4

    def test_the_bracket_can_tell_the_two_kernels_apart(self):
        """Guards the guard: the fixture has to see the difference it asserts on."""
        img = _base_scene()
        linear, lanczos = _kernel_bracket(img, _TYPICAL_WARP)
        assert lanczos > 1.15 * linear, (
            f"bilinear {linear:.1f} vs Lanczos4 {lanczos:.1f} - this input cannot "
            "distinguish the kernels, so the assertions below prove nothing")

    def test_warp_frame_resamples_with_the_wide_kernel(self):
        img = _base_scene()
        linear, lanczos = _kernel_bracket(img, _TYPICAL_WARP)
        h, w = img.shape[:2]

        got = _sharpness(_interior(_warp_frame(img, _TYPICAL_WARP, w, h)))

        assert got == pytest.approx(lanczos, rel=1e-6)
        assert got > linear

    @pytest.mark.parametrize("method", ["scale", "homography", "ecc"])
    def test_a_stage_does_not_warp_the_stack_down_to_bilinear(self, method):
        """End to end: the kernel survives the stage, not just the helper.

        Registration necessarily costs some sharpness - it is a resample - so
        the bar is the midpoint between the two kernels on this input rather
        than the Lanczos figure itself. The GPU path this file used to compare
        against landed below the bilinear end of that bracket.
        """
        frames = _drifting_stack()
        linear, lanczos = _kernel_bracket(frames[0], _TYPICAL_WARP)
        midpoint = 0.5 * (linear + lanczos)

        aligned = _register(frames, method)

        got = float(np.mean([_sharpness(_interior(f)) for f in aligned]))
        assert got > midpoint, (
            f"{method}: aligned frames average {got:.1f}, between bilinear "
            f"{linear:.1f} and Lanczos4 {lanczos:.1f} - the stack is being "
            "resampled with a narrower kernel than the one asked for")


class TestDriftIsRemoved:
    """A warp that applies its transform backwards makes the stack worse."""

    @pytest.mark.parametrize("method", ["ecc", "scale"])
    def test_registration_removes_the_drift_rather_than_doubling_it(self, method):
        """The defect this file was opened for.

        Registering a stack that drifts by a known step must leave every frame
        on top of the reference. Applying the inverse correction instead left
        each frame at roughly twice its original offset, so an assertion against
        the *unregistered* drift separates a fix from a regression.
        """
        step = 3.0
        frames = _drifting_stack(step=step)

        aligned = _register(frames, method)

        for i, frame in enumerate(aligned[1:], start=1):
            before = np.hypot(i * step, i * step * 0.6)
            after = _residual_shift(aligned[0], frame)
            assert after < 0.5 * before, (
                f"frame {i}: {after:.2f} px residual against {before:.2f} px of drift"
            )


class TestNoSecondWarpPath:
    """One warp, so there is nothing for a second one to disagree with."""

    @pytest.mark.parametrize("method", ["scale", "ecc"])
    def test_no_stage_reaches_for_a_gpu_array_library(self, method):
        """Every reintroduction of the deleted path starts with this import.

        Recorded rather than blocked: a machine that has CuPy installed for
        something else would otherwise pass by luck, and one that does not would
        pass without exercising anything.
        """
        seen = []

        def watched(name, *args, **kwargs):
            seen.append(name)
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = watched
        try:
            _register(_drifting_stack(size=192, n=3), method)
        finally:
            builtins.__import__ = _real_import

        gpu_imports = [n for n in seen if n.split(".")[0] in ("cupy", "cupyx", "torch")]
        assert not gpu_imports, f"{method} imported {sorted(set(gpu_imports))} while registering"
