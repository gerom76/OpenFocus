"""The GPU warp path must apply the same transform as the CPU one.

`cv2.warpPerspective(img, M, dsize)` computes `dst(x, y) = src(M^-1 . (x, y))` -
it takes the forward source-to-destination map and inverts it internally.
`map_coordinates`, which the CuPy path samples with, has no such convention: it
needs the destination-to-source map handed to it explicitly.

The ECC stage used to carry its own inlined copy of the GPU warp that skipped
that inversion, so on any CUDA machine it applied the inverse of the correction
it had just measured - doubling the misalignment instead of removing it. Nothing
in the suite exercised the CuPy branch, so the defect was invisible to it.

The bar the warp has to clear:

1. `_warp_perspective_gpu` interprets its matrix the same way `cv2` does, and a
   sign error shows up as a displacement in the opposite direction rather than
   as blur (`TestWarpConvention`).
2. Every stage that warps agrees with its own CPU path end to end, to well under
   a pixel (`TestDeviceAgreement`).

The CuPy tests skip on a machine without a usable CUDA device; the convention
tests that can run on the CPU alone do.

Run with:  python -m pytest tests/test_registration_gpu_warp.py -v
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
    _init_gpu_warp,
    _warp_perspective_gpu,
)
from tests.synthetic_stack import make_reference


_real_import = builtins.__import__


class without_cupy:
    """Run a block with the CuPy warp path unavailable, to force the CPU one.

    Lifted from tests/benchmark_registration.py, which uses it to compare the
    two devices on the same input.
    """

    def __enter__(self):
        def guarded(name, *args, **kwargs):
            if name.startswith("cupy"):
                raise ImportError("disabled by test_registration_gpu_warp")
            return _real_import(name, *args, **kwargs)
        builtins.__import__ = guarded
        return self

    def __exit__(self, *exc):
        builtins.__import__ = _real_import


def _gpu_or_skip():
    """The (cp, ndimage) pair, or skip - there is nothing to assert without one."""
    cp, cp_ndimage = _init_gpu_warp()
    if cp is None:
        pytest.skip("CuPy with a usable CUDA device is required for the GPU warp path")
    return cp, cp_ndimage


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


def _register(frames, method, force_cpu):
    context = without_cupy() if force_cpu else None
    if context:
        context.__enter__()
    try:
        return ImageRegistration(method=method, downscale_width=1024,
                                 reference_index=0).process(
                                     [f.copy() for f in frames],
                                     output_path=None, thread_count=2)
    finally:
        if context:
            context.__exit__()


class TestWarpConvention:
    """The matrix means the same thing on both devices."""

    def test_translation_moves_content_the_same_way_as_cv2(self):
        cp, cp_ndimage = _gpu_or_skip()
        img = _base_scene(size=128, seed=7)
        # A pure integer translation: the result is a copy, so the two paths
        # have to agree exactly rather than to within an interpolation kernel.
        M = np.array([[1, 0, 12], [0, 1, -7], [0, 0, 1]], dtype=np.float32)

        gpu = _warp_perspective_gpu(cp, cp_ndimage, img, M, 128, 128)
        cpu = cv2.warpPerspective(img, M, (128, 128),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT)

        # Compare inside the valid region; the borders differ only in what each
        # path pads with.
        assert np.array_equal(gpu[10:110, 20:110], cpu[10:110, 20:110])

    def test_inverting_the_matrix_is_not_a_no_op(self):
        """Guards the guard: the fixture above can actually see a sign error."""
        cp, cp_ndimage = _gpu_or_skip()
        img = _base_scene(size=128, seed=7)
        M = np.array([[1, 0, 12], [0, 1, -7], [0, 0, 1]], dtype=np.float32)
        M_wrong = np.linalg.inv(M).astype(np.float32)

        right = _warp_perspective_gpu(cp, cp_ndimage, img, M, 128, 128)
        wrong = _warp_perspective_gpu(cp, cp_ndimage, img, M_wrong, 128, 128)

        assert not np.array_equal(right[10:110, 20:110], wrong[10:110, 20:110])

    def test_single_channel_input_is_accepted(self):
        """The shared helper handles 2-D frames; the inlined ECC copy did not."""
        cp, cp_ndimage = _gpu_or_skip()
        gray = cv2.cvtColor(_base_scene(size=128, seed=7), cv2.COLOR_BGR2GRAY)
        M = np.array([[1, 0, 5], [0, 1, 5], [0, 0, 1]], dtype=np.float32)

        out = _warp_perspective_gpu(cp, cp_ndimage, gray, M, 128, 128)

        assert out.shape == (128, 128)
        assert out.dtype == gray.dtype


class TestDeviceAgreement:
    """Each stage lands in the same place whichever device warped it."""

    @pytest.mark.parametrize("method", ["ecc", "scale"])
    def test_gpu_and_cpu_agree_to_a_fraction_of_a_pixel(self, method):
        _gpu_or_skip()
        frames = _drifting_stack()

        gpu = _register(frames, method, force_cpu=False)
        cpu = _register(frames, method, force_cpu=True)

        assert len(gpu) == len(cpu)
        assert gpu[0].shape == cpu[0].shape, "the two devices cropped differently"
        for i, (g, c) in enumerate(zip(gpu, cpu)):
            # Bilinear against Lanczos4 changes the pixels, so compare where the
            # content landed rather than the samples themselves.
            assert _residual_shift(g, c) < 0.25, f"frame {i} landed elsewhere on the GPU"

    @pytest.mark.parametrize("method", ["ecc", "scale"])
    def test_gpu_removes_the_drift_rather_than_doubling_it(self, method):
        """The defect this file exists for: the sign error made frames worse.

        Registering a stack that drifts by a known step must leave every frame
        on top of the reference. Applying the inverse correction instead left
        each frame at roughly twice its original offset, so an assertion against
        the *unregistered* drift separates a fix from a regression.
        """
        _gpu_or_skip()
        step = 3.0
        frames = _drifting_stack(step=step)

        aligned = _register(frames, method, force_cpu=False)

        for i, frame in enumerate(aligned[1:], start=1):
            before = np.hypot(i * step, i * step * 0.6)
            after = _residual_shift(aligned[0], frame)
            assert after < 0.5 * before, (
                f"frame {i}: {after:.2f} px residual against {before:.2f} px of drift"
            )
