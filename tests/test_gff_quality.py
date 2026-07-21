"""
Quality and correctness tests for the Guided Filter fusion method (gff.py).

The quality tests fuse a synthetic focus stack whose all-in-focus ground truth
is known, then assert the result beats every input slice on the fusion metrics
and stays close to the reference. Thresholds are deliberately loose - they are
regression guards, not a leaderboard.

Run with:  python -m pytest tests/test_gff_quality.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.gff import gff_impl
from tests import fusion_metrics as fm
from tests.synthetic_stack import make_stack

KERNEL = 31


@pytest.fixture(scope="module")
def stack_fixture():
    """A 3-slice synthetic stack plus its all-in-focus reference."""
    stack, reference, _ = make_stack(num_slices=3, height=256, width=256, seed=7)
    return stack, reference


@pytest.fixture(scope="module")
def fused(stack_fixture):
    stack, _ = stack_fixture
    return gff_impl(stack, img_resize=None, kernel_size=KERNEL)


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

def test_output_shape_and_dtype(stack_fixture, fused):
    stack, _ = stack_fixture
    assert fused.shape == stack[0].shape
    assert fused.dtype == np.uint8


def test_even_kernel_is_accepted(stack_fixture):
    """gff_impl bumps an even kernel to the next odd size instead of failing."""
    stack, _ = stack_fixture
    out = gff_impl(stack, img_resize=None, kernel_size=30)
    assert out.shape == stack[0].shape


@pytest.mark.parametrize("kernel_size", [0, None])
def test_invalid_kernel_falls_back_to_default(stack_fixture, kernel_size):
    stack, _ = stack_fixture
    out = gff_impl(stack, img_resize=None, kernel_size=kernel_size)
    assert out.shape == stack[0].shape


def test_resize_is_applied(stack_fixture):
    stack, _ = stack_fixture
    out = gff_impl(stack, img_resize=(128, 96), kernel_size=KERNEL)
    assert out.shape == (96, 128, 3)


def test_single_image_stack_round_trips(stack_fixture):
    """A one-image stack has nothing to choose between, so it must come back intact."""
    stack, _ = stack_fixture
    out = gff_impl([stack[0]], img_resize=None, kernel_size=KERNEL)
    assert fm.psnr(out, stack[0]) > 40


def test_empty_input_raises():
    with pytest.raises(ValueError):
        gff_impl([], img_resize=None, kernel_size=KERNEL)


def test_folder_input(tmp_path, stack_fixture):
    """The folder code path loads and numerically sorts slices from disk."""
    stack, _ = stack_fixture
    for i, img in enumerate(stack):
        cv2.imwrite(str(tmp_path / f"slice_{i:02d}.png"), img)

    out = gff_impl(str(tmp_path), img_resize=None, kernel_size=KERNEL)
    assert out.shape == stack[0].shape


def test_thread_count_does_not_change_result(stack_fixture, fused):
    """
    The thread pool size must not change the output beyond rounding.

    Weights are accumulated in completion order, so float addition is not
    associative across thread counts and a handful of pixels can land 1 level
    apart. Anything larger than that is a real divergence.
    """
    stack, _ = stack_fixture
    single = gff_impl(stack, img_resize=None, kernel_size=KERNEL, thread_count=1)
    diff = np.abs(single.astype(np.int16) - fused.astype(np.int16))
    assert diff.max() <= 1
    assert np.mean(diff > 0) < 0.001


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------

def test_fused_beats_every_source_slice(stack_fixture, fused):
    """Each input is sharp on only one band, so fusion must beat all of them."""
    stack, reference = stack_fixture
    fused_psnr = fm.psnr(fused, reference)
    for i, src in enumerate(stack):
        assert fused_psnr > fm.psnr(src, reference) + 1.0, f"slice {i} not beaten"


def test_close_to_ground_truth(stack_fixture, fused):
    stack, reference = stack_fixture
    assert fm.psnr(fused, reference) > 20.0
    assert fm.ssim(fused, reference) > 0.80


def test_detail_is_recovered_not_smoothed(stack_fixture, fused):
    """Spatial frequency must approach the reference, not the blurred inputs."""
    stack, reference = stack_fixture
    sf_fused = fm.spatial_frequency(fused)
    sf_ref = fm.spatial_frequency(reference)
    sf_worst = min(fm.spatial_frequency(s) for s in stack)
    assert sf_fused > sf_worst
    assert sf_fused > 0.7 * sf_ref


def test_edge_information_is_preserved(stack_fixture, fused):
    """Q^AB/F is the no-reference headline metric; 0.5 is a weak lower bound."""
    stack, _ = stack_fixture
    assert fm.qabf(fused, stack) > 0.5


def test_each_band_comes_from_its_focused_slice(stack_fixture, fused):
    """
    Per band, the fused result should track the slice focused on that band
    more closely than the slices that are blurred there.
    """
    stack, reference = stack_fixture
    height = reference.shape[0]
    edges = np.linspace(0, height, len(stack) + 1).round().astype(int)

    for k in range(len(stack)):
        # Skip rows near a boundary, where the guided filter blends neighbours
        top = edges[k] + 12
        bottom = edges[k + 1] - 12
        band = slice(top, bottom)
        own = fm.psnr(fused[band], stack[k][band])
        others = [fm.psnr(fused[band], stack[j][band])
                  for j in range(len(stack)) if j != k]
        assert own > max(others), f"band {k} did not favour its focused slice"


@pytest.mark.parametrize("kernel_size", [7, 15, 31, 63])
def test_quality_holds_across_kernel_sizes(stack_fixture, kernel_size):
    """The kernel slider is user-facing; no setting may collapse quality."""
    stack, reference = stack_fixture
    out = gff_impl(stack, img_resize=None, kernel_size=kernel_size)
    assert fm.psnr(out, reference) > 18.0
    assert fm.qabf(out, stack) > 0.45


def test_no_saturation_artifacts(stack_fixture, fused):
    """
    Reconstruction sums the base and detail layers, so it can overshoot and clip.

    The synthetic reference is itself partly saturated, so the bar is relative:
    fusion may not clip meaningfully more than the ground truth already does.
    """
    _, reference = stack_fixture
    clipped_fused = np.mean((fused == 0) | (fused == 255))
    clipped_ref = np.mean((reference == 0) | (reference == 255))
    assert clipped_fused < clipped_ref + 0.02


# --------------------------------------------------------------------------
# CPU / GPU parity
# --------------------------------------------------------------------------

def test_torch_matches_cpu(stack_fixture, fused):
    """The GPU path mirrors gff.py, so both must agree within rounding noise."""
    torch = pytest.importorskip("torch")
    from fusion_methods.gff_torch import gff_torch_impl

    stack, _ = stack_fixture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = gff_torch_impl(stack, img_resize=None, kernel_size=KERNEL, device=device)
    assert fm.psnr(out, fused) > 35.0
