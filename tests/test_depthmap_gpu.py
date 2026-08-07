"""
Contract tests for the GPU depth-map fusion path (depthmap_torch.py).

The device path is a mirror of fusion_methods/depthmap.py, so almost everything
worth asserting is a comparison against the CPU implementation rather than an
absolute number. All of it runs on a plain ``cpu`` torch device, which is the
point: the batching, the chunk loop and the elliptical dilation are the parts
most likely to break, and none of them needs a card to be exercised. The
CPU/GPU quality parity on a real device is covered separately by
tests/test_fusion_quality.py.

Run with:  python -m pytest tests/test_depthmap_gpu.py -v
"""

import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

# Imported after the skip check, so a machine without PyTorch skips this file
# rather than failing to collect it.
from fusion_methods.depthmap import (  # noqa: E402
    MODE_AVERAGE, MODE_MAX, depthmap_impl,
)
from fusion_methods.depthmap_torch import (  # noqa: E402
    CHUNK_FALLBACK, CHUNK_MAX, CHUNK_MIN,
    _chunk_size, _dilate_ellipse, _ellipse_spans, depthmap_torch_impl,
)

DEV = "cpu"     # the algorithm is device-agnostic; see the module docstring
KERNEL = 9


def _stack(size=96, frames=4, seed=3, dtype=np.uint8):
    """A stack whose sharp band walks down the frame, one band per slice."""
    rng = np.random.default_rng(seed)
    scale = 255.0 if dtype == np.uint8 else 65535.0
    texture = rng.random((size, size, 3), dtype=np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 0.8) * scale

    edges = np.linspace(0, size, frames + 1).round().astype(int)
    stack = []
    for k in range(frames):
        frame = cv2.GaussianBlur(texture, (0, 0), 3.0)
        band = slice(edges[k], edges[k + 1])
        frame[band] = texture[band]
        stack.append(np.clip(frame, 0, scale).astype(dtype))
    return stack


@pytest.fixture(scope="module")
def stack():
    return _stack()


def _psnr(a, b):
    scale = 255.0 if a.dtype == np.uint8 else 65535.0
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10(scale * scale / mse)


# ---------------------------------------------------------------------------
# The elliptical element, which is where a GPU rewrite most easily drifts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5, 8, 12])
def test_spans_match_opencvs_element(radius):
    """The decomposed spans must describe cv2's ellipse row for row."""
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * radius + 1,) * 2)
    expected = {}
    for dy, row in enumerate(element, start=-radius):
        expected[dy] = (int(row.sum()) - 1) // 2

    got = {dy: half for half, offsets in _ellipse_spans(radius) for dy in offsets}
    assert got == expected


@pytest.mark.parametrize("radius", [1, 2, 4, 8])
def test_dilation_matches_opencv(radius):
    """Grey dilation on the device must equal cv2.dilate exactly, not roughly."""
    rng = np.random.default_rng(11)
    energy = rng.random((37, 41), dtype=np.float32)

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * radius + 1,) * 2)
    expected = cv2.dilate(energy, element)

    tensor = torch.from_numpy(energy).view(1, 1, *energy.shape)
    got = _dilate_ellipse(tensor, radius)[0, 0].numpy()

    assert np.array_equal(got, expected)


def test_zero_radius_is_a_no_op():
    energy = torch.rand(1, 1, 8, 8)
    assert _dilate_ellipse(energy, 0) is energy


# ---------------------------------------------------------------------------
# Agreement with the CPU implementation it mirrors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [MODE_MAX, MODE_AVERAGE])
def test_matches_the_cpu_path(stack, mode):
    cpu = depthmap_impl(list(stack), mode=mode, kernel_size=KERNEL)
    gpu = depthmap_torch_impl(list(stack), mode=mode, kernel_size=KERNEL,
                              device=DEV)

    assert gpu.shape == cpu.shape
    assert gpu.dtype == cpu.dtype
    # The two differ only in the border convention of the pooling filter, so
    # they agree everywhere except a window's reach from the edge.
    assert _psnr(gpu, cpu) > 40.0
    inner = (slice(KERNEL, -KERNEL), slice(KERNEL, -KERNEL))
    assert _psnr(gpu[inner], cpu[inner]) > 55.0


@pytest.mark.parametrize("mode", [MODE_MAX, MODE_AVERAGE])
def test_halo_radius_matches_the_cpu_path(stack, mode):
    cpu = depthmap_impl(list(stack), mode=mode, kernel_size=KERNEL, halo_radius=6)
    gpu = depthmap_torch_impl(list(stack), mode=mode, kernel_size=KERNEL,
                              halo_radius=6, device=DEV)
    inner = (slice(KERNEL, -KERNEL), slice(KERNEL, -KERNEL))
    assert _psnr(gpu[inner], cpu[inner]) > 55.0


# ---------------------------------------------------------------------------
# The chunking is an implementation detail and must stay one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selectivity", [0, 100])
def test_selectivity_matches_the_cpu_path(stack, selectivity):
    """The average's weighting dial, at both ends rather than at its default."""
    cpu = depthmap_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                        selectivity=selectivity)
    gpu = depthmap_torch_impl(list(stack), mode=MODE_AVERAGE, kernel_size=KERNEL,
                              selectivity=selectivity, device=DEV)
    inner = (slice(KERNEL, -KERNEL), slice(KERNEL, -KERNEL))
    assert _psnr(gpu[inner], cpu[inner]) > 55.0


@pytest.mark.parametrize("mode", [MODE_MAX, MODE_AVERAGE])
def test_chunk_size_does_not_change_the_result(stack, mode):
    """Whatever the VRAM sizing picks, the pixels must be the same."""
    one = depthmap_torch_impl(list(stack), mode=mode, kernel_size=KERNEL,
                              device=DEV, chunk_size=1)
    whole = depthmap_torch_impl(list(stack), mode=mode, kernel_size=KERNEL,
                                device=DEV, chunk_size=len(stack))
    if mode == MODE_MAX:
        # A hard select copies source pixels, so the batching cannot move them.
        assert np.array_equal(one, whole)
    else:
        # The blend sums a chunk at a time, so only the float ordering differs.
        assert _psnr(one, whole) > 60.0


def test_chunk_sizing_stays_inside_its_bounds():
    """A device that cannot report free memory gets the fixed fallback."""
    assert _chunk_size(3000, 4000, 3, 0, torch.device("cpu")) == CHUNK_FALLBACK

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        # A tile fits many times over; a frame far larger than any card does not.
        assert _chunk_size(1024, 1024, 3, 0, dev) == CHUNK_MAX
        assert _chunk_size(200000, 200000, 3, 0, dev) == CHUNK_MIN


def test_repeat_runs_are_identical(stack):
    first = depthmap_torch_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                                device=DEV)
    second = depthmap_torch_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                                 device=DEV)
    assert np.array_equal(first, second)


def test_frame_order_does_not_change_max_mode(stack):
    """The argmax is over the stack, so shuffling the inputs cannot move it."""
    forward = depthmap_torch_impl(list(stack), mode=MODE_MAX, kernel_size=KERNEL,
                                  device=DEV)
    reverse = depthmap_torch_impl(list(reversed(stack)), mode=MODE_MAX,
                                  kernel_size=KERNEL, device=DEV)
    assert _psnr(forward, reverse) > 45.0


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_sixteen_bit_stays_sixteen_bit():
    stack16 = _stack(dtype=np.uint16)
    fused = depthmap_torch_impl(stack16, mode=MODE_MAX, kernel_size=KERNEL,
                                device=DEV)
    assert fused.dtype == np.uint16
    assert fused.max() > 255      # not an 8-bit result wearing a wide dtype


def test_grayscale_is_refused_rather_than_mangled(stack):
    """The dispatcher falls back to the CPU on this, so it must raise."""
    grey = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in stack]
    with pytest.raises(ValueError):
        depthmap_torch_impl(grey, mode=MODE_MAX, kernel_size=KERNEL, device=DEV)


def test_too_small_for_the_window_is_refused(stack):
    tiny = [img[:4, :4] for img in stack]
    with pytest.raises(ValueError):
        depthmap_torch_impl(tiny, mode=MODE_MAX, kernel_size=31, device=DEV)


def test_unknown_mode_is_refused(stack):
    with pytest.raises(ValueError):
        depthmap_torch_impl(list(stack), mode="median", device=DEV)


def test_resize_is_honoured(stack):
    fused = depthmap_torch_impl(list(stack), img_resize=(64, 48), mode=MODE_MAX,
                                kernel_size=KERNEL, device=DEV)
    assert fused.shape == (48, 64, 3)
