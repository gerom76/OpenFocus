"""
The quality contract every fusion method has to satisfy.

test_gff_quality.py covers the Guided Filter in depth; this module applies the
same style of check to every implementation in the project - DCT, DTCWT,
GFG-FGF, StackMFF-V4, the IFCNN refinement stage and the GPU variants - through
the uniform interface in tests/fusion_registry.py.

A method whose dependency or weights file is missing skips with the reason
instead of failing, so the suite is meaningful on a partial install.

Run with:  python -m pytest tests/test_fusion_quality.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_metrics as fm
from tests import fusion_registry as reg
from tests.synthetic_stack import make_stack

# The comparison fixture is deliberately photographic rather than the noise
# texture used by the Guided Filter suite: variance-based focus measures cannot
# tell noise from sharpness, which would penalise DCT for the fixture's sake.
STACK_KWARGS = dict(num_slices=3, height=256, width=256, seed=7, style="photographic")

# Floors that hold for every method, regardless of algorithm
MIN_SSIM = 0.95
MIN_QABF = 0.45
MIN_SLICE_MARGIN_DB = 3.0      # fused must beat the best single slice by this
MIN_DETAIL_FRACTION = 0.70     # fused spatial frequency vs the ground truth


def _params():
    """One pytest param per registered method, pre-marked with its skip reason."""
    out = []
    for method in reg.METHODS:
        ok, reason = method.available()
        marks = [] if ok else [pytest.mark.skip(reason=f"{method.label}: {reason}")]
        out.append(pytest.param(method, id=method.key, marks=marks))
    return out


ALL_METHODS = _params()


@pytest.fixture(scope="module")
def fixture():
    """The shared stack, its ground truth, and the per-slice focus bands."""
    stack, reference, _ = make_stack(**STACK_KWARGS)
    return stack, reference


@pytest.fixture(scope="module")
def fused_cache():
    """Fusing StackMFF-V4 and IFCNN is slow; run each method at most once."""
    return {}


def _fused(method, fixture, cache):
    if method.key not in cache:
        stack, _ = fixture
        cache[method.key] = method.run(stack)
    return cache[method.key]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_output_contract(method, fixture, fused_cache):
    """Every method returns a uint8 BGR image matching the input geometry."""
    stack, _ = fixture
    out = _fused(method, fixture, fused_cache)
    assert out is not None, "fusion returned None"
    assert out.dtype == np.uint8
    assert out.shape == stack[0].shape


@pytest.mark.parametrize("method", ALL_METHODS)
def test_folder_input(method, fixture, fused_cache, tmp_path_factory):
    """Loading a stack from disk must produce the same geometry as in-memory."""
    if not method.supports_folder:
        pytest.skip(f"{method.label} takes a fused result, not a directory")

    stack, _ = fixture
    folder = tmp_path_factory.mktemp(f"stack_{method.key}")
    for i, img in enumerate(stack):
        cv2.imwrite(str(folder / f"slice_{i:02d}.png"), img)

    out = method.run(str(folder))
    assert out.shape == stack[0].shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("method", ALL_METHODS)
def test_resize(method, fixture):
    """img_resize is honoured, or refused outright - never silently ignored."""
    stack, _ = fixture
    target = (192, 128)  # (width, height); clears every method's minimum

    if not method.supports_resize:
        with pytest.raises(ValueError):
            method.run(stack, img_resize=target)
        return

    out = method.run(stack, img_resize=target)
    assert out.shape[:2] == (target[1], target[0])


@pytest.mark.parametrize("method", ALL_METHODS)
def test_resize_at_the_documented_minimum(method, fixture):
    """A method that declares a minimum edge must actually work at it."""
    if method.min_dimension is None:
        pytest.skip(f"{method.label} declares no minimum size")

    stack, _ = fixture
    edge = method.min_dimension
    out = method.run(stack, img_resize=(edge, edge))
    assert out.shape[:2] == (edge, edge)


@pytest.mark.xfail(reason="StackMFF-V4 raises an opaque torch max_pool2d error "
                          "below 112 px instead of rejecting the size clearly",
                   raises=RuntimeError, strict=False)
def test_undersized_resize_is_rejected_clearly(fixture):
    """
    Asking StackMFF-V4 for a preview smaller than its pooling depth allows
    should fail with an explanation, not a RuntimeError about a 480x0x1 tensor.
    Marked xfail because it currently does the latter - the app can hit this by
    setting a small preview size.
    """
    method = reg.get("stackmffv4")
    ok, reason = method.available()
    if not ok:
        pytest.skip(reason)

    stack, _ = fixture
    with pytest.raises(ValueError):
        method.run(stack, img_resize=(96, 96))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_output_size_for_awkward_dimensions(method, fixture):
    """
    A stack whose dimensions are neither even nor block-aligned.

    Most methods return the input geometry untouched. DCT crops down to a
    multiple of its block size and DTCWT pads an odd edge up to even, so those
    declare preserves_size=False and are held to a bounded deviation instead -
    a caller that assumes the output matches the input will misalign on them.
    """
    stack, _, _ = make_stack(num_slices=3, height=250, width=170, seed=7,
                             style="photographic")
    height, width = stack[0].shape[:2]
    out = method.run(stack)

    if method.preserves_size:
        assert out.shape == stack[0].shape, (
            f"{method.label} changed {height}x{width} to "
            f"{out.shape[0]}x{out.shape[1]} but claims to preserve size")
        return

    # Bounded: at most one block lost, or one row/column of padding gained
    assert abs(out.shape[0] - height) <= 8
    assert abs(out.shape[1] - width) <= 8
    assert out.shape[2] == stack[0].shape[2]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_repeat_run_is_stable(method, fixture, fused_cache):
    """
    Fusing the same stack twice must give the same answer.

    Methods that accumulate weights across a thread pool add in completion
    order, so a few pixels may land one level apart; nothing may drift further.
    """
    stack, _ = fixture
    first = _fused(method, fixture, fused_cache)
    second = method.run(stack)

    if method.deterministic:
        assert np.array_equal(first, second)
    else:
        diff = np.abs(first.astype(np.int16) - second.astype(np.int16))
        assert diff.max() <= 1
        assert np.mean(diff > 0) < 0.001


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_beats_every_source_slice(method, fixture, fused_cache):
    """No input is sharp everywhere, so fusion must beat all of them clearly."""
    stack, reference = fixture
    out = _fused(method, fixture, fused_cache)

    fused_psnr = fm.psnr(out, reference)
    best_slice = max(fm.psnr(s, reference) for s in stack)
    assert fused_psnr > best_slice + MIN_SLICE_MARGIN_DB, (
        f"{method.label}: {fused_psnr:.2f} dB vs best slice {best_slice:.2f} dB")


@pytest.mark.parametrize("method", ALL_METHODS)
def test_reconstruction_floor(method, fixture, fused_cache):
    """Per-method regression guard, set well below the measured value."""
    _, reference = fixture
    out = _fused(method, fixture, fused_cache)

    psnr = fm.psnr(out, reference)
    assert psnr > method.min_psnr, f"{method.label}: {psnr:.2f} dB"
    assert fm.ssim(out, reference) > MIN_SSIM


@pytest.mark.parametrize("method", ALL_METHODS)
def test_edge_information_is_preserved(method, fixture, fused_cache):
    """Q^AB/F is the metric that carries over to stacks with no ground truth."""
    stack, _ = fixture
    out = _fused(method, fixture, fused_cache)

    score = fm.qabf(out, stack)
    assert score > MIN_QABF, f"{method.label}: Q_ABF {score:.4f}"


@pytest.mark.parametrize("method", ALL_METHODS)
def test_detail_is_recovered_not_smoothed(method, fixture, fused_cache):
    """Spatial frequency must approach the truth, not the blurred inputs."""
    stack, reference = fixture
    out = _fused(method, fixture, fused_cache)

    fused_sf = fm.spatial_frequency(out)
    assert fused_sf > max(fm.spatial_frequency(s) for s in stack)
    assert fused_sf > MIN_DETAIL_FRACTION * fm.spatial_frequency(reference)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_each_band_comes_from_its_focused_slice(method, fixture, fused_cache):
    """
    Per band, the fused pixels must track the slice focused there more closely
    than the slices that are blurred there. Rows near a boundary are skipped:
    every method blends neighbours across the transition.
    """
    stack, reference = fixture
    out = _fused(method, fixture, fused_cache)

    height = reference.shape[0]
    edges = np.linspace(0, height, len(stack) + 1).round().astype(int)

    for k in range(len(stack)):
        band = slice(edges[k] + 12, edges[k + 1] - 12)
        own = fm.psnr(out[band], stack[k][band])
        others = [fm.psnr(out[band], stack[j][band])
                  for j in range(len(stack)) if j != k]
        assert own > max(others), f"{method.label}: band {k} favoured the wrong slice"


@pytest.mark.parametrize("method", ALL_METHODS)
def test_no_saturation_artifacts(method, fixture, fused_cache):
    """Fusion may not clip meaningfully more than the ground truth already does."""
    _, reference = fixture
    out = _fused(method, fixture, fused_cache)

    clipped = np.mean((out == 0) | (out == 255))
    clipped_ref = np.mean((reference == 0) | (reference == 255))
    assert clipped < clipped_ref + 0.02, f"{method.label}: {clipped:.3f} clipped"


# ---------------------------------------------------------------------------
# CPU / GPU parity
# ---------------------------------------------------------------------------

def _parity_params():
    out = []
    for cpu_key, gpu_key in reg.PARITY_PAIRS:
        cpu, gpu = reg.get(cpu_key), reg.get(gpu_key)
        cpu_ok, cpu_why = cpu.available()
        gpu_ok, gpu_why = gpu.available()
        marks = []
        if not (cpu_ok and gpu_ok):
            marks = [pytest.mark.skip(reason=cpu_why if not cpu_ok else gpu_why)]
        out.append(pytest.param(cpu, gpu, id=cpu_key, marks=marks))
    return out


@pytest.mark.parametrize("cpu,gpu", _parity_params())
def test_gpu_matches_cpu(cpu, gpu, fixture):
    """
    A GPU path mirrors its CPU implementation, so both must land in the same
    place. The bar is agreement in the same league, not bit equality - the two
    use different filter primitives and accumulate in a different order.
    """
    stack, reference = fixture
    cpu_out = cpu.run(stack)
    gpu_out = gpu.run(stack)

    assert gpu_out.shape == cpu_out.shape
    agreement = fm.psnr(gpu_out, cpu_out)
    assert agreement > 25.0, f"{cpu.label}: CPU/GPU agreement only {agreement:.2f} dB"

    # Neither side may be the worse implementation of the pair by a wide margin
    cpu_psnr = fm.psnr(cpu_out, reference)
    gpu_psnr = fm.psnr(gpu_out, reference)
    assert abs(cpu_psnr - gpu_psnr) < 5.0, (
        f"{cpu.label}: CPU {cpu_psnr:.2f} dB vs GPU {gpu_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

def test_registry_covers_the_dispatcher():
    """
    Every algorithm the app can dispatch has a registered test adapter, so a
    newly added fusion method cannot silently escape this suite.
    """
    from core.multi_focus_fusion import MultiFocusFusion

    covered = {m.key for m in reg.METHODS}
    for algorithm in MultiFocusFusion.SUPPORTED_ALGORITHMS:
        assert algorithm in covered, f"{algorithm} has no entry in fusion_registry"


def test_at_least_the_cpu_methods_are_available():
    """The pure-OpenCV methods have no optional dependency and must always run."""
    for key in ("guided_filter", "gfgfgf", "dct"):
        ok, reason = reg.get(key).available()
        assert ok, f"{key} unexpectedly unavailable: {reason}"
