"""
Defocus-wash tests for the DCT fusion method (dct.py, dct_torch.py).

The stack is built the way a macro shot records a bright object standing well
clear of its backdrop: the frame focused on the object carries that object's
defocus as a broad, smooth wash spread across the backdrop, wiping out the faint
fine markings the other frame renders sharply.

A focus measure that scores a block by its total contrast reads that wash as
detail - a smooth ramp crossing one block carries more energy than fine,
low-amplitude texture does - so the blurred frame wins whole regions and its
pixels are copied through verbatim, leaving flat homogeneous patches on the
block grid. That is item 17 in docs/ALGORITHM_IMPROVEMENTS.md, and it is what
these tests guard against coming back.

Thresholds are loose regression guards, not a leaderboard.

Run with:  python -m pytest tests/test_dct_defocus_wash.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.dct import dct_focus_stack_fusion

BLUR = 61     # heavy defocus, so the wash reaches far past one block
DISC = (110, 110, 52)  # centre x, centre y, radius of the bright object


def _wash_stack(size=384, seed=7):
    """Two frames: one focused on a bright disc, one on the marked backdrop.

    Returns (stack, reference, backdrop mask). The backdrop carries faint fine
    marks - the detail the wash destroys - and the disc carries its own texture
    so that frame is decidable on its own merits.
    """
    rng = np.random.default_rng(seed)

    reference = np.full((size, size, 3), 22, dtype=np.uint8)
    reference[:, :, 1] = 46  # dark board

    for _ in range(260):
        x, y = int(rng.integers(4, size - 14)), int(rng.integers(4, size - 8))
        v = int(rng.integers(70, 120))
        cv2.rectangle(reference, (x, y),
                      (x + int(rng.integers(2, 9)), y + 2), (v, v, v), -1)
    for _ in range(40):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        cv2.line(reference, (x, y), (x + int(rng.integers(10, 40)), y),
                 (140, 150, 140), 1)

    cx, cy, r = DISC
    near = np.zeros((size, size), dtype=np.float32)
    cv2.circle(reference, (cx, cy), r, (238, 244, 250), -1)
    cv2.circle(near, (cx, cy), r, 1.0, -1)
    for ring in range(8, r, 7):
        cv2.circle(reference, (cx, cy), ring, (150, 160, 170), 1)

    blurred = cv2.GaussianBlur(reference, (BLUR, BLUR), 0)
    stack = []
    for mask in (near, 1.0 - near):
        a = mask[:, :, None]
        img = reference.astype(np.float32) * a + blurred.astype(np.float32) * (1.0 - a)
        img += rng.normal(0.0, 2.0, img.shape)  # a little grain, as in a real shot
        stack.append(np.clip(img, 0, 255).astype(np.uint8))
    return stack, reference, near < 0.5


def _wrong_frame_share(fused, stack, backdrop):
    """Percentage of the backdrop taken from the frame that is blurred there.

    DCT copies pixels verbatim, so each output pixel can be attributed to the
    source it is closest to.
    """
    d_near = np.abs(fused.astype(np.int16) - stack[0].astype(np.int16)).sum(2)
    d_far = np.abs(fused.astype(np.int16) - stack[1].astype(np.int16)).sum(2)
    return float(np.mean((d_near < d_far)[backdrop]) * 100.0)


def _psnr(fused, reference):
    mse = np.mean((fused.astype(np.float64) - reference.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


@pytest.fixture(scope="module")
def scene():
    return _wash_stack()


# --------------------------------------------------------------------------
# The wash must not win the backdrop
# --------------------------------------------------------------------------

def test_backdrop_comes_from_the_frame_that_resolves_it(scene):
    """The measured failure was 35.7% of the backdrop; the fix leaves 2.0%."""
    stack, _, backdrop = scene
    fused = dct_focus_stack_fusion(list(stack))
    share = _wrong_frame_share(fused, stack, backdrop)
    assert share < 8.0, (
        f"{share:.1f}% of the backdrop was taken from the frame that is blurred "
        "there - the focus measure is scoring contrast rather than sharpness "
        "again (item 17)")


def test_fusing_beats_either_frame_on_this_scene(scene):
    """Before the fix DCT gained 0.55 dB here; the wash cost as much as fusing won."""
    stack, reference, _ = scene
    fused = dct_focus_stack_fusion(list(stack))
    best_slice = max(_psnr(s, reference) for s in stack)
    assert _psnr(fused, reference) > best_slice + 2.0


def test_no_large_flat_patch_replaces_the_markings(scene):
    """
    The artefact's signature: a wide region of the backdrop with none of the
    local detail the reference has there. Measured as the share of the backdrop
    whose high-pass energy collapsed against the reference's.
    """
    stack, reference, backdrop = scene
    fused = dct_focus_stack_fusion(list(stack))

    def detail(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = g - cv2.blur(g, (9, 9))
        return cv2.blur(d * d, (9, 9))

    ref_detail, fused_detail = detail(reference), detail(fused)
    alive = ref_detail > 4.0  # where the reference genuinely has markings
    zone = backdrop & alive
    blanked = float(np.mean((fused_detail[zone] < 0.25 * ref_detail[zone])) * 100.0)
    assert blanked < 15.0, (
        f"{blanked:.1f}% of the marked backdrop came through flat")


# --------------------------------------------------------------------------
# The GPU twin must not drift from the CPU path on the same scene
# --------------------------------------------------------------------------

def test_gpu_path_agrees_on_the_wash(scene):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("no GPU device available")
    from fusion_methods.dct_torch import dct_torch_impl

    stack, _, backdrop = scene
    cpu = dct_focus_stack_fusion(list(stack))
    gpu = dct_torch_impl(list(stack))
    assert _psnr(gpu, cpu) > 40.0
    assert _wrong_frame_share(gpu, stack, backdrop) < 8.0
