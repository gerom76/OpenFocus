"""
Flat-patch tests for the DCT fusion method (dct.py, dct_torch.py).

Any block whose focal plane lands on a frame that is blurred there puts a piece
of that blur into the render. Where such blocks join up, the result is a flat
homogeneous patch on the block grid - the artefact item 17 in
docs/ALGORITHM_IMPROVEMENTS.md is about. Two different mistakes produce it, and
there is a fixture for each:

* `_wash_stack` - a bright object standing clear of its backdrop, its defocus
  spread as a broad smooth wash over faint fine markings. A measure that scores
  a block by its total contrast reads that wash as detail, because a smooth ramp
  crossing one block carries more energy than fine, low-amplitude texture does.

* `_veil_stack` - a stack whose far end is focused in front of everything, so
  those frames are a near-uniform bright veil holding no detail at all. Photon
  noise grows with brightness, so in any block where nothing is in focus the
  veil carries more high-pass energy than the dark frames do, purely as grain,
  and wins it. Every such block comes from the same frame, which is why the
  patches all share one flat tone.

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


def _veil_stack(size=384, seed=17, frames=24, photon=0.16, read_noise=1.0):
    """A stack whose last frames are a near-uniform bright veil.

    Returns (stack, reference, body interior mask). The subject is a dark body
    smooth everywhere but its bristled rim - the region where no frame resolves
    anything and the veil's grain is therefore the largest signal on offer.
    """
    rng = np.random.default_rng(seed)

    board = np.full((size, size, 3), 20, np.float32)
    board[:, :, 1] = 42
    for _ in range(320):
        x, y = int(rng.integers(4, size - 14)), int(rng.integers(4, size - 8))
        v = float(rng.integers(58, 104))
        cv2.rectangle(board, (x, y), (x + int(rng.integers(2, 9)), y + 2), (v, v, v), -1)

    body = np.zeros((size, size, 3), np.float32)
    body_a = np.zeros((size, size), np.float32)
    cv2.ellipse(body, (185, 190), (120, 84), 15, 0, 360, (32, 30, 28), -1)
    cv2.ellipse(body_a, (185, 190), (120, 84), 15, 0, 360, 1.0, -1)
    for _ in range(150):
        ang = rng.uniform(0, 2 * np.pi)
        px, py = int(185 + 116 * np.cos(ang)), int(190 + 80 * np.sin(ang))
        tip = (px + int(14 * np.cos(ang)), py + int(14 * np.sin(ang)))
        cv2.line(body, (px, py), tip, (125, 132, 122), 1)
        cv2.line(body_a, (px, py), tip, 1.0, 1)

    a3 = body_a[:, :, None]
    reference = body * a3 + board * (1.0 - a3)

    def shot(img):
        """Photon noise: variance grows with signal, as a real sensor's does."""
        return (img + rng.normal(0, 1, img.shape) * np.sqrt(np.maximum(img, 0)) * photon
                + rng.normal(0, read_noise, img.shape))

    stack = []
    for k in range(16):          # a normal sweep through board and body
        t = k / 15.0
        board_sigma, body_sigma = abs(t - 0.9) * 22.0, abs(t - 0.35) * 22.0
        b = board if board_sigma < 0.3 else cv2.GaussianBlur(board, (0, 0), board_sigma)
        prem, al = body * a3, body_a
        if body_sigma >= 0.3:
            prem = cv2.GaussianBlur(prem, (0, 0), body_sigma)
            al = cv2.GaussianBlur(body_a, (0, 0), body_sigma)
        stack.append(np.clip(shot(prem + b * (1.0 - al[:, :, None])), 0, 255).astype(np.uint8))

    veil = np.ones((size, size, 3), np.float32) * np.array([214.0, 226.0, 238.0])
    for k in range(frames - 16):  # focused far in front: everything veiled
        strength = 0.72 + 0.03 * k
        img = cv2.GaussianBlur(reference * (1.0 - strength) + veil * strength, (0, 0), 25.0)
        stack.append(np.clip(shot(img), 0, 255).astype(np.uint8))

    interior = cv2.erode(body_a, np.ones((41, 41), np.uint8)) > 0.5
    return stack, np.clip(reference, 0, 255).astype(np.uint8), interior


def _veil_patch_share(fused, reference, interior):
    """Percentage of the dark body rendered far brighter than it should be."""
    gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((gray - ref_gray > 45)[interior]) * 100.0)


def _wrong_frame_share(fused, stack, backdrop):
    """Percentage of the backdrop taken from the frame that is blurred there.

    Each output pixel is attributed to whichever source it lands closest to.
    Compositing blends neighbouring frames, but a pixel drawn from the wrong end
    of the stack is still far closer to that end than to the right one.
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


@pytest.fixture(scope="module")
def veiled():
    return _veil_stack()


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
# A frame that resolves nothing must not win on grain alone
# --------------------------------------------------------------------------

def test_veil_frames_do_not_take_the_smooth_body(veiled):
    """The measured failure covered 37.2% of the body; the fix leaves none."""
    stack, reference, interior = veiled
    fused = dct_focus_stack_fusion(list(stack))
    share = _veil_patch_share(fused, reference, interior)
    assert share < 5.0, (
        f"{share:.1f}% of the smooth dark body came from a bright veil frame - "
        "grain is buying blocks that hold no detail again (item 17)")


def test_detail_free_regions_do_not_cost_the_rest_of_the_picture(veiled):
    """
    Guard against the cheap way to pass the test above: refusing to select at
    all. The rim bristles and the board markings are real detail and must still
    be picked up, so the fused result has to beat every single frame outright.
    """
    stack, reference, _ = veiled
    fused = dct_focus_stack_fusion(list(stack))
    best_slice = max(_psnr(s, reference) for s in stack)
    assert _psnr(fused, reference) > best_slice + 1.0


def test_veil_frames_still_win_where_they_are_the_sharp_ones(veiled):
    """
    The veil frames are not blacklisted - nothing here may depend on *which*
    frame is suspect. Fusing the veil frames alone must still return a picture
    made of them, not a refusal to choose.
    """
    stack, _, _ = veiled
    veil_frames = list(stack[16:])
    fused = dct_focus_stack_fusion(veil_frames)
    assert fused.shape == stack[0].shape

    # DCT composites neighbouring frames rather than copying one verbatim, so
    # the contract is that every pixel stays inside the envelope its sources
    # span - the result is built from these frames, not from anything else.
    lo = np.min(np.stack(veil_frames), axis=0).astype(np.int16)
    hi = np.max(np.stack(veil_frames), axis=0).astype(np.int16)
    out = fused.astype(np.int16)
    outside = np.maximum(lo - out, 0) + np.maximum(out - hi, 0)
    assert outside.max() <= 1, (
        "output does not come from the frames it was given "
        f"(off the source envelope by {outside.max()} levels)")


# --------------------------------------------------------------------------
# The GPU twin must not drift from the CPU path on the same scene
# --------------------------------------------------------------------------

def _require_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("no GPU device available")


def test_gpu_path_agrees_on_the_wash(scene):
    _require_gpu()
    from fusion_methods.dct_torch import dct_torch_impl

    stack, _, backdrop = scene
    cpu = dct_focus_stack_fusion(list(stack))
    gpu = dct_torch_impl(list(stack))
    assert _psnr(gpu, cpu) > 40.0
    assert _wrong_frame_share(gpu, stack, backdrop) < 8.0


def test_gpu_path_agrees_on_the_veil(veiled):
    _require_gpu()
    from fusion_methods.dct_torch import dct_torch_impl

    stack, reference, interior = veiled
    cpu = dct_focus_stack_fusion(list(stack))
    gpu = dct_torch_impl(list(stack))
    assert _psnr(gpu, cpu) > 30.0
    assert _veil_patch_share(gpu, reference, interior) < 5.0
