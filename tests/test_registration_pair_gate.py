"""A pair is chained on the RANSAC verdict, not on the match count.

Both feature stages used to accept a pair as soon as six matches survived the
ratio test - two above the four a homography needs - and then discarded the
inlier mask RANSAC had produced. The pair sitting on that floor is the pair that
goes wrong, and because the chain accumulates (`H_global = H_global @ H_local`)
a bad estimate is not a bad frame, it is a permanent offset carried by every
frame after it.

The stages now read the verdict: enough inliers, enough of the matches offered,
and a matrix that is a motion a camera could have made. A pair failing any of
that takes the skip path the stages already had - the running trajectory is kept
and the reference features are not advanced, so the next frame is matched
against the last frame that fitted. The bar:

1. The gate accepts a sound fit and names the fault in an unsound one
   (`TestGateVerdict`).
2. The pair the audit found - six matches, 3.5x the residual of its neighbours -
   is no longer chained, and that frame comes out better for it
   (`TestSixMatchPair`).
3. A bad pair in the middle of a chain stops costing every frame after it
   (`TestMidChainRecovery`).
4. Stacks that were fitting fine still fit - the gate is a floor, not a filter
   (`TestSoundStacksAreUntouched`).

Run with:  python -m pytest tests/test_registration_pair_gate.py -v
"""

import contextlib
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.registration as registration_module
from core.registration import ImageRegistration, _pair_rejection_reason

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "samples")

SHAPE = (720, 1280)      # the frame the perspective terms are judged against
BAD_FRAME = 7            # the frame degraded in the mid-chain test


# --------------------------------------------------------------- geometry

def _project(M, pts):
    q = np.asarray(M, dtype=np.float64) @ np.vstack([pts.T, np.ones(len(pts))])
    w = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    return (q[:2] / w).T


def _mask(inliers, total):
    """The mask RANSAC returns: `inliers` ones followed by zeros."""
    m = np.zeros((total, 1), dtype=np.uint8)
    m[:inliers] = 1
    return m


def _similarity(scale=1.0, degrees=0.0, tx=0.0, ty=0.0):
    M = cv2.getRotationMatrix2D((SHAPE[1] / 2.0, SHAPE[0] / 2.0), degrees, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return np.vstack([M, [0.0, 0.0, 1.0]])


def _keystoned(fraction):
    """A homography whose perspective term bends the frame by `fraction`."""
    H = np.eye(3)
    H[2, 0] = fraction / SHAPE[1]
    return H


# ------------------------------------------------- reading a stage back

class _capture_transforms:
    """Record the transforms and crop a registration call finally applies.

    The pipeline calls _compute_valid_region_from_transforms once per stage,
    with the transforms composed so far and just before folding in the crop
    translation, so the last call carries the whole map.
    """

    def __enter__(self):
        self.stages = []
        self._original = registration_module._compute_valid_region_from_transforms

        def spy(H_matrices, img_shape, margin=2):
            top, bottom, left, right = self._original(H_matrices, img_shape, margin)
            T_crop = np.eye(3)
            if not (top >= bottom or left >= right):
                T_crop[0, 2] = -left
                T_crop[1, 2] = -top
            self.stages.append(
                ([np.array(H, dtype=np.float64) for H in H_matrices], T_crop))
            return top, bottom, left, right

        registration_module._compute_valid_region_from_transforms = spy
        return self

    def __exit__(self, *exc):
        registration_module._compute_valid_region_from_transforms = self._original

    def maps(self):
        H_matrices, T_crop = self.stages[-1]
        return [T_crop @ H for H in H_matrices]


def _register(frames, method="scale"):
    with _capture_transforms() as captured:
        ImageRegistration(method=method, downscale_width=1024,
                          reference_index=0).process([f.copy() for f in frames],
                                                     output_path=None, thread_count=2)
    return captured.maps()


_GATE_NAMES = ("_MIN_PAIR_INLIERS", "_MIN_PAIR_INLIER_RATIO",
               "_MAX_PAIR_SCALE_DRIFT", "_MAX_PAIR_KEYSTONE")
# What ships, read once before any test can move it.
_GATE_ON = {name: getattr(registration_module, name) for name in _GATE_NAMES}
# What the stages did before item 8: accept on six matches, ignore the verdict.
_GATE_OFF = {"_MIN_PAIR_INLIERS": 6, "_MIN_PAIR_INLIER_RATIO": 0.0,
             "_MAX_PAIR_SCALE_DRIFT": 1e9, "_MAX_PAIR_KEYSTONE": 1e9}


@contextlib.contextmanager
def _gate(enabled):
    """Run the stages with the acceptance gate in place, or as they were."""
    saved = {name: getattr(registration_module, name) for name in _GATE_NAMES}
    for name, value in (_GATE_ON if enabled else _GATE_OFF).items():
        setattr(registration_module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(registration_module, name, value)


def _per_frame_residual(estimated, truth, shape, step=8):
    """Misregistration in output pixels, against the known per-frame motion.

    A point p of the scene lands at truth_i(p) in frame i and registration maps
    it to estimated_i(truth_i(p)); perfect registration makes that one map for
    every frame, so the spread against frame 0 is the error, independent of how
    the result was cropped.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    reference = _project(estimated[0] @ truth[0], pts)
    return np.array([
        np.sqrt(np.mean(np.sum(
            (_project(estimated[i] @ truth[i], pts) - reference) ** 2, axis=1)))
        for i in range(len(estimated))])


def _load_frames(name):
    frame_dir = os.path.join(SAMPLES_DIR, name, "frames")
    names = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))
    return [cv2.imread(os.path.join(frame_dir, f), cv2.IMREAD_UNCHANGED) for f in names]


def _load_scene(name):
    """Frames plus the per-frame motion the sample generator recorded."""
    with open(os.path.join(SAMPLES_DIR, name, "scene.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    if not meta.get("per_frame_affine"):
        pytest.skip(f"{name} carries no per_frame_affine")
    truth = [np.vstack([np.array(m, dtype=np.float64), [0.0, 0.0, 1.0]])
             for m in meta["per_frame_affine"]]
    return _load_frames(name), truth


def _spoil(frames, index=BAD_FRAME, sigma=5.0, noise=6.0, seed=4):
    """Defocus and noise one frame hard enough that its matches collapse."""
    rng = np.random.default_rng(seed)
    out = [f.copy() for f in frames]
    blurred = cv2.GaussianBlur(out[index], (0, 0), sigma)
    out[index] = np.clip(blurred.astype(np.float32)
                         + rng.normal(0.0, noise, blurred.shape),
                         0, 255).astype(out[index].dtype)
    return out


needs_samples = pytest.mark.skipif(
    not os.path.exists(os.path.join(SAMPLES_DIR, "manifest.json")),
    reason="samples/ not generated; run `python samples/generate_samples.py`")


# ------------------------------------------------------------------ tests

class TestGateVerdict:
    """What the gate reads, and what it says when it refuses."""

    def test_a_sound_fit_is_accepted(self):
        H = _similarity(scale=1.004, degrees=0.3, tx=2.0, ty=-1.0)

        assert _pair_rejection_reason(H, _mask(38, 42), 42, SHAPE) is None

    def test_a_fit_nobody_could_estimate_is_rejected(self):
        assert _pair_rejection_reason(None, None, 40, SHAPE) == "no transform"

    def test_too_few_inliers_is_rejected_however_many_matches_were_offered(self):
        """The audit's pair: RANSAC agreed with everything, and that was six."""
        reason = _pair_rejection_reason(_similarity(), _mask(6, 6), 6, SHAPE)

        assert reason == "6 inliers"

    def test_a_fit_ransac_half_believes_is_rejected(self):
        """Enough inliers in absolute terms, but a minority of the matches."""
        reason = _pair_rejection_reason(_similarity(), _mask(14, 60), 60, SHAPE)

        assert reason == "14/60 inliers"

    def test_a_reflection_is_rejected(self):
        flip = np.diag([-1.0, 1.0, 1.0])

        assert "determinant" in _pair_rejection_reason(flip, _mask(40, 44), 44, SHAPE)

    def test_an_implausible_magnification_is_rejected(self):
        """Focus breathing is a fraction of a percent per pair, not a third."""
        reason = _pair_rejection_reason(_similarity(scale=1.35), _mask(40, 44), 44, SHAPE)

        assert reason is not None and reason.startswith("scale")

    def test_a_keystoned_frame_is_rejected(self):
        reason = _pair_rejection_reason(_keystoned(0.06), _mask(40, 44), 44, SHAPE)

        assert reason is not None and reason.startswith("keystone")

    def test_the_perspective_a_real_tilt_produces_is_kept(self):
        """The bound has to sit above the keystone a camera actually makes."""
        assert _pair_rejection_reason(_keystoned(0.008), _mask(40, 44), 44, SHAPE) is None

    def test_a_non_finite_transform_is_rejected(self):
        H = _similarity()
        H[0, 2] = np.inf

        assert _pair_rejection_reason(H, _mask(40, 44), 44, SHAPE) == "non-finite transform"

    def test_no_mask_leaves_the_matrix_to_speak_for_itself(self):
        """An estimator that returns no verdict must not veto by silence."""
        assert _pair_rejection_reason(_similarity(1.003), None, 40, SHAPE) is None
        assert _pair_rejection_reason(_keystoned(0.06), None, 40, SHAPE) is not None


@needs_samples
class TestSixMatchPair:
    """handheld_drift's last pair: six matches, 4.5 px, 5% of keystone."""

    @pytest.mark.parametrize("method", ["scale", "homography"])
    def test_the_pair_is_not_chained(self, method):
        frames, _truth = _load_scene("handheld_drift")

        maps = _register(frames, method)

        # Skipped means the frame keeps the frame before it's trajectory - the
        # same composed map, not merely a similar one.
        assert np.allclose(maps[-1], maps[-2], atol=1e-9), (
            "the six-match pair was chained anyway")

    @pytest.mark.parametrize("method", ["scale", "homography"])
    def test_dropping_it_improves_the_frame_it_would_have_moved(self, method):
        frames, truth = _load_scene("handheld_drift")
        shape = frames[0].shape[:2]

        with _gate(False):
            before = _per_frame_residual(_register(frames, method), truth, shape)
        with _gate(True):
            after = _per_frame_residual(_register(frames, method), truth, shape)

        # Measured: 5.60 -> 4.66 px on the frame itself, 2.22 -> 2.16 px over
        # the stack. Every earlier frame is untouched, which is the other half.
        assert after[-1] < before[-1] * 0.95
        assert after.mean() < before.mean()
        assert np.allclose(after[:-1], before[:-1], atol=1e-6)


@needs_samples
class TestMidChainRecovery:
    """The cost of a bad pair is not the frame, it is everything after it."""

    def test_frames_after_a_bad_pair_stop_paying_for_it(self):
        frames, truth = _load_scene("handheld_drift")
        shape = frames[0].shape[:2]
        spoiled = _spoil(frames)
        tail = slice(BAD_FRAME, None)

        with _gate(False):
            ungated_tail = _per_frame_residual(_register(spoiled), truth, shape)[tail]
        with _gate(True):
            gated_tail = _per_frame_residual(_register(spoiled), truth, shape)[tail]
            intact_tail = _per_frame_residual(_register(frames), truth, shape)[tail]

        # Measured: 3.92 px from the degraded frame on, against 3.41 px with the
        # gate, where the stack that was never degraded scores 3.18 px. The
        # rejected pair costs the chain a little; chaining it costs it a lot.
        assert gated_tail.mean() < ungated_tail.mean() * 0.95
        assert gated_tail.mean() < intact_tail.mean() * 1.15


@needs_samples
class TestSoundStacksAreUntouched:
    """A floor that rejects sound pairs would silently stop registering."""

    @pytest.mark.parametrize("scene", ["flower01_handheld", "flower01_subject_hires"])
    @pytest.mark.parametrize("method", ["scale", "homography"])
    def test_no_pair_of_a_clean_stack_is_rejected(self, scene, method):
        frames = _load_frames(scene)

        with _gate(False):
            before = _register(frames, method)
        with _gate(True):
            after = _register(frames, method)

        assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(before, after)), (
            f"{scene}/{method}: the gate rejected a pair that was fitting fine")

    @pytest.mark.parametrize("method", ["scale", "homography"])
    def test_consecutive_frames_are_never_left_on_top_of_each_other(self, method):
        """Every accepted pair still moves its frame - the gate is not a no-op."""
        frames = _load_frames("flower01_handheld")

        maps = _register(frames, method)

        assert not any(np.allclose(maps[i], maps[i - 1], atol=1e-9)
                       for i in range(1, len(maps)))
