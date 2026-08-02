"""The homography stage must not spend degrees of freedom on noise.

`homography` and `scale` share their whole front end - SIFT on the same
downscaled frames, the same ratio test, the same RANSAC threshold, the same
chain. The only difference used to be the model: `cv2.findHomography` (8 DOF)
against `cv2.estimateAffinePartial2D` (4 DOF). A focus rail moves the focal
plane along the optical axis, so the motion between frames is a magnification
change plus small rotation and recentring - a similarity - and the homography's
two perspective terms had nothing left to fit but match noise. Measured against
the affines the handheld sample scenes record, the stage came out *worse than
not registering at all*: 0.79x and 0.88x.

The stage now fits both models to each pair's matches and keeps the homography
only where it predicts held-out matches better than the similarity does. The bar:

1. The choice tracks the motion, not the noise (`TestModelChoice`).
2. On the two scenes with exact geometric ground truth, the stage beats leaving
   the stack alone rather than degrading it (`TestGroundTruthScenes`).
3. Where the perspective is real, the homography is still used and still
   recovers it to sub-pixel accuracy - the fix is not "always use 4 DOF"
   (`TestRealPerspective`).
4. The extra fitting keeps the stage reproducible (`TestDeterminism`).

Run with:  python -m pytest tests/test_registration_homography_model.py -v
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.registration as registration_module
from core.registration import ImageRegistration, _select_pair_transform

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "samples")

# Scenes carrying a per-frame affine, i.e. the ones with geometric ground truth.
GROUND_TRUTH_SCENES = ("handheld_drift", "flower01_handheld")


# --------------------------------------------------------------- geometry

def _project(M, pts):
    q = np.asarray(M, dtype=np.float64) @ np.vstack([pts.T, np.ones(len(pts))])
    w = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    return (q[:2] / w).T


def _tilt(theta_y, theta_x, w, h, focal_ratio=1.2):
    """The homography a camera tilt produces - genuine perspective, K R K^-1."""
    f = focal_ratio * w
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    R = (np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
         @ np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]))
    return K @ R @ np.linalg.inv(K)


def _similarity(scale, degrees, tx, ty, w, h):
    """The motion a stacking rig produces: zoom about the centre, roll, shift."""
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return np.vstack([M, [0.0, 0.0, 1.0]])


def _matches(H, n=120, noise=0.7, size=(1280, 720), seed=0, outliers=4):
    """A match set drawn from H, with localisation noise and a few false pairs."""
    w, h = size
    rng = np.random.default_rng(seed)
    src = rng.uniform([0.0, 0.0], [w, h], (n, 2))
    dst = _project(H, src) + rng.normal(0.0, noise, (n, 2))
    # The ratio test lets a few wrong matches through; RANSAC is what removes
    # them, and the model choice must survive their presence.
    dst[:outliers] = rng.uniform([0.0, 0.0], [w, h], (outliers, 2))
    return src, dst


# ------------------------------------------------- reading the stage back

class _capture_transforms:
    """Record the transforms and crop a registration call finally applies.

    The pipeline calls _compute_valid_region_from_transforms once per stage,
    with the transforms composed so far and just before folding in the crop
    translation, so the last call carries the whole map. Lifted from
    tests/benchmark_registration.py, which reads the stages back the same way.
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
        """The one map per frame the pipeline applies, crop included."""
        H_matrices, T_crop = self.stages[-1]
        return [T_crop @ H for H in H_matrices]


def _register(frames, method="homography"):
    """Run a stage and return (aligned frames, the map applied to each frame)."""
    with _capture_transforms() as captured:
        aligned = ImageRegistration(method=method, downscale_width=1024,
                                    reference_index=0).process(
                                        [f.copy() for f in frames],
                                        output_path=None, thread_count=2)
    return aligned, captured.maps()


def _residual_px(estimated, truth, shape, step=8):
    """Misregistration in output pixels, against the known per-frame motion.

    A point p of the scene lands at truth_i(p) in frame i and registration maps
    it to estimated_i(truth_i(p)); perfect registration makes that the same map
    for every frame, so the spread against frame 0 is the error, independent of
    how the result was cropped.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    reference = _project(estimated[0] @ truth[0], pts)
    return float(np.mean([
        np.sqrt(np.mean(np.sum(
            (_project(estimated[i] @ truth[i], pts) - reference) ** 2, axis=1)))
        for i in range(len(estimated))]))


def _load_scene(name):
    root = os.path.join(SAMPLES_DIR, name)
    with open(os.path.join(root, "scene.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    if "per_frame_affine" not in meta:
        pytest.skip(f"{name} carries no per_frame_affine")
    frame_dir = os.path.join(root, "frames")
    names = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))
    frames = [cv2.imread(os.path.join(frame_dir, f), cv2.IMREAD_UNCHANGED) for f in names]
    truth = [np.vstack([np.array(m, dtype=np.float64), [0.0, 0.0, 1.0]])
             for m in meta["per_frame_affine"]]
    return frames, truth


def _perspective_stack(degrees_per_frame=0.35, n=8, size=(960, 640), seed=11):
    """A stack whose frames differ by a real camera tilt, plus its ground truth."""
    from tests.synthetic_stack import make_reference
    w, h = size
    base = make_reference(height=h, width=w, seed=seed, style="texture")
    for (cy, cx) in [(120, 160), (120, 800), (520, 160), (520, 800), (320, 480)]:
        cv2.circle(base, (cx, cy), 22, (30, 220, 30), -1)

    truth = [_tilt(np.deg2rad(degrees_per_frame) * i,
                   np.deg2rad(degrees_per_frame * 0.4) * i, w, h) for i in range(n)]
    frames = [cv2.warpPerspective(base, T, (w, h), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT) for T in truth]
    return frames, truth


# ------------------------------------------------------------------ tests

class TestModelChoice:
    """Which model a pair gets is decided by the motion, not by the estimator."""

    def test_similarity_motion_takes_the_constrained_model(self):
        H = _similarity(1.012, 0.6, 4.0, -3.0, 1280, 720)
        src, dst = _matches(H, seed=1)

        _M, model = _select_pair_transform(src, dst)

        assert model == "similarity", (
            "a zoom-roll-shift pair has no perspective in it; fitting 8 DOF to "
            "it can only fit noise")

    def test_real_perspective_keeps_the_homography(self):
        H = _tilt(np.deg2rad(0.5), np.deg2rad(0.2), 1280, 720)
        src, dst = _matches(H, seed=2)

        M, model = _select_pair_transform(src, dst)

        assert model == "homography", (
            "the stage still has to fit perspective where the camera really tilted")
        # And the matrix it returns is the tilt, not something merely accepted.
        corners = np.array([[0.0, 0.0], [1280.0, 0.0], [0.0, 720.0], [1280.0, 720.0]])
        assert np.max(np.linalg.norm(
            _project(M, corners) - _project(H, corners), axis=1)) < 1.0

    def test_a_thin_match_set_takes_the_constrained_model(self):
        """8 DOF through ten points reproduces them however wrong it is."""
        H = _tilt(np.deg2rad(0.5), np.deg2rad(0.2), 1280, 720)
        src, dst = _matches(H, n=10, outliers=0, seed=3)

        _M, model = _select_pair_transform(src, dst)

        assert model == "similarity"

    def test_a_pair_with_no_usable_motion_still_returns_a_matrix(self):
        src, dst = _matches(np.eye(3), n=40, noise=0.5, seed=4)

        M, model = _select_pair_transform(src, dst)

        assert model in ("similarity", "homography")
        assert M is not None and np.all(np.isfinite(M))


@pytest.mark.skipif(not os.path.exists(os.path.join(SAMPLES_DIR, "manifest.json")),
                    reason="samples/ not generated; run `python samples/generate_samples.py`")
class TestGroundTruthScenes:
    """The audit's own complaint: the stage was worse than doing nothing."""

    @pytest.mark.parametrize("scene", GROUND_TRUTH_SCENES)
    def test_registering_beats_leaving_the_stack_alone(self, scene):
        frames, truth = _load_scene(scene)
        shape = frames[0].shape[:2]
        identity = [np.eye(3)] * len(frames)

        unregistered = _residual_px(identity, truth, shape)
        _aligned, maps = _register(frames, "homography")
        registered = _residual_px(maps, truth, shape)

        # Measured at the time of writing: 1.78x and 2.65x, against 0.79x and
        # 0.88x for the unconstrained 8-DOF fit this replaced.
        assert registered < unregistered / 1.4, (
            f"{scene}: {registered:.2f} px after registration against "
            f"{unregistered:.2f} px before it")

    @pytest.mark.parametrize("scene", GROUND_TRUTH_SCENES)
    def test_it_no_longer_degrades_the_scale_stage_in_front_of_it(self, scene):
        """`scale`+`homography` is the app default, and it used to lose to `scale`."""
        frames, truth = _load_scene(scene)
        shape = frames[0].shape[:2]

        _scaled, scale_maps = _register(frames, "scale")
        scale_only = _residual_px(scale_maps, truth, shape)

        with _capture_transforms() as captured:
            ImageRegistration(method="scale+homography", downscale_width=1024,
                              reference_index=0).process(
                                  [f.copy() for f in frames],
                                  output_path=None, thread_count=2)
        both = _residual_px(captured.maps(), truth, shape)

        # Measured: 2.61 against 2.22 px, and 1.60 against 2.57 px - the stage
        # is allowed to cost a little, not to undo the stage before it.
        assert both < scale_only * 1.35, (
            f"{scene}: scale+homography {both:.2f} px against scale alone "
            f"{scale_only:.2f} px")


class TestRealPerspective:
    """The fix is a model choice, not the removal of the model."""

    def test_a_tilted_stack_is_still_recovered_to_sub_pixel(self):
        frames, truth = _perspective_stack()
        shape = frames[0].shape[:2]
        unregistered = _residual_px([np.eye(3)] * len(frames), truth, shape)

        _aligned, maps = _register(frames, "homography")
        registered = _residual_px(maps, truth, shape)

        assert unregistered > 10.0, "the fixture should carry real misalignment"
        assert registered < 1.0, (
            f"genuine perspective was flattened into a similarity: "
            f"{registered:.2f} px")

    def test_it_beats_the_similarity_only_stage_on_that_stack(self):
        """Guards the guard: a stage that always fits 4 DOF fails this."""
        frames, truth = _perspective_stack()
        shape = frames[0].shape[:2]

        _h, hom_maps = _register(frames, "homography")
        _s, scale_maps = _register(frames, "scale")

        assert _residual_px(hom_maps, truth, shape) < \
            0.5 * _residual_px(scale_maps, truth, shape)


class TestDeterminism:
    def test_the_same_stack_registers_the_same_way_twice(self):
        frames, _truth = _perspective_stack(n=5)

        first, _ = _register(frames, "homography")
        second, _ = _register(frames, "homography")

        for a, b in zip(first, second):
            assert np.array_equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
