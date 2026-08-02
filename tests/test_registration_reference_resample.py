"""The reference frame must be resampled like every other frame.

After re-referencing, the anchor frame's transform is exactly the identity, and
the crop translation folded in behind it used to be an integer offset. A warp by
an integer translation is a copy - every interpolation weight collapses to 1 -
so the anchor came out of registration untouched while every other frame was
softened by one interpolation.

That is an artificial sharpness advantage handed to one frame, and
selection-based fusion decides everything by asking which frame is locally
sharpest. It read the advantage as focus and sourced far more of the picture
from the anchor than belongs there: on the two ground-truth scenes, up to 68% of
the output came from a frame that is genuinely sharpest on under 3% of it - see
docs/REGISTRATION_IMPROVEMENTS.md item 3.

The fix folds a common quarter-pixel offset into the crop translation, so the
anchor goes through the same interpolator as the rest. Being common to every
frame, it cannot move them relative to each other.

The bar this has to clear:

1. The crop transform is a pure translation that no longer lands on the pixel
   grid, and its offset is the same for every frame (`TestCropTransform`).
2. On a stack whose frames carry identical content, the anchor comes out neither
   sharper nor softer than the frames it anchors - the assertion is two-sided,
   because over-correcting merely inverts the bias (`TestReferenceIsResampled`).
3. Disabling the offset brings the bias straight back, so the test fails in both
   directions (`TestReferenceIsResampled`).
4. Registration accuracy is unchanged - a shift shared by every frame is not a
   misalignment (`TestAlignmentIsUnaffected`).
5. On the real ground-truth scenes, the share of the picture the focus measure
   takes from the anchor comes back down to what is genuinely sharpest there
   (`TestSelectionOnGroundTruthScenes`, skipped until samples/ is generated).

Run with:  python -m pytest tests/test_registration_reference_resample.py -v
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.registration as registration_module
from core.registration import ImageRegistration, _build_crop_transform
from tests.synthetic_stack import make_reference

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "samples")

# The scenes carrying a focus_index, i.e. a per-pixel answer to "which frame is
# genuinely sharpest here".
GROUND_TRUTH_SCENES = ("handheld_drift", "flower01_handheld")


# --- a stack in which no frame is genuinely sharper than any other -----------

def _base_scene(size=512, seed=5):
    """A feature-rich BGR frame SIFT can lock onto reliably."""
    img = make_reference(height=size, width=size, seed=seed, style="texture")
    for (cy, cx) in [(120, 120), (120, 392), (392, 120), (392, 392), (256, 256)]:
        cv2.circle(img, (cx, cy), 20, (30, 220, 30), -1)
    return img


def _uniform_stack(size=512, n=7, step=2.5, seed=5):
    """One scene, displaced frame by frame, every frame resampled once.

    Frame 0 is offset by a fraction of a pixel like all the rest, so every frame
    enters registration having paid for exactly one interpolation, and the
    content is identical throughout. Any sharpness difference on the way out is
    therefore registration's doing and nothing else's.
    """
    base = _base_scene(size=size, seed=seed)
    stack = []
    for i in range(n):
        shift = (i + 0.37) * step
        M = np.array([[1.0, 0.0, shift], [0.0, 1.0, 0.6 * shift]], dtype=np.float32)
        stack.append(cv2.warpAffine(base, M, (size, size),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_REFLECT))
    return stack


# --- metrics -----------------------------------------------------------------

def _gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img.astype(np.float32)


def _sharpness(img, inset=24):
    """Laplacian variance away from the borders, where the warp runs out of data."""
    core = _gray(img)[inset:-inset, inset:-inset]
    return float(cv2.Laplacian(core, cv2.CV_32F).var())


def _reference_advantage(aligned, ref):
    """How much sharper the anchor reads than the mean of the frames it anchors.

    Zero is the honest answer on a stack of identical content. This is the
    robust measurement of the defect: unlike the selection share below, it does
    not depend on an argmax between frames that are nearly tied.
    """
    others = np.mean([_sharpness(a) for i, a in enumerate(aligned) if i != ref])
    return _sharpness(aligned[ref]) / others - 1.0


def _focus_share(aligned, ref, kernel=11, inset=24):
    """Fraction of the picture a selection-based fusion would take from the anchor.

    Pooled |Laplacian| argmax is the decision depthmap, pyramid and the rest all
    make. Meaningful on a scene with real focus variation; on the synthetic
    stack above, where every frame is tied, it is a knife-edge and only its
    gross behaviour is asserted.
    """
    maps = []
    for img in aligned:
        response = np.abs(cv2.Laplacian(_gray(img), cv2.CV_32F, ksize=3))
        maps.append(cv2.boxFilter(response, -1, (kernel, kernel))[inset:-inset, inset:-inset])
    picked = np.argmax(np.stack(maps, axis=0), axis=0)
    return float(np.mean(picked == ref))


def _mean_abs_diff(a, b):
    return float(np.mean(np.abs(_gray(a) - _gray(b))))


def _register(stack, ref, method="scale", offset=None, monkeypatch=None):
    """Run a stage, optionally with the reference-resample offset overridden."""
    if offset is not None:
        monkeypatch.setattr(registration_module, "_REFERENCE_RESAMPLE_OFFSET", offset)
    return ImageRegistration(method=method, downscale_width=512,
                             reference_index=ref).process(
        [f.copy() for f in stack], output_path=None, thread_count=2)


# --- the crop transform itself ----------------------------------------------

class TestCropTransform:
    def test_it_is_a_pure_translation(self):
        T = _build_crop_transform(-13, 7)
        assert np.allclose(T[:2, :2], np.eye(2))
        assert np.allclose(T[2], [0, 0, 1])

    def test_the_translation_is_off_the_pixel_grid(self):
        # An integral translation is a copy; that is the whole defect.
        T = _build_crop_transform(-13, 7)
        for value in (T[0, 2], T[1, 2]):
            assert abs(value - round(float(value))) > 1e-3, \
                "the crop translation still lands on the pixel grid"

    def test_it_stays_off_the_grid_without_a_crop(self):
        # A degenerate valid region skips the crop, but the anchor must still be
        # resampled - the offset is not conditional on there being a crop.
        T = _build_crop_transform(0, 0)
        assert abs(T[0, 2] - round(float(T[0, 2]))) > 1e-3
        assert abs(T[1, 2] - round(float(T[1, 2]))) > 1e-3

    def test_the_offset_is_identical_for_every_frame(self):
        # Two different crops must differ by exactly their integer offsets, so
        # the shift cancels out of any frame-to-frame comparison.
        a = _build_crop_transform(-13, 7)
        b = _build_crop_transform(4, -2)
        assert np.allclose(a[:, 2] - b[:, 2], [-17, 9, 0])

    def test_a_zero_offset_restores_the_old_integer_crop(self, monkeypatch):
        monkeypatch.setattr(registration_module, "_REFERENCE_RESAMPLE_OFFSET", 0.0)
        T = _build_crop_transform(-13, 7)
        assert np.allclose(T[:, 2], [-13, 7, 1])


# --- end to end on a stack with nothing to choose between frames -------------

class TestReferenceIsResampled:
    @pytest.mark.parametrize("ref", [0, 3])
    @pytest.mark.parametrize("method", ["scale", "homography"])
    def test_the_anchor_is_neither_sharper_nor_softer(self, ref, method):
        # Two-sided on purpose. A half-pixel offset would pass a one-sided
        # "not sharper" test while leaving the anchor 22-72% softer than the
        # stack, which inverts the bias rather than removing it.
        aligned = _register(_uniform_stack(n=7), ref, method=method)

        advantage = _reference_advantage(aligned, ref)
        assert abs(advantage) < 0.10, (
            f"the reference frame reads {advantage:+.1%} against the frames it "
            f"anchors, on a stack where every frame carries identical content"
        )

    def test_disabling_the_offset_brings_the_bias_back(self, monkeypatch):
        # The other direction: with the anchor special-cased into a copy again,
        # the measurement above must fail. This is what separates the fix from
        # its own quiet removal.
        stack = _uniform_stack(n=7)
        fixed = _register(stack, 0)
        unfixed = _register(stack, 0, offset=0.0, monkeypatch=monkeypatch)

        assert _reference_advantage(unfixed, 0) > 0.20
        assert _reference_advantage(fixed, 0) < _reference_advantage(unfixed, 0)
        # The tied-frame argmax is a knife-edge, so only the gross direction is
        # asserted: an unresampled anchor takes most of a stack it should split.
        assert _focus_share(unfixed, 0) > 0.40
        assert _focus_share(fixed, 0) < _focus_share(unfixed, 0)


class TestAlignmentIsUnaffected:
    def test_the_frames_still_register_onto_the_anchor(self):
        stack = _uniform_stack(n=7)
        ref = 3
        aligned = _register(stack, ref)

        assert len({a.shape for a in aligned}) == 1
        worst = max(_mean_abs_diff(aligned[ref], a)
                    for i, a in enumerate(aligned) if i != ref)
        assert worst < 8.0, f"a frame still drifts from its reference (worst={worst:.2f})"

    def test_the_offset_costs_no_alignment_accuracy(self, monkeypatch):
        # A shift shared by every frame is not a misalignment: registering with
        # and without it must be equally well aligned, and must crop the same.
        stack = _uniform_stack(n=7)
        ref = 3
        fixed = _register(stack, ref)
        unfixed = _register(stack, ref, offset=0.0, monkeypatch=monkeypatch)

        assert fixed[0].shape == unfixed[0].shape

        def worst(aligned):
            return max(_mean_abs_diff(aligned[ref], a)
                       for i, a in enumerate(aligned) if i != ref)

        assert worst(fixed) < worst(unfixed) + 1.0


# --- the scenes that know which frame is really sharpest ---------------------

def _load_scene(name):
    root = os.path.join(SAMPLES_DIR, name)
    frame_dir = os.path.join(root, "frames")
    index_path = os.path.join(root, "ground_truth", "focus_index.png")
    if not os.path.isdir(frame_dir) or not os.path.exists(index_path):
        pytest.skip(f"{name} has no focus_index ground truth")

    names = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))
    frames = [cv2.imread(os.path.join(frame_dir, f), cv2.IMREAD_UNCHANGED) for f in names]
    focus_index = cv2.imread(index_path, cv2.IMREAD_UNCHANGED)
    return frames, focus_index


@pytest.mark.skipif(not os.path.exists(os.path.join(SAMPLES_DIR, "manifest.json")),
                    reason="samples/ not generated; run `python samples/generate_samples.py`")
class TestSelectionOnGroundTruthScenes:
    """The measurement item 3 is stated in: which frame the fusion actually picks."""

    @pytest.mark.parametrize("scene", GROUND_TRUTH_SCENES)
    def test_the_anchor_is_picked_about_as_often_as_it_deserves(self, scene):
        frames, focus_index = _load_scene(scene)
        truth = float(np.mean(focus_index == 0))

        aligned = ImageRegistration(method="scale", downscale_width=1024,
                                    reference_index=0).process(
            [f.copy() for f in frames], output_path=None, thread_count=2)

        share = _focus_share(aligned, 0)
        # Pre-fix this was 11.7% and 14.7% on the CPU warp path and 53.6% and
        # 57.2% on the GPU one, against a truth of 2.9% and 0.0%.
        assert share < truth + 0.05, (
            f"{scene}: {share:.1%} of the picture is sourced from the anchor "
            f"frame, which is genuinely sharpest on {truth:.1%} of it"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
