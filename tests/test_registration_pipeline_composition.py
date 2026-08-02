"""A multi-stage pipeline must resample and crop the frames once, not once per stage.

Each stage used to be a complete registration - estimate, warp, crop - handing
its *pixels* to the next one. Resampling is not idempotent and the crops
compose, so `scale`+`homography`+`ECC` interpolated every frame three times and
took the intersection of the valid regions three times: 40% of the picture's
high-frequency energy gone where one stage costs 20%, and a seventh of the frame
thrown away. In a focus stacker that is the wrong currency to pay in - the
fusion stage picks each pixel from whichever frame is locally sharpest, and it
cannot see detail an earlier warp has already discarded. See
docs/REGISTRATION_IMPROVEMENTS.md item 7.

The stages now compose: each is still *measured* on the one before it, but their
matrices are multiplied together and the original frames are warped once, into
one crop. The bar:

1. Stages handed over together cost one interpolation, not one per stage
   (`TestOneResample`). Fails if the composition is dropped and the stages are
   chained internally again.
2. They cost one crop too (`TestOneCrop`).
3. The composition is arithmetically right: the stack ends at least as well
   aligned as chaining leaves it, against known geometric truth
   (`TestCompositionIsCorrect`). This is what fails if the conjugation through
   the intermediate canvas is dropped or inverted.
4. Each stage still measures on the previous stage's output pixels - the
   intermediate canvas is that output, bit for bit (`TestStagesStillSeeEachOther`).
   Fails if a later stage is quietly fed the raw frames instead.
5. A single stage is untouched by any of this (`TestSingleStageIsUnchanged`).
6. Method names, '+'-joined strings and sequences resolve to the same stages in
   pipeline order (`TestStageResolution`).

Run with:  python -m pytest tests/test_registration_pipeline_composition.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.registration as registration_module
from core.registration import ImageRegistration, resolve_stages
from tests.synthetic_stack import make_reference

# The pipeline the app defaults to, and the longest one it offers.
DEFAULT_PIPELINE = ["scale", "homography"]
FULL_PIPELINE = ["scale", "homography", "ecc"]


# --- input -------------------------------------------------------------------

def _base_scene(height=384, width=448, seed=5):
    """A feature-rich BGR frame both SIFT and ECC can lock onto."""
    img = make_reference(height=height, width=width, seed=seed, style="texture")
    for (cy, cx) in [(96, 112), (96, 336), (288, 112), (288, 336), (192, 224)]:
        cv2.circle(img, (cx, cy), 18, (30, 220, 30), -1)
    return img


def _drifting_stack(n=6, step=2.5, breathe=0.004, seed=5):
    """A stack that drifts and breathes by a known amount, with its truth.

    The motion is what a handheld focus stack produces - a uniform magnification
    about the centre plus a translation - so every stage has something real to
    find and none of them has perspective to invent.
    """
    base = _base_scene(seed=seed)
    h, w = base.shape[:2]
    frames, truth = [], []
    for i in range(n):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 1.0 + breathe * i)
        M[0, 2] += i * step
        M[1, 2] += i * step * 0.6
        frames.append(cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_REFLECT))
        truth.append(np.vstack([M, [0.0, 0.0, 1.0]]))
    return frames, truth


# --- how the two ways of running a pipeline are compared ---------------------

def _composed(frames, stages, width=512, thread_count=2):
    """The stages handed over together - one estimate chain, one warp."""
    return ImageRegistration(method=list(stages), downscale_width=width,
                             reference_index=0).process(
        [f.copy() for f in frames], output_path=None, thread_count=thread_count)


def _chained(frames, stages, width=512, thread_count=2):
    """One registration call per stage, i.e. what the code did before item 7."""
    images = [f.copy() for f in frames]
    for stage in stages:
        images = ImageRegistration(method=stage, downscale_width=width,
                                   reference_index=0).process(
            images, output_path=None, thread_count=thread_count)
    return images


def _gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img.astype(np.float32)


def _sharpness(stack, inset=24):
    """Mean high-frequency energy of a stack, away from the warp's borders.

    The measure item 4 is stated in and the one a selection-based fusion
    effectively asks each frame for, so it is what an extra interpolation costs
    in the currency the application cares about.
    """
    return float(np.mean([
        cv2.Laplacian(_gray(img)[inset:-inset, inset:-inset], cv2.CV_32F).var()
        for img in stack]))


def _kept(stack, frames):
    """Share of the original frame area that survived the crop."""
    h, w = frames[0].shape[:2]
    return stack[0].shape[0] * stack[0].shape[1] / float(h * w)


class _capture_maps:
    """The one map per frame a registration call finally applies.

    The pipeline computes its valid region once per stage, with the transforms
    composed so far, so the last call carries the whole map; earlier calls are
    the intermediate canvases the later estimators measured on.
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

    def total(self):
        H_matrices, T_crop = self.stages[-1]
        return [T_crop @ H for H in H_matrices]

    def composed_total(self):
        """What the maps would compose to if every stage warped and cropped."""
        composed = []
        for i in range(len(self.stages[0][0])):
            M = np.eye(3)
            for H_matrices, T_crop in self.stages:
                M = T_crop @ H_matrices[i] @ M
            composed.append(M)
        return composed


def _project(M, pts):
    q = np.asarray(M, dtype=np.float64) @ np.vstack([pts.T, np.ones(len(pts))])
    w = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    return (q[:2] / w).T


def _residual_px(estimated, truth, shape, step=8):
    """Misregistration in output pixels, against the known per-frame motion.

    A point p lands at truth_i(p) in frame i and registration maps it to
    estimated_i(truth_i(p)); perfect registration makes that one map for every
    frame, so the spread against frame 0 is the error - independent of how the
    result was cropped, which is what lets the two ways of running the pipeline
    be compared at all.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    reference = _project(estimated[0] @ truth[0], pts)
    return float(np.mean([
        np.sqrt(np.mean(np.sum(
            (_project(estimated[i] @ truth[i], pts) - reference) ** 2, axis=1)))
        for i in range(len(estimated))]))


# --- 1. one interpolation ----------------------------------------------------

class TestOneResample:
    """However many stages run, the output frames are resampled once."""

    def test_the_measurement_can_see_an_extra_interpolation(self):
        """Guards the guard: chaining has to cost something measurable here."""
        frames, _ = _drifting_stack()
        one = _sharpness(_composed(frames, ["scale"]))
        three = _sharpness(_chained(frames, FULL_PIPELINE))
        assert three < 0.9 * one, (
            f"three chained stages read {three:.1f} against {one:.1f} for one - "
            "this input cannot see the interpolation, so the assertions below "
            "prove nothing")

    @pytest.mark.parametrize("stages", [DEFAULT_PIPELINE, FULL_PIPELINE])
    def test_a_composed_pipeline_keeps_the_detail_a_chained_one_loses(self, stages):
        frames, _ = _drifting_stack()

        composed = _sharpness(_composed(frames, stages))
        chained = _sharpness(_chained(frames, stages))

        assert composed > 1.1 * chained, (
            f"{'+'.join(stages)}: composed {composed:.1f} against chained "
            f"{chained:.1f} - the stages are still resampling one after another")

    @pytest.mark.parametrize("stages", [DEFAULT_PIPELINE, FULL_PIPELINE])
    def test_it_costs_no_more_than_a_single_stage_does(self, stages):
        """The whole claim: the loss is one warp's worth, not one per stage."""
        frames, _ = _drifting_stack()

        one_stage = _sharpness(_composed(frames, ["scale"]))
        pipeline = _sharpness(_composed(frames, stages))

        assert pipeline > 0.95 * one_stage, (
            f"{'+'.join(stages)} reads {pipeline:.1f} against {one_stage:.1f} "
            f"for a single stage - it is paying for more than one resample")


# --- 2. one crop -------------------------------------------------------------

class TestOneCrop:
    """Every stage used to take the intersection of the valid regions again."""

    @pytest.mark.parametrize("stages", [DEFAULT_PIPELINE, FULL_PIPELINE])
    def test_a_composed_pipeline_keeps_more_of_the_frame(self, stages):
        frames, _ = _drifting_stack()

        composed = _kept(_composed(frames, stages), frames)
        chained = _kept(_chained(frames, stages), frames)

        assert composed > chained, (
            f"{'+'.join(stages)}: kept {100 * composed:.1f}% against "
            f"{100 * chained:.1f}% chained - the crops are still composing")

    @pytest.mark.parametrize("stages", [DEFAULT_PIPELINE, FULL_PIPELINE])
    def test_it_crops_about_as_much_as_a_single_stage(self, stages):
        frames, _ = _drifting_stack()

        one_stage = _kept(_composed(frames, ["scale"]), frames)
        pipeline = _kept(_composed(frames, stages), frames)

        # A later stage may still find a little more motion to crop away; what
        # it may not do is crop the previous stage's crop.
        assert pipeline > 0.97 * one_stage, (
            f"{'+'.join(stages)}: kept {100 * pipeline:.1f}% against "
            f"{100 * one_stage:.1f}% for one stage")

    def test_every_frame_comes_back_the_same_size(self):
        frames, _ = _drifting_stack()
        aligned = _composed(frames, FULL_PIPELINE)
        assert len({a.shape for a in aligned}) == 1


# --- 3. the composition is arithmetically right ------------------------------

class TestCompositionIsCorrect:
    """One warp is only worth having if it lands the frames where they belong."""

    @pytest.mark.parametrize("stages", [DEFAULT_PIPELINE, FULL_PIPELINE])
    def test_the_stack_is_at_least_as_well_aligned_as_when_chained(self, stages):
        frames, truth = _drifting_stack()
        shape = frames[0].shape[:2]

        with _capture_maps() as captured:
            _composed(frames, stages)
        composed = _residual_px(captured.total(), truth, shape)

        with _capture_maps() as captured:
            _chained(frames, stages)
        chained = _residual_px(captured.composed_total(), truth, shape)

        assert composed < max(1.2 * chained, chained + 0.05), (
            f"{'+'.join(stages)}: {composed:.3f} px composed against "
            f"{chained:.3f} px chained - the composition has lost the alignment")

    @pytest.mark.parametrize("stages", [DEFAULT_PIPELINE, FULL_PIPELINE])
    def test_the_pipeline_registers_the_stack(self, stages):
        """A composition that silently cancelled itself would still be smooth."""
        frames, truth = _drifting_stack()
        shape = frames[0].shape[:2]
        identity = [np.eye(3)] * len(frames)

        with _capture_maps() as captured:
            _composed(frames, stages)

        registered = _residual_px(captured.total(), truth, shape)
        unregistered = _residual_px(identity, truth, shape)
        assert unregistered > 3.0, "the fixture should carry real misalignment"
        assert registered < 0.25 * unregistered, (
            f"{'+'.join(stages)}: {registered:.2f} px after registration against "
            f"{unregistered:.2f} px before it")

    def test_the_reference_frame_is_the_fixed_point_of_the_composition(self):
        """Whatever the stages measure, the anchor's composed map is the identity.

        The conjugation through each intermediate canvas is what keeps this
        true; without it the crop translations accumulate into the map and the
        final crop is computed on a canvas that has drifted off the frame.
        """
        frames, _ = _drifting_stack()

        with _capture_maps() as captured:
            _composed(frames, FULL_PIPELINE)

        H_matrices, _T_crop = captured.stages[-1]
        assert np.allclose(H_matrices[0], np.eye(3), atol=1e-9)


# --- 4. the stages still measure on each other -------------------------------

class TestStagesStillSeeEachOther:
    """Composing the output must not change what the stages are measured on."""

    def test_the_second_stage_sees_the_first_stage_s_output(self):
        """Bit for bit: the intermediate canvas *is* the previous stage's result.

        This is what keeps every estimate in the pipeline what it was before the
        composition - the change is to where the pixels go, not to what the
        stages look at. A pipeline that fed the raw frames to the second stage
        would be cheaper and would fail here.
        """
        frames, _ = _drifting_stack()
        seen = []

        original = registration_module._STAGE_ESTIMATORS["homography"]

        def spy(images, downscale_width, thread_count, parallel_ecc=True):
            seen.append([np.array(img) for img in images])
            return original(images, downscale_width, thread_count, parallel_ecc)

        registration_module._STAGE_ESTIMATORS["homography"] = spy
        try:
            _composed(frames, DEFAULT_PIPELINE)
        finally:
            registration_module._STAGE_ESTIMATORS["homography"] = original

        assert len(seen) == 1
        scale_only = _composed(frames, ["scale"])
        assert len(seen[0]) == len(scale_only)
        for measured, standalone in zip(seen[0], scale_only):
            assert measured.shape == standalone.shape
            assert np.array_equal(measured, standalone)

    def test_every_stage_runs_exactly_once(self):
        frames, _ = _drifting_stack()
        calls = {name: 0 for name in FULL_PIPELINE}
        originals = dict(registration_module._STAGE_ESTIMATORS)

        def counted(name):
            def spy(images, downscale_width, thread_count, parallel_ecc=True):
                calls[name] += 1
                return originals[name](images, downscale_width, thread_count,
                                       parallel_ecc)
            return spy

        registration_module._STAGE_ESTIMATORS.update(
            {name: counted(name) for name in FULL_PIPELINE})
        try:
            _composed(frames, FULL_PIPELINE)
        finally:
            registration_module._STAGE_ESTIMATORS.update(originals)

        assert calls == {name: 1 for name in FULL_PIPELINE}


# --- 5. a single stage is untouched ------------------------------------------

class TestSingleStageIsUnchanged:
    """The composition is a no-op on one stage, and has to stay one."""

    @pytest.mark.parametrize("method", ["scale", "homography", "ecc"])
    def test_one_stage_composed_is_one_stage_chained(self, method):
        frames, _ = _drifting_stack()
        composed = _composed(frames, [method])
        chained = _chained(frames, [method])

        assert len(composed) == len(chained)
        for a, b in zip(composed, chained):
            assert a.shape == b.shape
            assert np.array_equal(a, b), f"{method} is no longer bit-identical"

    def test_a_stack_too_short_to_register_comes_back_untouched(self):
        frames, _ = _drifting_stack(n=1)
        out = _composed(frames, FULL_PIPELINE)
        assert len(out) == 1
        assert np.array_equal(out[0], frames[0])


# --- 6. naming ---------------------------------------------------------------

class TestStageResolution:
    """'both' is two stages, and stages come out in pipeline order."""

    @pytest.mark.parametrize("method,expected", [
        ("scale", ("scale",)),
        ("both", ("homography", "ecc")),
        ("scale+ecc", ("scale", "ecc")),
        ("scale+both", ("scale", "homography", "ecc")),
        (["ecc", "scale"], ("scale", "ecc")),
        (["scale", "both", "ecc"], ("scale", "homography", "ecc")),
        (None, ()),
    ])
    def test_it_resolves_to_the_stages_in_pipeline_order(self, method, expected):
        assert resolve_stages(method) == expected

    @pytest.mark.parametrize("method", ["zoom", "scale+zoom", ["scale", "zoom"]])
    def test_an_unknown_method_is_rejected(self, method):
        with pytest.raises(ValueError):
            ImageRegistration(method=method)

    def test_both_is_the_same_pipeline_as_naming_its_two_stages(self):
        frames, _ = _drifting_stack()
        by_alias = _composed(frames, ["both"])
        by_stages = _composed(frames, ["homography", "ecc"])
        for a, b in zip(by_alias, by_stages):
            assert np.array_equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
