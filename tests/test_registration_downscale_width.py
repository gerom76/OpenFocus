"""The detection width the caller asks for is the one that gets used.

All three stages measure their transform on a downscaled copy of the frames and
apply it at full resolution, so `downscale_width` trades detection accuracy
against time. It used to trade nothing: each stage carried

    max_dim = max(h_orig, w_orig)
    if max_dim >= 2048:
        downscale_width = 1024

which is an assignment rather than a cap, on a threshold every frame from every
camera made this century clears. Raising the setting to 2048 for a difficult
stack and lowering it to 512 for speed both produced byte-identical output - the
dialog, the `reg_downscale_width` config key and the argument threaded through
`ImageRegistration`, `RenderWorker` and `BatchWorker` were all inert on real
input. See docs/REGISTRATION_IMPROVEMENTS.md item 5.

The bar this has to clear:

1. The resolver returns what it was handed, at any frame size, and falls back to
   the shared default only for a missing or nonsensical value
   (`TestResolveDetectionWidth`).
2. On a frame over the old threshold, every stage detects at the width it was
   given - including a width large enough that no downscaling happens at all,
   which the override made unreachable (`TestStagesHonourTheSetting`).
3. Two different settings on such a frame produce different output, which is the
   defect stated as a user would meet it (`TestTheSettingChangesTheResult`).
4. The default is one value shared by the stages, `ImageRegistration` and the
   app's config, so a user who has not touched the setting sees no change from
   the override going away (`TestTheDefaultIsUnchanged`).
5. Registration still aligns a large frame at any setting, from a coarse
   detection width up to the full resolution the override used to forbid
   (`TestLargeFramesStillRegister`).

Reinstating the override fails 2, 3 and 5.

Run with:  python -m pytest tests/test_registration_downscale_width.py -v
"""

import inspect
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.registration as registration_module
from core.registration import (
    DEFAULT_DETECTION_WIDTH,
    ImageRegistration,
    _align_ecc_impl,
    _align_homography_impl,
    _align_scale_impl,
    _detection_scale,
    _resolve_detection_width,
)
from tests.synthetic_stack import make_reference

# Over the 2048 threshold on the longer side - which is what the override
# triggered on - but short enough that SIFT and ECC stay cheap.
WIDE = 2176
TALL = 544


def _wide_stack(n=4, step=3.0, seed=11):
    """A stack wider than the old threshold, drifting by a known translation."""
    base = make_reference(height=TALL, width=WIDE, seed=seed, style="texture")
    stack = []
    for i in range(n):
        shift = (i + 0.4) * step
        M = np.array([[1.0, 0.0, shift], [0.0, 1.0, 0.5 * shift]], dtype=np.float32)
        stack.append(cv2.warpAffine(base, M, (WIDE, TALL),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_REFLECT))
    return stack


class _record_detection_widths:
    """Capture the width each stage actually detects at.

    Every stage reaches its detection resolution through `_detection_scale`, so
    wrapping it records the size the frames were resampled to before SIFT or ECC
    ever saw them - the number the setting is supposed to control. An override
    reintroduced anywhere upstream of that call shows up here.
    """

    def __enter__(self):
        self.widths = set()
        self._original = registration_module._detection_scale

        def spy(width, downscale_width):
            scale = self._original(width, downscale_width)
            self.widths.add(int(round(width * scale)))
            return scale

        registration_module._detection_scale = spy
        return self

    def __exit__(self, *exc):
        registration_module._detection_scale = self._original


def _register(stack, method, width, thread_count=2):
    return ImageRegistration(method=method, downscale_width=width,
                             reference_index=0).process(
        [f.copy() for f in stack], output_path=None, thread_count=thread_count)


# --- the resolver ------------------------------------------------------------

class TestResolveDetectionWidth:
    @pytest.mark.parametrize("width", [256, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("frame", [(480, 640), (1430, 2560), (4000, 6000)])
    def test_it_returns_what_it_was_given(self, width, frame):
        # The whole item: the answer must not depend on the frame size.
        h, w = frame
        assert _resolve_detection_width(width, h, w, "test") == width

    @pytest.mark.parametrize("bad", [None, 0, -1, "", "wide"])
    def test_a_missing_or_nonsensical_value_falls_back_to_the_default(self, bad):
        assert _resolve_detection_width(bad, 1430, 2560, "test") == DEFAULT_DETECTION_WIDTH

    def test_a_string_number_is_accepted(self):
        assert _resolve_detection_width("768", 1430, 2560, "test") == 768

    def test_the_scale_never_upsamples(self):
        # A width above the frame means full-resolution detection, not a stretch.
        assert _detection_scale(2560, 4096) == 1.0
        assert _detection_scale(2560, 2560) == 1.0
        assert _detection_scale(2560, 1280) == pytest.approx(0.5)


# --- end to end, on a frame the override used to fire on ---------------------

class TestStagesHonourTheSetting:
    @pytest.mark.parametrize("method", ["scale", "homography", "ecc"])
    @pytest.mark.parametrize("width", [640, 1360])
    def test_the_stage_detects_at_the_width_it_was_given(self, method, width):
        stack = _wide_stack()
        with _record_detection_widths() as observed:
            _register(stack, method, width)
        assert observed.widths == {width}, (
            f"{method} detected at {sorted(observed.widths)} px on a {WIDE} px "
            f"frame, having been asked for {width} px"
        )

    @pytest.mark.parametrize("method", ["scale", "homography", "ecc"])
    def test_full_resolution_detection_is_reachable(self, method):
        # The override made this impossible: any frame over 2048 px was detected
        # at 1024 whatever the caller asked for.
        stack = _wide_stack()
        with _record_detection_widths() as observed:
            _register(stack, method, WIDE)
        assert observed.widths == {WIDE}


class TestTheSettingChangesTheResult:
    def test_two_settings_give_two_results(self):
        # Stated the way a user meets it: pre-fix, an 8x range of the setting
        # returned byte-identical pixels on any real camera file.
        stack = _wide_stack()
        coarse = _register(stack, "scale", 512)
        fine = _register(stack, "scale", WIDE)

        if coarse[0].shape == fine[0].shape:
            worst = max(float(np.max(np.abs(a.astype(np.int32) - b.astype(np.int32))))
                        for a, b in zip(coarse, fine))
            assert worst > 0, "the detection width still makes no difference"


class TestTheDefaultIsUnchanged:
    def test_the_stages_share_one_default(self):
        for impl in (_align_scale_impl, _align_homography_impl, _align_ecc_impl):
            default = inspect.signature(impl).parameters["downscale_width"].default
            assert default == DEFAULT_DETECTION_WIDTH, (
                f"{impl.__name__} defaults to {default}, not the shared "
                f"{DEFAULT_DETECTION_WIDTH}"
            )

    def test_the_registrar_shares_it_too(self):
        default = inspect.signature(ImageRegistration.__init__).parameters[
            "downscale_width"].default
        assert default == DEFAULT_DETECTION_WIDTH
        assert ImageRegistration(method="scale").downscale_width == DEFAULT_DETECTION_WIDTH
        assert ImageRegistration(method="scale",
                                 downscale_width=None).downscale_width == DEFAULT_DETECTION_WIDTH

    def test_it_is_what_the_app_ships(self):
        # The claim that removing the override changes nothing for a user who
        # has not touched the setting rests on these two agreeing.
        from constants import REG_DOWNSCALE_WIDTH
        assert REG_DOWNSCALE_WIDTH == DEFAULT_DETECTION_WIDTH


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _worst_frame_difference(stack, inset=32):
    """How far the frames still are from the one they were aligned onto.

    The stack is broadband texture, so a residual of a fraction of a pixel still
    shows as a sizeable mean difference - which is what makes this sensitive to
    the detection resolution rather than merely to whether a stage ran.
    """
    ref = _gray(stack[0])[inset:-inset, inset:-inset]
    return max(float(np.mean(np.abs(ref - _gray(a)[inset:-inset, inset:-inset])))
               for a in stack[1:])


class TestLargeFramesStillRegister:
    @pytest.mark.parametrize("width", [512, 1024, WIDE])
    def test_the_stack_aligns_at_any_setting(self, width):
        stack = _wide_stack()
        aligned = _register(stack, "scale", width)

        assert len({a.shape for a in aligned}) == 1
        assert _worst_frame_difference(aligned) < 0.7 * _worst_frame_difference(stack), (
            f"detection at {width} px leaves the stack no better aligned than it "
            f"arrived")

    def test_a_finer_detection_width_is_more_accurate(self):
        # What the setting exists to buy, and what the override forbade: on this
        # stack the residual runs 19.3 -> 9.0 -> 2.1 for 512, 1024 and full
        # resolution. With the override back, the last two are the same number.
        stack = _wide_stack()
        coarse = _worst_frame_difference(_register(stack, "scale", 1024))
        fine = _worst_frame_difference(_register(stack, "scale", WIDE))
        assert fine < 0.6 * coarse, (
            f"full-resolution detection ({fine:.2f}) is no more accurate than "
            f"detection at 1024 px ({coarse:.2f})")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
