"""One folder loader, one extension list, one sort - and the dead stage is gone.

Items 11 and 12 of docs/REGISTRATION_IMPROVEMENTS.md.

Item 12: the three stages each carried a copy of the same directory loader, and
the copies had drifted - ECC's extension list was missing '.tiff', so a folder
of .tiff frames aligned under `scale` and `homography` and silently lost every
frame under `ecc`. Item 7 made it one loader but still handed it whichever
stage's set the pipeline began with, which moved the disagreement rather than
removing it. The set now comes from `utils.image_utils`, derived from what
`read_image_any_depth` can actually decode, so it is the app's answer rather
than registration's and every stage gets the same one.

Item 11: `_stabilisation_impl` was 117 lines of Lucas-Kanade trajectory
smoothing that nothing dispatched to - a video idea, which deliberately *keeps*
low-frequency motion where a stack wants it removed - carrying the fourth copy
of the loader, a hardcoded 1.04 zoom crop and an `IndexError` on single-frame
input. It is deleted rather than revived, along with the commented-out
`_registration_impl` and the four commented-out aliases below it.

The bar here:

1. The extension list covers every container the loader can decode, and is one
   list rather than one per stage (`TestSupportedExtensions`).
2. Every stage loads the same folder to the same frames, for each of those
   extensions - the .tiff case is the defect itself (`TestEveryStageLoadsTheSameFolder`).
3. The sort orders by the last number in the name and does not raise on a
   folder that mixes numbered and unnumbered names (`TestStackOrder`).
4. Nothing is left of the dead stage or the commented-out blocks
   (`TestDeadCodeIsGone`).

Reinstating the per-stage extension sets fails 2; restoring the old sort key
fails 3; putting `_stabilisation_impl` back fails 4.

Run with:  python -m pytest tests/test_registration_stack_loading.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.registration as registration
from core.registration import ImageRegistration, _load_stack
from utils.image_utils import (
    load_image_folder,
    stack_sort_key,
    supported_input_extensions,
    write_image,
)
from utils import dng, jxl
from tests.synthetic_stack import make_reference


SIZE = 256
STAGES = ["scale", "homography", "ecc"]


def _base_scene(seed=11):
    """A feature-rich frame both SIFT and ECC can lock onto."""
    img = make_reference(height=SIZE, width=SIZE, seed=seed, style="texture")
    for (cy, cx) in [(64, 64), (64, 192), (192, 64), (192, 192), (128, 128)]:
        cv2.circle(img, (cx, cy), 12, (30, 220, 30), -1)
    return img


def _drifting_stack(n=4, step=2.0):
    """A stack that translates by a known, constant amount per frame."""
    frame = _base_scene()
    h, w = frame.shape[:2]
    return [
        cv2.warpAffine(frame, np.float32([[1, 0, i * step], [0, 1, i * step * 0.6]]),
                       (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        for i in range(n)
    ]


def _write_stack(folder, ext, stack=None):
    """Write a drifting stack into `folder` as frame_0<ext> .. frame_n<ext>."""
    os.makedirs(folder, exist_ok=True)
    stack = _drifting_stack() if stack is None else stack
    for i, img in enumerate(stack):
        assert cv2.imwrite(os.path.join(folder, f"frame_{i}{ext}"), img), ext
    return stack


class TestSupportedExtensions:
    """One list, and it matches what the loader can actually decode."""

    @pytest.mark.parametrize("ext", ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'])
    def test_every_opencv_container_is_accepted(self, ext):
        assert ext in supported_input_extensions()

    def test_tiff_is_in_the_list(self):
        """The exact entry ECC's copy was missing."""
        assert '.tiff' in supported_input_extensions()

    def test_the_optional_backends_decide_their_own_entries(self):
        """JPEG XL and DNG appear exactly when their backend does."""
        exts = supported_input_extensions()
        assert set(jxl.extensions()) <= exts
        assert set(dng.extensions()) <= exts
        if not jxl.extensions():
            assert '.jxl' not in exts

    def test_registration_no_longer_keeps_a_list_of_its_own(self):
        """Item 12 is the per-stage sets going away, not being reconciled."""
        for name in ("_FEATURE_EXTENSIONS", "_ECC_EXTENSIONS", "_STAGE_EXTENSIONS"):
            assert not hasattr(registration, name), (
                f"{name} is back: the stages can disagree about their input again"
            )


class TestEveryStageLoadsTheSameFolder:
    """The stages agree about what a folder of frames is."""

    @pytest.mark.parametrize("ext", ['.png', '.tif', '.tiff', '.webp', '.bmp'])
    @pytest.mark.parametrize("stage", STAGES)
    def test_stage_registers_a_folder_of_this_extension(self, tmp_path, ext, stage):
        folder = str(tmp_path / f"{stage}{ext.replace('.', '_')}")
        stack = _write_stack(folder, ext)
        aligned = ImageRegistration(method=stage).process(folder)
        assert len(aligned) == len(stack), (
            f"{stage} loaded {len(aligned)} of {len(stack)} {ext} frames"
        )

    def test_ecc_no_longer_loses_a_tiff_folder(self, tmp_path):
        """The defect stated directly: same folder, scale against ecc."""
        folder = str(tmp_path / "tiff_stack")
        _write_stack(folder, ".tiff")
        scale_names = _load_stack(folder, None)[1]
        assert len(scale_names) == 4
        for stage in STAGES:
            assert len(ImageRegistration(method=stage).process(folder)) == 4

    def test_a_mixed_extension_folder_loads_every_frame(self, tmp_path):
        """One list means one answer per file, not one per stage."""
        folder = str(tmp_path / "mixed")
        os.makedirs(folder)
        stack = _drifting_stack()
        for i, (img, ext) in enumerate(zip(stack, ['.png', '.tif', '.tiff', '.webp'])):
            cv2.imwrite(os.path.join(folder, f"frame_{i}{ext}"), img)
        images, names = _load_stack(folder, None)
        assert len(images) == 4 and len(names) == 4

    @pytest.mark.parametrize("ext", ['.jxl', '.dng'])
    @pytest.mark.parametrize("stage", STAGES)
    def test_stage_registers_an_optional_backend_folder(self, tmp_path, ext, stage):
        """JPEG XL and DNG are stack input everywhere else in the app."""
        if ext not in supported_input_extensions():
            pytest.skip(f"no backend for {ext} in this build")
        folder = tmp_path / f"{stage}{ext.replace('.', '_')}"
        folder.mkdir()
        stack = _drifting_stack()
        for i, img in enumerate(stack):
            write_image(str(folder / f"frame_{i}{ext}"), img)
        assert len(ImageRegistration(method=stage).process(str(folder))) == len(stack)

    def test_a_preloaded_list_is_still_handed_back_untouched(self):
        """Every in-app caller takes this path and must not be routed to disk."""
        stack = _drifting_stack()
        images, names = _load_stack(stack, None)
        assert images is stack and names is None

    def test_a_non_image_file_in_the_folder_is_ignored(self, tmp_path):
        folder = str(tmp_path / "with_junk")
        _write_stack(folder, ".png")
        (tmp_path / "with_junk" / "notes.txt").write_text("not a frame")
        images, names = _load_stack(folder, None)
        assert len(images) == 4 and all(n.endswith(".png") for n in names)


class TestStackOrder:
    """One sort, and it has a fallback that sorts rather than raising."""

    def test_frames_order_by_their_number_not_alphabetically(self, tmp_path):
        folder = str(tmp_path / "numbered")
        os.makedirs(folder)
        frame = _base_scene()
        for i in (1, 2, 10, 20):
            cv2.imwrite(os.path.join(folder, f"frame_{i}.png"), frame)
        assert load_image_folder(folder)[1] == [
            "frame_1.png", "frame_2.png", "frame_10.png", "frame_20.png"
        ]

    def test_a_folder_mixing_numbered_and_unnumbered_names_does_not_raise(self, tmp_path):
        """The old key returned int or str, so this was a TypeError."""
        folder = str(tmp_path / "mixed_names")
        os.makedirs(folder)
        frame = _base_scene()
        for name in ("img1.png", "cover.png", "img2.png"):
            cv2.imwrite(os.path.join(folder, name), frame)
        names = load_image_folder(folder)[1]
        assert names == ["img1.png", "img2.png", "cover.png"]

    def test_unnumbered_names_sort_among_themselves(self):
        names = sorted(["b.png", "a.png", "c.png"], key=stack_sort_key)
        assert names == ["a.png", "b.png", "c.png"]

    def test_the_key_reads_the_last_number_in_the_name(self):
        """A dated or sized prefix must not outrank the frame index."""
        names = sorted(["2024_shot_10.png", "2024_shot_2.png"], key=stack_sort_key)
        assert names == ["2024_shot_2.png", "2024_shot_10.png"]

    def test_the_key_is_type_stable(self):
        """Whatever the name, the tuples compare against each other."""
        keys = [stack_sort_key(n) for n in ("frame_1.png", "cover.png")]
        assert all(isinstance(k, tuple) and len(k) == 3 for k in keys)
        assert keys[0] < keys[1]


class TestDeadCodeIsGone:
    """Item 11: the unreachable stabiliser and the commented-out blocks."""

    @pytest.mark.parametrize("name", [
        "_stabilisation_impl", "_registration_impl", "_align_zoom_impl",
        "image_stack_stabilisation", "image_stack_registration",
        "image_stack_align_zoom", "process_image_stack",
    ])
    def test_the_symbol_is_not_in_the_module(self, name):
        assert not hasattr(registration, name)

    @pytest.mark.parametrize("token", [
        "_stabilisation_impl", "_registration_impl", "_align_zoom_impl",
    ])
    def test_the_source_does_not_carry_it_commented_out(self, token):
        """The aliases were commented out, which is why they survived this long."""
        with open(registration.__file__, encoding="utf-8") as handle:
            source = handle.read()
        assert token not in source

    def test_the_supported_methods_are_the_ones_that_run(self):
        """Nothing was dispatching to the stabiliser, and nothing does now."""
        assert ImageRegistration.SUPPORTED_METHODS == ['scale', 'homography', 'ecc', 'both']
        assert set(registration._STAGE_ESTIMATORS) == {'scale', 'homography', 'ecc'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
