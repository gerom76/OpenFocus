"""Loading and saving WebP.

WebP is encoded by OpenCV like PNG and TIFF, so the interesting part is not the
codec but the settings it is given and the limits it has. Three promises are
tested here:

1. `.webp` is written losslessly, so a fused result survives the round trip bit
   for bit rather than being quietly recompressed (`TestRoundTrip`).
2. It is an 8-bit container, so a 16-bit result is narrowed on the way out
   instead of being mangled by the encoder (`TestDepth`).
3. It is offered wherever the other formats are - the loader's input set, the
   save dialogs and the batch format list - and the one thing it cannot do,
   images past 16383 px, fails with a reason rather than a bare "could not
   write" (`TestAvailability`, `TestSizeLimit`).

Run with:  python -m pytest tests/test_webp.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import bitdepth
from utils.image_utils import (
    WEBP_MAX_DIMENSION,
    get_imwrite_params,
    read_image_any_depth,
    webp_size_error,
    write_image,
)


def _export_manager():
    """The export module, imported the way the app imports it.

    `ui` has to come first: importing `controllers` cold trips the ui/dialogs
    import cycle the app avoids by loading `ui` before its controllers.
    """
    pytest.importorskip("PyQt6.QtWidgets", reason="the dialogs need PyQt6")
    import ui  # noqa: F401
    from controllers import export_manager
    return export_manager


def _result8():
    rng = np.random.default_rng(21)
    return (rng.random((24, 32, 3)) * 255).astype(np.uint8)


def _result16():
    rng = np.random.default_rng(22)
    return (rng.random((24, 32, 3)) * 65535).astype(np.uint16)


class TestRoundTrip:
    def test_written_losslessly(self, tmp_path):
        image = _result8()
        out = str(tmp_path / "result.webp")

        assert write_image(out, image)

        # Noise is the worst case for a lossy encoder, so bit equality here can
        # only come from the lossless path.
        assert np.array_equal(read_image_any_depth(out), image)

    def test_quality_param_selects_lossless(self):
        # OpenCV switches to lossless only above 100; its default of 100 is lossy.
        flag, quality = get_imwrite_params(".webp")
        assert flag == cv2.IMWRITE_WEBP_QUALITY
        assert quality > 100

    def test_extension_is_matched_without_a_dot_and_case_insensitively(self):
        assert get_imwrite_params("WEBP") == get_imwrite_params(".webp")

    def test_greyscale_is_read_back_as_bgr(self, tmp_path):
        out = str(tmp_path / "grey.webp")
        grey = (np.random.default_rng(23).random((16, 16)) * 255).astype(np.uint8)

        assert cv2.imwrite(out, grey, get_imwrite_params(".webp"))

        read_back = read_image_any_depth(out)
        assert read_back.shape == (16, 16, 3)
        assert np.array_equal(read_back[:, :, 0], grey)


class TestDepth:
    def test_webp_is_an_8bit_container(self):
        assert not bitdepth.supports_16bit(".webp")
        assert bitdepth.prepare_for_write(_result16(), ".webp").dtype == np.uint8

    def test_16bit_result_is_narrowed_rather_than_refused(self, tmp_path):
        out = str(tmp_path / "deep.webp")

        assert write_image(out, _result16())

        read_back = read_image_any_depth(out)
        assert read_back.dtype == np.uint8


class TestSizeLimit:
    def test_oversized_image_is_reported_not_silently_failed(self, tmp_path):
        out = str(tmp_path / "wide.webp")
        wide = np.zeros((4, WEBP_MAX_DIMENSION + 1, 3), np.uint8)

        reason = webp_size_error(wide)
        assert reason and str(WEBP_MAX_DIMENSION) in reason
        # The save fails, and nothing half-written is left behind.
        assert not write_image(out, wide)
        assert not os.path.exists(out)

    def test_largest_allowed_image_still_writes(self, tmp_path):
        out = str(tmp_path / "edge.webp")
        edge = np.zeros((4, WEBP_MAX_DIMENSION, 3), np.uint8)

        assert webp_size_error(edge) is None
        assert write_image(out, edge)


class TestAvailability:
    def test_loader_accepts_webp_input(self):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        assert ".webp" in ImageStackLoader.SUPPORTED_FORMATS
        # .webm is a video; the two must not be confused for one another.
        assert ".webp" not in ImageStackLoader.SUPPORTED_VIDEO_FORMATS

    def test_save_dialog_offers_webp(self):
        export_manager = _export_manager()
        assert "*.webp" in export_manager.save_dialog_filter()
        assert "WebP Files (*.webp)" in export_manager.save_dialog_filter()

    def test_export_path_normalisation_keeps_webp(self):
        export_manager = _export_manager()
        manager = export_manager.ExportManager.__new__(export_manager.ExportManager)
        assert manager.normalize_export_path("out.webp") == "out.webp"
        assert manager.normalize_export_path("out", ".webp") == "out.webp"

    def test_remembered_format_survives_the_settings_round_trip(self):
        export_manager = _export_manager()
        from controllers.settings_manager import SettingsManager

        assert SettingsManager._sanitize_format("webp") == ".webp"
        assert ".webp" in export_manager.ALLOWED_EXPORT_EXTENSION_MAP
