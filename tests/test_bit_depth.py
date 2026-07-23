"""The 16-bit pipeline contract.

Three things have to hold for a 16-bit stack to be worth loading at 16 bits:

1. Conversions between depths are exact at the endpoints and round rather than
   truncate (`TestConversions`).
2. The loader decodes at the source depth, honours a forced mode, and reconciles
   a stack whose frames disagree (`TestLoader`).
3. Every fusion method is *dtype-preserving* and *scale-correct*: it returns
   uint16 for a uint16 stack, and the result matches what the same scene
   produces at 8 bits (`TestFusionDepth`).

Point 3 is the one that catches the failure this work existed to fix. A method
that kept a hardcoded /255 would normalise a 16-bit frame to ~257 instead of 1.0
and clip to white; comparing the two depths' results catches that immediately,
where an output-dtype assertion alone would not.

A method whose dependency or weights file is missing skips with the reason
rather than failing, matching the rest of the suite.

Run with:  python -m pytest tests/test_bit_depth.py -v
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_registry as reg
from tests.synthetic_stack import make_stack
from utils import bitdepth

STACK_KWARGS = dict(num_slices=3, height=256, width=256, seed=7, style="photographic")


@pytest.fixture(autouse=True)
def restore_mode():
    """Every test leaves the global depth mode as it found it."""
    previous = bitdepth.get_mode()
    yield
    bitdepth.set_mode(previous)


def _stack8():
    return [np.ascontiguousarray(img) for img in make_stack(**STACK_KWARGS)[0]]


def _stack16(stack8):
    return [bitdepth.convert(img, np.uint16) for img in stack8]


class TestConversions:
    def test_full_scale_is_preserved_upward(self):
        """8-bit white must become 16-bit white, not 65280."""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        out = bitdepth.convert(img, np.uint16)
        assert out[0, 0, 0] == 0
        assert out[0, 0, 2] == 65535

    def test_round_trip_is_lossless(self):
        """Every 8-bit level survives a trip through 16-bit and back."""
        levels = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        widened = bitdepth.convert(levels, np.uint16)
        assert np.array_equal(bitdepth.convert(widened, np.uint8), levels)

    def test_normalisation_matches_depth(self):
        assert bitdepth.to_float01(np.array([255], dtype=np.uint8))[0] == pytest.approx(1.0)
        assert bitdepth.to_float01(np.array([65535], dtype=np.uint16))[0] == pytest.approx(1.0)

    def test_from_float01_rounds_not_truncates(self):
        """Truncation would bias every value down half a level.

        See docs/ALGORITHM_IMPROVEMENTS.md item 5 - the same defect, in the
        conversion this module now centralises.
        """
        value = np.array([200.6 / 255.0], dtype=np.float32)
        assert bitdepth.from_float01(value, np.uint8)[0] == 201

    def test_from_float01_clips_out_of_range(self):
        values = np.array([-0.5, 1.5], dtype=np.float32)
        out = bitdepth.from_float01(values, np.uint16)
        assert out[0] == 0 and out[1] == 65535

    def test_stack_dtype_promotes_on_any_16bit_frame(self):
        mixed = [np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint16)]
        assert bitdepth.stack_dtype(mixed) == np.uint16
        assert all(img.dtype == np.uint16 for img in bitdepth.unify(mixed))

    def test_container_capability(self):
        assert bitdepth.supports_16bit(".png") and bitdepth.supports_16bit(".tif")
        assert not bitdepth.supports_16bit(".jpg") and not bitdepth.supports_16bit(".bmp")

    def test_prepare_for_write_narrows_only_when_needed(self):
        img = np.full((2, 2, 3), 65535, np.uint16)
        assert bitdepth.prepare_for_write(img, ".tif").dtype == np.uint16
        assert bitdepth.prepare_for_write(img, ".jpg").dtype == np.uint8


class TestLoader:
    @staticmethod
    def _write_fixture(folder):
        rng = np.random.default_rng(3)
        p16 = os.path.join(folder, "a.tif")
        p8 = os.path.join(folder, "b.png")
        cv2.imwrite(p16, (rng.random((32, 32, 3)) * 65535).astype(np.uint16))
        cv2.imwrite(p8, (rng.random((32, 32, 3)) * 255).astype(np.uint8))
        return p16, p8

    def test_auto_mode_follows_the_source(self):
        from core.image_loader import ImageStackLoader
        with tempfile.TemporaryDirectory() as folder:
            p16, p8 = self._write_fixture(folder)
            bitdepth.set_mode(bitdepth.MODE_AUTO)
            assert ImageStackLoader.read_image_bgr(p16).dtype == np.uint16
            assert ImageStackLoader.read_image_bgr(p8).dtype == np.uint8

    def test_forced_modes_override_the_source(self):
        from core.image_loader import ImageStackLoader
        with tempfile.TemporaryDirectory() as folder:
            p16, p8 = self._write_fixture(folder)

            bitdepth.set_mode(bitdepth.MODE_8)
            assert ImageStackLoader.read_image_bgr(p16).dtype == np.uint8
            assert ImageStackLoader.read_image_bgr(p8).dtype == np.uint8

            bitdepth.set_mode(bitdepth.MODE_16)
            assert ImageStackLoader.read_image_bgr(p16).dtype == np.uint16
            assert ImageStackLoader.read_image_bgr(p8).dtype == np.uint16

    def test_mixed_depth_folder_is_unified(self):
        """A folder holding both a TIFF and a JPEG must not reach fusion split."""
        from core.image_loader import ImageStackLoader
        with tempfile.TemporaryDirectory() as folder:
            self._write_fixture(folder)
            bitdepth.set_mode(bitdepth.MODE_AUTO)
            ok, _, images, _ = ImageStackLoader().load_from_folder(folder)
            assert ok and len(images) == 2
            assert {img.dtype for img in images} == {np.dtype(np.uint16)}

    def test_greyscale_source_still_arrives_as_bgr(self):
        """IMREAD_UNCHANGED keeps channel count as well as depth."""
        from core.image_loader import ImageStackLoader
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "grey.png")
            cv2.imwrite(path, np.full((8, 8), 4096, np.uint16))
            bitdepth.set_mode(bitdepth.MODE_AUTO)
            img = ImageStackLoader.read_image_bgr(path)
            assert img.shape == (8, 8, 3) and img.dtype == np.uint16


class TestWriteRoundTrip:
    def test_16bit_survives_png_and_tiff(self):
        from utils.image_utils import write_image
        source = np.array([[[0, 4096, 65535]]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            for ext in (".png", ".tif"):
                path = os.path.join(folder, f"out{ext}")
                assert write_image(path, source)
                back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                assert back.dtype == np.uint16
                assert np.array_equal(back.reshape(source.shape), source)

    def test_jpeg_is_narrowed_rather_than_mangled(self):
        """JPEG cannot hold 16 bits, so the data must be rescaled down first.

        The failure this guards against is clipping instead of scaling, which
        would drive every level above 255 to white.
        """
        from utils.image_utils import write_image
        source = np.full((16, 16, 3), 32768, np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.jpg")
            assert write_image(path, source)
            back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            assert back.dtype == np.uint8
            # 32768/65535 * 255 = 127.5; JPEG is lossy, so allow a little slack
            assert abs(int(back[8, 8, 0]) - 128) <= 2


class TestTiledFusion:
    """Large images take a separate code path, and it has its own depth bugs.

    MultiFocusFusion switches to tiled fusion above `tile_threshold`, feathering
    the tiles together in a float32 accumulator that holds *levels* rather than
    normalised values. Clipping that accumulator at 255 - as it originally did -
    drives every 16-bit level above 255 to white, so a real render came back
    blown out while every method's own unit test passed.

    The fixture is deliberately just over a small threshold so the tiled branch
    is exercised without a slow render.
    """

    @staticmethod
    def _tiled_fusion(algorithm="guided_filter"):
        from core.multi_focus_fusion import MultiFocusFusion
        fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False)
        fusion.set_tile_mode(True)
        fusion.set_tile_params(block_size=128, overlap=32, threshold=192)
        return fusion

    def test_tiled_path_is_actually_taken(self, capsys):
        stack = [np.ascontiguousarray(img) for img in
                 make_stack(num_slices=3, height=256, width=256, seed=4,
                            style="photographic")[0]]
        self._tiled_fusion().fuse(stack)
        assert "Tiled fusion" in capsys.readouterr().out, (
            "fixture no longer crosses the tile threshold, so this class is "
            "testing the untiled path and proves nothing")

    def test_tiled_preserves_depth_and_range(self):
        stack8 = [np.ascontiguousarray(img) for img in
                  make_stack(num_slices=3, height=256, width=256, seed=4,
                             style="photographic")[0]]
        result = self._tiled_fusion().fuse([bitdepth.convert(i, np.uint16) for i in stack8])
        assert result.dtype == np.uint16
        assert result.max() > 255, (
            f"tiled fusion clipped a 16-bit stack to {result.max()}")

    def test_tiled_16bit_agrees_with_8bit(self):
        """The blown-out render reduced to a single assertion."""
        stack8 = [np.ascontiguousarray(img) for img in
                  make_stack(num_slices=3, height=256, width=256, seed=4,
                             style="photographic")[0]]
        expected = self._tiled_fusion().fuse(stack8)
        result16 = self._tiled_fusion().fuse([bitdepth.convert(i, np.uint16) for i in stack8])

        narrowed = bitdepth.convert(result16, np.uint8).astype(np.int16)
        difference = np.abs(narrowed - expected.astype(np.int16))
        assert difference.mean() < 1.0, (
            f"tiled 16-bit differs from 8-bit by {difference.mean():.2f} levels "
            f"on average (max {difference.max()})")


class TestDisplayConversion:
    """Handing 16-bit pixels to Qt.

    QImage takes a raw buffer plus a declared format and stride; it does not
    inspect the array. Passing a 16-bit buffer while declaring Format_RGB888 and
    a stride of 3*width makes Qt read the low and high byte of each sample as
    two different colour channels, and advance only half a row at a time - which
    renders as scrambled colour over stretched, torn geometry rather than as an
    error. Every display path therefore has to narrow to 8-bit first.

    GPU fusion results are additionally non-contiguous (a permuted view of
    channel-planar memory), so the buffer handed to Qt must be contiguous too.
    """

    @pytest.fixture(autouse=True, scope="class")
    def qt_app(self):
        """QPixmap needs a live QGuiApplication; without one Qt aborts the
        process rather than raising, so this skips instead when it cannot start."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError as exc:
            pytest.skip(f"PyQt6 unavailable: {exc}")
        app = QApplication.instance() or QApplication([])
        yield app

    @staticmethod
    def _pixmap_to_rgb(pixmap):
        from PyQt6.QtGui import QImage
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.bytesPerLine() * height)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, image.bytesPerLine()))
        # Copy before returning: the array is a view into `image`, which is
        # freed on return, and reading it afterwards yields whatever the
        # allocator has since put there.
        return arr[:, : width * 3].reshape(height, width, 3).copy()

    def _roundtrip(self, bgr):
        from utils.image_utils import cv2_to_pixmap
        pixmap = cv2_to_pixmap(bgr)
        assert not pixmap.isNull(), "conversion produced a null pixmap"
        return self._pixmap_to_rgb(pixmap)

    def test_16bit_displays_the_same_picture_as_8bit(self):
        rng = np.random.default_rng(12)
        bgr8 = (rng.random((24, 40, 3)) * 255).astype(np.uint8)
        shown8 = self._roundtrip(bgr8)
        shown16 = self._roundtrip(bitdepth.convert(bgr8, np.uint16))
        difference = np.abs(shown16.astype(np.int16) - shown8.astype(np.int16))
        assert difference.max() <= 1, (
            f"16-bit display differs from 8-bit by up to {difference.max()} levels")

    def test_16bit_channels_are_not_scrambled(self):
        """A pure-blue frame must still read as blue, not magenta.

        This is the reported symptom reduced to one assertion: byte-pair
        misreading turns a saturated single-channel colour into a mix.
        """
        bgr = np.zeros((16, 24, 3), np.uint16)
        bgr[:, :, 0] = 65535                      # BGR: pure blue
        shown = self._roundtrip(bgr)              # RGB out
        assert shown[:, :, 2].min() > 250, "blue channel lost"
        assert shown[:, :, 0].max() < 5 and shown[:, :, 1].max() < 5, (
            f"colour bled into other channels: mean RGB "
            f"{[round(float(shown[:, :, c].mean()), 1) for c in range(3)]}")

    def test_non_contiguous_16bit_survives(self):
        """GPU results arrive as permuted views of channel-planar memory."""
        planar = np.zeros((3, 16, 24), np.uint16)
        planar[0] = 65535                          # blue plane
        strided = np.transpose(planar, (1, 2, 0))
        assert not strided.flags["C_CONTIGUOUS"]
        shown = self._roundtrip(strided)
        assert shown[:, :, 2].min() > 250
        assert shown[:, :, 0].max() < 5 and shown[:, :, 1].max() < 5


class TestOverlayColours:
    """Overlay colours are authored in 8-bit units and must be scaled.

    White text drawn unscaled onto a 16-bit frame lands at level 255 of 65535 -
    all but black, and invisible against the image.
    """

    @staticmethod
    def _label(dtype):
        from utils.validators import LabelAdder
        adder = LabelAdder()
        adder.config.format = "{value}"
        adder.config.font_size = 60
        adder.config.transparent_bg = True
        canvas = np.zeros((160, 320, 3), dtype)
        return adder.add_label_to_image(canvas, 0)

    def test_white_label_is_white_at_16bit(self):
        drawn = self._label(np.uint16)
        assert drawn.max() > 60000, (
            f"label drawn at level {drawn.max()} on a 16-bit frame; "
            f"an 8-bit colour was used unscaled")

    def test_8bit_labels_are_unchanged(self):
        assert self._label(np.uint8).max() == 255


def _methods():
    """Every registered method that can run here, GPU ones included."""
    return [m for m in reg.available_methods(include_gpu=True) if m.available()[0]]


@pytest.mark.parametrize("method", _methods(), ids=lambda m: m.key)
class TestFusionDepth:
    def test_preserves_input_dtype(self, method):
        """A 16-bit stack in, a 16-bit result out."""
        result = method.run(_stack16(_stack8()))
        assert result.dtype == np.uint16, (
            f"{method.label} narrowed a 16-bit stack to {result.dtype}")

    def test_still_returns_8bit_for_8bit_input(self, method):
        """The 8-bit path must not be widened as a side effect."""
        assert method.run(_stack8()).dtype == np.uint8

    def test_16bit_result_agrees_with_8bit(self, method):
        """The two depths must describe the same picture.

        This is the scale check. A method still dividing by 255 would map a
        16-bit frame to ~257.0 instead of 1.0 and clip the result to white,
        which shows up here as a large mean difference even though the output
        dtype would look correct.
        """
        stack8 = _stack8()
        expected = method.run(stack8)
        result16 = method.run(_stack16(stack8))

        narrowed = bitdepth.convert(result16, np.uint8).astype(np.int16)
        difference = np.abs(narrowed - expected.astype(np.int16))

        # A couple of levels of slack: the extra precision genuinely changes
        # rounding at block and weight boundaries, and the argmax in the
        # decision-map methods can legitimately pick a different frame where two
        # focus scores are all but tied.
        assert difference.mean() < 1.0, (
            f"{method.label}: 16-bit result differs from 8-bit by "
            f"{difference.mean():.2f} levels on average (max {difference.max()})")

    def test_16bit_output_uses_the_full_range(self, method):
        """Guards the opposite error: a result computed at 8-bit precision and
        merely stored in a uint16 buffer, which would leave every value below
        256 and render as near-black."""
        result = method.run(_stack16(_stack8()))
        assert result.max() > 255, (
            f"{method.label} produced a uint16 image whose maximum is "
            f"{result.max()}, so it is 8-bit data in a 16-bit container")
