"""Reading and writing DNG.

DNG is the one container OpenCV can neither write nor read, and LibRaw can only
read, so utils.dng assembles it by hand. Five promises are tested here:

1. `.dng` is routed to that writer by write_image, and what comes back out is
   bit-for-bit what went in - at 8 bits, at 16, in colour and in mono, and
   across the multi-strip boundary (`TestRoundTrip`).
2. The bytes really are a DNG: a little-endian TIFF whose tags say LinearRaw,
   carry a DNGVersion, and describe the pixels' sRGB encoding through a
   LinearizationTable rather than leaving a converter to guess (`TestContainer`).
3. An independent reader agrees. LibRaw opens the file, finds the right
   dimensions, and renders a neutral ramp back as neutral - which only holds if
   the colour matrix, the neutral and the linearization table are all right
   (`TestInterop`).
4. `.dng` is a supported input, read verbatim when OpenFocus wrote it and left to
   LibRaw when a camera did (`TestLoading`).
5. A build without rawpy offers DNG nowhere, but can still read back the files
   it wrote itself (`TestWithoutRawpy`).

The layout is parsed here with utils.dng's own reader, which on its own would
only prove the module agrees with itself; `TestInterop` is what independently
establishes that the file is real.

Run with:  python -m pytest tests/test_dng.py -v
"""

import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import bitdepth, dng
from utils.image_utils import read_image_any_depth, write_image

needs_rawpy = pytest.mark.skipif(
    not dng.is_available(), reason="DNG development needs rawpy (LibRaw)"
)


def _result8():
    rng = np.random.default_rng(21)
    return (rng.random((24, 32, 3)) * 255).astype(np.uint8)


def _result16():
    rng = np.random.default_rng(22)
    return (rng.random((24, 32, 3)) * 65535).astype(np.uint16)


def _grey_ramp():
    """A neutral 8-bit ramp - the input that makes a colour error visible."""
    ramp = np.linspace(0, 255, 256, dtype=np.uint8)
    return np.dstack([np.tile(ramp, (32, 1))] * 3)


def _tags(path):
    """IFD 0 of a written file, via utils.dng's own reader."""
    tags = dng._read_ifd0(path)
    assert tags is not None, "the file is not a little-endian TIFF"
    return tags


def _rationals(payload, signed=False):
    """Decode a (S)RATIONAL tag's payload into the fractions it stands for.

    utils.dng's reader hands these back as raw bytes - it only unpacks the
    integer tags it needs to locate the strips - so the numerator/denominator
    pairs are unpacked here, which also pins down how they are stored.
    """
    dtype = "<i4" if signed else "<u4"
    values = np.frombuffer(payload, dtype=dtype).astype(np.float64)
    return values[0::2] / values[1::2]


class TestRoundTrip:
    def test_write_image_routes_dng_to_the_writer(self, tmp_path):
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())
        assert os.path.getsize(out) > 0
        # OpenCV cannot produce this container at all, so a readable DNG here is
        # itself proof the write did not fall through to cv2.imwrite.
        assert dng.read_linear(out) is not None

    @pytest.mark.parametrize("image", [_result8(), _result16()], ids=["8bit", "16bit"])
    def test_the_pixels_survive_exactly(self, tmp_path, image):
        out = str(tmp_path / "result.dng")
        assert write_image(out, image)

        read_back = dng.read_linear(out)
        assert read_back is not None
        assert read_back.dtype == image.dtype
        assert read_back.shape == image.shape
        # Uncompressed and undeveloped, so this is an identity - not an
        # approximation the way a JPEG round trip would be.
        assert np.array_equal(read_back, image)

    def test_sixteen_bits_are_not_narrowed_on_the_way_out(self, tmp_path):
        assert bitdepth.supports_16bit(".dng")
        assert bitdepth.prepare_for_write(_result16(), ".dng").dtype == np.uint16

        out = str(tmp_path / "deep.dng")
        assert write_image(out, _result16())
        assert dng.read_linear(out).dtype == np.uint16

    def test_greyscale_stays_a_single_channel(self, tmp_path):
        out = str(tmp_path / "grey.dng")
        grey = _result8()[:, :, 0]
        assert write_image(out, grey)

        read_back = dng.read_linear(out)
        assert read_back is not None
        assert read_back.shape == grey.shape
        assert np.array_equal(read_back, grey)

    def test_an_alpha_channel_is_dropped(self, tmp_path):
        out = str(tmp_path / "alpha.dng")
        bgr = _result8()
        bgra = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
        assert write_image(out, bgra)

        read_back = dng.read_linear(out)
        # DNG raw data has no place for alpha, so the colour survives and the
        # channel does not.
        assert read_back.shape == bgr.shape
        assert np.array_equal(read_back, bgr)

    def test_an_image_spanning_many_strips_round_trips(self, tmp_path, monkeypatch):
        # Rather than allocate the tens of megabytes a real multi-strip image
        # needs, the strip target is shrunk so a small image crosses it.
        monkeypatch.setattr(dng, "_STRIP_TARGET_BYTES", 64)
        image = _result16()
        out = str(tmp_path / "strips.dng")
        assert write_image(out, image)

        tags = _tags(out)
        assert len(tags[dng._STRIP_OFFSETS]) > 1, "expected more than one strip"
        assert np.array_equal(dng.read_linear(out), image)

    def test_a_written_dng_reloads_through_the_shared_read_path(self, tmp_path):
        out = str(tmp_path / "frame.dng")
        image = _result16()
        assert write_image(out, image)
        # read_image_any_depth is what the folder-input fusion methods use.
        assert np.array_equal(read_image_any_depth(out), image)


class TestContainer:
    def test_it_is_a_little_endian_tiff(self, tmp_path):
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())
        with open(out, "rb") as handle:
            head = handle.read(4)
        assert head == b"II" + struct.pack("<H", 42)

    def test_the_dng_tags_say_linear_raw(self, tmp_path):
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())
        tags = _tags(out)

        assert tags[dng._PHOTOMETRIC] == [dng._LINEAR_RAW]
        assert tags[dng._COMPRESSION] == [1]  # uncompressed
        assert tags[dng._SAMPLES_PER_PIXEL] == [3]
        assert tags[dng._PLANAR_CONFIG] == [1]
        assert tags[dng._NEW_SUBFILE_TYPE] == [0]  # the full-resolution image
        assert bytes(tags[dng._DNG_VERSION]) == dng._DNG_VERSION_BYTES
        assert tags[dng._UNIQUE_CAMERA_MODEL] == dng.CAMERA_MODEL

    def test_the_pixels_are_declared_srgb_and_neutral(self, tmp_path):
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())
        tags = _tags(out)

        # D65, the white point sRGB is defined against.
        assert tags[dng._CALIBRATION_ILLUMINANT_1] == [dng._ILLUMINANT_D65]
        # AsShotNeutral (1,1,1): already balanced, so no converter re-does it.
        assert np.array_equal(_rationals(tags[dng._AS_SHOT_NEUTRAL]), [1.0, 1.0, 1.0])

        matrix = _rationals(tags[dng._COLOR_MATRIX_1], signed=True)
        assert np.allclose(matrix.reshape(3, 3), np.asarray(dng._XYZ_D65_TO_SRGB), atol=1e-6)

    @pytest.mark.parametrize("image,bits", [(_result8(), 8), (_result16(), 16)])
    def test_the_linearization_table_is_the_srgb_transfer_function(self, tmp_path, image, bits):
        out = str(tmp_path / "result.dng")
        assert write_image(out, image)
        tags = _tags(out)

        table = np.asarray(tags[dng._LINEARIZATION_TABLE], dtype=np.int64)
        assert len(table) == 1 << bits

        # This tag is what stops a raw converter from treating already-encoded
        # values as scene-linear and rendering the image far too bright, so it is
        # checked against the sRGB EOTF itself rather than against the writer.
        encoded = np.linspace(0.0, 1.0, 1 << bits)
        expected = np.where(
            encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
        )
        assert np.array_equal(table, np.rint(expected * dng._LINEAR_MAX))

        # WhiteLevel has to agree with what the table maps onto, not with the
        # stored sample range, or the top of the image would clip or fall short.
        assert tags[dng._WHITE_LEVEL] == [dng._LINEAR_MAX] * 3
        assert table[-1] == dng._LINEAR_MAX

    def test_a_mono_dng_carries_no_colorimetry(self, tmp_path):
        out = str(tmp_path / "grey.dng")
        assert write_image(out, _result8()[:, :, 0])
        tags = _tags(out)

        assert tags[dng._SAMPLES_PER_PIXEL] == [1]
        # A matrix and a neutral would be meaningless for a single channel, and
        # DNG 1.4 expects a monochrome file to omit them.
        assert dng._COLOR_MATRIX_1 not in tags
        assert dng._AS_SHOT_NEUTRAL not in tags


class TestInterop:
    """Whether a reader that is not this module can make sense of the file."""

    @needs_rawpy
    def test_libraw_opens_it_at_the_right_size(self, tmp_path):
        import rawpy

        out = str(tmp_path / "result.dng")
        image = _result16()
        assert write_image(out, image)

        with open(out, "rb") as handle:
            with rawpy.imread(handle) as raw:
                assert (raw.sizes.width, raw.sizes.height) == image.shape[1::-1]

    @needs_rawpy
    def test_libraw_renders_a_neutral_ramp_as_neutral(self, tmp_path):
        import rawpy

        out = str(tmp_path / "ramp.dng")
        ramp = _grey_ramp()
        assert write_image(out, ramp)

        with open(out, "rb") as handle:
            with rawpy.imread(handle) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True)

        row = rgb[rgb.shape[0] // 2].astype(np.int16)
        # Grey in, grey out: a wrong ColorMatrix1 or AsShotNeutral would tint it.
        assert np.abs(row[:, 0] - row[:, 1]).max() <= 1
        assert np.abs(row[:, 1] - row[:, 2]).max() <= 1

        # The endpoints must land exactly, and the ramp must stay monotonic. The
        # midtones are deliberately not pinned: LibRaw applies its own BT.709
        # output curve, which is not identical to sRGB's, and a converter
        # choosing its own rendering is the whole point of a raw file. What is
        # pinned is that no *second* encoding is applied - were the
        # LinearizationTable missing, mid grey would come back near 186, not 128.
        assert row[0, 0] == 0
        assert row[-1, 0] == 255
        assert np.all(np.diff(row[:, 0].astype(np.int32)) >= 0)
        assert abs(int(row[128, 0]) - 128) < 20


class TestLoading:
    def test_the_loader_accepts_dng_input(self):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        assert ".dng" in ImageStackLoader.SUPPORTED_FORMATS
        assert ".dng" in ImageStackLoader.RAW_FORMATS
        # Input only: a stack of stills, never a video container.
        assert ".dng" not in ImageStackLoader.SUPPORTED_VIDEO_FORMATS

    def test_the_loader_reads_its_own_dng_verbatim(self, tmp_path):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        out = str(tmp_path / "frame.dng")
        image = _result16()
        assert write_image(out, image)

        # Not developed: no white balance and no tone curve are applied to a
        # frame that already carries them, so a saved result reloads unchanged.
        assert np.array_equal(ImageStackLoader.read_image_bgr(out), image)

    def test_a_saved_result_is_a_valid_source_for_the_next_stack(self, tmp_path):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        for index in range(2):
            assert write_image(str(tmp_path / f"frame{index}.dng"), _result8())

        ok, message, images, filenames = ImageStackLoader().load_from_folder(str(tmp_path))
        assert ok, message
        assert len(images) == 2 and len(filenames) == 2

    def test_the_frame_size_is_probed_from_the_header(self, tmp_path):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        out = str(tmp_path / "frame.dng")
        image = _result16()
        assert write_image(out, image)

        height, width = image.shape[:2]
        assert dng.probe(out) == (width, height, 16)
        # The stored depth, not the 16 bits LibRaw would claim regardless.
        assert ImageStackLoader._probe_frame_bytes(out) == width * height * 3 * 2

    def test_the_save_dialog_offers_dng(self):
        pytest.importorskip("PyQt6.QtWidgets", reason="the dialogs need PyQt6")
        # `ui` first: importing `controllers` cold trips the ui/dialogs import
        # cycle that the app avoids by loading `ui` before its controllers.
        import ui  # noqa: F401
        from controllers import export_manager

        assert "DNG Files (*.dng)" in export_manager.save_dialog_filter()
        assert (".dng", "DNG") in export_manager.export_format_choices()
        assert export_manager.save_dialog_selected_filter(".dng") == "DNG Files (*.dng)"


class TestNotOurOwn:
    def test_a_file_without_the_marker_is_left_to_libraw(self, tmp_path):
        out = tmp_path / "camera.dng"
        assert write_image(str(out), _result8())

        # A camera DNG names its camera in UniqueCameraModel, so the verbatim
        # path must decline it; renaming the marker here reproduces that without
        # needing a real raw file in the repository.
        data = out.read_bytes()
        assert dng.CAMERA_MODEL.encode() in data
        out.write_bytes(data.replace(dng.CAMERA_MODEL.encode(), b"SomeCam\x00", 1))

        assert dng.read_linear(str(out)) is None
        assert dng.probe(str(out)) is None

    def test_a_truncated_file_reads_as_none(self, tmp_path):
        out = tmp_path / "short.dng"
        assert write_image(str(out), _result16())
        out.write_bytes(out.read_bytes()[:512])

        # The strips are gone; nothing should raise on the way to None.
        assert dng.read_linear(str(out)) is None
        assert dng.read(str(out)) is None
        assert read_image_any_depth(str(out)) is None

    def test_a_file_that_is_not_a_tiff_reads_as_none(self, tmp_path):
        out = tmp_path / "broken.dng"
        out.write_bytes(b"not a tiff at all")

        assert dng.read_linear(str(out)) is None
        assert dng.read(str(out)) is None
        assert read_image_any_depth(str(out)) is None

    def test_a_missing_file_reads_as_none(self, tmp_path):
        assert dng.read_linear(str(tmp_path / "absent.dng")) is None
        assert dng.read(str(tmp_path / "absent.dng")) is None

    def test_an_unsupported_dtype_fails_cleanly(self, tmp_path):
        out = str(tmp_path / "float.dng")
        # DNG's integer sample formats cannot hold this; it must be reported
        # rather than written as reinterpreted bytes.
        assert write_image(out, _result8().astype(np.float32)) is False


class TestWithoutRawpy:
    def test_it_is_not_offered_as_an_input_format(self, monkeypatch):
        monkeypatch.setattr(dng, "_RAWPY_AVAILABLE", False)
        assert dng.extensions() == ()
        assert dng.unavailable_reason()

        monkeypatch.setattr(dng, "_RAWPY_AVAILABLE", True)
        assert ".dng" in dng.extensions()
        assert dng.unavailable_reason() == ""

    def test_the_format_is_left_out_of_the_save_dialog(self, monkeypatch):
        pytest.importorskip("PyQt6.QtWidgets", reason="the dialogs need PyQt6")
        import ui  # noqa: F401
        from controllers import export_manager

        monkeypatch.setattr(dng, "_RAWPY_AVAILABLE", False)
        assert "*.dng" not in export_manager.save_dialog_filter()

        monkeypatch.setattr(dng, "_RAWPY_AVAILABLE", True)
        assert "DNG Files (*.dng)" in export_manager.save_dialog_filter()

    def test_its_own_files_still_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dng, "_RAWPY_AVAILABLE", False)
        out = str(tmp_path / "frame.dng")
        image = _result16()

        # Writing needs nothing but numpy, and reading back a linear DNG never
        # goes near LibRaw - only developing a camera DNG does.
        assert write_image(out, image)
        assert np.array_equal(dng.read_linear(out), image)
        assert np.array_equal(dng.read(out), image)

    def test_a_camera_dng_cannot_be_developed(self, tmp_path, monkeypatch):
        out = tmp_path / "camera.dng"
        assert write_image(str(out), _result8())
        data = out.read_bytes()
        out.write_bytes(data.replace(dng.CAMERA_MODEL.encode(), b"SomeCam\x00", 1))

        monkeypatch.setattr(dng, "_RAWPY_AVAILABLE", False)
        assert dng.read(str(out)) is None
