"""Reading and writing JPEG XL.

JPEG XL is the one container OpenCV neither encodes nor decodes, so both
directions go through utils.jxl and libjxl instead. Four promises are tested
here:

1. `.jxl` is routed to that encoder by write_image, and what comes back out is
   the image that went in - at 8 bits and, unlike JPEG, at 16 (`TestRoundTrip`).
2. The file is a JPEG XL *container*, and the EXIF and XMP boxes are spliced
   into it the way they are for JPEG and PNG (`TestMetadata`).
3. `.jxl` is a supported input: the loader accepts it, decodes it at its native
   depth, and reads the EXIF back out of the container (`TestLoading`).
4. A build without the codec offers no JPEG XL anywhere - not in the save
   dialogs, and not as an input format (`TestWithoutTheEncoder`).

Run with:  python -m pytest tests/test_jxl.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import bitdepth, jxl
from utils import metadata as meta
from utils.image_utils import read_image_any_depth, write_image
from utils.metadata import RenderMetadata

needs_encoder = pytest.mark.skipif(
    not jxl.is_available(), reason="JPEG XL encoder (imagecodecs) is not installed"
)

_CONTAINER_SIGNATURE = b"\x00\x00\x00\x0cJXL \r\n\x87\n"


def _result8():
    rng = np.random.default_rng(11)
    return (rng.random((24, 32, 3)) * 255).astype(np.uint8)


def _result16():
    rng = np.random.default_rng(12)
    return (rng.random((24, 32, 3)) * 65535).astype(np.uint16)


def _source_with_exif(path):
    """A small JPEG carrying an EXIF block, as the first frame of a stack would."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow is needed to write the source")

    exif = Image.Exif()
    exif[0x010F] = "OpenFocus Test Camera"
    exif[0x0110] = "Stack 1"
    Image.new("RGB", (32, 24), (40, 80, 120)).save(path, exif=exif.tobytes())
    return path


def _boxes(path):
    """The container boxes of a JPEG XL file, as an ordered list of (type, payload)."""
    with open(path, "rb") as handle:
        data = handle.read()
    assert data.startswith(_CONTAINER_SIGNATURE)

    found = []
    position = 0
    while position + 8 <= len(data):
        size = int.from_bytes(data[position:position + 4], "big")
        box_type = data[position + 4:position + 8]
        if size == 0:  # Runs to the end of the file.
            found.append((box_type, data[position + 8:]))
            break
        if size < 8:
            break
        found.append((box_type, data[position + 8:position + size]))
        position += size
    return found


class TestRoundTrip:
    @needs_encoder
    def test_write_image_routes_jxl_to_the_encoder(self, tmp_path):
        out = str(tmp_path / "result.jxl")

        assert write_image(out, _result8())

        with open(out, "rb") as handle:
            assert handle.read(12) == _CONTAINER_SIGNATURE

    @needs_encoder
    def test_8bit_survives_unchanged(self, tmp_path):
        image = _result8()
        out = str(tmp_path / "result.jxl")

        assert write_image(out, image)

        with open(out, "rb") as handle:
            read_back = jxl.decode(handle.read())
        # Written lossless, so noise comes back bit for bit.
        assert read_back.dtype == np.uint8
        assert np.array_equal(read_back, image)

    @needs_encoder
    def test_16bit_stays_16bit(self, tmp_path):
        image = _result16()
        out = str(tmp_path / "result.jxl")

        assert write_image(out, image, announce=True)

        with open(out, "rb") as handle:
            read_back = jxl.decode(handle.read())
        assert read_back.dtype == np.uint16
        assert np.array_equal(read_back, image)

    def test_the_container_is_declared_16bit_capable(self):
        # The depth policy has to agree, or write_image would narrow first.
        assert bitdepth.supports_16bit(".jxl")
        assert bitdepth.prepare_for_write(_result16(), ".jxl").dtype == np.uint16

    @needs_encoder
    def test_greyscale_is_written_as_one_plane(self, tmp_path):
        rng = np.random.default_rng(13)
        image = (rng.random((16, 20)) * 255).astype(np.uint8)
        out = str(tmp_path / "grey.jxl")

        assert write_image(out, image)

        with open(out, "rb") as handle:
            read_back = jxl.decode(handle.read())
        assert np.array_equal(np.squeeze(read_back), image)

    @needs_encoder
    def test_lossy_is_smaller_than_lossless(self, tmp_path):
        image = _result8()

        lossless = jxl.encode(image)
        lossy = jxl.encode(image, distance=3.0)
        assert len(lossy) < len(lossless)

    def test_extension_matching_ignores_case_and_the_dot(self):
        assert jxl.is_jxl(".jxl") and jxl.is_jxl("JXL") and jxl.is_jxl(".JXL")
        assert not jxl.is_jxl(".jpg")


class TestMetadata:
    @needs_encoder
    def test_the_openfocus_group_is_written_as_an_xmp_box(self, tmp_path):
        out = str(tmp_path / "result.jxl")

        assert write_image(out, _result8(), metadata=RenderMetadata(
            duration_s=12.5, options={"FusionMethod": "DCT"},
        ))

        xmp = dict(_boxes(out))[b"xml "].decode("utf-8")
        assert f'xmlns:OpenFocus="{meta.OPENFOCUS_NS}"' in xmp
        assert "<OpenFocus:RenderDuration>12.50 s</OpenFocus:RenderDuration>" in xmp
        assert "<OpenFocus:FusionMethod>DCT</OpenFocus:FusionMethod>" in xmp

    @needs_encoder
    def test_source_exif_reaches_the_result(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "result.jxl")

        assert write_image(out, _result8(), metadata=RenderMetadata(source_path=source))

        payload = dict(_boxes(out))[b"Exif"]
        # A JPEG XL Exif box opens with the offset of the TIFF header inside it.
        assert payload[:4] == b"\x00\x00\x00\x00"
        assert payload[4:8] in (b"II*\x00", b"MM\x00*")
        assert b"OpenFocus Test Camera" in payload

    @needs_encoder
    def test_metadata_boxes_precede_the_codestream(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "result.jxl")

        assert write_image(out, _result8(), metadata=RenderMetadata(source_path=source))

        types = [box_type for box_type, _ in _boxes(out)]
        assert types[:4] == [b"JXL ", b"ftyp", b"Exif", b"xml "]
        assert b"jxlc" in types or b"jxlp" in types

    @needs_encoder
    def test_pixels_survive_tagging(self, tmp_path):
        image = _result16()
        untagged = str(tmp_path / "untagged.jxl")
        tagged = str(tmp_path / "tagged.jxl")

        assert write_image(untagged, image)
        assert write_image(tagged, image, metadata=RenderMetadata(duration_s=1.0))

        with open(tagged, "rb") as handle:
            read_back = jxl.decode(handle.read())
        assert np.array_equal(read_back, image)
        # Only the inserted boxes made it bigger; the codestream was not re-encoded.
        assert os.path.getsize(tagged) > os.path.getsize(untagged)

    def test_a_bare_codestream_cannot_be_tagged(self):
        # utils.jxl never writes one, but a file that is not a container has
        # nowhere to put a box and must be reported as untagged rather than cut.
        assert meta._jxl_with_metadata(b"\xff\x0a" + b"\x00" * 32, None, b"<xmp/>") is None


class TestLoading:
    @needs_encoder
    def test_read_image_any_depth_decodes_jxl(self, tmp_path):
        image = _result8()
        out = str(tmp_path / "frame.jxl")

        assert write_image(out, image)

        # The same entry point every folder-input path uses, so this is what a
        # `.jxl` stack actually loads through.
        assert np.array_equal(read_image_any_depth(out), image)

    @needs_encoder
    def test_16bit_source_loads_at_16_bits(self, tmp_path):
        image = _result16()
        out = str(tmp_path / "deep.jxl")

        assert write_image(out, image)

        # Auto mode keeps the file's own depth; the point of decoding through
        # libjxl is that the extra bits are there to keep.
        read_back = read_image_any_depth(out)
        assert read_back.dtype == np.uint16
        assert np.array_equal(read_back, image)

    @needs_encoder
    def test_the_depth_mode_still_decides_the_stored_dtype(self, tmp_path):
        out = str(tmp_path / "deep.jxl")
        assert write_image(out, _result16())

        previous = bitdepth.get_mode()
        try:
            bitdepth.set_mode(bitdepth.MODE_8)
            assert read_image_any_depth(out).dtype == np.uint8
            bitdepth.set_mode(bitdepth.MODE_16)
            assert read_image_any_depth(out).dtype == np.uint16
        finally:
            bitdepth.set_mode(previous)

    @needs_encoder
    def test_greyscale_is_read_back_as_bgr(self, tmp_path):
        rng = np.random.default_rng(14)
        grey = (rng.random((16, 16)) * 255).astype(np.uint8)
        out = str(tmp_path / "grey.jxl")

        assert write_image(out, grey)

        read_back = read_image_any_depth(out)
        assert read_back.shape == (16, 16, 3)
        assert np.array_equal(read_back[:, :, 0], grey)

    @needs_encoder
    def test_the_loader_accepts_jxl_input(self):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        assert ".jxl" in ImageStackLoader.SUPPORTED_FORMATS
        # Input only: a stack of stills, never a video container.
        assert ".jxl" not in ImageStackLoader.SUPPORTED_VIDEO_FORMATS

    @needs_encoder
    def test_the_loader_decodes_a_jxl_frame(self, tmp_path):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        image = _result8()
        out = str(tmp_path / "frame.jxl")
        assert write_image(out, image)

        assert np.array_equal(ImageStackLoader.read_image_bgr(out), image)

    @needs_encoder
    def test_exif_is_read_back_out_of_the_container(self, tmp_path):
        # A .jxl written by OpenFocus is itself a valid source for the next
        # render, so the Exif box it carries has to be readable again - Pillow
        # cannot open the file to do it.
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        frame = str(tmp_path / "frame.jxl")
        assert write_image(frame, _result8(), metadata=RenderMetadata(source_path=source))

        blob = meta.read_source_exif(frame)
        assert blob is not None
        assert blob[:4] in (b"II*\x00", b"MM\x00*")
        assert b"OpenFocus Test Camera" in blob

    @needs_encoder
    def test_a_jxl_without_exif_yields_none_rather_than_failing(self, tmp_path):
        out = str(tmp_path / "bare.jxl")
        assert write_image(out, _result8())

        assert meta.read_source_exif(out) is None

    @needs_encoder
    def test_a_file_that_is_not_jpeg_xl_reads_as_none(self, tmp_path):
        not_jxl = tmp_path / "broken.jxl"
        not_jxl.write_bytes(b"not a codestream")

        assert jxl.read(str(not_jxl)) is None
        assert read_image_any_depth(str(not_jxl)) is None

    def test_a_missing_file_reads_as_none(self, tmp_path):
        assert jxl.read(str(tmp_path / "absent.jxl")) is None


class TestWithoutTheEncoder:
    def test_saving_fails_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jxl, "_AVAILABLE", False)
        out = str(tmp_path / "result.jxl")

        assert write_image(out, _result8()) is False
        assert jxl.unavailable_reason()

    def test_the_format_is_left_out_of_the_save_dialog(self, monkeypatch):
        pytest.importorskip("PyQt6.QtWidgets", reason="the dialogs need PyQt6")
        # `ui` first: importing `controllers` cold trips the ui/dialogs import
        # cycle that the app avoids by loading `ui` before its controllers.
        import ui  # noqa: F401
        from controllers import export_manager

        monkeypatch.setattr(jxl, "_AVAILABLE", False)
        assert "*.jxl" not in export_manager.save_dialog_filter()

        monkeypatch.setattr(jxl, "_AVAILABLE", True)
        assert "JPEG XL Files (*.jxl)" in export_manager.save_dialog_filter()

    def test_it_is_not_offered_as_an_input_format(self, monkeypatch):
        monkeypatch.setattr(jxl, "_AVAILABLE", False)
        assert jxl.extensions() == ()

        monkeypatch.setattr(jxl, "_AVAILABLE", True)
        assert ".jxl" in jxl.extensions()

    def test_loading_fails_cleanly(self, tmp_path, monkeypatch):
        out = str(tmp_path / "frame.jxl")
        if jxl.is_available():
            assert write_image(out, _result8())

        monkeypatch.setattr(jxl, "_AVAILABLE", False)
        assert jxl.read(out) is None
