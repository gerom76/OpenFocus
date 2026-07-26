"""What a saved result carries besides pixels.

Two promises are tested here:

1. The EXIF block of the first source frame reaches the saved JPEG and PNG, and
   arrives in a form a normal reader (Pillow) understands (`TestExif`).
2. The OpenFocus XMP group is present and holds the version, the render date and
   the time the render took (`TestXmp`).

Both are spliced into an already-encoded file, so the third promise - that the
pixels are left exactly as the encoder wrote them, including 16-bit PNG - is
what `TestPixelsUntouched` checks.

Run with:  python -m pytest tests/test_metadata.py -v
"""

import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import metadata as meta
from utils.image_utils import write_image
from utils.metadata import RenderMetadata

PIL = pytest.importorskip("PIL.Image", reason="Pillow is needed to read the files back")
from PIL import Image  # noqa: E402


def _source_with_exif(path, make="OpenFocus Test Camera", model="Stack 1"):
    """Write a small JPEG carrying a recognisable EXIF block."""
    exif = Image.Exif()
    exif[0x010F] = make   # Make
    exif[0x0110] = model  # Model
    Image.new("RGB", (32, 24), (40, 80, 120)).save(path, exif=exif.tobytes())
    return path


def _result8():
    rng = np.random.default_rng(4)
    return (rng.random((24, 32, 3)) * 255).astype(np.uint8)


def _result16():
    rng = np.random.default_rng(5)
    return (rng.random((24, 32, 3)) * 65535).astype(np.uint16)


def _metadata(source_path, duration=12.5):
    return RenderMetadata(
        source_path=source_path,
        rendered_at=datetime(2026, 7, 26, 14, 30, 15),
        duration_s=duration,
    )


def _read_xmp(path):
    """The XMP packet of a saved file, as text, or None if it has none."""
    with Image.open(path) as img:
        img.load()
        packet = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp")
    if packet is None:
        return None
    return packet.decode("utf-8") if isinstance(packet, bytes) else packet


class TestExif:
    @pytest.mark.parametrize("ext", [".jpg", ".png"])
    def test_source_exif_reaches_the_result(self, tmp_path, ext):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / f"result{ext}")

        assert write_image(out, _result8(), metadata=_metadata(source))

        with Image.open(out) as img:
            img.load()
            tags = dict(img.getexif())
        assert tags.get(0x010F) == "OpenFocus Test Camera"
        assert tags.get(0x0110) == "Stack 1"

    def test_a_source_without_exif_still_saves(self, tmp_path):
        source = str(tmp_path / "plain.png")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
        out = str(tmp_path / "result.jpg")

        assert write_image(out, _result8(), metadata=_metadata(source))
        # No EXIF to inherit, but the OpenFocus group is still written.
        assert meta.OPENFOCUS_NS in _read_xmp(out)

    def test_a_missing_source_is_not_an_error(self, tmp_path):
        out = str(tmp_path / "result.png")

        assert write_image(out, _result8(), metadata=_metadata(str(tmp_path / "gone.jpg")))
        assert meta.OPENFOCUS_NS in _read_xmp(out)

    def test_exif_is_read_with_and_without_the_exif_prefix(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        blob = meta.read_source_exif(source)

        # Normalised to a bare TIFF stream, whichever form Pillow reported.
        assert blob is not None
        assert not blob.startswith(b"Exif\x00\x00")
        assert blob[:4] in (b"II*\x00", b"MM\x00*")


class TestXmp:
    @pytest.mark.parametrize("ext", [".jpg", ".png"])
    def test_openfocus_group_holds_version_date_and_duration(self, tmp_path, ext):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / f"result{ext}")

        assert write_image(out, _result8(), metadata=_metadata(source, duration=12.5))

        packet = _read_xmp(out)
        assert f'xmlns:OpenFocus="{meta.OPENFOCUS_NS}"' in packet
        assert f"<OpenFocus:Version>{meta.app_version()}</OpenFocus:Version>" in packet
        assert "<OpenFocus:RenderDuration>12.50</OpenFocus:RenderDuration>" in packet

        rendered = re.search(r"<OpenFocus:RenderDate>(.*?)</OpenFocus:RenderDate>", packet)
        assert rendered is not None
        # Written with the local UTC offset, so it parses back as an aware time.
        parsed = datetime.fromisoformat(rendered.group(1))
        assert parsed.tzinfo is not None
        assert parsed.replace(tzinfo=None) == datetime(2026, 7, 26, 14, 30, 15)

    def test_no_metadata_leaves_the_file_as_before(self, tmp_path):
        out = str(tmp_path / "plain.png")

        assert write_image(out, _result8())
        assert _read_xmp(out) is None

    def test_unsupported_container_is_skipped(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "result.tif")

        assert write_image(out, _result8(), metadata=_metadata(source))
        # TIFF is written as it always was; only JPEG and PNG are tagged.
        assert meta.embed(out, _metadata(source)) is False


class TestPixelsUntouched:
    def test_png_pixels_survive_tagging(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        image = _result8()
        untagged = str(tmp_path / "untagged.png")
        tagged = str(tmp_path / "tagged.png")

        assert write_image(untagged, image)
        assert write_image(tagged, image, metadata=_metadata(source))

        assert np.array_equal(cv2.imread(tagged, cv2.IMREAD_UNCHANGED),
                              cv2.imread(untagged, cv2.IMREAD_UNCHANGED))

    def test_16bit_png_stays_16bit(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        image = _result16()
        out = str(tmp_path / "result.png")

        assert write_image(out, image, metadata=_metadata(source))

        read_back = cv2.imread(out, cv2.IMREAD_UNCHANGED)
        assert read_back.dtype == np.uint16
        assert np.array_equal(read_back, image)

    def test_jpeg_scan_is_not_recompressed(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        image = _result8()
        untagged = str(tmp_path / "untagged.jpg")
        tagged = str(tmp_path / "tagged.jpg")

        assert write_image(untagged, image)
        assert write_image(tagged, image, metadata=_metadata(source))

        # Same pixels out, and the tagged file is only larger by the segments
        # that were inserted - nothing was re-encoded.
        assert np.array_equal(cv2.imread(tagged), cv2.imread(untagged))
        assert os.path.getsize(tagged) > os.path.getsize(untagged)
