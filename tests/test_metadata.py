"""What a saved result carries besides pixels.

Three promises are tested here:

1. The EXIF block of the first source frame reaches the saved JPEG and PNG, and
   arrives in a form a normal reader (Pillow) understands (`TestExif`).
2. The OpenFocus XMP group is present and holds the version, the render date and
   the time the render took (`TestXmp`), and the Camera section next to it
   repeats the source's camera tags as readable text (`TestCameraSection`).
3. A TIFF-based source - a raw file - hands its block over as the camera wrote
   it, down to the field types and the unreduced rationals (`TestTiffSources`).
4. Orientation is the one tag that does not travel: every container states 1,
   because the result is already upright and the containers are not read alike
   (`TestOrientation`).

The first two are spliced into an already-encoded file, so the last promise -
that the pixels are left exactly as the encoder wrote them, including 16-bit
PNG - is what `TestPixelsUntouched` checks.

Run with:  python -m pytest tests/test_metadata.py -v
"""

import os
import re
import struct
import sys
from datetime import datetime
from xml.etree import ElementTree

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import metadata as meta
from utils.image_utils import write_image
from utils.metadata import RenderMetadata

PIL = pytest.importorskip("PIL.Image", reason="Pillow is needed to read the files back")
from PIL import Image  # noqa: E402
from PIL.TiffImagePlugin import IFDRational  # noqa: E402


def _source_with_exif(path, make="OpenFocus Test Camera", model="Stack 1"):
    """Write a small JPEG carrying a recognisable EXIF block.

    Tags are spread over the three IFDs a real camera fills - main image, EXIF
    sub-IFD and GPS - so the Camera section has to walk all of them, and a
    binary MakerNote is included because that is the value it must skip.
    """
    exif = Image.Exif()
    exif[0x010F] = make   # Make
    exif[0x0110] = model  # Model
    exif[0x0112] = 6      # Orientation

    camera = exif.get_ifd(0x8769)
    camera[0x829A] = IFDRational(1, 200)   # ExposureTime
    camera[0x829D] = IFDRational(56, 10)   # FNumber
    camera[0x8827] = 400                   # ISOSpeedRatings
    camera[0x9003] = "2026:07:20 11:22:33"  # DateTimeOriginal
    camera[0x927C] = b"\x00\x01binary maker note"

    gps = exif.get_ifd(0x8825)
    gps[1] = "N"  # GPSLatitudeRef

    Image.new("RGB", (32, 24), (40, 80, 120)).save(path, exif=exif.tobytes())
    return path


def _result8():
    rng = np.random.default_rng(4)
    return (rng.random((24, 32, 3)) * 255).astype(np.uint8)


def _result16():
    rng = np.random.default_rng(5)
    return (rng.random((24, 32, 3)) * 65535).astype(np.uint16)


def _metadata(source_path, duration=12.5, options=None):
    return RenderMetadata(
        source_path=source_path,
        rendered_at=datetime(2026, 7, 26, 14, 30, 15),
        duration_s=duration,
        options=options,
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
        # The duration carries its unit, so the value reads on its own.
        assert "<OpenFocus:RenderDuration>12.50 s</OpenFocus:RenderDuration>" in packet

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


class TestRenderOptions:
    @pytest.mark.parametrize("ext", [".jpg", ".png"])
    def test_options_are_written_into_the_openfocus_group(self, tmp_path, ext):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / f"result{ext}")
        options = {
            "SourceImages": "12 of 20 frames",
            "Registration": "Scale + ECC",
            "FusionMethod": "Guided Filter",
            "KernelSize": "31 px",
        }

        assert write_image(out, _result8(), metadata=_metadata(source, options=options))

        packet = _read_xmp(out)
        for name, value in options.items():
            assert f"<OpenFocus:{name}>{value}</OpenFocus:{name}>" in packet
        # Written in the order given, after the fields the group always has.
        assert packet.index("RenderDuration") < packet.index("SourceImages")
        assert packet.index("SourceImages") < packet.index("KernelSize")

    def test_option_values_are_escaped(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "result.png")
        options = {"ProcessingUnit": "CUDA <RTX 4090> & co"}

        assert write_image(out, _result8(), metadata=_metadata(source, options=options))

        root = ElementTree.fromstring(_read_xmp(out).split("?>", 1)[1].rsplit("<?xpacket", 1)[0])
        unit = root.find(f".//{{{meta.OPENFOCUS_NS}}}ProcessingUnit")
        assert unit is not None and unit.text == "CUDA <RTX 4090> & co"

    def test_a_name_that_cannot_be_xml_is_dropped_not_written(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "result.png")

        assert write_image(
            out, _result8(),
            metadata=_metadata(source, options={"2Bad": "x", "Good Name": "y"}),
        )

        packet = _read_xmp(out)
        assert "2Bad" not in packet
        # Spaces are not legal in a name, but the rest of it still carries.
        assert "<OpenFocus:GoodName>y</OpenFocus:GoodName>" in packet


class TestCameraSection:
    @pytest.mark.parametrize("ext", [".jpg", ".png"])
    def test_camera_tags_are_cloned_into_the_packet(self, tmp_path, ext):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / f"result{ext}")

        assert write_image(out, _result8(), metadata=_metadata(source))

        packet = _read_xmp(out)
        assert f'xmlns:Camera="{meta.CAMERA_NS}"' in packet
        # One tag from each of the three IFDs a camera fills.
        assert "<Camera:Make>OpenFocus Test Camera</Camera:Make>" in packet
        assert "<Camera:DateTimeOriginal>2026:07:20 11:22:33</Camera:DateTimeOriginal>" in packet
        assert "<Camera:GPSLatitudeRef>N</Camera:GPSLatitudeRef>" in packet

    def test_rationals_read_the_way_the_value_is_quoted(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))

        tags = dict(meta.read_camera_tags(meta.read_source_exif(source)))
        # A shutter speed is a fraction, an aperture is a decimal.
        assert tags["ExposureTime"] == "1/200"
        assert tags["FNumber"] == "5.6"
        assert tags["ISOSpeedRatings"] == "400"

    def test_binary_and_structural_tags_are_left_out(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))

        names = [name for name, _ in meta.read_camera_tags(meta.read_source_exif(source))]
        # The MakerNote stays in the EXIF block, which stores it losslessly.
        assert "MakerNote" not in names
        # Pointers between IFDs describe the source file, not the shot.
        assert "ExifOffset" not in names
        assert "GPSInfo" not in names

    def test_the_packet_is_well_formed_xml(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"), make="Ampersand & <Angle>")
        out = str(tmp_path / "result.png")

        assert write_image(out, _result8(), metadata=_metadata(source))

        # Values that need escaping must not be able to break the packet.
        root = ElementTree.fromstring(_read_xmp(out).split("?>", 1)[1].rsplit("<?xpacket", 1)[0])
        make = root.find(f".//{{{meta.CAMERA_NS}}}Make")
        assert make is not None and make.text == "Ampersand & <Angle>"

    def test_a_source_without_exif_has_no_camera_section(self, tmp_path):
        source = str(tmp_path / "plain.png")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
        out = str(tmp_path / "result.jpg")

        assert write_image(out, _result8(), metadata=_metadata(source))
        assert meta.CAMERA_NS not in _read_xmp(out)


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
        # that were inserted - nothing was re-encoded. Read unchanged, which is
        # what the loader does as well: an Orientation tag must not be able to
        # turn one of the two on the way in. See `TestOrientation` for the
        # value that reaches the file in the first place.
        assert np.array_equal(cv2.imread(tagged, cv2.IMREAD_UNCHANGED),
                              cv2.imread(untagged, cv2.IMREAD_UNCHANGED))
        assert os.path.getsize(tagged) > os.path.getsize(untagged)


class TestCopyExif:
    """The stack-export path: each frame keeps its own camera block, alone.

    `embed` is for a render, and says so in an XMP packet. A processed input
    frame has no render to describe - it is the frame that came out of the
    camera, transformed - so `copy_exif` carries the EXIF across and stops there.
    """

    @pytest.mark.parametrize("ext", [".jpg", ".png"])
    def test_the_source_block_reaches_the_saved_frame(self, tmp_path, ext):
        source = _source_with_exif(str(tmp_path / "src.jpg"), make="Nikon", model="Z 8")
        out = str(tmp_path / f"processed{ext}")

        assert write_image(out, _result8(), source_path=source)

        with Image.open(out) as saved:
            saved.load()
            exif = saved.getexif()
            assert exif.get(0x010F) == "Nikon"
            assert exif.get(0x0110) == "Z 8"
            assert exif.get_ifd(0x8769).get(0x9003) == "2026:07:20 11:22:33"

    def test_it_writes_no_render_record(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "processed.jpg")

        assert write_image(out, _result8(), source_path=source)

        # Nothing was rendered, so nothing may claim there was: no XMP packet,
        # and so no version, duration or Camera section either.
        assert _read_xmp(out) is None

    def test_each_frame_keeps_its_own_block(self, tmp_path):
        # A stack export walks frames and sources together; the exposure of one
        # frame must not be written onto another.
        first = _source_with_exif(str(tmp_path / "a.jpg"), model="Frame A")
        second = _source_with_exif(str(tmp_path / "b.jpg"), model="Frame B")
        outputs = []
        for index, source in enumerate((first, second)):
            out = str(tmp_path / f"out{index}.jpg")
            assert write_image(out, _result8(), source_path=source)
            outputs.append(out)

        models = []
        for path in outputs:
            with Image.open(path) as saved:
                saved.load()
                models.append(saved.getexif().get(0x0110))
        assert models == ["Frame A", "Frame B"]

    def test_a_render_record_takes_precedence(self, tmp_path):
        rendered = _source_with_exif(str(tmp_path / "rendered.jpg"), model="Rendered")
        other = _source_with_exif(str(tmp_path / "other.jpg"), model="Other")
        out = str(tmp_path / "result.jpg")

        # A fused result inherits from the source its own record names, which is
        # the first frame of the stack rather than whatever else is passed.
        assert write_image(out, _result8(), metadata=_metadata(rendered), source_path=other)

        with Image.open(out) as saved:
            saved.load()
            assert saved.getexif().get(0x0110) == "Rendered"
        assert meta.OPENFOCUS_NS in _read_xmp(out)

    def test_a_source_without_exif_is_not_an_error(self, tmp_path):
        source = str(tmp_path / "plain.png")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
        out = str(tmp_path / "processed.jpg")

        assert write_image(out, _result8(), source_path=source)
        assert meta.copy_exif(out, source) is False
        assert meta.copy_exif(out, str(tmp_path / "absent.jpg")) is False

    def test_pixels_are_untouched(self, tmp_path):
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        image = _result16()
        untagged = str(tmp_path / "untagged.png")
        tagged = str(tmp_path / "tagged.png")

        assert write_image(untagged, image)
        assert write_image(tagged, image, source_path=source)

        read_back = cv2.imread(tagged, cv2.IMREAD_UNCHANGED)
        assert read_back.dtype == np.uint16
        assert np.array_equal(read_back, cv2.imread(untagged, cv2.IMREAD_UNCHANGED))

    def test_containers_that_cannot_carry_it_say_so(self, tmp_path):
        # What the stack export checks before the loop, so it can warn once
        # rather than write a folder of frames that quietly lost their tags.
        assert [ext for ext in (".jpg", ".png", ".jxl", ".dng") if meta.carries_exif(ext)] == \
            [".jpg", ".png", ".jxl", ".dng"]
        assert not any(meta.carries_exif(ext) for ext in (".tif", ".tiff", ".bmp", ".webp"))
        assert meta.carries_exif("jpg") and not meta.carries_exif("")

        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "processed.tif")
        # The save still succeeds; only the tags are dropped.
        assert write_image(out, _result8(), source_path=source)
        assert meta.copy_exif(out, source) is False


# ----------------------------------------------------------------------
# TIFF-based sources
# ----------------------------------------------------------------------
def _pack_ifd(entries, base, endian):
    """Lay out one IFD and its out-of-line values at `base`.

    Written out here rather than borrowed from utils.metadata, so that the
    fixture and the code reading it do not share one layout.
    """
    entries = sorted(entries)
    values_offset = base + 2 + 12 * len(entries) + 4
    table = struct.pack(endian + "H", len(entries))
    values = b""
    positions = {}

    for tag, field_type, count, payload in entries:
        head = struct.pack(endian + "HHI", tag, field_type, count)
        if len(payload) <= 4:
            table += head + payload.ljust(4, b"\x00")
            positions[tag] = base + len(table) - 4
        else:
            positions[tag] = values_offset + len(values)
            table += head + struct.pack(endian + "I", positions[tag])
            values += payload
            if len(values) % 2:
                values += b"\x00"
    table += struct.pack(endian + "I", 0)
    return table + values, positions


def _read_ifd(blob, endian, offset):
    """{tag: (type, count, payload)} of the IFD at `offset` in `blob`."""
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    (count,) = struct.unpack_from(endian + "H", blob, offset)
    tags = {}
    for index in range(offset + 2, offset + 2 + 12 * count, 12):
        tag, field_type, values = struct.unpack_from(endian + "HHI", blob, index)
        total = sizes[field_type] * values
        if total <= 4:
            payload = blob[index + 8:index + 8 + total]
        else:
            (position,) = struct.unpack_from(endian + "I", blob, index + 8)
            payload = blob[position:position + total]
        tags[tag] = (field_type, values, payload)
    return tags


def _maker_note(size=24):
    """A vendor note of `size` bytes, recognisable wherever it ends up."""
    return (b"\x00\x01vendor note" * size)[:size]


def _tiff_source(path, endian="<", maker_note=None):
    """A TIFF file whose tags are exactly the bytes written here.

    It stands in for a camera raw, so the awkward parts of one are present on
    purpose: rationals the camera never reduced - a Nikon quotes 1/80 s in
    tenths of a millisecond and 100 mm in tenths of a millimetre - UNDEFINED
    fields, a MakerNote, and IFD 0 tags that describe the file's own embedded
    thumbnail rather than the shot.
    """
    note = _maker_note() if maker_note is None else maker_note
    exif = [
        (0x829A, 5, 1, struct.pack(endian + "2I", 10, 800)),    # ExposureTime
        (0x829D, 5, 1, struct.pack(endian + "2I", 280, 100)),   # FNumber
        (0x920A, 5, 1, struct.pack(endian + "2I", 1000, 10)),   # FocalLength
        (0x9204, 10, 1, struct.pack(endian + "2i", 0, 6)),      # ExposureBiasValue
        (0x8832, 4, 1, struct.pack(endian + "I", 140)),         # RecommendedExposureIndex
        (0x9003, 2, 20, b"2025:05:21 13:17:29\x00"),            # DateTimeOriginal
        (0x9286, 7, 13, b"ASCII\x00\x00\x00hello"),             # UserComment
        (0xA300, 7, 1, b"\x03"),                                # FileSource
        (0xA434, 2, 15, b"NIKKOR Z 100mm\x00"),                 # LensModel
        (0x927C, 7, len(note), note),                           # MakerNote
    ]
    ifd0 = [
        (256, 4, 1, struct.pack(endian + "I", 160)),            # ImageWidth - the thumbnail's
        (257, 4, 1, struct.pack(endian + "I", 120)),            # ImageLength
        (273, 4, 1, struct.pack(endian + "I", 0)),              # StripOffsets
        (271, 2, 18, b"NIKON CORPORATION\x00"),                 # Make
        (272, 2, 12, b"NIKON Z 6_2\x00"),                       # Model
        (282, 5, 1, struct.pack(endian + "2I", 300, 1)),        # XResolution
        (283, 5, 1, struct.pack(endian + "2I", 300, 1)),        # YResolution
        (296, 3, 1, struct.pack(endian + "H", 2)),              # ResolutionUnit
        (306, 2, 20, b"2025:05:21 13:17:29\x00"),               # DateTime
        (50712, 3, 4, struct.pack(endian + "4H", 0, 1, 2, 3)),  # LinearizationTable
        (0x8769, 4, 1, b"\x00" * 4),                            # ExifOffset - patched below
    ]

    table, positions = _pack_ifd(ifd0, 8, endian)
    order = b"II" if endian == "<" else b"MM"
    blob = bytearray(struct.pack(endian + "2sHI", order, 42, 8)) + table
    struct.pack_into(endian + "I", blob, positions[0x8769], len(blob))
    blob += _pack_ifd(exif, len(blob), endian)[0]

    with open(str(path), "wb") as handle:
        handle.write(bytes(blob))
    return str(path)


class TestTiffSources:
    """A TIFF-based source's EXIF, read as the camera wrote it.

    A JPEG hands its block over as the bytes it was stored as, so nothing can
    happen to it on the way. A NEF, a DNG or a .tif has no such segment, and
    letting an imaging library parse and re-encode the tags changes them - which
    is what these pin down does not happen any more.
    """

    @staticmethod
    def _exif_ifd(blob):
        endian = "<" if blob[:2] == b"II" else ">"
        (offset,) = struct.unpack(endian + "I", blob[4:8])
        ifd0 = _read_ifd(blob, endian, offset)
        (pointer,) = struct.unpack(endian + "I", ifd0[0x8769][2])
        return endian, ifd0, _read_ifd(blob, endian, pointer)

    def test_rationals_keep_the_denominator_the_camera_wrote(self, tmp_path):
        blob = meta.read_source_exif(_tiff_source(tmp_path / "src.tif"))
        endian, ifd0, camera = self._exif_ifd(blob)

        # Reduced, these would be 1/80, 14/5, 100/1 and 0/1: the same numbers,
        # and no longer the precision the camera claimed for them.
        assert struct.unpack(endian + "2I", camera[0x829A][2]) == (10, 800)
        assert struct.unpack(endian + "2I", camera[0x829D][2]) == (280, 100)
        assert struct.unpack(endian + "2I", camera[0x920A][2]) == (1000, 10)
        assert struct.unpack(endian + "2i", camera[0x9204][2]) == (0, 6)
        assert struct.unpack(endian + "2I", ifd0[282][2]) == (300, 1)

    def test_field_types_are_the_ones_the_source_used(self, tmp_path):
        blob = meta.read_source_exif(_tiff_source(tmp_path / "src.tif"))
        _endian, _ifd0, camera = self._exif_ifd(blob)

        # UNDEFINED, not BYTE: UserComment's leading character-set marker only
        # means anything in an UNDEFINED field.
        assert camera[0x9286][0] == 7
        assert camera[0x9286][2] == b"ASCII\x00\x00\x00hello"
        assert camera[0xA300][0] == 7
        # LONG, not the SHORT it would fit in.
        assert camera[0x8832][0] == 4

    def test_the_vendors_own_note_travels_whole(self, tmp_path):
        blob = meta.read_source_exif(_tiff_source(tmp_path / "src.tif"))
        _endian, _ifd0, camera = self._exif_ifd(blob)

        assert camera[0x927C][2] == _maker_note()
        assert camera[0xA434][2].rstrip(b"\x00") == b"NIKKOR Z 100mm"

    def test_tags_describing_the_source_file_are_left_behind(self, tmp_path):
        blob = meta.read_source_exif(_tiff_source(tmp_path / "src.tif"))
        _endian, ifd0, _camera = self._exif_ifd(blob)

        # The geometry and strip offsets belong to the source's own thumbnail,
        # and a LinearizationTable to its raw data; carried into another file's
        # block they would describe pixels that are not there.
        for tag in (256, 257, 273, 50712):
            assert tag not in ifd0
        # What the shot is actually recorded in does travel.
        assert ifd0[271][2].rstrip(b"\x00") == b"NIKON CORPORATION"
        assert ifd0[296][2][:2] == struct.pack("<H", 2)
        assert 306 in ifd0 and 283 in ifd0

    def test_a_big_endian_source_keeps_its_byte_order(self, tmp_path):
        blob = meta.read_source_exif(_tiff_source(tmp_path / "src.tif", endian=">"))
        endian, _ifd0, camera = self._exif_ifd(blob)

        assert blob[:4] == b"MM\x00*"
        assert endian == ">"
        assert struct.unpack(">2I", camera[0x829A][2]) == (10, 800)

    def test_the_camera_section_still_reads_it(self, tmp_path):
        # The readable XMP group is built from the same block, so a change to
        # how it is produced has to leave Pillow able to parse it.
        blob = meta.read_source_exif(_tiff_source(tmp_path / "src.tif"))
        tags = dict(meta.read_camera_tags(blob))

        assert tags["Make"] == "NIKON CORPORATION"
        assert tags["LensModel"] == "NIKKOR Z 100mm"
        assert tags["ExposureTime"] == "10/800"

    def test_a_source_that_is_not_a_tiff_is_unaffected(self, tmp_path):
        # JPEG keeps the Pillow path, which hands the APP1 segment over as read.
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        blob = meta.read_source_exif(source)

        assert blob is not None and blob[:4] in (b"II*\x00", b"MM\x00*")
        assert dict(meta.read_camera_tags(blob))["Model"] == "Stack 1"

    def test_an_oversized_block_loses_only_the_maker_note(self, tmp_path):
        # A Nikon MakerNote runs to about 180 KB, well past the 64 KB a JPEG
        # segment holds. Before, the camera tags went over the side with it.
        source = _tiff_source(tmp_path / "src.tif", maker_note=_maker_note(80_000))
        out = str(tmp_path / "result.jpg")

        assert write_image(out, _result8(), metadata=_metadata(source))

        with Image.open(out) as img:
            img.load()
            tags = dict(img.getexif())
        assert tags.get(0x010F) == "NIKON CORPORATION"
        with open(out, "rb") as handle:
            assert _maker_note(80_000) not in handle.read()


# ----------------------------------------------------------------------
# Orientation
# ----------------------------------------------------------------------
def _ifd0_orientation(blob):
    """Orientation as IFD 0 of a bare TIFF stream states it, or None."""
    endian = "<" if blob[:2] == b"II" else ">"
    (offset,) = struct.unpack(endian + "I", blob[4:8])
    entry = _read_ifd(blob, endian, offset).get(0x0112)
    return None if entry is None else struct.unpack(endian + "H", entry[2][:2])[0]


def _saved_orientation(path):
    """Orientation of a saved file, read out of its container by hand.

    Deliberately not through `read_source_exif`: that is the funnel the fix
    lives in, so asking it would answer 1 whatever the file says. These parse
    the four containers where the tag actually sits - the APP1 segment of a
    JPEG, the `eXIf` chunk of a PNG, the Exif box of a JPEG XL, and IFD 0 of a
    DNG, which is a TIFF and needs no unwrapping.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dng":
        return _ifd0_orientation(data)

    if ext in (".jpg", ".jpeg"):
        position = 2
        while position + 4 <= len(data) and data[position] == 0xFF:
            marker = data[position + 1]
            (length,) = struct.unpack(">H", data[position + 2:position + 4])
            if marker == 0xE1 and data[position + 4:position + 10] == b"Exif\x00\x00":
                return _ifd0_orientation(data[position + 10:position + 2 + length])
            if marker == 0xDA:  # the scan; no headers past it
                break
            position += 2 + length
        return None

    if ext == ".png":
        position = 8
        while position + 8 <= len(data):
            (length,) = struct.unpack(">I", data[position:position + 4])
            kind = data[position + 4:position + 8]
            if kind == b"eXIf":
                return _ifd0_orientation(data[position + 8:position + 8 + length])
            if kind == b"IDAT":
                break
            position += 12 + length
        return None

    assert ext == ".jxl"
    position = 0
    while position + 8 <= len(data):
        (length,) = struct.unpack(">I", data[position:position + 4])
        if data[position + 4:position + 8] == b"Exif":
            payload = data[position + 8:position + 8 + length - 8]
            # The box opens with the offset of the TIFF header inside it.
            (start,) = struct.unpack(">I", payload[:4])
            return _ifd0_orientation(payload[4 + start:])
        if length == 0:
            break
        position += length
    return None


class TestOrientation:
    """One answer, whatever the format: the saved result is upright.

    The tag is applied by a different set of readers in every container - a
    PNG's `eXIf` chunk is honoured by some viewers and ignored by others, and a
    JPEG XL's Exif box by the other half of them - so a rotation inherited from
    the source did not merely turn the result, it turned it in some viewers and
    not in others, and the same render saved three ways disagreed with itself.
    """

    @pytest.mark.parametrize("ext", [".jpg", ".png", ".jxl", ".dng"])
    def test_a_rotated_source_does_not_turn_the_result(self, tmp_path, ext):
        if ext == ".jxl":
            pytest.importorskip("imagecodecs", reason="JPEG XL encoder is not installed")
        # The source says 6: rotate a quarter turn clockwise to display. It was
        # applied when the frames were decoded, so the pixels here are upright.
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / f"result{ext}")

        assert write_image(out, _result8(), metadata=_metadata(source))
        assert _saved_orientation(out) == 1

    def test_every_format_agrees(self, tmp_path):
        # The point of the whole thing: no two containers may differ, since a
        # user comparing them side by side sees one render, not four.
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        formats = [".jpg", ".png", ".dng"]
        if meta.carries_exif(".jxl"):
            try:
                import imagecodecs  # noqa: F401
                formats.append(".jxl")
            except ImportError:
                pass

        saved = {}
        for ext in formats:
            out = str(tmp_path / f"result{ext}")
            assert write_image(out, _result8(), metadata=_metadata(source))
            saved[ext] = _saved_orientation(out)

        assert set(saved.values()) == {1}, saved

    def test_a_source_that_never_said_says_it_now(self, tmp_path):
        # The raw fixture writes no Orientation at all. Absent already means 1,
        # but stating it keeps a block read back from one save indistinguishable
        # from one read back from another.
        blob = meta.read_source_exif(_tiff_source(tmp_path / "src.tif"))

        assert _ifd0_orientation(blob) == 1

    def test_the_camera_section_says_the_same(self, tmp_path):
        # The readable copy is cloned from the block, so it inherits the fix
        # rather than needing one of its own.
        source = _source_with_exif(str(tmp_path / "src.jpg"))

        assert dict(meta.read_camera_tags(meta.read_source_exif(source)))["Orientation"] == "1"

    def test_a_processed_frame_is_normalised_too(self, tmp_path):
        # `copy_exif` carries a frame's own block across without a render record.
        # It reads through the same funnel, so it cannot disagree.
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "processed.png")

        assert write_image(out, _result8(), source_path=source)
        assert _saved_orientation(out) == 1

    def test_a_block_with_nothing_to_patch_is_returned_as_it_was(self):
        # `_upright` runs on every block read, including ones it has no business
        # touching, so the shapes it must pass through unchanged are pinned here.
        assert meta._upright(None) is None
        assert meta._upright(b"") == b""
        assert meta._upright(b"II*\x00") == b"II*\x00"          # header and no more
        assert meta._upright(b"not a tiff") == b"not a tiff"
        # A truncated IFD: the count promises entries the block does not hold.
        truncated = struct.pack("<2sHIH", b"II", 42, 8, 3) + b"\x00" * 8
        assert meta._upright(truncated) == truncated
