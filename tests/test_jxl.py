"""Reading and writing JPEG XL.

JPEG XL is the one container OpenCV neither encodes nor decodes, so both
directions go through utils.jxl and libjxl instead. Six promises are tested
here:

1. `.jxl` is routed to that encoder by write_image, and what comes back out is
   the image that went in - at 8 bits and, unlike JPEG, at 16 (`TestRoundTrip`).
2. The file is a JPEG XL *container*, and the EXIF and XMP boxes are spliced
   into it the way they are for JPEG and PNG (`TestMetadata`).
3. `.jxl` is a supported input: the loader accepts it, decodes it at its native
   depth, and reads the EXIF back out of the container (`TestLoading`).
4. libjxl is given more than the one thread `imagecodecs` defaults to, and the
   loader divides the cores between its own workers and libjxl's rather than
   letting the two multiply (`TestThreading`).
5. The dimensions and bit depth can be read from the header without decoding,
   so a `.jxl` stack is sized up front like every other format (`TestProbe`).
6. A build without the codec offers no JPEG XL anywhere - not in the save
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
        assert types[:2] == [b"JXL ", b"ftyp"]
        assert types.index(b"Exif") < types.index(b"xml ")
        assert b"jxlc" in types or b"jxlp" in types
        assert types.index(b"xml ") < min(types.index(t) for t in types
                                          if t in (b"jxlc", b"jxlp"))

    @needs_encoder
    def test_the_level_box_stays_directly_behind_ftyp(self, tmp_path):
        # libjxl emits the jxll level box for a 16-bit image - level 5 caps the
        # bit depth, so >8 bits declares level 10 - and ISO/IEC 18181-2 pins that
        # box directly behind ftyp. Inserting the metadata ahead of it produced a
        # container whose declared level no longer applied, which readers fall
        # back from to their 8-bit default: a 16-bit save read as 8-bit.
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "deep.jxl")

        assert write_image(out, _result16(), metadata=RenderMetadata(source_path=source))

        types = [box_type for box_type, _ in _boxes(out)]
        assert b"jxll" in types, "a 16-bit codestream should carry a level box"
        assert types[:3] == [b"JXL ", b"ftyp", b"jxll"]

    @needs_encoder
    def test_an_untagged_16bit_file_already_orders_its_boxes(self, tmp_path):
        # The guard for the test above: the order it checks has to come from the
        # splice, not from a file that never had a level box to begin with.
        out = str(tmp_path / "plain.jxl")

        assert write_image(out, _result16())

        assert [box_type for box_type, _ in _boxes(out)][:3] == [b"JXL ", b"ftyp", b"jxll"]

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


class TestThreading:
    def test_a_single_worker_is_left_to_libjxls_own_pool_sizing(self):
        assert jxl.threads_for_workers(1) == jxl.AUTO_THREADS
        assert jxl.threads_for_workers(0) == jxl.AUTO_THREADS

    def test_the_cores_are_divided_between_the_workers(self):
        cores = os.cpu_count() or 1
        for workers in (2, 3, 4, 8):
            # Rounded up: a remainder must not be left on the floor.
            assert jxl.threads_for_workers(workers) == max(1, -(-cores // workers))

    def test_the_split_covers_every_core(self):
        cores = os.cpu_count() or 1
        for workers in range(2, cores * 2 + 1):
            assert jxl.threads_for_workers(workers) * workers >= cores or \
                jxl.threads_for_workers(workers) == 1

    def test_more_workers_than_cores_still_leaves_a_thread_each(self):
        # Oversubscribed already; libjxl must not be told to use 0 threads,
        # which would mean "one per core" and multiply the oversubscription.
        assert jxl.threads_for_workers((os.cpu_count() or 1) * 4) >= 1

    def test_the_budget_is_restored_afterwards(self):
        before = jxl.get_decode_threads()
        with jxl.decode_thread_budget(3):
            assert jxl.get_decode_threads() == 3
        assert jxl.get_decode_threads() == before

    def test_the_budget_is_restored_after_a_failure(self):
        before = jxl.get_decode_threads()
        with pytest.raises(RuntimeError):
            with jxl.decode_thread_budget(3):
                raise RuntimeError("decode blew up")
        assert jxl.get_decode_threads() == before

    @needs_encoder
    def test_the_thread_count_does_not_change_the_pixels(self, tmp_path):
        # Lossless is lossless however many threads libjxl splits the frame
        # over, so a stack must not depend on the budget it happened to load at.
        image = _result16()

        one = jxl.encode(image, threads=1)
        many = jxl.encode(image, threads=jxl.AUTO_THREADS)
        assert np.array_equal(jxl.decode(one, threads=1), image)
        assert np.array_equal(jxl.decode(many, threads=1), image)
        assert np.array_equal(jxl.decode(one, threads=jxl.AUTO_THREADS), image)

    @needs_encoder
    def test_the_loader_narrows_the_budget_for_the_stack_it_decodes(self, tmp_path, monkeypatch):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        for index in range(4):
            assert write_image(str(tmp_path / f"frame{index}.jxl"), _result8())

        # What each decode was allowed while the loader's own pool was running.
        seen = []
        original = jxl.decode

        def record(payload, threads=None):
            seen.append(jxl.get_decode_threads())
            return original(payload, threads=threads)

        monkeypatch.setattr(jxl, "decode", record)
        loader = ImageStackLoader()
        entries = [(f"frame{i}.jxl", str(tmp_path / f"frame{i}.jxl")) for i in range(4)]
        results = loader._load_files_parallel(entries, max_workers=4)

        assert all(img is not None for img in results)
        assert seen and set(seen) == {jxl.threads_for_workers(4)}
        # And the process-wide default is back to what it was.
        assert jxl.get_decode_threads() == jxl.AUTO_THREADS

    @needs_encoder
    def test_rgba_comes_back_as_bgra(self):
        # The in-place channel swap has a four-channel path of its own, which
        # nothing else in this file exercises.
        rng = np.random.default_rng(15)
        bgra = (rng.random((12, 10, 4)) * 255).astype(np.uint8)

        read_back = jxl.decode(jxl.encode(bgra))
        assert read_back.shape == (12, 10, 4)
        assert np.array_equal(read_back, bgra)


class TestProbe:
    @needs_encoder
    @pytest.mark.parametrize("shape,dtype,bits", [
        ((24, 32, 3), np.uint8, 8),
        ((24, 32, 3), np.uint16, 16),
        ((800, 800, 3), np.uint8, 8),        # a 1:1 aspect ratio, stored as one
        ((480, 640, 3), np.uint16, 16),      # 4:3, likewise
        ((1080, 1920, 3), np.uint8, 8),      # 16:9
        ((8, 8, 3), np.uint8, 8),            # small enough for the div-8 path
        ((17, 4001, 3), np.uint8, 8),        # and wide enough to escape it
        ((16, 20), np.uint8, 8),             # greyscale
        ((16, 20), np.uint16, 16),
    ])
    def test_the_header_reports_what_the_decoder_produces(self, tmp_path, shape, dtype, bits):
        rng = np.random.default_rng(16)
        scale = 255 if dtype is np.uint8 else 65535
        image = (rng.random(shape) * scale).astype(dtype)
        out = str(tmp_path / "frame.jxl")

        assert write_image(out, image)

        height, width = shape[:2]
        assert jxl.probe(out) == (width, height, bits)

    @needs_encoder
    def test_a_tagged_file_is_still_probeable(self, tmp_path):
        # The Exif and XMP boxes sit ahead of the codestream box, so the walk
        # has to step over them rather than assume the codestream comes first.
        source = _source_with_exif(str(tmp_path / "src.jpg"))
        out = str(tmp_path / "tagged.jxl")

        assert write_image(out, _result16(), metadata=RenderMetadata(source_path=source))

        assert jxl.probe(out) == (32, 24, 16)

    @needs_encoder
    def test_a_large_metadata_block_does_not_hide_the_codestream(self, tmp_path):
        # A camera's EXIF is not small - a MakerNote alone runs to a few hundred
        # KB on a DNG - and the whole block goes into the Exif box ahead of the
        # codestream. Probing used to read a fixed window off the front of the
        # file, so every save from such a source came back unprobeable.
        out = tmp_path / "fat.jxl"
        out.write_bytes(jxl.encode(_result16()))
        bulky = meta._jxl_with_metadata(
            out.read_bytes(), b"II*\x00" + b"\x00" * (1 << 18), None,
        )
        assert bulky is not None
        out.write_bytes(bulky)

        assert jxl.probe(str(out)) == (32, 24, 16)

    @needs_encoder
    def test_a_bare_codestream_is_probeable_too(self, tmp_path):
        # utils.jxl never writes one, but a file from another encoder may be one.
        import imagecodecs
        out = tmp_path / "bare.jxl"
        out.write_bytes(imagecodecs.jpegxl_encode(_result8(), lossless=True, usecontainer=False))

        assert jxl.probe(str(out)) == (32, 24, 8)

    def test_a_file_that_is_not_jpeg_xl_probes_as_none(self, tmp_path):
        not_jxl = tmp_path / "broken.jxl"
        not_jxl.write_bytes(b"not a codestream at all")

        assert jxl.probe(str(not_jxl)) is None

    def test_a_truncated_header_probes_as_none_rather_than_raising(self, tmp_path):
        cut = tmp_path / "cut.jxl"
        cut.write_bytes(b"\xff\x0a")  # the signature and nothing behind it

        assert jxl.probe(str(cut)) is None

    def test_a_missing_file_probes_as_none(self, tmp_path):
        assert jxl.probe(str(tmp_path / "absent.jxl")) is None

    @needs_encoder
    def test_probing_does_not_need_the_codec(self, tmp_path, monkeypatch):
        # The header parser is this module's own, so a build that cannot decode
        # JPEG XL can still say how big a stack of it would be.
        out = str(tmp_path / "frame.jxl")
        assert write_image(out, _result16())

        jxl.clear_probe_cache()
        monkeypatch.setattr(jxl, "_AVAILABLE", False)
        assert jxl.probe(out) == (32, 24, 16)

    @needs_encoder
    def test_a_rewritten_file_is_probed_again(self, tmp_path):
        out = str(tmp_path / "frame.jxl")
        assert write_image(out, _result8())
        assert jxl.probe(out) == (32, 24, 8)

        # Same path, different image: the cache key carries size and mtime, so
        # the stale answer must not survive.
        rng = np.random.default_rng(17)
        assert write_image(out, (rng.random((40, 50, 3)) * 65535).astype(np.uint16))
        assert jxl.probe(out) == (50, 40, 16)

    @needs_encoder
    def test_the_loader_sizes_a_jxl_frame_from_its_header(self, tmp_path):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        out = str(tmp_path / "deep.jxl")
        assert write_image(out, _result16())

        # 32x24 at 16 bits, stored as 3-channel BGR - the same arithmetic the
        # DNG and RAW paths are held to.
        assert ImageStackLoader._probe_frame_bytes(out) == 32 * 24 * 3 * 2

    @needs_encoder
    def test_the_estimate_follows_the_scale_factor(self, tmp_path):
        pytest.importorskip("PyQt6.QtGui", reason="the loader needs PyQt6")
        from core.image_loader import ImageStackLoader

        out = str(tmp_path / "frame.jxl")
        assert write_image(out, _result8())

        assert ImageStackLoader._probe_frame_bytes(out, 0.5) == 16 * 12 * 3 * 1


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
