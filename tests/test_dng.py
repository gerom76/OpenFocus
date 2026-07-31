"""Reading and writing DNG.

DNG is the one container OpenCV can neither write nor read, and LibRaw can only
read, so utils.dng assembles it by hand. Nine promises are tested here:

1. `.dng` is routed to that writer by write_image, and what comes back out is
   what went in - at 8 bits, at 16, in colour and in mono, and across the
   multi-strip boundary. Within a couple of parts in 65535, because the samples
   are stored as scene-linear light and the curve that puts them there cannot be
   inverted exactly at the bottom of the range (`TestRoundTrip`).
2. The bytes really are a DNG: a little-endian TIFF whose tags say LinearRaw and
   carry a DNGVersion, whose samples have been linearised rather than left
   encoded with a LinearizationTable to explain them - a tag Luminar Neo does
   not read - and whose curve is the one the develop actually applied, not a
   different standard's (`TestContainer`).
3. An independent reader agrees. LibRaw opens the file, finds the right
   dimensions, and renders a neutral ramp back as the ramp that went in - which
   only holds if the colour matrix, the neutral and the transfer function are
   all right (`TestInterop`).
4. `.dng` is a supported input, read verbatim when OpenFocus wrote it and left to
   LibRaw when a camera did (`TestLoading`).
5. A build without rawpy offers DNG nowhere, but can still read back the files
   it wrote itself (`TestWithoutRawpy`).
6. The compression modes say what they mean: lossless is exactly lossless and
   actually smaller, lossy is 8-bit only, and every mode survives LibRaw
   (`TestCompression`).
7. The fast-load preview is a real, findable preview - a reduced-resolution
   JPEG SubIFD that LibRaw hands back as a thumbnail (`TestFastLoad`).
8. A source file's EXIF is carried in as tags of the DNG's own - camera, shot
   and position - with the offsets that cannot move left behind, a MakerNote
   kept only when it is measured from itself, and without disturbing the pixels
   or LibRaw (`TestSourceExif`).
9. A write that fails part way leaves no file, since the file is streamed rather
   than assembled in memory first (`TestPartialWrites`), and a camera DNG is
   developed on the GPU wherever the stack loader would have been, falling back
   to LibRaw rather than to a failed frame (`TestCameraDevelop`).

The layout is parsed here with utils.dng's own reader, which on its own would
only prove the module agrees with itself; `TestInterop` and the LibRaw checks in
`TestCompression` and `TestFastLoad` are what independently establish that the
file is real.

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

needs_lossless = pytest.mark.skipif(
    not dng.lossless_available(),
    reason="lossless DNG round trip needs imagecodecs",
)


@pytest.fixture(autouse=True)
def _restore_dng_settings():
    """Put the module's write settings back, whatever a test did to them.

    They are module-level - the writer is reached through write_image, which
    carries no codec choice - so a test that changes one would otherwise change
    the format every later test writes in.
    """
    saved = (dng.get_compression(), dng.get_lossy_quality(), dng.get_fast_load())
    yield
    dng.set_compression(saved[0])
    dng.set_lossy_quality(saved[1])
    dng.set_fast_load(saved[2])


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


# What a round trip through the container may cost one sample. The file stores
# scene-linear light while the pipeline's frames are display-referred, so a save
# applies the BT.709 curve and a load undoes it - and near black that curve is
# many-to-one, so no inverse can be exact there. Two parts in 65535 is what it
# measures, which is a hundredth of one level at 8 bits.
_ROUND_TRIP_TOLERANCE = 2


def _as_stored(image):
    """`image` at the depth and scale the container gives it back at.

    An 8-bit frame comes back 16-bit, because it is *linear* samples that get
    stored and 8 bits of those band the shadows into uselessness - so the writer
    promotes on the way out and the reader has 16 bits to hand back. Full scale
    stays full scale, hence 257 rather than a shift.
    """
    return image.astype(np.uint16) * 257 if image.dtype == np.uint8 else image


def _stored_samples(path):
    """The samples an uncompressed DNG actually holds, in file order.

    `dng.read_linear` is no use for this: it puts the samples back through the
    transfer function on the way out, which is exactly what a test of what was
    *written* has to see past. So the strips are read directly, and the order is
    the file's own - RGB, against the pipeline's BGR.
    """
    tags = _tags(path)
    height = int(tags[dng._IMAGE_LENGTH][0])
    width = int(tags[dng._IMAGE_WIDTH][0])
    samples = int(tags[dng._SAMPLES_PER_PIXEL][0])
    bits = int(tags[dng._BITS_PER_SAMPLE][0])

    payload = bytearray()
    with open(path, "rb") as handle:
        for offset, count in zip(tags[dng._STRIP_OFFSETS], tags[dng._STRIP_BYTE_COUNTS]):
            handle.seek(int(offset))
            payload += handle.read(int(count))

    dtype = np.dtype("<u2") if bits == 16 else np.dtype("u1")
    return np.frombuffer(bytes(payload), dtype=dtype).reshape(height, width, samples)


def _round_tripped(read_back, image):
    """Whether a frame came back out of the container as the one that went in."""
    expected = _as_stored(image)
    if read_back is None or read_back.dtype != expected.dtype:
        return False
    if read_back.shape != expected.shape:
        return False
    difference = np.abs(read_back.astype(np.int32) - expected.astype(np.int32))
    return int(difference.max()) <= _ROUND_TRIP_TOLERANCE


class TestRoundTrip:
    def test_write_image_routes_dng_to_the_writer(self, tmp_path):
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())
        assert os.path.getsize(out) > 0
        # OpenCV cannot produce this container at all, so a readable DNG here is
        # itself proof the write did not fall through to cv2.imwrite.
        assert dng.read_linear(out) is not None

    @pytest.mark.parametrize("image", [_result8(), _result16()], ids=["8bit", "16bit"])
    def test_the_pixels_survive_the_round_trip(self, tmp_path, image):
        out = str(tmp_path / "result.dng")
        assert write_image(out, image)

        read_back = dng.read_linear(out)
        # Uncompressed and undeveloped: nothing is resampled, no colour is
        # twisted and no tone curve is applied, so the only thing between the
        # frame going in and coming out is the transfer function - see
        # `_ROUND_TRIP_TOLERANCE` for the two parts in 65535 that costs.
        assert _round_tripped(read_back, image)

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

        assert _round_tripped(dng.read_linear(out), grey)

    def test_an_alpha_channel_is_dropped(self, tmp_path):
        out = str(tmp_path / "alpha.dng")
        bgr = _result8()
        bgra = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
        assert write_image(out, bgra)

        read_back = dng.read_linear(out)
        # DNG raw data has no place for alpha, so the colour survives and the
        # channel does not.
        assert _round_tripped(read_back, bgr)

    def test_an_image_spanning_many_strips_round_trips(self, tmp_path, monkeypatch):
        # Rather than allocate the tens of megabytes a real multi-strip image
        # needs, the strip target is shrunk so a small image crosses it.
        monkeypatch.setattr(dng, "_STRIP_TARGET_BYTES", 64)
        image = _result16()
        out = str(tmp_path / "strips.dng")
        assert write_image(out, image)

        tags = _tags(out)
        assert len(tags[dng._STRIP_OFFSETS]) > 1, "expected more than one strip"
        assert _round_tripped(dng.read_linear(out), image)

    def test_a_written_dng_reloads_through_the_shared_read_path(self, tmp_path):
        out = str(tmp_path / "frame.dng")
        image = _result16()
        assert write_image(out, image)
        # read_image_any_depth is what the folder-input fusion methods use.
        assert _round_tripped(read_image_any_depth(out), image)


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

    def test_the_embedded_profile_asks_for_no_rendering(self, tmp_path):
        """The tags that stop a converter developing an already-developed frame.

        A raw converter's job is to render scene-referred data, so left to its
        own devices it puts a baseline tone curve, a black-point rendering and a
        baseline exposure on top of whatever it opens. On a fused result that is
        a second rendering, and it is what made a saved DNG open brighter and
        flatter than the frame it was saved from even once the
        LinearizationTable described the right curve.
        """
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())
        tags = _tags(out)

        # An identity tone curve - the two endpoints and nothing between them -
        # is how a profile says "no curve"; absent the tag the converter's own
        # baseline curve applies instead.
        assert tags[dng._PROFILE_TONE_CURVE] == [0.0, 0.0, 1.0, 1.0]
        assert tags[dng._DEFAULT_BLACK_RENDER] == [dng._DEFAULT_BLACK_RENDER_NONE]
        assert _rationals(tags[dng._BASELINE_EXPOSURE], signed=True) == 0.0
        assert _rationals(tags[dng._BASELINE_EXPOSURE_OFFSET], signed=True) == 0.0
        # Named, so a converter shows the result's own profile rather than the
        # camera's - the pixels left camera space during the develop.
        assert tags[dng._PROFILE_NAME] == dng._PROFILE_NAME_TEXT

    def test_no_camera_profile_may_be_substituted_for_the_embedded_one(self, tmp_path):
        """The file names its profile, and refuses every other.

        Naming the profile is only half of it: a converter that recognises a
        camera will reach for that camera's profile regardless, and a camera
        profile's matrix expects that sensor's RGB. These samples are sRGB, so
        the result is not a slight cast - a stack shot on a Nikon rendered
        magenta in Luminar Neo, which had matched the source camera and gone
        looking for its own profile for it.

        Matching calibration signatures are the spec's answer: a profile whose
        signature differs from the file's must not be applied, and Adobe signs
        its camera profiles with its own.
        """
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())
        tags = _tags(out)

        assert tags[dng._AS_SHOT_PROFILE_NAME] == dng._PROFILE_NAME_TEXT
        assert tags[dng._CAMERA_CALIBRATION_SIGNATURE] == dng._CALIBRATION_SIGNATURE
        assert tags[dng._PROFILE_CALIBRATION_SIGNATURE] == dng._CALIBRATION_SIGNATURE

        # The two stages between the samples and ColorMatrix1, both stated as
        # the identity rather than left to default, so that a converter has
        # nowhere left to insert a correction of its own.
        assert np.allclose(
            _rationals(tags[dng._CAMERA_CALIBRATION_1], signed=True).reshape(3, 3),
            np.eye(3),
        )
        assert np.allclose(_rationals(tags[dng._ANALOG_BALANCE]), [1.0, 1.0, 1.0])

    def test_the_forward_matrix_sends_the_neutral_to_d50(self, tmp_path):
        out = str(tmp_path / "result.dng")
        assert write_image(out, _result8())

        forward = _rationals(_tags(out)[dng._FORWARD_MATRIX_1], signed=True).reshape(3, 3)
        # DNG requires a ForwardMatrix to map the camera neutral onto the D50
        # white point of the connection space. AsShotNeutral is (1,1,1) here, so
        # the neutral is (1,1,1) and the requirement is that the rows sum to D50
        # - which is what lets a profile-aware converter use this matrix instead
        # of inverting ColorMatrix1 and guessing at the white balance.
        assert np.allclose(forward.sum(axis=1), [0.96422, 1.0, 0.82521], atol=2e-4)

    def test_a_mono_result_still_asks_for_no_rendering(self, tmp_path):
        out = str(tmp_path / "grey.dng")
        assert write_image(out, _result8()[:, :, 0])
        tags = _tags(out)

        # The profile is colorimetry and goes with the colour tags, but "the
        # tones are finished" is as true of one channel as of three.
        assert dng._PROFILE_TONE_CURVE not in tags
        assert dng._FORWARD_MATRIX_1 not in tags
        assert tags[dng._DEFAULT_BLACK_RENDER] == [dng._DEFAULT_BLACK_RENDER_NONE]
        assert _rationals(tags[dng._BASELINE_EXPOSURE], signed=True) == 0.0

    @pytest.mark.parametrize("image,bits", [(_result8(), 8), (_result16(), 16)])
    def test_the_stored_samples_are_scene_linear(self, tmp_path, image, bits):
        """The curve is applied to the pixels, not declared alongside them.

        A raw converter assumes the samples it reads are scene-linear light, and
        the pipeline's frames are display-referred, so one of the two has to
        give. Declaring the encoding through `LinearizationTable` is the tidier
        answer and is what this module used to do - but Luminar Neo does not read
        the tag, took the encoded samples for linear and applied its own gamma on
        top, and rendered a fused stack washed out next to the JPEG XL written
        from the same pixels. A converter cannot ignore an encoding that is not
        there.

        Checked against the EOTF itself rather than against the writer's own
        table, and BT.709 rather than sRGB, because the develop encodes with
        dcraw's default gamma - the other curve is what once made a saved frame
        darker than the raw it came from.
        """
        out = str(tmp_path / "result.dng")
        assert write_image(out, image)
        tags = _tags(out)

        assert dng._LINEARIZATION_TABLE not in tags
        # Always 16, whatever went in: linear light spends its codes on the
        # highlights, and 8 bits of it band the shadows into uselessness.
        assert tags[dng._BITS_PER_SAMPLE] == [16] * 3

        encoded = image[:, :, ::-1].astype(np.float64) / ((1 << bits) - 1)
        expected = np.where(
            encoded < 0.081, encoded / 4.5, ((encoded + 0.099) / 1.099) ** 2.222
        )
        # The writer uses the knee dcraw solves for rather than the figures the
        # standard quotes, which is a difference of a few parts in 65535.
        stored = _stored_samples(out).astype(np.float64)
        assert np.abs(stored - np.rint(expected * dng._LINEAR_MAX)).max() <= 8

        # WhiteLevel has to be what full scale linearises to, or the top of the
        # image would clip or fall short.
        assert tags[dng._WHITE_LEVEL] == [dng._LINEAR_MAX] * 3
        assert stored.max() <= dng._LINEAR_MAX

    def test_the_lossy_mode_still_declares_its_encoding(self, tmp_path):
        """The one mode that cannot store linear samples, and so has to describe them.

        DNG restricts lossy JPEG to 8-bit, and 8 bits of *linear* light collapses
        the whole shadow half of the range into two or three codes. So the lossy
        mode keeps display-referred samples and the table that explains them -
        which costs it nothing, because a reader able to open a lossy DNG at all
        supports the tag: Adobe's own lossy files are built the same way.
        """
        out = str(tmp_path / "proxy.dng")
        assert write_image(out, _result8())
        assert dng._LINEARIZATION_TABLE not in _tags(out)

        dng.set_compression(dng.COMPRESSION_LOSSY)
        assert write_image(out, _result8())
        tags = _tags(out)

        assert tags[dng._BITS_PER_SAMPLE] == [8] * 3
        table = np.asarray(tags[dng._LINEARIZATION_TABLE], dtype=np.int64)
        assert len(table) == 256
        assert table[-1] == dng._LINEAR_MAX
        assert tags[dng._WHITE_LEVEL] == [dng._LINEAR_MAX] * 3

    def test_the_curve_is_the_one_the_develop_applies(self, tmp_path):
        """The curve has to describe *this* pipeline, not a standard in general.

        The bug it guards against was a disagreement between two modules: the
        develop encoded with dcraw's gamma while the writer assumed sRGB, so the
        samples were linearised through a curve they had never been put through
        and a saved frame rendered some 13 levels in 255 darker than the raw it
        was made from. Nothing inside either module was wrong on its own, which
        is why the check has to span both.
        """
        # core.gpu_decode's output curve, on the scene-linear values the writer
        # turns each display-referred sample into. Encoding those has to give the
        # sample back, or the two modules disagree about what the pixels are.
        linear = dng._linearization_table(8).astype(np.float64) / dng._LINEAR_MAX
        encoded = np.where(linear < 0.018,
                           linear * 4.5,
                           1.099 * np.maximum(linear, 0.018) ** (1.0 / 2.222) - 0.099)
        assert np.abs(encoded * 255.0 - np.arange(256)).max() <= 0.5

    def test_the_reader_undoes_exactly_what_the_writer_applied(self, tmp_path):
        """The inverse is taken analytically, so it is worth pinning to the forward table.

        Near black the forward curve is many-to-one - it spreads one linear code
        over some 4.5 encoded ones - so the round trip cannot be exact there and
        the question is only how far off it lands.
        """
        forward = dng._linearization_table(16).astype(np.int64)
        recovered = dng._display_table().astype(np.int64)[forward]
        drift = np.abs(recovered - np.arange(dng._LINEAR_MAX + 1))
        assert drift.max() <= _ROUND_TRIP_TOLERANCE

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

        # The ramp has to come back as the ramp that went in. LibRaw's output
        # curve is dcraw's default gamma, which is the curve the
        # LinearizationTable now declares, so the two cancel and the render is
        # the stored samples again - where a table describing sRGB instead left
        # mid grey near 115, and no table at all would leave it near 186.
        assert row[0, 0] == 0
        assert row[-1, 0] == 255
        assert np.all(np.diff(row[:, 0].astype(np.int32)) >= 0)
        assert np.abs(row[:, 0] - _grey_ramp()[ramp.shape[0] // 2][:, 0].astype(np.int16)).max() <= 2


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
        assert _round_tripped(ImageStackLoader.read_image_bgr(out), image)

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


def _detailed(height=96, width=128):
    """An image with structure, so a codec that drops data cannot pass by luck.

    Random noise would compress to nothing useful and a flat field would compress
    to nothing at all; a gradient with edges in it is what a fused result looks
    like to an entropy coder.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    base = ((xs * 2 + ys) % 256).astype(np.uint8)
    edges = np.where((xs // 16 + ys // 16) % 2 == 0, 40, 0).astype(np.uint8)
    plane = np.clip(base.astype(np.int16) + edges, 0, 255).astype(np.uint8)
    return np.dstack([plane, np.roll(plane, 7, axis=1), np.roll(plane, 13, axis=0)])


class TestCompression:
    """What each compression mode promises about the pixels and the file."""

    def test_the_default_is_uncompressed(self):
        # Uncompressed costs no dependency and no encode time, and is what every
        # DNG written before the modes existed looks like.
        assert dng.DEFAULT_COMPRESSION == dng.COMPRESSION_NONE

    def test_only_round_trippable_modes_are_offered(self):
        offered = dng.available_compressions()
        assert dng.COMPRESSION_NONE in offered
        assert dng.COMPRESSION_LOSSY in offered
        # Writing lossless needs nothing optional; reading it back needs
        # imagecodecs, so the mode is offered only when it can do both.
        assert (dng.COMPRESSION_LOSSLESS in offered) == dng.lossless_available()
        assert all(mode in dng.VALID_COMPRESSIONS for mode in offered)

    @needs_lossless
    @pytest.mark.parametrize("image", [_detailed(), _detailed().astype(np.uint16) * 257],
                             ids=["8bit", "16bit"])
    def test_lossless_is_exactly_lossless_and_smaller(self, tmp_path, image):
        plain = str(tmp_path / "plain.dng")
        packed = str(tmp_path / "packed.dng")
        assert write_image(plain, image)
        assert dng.write(packed, image, compression=dng.COMPRESSION_LOSSLESS)

        read_back = dng.read_linear(packed)
        # The codec stores prediction errors, not approximations, so it adds
        # nothing to the transfer function's own couple of parts in 65535.
        assert _round_tripped(read_back, image)
        assert os.path.getsize(packed) < os.path.getsize(plain)

    @needs_lossless
    def test_lossless_greyscale_round_trips(self, tmp_path):
        grey = _detailed()[:, :, 0]
        out = str(tmp_path / "grey.dng")
        assert dng.write(out, grey, compression=dng.COMPRESSION_LOSSLESS)

        assert _round_tripped(dng.read_linear(out), grey)

    def test_lossy_is_close_but_not_exact(self, tmp_path):
        image = _detailed()
        out = str(tmp_path / "lossy.dng")
        assert dng.write(out, image, compression=dng.COMPRESSION_LOSSY)

        read_back = dng.read_linear(out)
        assert read_back is not None
        assert read_back.shape == image.shape
        error = np.abs(read_back.astype(int) - image.astype(int))
        assert error.max() > 0, "a lossy codec that changed nothing is not lossy"
        assert error.mean() < 8, f"quality 92 should stay close, got {error.mean()}"

    def test_lossy_narrows_sixteen_bit_because_dng_requires_it(self, tmp_path, capsys):
        out = str(tmp_path / "deep_lossy.dng")
        assert dng.write(out, _detailed().astype(np.uint16) * 257,
                         compression=dng.COMPRESSION_LOSSY)

        # DNG allows lossy JPEG for 8-bit data only, so the depth genuinely goes
        # - and the loss is announced rather than left for the user to discover.
        assert dng.probe(out)[2] == 8
        assert dng.read_linear(out).dtype == np.uint8
        assert "8-bit" in capsys.readouterr().out

    def test_a_mono_lossy_image_falls_back(self, tmp_path, capsys):
        out = str(tmp_path / "mono_lossy.dng")
        assert dng.write(out, _detailed()[:, :, 0], compression=dng.COMPRESSION_LOSSY)

        # LibRaw reads three components per pixel from a lossy DNG whatever the
        # file says, so a one-sample lossy file is one nothing would open.
        tags = _tags(out)
        assert tags[dng._COMPRESSION] != [dng._COMPRESSION_LOSSY_JPEG]
        assert "monochrome" in capsys.readouterr().out

    @pytest.mark.parametrize("mode", dng.VALID_COMPRESSIONS)
    def test_the_compression_tag_says_which_codec_was_used(self, tmp_path, mode):
        if mode == dng.COMPRESSION_LOSSLESS and not dng.lossless_available():
            pytest.skip("lossless needs imagecodecs")
        out = str(tmp_path / f"{mode}.dng")
        assert dng.write(out, _detailed(), compression=mode)

        tags = _tags(out)
        assert tags[dng._COMPRESSION] == [dng._COMPRESSION_CODES[mode]]
        # Whatever the codec, the pixels are still declared LinearRaw - the file
        # is a demosaiced raw, not a rendered image.
        assert tags[dng._PHOTOMETRIC] == [dng._LINEAR_RAW]

    def test_lossy_declares_the_version_that_introduced_it(self, tmp_path):
        out = str(tmp_path / "lossy.dng")
        assert dng.write(out, _detailed(), compression=dng.COMPRESSION_LOSSY)
        # Compression 34892 arrived in DNG 1.4, and the spec says to say so, or a
        # 1.1 reader would accept the file and fail on the strips.
        assert bytes(_tags(out)[dng._DNG_BACKWARD_VERSION]) == b"\x01\x04\x00\x00"

        plain = str(tmp_path / "plain.dng")
        assert dng.write(plain, _detailed(), compression=dng.COMPRESSION_NONE)
        assert bytes(_tags(plain)[dng._DNG_BACKWARD_VERSION]) == b"\x01\x01\x00\x00"

    def test_a_compressed_image_is_one_strip(self, tmp_path, monkeypatch):
        # LibRaw reads later compressed strips from wherever the codec left the
        # file pointer, not from StripOffsets, so more than one strip decodes to
        # a mostly black image. The strip target must not be able to force it.
        monkeypatch.setattr(dng, "_STRIP_TARGET_BYTES", 64)
        for mode in dng.available_compressions():
            out = str(tmp_path / f"strips_{mode}.dng")
            assert dng.write(out, _detailed(), compression=mode)
            strips = len(_tags(out)[dng._STRIP_OFFSETS])
            if mode == dng.COMPRESSION_NONE:
                assert strips > 1, "uncompressed should still be cut into strips"
            else:
                assert strips == 1, f"{mode} must be a single strip, got {strips}"

    def test_the_settings_are_module_level_and_validated(self):
        dng.set_compression(dng.COMPRESSION_NONE)
        assert dng.get_compression() == dng.COMPRESSION_NONE
        assert dng.set_lossy_quality(10_000) == dng.MAX_LOSSY_QUALITY
        assert dng.set_lossy_quality(-1) == dng.MIN_LOSSY_QUALITY
        assert dng.set_fast_load(False) is False

        with pytest.raises(ValueError):
            dng.set_compression("brotli")

    def test_the_module_setting_is_what_write_image_uses(self, tmp_path):
        # The save paths call write_image, which carries no codec argument, so
        # the setting is the only way the choice can reach the writer.
        dng.set_compression(dng.COMPRESSION_LOSSY)
        out = str(tmp_path / "via_setting.dng")
        assert write_image(out, _detailed())
        assert _tags(out)[dng._COMPRESSION] == [dng._COMPRESSION_LOSSY_JPEG]

    def test_lossless_cannot_be_selected_without_its_decoder(self, monkeypatch):
        monkeypatch.setattr(dng, "_LJPEG_AVAILABLE", False)
        assert dng.COMPRESSION_LOSSLESS not in dng.available_compressions()
        assert dng.lossless_unavailable_reason()
        # Refused rather than downgraded: a caller that asked for lossless and
        # silently got something else would only notice from the file size.
        with pytest.raises(RuntimeError):
            dng.set_compression(dng.COMPRESSION_LOSSLESS)

    @needs_rawpy
    @pytest.mark.parametrize("mode", dng.VALID_COMPRESSIONS)
    def test_libraw_renders_every_mode_the_same_way(self, tmp_path, mode):
        if mode == dng.COMPRESSION_LOSSLESS and not dng.lossless_available():
            pytest.skip("lossless needs imagecodecs")
        import rawpy

        image = _detailed(256, 256)
        rendered = []
        for name, compression in (("plain", dng.COMPRESSION_NONE), ("test", mode)):
            path = str(tmp_path / f"{name}.dng")
            assert dng.write(path, image, compression=compression)
            with open(path, "rb") as handle:
                with rawpy.imread(handle) as raw:
                    assert (raw.sizes.width, raw.sizes.height) == image.shape[1::-1]
                    rendered.append(raw.postprocess(
                        use_camera_wb=True, output_bps=8, no_auto_bright=True))

        difference = np.abs(rendered[1].astype(int) - rendered[0].astype(int)).mean()
        # The lossy mode is allowed a wide margin here because `_detailed` is a
        # worst case for it: a sawtooth whose three planes are rolled copies puts
        # the signal in the chroma, which 4:2:0 subsampling is exactly what
        # discards. The margin was 6 while the LinearizationTable declared sRGB -
        # a curve the pixels had not been through, which rendered the file
        # compressed towards black and shrank every difference measured on it
        # along with it. Against the corrected table the same file's error
        # measures at its true amplitude, near 11.
        limit = 14 if mode == dng.COMPRESSION_LOSSY else 0
        # A codec LibRaw mis-decodes does not fail loudly - it renders part of
        # the frame black - so the whole image is compared, not just the header.
        assert difference <= limit, f"{mode} renders differently: mean {difference}"
        # A mean alone would let a black band hide behind a generous limit, so
        # the overall level has to survive the codec too.
        assert abs(rendered[1].mean() - rendered[0].mean()) <= limit


class TestFastLoad:
    """The embedded preview: what it is, and that a reader can find it."""

    @staticmethod
    def _preview_ifd(path):
        """The SubIFD holding the preview, via utils.dng's own IFD reader."""
        tags = _tags(path)
        assert dng._SUB_IFDS in tags, "no SubIFDs tag, so no preview to find"
        offset = int(tags[dng._SUB_IFDS][0])
        with open(path, "rb") as handle:
            data = handle.read()
        (count,) = struct.unpack_from("<H", data, offset)
        sub = {}
        for index in range(count):
            tag, field_type, n = struct.unpack_from("<HHI", data, offset + 2 + 12 * index)
            payload = data[offset + 2 + 12 * index + 8:offset + 2 + 12 * index + 12]
            size = dng._READ_TYPE_SIZE.get(field_type, 1) * n
            if size > 4:
                (where,) = struct.unpack_from("<I", payload)
                payload = data[where:where + size]
            sub[tag] = payload[:size]
        return sub, data

    def test_it_is_off_the_main_image_not_in_place_of_it(self, tmp_path):
        image = _detailed(400, 600)
        out = str(tmp_path / "preview.dng")
        assert dng.write(out, image, fast_load=True)

        # The full-resolution image stays in IFD 0, so a file written by an
        # earlier version and one written now are read back the same way.
        assert _round_tripped(dng.read_linear(out), image)
        assert _tags(out)[dng._NEW_SUBFILE_TYPE] == [0]

        sub, _ = self._preview_ifd(out)
        assert struct.unpack_from("<I", sub[dng._NEW_SUBFILE_TYPE])[0] == 1
        assert struct.unpack_from("<H", sub[dng._PHOTOMETRIC])[0] == dng._YCBCR
        assert struct.unpack_from("<H", sub[dng._COMPRESSION])[0] == dng._COMPRESSION_JPEG
        assert struct.unpack_from("<I", sub[dng._PREVIEW_COLOR_SPACE])[0] == 2  # sRGB
        assert sub[dng._PREVIEW_APPLICATION_NAME].startswith(b"OpenFocus")

    def test_the_preview_is_a_decodable_half_size_jpeg(self, tmp_path):
        import cv2

        image = _detailed(400, 600)
        out = str(tmp_path / "preview.dng")
        assert dng.write(out, image, fast_load=True)

        sub, data = self._preview_ifd(out)
        width = struct.unpack_from("<I", sub[dng._IMAGE_WIDTH])[0]
        height = struct.unpack_from("<I", sub[dng._IMAGE_LENGTH])[0]
        assert (width, height) == (600 // 2, 400 // 2)

        offset = struct.unpack_from("<I", sub[dng._STRIP_OFFSETS])[0]
        length = struct.unpack_from("<I", sub[dng._STRIP_BYTE_COUNTS])[0]
        jpeg = data[offset:offset + length]
        assert jpeg[:2] == b"\xff\xd8", "the preview strip is not a JPEG"

        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == (height, width, 3)

        # It has to be a picture of *this* image, not merely a valid JPEG. The
        # absolute bound is loose because the test pattern is far harsher than a
        # photograph - it is built from sharp wraps and a checkerboard, which is
        # exactly what subsampled JPEG handles worst - so the comparison that
        # carries the weight is the relative one: the preview must sit much
        # closer to the downscaled original than to a shifted version of it.
        expected = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        matched = np.abs(decoded.astype(int) - expected.astype(int)).mean()
        shifted = np.abs(decoded.astype(int)
                         - np.roll(expected, 5, axis=1).astype(int)).mean()
        assert matched < 15, f"preview does not resemble the image: {matched}"
        assert matched < shifted / 3, f"preview is not aligned with the image: {matched} vs {shifted}"

    def test_it_can_be_turned_off(self, tmp_path):
        image = _detailed(400, 600)
        without = str(tmp_path / "without.dng")
        with_preview = str(tmp_path / "with.dng")
        assert dng.write(without, image, fast_load=False)
        assert dng.write(with_preview, image, fast_load=True)

        assert dng._SUB_IFDS not in _tags(without)
        assert os.path.getsize(with_preview) > os.path.getsize(without)
        # Off or on, the raw is untouched - the preview is an addition.
        assert np.array_equal(dng.read_linear(without), dng.read_linear(with_preview))

    def test_a_tiny_image_gets_no_preview(self, tmp_path):
        # Below the threshold the main image is already thumbnail-sized, so a
        # preview would cost bytes to save a decode that costs nothing.
        out = str(tmp_path / "tiny.dng")
        assert dng.write(out, _detailed(32, 32), fast_load=True)
        assert dng._SUB_IFDS not in _tags(out)

    def test_a_mono_result_still_gets_a_colour_preview(self, tmp_path):
        out = str(tmp_path / "mono.dng")
        assert dng.write(out, _detailed(400, 600)[:, :, 0], fast_load=True)

        sub, _ = self._preview_ifd(out)
        # Three channels, so nothing displaying the preview has to know the main
        # image was a single plane.
        assert struct.unpack_from("<H", sub[dng._SAMPLES_PER_PIXEL])[0] == 3

    @needs_rawpy
    def test_libraw_finds_it_as_a_thumbnail(self, tmp_path):
        import rawpy

        out = str(tmp_path / "preview.dng")
        assert dng.write(out, _detailed(400, 600), fast_load=True)

        # The point of the whole feature: a reader gets pixels without touching
        # the full-resolution LinearRaw.
        with open(out, "rb") as handle:
            with rawpy.imread(handle) as raw:
                thumb = raw.extract_thumb()
        assert thumb.format == rawpy.ThumbFormat.JPEG
        assert thumb.data[:2] == b"\xff\xd8"

    @needs_rawpy
    def test_a_file_without_one_has_no_thumbnail(self, tmp_path):
        import rawpy

        out = str(tmp_path / "bare.dng")
        assert dng.write(out, _detailed(400, 600), fast_load=False)
        with open(out, "rb") as handle:
            with rawpy.imread(handle) as raw:
                with pytest.raises(rawpy.LibRawNoThumbnailError):
                    raw.extract_thumb()


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
        assert _round_tripped(dng.read_linear(out), image)
        assert _round_tripped(dng.read(out), image)

    def test_a_camera_dng_cannot_be_developed(self, tmp_path, monkeypatch):
        out = tmp_path / "camera.dng"
        assert write_image(str(out), _result8())
        data = out.read_bytes()
        out.write_bytes(data.replace(dng.CAMERA_MODEL.encode(), b"SomeCam\x00", 1))

        monkeypatch.setattr(dng, "_RAWPY_AVAILABLE", False)
        assert dng.read(str(out)) is None


class TestSourceExif:
    """A source file's EXIF, carried into the DNG as tags of its own.

    DNG cannot be tagged after the fact the way a JPEG can - a TIFF's offsets all
    move when anything is inserted - so the block is taken apart and re-laid out
    while the file is being written. That is what these check: the tags arrive,
    they arrive as tags and not as a copied blob, the ones whose values are
    offsets into the source are decided on their merits, and the pixels are
    untouched by any of it.
    """

    @staticmethod
    def _source(path, maker_note=b"\x00\x01binary maker note"):
        """A small JPEG with a camera-like EXIF block, and that block.

        Pillow writes big-endian ("MM") blocks, which is what makes this a test
        of the byte-order conversion as well: nothing written here is.
        """
        from PIL import Image
        from PIL.TiffImagePlugin import IFDRational

        exif = Image.Exif()
        exif[dng._MAKE] = "Nikon"
        exif[dng._MODEL] = "Z 8"
        exif[dng._ARTIST] = "A Photographer"
        exif[dng._X_RESOLUTION] = IFDRational(300, 1)
        exif[dng._Y_RESOLUTION] = IFDRational(300, 1)
        exif[dng._RESOLUTION_UNIT] = 2
        exif[dng._DATE_TIME_ORIGINAL] = "2026:07:20 11:22:33"
        camera = exif.get_ifd(0x8769)
        camera[0x829A] = IFDRational(1, 200)      # ExposureTime
        camera[0x8827] = 400                      # ISOSpeedRatings
        camera[0x9003] = "2026:07:20 11:22:33"    # DateTimeOriginal
        camera[0x927C] = maker_note
        gps = exif.get_ifd(0x8825)
        gps[1] = "N"
        Image.new("RGB", (16, 16), (9, 9, 9)).save(str(path), exif=exif.tobytes())

        blob = exif.tobytes()
        blob = blob[len(b"Exif\x00\x00"):] if blob.startswith(b"Exif\x00\x00") else blob
        assert blob.startswith(b"MM"), "expected Pillow to write a big-endian block"
        return blob

    # A recognisable stand-in for the JPEG a Nikon note embeds. It opens with a
    # start-of-image marker, because half the point of removing it is that a
    # reader scanning for one must not find it.
    PREVIEW = b"\xff\xd8\xff" + b"not the fused result" + b"\xff\xd9"

    @classmethod
    def _nikon_note(cls, lens=b"NIKKOR Z 100mm\x00", values_at=None, preview=None):
        """A MakerNote in the shape Nikon writes: a header, then a TIFF of its own.

        Everything inside is measured from that embedded header rather than from
        the file around it, which is what makes such a note safe to write at a
        new position - and what utils.dng checks before it agrees to. Passing
        `values_at` overrides where the lens value claims to be, which is how
        that check is shown to be a check and not a guess at the header alone.

        With `preview`, the note also carries a preview IFD holding a JPEG, as a
        real one does - the part that has to be taken out on the way into a DNG.
        """
        entries = 1 if preview is None else 2
        lens_at = 8 + 2 + 12 * entries + 4
        preview_ifd_at = lens_at + len(lens)
        preview_at = preview_ifd_at + 2 + 12 * 2 + 4

        inner = struct.pack("<2sHI", b"II", 42, 8) + struct.pack("<H", entries)
        if preview is not None:
            inner += struct.pack("<HHII", 0x0011, 4, 1, preview_ifd_at)
        inner += struct.pack("<HHII", 0x0084, 2, len(lens),
                             lens_at if values_at is None else values_at)
        inner += struct.pack("<I", 0) + lens
        if preview is not None:
            inner += struct.pack("<H", 2)
            inner += struct.pack("<HHII", 0x0201, 4, 1, preview_at)
            inner += struct.pack("<HHII", 0x0202, 4, 1, len(preview))
            inner += struct.pack("<I", 0) + preview
        return b"Nikon\x00\x02\x11\x00\x00" + inner

    @staticmethod
    def _sub_ifd(path, pointer):
        """The sub-IFD `pointer` names, read back with utils.dng's own parser."""
        with open(path, "rb") as handle:
            data = handle.read()
        tags = _tags(path)
        assert pointer in tags, f"no pointer tag {pointer} in IFD 0"
        entries = dng._blob_entries(data, "<", int(tags[pointer][0]))
        return {tag: payload for tag, _type, _count, payload in entries}

    def test_the_camera_reaches_the_dng(self, tmp_path):
        source = self._source(tmp_path / "src.jpg")
        out = str(tmp_path / "with_exif.dng")
        assert dng.write(out, _result16(), exif=source)

        tags = _tags(out)
        # Who shot it travels; what shot it does not sit in IFD 0. Make and
        # Model are what a raw converter reads to pick a camera profile, and a
        # camera's profile expects that sensor's RGB - applied to samples that
        # are already sRGB it throws the frame towards magenta, which is what a
        # stack from a Nikon did in Luminar Neo while these said "Nikon".
        assert tags[dng._MAKE] == dng.CAMERA_MODEL
        assert tags[dng._MODEL] == dng.CAMERA_MODEL
        assert tags[dng._ARTIST] == "A Photographer"
        assert tags[dng._UNIQUE_CAMERA_MODEL] == dng.CAMERA_MODEL
        assert tags[dng._SOFTWARE] == dng.CAMERA_MODEL

    def test_the_shot_reaches_the_exif_ifd(self, tmp_path):
        source = self._source(tmp_path / "src.jpg")
        out = str(tmp_path / "with_exif.dng")
        assert dng.write(out, _result16(), exif=source)

        camera = self._sub_ifd(out, dng._EXIF_IFD)
        assert camera[0x9003].split(b"\x00")[0] == b"2026:07:20 11:22:33"
        # The source block is big-endian; everything written here is not, so a
        # value that survives the move proves the conversion happened.
        assert struct.unpack("<H", camera[0x8827])[0] == 400
        assert struct.unpack("<2I", camera[0x829A]) == (1, 200)

        assert 1 in self._sub_ifd(out, dng._GPS_IFD)

    def test_the_shot_reaches_ifd0_where_the_source_put_it_there_too(self, tmp_path):
        source = self._source(tmp_path / "src.jpg")
        out = str(tmp_path / "with_exif.dng")
        assert dng.write(out, _result16(), exif=source)

        tags = _tags(out)
        # A raw file records its nominal resolution and, per TIFF/EP, the
        # capture date in IFD 0 as well as in the EXIF IFD; a DNG made from it
        # should not come out filling one of the two.
        assert _rationals(tags[dng._X_RESOLUTION]) == 300
        assert _rationals(tags[dng._Y_RESOLUTION]) == 300
        assert tags[dng._RESOLUTION_UNIT] == [2]
        assert tags[dng._DATE_TIME_ORIGINAL] == "2026:07:20 11:22:33"
        # DateTime is when *this* file was made, so it stays the writer's own.
        assert tags[dng._DATE_TIME] != "2026:07:20 11:22:33"
        # Orientation is not copied: the result is already the right way up.
        assert tags[dng._ORIENTATION] == [1]

    def test_offsets_that_cannot_move_are_left_behind(self, tmp_path):
        source = self._source(tmp_path / "src.jpg")
        out = str(tmp_path / "with_exif.dng")
        assert dng.write(out, _result16(), exif=source)

        # A MakerNote whose offsets run from the file it came from decodes as
        # noise anywhere else; dropping it beats writing it broken.
        camera = self._sub_ifd(out, dng._EXIF_IFD)
        assert dng._MAKER_NOTE not in camera
        assert dng._INTEROP_IFD not in camera
        with open(out, "rb") as handle:
            assert b"binary maker note" not in handle.read()

    def test_a_self_contained_maker_note_travels(self, tmp_path):
        # A Nikon note brings its own TIFF header and is measured from it, so it
        # can be moved - and it is where the lens, the shutter count and the
        # rest of a Nikon's private record live.
        note = self._nikon_note()
        source = self._source(tmp_path / "src.jpg", maker_note=note)
        out = str(tmp_path / "with_exif.dng")
        assert dng.write(out, _result16(), exif=source)

        camera = self._sub_ifd(out, dng._EXIF_IFD)
        assert camera[dng._MAKER_NOTE] == note

    def test_the_preview_inside_a_note_does_not_travel(self, tmp_path):
        # A Nikon note embeds a JPEG of the frame it came from. A DNG is a raw
        # file, so a browser takes any preview it finds in preference to
        # decoding: left in, that JPEG is what gets shown instead of the fused
        # result. The rest of the note is what it is carried for and stays.
        note = self._nikon_note(preview=self.PREVIEW)
        source = self._source(tmp_path / "src.jpg", maker_note=note)
        out = str(tmp_path / "with_exif.dng")
        assert dng.write(out, _result16(), exif=source, fast_load=False)

        carried = self._sub_ifd(out, dng._EXIF_IFD)[dng._MAKER_NOTE]
        assert len(carried) == len(note), "the note may not change length"
        assert b"NIKKOR Z 100mm" in carried
        assert b"not the fused result" not in carried

        # Nothing anywhere in the file for a reader to mistake for a preview.
        with open(out, "rb") as handle:
            assert b"\xff\xd8\xff" not in handle.read()

    def test_the_written_preview_is_the_only_one(self, tmp_path):
        # With the fast-load preview on there is exactly one JPEG in the file,
        # and it is the one this module rendered from the result.
        note = self._nikon_note(preview=self.PREVIEW)
        source = self._source(tmp_path / "src.jpg", maker_note=note)
        out = str(tmp_path / "with_preview.dng")
        assert dng.write(out, _detailed(400, 400), exif=source, fast_load=True)

        tags = _tags(out)
        assert dng._SUB_IFDS in tags
        with open(out, "rb") as handle:
            data = handle.read()
        # The tag table is where the note sits, and it holds no image of its
        # own; the preview this module rendered is written past the strips.
        assert b"\xff\xd8\xff" not in data[:min(tags[dng._STRIP_OFFSETS])]
        assert b"not the fused result" not in data

    def test_a_note_whose_offsets_do_not_fit_is_left_behind(self, tmp_path):
        # The header shape is not taken on trust: a note claiming a value past
        # its own end is one whose offsets mean something else.
        note = self._nikon_note(values_at=4096)
        source = self._source(tmp_path / "src.jpg", maker_note=note)
        out = str(tmp_path / "with_exif.dng")
        assert dng.write(out, _result16(), exif=source)

        assert dng._MAKER_NOTE not in self._sub_ifd(out, dng._EXIF_IFD)

    def test_the_pixels_are_the_same_either_way(self, tmp_path):
        source = self._source(tmp_path / "src.jpg")
        image = _result16()
        plain = str(tmp_path / "plain.dng")
        tagged = str(tmp_path / "tagged.dng")
        assert dng.write(plain, image)
        assert dng.write(tagged, image, exif=source)

        assert _round_tripped(dng.read_linear(tagged), image)
        assert _round_tripped(dng.read_linear(plain), image)
        assert os.path.getsize(tagged) > os.path.getsize(plain)

    def test_write_image_passes_the_source_through(self, tmp_path):
        source_jpg = tmp_path / "src.jpg"
        self._source(source_jpg)
        out = str(tmp_path / "processed.dng")

        # The stack export hands write_image the file each frame came from; DNG
        # is the one container that has to receive it before the encode.
        assert write_image(out, _result16(), source_path=str(source_jpg))
        tags = _tags(out)
        # Artist rather than Make, because Make is one of the two tags a source
        # block deliberately does not bring with it - see
        # `test_the_camera_reaches_the_dng`. It carries the same proof: it is in
        # the source and nowhere else, so it can only have come from the block.
        assert tags[dng._ARTIST] == "A Photographer"
        assert dng._EXIF_IFD in tags

    def test_a_block_that_is_not_exif_is_ignored(self, tmp_path):
        out = str(tmp_path / "junk.dng")
        image = _result8()
        for blob in (b"", b"not a tiff", b"II*\x00\xff\xff\xff\xff", b"MM\x00*" + b"\x00" * 4):
            assert dng.write(out, image, exif=blob)
            assert _round_tripped(dng.read_linear(out), image)
            assert _tags(out)[dng._MAKE] == dng.CAMERA_MODEL

    @needs_rawpy
    def test_libraw_still_opens_a_tagged_file(self, tmp_path):
        import rawpy

        source = self._source(tmp_path / "src.jpg")
        image = _detailed(128, 128)
        rendered = []
        for name, blob in (("plain", None), ("tagged", source)):
            path = str(tmp_path / f"{name}.dng")
            assert dng.write(path, image, exif=blob)
            with open(path, "rb") as handle:
                with rawpy.imread(handle) as raw:
                    rendered.append(raw.postprocess(
                        use_camera_wb=True, output_bps=8, no_auto_bright=True))
        # The extra IFDs must not disturb what a raw converter reads.
        assert np.array_equal(rendered[0], rendered[1])


class TestPartialWrites:
    def test_a_failed_write_leaves_no_file(self, tmp_path, monkeypatch):
        out = str(tmp_path / "broken.dng")

        # The file is streamed rather than assembled first, so a failure part
        # way through would otherwise leave a header with no pixels behind it -
        # a file that parses far enough to be opened and then cannot be read.
        def explode(*args, **kwargs):
            raise MemoryError("no room for the strip")

        monkeypatch.setattr(dng, "_file_order", explode)
        assert dng.write(out, _result16()) is False
        assert not os.path.exists(out)


class TestCameraDevelop:
    """Which develop a camera DNG gets, and what happens when the GPU declines.

    A DNG OpenFocus wrote is read from its strips and never reaches this; a
    camera one is developed, and that develop is the expensive half of loading a
    RAW stack. `utils.dng.read` used to go straight to LibRaw while the stack
    loader used the GPU, so the same file developed differently depending on
    which path opened it. Both now ask core.gpu_decode.

    Stubs stand in for rawpy, because what is under test is the routing rather
    than either develop.
    """

    class _FakeRaw:
        """Just enough of a rawpy object to record whether LibRaw was asked."""

        def __init__(self):
            self.calls = []

        def postprocess(self, **options):
            self.calls.append(options)
            return np.dstack([np.full((4, 4), 10, np.uint16),
                              np.full((4, 4), 20, np.uint16),
                              np.full((4, 4), 30, np.uint16)])

    @pytest.fixture
    def gpu(self, monkeypatch):
        from core import gpu_decode
        monkeypatch.setattr(gpu_decode, "is_available", lambda: True)
        return gpu_decode

    def test_the_gpu_result_is_used_when_there_is_one(self, gpu, monkeypatch):
        developed = np.zeros((4, 4, 3), np.uint16)
        monkeypatch.setattr(gpu, "postprocess_raw", lambda raw, bps: developed)
        raw = self._FakeRaw()

        assert dng._develop(raw, 16) is developed
        assert not raw.calls, "LibRaw was asked for a frame the GPU had already produced"

    def test_libraw_finishes_what_the_gpu_will_not(self, gpu, monkeypatch):
        # postprocess_raw returns None for an unusual sensor, a busy GPU or one
        # that is out of memory - all of which have to end in a frame, not a
        # failed load.
        monkeypatch.setattr(gpu, "postprocess_raw", lambda raw, bps: None)
        raw = self._FakeRaw()

        developed = dng._develop(raw, 16)
        assert raw.calls == [{"use_camera_wb": True, "output_bps": 16}]
        # RGB from LibRaw, BGR to the pipeline.
        assert tuple(developed[0, 0]) == (30, 20, 10)

    def test_a_gpu_failure_is_not_a_failed_frame(self, gpu, monkeypatch):
        def explode(raw, bps):
            raise RuntimeError("CUDA error")

        monkeypatch.setattr(gpu, "postprocess_raw", explode)
        raw = self._FakeRaw()

        assert dng._develop(raw, 8) is not None
        assert raw.calls, "the frame should have fallen through to LibRaw"
