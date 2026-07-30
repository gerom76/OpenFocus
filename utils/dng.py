"""Adobe DNG reading and writing.

DNG is a TIFF-based raw container, and the two directions are asymmetric enough
that they are worth describing separately.

**Reading** covers two quite different kinds of file that share the extension. A
camera DNG (or one produced by Adobe DNG Converter) holds undemosaiced sensor
data, and is developed by LibRaw through `rawpy` exactly as `.nef` is - the
loader routes it through the same RAW path, so it also picks up the GPU
postprocess and the 8/16-bit depth mode. A DNG written by OpenFocus holds
already-demosaiced pixels, and is read back verbatim from its strips by this
module: no development, no white balance, no tone curve, so a saved result
reloads bit-for-bit as the frame that was saved.

**Writing** produces an uncompressed *linear* DNG - `PhotometricInterpretation`
= LinearRaw, one sample per channel, no CFA - because a fused result is already
demosaiced and there is no sensor mosaic left to describe. OpenCV cannot write
DNG and LibRaw cannot write at all, so the container is assembled here: an
uncompressed TIFF is a length-prefixed tag table followed by pixel strips, and
the DNG-specific part is the handful of extra tags that tell a raw converter how
to interpret those pixels.

Those tags are what make the file mean something to Lightroom or RawTherapee
rather than merely parse:

* `ColorMatrix1` with `CalibrationIlluminant1` = D65 declares the data to be in
  sRGB primaries, so no colour twist is applied on top of it.
* `AsShotNeutral` = (1, 1, 1) declares it already neutral, so no white balance
  is applied either.
* `LinearizationTable` carries the sRGB transfer function. The pipeline's frames
  are display-referred (sRGB-encoded), while a raw converter assumes the values
  it reads are scene-linear; without the table it would apply its own tone curve
  on top of the encoding and render the image far too bright. The table is the
  tag meant for exactly this, and it keeps the stored samples untouched - the
  alternative, linearising the pixels on the way out, would cost precision and
  band badly on 8-bit input.

Unlike JPEG XL, nothing optional is needed to *write* DNG - only numpy. Reading
a camera DNG needs `rawpy`, which is a core requirement, so `is_available()`
tracks that the same way the loader's RAW support does.
"""

import datetime
import os
import struct
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Extensions routed to this module instead of cv2.imwrite / cv2.imdecode.
EXTENSIONS = (".dng",)

# Written into UniqueCameraModel, and the marker `read` uses to recognise its own
# output and take the verbatim path instead of developing the file.
CAMERA_MODEL = "OpenFocus"

# Strips are sized to about this many bytes. A single strip spanning a 24 MP
# 16-bit image would be a 144 MB run, which some readers handle poorly; a few
# megabytes per strip is what camera DNGs use.
_STRIP_TARGET_BYTES = 8 << 20

try:
    import rawpy
    _RAWPY_AVAILABLE = True
except ImportError:
    rawpy = None
    _RAWPY_AVAILABLE = False


# ----------------------------------------------------------------------
# TIFF field types and tags
# ----------------------------------------------------------------------
_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL, _SRATIONAL = 1, 2, 3, 4, 5, 10

_NEW_SUBFILE_TYPE = 254
_IMAGE_WIDTH = 256
_IMAGE_LENGTH = 257
_BITS_PER_SAMPLE = 258
_COMPRESSION = 259
_PHOTOMETRIC = 262
_MAKE = 271
_MODEL = 272
_STRIP_OFFSETS = 273
_ORIENTATION = 274
_SAMPLES_PER_PIXEL = 277
_ROWS_PER_STRIP = 278
_STRIP_BYTE_COUNTS = 279
_PLANAR_CONFIG = 284
_SOFTWARE = 305
_DATE_TIME = 306
_SAMPLE_FORMAT = 339
_DNG_VERSION = 50706
_DNG_BACKWARD_VERSION = 50707
_UNIQUE_CAMERA_MODEL = 50708
_LINEARIZATION_TABLE = 50712
_WHITE_LEVEL = 50717
_COLOR_MATRIX_1 = 50721
_AS_SHOT_NEUTRAL = 50728
_CALIBRATION_ILLUMINANT_1 = 50778

# PhotometricInterpretation for demosaiced raw data - the whole point of a
# linear DNG, as opposed to 32803 (CFA) for a sensor mosaic.
_LINEAR_RAW = 34892

# CalibrationIlluminant code for D65, the white point sRGB is defined against.
_ILLUMINANT_D65 = 21

# Everything the LinearizationTable maps into, and therefore WhiteLevel. 8-bit
# samples are given the same 16-bit linear range as 16-bit ones: linearising
# sRGB compresses the shadows hard, and 256 output levels there would band.
_LINEAR_MAX = 65535

# XYZ (D65) -> linear sRGB. ColorMatrix1 is defined in that direction: it takes
# XYZ under the calibration illuminant into the "camera" space, which for this
# writer is sRGB itself.
_XYZ_D65_TO_SRGB = (
    (3.2406, -1.5372, -0.4986),
    (-0.9689, 1.8758, 0.0415),
    (0.0557, -0.2040, 1.0570),
)

# Denominator for the SRATIONAL colour matrix entries; six digits is well past
# the precision the matrix itself is quoted to.
_MATRIX_DENOMINATOR = 1000000

# DNG 1.4 is claimed so monochrome results are legal; the backward version stays
# at 1.1 because the layout used here - full-resolution image in IFD 0, no
# SubIFDs - is the one DNG has accepted since 1.0.
_DNG_VERSION_BYTES = bytes((1, 4, 0, 0))
_DNG_BACKWARD_VERSION_BYTES = bytes((1, 1, 0, 0))


def is_available() -> bool:
    """Whether this build can read DNG.

    Writing needs nothing beyond numpy, but a build that cannot develop a camera
    DNG cannot honestly offer the format, so this tracks `rawpy` - the same
    dependency the loader's RAW formats have.
    """
    return _RAWPY_AVAILABLE


def is_dng(extension: str) -> bool:
    """Whether a file extension names the DNG container."""
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext in EXTENSIONS


def unavailable_reason() -> str:
    """One line explaining why DNG is off, for logs and message boxes."""
    if _RAWPY_AVAILABLE:
        return ""
    return "DNG support needs the 'rawpy' package: pip install rawpy"


def extensions() -> tuple:
    """The extensions this build can handle - empty when rawpy is absent.

    Mirrors utils.jxl.extensions, so the loader and the folder-scanning fusion
    methods can fold DNG into their supported sets without each repeating the
    availability check.
    """
    return EXTENSIONS if _RAWPY_AVAILABLE else ()


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def _srgb_linearization_table(bits: int) -> np.ndarray:
    """The sRGB EOTF, tabulated for every value a sample of `bits` can hold.

    Entry *i* is the scene-linear value that stored sample *i* stands for, scaled
    to 0..`_LINEAR_MAX`. This is what lets the samples themselves be written
    untouched: the encoding is described rather than undone.
    """
    encoded = np.linspace(0.0, 1.0, 1 << bits, dtype=np.float64)
    linear = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )
    return np.rint(linear * _LINEAR_MAX).astype(np.uint16)


def _pack(field_type: int, values) -> bytes:
    """Serialise a tag's values as little-endian TIFF field data."""
    if field_type == _BYTE:
        return bytes(values)
    if field_type == _ASCII:
        return values.encode("ascii", "replace") + b"\x00"
    if field_type == _SHORT:
        return np.asarray(values, dtype="<u2").tobytes()
    if field_type == _LONG:
        return np.asarray(values, dtype="<u4").tobytes()
    if field_type == _RATIONAL:
        return np.asarray(values, dtype="<u4").tobytes()
    if field_type == _SRATIONAL:
        return np.asarray(values, dtype="<i4").tobytes()
    raise ValueError(f"Unsupported TIFF field type: {field_type}")


def _count(field_type: int, values) -> int:
    """The TIFF `count` for a tag - the number of values, not of bytes.

    RATIONALs are stored as numerator/denominator pairs, so the pair is one
    value; ASCII counts the terminating NUL.
    """
    if field_type == _ASCII:
        return len(values.encode("ascii", "replace")) + 1
    if field_type == _BYTE:
        return len(values)
    size = len(np.asarray(values).ravel())
    if field_type in (_RATIONAL, _SRATIONAL):
        return size // 2
    return size


def _rational_matrix(matrix) -> list:
    """Flatten a 3x3 float matrix into SRATIONAL numerator/denominator pairs."""
    flat = []
    for row in matrix:
        for value in row:
            flat.append(int(round(value * _MATRIX_DENOMINATOR)))
            flat.append(_MATRIX_DENOMINATOR)
    return flat


def _build_ifd(entries: list, strip_count: int) -> Tuple[bytes, list]:
    """Lay out an IFD and its out-of-line values, and say where the strips land.

    `entries` are (tag, type, values) triples; StripOffsets is passed in with
    placeholder zeros, because a strip's offset is only known once the table that
    records it has been sized. Returns the assembled bytes up to the start of the
    pixel data, and the byte position of each strip offset within them so the
    caller can patch the real values in.

    TIFF requires the entries be sorted by tag, and any value longer than the
    four bytes an entry has room for to live elsewhere in the file and be
    referenced by offset.
    """
    entries = sorted(entries, key=lambda item: item[0])

    ifd_offset = 8  # straight after the 8-byte TIFF header
    ifd_size = 2 + 12 * len(entries) + 4  # count, entries, next-IFD pointer
    values_offset = ifd_offset + ifd_size

    table = bytearray()
    values = bytearray()
    # Where, in the finished file, each strip offset is written. Filled for the
    # StripOffsets tag alone, whose values are patched after the layout is fixed.
    strip_offset_positions: list = []

    table += struct.pack("<H", len(entries))
    for tag, field_type, raw in entries:
        payload = _pack(field_type, raw)
        count = _count(field_type, raw)
        entry = struct.pack("<HHI", tag, field_type, count)
        if len(payload) <= 4:
            table += entry + payload.ljust(4, b"\x00")
            if tag == _STRIP_OFFSETS:
                # Single strip: the offset sits inline in the entry itself. The
                # position is file-absolute, hence past the header.
                strip_offset_positions.append(ifd_offset + len(table) - 4)
        else:
            position = values_offset + len(values)
            table += entry + struct.pack("<I", position)
            if tag == _STRIP_OFFSETS:
                strip_offset_positions.extend(position + 4 * i for i in range(strip_count))
            values += payload
            if len(values) % 2:
                values += b"\x00"  # keep the next value word-aligned
    table += struct.pack("<I", 0)  # no further IFDs

    header = struct.pack("<2sHI", b"II", 42, ifd_offset)
    return bytes(header + table + values), strip_offset_positions


def encode(image: np.ndarray, software: str = CAMERA_MODEL) -> bytes:
    """Encode a BGR (or greyscale) image as an uncompressed linear DNG.

    The samples are written exactly as they arrive - the depth is whatever the
    array carries, 8 or 16 bits - and it is the tags that describe what they
    mean; see the module docstring for which ones and why.
    """
    if image is None:
        raise ValueError("No image to encode.")

    # DNG has no place for an alpha channel in raw data, so it is dropped rather
    # than written as an extra sample no converter would look at.
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    if image.ndim == 2:
        samples = 1
        data = image
    elif image.ndim == 3 and image.shape[2] == 3:
        samples = 3
        # The pipeline carries BGR; DNG, like TIFF, stores channels in order.
        data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape for DNG: {image.shape}")

    if data.dtype == np.uint8:
        bits = 8
    elif data.dtype == np.uint16:
        bits = 16
    else:
        raise ValueError(f"Unsupported image dtype for DNG: {data.dtype}")

    data = np.ascontiguousarray(data)
    height, width = data.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    row_bytes = width * samples * (bits // 8)
    rows_per_strip = max(1, min(height, _STRIP_TARGET_BYTES // max(1, row_bytes)))
    strip_count = (height + rows_per_strip - 1) // rows_per_strip
    strip_byte_counts = [
        row_bytes * min(rows_per_strip, height - first)
        for first in range(0, height, rows_per_strip)
    ]

    now = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    entries = [
        (_NEW_SUBFILE_TYPE, _LONG, [0]),  # this is the full-resolution image
        (_IMAGE_WIDTH, _LONG, [width]),
        (_IMAGE_LENGTH, _LONG, [height]),
        (_BITS_PER_SAMPLE, _SHORT, [bits] * samples),
        (_COMPRESSION, _SHORT, [1]),  # uncompressed
        (_PHOTOMETRIC, _SHORT, [_LINEAR_RAW]),
        (_MAKE, _ASCII, CAMERA_MODEL),
        (_MODEL, _ASCII, CAMERA_MODEL),
        (_STRIP_OFFSETS, _LONG, [0] * strip_count),  # patched after layout
        (_ORIENTATION, _SHORT, [1]),
        (_SAMPLES_PER_PIXEL, _SHORT, [samples]),
        (_ROWS_PER_STRIP, _LONG, [rows_per_strip]),
        (_STRIP_BYTE_COUNTS, _LONG, strip_byte_counts),
        (_PLANAR_CONFIG, _SHORT, [1]),  # interleaved
        (_SOFTWARE, _ASCII, software),
        (_DATE_TIME, _ASCII, now),
        (_SAMPLE_FORMAT, _SHORT, [1] * samples),  # unsigned integer
        (_DNG_VERSION, _BYTE, _DNG_VERSION_BYTES),
        (_DNG_BACKWARD_VERSION, _BYTE, _DNG_BACKWARD_VERSION_BYTES),
        (_UNIQUE_CAMERA_MODEL, _ASCII, CAMERA_MODEL),
        (_LINEARIZATION_TABLE, _SHORT, _srgb_linearization_table(bits)),
        (_WHITE_LEVEL, _LONG, [_LINEAR_MAX] * samples),
    ]

    if samples == 3:
        # Colorimetry only means anything for a colour image. A monochrome DNG
        # carries no matrix and no neutral, which is what DNG 1.4 expects.
        entries.append((_COLOR_MATRIX_1, _SRATIONAL, _rational_matrix(_XYZ_D65_TO_SRGB)))
        entries.append((_CALIBRATION_ILLUMINANT_1, _SHORT, [_ILLUMINANT_D65]))
        entries.append((_AS_SHOT_NEUTRAL, _RATIONAL, [1, 1, 1, 1, 1, 1]))

    prefix, strip_offset_positions = _build_ifd(entries, strip_count)

    blob = bytearray(prefix)
    offsets = []
    for first in range(0, height, rows_per_strip):
        offsets.append(len(blob))
        blob += data[first:first + rows_per_strip].tobytes()

    for position, offset in zip(strip_offset_positions, offsets):
        blob[position:position + 4] = struct.pack("<I", offset)
    return bytes(blob)


def write(file_path: str, image: np.ndarray, software: str = CAMERA_MODEL) -> bool:
    """Encode and write a DNG file. Returns False instead of raising.

    Mirrors what cv2.imwrite gives the callers in utils.image_utils: a boolean,
    with the reason printed, so a failed DNG save is reported through the same
    "could not write" path as any other format.
    """
    try:
        payload = encode(image, software=software)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[DNG] Could not encode {os.path.basename(file_path)}: {exc}", flush=True)
        return False

    try:
        with open(file_path, "wb") as handle:
            handle.write(payload)
    except OSError as exc:
        print(f"[DNG] Could not write {os.path.basename(file_path)}: {exc}", flush=True)
        return False
    return True


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
# The scalar field types worth unpacking. RATIONAL (5) and SRATIONAL (10) are
# absent on purpose: nothing needed to find the strips is stored as a fraction,
# so they are left as raw bytes rather than decoded and thrown away.
_UNPACK = {1: "<B", 3: "<H", 4: "<I", 6: "<b", 8: "<h", 9: "<i", 11: "<f", 12: "<d"}
_READ_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _read_ifd0(path: str) -> Optional[Dict[int, object]]:
    """Read the tags of a little-endian TIFF's first IFD, or None if it is not one.

    Only enough of TIFF is parsed to recognise this module's own output and find
    its strips: little-endian files, no rationals, no following IFDs. A camera
    DNG is big-endian as often as not and keeps its raw data in a SubIFD, so it
    simply fails one of the checks in `_is_own_output` and goes to LibRaw, which
    is where it belongs anyway.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) < 8 or head[:2] != b"II":
                return None
            magic, ifd_offset = struct.unpack("<HI", head[2:8])
            if magic != 42:
                return None
            handle.seek(ifd_offset)
            raw_count = handle.read(2)
            if len(raw_count) < 2:
                return None
            (entry_count,) = struct.unpack("<H", raw_count)
            table = handle.read(12 * entry_count)
            if len(table) < 12 * entry_count:
                return None

            tags: Dict[int, object] = {}
            for index in range(entry_count):
                tag, field_type, count = struct.unpack_from("<HHI", table, 12 * index)
                payload = table[12 * index + 8:12 * index + 12]
                size = _READ_TYPE_SIZE.get(field_type)
                if size is None:
                    continue
                total = size * count
                if total > 4:
                    (offset,) = struct.unpack("<I", payload)
                    handle.seek(offset)
                    payload = handle.read(total)
                    if len(payload) < total:
                        return None
                else:
                    payload = payload[:total]

                if field_type in (2, 7):
                    tags[tag] = payload.split(b"\x00")[0].decode("ascii", "replace")
                elif field_type in _UNPACK:
                    fmt = _UNPACK[field_type]
                    tags[tag] = [
                        struct.unpack_from(fmt, payload, size * i)[0] for i in range(count)
                    ]
                else:
                    tags[tag] = payload
            return tags
    except (OSError, struct.error):
        return None


def _is_own_output(tags: Dict[int, object]) -> bool:
    """Whether these IFD 0 tags describe a DNG this module wrote.

    All three conditions matter. A camera DNG names its camera in
    UniqueCameraModel, and puts a small preview - not LinearRaw, usually
    compressed - in IFD 0, so it cannot pass by accident.
    """
    return (
        tags.get(_UNIQUE_CAMERA_MODEL) == CAMERA_MODEL
        and tags.get(_PHOTOMETRIC) == [_LINEAR_RAW]
        and tags.get(_COMPRESSION) == [1]
    )


def probe(path: str) -> Optional[Tuple[int, int, int]]:
    """(width, height, bits) of an OpenFocus linear DNG, read from its header.

    Lets the loader size a `.dng` stack up front without decoding a frame, and
    without opening it through LibRaw only to be told a depth the file does not
    actually have. Returns None for a camera DNG, whose dimensions the loader
    gets from rawpy instead.
    """
    tags = _read_ifd0(path)
    if tags is None or not _is_own_output(tags):
        return None
    try:
        return (
            int(tags[_IMAGE_WIDTH][0]),
            int(tags[_IMAGE_LENGTH][0]),
            int(tags[_BITS_PER_SAMPLE][0]),
        )
    except (KeyError, IndexError, TypeError):
        return None


def read_linear(path: str) -> Optional[np.ndarray]:
    """Read an OpenFocus linear DNG straight from its strips, as BGR.

    No development happens: the samples come back at the depth they were stored
    at, in the order they were stored in, so saving a frame and loading it again
    is an identity. Returns None for anything that is not this module's own
    output, which is how `read` decides to hand the file to LibRaw instead.
    """
    tags = _read_ifd0(path)
    if tags is None or not _is_own_output(tags):
        return None

    try:
        width = int(tags[_IMAGE_WIDTH][0])
        height = int(tags[_IMAGE_LENGTH][0])
        samples = int(tags[_SAMPLES_PER_PIXEL][0])
        bits = [int(value) for value in tags[_BITS_PER_SAMPLE]]
        offsets = [int(value) for value in tags[_STRIP_OFFSETS]]
        counts = [int(value) for value in tags[_STRIP_BYTE_COUNTS]]
    except (KeyError, IndexError, TypeError):
        return None

    if tags.get(_PLANAR_CONFIG, [1]) != [1] or len(set(bits)) != 1:
        return None
    dtype = {8: np.uint8, 16: np.uint16}.get(bits[0])
    if dtype is None or samples not in (1, 3) or len(offsets) != len(counts):
        return None
    if width <= 0 or height <= 0:
        return None
    if sum(counts) != width * height * samples * (bits[0] // 8):
        return None

    try:
        with open(path, "rb") as handle:
            chunks = []
            for offset, count in zip(offsets, counts):
                handle.seek(offset)
                chunk = handle.read(count)
                if len(chunk) < count:
                    return None
                chunks.append(chunk)
    except OSError:
        return None

    # Samples are little-endian on disk regardless of the host, so the dtype is
    # byte-ordered explicitly and then normalised to native for the pipeline.
    flat = np.frombuffer(b"".join(chunks), dtype=np.dtype(dtype).newbyteorder("<"))
    pixels = flat.reshape(height, width, samples).astype(dtype, copy=False)
    if samples == 1:
        return np.ascontiguousarray(pixels[:, :, 0])
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


def read(path: str, output_bps: int = 16) -> Optional[np.ndarray]:
    """Read a DNG as BGR, or None if it cannot be read.

    An OpenFocus linear DNG is returned verbatim; anything else is a camera raw
    and is developed by LibRaw at `output_bps` bits, with the camera's own white
    balance, which is what `.nef` already gets.

    Returns None - rather than raising - for a missing, truncated or unreadable
    file, because the callers report a failed frame themselves and carry on with
    the rest of the stack.
    """
    linear = read_linear(path)
    if linear is not None:
        return linear
    if not _RAWPY_AVAILABLE:
        return None
    try:
        # A file object is passed so paths with non-ASCII characters work, the
        # same way the loader's RAW path does it.
        with open(path, "rb") as handle:
            with rawpy.imread(handle) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=output_bps)
        # RGB->BGR is a channel swap, done in place so a full-frame temporary is
        # not added to the heaviest allocation on the load path.
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR, dst=rgb)
    except Exception:  # pylint: disable=broad-except
        return None
