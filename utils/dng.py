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

**Writing** produces a *linear* DNG - `PhotometricInterpretation` = LinearRaw,
one sample per channel, no CFA - because a fused result is already demosaiced
and there is no sensor mosaic left to describe. OpenCV cannot write DNG and
LibRaw cannot write at all, so the container is assembled here: a TIFF is a
length-prefixed tag table followed by pixel strips, and the DNG-specific part is
the handful of extra tags that tell a raw converter how to interpret those
pixels.

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

Compression
-----------
DNG is strict about which codec may carry which kind of data, and the rules
decide the three modes offered here (`COMPRESSION_*`):

- ``none``     `Compression` = 1. What this module wrote before the modes
               existed, and still the default: no dependency, no encode cost,
               and the strips are the pixels.
- ``lossless`` `Compression` = 7, lossless Huffman JPEG. The *only* lossless
               codec DNG permits for 16-bit LinearRaw data - Deflate (8) is
               restricted by the spec to floating point, 32-bit integer,
               transparency mask and depth map data, so it is not an option for
               integer image data however well it would compress. Costs nothing
               in quality and takes a fused master to roughly 50-55% of its
               uncompressed size at 16 bits, 35-40% at 8.
- ``lossy``    `Compression` = 34892, baseline DCT JPEG. The spec allows this
               only for **8-bit** LinearRaw, so choosing it narrows a 16-bit
               result on the way out; `encode` says so rather than doing it
               quietly. Meant for proxies, not for masters.

A compressed image is written as one strip - not for tidiness, but because
LibRaw walks the strips of a compressed *stripped* DNG by continuing from
wherever the codec left the file pointer rather than by consulting
`StripOffsets`. A multi-strip lossless file therefore decodes its first strip
and renders the rest black. `StripByteCounts` also holds compressed lengths, so
the strip has to be encoded before the tag table around it can be laid out.

The lossless codestream is produced by utils.ljpeg, written for this, rather
than by a library: see that module for why the single-component encoding the
spec would have allowed - and which `imagecodecs` could have written - is not
usable in practice.

Fast-load preview
-----------------
`fast_load` embeds a half-resolution, JPEG-compressed rendering of the result in
a SubIFD marked `NewSubFileType` = 1, alongside the `Preview*` tags that say
what it is and which colour space it is in. Without it a viewer has nothing to
show but the LinearRaw itself, so every thumbnail costs a full decode of the
main image - and for a fused stack that is the slowest thing in the file. With
it, browsers, Explorer and raw converters draw from a few hundred kilobytes of
baseline JPEG.

It is reached through the `SubIFDs` tag rather than by moving the raw out of
IFD 0: DNG only *recommends* a thumbnail in the first IFD, and keeping the
full-resolution image where it has always been means files written by earlier
versions still read back through the same path.

This is not Adobe's "Embed Fast Load Data", which is a partially-processed
Camera Raw cache in a private SubIFD and is not part of the DNG specification;
its contents cannot be reproduced from outside Adobe's converter. What is
embedded here is the spec's own preview mechanism, which is what gives
non-Adobe readers - and Adobe's own browsers - a fast path to pixels.

Dependencies
------------
Every mode *writes* with numpy and OpenCV alone. Reading back a lossless file is
what needs `imagecodecs` - the same optional package JPEG XL uses - because
Huffman decoding is inherently serial and a numpy implementation of it would be
far slower than a stack load can afford. The mode is therefore offered only when
that package is present: a format this app can write but not reopen would break
the one thing a saved result has to do, which is serve as the input to the next
stack. `available_compressions` reports what this build can actually round trip,
and the setting is validated against it.

Reading a camera DNG needs `rawpy`, which is a core requirement, so
`is_available()` tracks that the same way the loader's RAW support does.
"""

import datetime
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import ljpeg

# Extensions routed to this module instead of cv2.imwrite / cv2.imdecode.
EXTENSIONS = (".dng",)

# Written into UniqueCameraModel, and the marker `read` uses to recognise its own
# output and take the verbatim path instead of developing the file.
CAMERA_MODEL = "OpenFocus"

# Strips are sized to about this many bytes of *uncompressed* pixels. A single
# strip spanning a 24 MP 16-bit image would be a 144 MB run, which some readers
# handle poorly; a few megabytes per strip is what camera DNGs use. It also
# bounds the working set of a compressed strip's encode and decode.
_STRIP_TARGET_BYTES = 8 << 20

try:
    import rawpy
    _RAWPY_AVAILABLE = True
except ImportError:
    rawpy = None
    _RAWPY_AVAILABLE = False

try:
    import imagecodecs
    _LJPEG_AVAILABLE = bool(
        getattr(imagecodecs, "LJPEG", None) and imagecodecs.LJPEG.available
    )
except Exception:  # pylint: disable=broad-except
    imagecodecs = None
    _LJPEG_AVAILABLE = False


# ----------------------------------------------------------------------
# Compression modes
# ----------------------------------------------------------------------
COMPRESSION_NONE = "none"
COMPRESSION_LOSSLESS = "lossless"
COMPRESSION_LOSSY = "lossy"

# Every mode this module knows, in increasing order of what it costs the pixels.
VALID_COMPRESSIONS = (COMPRESSION_NONE, COMPRESSION_LOSSLESS, COMPRESSION_LOSSY)

# Uncompressed, so a build that gains or loses `imagecodecs` writes the same
# file, and so the default never trades quality or a dependency for size.
DEFAULT_COMPRESSION = COMPRESSION_NONE

# Quality for the lossy mode, on OpenCV's 1-100 JPEG scale. 92 is high enough
# that the DCT is not the thing limiting a proxy, and well short of the point
# where the file grows for no visible return.
DEFAULT_LOSSY_QUALITY = 92
MIN_LOSSY_QUALITY = 1
MAX_LOSSY_QUALITY = 100

# Whether a fast-load preview is embedded by default. On: the preview costs a
# few hundred kilobytes and one downscale, and is the difference between a
# thumbnail appearing at once and a viewer decoding the whole LinearRaw first.
DEFAULT_FAST_LOAD = True

# Preview quality, and the divisor applied to each side. Half resolution is what
# Adobe's own fast-load data uses: enough to fill a develop-module window
# without approaching the size of the raw it stands in for.
_PREVIEW_QUALITY = 85
_PREVIEW_DIVISOR = 2

# A preview smaller than this on its long side is not worth a SubIFD - the main
# image is already thumbnail-sized, so decoding it costs nothing to avoid.
_PREVIEW_MIN_EDGE = 160


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
_SUB_IFDS = 330
_SOFTWARE = 305
_DATE_TIME = 306
_YCBCR_SUB_SAMPLING = 530
_SAMPLE_FORMAT = 339
_PREVIEW_APPLICATION_NAME = 50966
_PREVIEW_APPLICATION_VERSION = 50967
_PREVIEW_COLOR_SPACE = 50970
_PREVIEW_DATE_TIME = 50971
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

# PhotometricInterpretation for a JPEG-compressed preview. The spec names this
# as the value to use for one, and pins the JPEG to baseline DCT when it is
# paired with 8/8/8 BitsPerSample - which is what OpenCV writes.
_YCBCR = 6

# Compression tag values, and the mode each one stands for. Lossy JPEG shares
# its code with LinearRaw's PhotometricInterpretation, which is a coincidence of
# the spec's numbering and not a relationship.
_COMPRESSION_UNCOMPRESSED = 1
_COMPRESSION_JPEG = 7
_COMPRESSION_LOSSY_JPEG = 34892

_COMPRESSION_CODES = {
    COMPRESSION_NONE: _COMPRESSION_UNCOMPRESSED,
    COMPRESSION_LOSSLESS: _COMPRESSION_JPEG,
    COMPRESSION_LOSSY: _COMPRESSION_LOSSY_JPEG,
}
_COMPRESSION_MODES = {code: mode for mode, code in _COMPRESSION_CODES.items()}

# PreviewColorSpace: the preview is rendered from sRGB-encoded pipeline pixels.
_PREVIEW_COLOR_SPACE_SRGB = 2

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

# DNG 1.4 is claimed so monochrome results are legal. The backward version is
# 1.1 for the uncompressed and lossless modes - the layout they use, a
# full-resolution image in IFD 0 with an optional preview SubIFD, is the one DNG
# has accepted since 1.0 - and 1.4 for the lossy mode, whose compression code
# the spec says to declare that way.
_DNG_VERSION_BYTES = bytes((1, 4, 0, 0))
_DNG_BACKWARD_VERSION_BYTES = bytes((1, 1, 0, 0))
_DNG_BACKWARD_VERSION_LOSSY_BYTES = bytes((1, 4, 0, 0))


def is_available() -> bool:
    """Whether this build can read DNG.

    Writing needs nothing beyond numpy and OpenCV, but a build that cannot
    develop a camera DNG cannot honestly offer the format, so this tracks
    `rawpy` - the same dependency the loader's RAW formats have.
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
# Compression settings
# ----------------------------------------------------------------------
# Module-level rather than arguments threaded through every caller, the same way
# utils.bitdepth holds the depth mode and utils.jxl its decode thread budget:
# the writer is reached through image_utils.write_image, which every save path in
# the app shares and none of which has an opinion about DNG's codec. Only the
# settings UI sets these.
_compression = DEFAULT_COMPRESSION
_lossy_quality = DEFAULT_LOSSY_QUALITY
_fast_load = DEFAULT_FAST_LOAD


def lossless_available() -> bool:
    """Whether the lossless mode can be both written and read by this build."""
    return _LJPEG_AVAILABLE


def lossless_unavailable_reason() -> str:
    """One line explaining why the lossless mode is off, for logs and dialogs."""
    if _LJPEG_AVAILABLE:
        return ""
    return ("Reading back lossless DNG needs the 'imagecodecs' package: "
            "pip install imagecodecs")


def available_compressions() -> tuple:
    """The compression modes this build can round trip, in the documented order.

    The lossless mode is dropped when `imagecodecs` is absent - writing it would
    work, but the result could not be reopened - the same kind of gate JPEG XL
    applies to the format as a whole.
    """
    return tuple(mode for mode in VALID_COMPRESSIONS
                 if mode != COMPRESSION_LOSSLESS or _LJPEG_AVAILABLE)


def set_compression(mode: str) -> str:
    """Set the compression mode for subsequent writes. Returns the mode applied.

    An unknown mode, or the lossless mode on a build without `imagecodecs`,
    raises rather than being quietly downgraded: a caller asking for lossless
    and getting uncompressed would only find out from the file size.
    """
    global _compression
    if mode not in VALID_COMPRESSIONS:
        raise ValueError(
            f"Unknown DNG compression mode: {mode!r}. Expected one of {VALID_COMPRESSIONS}."
        )
    if mode == COMPRESSION_LOSSLESS and not _LJPEG_AVAILABLE:
        raise RuntimeError(lossless_unavailable_reason())
    _compression = mode
    return _compression


def get_compression() -> str:
    """The compression mode subsequent writes will use."""
    return _compression


def set_lossy_quality(quality: int) -> int:
    """Set the lossy mode's JPEG quality, clamped to 1-100. Returns what stuck."""
    global _lossy_quality
    _lossy_quality = int(max(MIN_LOSSY_QUALITY, min(MAX_LOSSY_QUALITY, int(quality))))
    return _lossy_quality


def get_lossy_quality() -> int:
    """The JPEG quality the lossy mode will use."""
    return _lossy_quality


def set_fast_load(enabled: bool) -> bool:
    """Set whether writes embed a fast-load preview. Returns what stuck."""
    global _fast_load
    _fast_load = bool(enabled)
    return _fast_load


def get_fast_load() -> bool:
    """Whether writes embed a fast-load preview."""
    return _fast_load


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


def _ifd_size(entry_count: int) -> int:
    """Bytes an IFD's tag table occupies: count, entries, next-IFD pointer."""
    return 2 + 12 * entry_count + 4


def _build_ifd(entries: list, base_offset: int) -> Tuple[bytes, Dict[int, int]]:
    """Lay out one IFD, and its out-of-line values, at `base_offset`.

    `entries` are (tag, type, values) triples. Returns the assembled bytes and,
    for every tag, the file-absolute position at which its values begin -
    whether that is inline in the 4-byte entry or out in the value block. The
    caller patches through those positions the offsets it could not know when it
    built the entries: where each strip landed, and where the preview SubIFD
    starts. Both are chicken-and-egg, since recording them is what fixes the
    size of the table they are recorded in.

    TIFF requires the entries be sorted by tag, and any value longer than the
    four bytes an entry has room for to live elsewhere in the file and be
    referenced by offset.
    """
    entries = sorted(entries, key=lambda item: item[0])
    values_offset = base_offset + _ifd_size(len(entries))

    table = bytearray()
    values = bytearray()
    positions: Dict[int, int] = {}

    table += struct.pack("<H", len(entries))
    for tag, field_type, raw in entries:
        payload = _pack(field_type, raw)
        count = _count(field_type, raw)
        entry = struct.pack("<HHI", tag, field_type, count)
        if len(payload) <= 4:
            table += entry + payload.ljust(4, b"\x00")
            # The value sits inline in the entry itself; its position is
            # file-absolute, hence measured from the IFD's own base.
            positions[tag] = base_offset + len(table) - 4
        else:
            positions[tag] = values_offset + len(values)
            table += entry + struct.pack("<I", positions[tag])
            values += payload
            if len(values) % 2:
                values += b"\x00"  # keep the next value word-aligned
    table += struct.pack("<I", 0)  # no further IFDs; DNG uses SubIFD trees

    return bytes(table + values), positions


# ----------------------------------------------------------------------
# Strip codecs
# ----------------------------------------------------------------------
def _encode_strip_lossless(strip: np.ndarray, bits: int) -> bytes:
    """Lossless Huffman JPEG for one strip of file-order samples.

    The strip's own geometry is what reaches the codestream - one JPEG component
    per DNG sample - because that is what readers assume; see utils.ljpeg for
    why the alternative the spec permits does not survive contact with LibRaw.
    """
    return ljpeg.encode(strip, bits=bits)


def _encode_strip_lossy(strip: np.ndarray, quality: int) -> bytes:
    """Baseline DCT JPEG for one strip of 8-bit file-order samples.

    OpenCV encodes from BGR, so a three-sample strip - which is held here in
    file order, red first - is reversed on the way in. That way the JPEG's own
    first component is the red one, and a reader that simply decodes it gets the
    samples in the order the strip claims rather than mirrored.
    """
    params = [
        cv2.IMWRITE_JPEG_QUALITY, int(quality),
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
    ]
    if strip.ndim == 3 and strip.shape[2] == 3:
        strip = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(strip), params)
    if not ok:
        raise ValueError("OpenCV could not encode a lossy DNG strip.")
    return buffer.tobytes()


def _decode_strip_lossless(payload: bytes) -> Optional[np.ndarray]:
    """Flat sample array from a lossless-JPEG strip, or None if it will not decode.

    The internal geometry is deliberately not checked against the strip's: DNG
    lets them differ, so only the total sample count means anything and the
    caller is the one that knows what it should be.
    """
    if not _LJPEG_AVAILABLE:
        return None
    try:
        return np.asarray(imagecodecs.ljpeg_decode(payload)).ravel()
    except Exception:  # pylint: disable=broad-except
        return None


def _decode_strip_lossy(payload: bytes, samples: int) -> Optional[np.ndarray]:
    """Flat sample array from a lossy-JPEG strip, or None if it will not decode.

    The mirror of `_encode_strip_lossy`: OpenCV hands back BGR, which is
    reversed again to put the samples in the file order the strip describes.
    """
    flags = cv2.IMREAD_COLOR if samples == 3 else cv2.IMREAD_GRAYSCALE
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    if decoded is None:
        return None
    if samples == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return np.asarray(decoded).ravel()


# ----------------------------------------------------------------------
# Fast-load preview
# ----------------------------------------------------------------------
def _preview_jpeg(data: np.ndarray, bits: int) -> Optional[Tuple[bytes, int, int]]:
    """A half-resolution baseline JPEG of the image, with its dimensions.

    `data` is the full-resolution frame in file order (RGB, or a single plane).
    Returns None when the image is too small for a preview to save a reader any
    work, or when OpenCV declines to encode it.
    """
    height, width = data.shape[:2]
    target_w = max(1, width // _PREVIEW_DIVISOR)
    target_h = max(1, height // _PREVIEW_DIVISOR)
    if max(target_w, target_h) < _PREVIEW_MIN_EDGE:
        return None

    # INTER_AREA is the right filter for a pure downscale: it averages every
    # source pixel that falls in a target one, so the preview does not alias the
    # fine detail a focus-stacked frame is full of.
    small = cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if bits == 16:
        # A JPEG preview is 8-bit whatever the raw is. Scaling by 257 rather
        # than shifting keeps full scale at full scale.
        small = np.rint(small.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)

    # The preview is a rendering, not raw data, so it is always three-channel:
    # a viewer showing it never has to know the main image was monochrome.
    if small.ndim == 2:
        bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    ok, buffer = cv2.imencode(
        ".jpg", bgr,
        [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_QUALITY,
         cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420],
    )
    if not ok:
        return None
    return buffer.tobytes(), target_w, target_h


def _preview_entries(width: int, height: int, byte_count: int,
                     software: str, now: str) -> list:
    """IFD entries for the preview SubIFD, with a placeholder strip offset.

    `NewSubFileType` = 1 is what marks this as the *primary* preview, which is
    the one a reader is meant to display by default; PhotometricInterpretation
    6 with 8/8/8 samples is the combination the spec reserves for a baseline
    JPEG preview, and the `Preview*` tags say who rendered it and in which
    colour space, so no reader has to guess at either.
    """
    return [
        (_NEW_SUBFILE_TYPE, _LONG, [1]),
        (_IMAGE_WIDTH, _LONG, [width]),
        (_IMAGE_LENGTH, _LONG, [height]),
        (_BITS_PER_SAMPLE, _SHORT, [8, 8, 8]),
        (_COMPRESSION, _SHORT, [_COMPRESSION_JPEG]),
        (_PHOTOMETRIC, _SHORT, [_YCBCR]),
        (_STRIP_OFFSETS, _LONG, [0]),  # patched once the layout is fixed
        (_ORIENTATION, _SHORT, [1]),
        (_SAMPLES_PER_PIXEL, _SHORT, [3]),
        (_ROWS_PER_STRIP, _LONG, [height]),  # the whole JPEG is one strip
        (_STRIP_BYTE_COUNTS, _LONG, [byte_count]),
        (_PLANAR_CONFIG, _SHORT, [1]),
        (_YCBCR_SUB_SAMPLING, _SHORT, [2, 2]),  # matches the 4:2:0 encode above
        (_PREVIEW_APPLICATION_NAME, _ASCII, CAMERA_MODEL),
        (_PREVIEW_APPLICATION_VERSION, _ASCII, software),
        (_PREVIEW_COLOR_SPACE, _LONG, [_PREVIEW_COLOR_SPACE_SRGB]),
        (_PREVIEW_DATE_TIME, _ASCII, now),
    ]


def _rows_per_strip(mode: str, width: int, height: int, samples: int, bits: int) -> int:
    """How many rows one strip holds, which the codec decides.

    An uncompressed image is cut into several strips, because one strip spanning
    a 24 MP 16-bit frame is a 144 MB run that some readers handle poorly.

    A compressed image is written as a **single** strip, and that is not a
    preference. In a stripped - as opposed to tiled - DNG, LibRaw reads the
    second and later compressed strips by continuing on from wherever the codec
    left the file pointer rather than by consulting `StripOffsets`, so anything
    the decoder does not consume byte-for-byte desynchronises it: a multi-strip
    lossless file decodes its first strip and renders the rest black. One strip
    per image removes the question, at the cost of holding the frame and its
    encoded form at once - which the uncompressed path does anyway, since the
    whole file is assembled in memory before it is written.
    """
    if mode != COMPRESSION_NONE:
        return height
    row_bytes = width * samples * (bits // 8)
    return max(1, min(height, _STRIP_TARGET_BYTES // max(1, row_bytes)))


def _resolve_options(compression: Optional[str], lossy_quality: Optional[int],
                     fast_load: Optional[bool]) -> Tuple[str, int, bool]:
    """Fill unset encode options from the module settings, and validate them."""
    mode = _compression if compression is None else str(compression)
    if mode not in VALID_COMPRESSIONS:
        raise ValueError(
            f"Unknown DNG compression mode: {mode!r}. Expected one of {VALID_COMPRESSIONS}."
        )
    if mode == COMPRESSION_LOSSLESS and not _LJPEG_AVAILABLE:
        raise RuntimeError(lossless_unavailable_reason())

    quality = _lossy_quality if lossy_quality is None else int(lossy_quality)
    quality = max(MIN_LOSSY_QUALITY, min(MAX_LOSSY_QUALITY, quality))
    return mode, quality, _fast_load if fast_load is None else bool(fast_load)


def _as_file_order(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """A BGR or greyscale image as contiguous file-order samples, and their count.

    DNG, like TIFF, stores a pixel's channels in order, so the pipeline's BGR
    becomes RGB here. An alpha channel is dropped rather than written as an
    extra sample no converter would look at - DNG raw data has no place for one.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    if image.ndim == 2:
        return np.ascontiguousarray(image), 1
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), 3
    raise ValueError(f"Unsupported image shape for DNG: {image.shape}")


def encode(image: np.ndarray,
           software: str = CAMERA_MODEL,
           compression: Optional[str] = None,
           lossy_quality: Optional[int] = None,
           fast_load: Optional[bool] = None) -> bytes:
    """Encode a BGR (or greyscale) image as a linear DNG.

    The samples are written at whatever depth the array carries, 8 or 16 bits,
    and it is the tags that describe what they mean; see the module docstring for
    which ones and why. `compression`, `lossy_quality` and `fast_load` default
    to the module settings, so a caller that has no opinion about the codec does
    not have to form one.

    The lossy mode is 8-bit only in DNG, so a 16-bit image handed to it is
    narrowed here, with a line on stdout - the same way image_utils announces a
    container that cannot hold the depth it was given.
    """
    if image is None:
        raise ValueError("No image to encode.")
    mode, quality, want_preview = _resolve_options(compression, lossy_quality, fast_load)

    data, samples = _as_file_order(image)

    if data.dtype == np.uint8:
        bits = 8
    elif data.dtype == np.uint16:
        bits = 16
    else:
        raise ValueError(f"Unsupported image dtype for DNG: {data.dtype}")

    if mode == COMPRESSION_LOSSY and samples == 1:
        # LibRaw's lossy-DNG path reads three components per pixel unconditionally,
        # so a single-sample lossy file is one no mainstream raw reader will open.
        # Falling back keeps the promise that every DNG written here is readable,
        # which matters more than honouring a codec choice for a mono result.
        fallback = COMPRESSION_LOSSLESS if _LJPEG_AVAILABLE else COMPRESSION_NONE
        print(f"[DNG] Lossy compression cannot be read back for a monochrome image; "
              f"writing {fallback} instead.", flush=True)
        mode = fallback

    if mode == COMPRESSION_LOSSY and bits == 16:
        print("[DNG] Lossy compression is 8-bit only in DNG; saving 8-bit. "
              "Use the lossless or uncompressed mode to keep 16 bits.", flush=True)
        data = np.rint(data.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
        bits = 8

    height, width = data.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    rows_per_strip = _rows_per_strip(mode, width, height, samples, bits)

    strips: List[bytes] = []
    for first in range(0, height, rows_per_strip):
        strip = data[first:first + rows_per_strip]
        if mode == COMPRESSION_LOSSLESS:
            strips.append(_encode_strip_lossless(strip, bits))
        elif mode == COMPRESSION_LOSSY:
            strips.append(_encode_strip_lossy(strip, quality))
        else:
            strips.append(np.ascontiguousarray(strip).tobytes())

    preview = _preview_jpeg(data, bits) if want_preview else None

    now = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    backward = (_DNG_BACKWARD_VERSION_LOSSY_BYTES if mode == COMPRESSION_LOSSY
                else _DNG_BACKWARD_VERSION_BYTES)
    entries = [
        (_NEW_SUBFILE_TYPE, _LONG, [0]),  # this is the full-resolution image
        (_IMAGE_WIDTH, _LONG, [width]),
        (_IMAGE_LENGTH, _LONG, [height]),
        (_BITS_PER_SAMPLE, _SHORT, [bits] * samples),
        (_COMPRESSION, _SHORT, [_COMPRESSION_CODES[mode]]),
        (_PHOTOMETRIC, _SHORT, [_LINEAR_RAW]),
        (_MAKE, _ASCII, CAMERA_MODEL),
        (_MODEL, _ASCII, CAMERA_MODEL),
        (_STRIP_OFFSETS, _LONG, [0] * len(strips)),  # patched after layout
        (_ORIENTATION, _SHORT, [1]),
        (_SAMPLES_PER_PIXEL, _SHORT, [samples]),
        (_ROWS_PER_STRIP, _LONG, [rows_per_strip]),
        (_STRIP_BYTE_COUNTS, _LONG, [len(blob) for blob in strips]),
        (_PLANAR_CONFIG, _SHORT, [1]),  # interleaved
        (_SOFTWARE, _ASCII, software),
        (_DATE_TIME, _ASCII, now),
        (_SAMPLE_FORMAT, _SHORT, [1] * samples),  # unsigned integer
        (_DNG_VERSION, _BYTE, _DNG_VERSION_BYTES),
        (_DNG_BACKWARD_VERSION, _BYTE, backward),
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
    if preview is not None:
        entries.append((_SUB_IFDS, _LONG, [0]))  # patched after layout

    # IFD 0 sits straight after the 8-byte TIFF header; the preview's IFD
    # follows it, and the pixel data follows both, so that every offset either
    # IFD records points forward into a block whose size is already known.
    main_ifd, main_positions = _build_ifd(entries, 8)
    blob = bytearray(struct.pack("<2sHI", b"II", 42, 8))
    blob += main_ifd

    preview_positions: Dict[int, int] = {}
    if preview is not None:
        payload, preview_width, preview_height = preview
        preview_ifd, preview_positions = _build_ifd(
            _preview_entries(preview_width, preview_height, len(payload), software, now),
            len(blob),
        )
        struct.pack_into("<I", blob, main_positions[_SUB_IFDS], len(blob))
        blob += preview_ifd

    for index, strip in enumerate(strips):
        struct.pack_into("<I", blob, main_positions[_STRIP_OFFSETS] + 4 * index, len(blob))
        blob += strip

    if preview is not None:
        struct.pack_into("<I", blob, preview_positions[_STRIP_OFFSETS], len(blob))
        blob += preview[0]

    return bytes(blob)


def write(file_path: str,
          image: np.ndarray,
          software: str = CAMERA_MODEL,
          compression: Optional[str] = None,
          lossy_quality: Optional[int] = None,
          fast_load: Optional[bool] = None) -> bool:
    """Encode and write a DNG file. Returns False instead of raising.

    Mirrors what cv2.imwrite gives the callers in utils.image_utils: a boolean,
    with the reason printed, so a failed DNG save is reported through the same
    "could not write" path as any other format.
    """
    try:
        payload = encode(image, software=software, compression=compression,
                         lossy_quality=lossy_quality, fast_load=fast_load)
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


def _own_compression(tags: Dict[int, object]) -> Optional[str]:
    """The compression mode of a DNG this module wrote, or None if it did not.

    Both other conditions matter as much as the codec. A camera DNG names its
    camera in UniqueCameraModel, and puts a small preview - not LinearRaw - in
    IFD 0, so it cannot pass by accident.
    """
    if tags.get(_UNIQUE_CAMERA_MODEL) != CAMERA_MODEL:
        return None
    if tags.get(_PHOTOMETRIC) != [_LINEAR_RAW]:
        return None
    code = tags.get(_COMPRESSION)
    if not isinstance(code, list) or len(code) != 1:
        return None
    return _COMPRESSION_MODES.get(int(code[0]))


def _is_own_output(tags: Dict[int, object]) -> bool:
    """Whether these IFD 0 tags describe a DNG this module wrote."""
    return _own_compression(tags) is not None


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


def _strip_payloads(path: str, offsets: Sequence[int],
                    counts: Sequence[int]) -> Optional[List[bytes]]:
    """Read each strip's bytes, or None if any of them is short or unreadable."""
    try:
        with open(path, "rb") as handle:
            chunks = []
            for offset, count in zip(offsets, counts):
                handle.seek(offset)
                chunk = handle.read(count)
                if len(chunk) < count:
                    return None
                chunks.append(chunk)
            return chunks
    except OSError:
        return None


def read_linear(path: str) -> Optional[np.ndarray]:
    """Read an OpenFocus linear DNG straight from its strips, as BGR.

    No development happens: the samples come back at the depth they were stored
    at, in the order they were stored in, so saving a frame and loading it again
    is an identity - exactly so for the uncompressed and lossless modes, and to
    within the DCT for the lossy one. Returns None for anything that is not this
    module's own output, which is how `read` decides to hand the file to LibRaw
    instead, and for a file whose own mode this build cannot decode.
    """
    tags = _read_ifd0(path)
    if tags is None:
        return None
    mode = _own_compression(tags)
    if mode is None:
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
    expected = width * height * samples

    chunks = _strip_payloads(path, offsets, counts)
    if chunks is None:
        return None

    if mode == COMPRESSION_NONE:
        if sum(counts) != expected * (bits[0] // 8):
            return None
        # Samples are little-endian on disk regardless of the host, so the dtype
        # is byte-ordered explicitly and then normalised to native.
        flat = np.frombuffer(b"".join(chunks), dtype=np.dtype(dtype).newbyteorder("<"))
        pixels = flat.reshape(height, width, samples).astype(dtype, copy=False)
    else:
        decoded = []
        for chunk in chunks:
            if mode == COMPRESSION_LOSSLESS:
                part = _decode_strip_lossless(chunk)
            else:
                part = _decode_strip_lossy(chunk, samples)
            if part is None:
                return None
            decoded.append(part)
        flat = np.concatenate(decoded) if decoded else np.empty(0, dtype=dtype)
        if flat.size != expected:
            return None
        pixels = flat.astype(dtype, copy=False).reshape(height, width, samples)

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
