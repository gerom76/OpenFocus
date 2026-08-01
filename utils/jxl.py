"""JPEG XL encoding and decoding.

OpenCV's wheels are not built with libjxl, so `.jxl` is the one container the
usual `cv2.imwrite` / `cv2.imdecode` paths can neither write nor read. It is
handled here instead, through `imagecodecs`, which wraps libjxl and speaks numpy
arrays directly - the same kind of array the rest of the pipeline already
carries. That matters for depth in both directions: JPEG XL stores 16 bits per
channel, so encoding straight from the uint16 buffer keeps the full depth PNG
and TIFF also keep, and a 16-bit `.jxl` stack loads at 16 bits rather than
arriving pre-narrowed.

The backend is optional. Without it `is_available()` is False, the format is
left out of the save dialogs, the batch format list and the loader's supported
extensions, and nothing else in the app changes - so a build without
`imagecodecs` behaves exactly as before.

Files are always written as a **container** (the ISOBMFF-style box layout) rather
than a bare codestream, because that is what lets utils.metadata splice the EXIF
and XMP boxes in afterwards. The 32 bytes it costs are the header boxes.

Threading
---------
libjxl does the work in its own pool, and `imagecodecs` sizes that pool at **one
thread** unless told otherwise - which is what every call here used to get. On a
12 MP frame that single thread costs 26 s to encode losslessly at 16 bits and
1.7 s to decode, against 1.5 s and 0.10 s with the cores this machine has. So
the pool size is now always passed:

- **Encoding** takes every core. Nothing in the app encodes more than one image
  at a time - exports, batch results and aligned-stack saves all run in a plain
  loop - so there is no other encode to share the machine with.
- **Decoding** answers to `set_decode_threads`, because the loader decodes a
  stack through a thread pool of its own. It divides the cores among its
  workers, so the two levels of parallelism multiply out to the core count
  instead of oversubscribing it; a stack with fewer frames than cores still
  gets the whole machine, one frame at a time.

There is no GPU path. Neither nvJPEG nor nvImageCodec implements JPEG XL and
libjxl has no CUDA backend, so unlike JPEG (see core.gpu_decode) a `.jxl` frame
cannot be handed to the GPU to decode; the cores above are the whole budget.
The frame reaches the GPU the same way any other decoded frame does, once
fusion uploads it.

Memory
------
A libjxl decode peaks at about three times the size of the frame it produces -
the codec works in float internally - which is why `probe` exists: it reads the
dimensions and the bit depth out of the codestream header, without decoding,
so the loader can size its thread pool and its up-front memory estimate for a
`.jxl` stack the way it already does for RAW. The RGB->BGR conversion afterwards
is done in place, so a decode does not also hold a second copy of the frame.
"""

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Optional, Tuple

import cv2
import numpy as np

# Extensions routed to this module instead of cv2.imwrite.
EXTENSIONS = (".jxl",)

# Encoder effort, 1 (fastest) to 9 (slowest). 7 is libjxl's own default and the
# point where more effort stops buying much on photographic input: on a 12 MP
# frame, 9 is 3.7x the time of 7 for under 1% off the file.
DEFAULT_EFFORT = 7

# Pool size meaning "one worker per core", which is what libjxl picks for
# itself when it is allowed to. `imagecodecs` defaults to 1 instead, so this is
# passed explicitly everywhere rather than left off.
AUTO_THREADS = 0

# Bytes of the codestream `probe` reads once it has found it. The SizeHeader and
# ImageMetadata sit within the first few dozen bytes; the rest is slack for the
# optional fields the parser steps over. The boxes ahead of the codestream are
# seeked over rather than read, so their size does not enter into this - a file
# carrying a camera's MakerNote can put a few hundred KB in front of the
# codestream and still be probed with two short reads.
_PROBE_BYTES = 1 << 12

try:
    import imagecodecs
    _AVAILABLE = bool(getattr(imagecodecs, "JPEGXL", None) and imagecodecs.JPEGXL.available)
except Exception:  # pylint: disable=broad-except
    imagecodecs = None
    _AVAILABLE = False


def is_available() -> bool:
    """Whether this build can read and write JPEG XL."""
    return _AVAILABLE


def is_jxl(extension: str) -> bool:
    """Whether a file extension names the JPEG XL container."""
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext in EXTENSIONS


def unavailable_reason() -> str:
    """One line explaining why JPEG XL is off, for logs and message boxes."""
    if _AVAILABLE:
        return ""
    return ("JPEG XL support needs the 'imagecodecs' package: "
            "pip install imagecodecs")


def extensions() -> tuple:
    """The extensions this build can handle - empty when the backend is absent.

    Lets the loader and the folder-scanning fusion methods fold JPEG XL into
    their supported sets without each of them repeating the availability check.
    """
    return EXTENSIONS if _AVAILABLE else ()


# --------------------------------------------------------------------------
# Decoder thread budget
# --------------------------------------------------------------------------

# How many threads libjxl may use per decode. A module-level setting rather
# than an argument because the decode is reached through
# image_utils.read_image_any_depth, which every folder-input path in the app
# shares and none of which has an opinion about threading; only the loader,
# which owns the pool the decodes run in, sets it.
_decode_threads = AUTO_THREADS


def set_decode_threads(count: int) -> None:
    """Set how many threads a single decode may use; 0 means one per core."""
    global _decode_threads
    _decode_threads = max(0, int(count))


def get_decode_threads() -> int:
    """The current per-decode thread budget."""
    return _decode_threads


def threads_for_workers(workers: int) -> int:
    """Split the cores between `workers` concurrent decodes, at least 1 each.

    Rounded up rather than down. Rounding down leaves cores idle whenever the
    frame count does not divide into them, and the miss is worst exactly where
    it hurts: 24 frames on 32 cores floors to one thread each and takes twice
    as long as the two the same split rounds up to. A libjxl pool is not busy
    every moment of a decode, so the slight oversubscription that rounding up
    can produce costs less than the idle cores rounding down leaves behind.

    Returns AUTO_THREADS when there is only one decode in flight, so a
    single-frame load gets libjxl's own pool sizing rather than a count frozen
    at load time.
    """
    cores = os.cpu_count() or 1
    workers = max(1, int(workers))
    if workers <= 1:
        return AUTO_THREADS
    return max(1, -(-cores // workers))


@contextmanager
def decode_thread_budget(count: int):
    """Apply a per-decode thread budget for the duration of a block."""
    previous = _decode_threads
    set_decode_threads(count)
    try:
        yield
    finally:
        set_decode_threads(previous)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def encode(image: np.ndarray,
           lossless: bool = True,
           distance: Optional[float] = None,
           effort: int = DEFAULT_EFFORT,
           threads: Optional[int] = None) -> bytes:
    """Encode a BGR (or greyscale) image as a JPEG XL container.

    `lossless` is the default because every other format this app writes is
    written at its maximum quality setting - JPEG at 100, PNG uncompressed,
    TIFF uncompressed - and a fused result is a master, not a delivery file.
    Passing a `distance` (libjxl's butteraugli target, 0 = lossless, 1 ~ visually
    lossless) selects the lossy path instead.

    `threads` defaults to every core: see the module docstring for why encoding
    never has to share the machine with another encode.
    """
    if not _AVAILABLE:
        raise RuntimeError(unavailable_reason())
    if image is None:
        raise ValueError("No image to encode.")

    # libjxl works in RGB; the pipeline carries BGR. Greyscale is passed through
    # as a single plane, which JPEG XL stores natively. The conversion cannot be
    # done in place here the way it can on the way back out - the array belongs
    # to the caller, which still wants it in BGR afterwards.
    if image.ndim == 3:
        if image.shape[2] == 3:
            data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            data = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 1:
            data = image[:, :, 0]
        else:
            raise ValueError(f"Unsupported image shape for JPEG XL: {image.shape}")
    elif image.ndim == 2:
        data = image
    else:
        raise ValueError(f"Unsupported image shape for JPEG XL: {image.shape}")

    data = np.ascontiguousarray(data)
    numthreads = AUTO_THREADS if threads is None else max(0, int(threads))

    if distance is None or float(distance) <= 0.0:
        return imagecodecs.jpegxl_encode(
            data, lossless=True, effort=effort, usecontainer=True,
            numthreads=numthreads,
        )
    return imagecodecs.jpegxl_encode(
        data, lossless=False, distance=float(distance), effort=effort,
        usecontainer=True, numthreads=numthreads,
    )


def write(file_path: str,
          image: np.ndarray,
          lossless: bool = True,
          distance: Optional[float] = None,
          effort: int = DEFAULT_EFFORT,
          threads: Optional[int] = None) -> bool:
    """Encode and write a JPEG XL file. Returns False instead of raising.

    Mirrors what cv2.imwrite gives the callers in utils.image_utils: a boolean,
    with the reason printed, so a failed JPEG XL save is reported through the
    same "could not write" path as any other format.
    """
    try:
        payload = encode(image, lossless=lossless, distance=distance,
                         effort=effort, threads=threads)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[JPEG XL] Could not encode {os.path.basename(file_path)}: {exc}", flush=True)
        return False

    try:
        with open(file_path, "wb") as handle:
            handle.write(payload)
    except OSError as exc:
        print(f"[JPEG XL] Could not write {os.path.basename(file_path)}: {exc}", flush=True)
        return False
    return True


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

def decode(payload: bytes, threads: Optional[int] = None) -> Optional[np.ndarray]:
    """Decode a JPEG XL byte string to BGR, or None if it cannot be read.

    The array comes back at the depth the file was written at - uint8, uint16 or
    float32 - and is handed on unchanged, so the caller's depth mode is what
    decides the frame's storage dtype, exactly as for a 16-bit PNG or TIFF.

    `threads` defaults to the budget the loader set for the stack it is
    decoding; see `set_decode_threads`.
    """
    if not _AVAILABLE:
        return None
    try:
        data = imagecodecs.jpegxl_decode(
            payload,
            numthreads=_decode_threads if threads is None else max(0, int(threads)),
        )
    except Exception:  # pylint: disable=broad-except
        return None
    # The decoded array belongs to us, so the channel swap writes back into it
    # rather than allocating a second full frame beside libjxl's own buffers.
    if data.ndim == 3 and data.shape[2] in (3, 4):
        code = cv2.COLOR_RGB2BGR if data.shape[2] == 3 else cv2.COLOR_RGBA2BGRA
        if data.flags.c_contiguous and data.flags.writeable:
            return cv2.cvtColor(data, code, dst=data)
        return cv2.cvtColor(data, code)
    return data


def read(file_path: str, threads: Optional[int] = None) -> Optional[np.ndarray]:
    """Read a JPEG XL file from disk as BGR, or None if it cannot be read.

    The counterpart of `write`, and the read side of what cv2.imdecode does for
    every other input format. The file is pulled into memory first rather than
    handed to libjxl by name, so paths with non-ASCII characters load the same
    way they do everywhere else in the loader.

    Returns None - rather than raising - for a missing, truncated or non-JPEG XL
    file, because the callers report a failed frame themselves and carry on with
    the rest of the stack.
    """
    if not _AVAILABLE:
        return None
    try:
        with open(file_path, "rb") as handle:
            payload = handle.read()
    except OSError:
        return None
    return decode(payload, threads=threads)


# --------------------------------------------------------------------------
# Header probing
# --------------------------------------------------------------------------

_CODESTREAM_SIGNATURE = b"\xff\x0a"
_CONTAINER_SIGNATURE = b"\x00\x00\x00\x0cJXL \r\n\x87\n"

# Codestream boxes: `jxlc` holds the whole codestream, `jxlp` the first slice of
# a split one, behind a 4-byte sequence index.
_BOX_CODESTREAM = b"jxlc"
_BOX_PARTIAL = b"jxlp"

# SizeHeader's aspect-ratio table, indexed by the 3-bit ratio field; 0 means the
# width is stored explicitly instead.
_RATIOS = ((0, 0), (1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1))


class _BitReader:
    """The codestream's bit packing: least significant bit of each byte first."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def bits(self, count: int) -> int:
        value = 0
        for index in range(count):
            byte, bit = divmod(self._pos, 8)
            if byte >= len(self._data):
                raise EOFError("JPEG XL header is truncated")
            value |= ((self._data[byte] >> bit) & 1) << index
            self._pos += 1
        return value

    def flag(self) -> bool:
        return bool(self.bits(1))

    def u32(self, *distributions) -> int:
        """A U32 field: two selector bits choose one of four (width, offset) pairs."""
        width, offset = distributions[self.bits(2)]
        return (self.bits(width) if width else 0) + offset


def _dimension(reader: _BitReader, small: bool) -> int:
    """One side length from a SizeHeader; `small` is its shared div-8 flag."""
    if small:
        return (reader.bits(5) + 1) * 8
    return reader.u32((9, 1), (13, 1), (18, 1), (30, 1))


def _skip_optional_size(reader: _BitReader) -> None:
    """Skip an optional nested SizeHeader (the intrinsic size, or the preview)."""
    if reader.flag():
        small = reader.flag()
        _dimension(reader, small)
        if reader.bits(3) == 0:
            _dimension(reader, small)


def _parse_codestream(codestream: bytes) -> Optional[Tuple[int, int, int]]:
    """(width, height, bits) from a codestream's SizeHeader and ImageMetadata.

    Only the fields ahead of the bit depth are decoded - the colour encoding and
    the extra channels behind it say nothing the loader needs, and every field
    parsed is a field that can go wrong on a file libjxl would still open.
    """
    reader = _BitReader(codestream)
    if reader.bits(8) != 0xFF or reader.bits(8) != 0x0A:
        return None

    # SizeHeader: the height first, then either an aspect ratio or the width.
    small = reader.flag()
    height = _dimension(reader, small)
    ratio = reader.bits(3)
    if ratio == 0:
        width = _dimension(reader, small)
    else:
        numerator, denominator = _RATIOS[ratio]
        width = height * numerator // denominator

    # ImageMetadata. Its all-default case is 8-bit sRGB, which is most files.
    bits = 8
    if not reader.flag():
        if reader.flag():                # extra fields
            reader.bits(3)               # orientation
            _skip_optional_size(reader)  # intrinsic size
            _skip_optional_size(reader)  # preview
            if reader.flag():            # animation
                reader.u32((0, 100), (0, 1000), (10, 1), (30, 1))  # ticks numerator
                reader.u32((0, 1), (0, 1001), (8, 1), (10, 1))     # ticks denominator
                reader.u32((0, 0), (3, 0), (16, 0), (32, 0))       # loop count
                reader.flag()                                      # have timecodes
        if reader.flag():                # float samples
            bits = reader.u32((0, 32), (0, 16), (0, 24), (6, 1))
        else:
            bits = reader.u32((0, 8), (0, 10), (0, 12), (6, 1))

    if width <= 0 or height <= 0 or not 1 <= bits <= 64:
        return None
    return width, height, bits


def _codestream_head(handle) -> Optional[bytes]:
    """The front of the codestream, found without reading what sits before it.

    A bare codestream is read from the start of the file. A container is walked
    box by box, seeking over each payload rather than reading it, because the
    boxes ahead of the codestream can be large: a result saved from a camera
    source carries that camera's EXIF, and a MakerNote alone is routinely a few
    hundred KB. Reading a fixed window off the front of the file would miss the
    codestream entirely on exactly those files - which is what used to happen,
    and left every DNG-sourced save unprobeable.

    Returns None for a file that is not JPEG XL, or whose boxes run out before a
    codestream appears.
    """
    head = handle.read(len(_CONTAINER_SIGNATURE))
    if head.startswith(_CODESTREAM_SIGNATURE):
        return head + handle.read(_PROBE_BYTES)
    if not head.startswith(_CONTAINER_SIGNATURE):
        return None

    while True:
        header = handle.read(8)
        if len(header) < 8:
            return None
        size = int.from_bytes(header[0:4], "big")
        box_type = header[4:8]

        if size == 1:
            # A 64-bit size, in the eight bytes behind the type. It counts the
            # whole box, header included, so 8 comes off to match the 8-byte
            # header the branches below assume.
            extended = handle.read(8)
            if len(extended) < 8:
                return None
            size = int.from_bytes(extended, "big")
            if size < 16:
                return None
            size -= 8

        if box_type == _BOX_CODESTREAM:
            return handle.read(_PROBE_BYTES)
        if box_type == _BOX_PARTIAL:
            handle.seek(4, os.SEEK_CUR)  # over the 4-byte sequence index
            return handle.read(_PROBE_BYTES)
        if size < 8:
            # 0 means the box runs to the end of the file, and this one is not a
            # codestream; anything else below 8 is malformed.
            return None
        handle.seek(size - 8, os.SEEK_CUR)


# Probe results, keyed by (path, size, mtime) so an edited file is re-read. The
# loader probes from its worker threads, and folder scans revisit the same
# files across dialogs, so the cache is both shared and locked.
_PROBE_CACHE_LIMIT = 512
_probe_cache: "OrderedDict[tuple, Optional[Tuple[int, int, int]]]" = OrderedDict()
_probe_lock = threading.Lock()


def clear_probe_cache() -> None:
    """Forget every cached probe result."""
    with _probe_lock:
        _probe_cache.clear()


def probe(path: str) -> Optional[Tuple[int, int, int]]:
    """(width, height, bits) of a JPEG XL file, read from its header.

    Mirrors dng.probe, and answers the same question for the same caller: it
    lets the loader size a `.jxl` stack - both its memory estimate and how many
    decodes it dares run at once - without decoding a frame to find out. Only
    the first few bytes of the file are touched.

    Returns None for a file that is not JPEG XL, or whose header is truncated
    or uses a layout this parser does not follow; the loader then carries on
    without the estimate, exactly as it does for any other unprobeable format.
    Unlike the rest of this module it does not need the codec, so a build
    without `imagecodecs` can still size a stack it cannot decode.
    """
    try:
        stat = os.stat(path)
        key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return None

    with _probe_lock:
        if key in _probe_cache:
            _probe_cache.move_to_end(key)
            return _probe_cache[key]

    try:
        with open(path, "rb") as handle:
            codestream = _codestream_head(handle)
    except OSError:
        return None

    try:
        result = _parse_codestream(codestream) if codestream else None
    except (EOFError, IndexError, ValueError):
        result = None

    with _probe_lock:
        _probe_cache[key] = result
        _probe_cache.move_to_end(key)
        while len(_probe_cache) > _PROBE_CACHE_LIMIT:
            _probe_cache.popitem(last=False)
    return result
