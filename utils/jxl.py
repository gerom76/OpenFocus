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
"""

import os
from typing import Optional

import cv2
import numpy as np

# Extensions routed to this module instead of cv2.imwrite.
EXTENSIONS = (".jxl",)

# Encoder effort, 1 (fastest) to 9 (slowest). 7 is libjxl's own default and the
# point where more effort stops buying much on photographic input.
DEFAULT_EFFORT = 7

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


def encode(image: np.ndarray,
           lossless: bool = True,
           distance: Optional[float] = None,
           effort: int = DEFAULT_EFFORT) -> bytes:
    """Encode a BGR (or greyscale) image as a JPEG XL container.

    `lossless` is the default because every other format this app writes is
    written at its maximum quality setting - JPEG at 100, PNG uncompressed,
    TIFF uncompressed - and a fused result is a master, not a delivery file.
    Passing a `distance` (libjxl's butteraugli target, 0 = lossless, 1 ~ visually
    lossless) selects the lossy path instead.
    """
    if not _AVAILABLE:
        raise RuntimeError(unavailable_reason())
    if image is None:
        raise ValueError("No image to encode.")

    # libjxl works in RGB; the pipeline carries BGR. Greyscale is passed through
    # as a single plane, which JPEG XL stores natively.
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

    if distance is None or float(distance) <= 0.0:
        return imagecodecs.jpegxl_encode(
            data, lossless=True, effort=effort, usecontainer=True
        )
    return imagecodecs.jpegxl_encode(
        data, lossless=False, distance=float(distance), effort=effort,
        usecontainer=True,
    )


def write(file_path: str,
          image: np.ndarray,
          lossless: bool = True,
          distance: Optional[float] = None,
          effort: int = DEFAULT_EFFORT) -> bool:
    """Encode and write a JPEG XL file. Returns False instead of raising.

    Mirrors what cv2.imwrite gives the callers in utils.image_utils: a boolean,
    with the reason printed, so a failed JPEG XL save is reported through the
    same "could not write" path as any other format.
    """
    try:
        payload = encode(image, lossless=lossless, distance=distance, effort=effort)
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


def decode(payload: bytes) -> Optional[np.ndarray]:
    """Decode a JPEG XL byte string to BGR, or None if it cannot be read.

    The array comes back at the depth the file was written at - uint8, uint16 or
    float32 - and is handed on unchanged, so the caller's depth mode is what
    decides the frame's storage dtype, exactly as for a 16-bit PNG or TIFF.
    """
    if not _AVAILABLE:
        return None
    try:
        data = imagecodecs.jpegxl_decode(payload)
    except Exception:  # pylint: disable=broad-except
        return None
    if data.ndim == 3 and data.shape[2] == 3:
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)
    return data


def read(file_path: str) -> Optional[np.ndarray]:
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
    return decode(payload)
