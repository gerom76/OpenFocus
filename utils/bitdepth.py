"""Bit-depth policy and conversion for the processing pipeline.

The pipeline computes in float32 [0, 1] and stores frames as either uint8 or
uint16. Every stage is written to be **dtype-preserving**: a stage handed uint16
frames returns a uint16 result. That keeps the depth mode a pure *loading*
policy - it decides which dtype frames enter the pipeline as, and no stage
downstream has to know which mode is active.

The three modes:

- ``MODE_AUTO``   load at the file's native depth, so >8-bit sources engage the
                  16-bit path automatically. This is the default.
- ``MODE_8``      force everything to 8-bit, halving memory and matching the
                  behaviour of releases before 1.6.0.
- ``MODE_16``     force everything to 16-bit, including 8-bit sources. Useful
                  when a stack mixes depths, or to keep blending headroom on
                  8-bit input.

Scaling between depths is multiplicative, not a bit shift: 8-bit 255 maps to
16-bit 65535 rather than 65280, so a round trip through either direction is
lossless at the endpoints and white stays white.
"""

from typing import List, Optional, Sequence

import numpy as np

MODE_AUTO = "auto"
MODE_8 = "8"
MODE_16 = "16"
VALID_MODES = (MODE_AUTO, MODE_8, MODE_16)

UINT8 = np.dtype(np.uint8)
UINT16 = np.dtype(np.uint16)

_MAX_FOR_DTYPE = {UINT8: 255.0, UINT16: 65535.0}
_DTYPE_FOR_BITS = {8: UINT8, 16: UINT16}

# Module-level because the depth policy is a single global property of a run,
# the same way tile mode is handled in core.multi_focus_fusion. Workers read it
# through get_mode() rather than being passed it, so a mode switch applies to
# the next render without re-threading it through every call site.
_mode = MODE_AUTO


def set_mode(mode: str) -> str:
    """Set the active depth mode. Returns the mode actually applied."""
    global _mode
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown bit-depth mode: {mode!r}. Expected one of {VALID_MODES}.")
    _mode = mode
    return _mode


def get_mode() -> str:
    """Return the active depth mode."""
    return _mode


def max_value(dtype) -> float:
    """Full-scale value for an integer image dtype.

    Float arrays are taken to be normalised already, so their full scale is 1.0.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return 1.0
    try:
        return _MAX_FOR_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported image dtype: {dtype}. Expected uint8, uint16 or float.")


def bits_for(dtype) -> int:
    """Bits per channel for an image dtype (float counts as 32)."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return 32
    return 16 if dtype == UINT16 else 8


def dtype_for_bits(bits: int) -> np.dtype:
    """Image dtype for a bit count. Anything above 8 rounds up to 16."""
    return _DTYPE_FOR_BITS[16] if int(bits) > 8 else _DTYPE_FOR_BITS[8]


def is_high_depth(value) -> bool:
    """True when an image, dtype or stack carries more than 8 bits per channel."""
    if isinstance(value, np.ndarray):
        return np.dtype(value.dtype) == UINT16
    if isinstance(value, (list, tuple)):
        return any(is_high_depth(item) for item in value)
    return np.dtype(value) == UINT16


def to_float01(img: np.ndarray) -> np.ndarray:
    """Normalise an image to float32 in [0, 1] using its own full scale.

    Float inputs are passed through as float32 without rescaling: they are
    already normalised by convention.
    """
    if img.dtype.kind == "f":
        return np.asarray(img, dtype=np.float32)
    return img.astype(np.float32) / max_value(img.dtype)


def from_float01(arr: np.ndarray, dtype) -> np.ndarray:
    """Convert normalised float back to an integer image of `dtype`.

    Rounds rather than truncates. Truncation biases every value down by half a
    level, which showed up as systematic darkening (see
    docs/ALGORITHM_IMPROVEMENTS.md item 5).
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return np.asarray(arr, dtype=dtype)
    scale = max_value(dtype)
    return np.rint(np.clip(arr, 0.0, 1.0) * scale).astype(dtype)


def convert(img: np.ndarray, target_dtype) -> np.ndarray:
    """Rescale an image between 8- and 16-bit, preserving full scale.

    Returns the input untouched when it is already the requested dtype.
    """
    target_dtype = np.dtype(target_dtype)
    if img.dtype == target_dtype:
        return img
    source_max = max_value(img.dtype)
    target_max = max_value(target_dtype)
    scaled = img.astype(np.float32) * (target_max / source_max)
    if target_dtype.kind == "f":
        return scaled
    return np.rint(np.clip(scaled, 0.0, target_max)).astype(target_dtype)


def resolve_load_dtype(native_dtype) -> np.dtype:
    """Dtype a freshly decoded frame should be stored as, under the active mode."""
    native_dtype = np.dtype(native_dtype)
    mode = get_mode()
    if mode == MODE_8:
        return UINT8
    if mode == MODE_16:
        return UINT16
    # Auto: keep whatever the file gave us, but only 8 and 16 are storage
    # formats; anything else (float TIFF, for instance) lands on 16-bit.
    return UINT8 if native_dtype == UINT8 else UINT16


def apply_load_mode(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Bring a decoded frame to the dtype the active mode calls for."""
    if img is None:
        return None
    return convert(img, resolve_load_dtype(img.dtype))


def stack_dtype(images: Sequence[np.ndarray]) -> np.dtype:
    """Common storage dtype for a stack: 16-bit if any frame is 16-bit."""
    for img in images:
        if img is not None and np.dtype(img.dtype) == UINT16:
            return UINT16
    return UINT8


def unify(images: Sequence[np.ndarray], dtype=None) -> List[np.ndarray]:
    """Convert a stack so every frame shares one dtype.

    A stack mixing 8- and 16-bit frames - a folder holding both JPEGs and 16-bit
    TIFFs, say - would otherwise break the fusion methods, which assume one full
    scale across the stack.
    """
    if not images:
        return list(images)
    target = np.dtype(dtype) if dtype is not None else stack_dtype(images)
    return [convert(img, target) if img is not None else None for img in images]


def to_display8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """8-bit copy for anything that has to hand pixels to Qt or an 8-bit codec."""
    if img is None:
        return None
    if np.dtype(img.dtype) == UINT8:
        return img
    return convert(img, UINT8)


def to_analysis8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """8-bit copy for CV routines that only accept 8-bit input.

    SIFT and findTransformECC are the two that matter here. Both are used to
    *measure* a transform, which is then applied to the full-depth frame, so
    dropping to 8 bits for the measurement costs nothing in the output.
    """
    return to_display8(img)


def describe(dtype) -> str:
    """Short human-readable depth label for logs and the status bar."""
    return f"{bits_for(dtype)}-bit"


def stack_summary(images: Sequence[np.ndarray]) -> str:
    """One-line depth description of a loaded stack, for the console."""
    if not images:
        return "empty stack"
    depths = sorted({bits_for(img.dtype) for img in images if img is not None})
    if len(depths) == 1:
        return f"{len(images)} frame(s), {depths[0]}-bit"
    joined = "/".join(f"{d}-bit" for d in depths)
    return f"{len(images)} frame(s), mixed depth ({joined})"


def supports_16bit(extension: str) -> bool:
    """Whether a container can actually store 16 bits per channel.

    PNG and TIFF can; JPEG and BMP cannot, so writing 16-bit data to them has to
    be narrowed first rather than silently mangled by the encoder.
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext in (".png", ".tif", ".tiff")


def prepare_for_write(img: np.ndarray, extension: str) -> np.ndarray:
    """Narrow an image to 8-bit when the target container cannot hold 16."""
    if img is None or not is_high_depth(img):
        return img
    if supports_16bit(extension):
        return img
    return to_display8(img)
