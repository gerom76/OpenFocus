"""
How much a frame shrinks as it is loaded.

The downsample dialog offers two ways to ask for the same thing: a percentage
of the source size, or a target length for the longer edge in pixels. A
percentage is uniform over a stack whatever the frames measure; a long-edge
target is resolved per frame, so a stack of mixed sizes still comes out with
one common long edge. Neither ever enlarges a frame - downsampling is where
memory is saved, not where resolution is invented.
"""

from typing import Optional, Tuple

# Offered when the dialog has no source frame to take a default from.
DEFAULT_LONG_EDGE = 2000

# The spin box bounds: below 64 px nothing survives focus detection, and the
# upper end is well past any sensor the loader is likely to meet.
MIN_LONG_EDGE = 64
MAX_LONG_EDGE = 60000


def target_size(
    width: int,
    height: int,
    scale_factor: float = 1.0,
    target_long_edge: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """(width, height) this frame should be resized to, or None to leave it alone.

    A long-edge target wins over the scale factor, so callers can pass the
    scale they already had alongside the pixel target the user just picked.
    """
    if target_long_edge is not None and target_long_edge > 0:
        longest = max(width, height)
        if longest <= target_long_edge:
            return None
        ratio = target_long_edge / longest
        return max(1, round(width * ratio)), max(1, round(height * ratio))

    if scale_factor != 1.0 and 0 < scale_factor < 1.0:
        return max(1, int(width * scale_factor)), max(1, int(height * scale_factor))

    return None
