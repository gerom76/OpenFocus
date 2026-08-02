"""Release memory the render pipeline no longer needs.

torch keeps freed GPU allocations in a process-wide caching pool so the next
operation can reuse them without another driver round-trip. After a large render
that looks like leaked VRAM (and pinned host memory) sitting on the app. This
module is the single place that hands that cache back once a render or batch job
is done; the next render simply re-warms the pool on first use.

CuPy's pools were freed here too, until registration stopped warping on the GPU
(docs/REGISTRATION_IMPROVEMENTS.md item 4) and left nothing in the app that
allocates through CuPy at all.

Only libraries already present in sys.modules are touched, so a CPU-only run
never pays a GPU library import just to free nothing.
"""

import gc
import sys
from typing import Optional

try:
    import psutil
except ImportError:  # psutil is a hard requirement, but never fail a load over it
    psutil = None

# Share of the memory free at load time that the transient decode buffers are
# allowed to occupy. Decoding is bounded by this rather than by the CPU count
# because a RAW worker holds several full frames at once: at 24 MP that is
# ~600 MB per thread, so one thread per core alone reserves more than a
# modest machine has, on top of the stack it is filling.
DECODE_MEMORY_SHARE = 0.12


def available_bytes() -> Optional[int]:
    """Physical memory that can be handed out without swapping, or None."""
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def format_bytes(count: float) -> str:
    """Byte count as a short human-readable string, for logs and warnings."""
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.1f} GB"


def decode_worker_limit(frame_bytes: Optional[int], frames_in_flight: int, requested: int) -> int:
    """Cap decode threads so their scratch buffers fit in free memory.

    `frame_bytes` is the decoded size of one frame and `frames_in_flight` how
    many of those a single worker holds at its peak. Returns `requested`
    unchanged when the frame size or the free memory is unknown, and never
    goes below 2 - a stack that cannot afford two decode threads will not fit
    in memory anyway, and the caller warns about that separately.
    """
    available = available_bytes()
    if not frame_bytes or available is None:
        return requested
    per_worker = frame_bytes * max(1, frames_in_flight)
    affordable = int(available * DECODE_MEMORY_SHARE) // per_worker
    return max(2, min(requested, int(affordable)))


def release_render_memory() -> None:
    """Return cached render allocations to the OS and GPU driver.

    Safe to call from any thread. Cached fusion models (IFCNN, StackMFF-V4)
    are deliberately left loaded - they are small compared to the image
    buffers and reloading them would slow every render.
    """
    # Collect Python-side garbage first so the pool below actually sees the
    # dropped tensors as free blocks.
    gc.collect()

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

