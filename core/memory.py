"""Release memory the render pipeline no longer needs.

torch and cupy keep freed GPU allocations in process-wide caching pools so the
next operation can reuse them without another driver round-trip. After a large
render that looks like leaked VRAM (and pinned host memory) sitting on the app.
This module is the single place that hands those caches back once a render or
batch job is done; the next render simply re-warms the pools on first use.

Only libraries already present in sys.modules are touched, so a CPU-only run
never pays a GPU library import just to free nothing.
"""

import gc
import sys


def release_render_memory() -> None:
    """Return cached render allocations to the OS and GPU driver.

    Safe to call from any thread. Cached fusion models (IFCNN, StackMFF-V4)
    are deliberately left loaded - they are small compared to the image
    buffers and reloading them would slow every render.
    """
    # Collect Python-side garbage first so the pools below actually see the
    # dropped arrays/tensors as free blocks.
    gc.collect()

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
