"""
GPU-accelerated JPEG decoding via torchvision's nvJPEG bindings.

The loader hands whole stacks of JPEG byte buffers here when an NVIDIA GPU is
available; everything else (PNG/TIFF/RAW, machines without CUDA, or files
nvJPEG cannot handle, such as progressive JPEGs) stays on the OpenCV path.
Decoded frames come back as BGR uint8 numpy arrays, so callers cannot tell
which path produced a frame.
"""

from typing import List, Optional

import numpy as np

# Below this many JPEGs the CUDA context / nvJPEG spin-up costs more than it
# saves, so callers should not bother asking.
MIN_IMAGES = 4

# Frames per nvJPEG call. Bounds VRAM: a 24 MP frame is ~72 MB decoded, so a
# batch of 8 stays well under 1 GB even before resize scratch buffers.
_BATCH = 8

# None = not probed yet, then True/False for the life of the process.
_available: Optional[bool] = None


def is_available() -> bool:
    """True when torch, torchvision and a working nvJPEG decoder are present.

    The first call pays for CUDA context creation and a 1-frame smoke decode;
    torchvision can import fine on machines where nvJPEG itself is broken, so
    availability is proven by decoding, not by imports.
    """
    global _available
    if _available is None:
        _available = _probe()
    return _available


def _probe() -> bool:
    try:
        import cv2
        import torch
        from torchvision.io import decode_jpeg
        if not torch.cuda.is_available():
            return False
        ok, buf = cv2.imencode('.jpg', np.zeros((8, 8, 3), np.uint8))
        if not ok:
            return False
        decode_jpeg(torch.from_numpy(buf.reshape(-1)), device='cuda')
        return True
    except Exception:
        return False


def device_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "CUDA"


def decode_jpegs(
    datas: List[Optional[np.ndarray]],
    scale_factor: float = 1.0,
) -> List[Optional[np.ndarray]]:
    """Decode JPEG byte buffers on the GPU, in input order.

    Each entry of `datas` is the raw file content as a uint8 array (or None
    for files that could not be read). The result holds a BGR uint8 image per
    entry, or None where nvJPEG refused the file - the caller is expected to
    retry those on the CPU path. Downscaling happens on the GPU before the
    copy back, so a reduced stack also transfers less over PCIe.
    """
    import torch
    from torchvision.io import decode_jpeg, ImageReadMode

    out: List[Optional[np.ndarray]] = [None] * len(datas)
    pending = [(i, torch.from_numpy(d)) for i, d in enumerate(datas) if d is not None and d.size > 0]

    for start in range(0, len(pending), _BATCH):
        chunk = pending[start:start + _BATCH]
        try:
            decoded = decode_jpeg([t for _, t in chunk], mode=ImageReadMode.RGB, device='cuda')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue  # whole chunk falls back to CPU
        except Exception:
            # One bad file (progressive JPEG, truncated data) fails the whole
            # batched call; retry singly so the rest still decode on the GPU.
            decoded = []
            for _, t in chunk:
                try:
                    decoded.append(decode_jpeg(t, mode=ImageReadMode.RGB, device='cuda'))
                except Exception:
                    decoded.append(None)
        for (index, _), img in zip(chunk, decoded):
            if img is not None:
                out[index] = _to_bgr_numpy(img, scale_factor)
    return out


def _to_bgr_numpy(img, scale_factor: float) -> np.ndarray:
    """(3, H, W) RGB uint8 CUDA tensor -> (H, W, 3) BGR uint8 numpy array."""
    import torch
    import torch.nn.functional as F

    if scale_factor != 1.0 and 0 < scale_factor < 1.0:
        # Area interpolation is the tensor analogue of cv2.INTER_AREA, which
        # the CPU path uses for the same downscale.
        height = int(img.shape[1] * scale_factor)
        width = int(img.shape[2] * scale_factor)
        img = F.interpolate(img.unsqueeze(0).float(), size=(height, width), mode='area')
        img = img.squeeze(0).round_().clamp_(0, 255).to(torch.uint8)
    return img.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
