"""
GPU-accelerated image decoding.

Two entry points, both used by the loader when an NVIDIA GPU is available:

- ``decode_jpegs`` decodes JPEG byte buffers with torchvision's nvJPEG
  bindings. Files nvJPEG cannot handle (progressive JPEGs, bad data) come
  back as None and stay on the OpenCV path.
- ``postprocess_raw`` runs the develop stage of RAW loading on the GPU:
  LibRaw still unpacks the mosaic on the CPU (there is no GPU LibRaw), but
  the expensive part - demosaicing plus white balance, colour matrix,
  auto-brightness and gamma - is reimplemented in torch, mirroring
  ``rawpy.postprocess(use_camera_wb=True)``. Demosaicing uses the
  Malvar-He-Cutler linear kernels, so results differ from LibRaw's AHD only
  in fine edge detail. Returns None for anything unusual (non-Bayer sensor,
  missing matrices, exotic orientation), which sends the file back to LibRaw.

Decoded frames come back as BGR numpy arrays, so callers cannot tell which
path produced a frame.
"""

import threading
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

# User-facing switch (Settings > GPU Image Loading), persisted by the settings
# manager. Kept separate from the capability probe so re-enabling never has to
# re-detect the hardware.
_enabled: bool = True


def set_enabled(enabled: bool) -> None:
    """Turn GPU image loading on or off; loaders honour it via is_available()."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    """The user's setting alone, regardless of whether a GPU is present."""
    return _enabled


# The probe creates the CUDA context. Loader worker threads all ask
# is_available() around the same moment on the first RAW stack, and letting
# each of them run its own probe concurrently serialises inside the driver
# and can stall loading for many seconds - so exactly one thread probes.
_PROBE_LOCK = threading.Lock()


def is_available() -> bool:
    """True when GPU loading is enabled and a working nvJPEG decoder exists.

    Checked before the capability probe so a disabled setting also skips CUDA
    context creation. The first probe pays for that context plus a 1-frame
    smoke decode; torchvision can import fine on machines where nvJPEG itself
    is broken, so availability is proven by decoding, not by imports.
    """
    global _available
    if not _enabled:
        return False
    if _available is None:
        with _PROBE_LOCK:
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


# sRGB -> XYZ (D65), the same constants dcraw uses to derive the camera matrix.
_XYZ_FROM_RGB = np.array([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
])

# LibRaw flip codes -> np.rot90 quarter turns (0 none, 3 = 180, 5/6 = 90).
_FLIP_TO_ROT90 = {0: 0, 3: 2, 5: 1, 6: 3}

# dcraw's default auto-brightness: scale so 1% of pixels clip.
_AUTO_BRIGHT_CLIP = 0.01
_HIST_BINS = 8192

# Malvar-He-Cutler 5x5 demosaic kernels (all / 8).
# _K_GREEN estimates G at an R or B site; _K_SAME_ROW estimates R (or B) at a
# G site whose horizontal neighbours carry that colour, _K_SAME_COL is its
# transpose for vertical neighbours; _K_DIAG estimates R at B sites and B at
# R sites, whose known samples sit on the diagonals.
_K_GREEN = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0],
], dtype=np.float32) / 8.0
_K_SAME_ROW = np.array([
    [0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0],
], dtype=np.float32) / 8.0
_K_SAME_COL = _K_SAME_ROW.T.copy()
_K_DIAG = np.array([
    [0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0],
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0],
], dtype=np.float32) / 8.0


# One RAW image on the GPU at a time: concurrent develops from the loader's
# worker threads fight over the allocator and can push VRAM into out-of-memory
# retries. Non-blocking - a thread that finds the GPU busy develops on the CPU
# instead of idling a core in a queue.
_RAW_GPU_LOCK = threading.Lock()

# 5x5 kernels uploaded once per process instead of once per image.
_kernel_cache: dict = {}


def postprocess_raw(raw, output_bps: int = 16) -> Optional[np.ndarray]:
    """Develop an opened rawpy image on the GPU; BGR uint8/uint16 out.

    Mirrors ``raw.postprocess(use_camera_wb=True, output_bps=...)`` closely
    enough for fusion work: camera white balance, dcraw's highlight clip and
    1%-clip auto-brightness, BT.709 gamma. Returns None whenever the file
    falls outside the straightforward Bayer case - or whenever the GPU is
    busy or out of memory - so the caller can fall back to LibRaw's own
    postprocess.
    """
    import torch

    pattern = np.asarray(raw.raw_pattern)
    if pattern.shape != (2, 2):
        return None
    desc = raw.color_desc.decode('ascii', 'replace')
    cell = [[desc[pattern[y, x]] for x in range(2)] for y in range(2)]
    if sorted(cell[0] + cell[1]) != ['B', 'G', 'G', 'R']:
        return None

    wb = list(raw.camera_whitebalance)
    if len(wb) < 3 or min(wb[:3]) <= 0:
        return None
    cam_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    if not np.any(cam_xyz):
        return None
    flip = raw.sizes.flip
    if flip not in _FLIP_TO_ROT90:
        return None

    # cam_rgb = cam_xyz @ (sRGB -> XYZ), rows normalised so camera white maps
    # to white, then inverted - exactly dcraw's cam_xyz_coeff().
    cam_rgb = cam_xyz @ _XYZ_FROM_RGB
    row_sums = cam_rgb.sum(axis=1, keepdims=True)
    if np.any(np.abs(row_sums) < 1e-8):
        return None
    rgb_cam = np.linalg.inv(cam_rgb / row_sums)

    cfa_np = raw.raw_image_visible
    height, width = cfa_np.shape[0] & ~1, cfa_np.shape[1] & ~1
    if height < 6 or width < 6:
        return None
    cfa_np = cfa_np[:height, :width]

    black = raw.black_level_per_channel
    white = float(raw.white_level)
    # Per-CFA-site black level and white-balance gain, as 2x2 tiles. G2 shares
    # the G gain when the camera reports 0 for it, matching LibRaw.
    gains = {'R': wb[0], 'G': wb[1], 'B': wb[2]}
    wb_norm = min(gains.values())
    black_tile = np.array([[black[pattern[y, x]] for x in range(2)] for y in range(2)], dtype=np.float32)
    gain_tile = np.array([[gains[cell[y][x]] / wb_norm for x in range(2)] for y in range(2)], dtype=np.float32)

    scale = white - black_tile.max()
    if scale <= 0:
        return None

    if not _RAW_GPU_LOCK.acquire(blocking=False):
        return None
    try:
        return _develop_raw(cfa_np, cell, black_tile, gain_tile, scale, rgb_cam, flip, output_bps)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        _RAW_GPU_LOCK.release()


def _develop_raw(cfa_np, cell, black_tile, gain_tile, scale, rgb_cam, flip, output_bps) -> np.ndarray:
    """The GPU develop itself; caller holds the lock and handles OOM."""
    import torch

    height, width = cfa_np.shape
    device = torch.device('cuda')

    # Upload at the source's 16 bits and widen on the GPU: half the PCIe
    # traffic of sending float32.
    cfa = torch.from_numpy(np.ascontiguousarray(cfa_np)).to(device).to(torch.float32)
    # dcraw's scale_colors with highlight mode 0: the smallest gain lands at
    # 1.0 and everything above full scale is clipped. Applied in place per
    # 2x2 phase, so no full-size black/gain planes are ever materialised.
    for y in range(2):
        for x in range(2):
            cfa[y::2, x::2].sub_(float(black_tile[y, x])).mul_(float(gain_tile[y, x]) / scale)
    cfa.clamp_(0.0, 1.0)

    rgb = _demosaic_mhc(cfa, cell)
    del cfa

    matrix = torch.from_numpy(rgb_cam.astype(np.float32)).to(device)
    rgb = torch.einsum('ij,jhw->ihw', matrix, rgb).clamp_(0.0, 1.0)

    # Auto-brightness: per channel, find the level whose top tail holds 1% of
    # the pixels; the largest such level becomes the new white point. Batched
    # so the whole search costs a single host sync.
    clip_count = float(height * width * _AUTO_BRIGHT_CLIP)
    hists = torch.stack([torch.histc(rgb[c], bins=_HIST_BINS, min=0.0, max=1.0) for c in range(3)])
    tails = hists.flip(1).cumsum(1)
    limits = torch.full((3, 1), clip_count, device=device)
    first = int(torch.searchsorted(tails, limits).min().item())
    white_point = (_HIST_BINS - first) / _HIST_BINS
    if white_point > 1e-4:
        rgb = rgb.div_(white_point).clamp_(0.0, 1.0)

    # BT.709 transfer curve, dcraw's default gamma (2.222, 4.5).
    rgb = torch.where(rgb < 0.018, rgb * 4.5, 1.099 * rgb.clamp(min=0.018).pow(1.0 / 2.222) - 0.099)

    if output_bps == 8:
        out = (rgb * 255.0).round_().clamp_(0, 255).to(torch.uint8)
    else:
        out = (rgb * 65535.0).round_().clamp_(0, 65535).to(torch.uint16)
    bgr = out.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()

    turns = _FLIP_TO_ROT90[flip]
    if turns:
        bgr = np.ascontiguousarray(np.rot90(bgr, turns))
    return bgr


def _cached_kernel(kernel: np.ndarray, device):
    """The 5x5 kernel as a (1, 1, 5, 5) tensor, uploaded once per process."""
    import torch
    tensor = _kernel_cache.get(id(kernel))
    if tensor is None:
        tensor = torch.from_numpy(kernel).to(device).reshape(1, 1, 5, 5)
        _kernel_cache[id(kernel)] = tensor
    return tensor


def _demosaic_mhc(cfa, cell):
    """Malvar-He-Cutler demosaic of a white-balanced CFA plane.

    `cfa` is a (H, W) float CUDA tensor in [0, 1]; `cell` the 2x2 colour
    letters. Returns a (3, H, W) RGB tensor. Four fixed 5x5 convolutions
    produce every missing sample; strided slice assignment then picks the
    right estimate per site, so no mask planes are allocated.
    """
    import torch
    import torch.nn.functional as F

    padded = F.pad(cfa[None, None], (2, 2, 2, 2), mode='reflect')
    est_green = F.conv2d(padded, _cached_kernel(_K_GREEN, cfa.device))[0, 0]
    est_row = F.conv2d(padded, _cached_kernel(_K_SAME_ROW, cfa.device))[0, 0]
    est_col = F.conv2d(padded, _cached_kernel(_K_SAME_COL, cfa.device))[0, 0]
    est_diag = F.conv2d(padded, _cached_kernel(_K_DIAG, cfa.device))[0, 0]
    del padded

    red = torch.empty_like(cfa)
    green = torch.empty_like(cfa)
    blue = torch.empty_like(cfa)
    for y in range(2):
        for x in range(2):
            site = (slice(y, None, 2), slice(x, None, 2))
            colour = cell[y][x]
            if colour == 'R':
                red[site] = cfa[site]
                green[site] = est_green[site]
                blue[site] = est_diag[site]
            elif colour == 'B':
                blue[site] = cfa[site]
                green[site] = est_green[site]
                red[site] = est_diag[site]
            else:
                green[site] = cfa[site]
                # A G site with R beside it has B above/below, and vice versa.
                if cell[y][1 - x] == 'R':
                    red[site] = est_row[site]
                    blue[site] = est_col[site]
                else:
                    blue[site] = est_row[site]
                    red[site] = est_col[site]
    return torch.stack((red, green, blue)).clamp_(0.0, 1.0)


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
