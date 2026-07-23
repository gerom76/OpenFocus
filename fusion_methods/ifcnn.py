"""IFCNN refinement stage.

Runs after a fusion method has produced an all-in-focus candidate. The candidate
and the aligned source stack are encoded by IFCNN and merged element-wise in
feature space, so detail the fusion step missed (blur bleeding around edges,
pixels picked from the wrong slice) can be recovered from whichever source frame
actually holds it. Only the difference the merge makes is applied to the
candidate, so the network's lossy round trip is not charged to the whole frame;
see ``_refine_block``.

The pretrained weights are not bundled: drop the official IFCNN checkpoint at
``weights/ifcnn.pth`` to enable the stage.
"""

# Conditional import, to avoid errors when it is not needed
try:
    import torch
except ImportError:
    torch = None

import os
import numpy as np
import cv2

from utils import resource_path, bitdepth

# ImageNet statistics; IFCNN was trained on inputs normalized this way
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MODEL_PATH = os.path.join("weights", "ifcnn.pth")

# ================= Global cache variables =================
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None

# Cache the available accelerator type (detected once at module load)
_MPS_AVAILABLE = None
_CUDA_AVAILABLE = None


def _detect_accelerators():
    """Detect the available accelerator type on the system, run once at module load"""
    global _MPS_AVAILABLE, _CUDA_AVAILABLE
    try:
        import torch as _torch
        _MPS_AVAILABLE = hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available()
        _CUDA_AVAILABLE = _torch.cuda.is_available()
    except Exception:
        _MPS_AVAILABLE = False
        _CUDA_AVAILABLE = False


_detect_accelerators()


def get_ifcnn_model_path(model_path=None) -> str:
    """Resolve the IFCNN weights path, relative paths against the app resources."""
    if not model_path:
        model_path = DEFAULT_MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = resource_path(model_path)
    return model_path


def is_ifcnn_available(model_path=None) -> bool:
    """Return True when PyTorch is importable and the IFCNN weights are present."""
    if torch is None:
        return False
    return os.path.isfile(get_ifcnn_model_path(model_path))


def _load_state_dict(model, state_dict):
    """Load a checkpoint, tolerating only the batch-norm counters.

    Checkpoints published before ``num_batches_tracked`` became a batch-norm
    buffer lack those keys. They only steer momentum while training, so an eval
    run does not need them -- but every other key must still match exactly, so a
    genuinely wrong file is rejected instead of silently loading half a network.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unloadable = [key for key in missing if not key.endswith('num_batches_tracked')]
    if unloadable or unexpected:
        raise RuntimeError(
            f"IFCNN checkpoint does not match the network "
            f"(missing: {unloadable}, unexpected: {list(unexpected)})"
        )


def _get_model_and_device(model_path, use_gpu):
    """Get or initialize the global IFCNN model and device"""
    if torch is None:
        raise ImportError("PyTorch not installed")
    from core.models.ifcnn_network import build_ifcnn_from_state_dict

    if use_gpu and _MPS_AVAILABLE:
        device = torch.device('mps')
    elif use_gpu and _CUDA_AVAILABLE:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    global _GLOBAL_MODEL, _GLOBAL_DEVICE

    if _GLOBAL_MODEL is None:
        print("Loading IFCNN model (once)...")
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model = build_ifcnn_from_state_dict(state_dict)
        _load_state_dict(model, state_dict)
        model.to(device)
        model.eval()
        _GLOBAL_MODEL = model
        _GLOBAL_DEVICE = device
    else:
        model = _GLOBAL_MODEL
        if _GLOBAL_DEVICE != device:
            model.to(device)
            _GLOBAL_DEVICE = device

    return model, device


def _to_rgb(bgr_image):
    """BGR uint8/uint16 -> float RGB in [0, 1].

    Normalised by the frame's own full scale, so the network sees the same range
    at either depth.
    """
    return bitdepth.to_float01(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def _to_tensor(bgr_image, device):
    """BGR uint8/uint16 -> normalized RGB tensor of shape [1, 3, H, W]."""
    rgb = (_to_rgb(bgr_image) - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return tensor.unsqueeze(0).to(device)


def _from_tensor(tensor):
    """Normalized RGB tensor of shape [1, 3, H, W] -> float RGB, left unclipped.

    Clipping is deferred to _to_bgr so the difference of two decoded images can
    be taken at full precision.
    """
    rgb = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return rgb * _IMAGENET_STD + _IMAGENET_MEAN


def _to_bgr(rgb, out_dtype=np.uint8):
    """Float RGB in [0, 1] -> BGR image at `out_dtype`.

    from_float01 rounds rather than truncating: a bare cast drops half a level
    from every pixel, which shows up as a systematic darkening of the refined
    image.
    """
    return cv2.cvtColor(bitdepth.from_float01(rgb, out_dtype), cv2.COLOR_RGB2BGR)


def _refine_block(model, device, block_images):
    """Merge one block of images in feature space, as a correction to the first.

    ``block_images[0]`` is the fused candidate; the rest are the sources it may
    have taken detail from.

    The decoder's output is not returned neat. IFCNN's encode/decode round trip
    is not an identity map -- push an image through unchanged and it comes back
    with its colours moved several levels -- so adopting the decoded picture
    everywhere pays that cost over the whole frame in exchange for repairs that
    only occur near edges. Decoding the candidate's own features alone measures
    the round-trip error by itself, and subtracting it leaves just what merging
    the sources contributed. Where IFCNN found nothing to add, the candidate
    comes back untouched instead of drifting.
    """
    out_dtype = block_images[0].dtype
    if len(block_images) < 2:
        # Nothing to merge, so the round trip could only cost colour
        return block_images[0].copy()

    with torch.no_grad():
        lead_features = None
        running = None
        for image in block_images:
            features = model.encode(_to_tensor(image, device))
            if lead_features is None:
                lead_features = features
            running = model.accumulate(running, features)
        fused = model.finalize(running, len(block_images))
        merged = _from_tensor(model.decode(fused))
        round_trip = _from_tensor(model.decode(lead_features))

    return _to_bgr(_to_rgb(block_images[0]) + (merged - round_trip), out_dtype)


def _feather_weights(x0, y0, x1, y1, w, h, overlap):
    """Separable linear ramp used to blend overlapping tiles seamlessly."""
    tw, th = x1 - x0, y1 - y0

    def ramp(length, fade_start, fade_end):
        if length <= 1:
            return np.ones((1,), dtype=np.float32)
        index = np.arange(length, dtype=np.float32)
        weights = np.ones((length,), dtype=np.float32)
        limit = min(overlap, length - 1)
        if fade_start and limit > 0:
            weights = np.minimum(weights, np.clip(index / float(limit), 0.0, 1.0))
        if fade_end and limit > 0:
            weights = np.minimum(weights, np.clip((length - 1 - index) / float(limit), 0.0, 1.0))
        return weights

    wx = ramp(tw, x0 > 0, x1 < w)
    wy = ramp(th, y0 > 0, y1 < h)
    return np.outer(wy, wx).astype(np.float32)


def _refine_tiled(model, device, images, block_size, overlap):
    """Refine a large image tile by tile, blending the overlaps."""
    h, w = images[0].shape[:2]
    out_dtype = images[0].dtype
    full_scale = bitdepth.max_value(out_dtype)
    step = max(1, block_size - overlap)

    accumulator = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)

    for y0 in range(0, h, step):
        y1 = min(y0 + block_size, h)
        for x0 in range(0, w, step):
            x1 = min(x0 + block_size, w)

            crops = [image[y0:y1, x0:x1] for image in images]
            refined = _refine_block(model, device, crops).astype(np.float32)

            weights = _feather_weights(x0, y0, x1, y1, w, h, overlap)[:, :, None]
            accumulator[y0:y1, x0:x1] += refined * weights
            weight_sum[y0:y1, x0:x1] += weights

            if x1 >= w:
                break
        if y1 >= h:
            break

    weight_sum = np.maximum(weight_sum, 1e-6)
    # The accumulator holds levels, not normalised values, so it is clipped
    # against the stack's own full scale.
    return np.rint(np.clip(accumulator / weight_sum, 0, full_scale)).astype(out_dtype)


def _ifcnn_refine_impl(fusion_result, source_images, model_path, use_gpu,
                       tile_enabled=True, tile_block_size=1024, tile_overlap=128,
                       tile_threshold=2048):
    """
    Refine a fused image with IFCNN, using the aligned source stack as reference.

    Args:
        fusion_result: fused image produced by the fusion stage (BGR, uint8)
        source_images: aligned source stack (list of BGR uint8 images)
        model_path: path to the IFCNN weights file
        use_gpu: whether to use the GPU
        tile_enabled: process large images tile by tile
        tile_block_size: tile size in pixels
        tile_overlap: overlap between neighbouring tiles in pixels
        tile_threshold: longest side above which tiling kicks in

    Returns:
        the refined image (BGR format, uint8)
    """
    if torch is None:
        raise ImportError("PyTorch not installed")
    if fusion_result is None:
        raise ValueError("IFCNN refinement needs a fusion result")

    model, device = _get_model_and_device(model_path, use_gpu)

    device_name = 'CUDA' if device.type == 'cuda' else device.type.upper()
    print(f"Running IFCNN refinement on {device_name}...")

    # The fused candidate leads the stack; the sources only contribute detail it
    # is missing. Sources of a different size (ROI crop) cannot be referenced.
    h, w = fusion_result.shape[:2]
    images = [fusion_result]
    for image in (source_images or []):
        if image.shape[:2] == (h, w):
            images.append(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    print(f"IFCNN refinement: {len(images)} inputs ({w}x{h})")

    use_tiles = tile_enabled and max(h, w) > tile_threshold
    try:
        if use_tiles:
            return _refine_tiled(model, device, images, int(tile_block_size), int(tile_overlap))
        return _refine_block(model, device, images)
    except RuntimeError as exc:
        if device.type == 'cpu' or 'out of memory' not in str(exc).lower():
            raise
        # Retry on the CPU rather than losing the render to a GPU memory spike
        print("IFCNN refinement ran out of GPU memory, retrying on CPU...")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model, device = _get_model_and_device(model_path, use_gpu=False)
        if use_tiles:
            return _refine_tiled(model, device, images, int(tile_block_size), int(tile_overlap))
        return _refine_block(model, device, images)
