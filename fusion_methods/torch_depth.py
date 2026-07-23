"""Depth-aware numpy <-> torch conversion for the GPU fusion paths.

torch has no uint16 dtype, so a 16-bit stack cannot be handed to
``torch.from_numpy`` the way an 8-bit one can. Rather than transferring raw
levels and normalising on the device, 16-bit frames are normalised on the host
and uploaded as float32.

That costs 4 bytes per pixel on the host and over the bus instead of 2, but the
*device*-side peak is unchanged: every GPU path converts to float32 immediately
on arrival anyway, so nothing extra is resident on the GPU. The 8-bit path is
left exactly as it was - upload bytes, divide on device - so no existing run
gets slower.
"""

import numpy as np
import torch

from utils import bitdepth


def stack_to_float01(chunk, dev):
    """Upload a list of BGR frames as a normalised (B, 3, H, W) float32 tensor.

    Accepts uint8 or uint16 frames and returns values in [0, 1] either way.
    """
    arr = np.stack([np.ascontiguousarray(img) for img in chunk])
    if arr.dtype == np.uint8:
        # Cheap path: 1 byte per channel over the bus, scale on the device.
        return torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255.0
    # 16-bit (or anything else): normalise on the host, since torch cannot hold
    # uint16 at all.
    normalised = bitdepth.to_float01(arr)
    return torch.from_numpy(normalised).to(dev).permute(0, 3, 1, 2)


def image_to_float01(img, dev):
    """Upload a single BGR frame as a normalised (1, 3, H, W) float32 tensor."""
    return stack_to_float01([img], dev)


def from_float01(tensor, out_dtype):
    """Bring a normalised (3, H, W) or (H, W, 3) tensor back to a numpy image.

    `tensor` is expected in [0, 1]; the result is clipped, rounded and returned
    at `out_dtype`. uint8 is finished on the device (where the rounding is free
    and the transfer is smallest); uint16 is finished on the host because torch
    cannot express the target dtype.
    """
    out_dtype = np.dtype(out_dtype)
    scale = bitdepth.max_value(out_dtype)
    if out_dtype == bitdepth.UINT8:
        # .round() before the cast: a bare cast truncates, which biases every
        # value down half a level (docs/ALGORITHM_IMPROVEMENTS.md item 5).
        return (tensor * scale).round_().clamp_(0, scale).to(torch.uint8).cpu().numpy()
    scaled = (tensor * scale).round_().clamp_(0, scale).cpu().numpy()
    return scaled.astype(out_dtype)
