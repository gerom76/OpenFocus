import os
import glob
import re
import cv2
import numpy as np

from utils import bitdepth
from utils.image_utils import read_image_any_depth
from fusion_methods.gff import _map_in_order

# Reference:
# Burt P J, Adelson E H. The Laplacian pyramid as a compact image code[J].
# IEEE Transactions on Communications, 1983, 31(4): 532-540.
# Multi-focus fusion selects, band by band, the coefficient carrying the most
# local energy - the classic "choose-max" rule on a Laplacian decomposition.

# Number of pyramid levels (pyrDown steps) when the caller does not ask for a
# specific depth. Five bands is enough to separate coarse structure from fine
# texture on the image sizes this app renders, while staying cheap.
DEFAULT_LEVELS = 5

# Side of the window the per-band focus energy is pooled over before the
# choose-max decision. Pooling turns a per-pixel |Laplacian| - which is noisy
# and zero on flat-but-in-focus areas - into a region measure, so the selection
# follows genuinely sharp regions instead of speckling between slices. Kept
# internal: it trades off with level count, which is the user-facing dial.
_ENERGY_WINDOW = 5


def _resolve_levels(height, width, requested):
    """Clamp the decomposition depth so the coarsest band stays workable.

    Each level halves both sides; decomposing past a handful of pixels buys no
    focus information and risks a degenerate 1-pixel band. The depth is capped
    so the smallest Gaussian level keeps both sides >= 2.
    """
    requested = DEFAULT_LEVELS if requested is None else max(1, int(requested))
    depth = 0
    smallest = min(height, width)
    while depth < requested and (smallest + 1) // 2 >= 2:
        smallest = (smallest + 1) // 2
        depth += 1
    return max(1, depth)


def _laplacian_pyramid(img, levels):
    """Return (detail levels 0..levels-1, coarsest Gaussian base).

    pyrUp is given an explicit destination size at every step so an odd-sized
    level reconstructs to exactly the size it came from - the pipeline hands
    back the geometry it was given, and the tiled renderer accumulates fused
    tiles at fixed offsets.
    """
    gaussian = [img]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

    detail = []
    for i in range(levels):
        target = (gaussian[i].shape[1], gaussian[i].shape[0])
        up = cv2.pyrUp(gaussian[i + 1], dstsize=target)
        detail.append(gaussian[i] - up)
    return detail, gaussian[levels]


def _band_energy(detail):
    """Local focus energy of one detail band: pooled squared response.

    Summing the squared coefficients across colour channels first keeps the
    decision on a single grey activity map, so all three channels of a pixel are
    taken from the same slice and colour cannot split across sources.
    """
    if detail.ndim == 3:
        squared = np.sum(detail * detail, axis=2)
    else:
        squared = detail * detail
    return cv2.boxFilter(squared, cv2.CV_32F, (_ENERGY_WINDOW, _ENERGY_WINDOW),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def pyramid_impl(input_source, img_resize=None, levels=None, thread_count=None):
    """Laplacian-pyramid multi-focus fusion (choose-max on band energy).

    Each frame is split into band-pass detail levels plus a low-frequency base.
    For every detail band the coefficient with the highest pooled energy across
    the stack wins the pixel; the base, which the focus stack shares, is
    averaged. Collapsing the fused pyramid gives the all-in-focus image.

    The stack's own depth is preserved end to end: an 8-bit stack returns
    uint8, a 16-bit stack returns uint16.
    """
    if thread_count is None:
        max_workers = min(8, os.cpu_count() or 4)
    else:
        try:
            max_workers = max(1, int(thread_count))
        except (TypeError, ValueError):
            max_workers = min(8, os.cpu_count() or 4)

    # ---------- Data loading ----------
    if isinstance(input_source, str):
        filenames = os.listdir(input_source)
        if not filenames:
            raise ValueError("Input folder is empty or no images were found")
        suffixes = [os.path.splitext(f)[1] for f in filenames if os.path.splitext(f)[1]]
        img_ext = suffixes[0] if suffixes else ''
        img_paths = glob.glob(os.path.join(input_source, '*' + img_ext))

        def sort_key(path):
            nums = re.findall(r"\d+", os.path.basename(path))
            return int(nums[-1]) if nums else path
        img_paths.sort(key=sort_key)

        stack_ori = [read_image_any_depth(p) for p in img_paths]
        stack_ori = [img for img in stack_ori if img is not None]
    else:
        stack_ori = list(input_source)

    if not stack_ori:
        raise ValueError("No image data was loaded")

    out_dtype = bitdepth.stack_dtype(stack_ori)
    num_images = len(stack_ori)

    if img_resize:
        cols, rows = int(img_resize[0]), int(img_resize[1])
    else:
        rows, cols = stack_ori[0].shape[:2]

    levels = _resolve_levels(rows, cols, levels)

    def load_float(k):
        # Per-frame resize and float conversion, so only the handful of frames
        # in flight are ever held as float32 rather than the whole stack.
        img = stack_ori[k]
        if img_resize and (img.shape[1], img.shape[0]) != (cols, rows):
            img = cv2.resize(img, (cols, rows))
        return bitdepth.to_float01(img)

    # A single frame decides every band's shape; all frames share it because
    # they share (rows, cols) after the resize step above.
    template_detail, template_base = _laplacian_pyramid(load_float(0), levels)

    fused_detail = [np.zeros_like(band) for band in template_detail]
    best_energy = [np.full(band.shape[:2], -np.inf, dtype=np.float32)
                   for band in template_detail]
    base_accumulator = np.zeros_like(template_base)
    del template_detail, template_base

    def decompose(k):
        return _laplacian_pyramid(load_float(k), levels)

    # Frames are decomposed on a thread pool but reduced in index order, so the
    # base sum and the strict-'>' tie-break are bitwise-stable run to run.
    for _, (detail, base) in _map_in_order(decompose, num_images, max_workers):
        base_accumulator += base
        for i, band in enumerate(detail):
            energy = _band_energy(band)
            better = energy > best_energy[i]
            np.copyto(best_energy[i], energy, where=better)
            if band.ndim == 3:
                np.copyto(fused_detail[i], band, where=better[:, :, np.newaxis])
            else:
                np.copyto(fused_detail[i], band, where=better)
        del detail, base

    # ---------- Reconstruction ----------
    fused = base_accumulator / num_images
    for i in range(levels - 1, -1, -1):
        target = (fused_detail[i].shape[1], fused_detail[i].shape[0])
        fused = fused_detail[i] + cv2.pyrUp(fused, dstsize=target)

    return bitdepth.from_float01(fused, out_dtype)
