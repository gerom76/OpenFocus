# Reference:
# Haghighat M B A, Aghagolzadeh A, Seyedarabi H. Multi-focus image fusion for visual sensor networks in DCT domain[J]. Computers & Electrical Engineering, 2011, 37(5): 789-797.
import glob
import os
import time
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np

from utils import bitdepth, dng, jxl
from utils.image_utils import read_image_any_depth

ArraySource = Sequence[np.ndarray]

# --- Focus-measure tuning -------------------------------------------------
# The paper scores a block by the energy of all its AC coefficients, which is
# the block's plain pixel variance. That measure answers "how much contrast is
# here", not "how sharp is it": a heavily defocused highlight lays a smooth
# brightness ramp across the frame, and a ramp crossing one block carries more
# variance than the fine, low-amplitude texture that is genuinely in focus
# there. The blurred frame then wins whole regions, which is what produced the
# flat homogeneous patches this measure replaces. Dropping the lowest AC band -
# everything coarser than a block - leaves only detail a block can actually
# resolve, so a defocused wash scores near zero however bright it is.
#
# The cutoff is the block itself: the high-pass window is the block size, so
# structure spanning more than one block is subtracted off before the energy is
# measured. By Parseval the result is still DCT AC energy, just band-limited
# from below.
_HIGHPASS_SCALE = 1.0

# Side of the window the block energies are pooled over before the choose-max,
# in blocks. A single block's energy is noisy, and the frame that happens to win
# it is noisier still; pooling turns the decision into a regional one, the same
# reason pyramid.py pools its band energy before choosing (_ENERGY_WINDOW).
_POOL_WINDOW = 3

# Each frame's energies are divided by its own noise level before frames are
# compared, and that level is read off the frame as this percentile of its block
# energies - low enough to land in whatever the frame's quietest region is,
# which in a focus stack is always somewhere out of focus.
#
# Without this, sensor noise decides every block that holds no detail, and it
# does not decide them evenly: photon noise grows with brightness, so a frame
# that is a bright defocused veil carries 4-12x the high-pass energy of a dark
# smooth one *in grain alone* (measured on flat patches at 8-bit levels 30 vs
# 232). The veil then wins every detail-free block by a wide enough margin to
# look decisive, and its flat tone is copied through - one washed-out frame
# stamping identically coloured patches across the render. Dividing by the
# frame's own noise puts every frame at about 1.0 where it resolves nothing, so
# those blocks tie instead, and the tie is settled below.
_NOISE_PERCENTILE = 10.0

# Guards the division above when a frame is perfectly flat (a synthetic or
# fully clipped frame percentiles to zero). Well under the energy of a flat
# 8-bit patch dithering by half a level, so it never displaces a real estimate.
_NOISE_FLOOR = 1e-7

# A block is not decided by a single winner but by every frame that comes within
# this fraction of the peak energy - the block's own depth of field - and it
# takes the middle of that run as its focal plane.
#
# Picking one winner does not survive a deep stack. Neighbouring frames of a
# 274-frame stack differ by well under any usable margin, so a "is the winner
# clearly ahead" test fails on 91% of blocks, and whatever settles those blocks
# is what actually draws the picture. Settling them by their neighbours' choice
# is worse than it sounds: the nearest decided block can be 200 frames away in
# the stack, and frames that far apart look nothing alike in a defocused area,
# so the render tore into chunks along the block lattice.
#
# The middle of the in-focus set has no such failure mode. In focus the set is a
# short run around the true focal plane and its middle is that plane, to
# sub-frame accuracy. Out of focus every frame is equally poor, the set is the
# whole stack, and the middle is mid-stack - the same answer everywhere, so
# defocused regions come out uniform instead of patchwork. The field it produces
# is continuous, so neighbouring blocks land on neighbouring frames, which look
# alike.
#
# The width matters as much as the idea, and it is a two-sided choice.
#
# Too narrow and the plateau is measured against a noise outlier wherever
# nothing is in focus: frames sit within 7% of each other in such a block
# (measured on the veil fixture), so a narrow band admits only the frames that
# happen to be noisiest there, which is how a bright veil frame took every
# featureless block. At 0.90 that fixture fails outright - 22.5% of a smooth
# dark body comes back veiled.
#
# Too wide and regions that are never quite in focus - a background beyond the
# stack's reach - average over frames that do resolve them a little, and come
# out flatter than the best frame would render them. On the deep_stack scenario
# that costs 1.6 dB between 0.80 and 0.70.
#
# 0.80 sits between the two, with margin on the side that produces an artefact
# rather than a softness: the veil fixture is clean from 0.85 down, and this is
# comfortably clear of that edge.
_PLATEAU = 0.8

# How far a frame has to fall, as a fraction of the in-focus bar above, before
# it is taken to have ended the run rather than to have dipped inside it.
#
# The in-focus set is read as an interval - the frames from where the subject
# comes into focus to where it leaves again - so the plane is taken over the run
# containing the peak rather than over every frame in the stack that happens to
# clear the bar. Averaging the whole set instead lets one distant frame that
# clears it on grain drag the plane into a part of the stack that resolves
# nothing there: measured on the deep_stack scenario, 6.6% of blocks were given
# a focal plane no frame was actually in focus at, and 43% of blocks had an
# in-focus set that was not one run.
#
# Reading it as a run only works if the run survives noise, and that is what
# this second, lower bar is for. Where nothing is in focus every frame sits at
# about the same energy, the peak is whichever one grain favoured, and frames
# dip below the in-focus bar at random - so a run that ends at the first dip
# collapses onto the peak frame, and item 17's veil artefact comes straight back
# (71% of the veil fixture's smooth body, against 0% here). Hysteresis is the
# right shape for that because it asks how deep the dip is rather than how long
# it is: grain moves a defocused frame by a few percent, while leaving the depth
# of field costs it most of its energy.
#
# The two ends of the range are both failures: 1.0 ends the run at the first dip
# (71% of the veil body at the default plateau), and 0.0 never ends one, which
# puts the far end of the stack back inside the interval and gives up most of
# what this is for - order agreement on deep_stack falls from 34.3 dB to 26.6.
#
# The usable end is the top. Measured on the veil fixture at the strictest
# plateau a caller can ask for (0.85), the artefact is absent at 0.85 and below
# and takes 24% of the body at 0.90, so the working range stops just above this.
# Below 0.70 the fixtures stop moving at all. 0.75 therefore sits inside the
# flat part with a clear step of margin on the side that produces an artefact,
# which is the same way _PLATEAU above is placed. It costs 0.5 dB on deep_stack
# against sitting at the cliff edge, all of it in a background no frame resolves
# - the softness half of the trade item 18 describes, not detail.
_RUN_EXIT = 0.75

# The focal-plane field is turned into an image by weighting each frame by how
# close it is to the field, and upsampling those weights across the block
# lattice rather than the frame indices. Interpolating between two neighbouring
# frames costs no meaningful sharpness - they are one focus step apart - and it
# is what removes the last of the staircase, since the weights vary continuously
# across a block boundary where a frame index cannot.
_BLEND = True

# Steps per frame the focal-plane map is quantised to before it is filtered.
#
# The run's midpoint is a multiple of half a frame, so two steps hold it exactly
# and the map can be carried through the median filter - which needs integers -
# without being rounded to a whole frame first. That rounding was worth about a
# decibel on its own (sensor_noise 38.2 -> 41.8, long_stack 28.3 -> 30.5): a
# block whose plane sits between two frames is composited from both rather than
# snapped to one. It also makes the method exactly invariant to reversing the
# stack, which rounding a half to even cannot be - reversing turns k + 0.5 into
# (n - 1 - k) - 0.5 and the two round to frames that are not each other's mirror.
_SUBFRAME = 2


def _odd(value: int, limit: int) -> int:
    """An odd filter window of about `value`, never wider than `limit`.

    Returns 1 - a no-op window - when the map is too small to filter at all.
    """
    value = max(3, int(value) | 1)
    limit = max(1, int(limit))
    if limit < 3:
        return 1
    return min(value, limit if limit % 2 else limit - 1)


def _ensure_color_image(image: np.ndarray) -> np.ndarray:
    """Ensure the input image is in three-channel BGR format."""
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")

def _collect_images_from_folder(source_folder: str) -> Tuple[List[np.ndarray], List[str]]:
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp', '*.webp']
    extensions += [f"*{ext}" for ext in jxl.extensions()]
    extensions += [f"*{ext}" for ext in dng.extensions()]
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(source_folder, ext)))
    img_paths.sort()
    
    images = []
    for path in img_paths:
        img = read_image_any_depth(path)
        if img is not None:
            images.append(img)
    return images, img_paths

def _normalize_image_stack(images: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Unify image sizes and return a list of BGR images."""
    if not images:
        raise ValueError("Image stack is empty")
    
    # Get the reference size
    ref_img = images[0]
    target_h, target_w = ref_img.shape[:2]
    
    normalized = []
    for img in images:
        img_bgr = _ensure_color_image(np.ascontiguousarray(img))
        if img_bgr.shape[:2] != (target_h, target_w):
            img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        normalized.append(img_bgr)
        
    if target_h < 8 or target_w < 8:
        raise ValueError("Image size is too small")

    return normalized, (target_h, target_w)

def _normalised_energy(gray: np.ndarray, hp_window: int, pool_window: int,
                       map_w: int, map_h: int) -> np.ndarray:
    """One frame's per-block detail energy, in units of its own noise.

    High-pass, block-mean the squared detail, pool over neighbouring blocks,
    then divide by the frame's own noise level. The result is a signal-to-noise
    ratio: about 1 wherever the frame resolves nothing, far above it wherever
    the frame is in focus, and - the point of the division - on the same scale
    for a dark frame and a bright one.
    """
    # A box blur is used rather than a Gaussian because it costs the same per
    # pixel at any window size and is reproduced exactly by an average pool on
    # the GPU side (dct_torch.py).
    detail = gray - cv2.blur(gray, (hp_window, hp_window))

    # cv2.resize with INTER_AREA over an integer ratio is exactly a block mean,
    # and much faster than a per-block loop. The detail image has zero mean by
    # construction, so the mean of squares is already the variance - no E[X]^2
    # term to subtract.
    energy = cv2.resize(detail * detail, (map_w, map_h),
                        interpolation=cv2.INTER_AREA)
    pooled = cv2.boxFilter(energy, -1, (pool_window, pool_window),
                           normalize=True, borderType=cv2.BORDER_DEFAULT)
    noise = max(float(np.percentile(pooled, _NOISE_PERCENTILE)), _NOISE_FLOOR)
    return pooled / noise


def _compose(images: Sequence[np.ndarray], field: np.ndarray,
             block_size: int, shape: Tuple[int, int],
             out_dtype: np.dtype) -> np.ndarray:
    """Build the picture from the focal-plane map.

    Each frame is weighted by how close the map is to it, and the weights - not
    the frame indices - are what gets upsampled from the block lattice to
    pixels. Upsampling an index can only step from one frame to the next at a
    block edge, which is a seam; upsampling the weight ramps between them across
    the block, which is not. Where a whole region shares one frame the weight is
    exactly 1 and the pixels come through untouched.

    The map is fractional - a plane halfway between two frames draws half from
    each - so the frames that can contribute are the two the plane lies between,
    and each is composited over the bounding box of its own weight rather than
    the whole frame. A deep stack therefore costs little more than a shallow one.
    """
    h, w = shape
    h_trim, w_trim = field.shape[0] * block_size, field.shape[1] * block_size
    field = field.astype(np.float32, copy=False)

    accum = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)

    candidates = np.unique(np.concatenate(
        (np.floor(field), np.ceil(field)), axis=None)).astype(np.int64)
    for idx in candidates:
        block_w = np.clip(1.0 - np.abs(field - float(idx)), 0.0, 1.0)
        rows, cols = np.nonzero(block_w)
        if rows.size == 0:
            continue
        # Bounding box in blocks, one block of margin for the interpolation ramp
        r0, r1 = max(int(rows.min()) - 1, 0), min(int(rows.max()) + 2, field.shape[0])
        c0, c1 = max(int(cols.min()) - 1, 0), min(int(cols.max()) + 2, field.shape[1])
        y0, y1 = r0 * block_size, r1 * block_size
        x0, x1 = c0 * block_size, c1 * block_size

        patch = cv2.resize(block_w[r0:r1, c0:c1], (x1 - x0, y1 - y0),
                           interpolation=cv2.INTER_LINEAR)
        # The trimmed edge strip has no blocks of its own; extend the last row
        # and column of weights over it so the output keeps its full geometry.
        if y1 == h_trim and h_trim != h:
            patch = cv2.copyMakeBorder(patch, 0, h - h_trim, 0, 0, cv2.BORDER_REPLICATE)
            y1 = h
        if x1 == w_trim and w_trim != w:
            patch = cv2.copyMakeBorder(patch, 0, 0, 0, w - w_trim, cv2.BORDER_REPLICATE)
            x1 = w

        source = images[int(idx)][y0:y1, x0:x1]
        accum[y0:y1, x0:x1] += source.astype(np.float32) * patch[:, :, None]
        weights[y0:y1, x0:x1] += patch

    np.maximum(weights, 1e-6, out=weights)
    accum /= weights[:, :, None]
    ceiling = 65535 if out_dtype == bitdepth.UINT16 else 255
    return np.clip(np.rint(accum, out=accum), 0, ceiling).astype(out_dtype)


def _median_filter_map(block_map: np.ndarray, kernel_size: int) -> np.ndarray:
    """Median-filter a uint16 block map.

    cv2.medianBlur accepts 16-bit input only at apertures 3 and 5, so larger
    kernels are approximated by iterating the 5-aperture filter: each pass
    extends the effective radius by 2, so we run enough passes to cover the
    requested kernel's radius.
    """
    if kernel_size < 3:
        return block_map
    if kernel_size <= 5:
        return cv2.medianBlur(block_map, kernel_size)
    passes = ((kernel_size - 1) // 2 + 1) // 2
    for _ in range(passes):
        block_map = cv2.medianBlur(block_map, 5)
    return block_map


def _focal_plane(energy_of, count: int, threshold: np.ndarray) -> np.ndarray:
    """Each block's focal plane: the middle of its in-focus run, in frames.

    One streaming pass. `energy_of(idx)` returns frame `idx`'s normalised block
    energies and `threshold` is the bar a frame clears to count as in focus
    there, so nothing per-frame is retained and the cost is O(1) in stack depth.

    The run tracked is the one containing the peak - the frames on either side
    of the sharpest that are still within its depth of field. Frames elsewhere
    in the stack that clear the bar are not part of it, so a lone noisy frame at
    the far end of the sweep cannot pull the plane away from the depth the block
    is actually at. A run ends where a frame falls to _RUN_EXIT of the bar, not
    where one merely dips below it; only frames above the bar itself extend the
    run's ends, so the lower bar lengthens no plateau, it only keeps grain from
    cutting one in half.

    Where nothing is in focus every frame clears the bar, the run is the whole
    stack and the plane is mid-stack - the same answer everywhere, which is what
    keeps a defocused region uniform instead of patchwork.
    """
    shape = threshold.shape
    exit_bar = threshold * _RUN_EXIT
    lo = np.zeros(shape, np.float32)          # first frame of the peak's run
    hi = np.zeros(shape, np.float32)          # last frame of it
    best = np.full(shape, -np.inf, np.float32)
    run_start = np.zeros(shape, np.float32)   # first frame of the current run
    inside = np.zeros(shape, bool)            # is any run open?
    alive = np.zeros(shape, bool)             # is the peak's own run open?

    for idx in range(count):
        energy = energy_of(idx)
        in_focus = energy >= threshold
        # A run opens on a frame above the bar and stays open until one falls
        # below the lower bar, so a dip inside a plateau does not end it.
        was_inside = inside
        inside = (inside & (energy >= exit_bar)) | in_focus
        np.copyto(run_start, np.float32(idx), where=inside & ~was_inside)
        alive &= inside

        # The peak always clears its own threshold, so the run it lands in has
        # opened and run_start is the frame it opened at.
        peak_here = energy > best
        best = np.where(peak_here, energy, best)
        np.copyto(lo, run_start, where=peak_here)
        np.copyto(hi, np.float32(idx), where=peak_here | (alive & in_focus))
        alive |= peak_here

    return (lo + hi) * 0.5


def dct_focus_stack_fusion(
    source: Union[str, ArraySource],
    output_path: str = None,
    block_size: int = 8,
    kernel_size: int = 7,
    plateau: float = None,
    blend: bool = None,
) -> np.ndarray:
    """
    DCT-domain multi-focus fusion.

    Each block's detail energy is measured in every frame; the frames that come
    within _PLATEAU of the best are the ones in focus there, and the middle of
    the run of them containing the peak is the block's focal plane, to half a
    frame. The resulting map is median-filtered and then turned back into an
    image by weighting each frame by its distance from it - see the tuning notes
    at the top of this module for why a single winner per block does not survive
    a deep stack.

    The stack is read as a focus sweep, so the result depends on the order the
    frames arrive in - deliberately, since the plane it estimates is a position
    along that sweep. It is exactly invariant to the one re-ordering that is
    still a sweep, reversal.

    Optimization notes:
    By Parseval's theorem the energy of a block's DCT coefficients equals the
    energy of its pixels, so the measure needs no transform at all: a box
    high-pass (which drops the coefficients below the block's own frequency)
    followed by a block mean of the squared result, both done with cv2 filters
    and cv2.resize(INTER_AREA), replaces the originally extremely slow per-block
    DCT loop.

    Args:
        block_size: Side of the block every decision is made over.
        kernel_size: Median filter applied to the focal-plane map.
        plateau: How far below the peak a frame still counts as in focus.
            None uses _PLATEAU. Clamped: above ~0.85 detail-free regions start
            being decided by grain again (item 17), which is a defect rather
            than a preference, so the range stops short of it.
        blend: Composite by weighting neighbouring frames (the default) or copy
            each block from one frame verbatim. None uses _BLEND.
    """

    # --- 1. Parameter validation and preparation ---
    if kernel_size % 2 == 0:
        kernel_size += 1
    block_size = max(2, int(block_size))
    plateau = _PLATEAU if plateau is None else min(max(float(plateau), 0.1), 0.85)
    blend = _BLEND if blend is None else bool(blend)

    if isinstance(source, str):
        images, _ = _collect_images_from_folder(source)
    elif isinstance(source, (list, tuple)):
        images = [img for img in source if img is not None]
    else:
        raise TypeError("Source must be folder path or image list.")

    if len(images) < 2:
        raise ValueError("At least 2 images are required for fusion.")

    # Normalize and get the size
    normalized_images, (h, w) = _normalize_image_stack(images)

    # Compute the aligned size (must be an integer multiple of block_size)
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    map_h = h_trim // block_size
    map_w = w_trim // block_size

    if map_h == 0 or map_w == 0:
        raise ValueError("Block size is too large for image size.")

    # --- 2. Quickly compute the block energy map (core optimization) ---
    hp_window = _odd(round(block_size * _HIGHPASS_SCALE), min(h_trim, w_trim))
    pool_window = _odd(_POOL_WINDOW, min(map_h, map_w))

    def block_energy(idx: int) -> np.ndarray:
        # Grayscale, normalised to [0, 1] before squaring. Normalising is what
        # makes the energy measure usable at 16 bits: squaring raw 16-bit
        # levels reaches 4.3e9, where float32's 24-bit mantissa has a ULP of
        # ~256, so everything below that would quantise away. Working in [0, 1]
        # gives both depths the same headroom. Fusion decisions are unaffected
        # by the rescaling itself: scaling every frame by the same constant
        # leaves the per-block ranking unchanged.
        img_trim = normalized_images[idx][:h_trim, :w_trim]
        gray = bitdepth.to_float01(cv2.cvtColor(img_trim, cv2.COLOR_BGR2GRAY))
        return _normalised_energy(gray, hp_window, pool_window, map_w, map_h)

    # Pass 1: the best any frame manages on each block, which sets the bar for
    # what counts as in focus there. Running maximum, so memory does not grow
    # with the stack.
    peak_energy = np.full((map_h, map_w), -1.0, dtype=np.float32)
    for idx in range(len(normalized_images)):
        np.maximum(peak_energy, block_energy(idx), out=peak_energy)

    # Pass 2: the middle of the in-focus run around that peak - the block's
    # focal plane. Streaming, so this stays O(1) in stack depth too.
    field = _focal_plane(block_energy, len(normalized_images),
                         peak_energy * plateau)

    # --- 3. Consistency verification (median filtering) ---
    # The map is carried in units of 1/_SUBFRAME of a frame, so a plane that
    # sits between two frames survives the filter instead of being snapped to
    # one of them. uint16 covers any realistic stack depth at that resolution
    # and, unlike uint8, does not wrap indices at 256 frames; medianBlur
    # (apertures 3/5) accepts it. Without the median a speck of dust that is
    # sharp in exactly one frame drags its block onto that frame, and the block
    # shows up as a square in a defocused area.
    plane_map = np.clip(np.rint(field * _SUBFRAME), 0,
                        (len(normalized_images) - 1) * _SUBFRAME).astype(np.uint16)
    plane_map = _median_filter_map(plane_map, kernel_size)
    plane_map = _median_filter_map(plane_map, kernel_size)
    field = plane_map.astype(np.float32) / _SUBFRAME

    # --- 4. Reconstruction ---
    out_dtype = bitdepth.stack_dtype(normalized_images)
    if blend:
        fused_image = _compose(normalized_images, field, block_size,
                               (h, w), out_dtype)
    else:
        index_map = np.rint(field).astype(np.uint16)
        full_size_indices = cv2.resize(index_map, (w_trim, h_trim),
                                       interpolation=cv2.INTER_NEAREST)
        # The block grid only covers a multiple of block_size, so an image whose
        # dimensions do not divide evenly leaves a strip on the right and bottom.
        # Extend the last row/column of decisions over it rather than returning a
        # smaller image than we were given.
        if (h_trim, w_trim) != (h, w):
            full_size_indices = cv2.copyMakeBorder(
                full_size_indices, 0, h - h_trim, 0, w - w_trim,
                cv2.BORDER_REPLICATE)
        fused_image = np.zeros((h, w, 3), dtype=out_dtype)
        for idx in np.unique(index_map):
            mask = (full_size_indices == idx)
            fused_image[mask] = normalized_images[idx][mask]

    if output_path:
        cv2.imwrite(output_path, fused_image)

    return fused_image


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    t_start = time.time()
    
    # Change this path to your actual image folder
    TARGET_DIR = r"C:\Users\dell\Pictures\Helicon Focus\StackMFF V2 Used\Bug"
    OUTPUT_FILE = os.path.join(TARGET_DIR, "Fused_Result_Optimized.tif")
    
    BLOCK_SIZE = 8
    KERNEL_SIZE = 7
    
    if os.path.exists(TARGET_DIR):
        try:
            print(f"Start processing: {TARGET_DIR}")
            result = dct_focus_stack_fusion(
                source=TARGET_DIR,  # corrected parameter name
                output_path=OUTPUT_FILE,
                block_size=BLOCK_SIZE,
                kernel_size=KERNEL_SIZE
            )
            
            elapsed = time.time() - t_start
            print(f"Processing done, elapsed: {elapsed:.4f} s")
            
            if result is not None:
                # Show the result (limit the maximum display size)
                h, w = result.shape[:2]
                max_dim = 800
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    show_w, show_h = int(w * scale), int(h * scale)
                    show_img = cv2.resize(result, (show_w, show_h))
                else:
                    show_img = result
                    
                cv2.imshow("Optimized Fusion Result", show_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("The folder path does not exist, please check the configuration.")