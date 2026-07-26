# Reference:
# Haghighat M B A, Aghagolzadeh A, Seyedarabi H. Multi-focus image fusion for visual sensor networks in DCT domain[J]. Computers & Electrical Engineering, 2011, 37(5): 789-797.
import glob
import os
import time
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np

from utils import bitdepth
from utils.image_utils import read_image_any_depth

ArraySource = Sequence[np.ndarray]

# --- Focus-measure tuning -------------------------------------------------
# The paper scores a block by the energy of all its AC coefficients, which is
# the block's plain pixel variance. That measure answers "how much contrast is
# here", not "how sharp is it": a heavily defocused highlight lays a smooth
# brightness ramp across the frame, and a ramp crossing one block carries more
# variance than the fine, low-amplitude texture that is genuinely in focus
# there. The blurred frame then wins whole regions and its pixels are copied
# verbatim, which is what produced the flat homogeneous patches this measure
# replaces. Dropping the lowest AC band - everything coarser than a block -
# leaves only detail a block can actually resolve, so a defocused wash scores
# near zero however bright it is.
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

# How far above its own noise the winner must be for the block to count as
# holding real detail. The measure is a signal-to-noise ratio once normalised,
# so this reads directly: below 2 the best frame is no more than twice its own
# grain. Measured on the fixtures, the winner's SNR sits at 1.1-2.2 across
# genuinely featureless regions and never below 93 in the detailed scenarios,
# so the two populations are far apart and the exact value is not delicate.
_DETAIL_SNR = 2.0

# A winner must also beat the runner-up by this fraction of its own energy.
# Below it the two frames are equally sharp here by any honest reading, so the
# block is left to the neighbourhood rather than a coin toss. This takes exact
# ties off the lowest-index default they used to fall to, but it does not settle
# item 4 in docs/ALGORITHM_IMPROVEMENTS.md.
_DECISION_MARGIN = 0.10


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
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
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


def _fill_undecided(index_map: np.ndarray, undecided: np.ndarray) -> np.ndarray:
    """Give each undecided block the choice of the nearest decided one.

    Where no frame resolves anything there is no focus information to decide on,
    and asking the energies anyway just hands the block to whichever frame has
    the most grain. Taking the nearest confident neighbour's frame instead keeps
    such regions continuous with their surroundings, which is what stops a
    detail-free area from being stamped out in one frame's flat tone.

    The seed-to-label mapping is read back out of the transform's own output -
    a seed is its own nearest seed - so nothing here depends on the order
    OpenCV happens to number labels in.
    """
    if not undecided.any() or undecided.all():
        return index_map

    _, labels = cv2.distanceTransformWithLabels(
        undecided.astype(np.uint8), cv2.DIST_L2, 3,
        labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros(int(labels.max()) + 1, dtype=index_map.dtype)
    decided = ~undecided
    lookup[labels[decided]] = index_map[decided]
    return lookup[labels]


def _median_filter_index_map(index_map: np.ndarray, kernel_size: int) -> np.ndarray:
    """Median-filter a uint16 index map.

    cv2.medianBlur accepts 16-bit input only at apertures 3 and 5, so larger
    kernels are approximated by iterating the 5-aperture filter: each pass
    extends the effective radius by 2, so we run enough passes to cover the
    requested kernel's radius.
    """
    if kernel_size < 3:
        return index_map
    if kernel_size <= 5:
        return cv2.medianBlur(index_map, kernel_size)
    passes = ((kernel_size - 1) // 2 + 1) // 2
    for _ in range(passes):
        index_map = cv2.medianBlur(index_map, 5)
    return index_map

def dct_focus_stack_fusion(
    source: Union[str, ArraySource],
    output_path: str = None,
    block_size: int = 8,
    kernel_size: int = 7,
) -> np.ndarray:
    """
    DCT-domain multi-focus fusion: each block is taken from the frame carrying
    the most detail energy there, and its pixels are copied through verbatim.

    Optimization notes:
    By Parseval's theorem the energy of a block's DCT coefficients equals the
    energy of its pixels, so the measure needs no transform at all: a box
    high-pass (which drops the coefficients below the block's own frequency -
    see the tuning notes at the top of this module) followed by a block mean of
    the squared result, both done with cv2 filters and cv2.resize(INTER_AREA),
    replaces the originally extremely slow per-block DCT loop.
    """
    
    # --- 1. Parameter validation and preparation ---
    if kernel_size % 2 == 0:
        kernel_size += 1
    block_size = max(2, int(block_size))

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

    # Running top two energies per block, and the frame that holds the top one.
    # Only the top two are kept, so memory stays independent of stack depth.
    best_energy = np.full((map_h, map_w), -1.0, dtype=np.float32)
    second_energy = np.full((map_h, map_w), -1.0, dtype=np.float32)
    # uint16 covers any realistic stack depth and, unlike uint8, does not wrap
    # indices at 256 frames; medianBlur (apertures 3/5) and INTER_NEAREST
    # resize both accept it, so the map stays uint16 end to end.
    best_index_map = np.zeros((map_h, map_w), dtype=np.uint16)

    for idx, bgr_img in enumerate(normalized_images):
        # Crop the edges to match the block tiling
        img_trim = bgr_img[:h_trim, :w_trim]

        # Grayscale, normalised to [0, 1] before squaring. Normalising is what
        # makes the energy measure usable at 16 bits: squaring raw 16-bit
        # levels reaches 4.3e9, where float32's 24-bit mantissa has a ULP of
        # ~256, so everything below that would quantise away. Working in [0, 1]
        # gives both depths the same headroom. Fusion decisions are unaffected
        # by the rescaling itself: scaling every frame by the same constant
        # leaves the per-block ranking unchanged.
        gray = bitdepth.to_float01(cv2.cvtColor(img_trim, cv2.COLOR_BGR2GRAY))
        pooled = _normalised_energy(gray, hp_window, pool_window, map_w, map_h)

        # Running top-two update: the previous best is demoted to runner-up when
        # it is beaten, otherwise the new frame competes for the runner-up slot.
        wins = pooled > best_energy
        np.copyto(second_energy, best_energy, where=wins)
        np.copyto(best_energy, pooled, where=wins)
        np.copyto(best_index_map, np.uint16(idx), where=wins)
        np.copyto(second_energy, pooled, where=~wins & (pooled > second_energy))

    # Two ways a block can fail to decide itself: no frame holds real detail
    # there, or two frames hold the same amount. Either way the energies have
    # nothing left to say, so the block takes its neighbourhood's choice rather
    # than a coin toss. An exact tie lands here too (margin of zero), so the
    # winner no longer depends on which frame happened to be passed first.
    undecided = (best_energy < _DETAIL_SNR) | (
        (best_energy - second_energy) <= _DECISION_MARGIN * best_energy)
    best_index_map = _fill_undecided(best_index_map, undecided)

    # --- 3. Consistency verification (median filtering) ---
    # Two passes of median filtering to remove noise
    filtered_map = _median_filter_index_map(best_index_map, kernel_size)
    final_index_map = _median_filter_index_map(filtered_map, kernel_size)

    # --- 4. Fast reconstruction ---
    # Scale the small index map back up to the original size in one go (Nearest Neighbor)
    full_size_indices = cv2.resize(
        final_index_map,
        (w_trim, h_trim),
        interpolation=cv2.INTER_NEAREST
    )

    # The block grid only covers a multiple of block_size, so an image whose
    # dimensions do not divide evenly leaves a strip on the right and bottom.
    # Extend the last row/column of decisions over it rather than returning a
    # smaller image than we were given.
    if (h_trim, w_trim) != (h, w):
        full_size_indices = cv2.copyMakeBorder(
            full_size_indices, 0, h - h_trim, 0, w - w_trim,
            cv2.BORDER_REPLICATE
        )

    # Output pixels are copied verbatim from the source frames, so the result
    # only has to be allocated at the stack's own depth to stay lossless.
    fused_image = np.zeros((h, w, 3), dtype=bitdepth.stack_dtype(normalized_images))

    # Iterate only over the source-image indices that are used, to fill in
    unique_indices = np.unique(final_index_map)
    
    for idx in unique_indices:
        # Generate the mask: True wherever this image is needed
        mask = (full_size_indices == idx)
        
        # Even though this is a Python loop, it operates on whole-image masks, so it is fast
        source_layer = normalized_images[idx]

        # Assign
        fused_image[mask] = source_layer[mask]

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