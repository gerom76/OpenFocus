import os
import glob
import re
import cv2
import numpy as np
import concurrent.futures
from collections import deque

from utils import bitdepth
from utils.image_utils import read_image_any_depth

# Reference:
# https://github.com/RCharradi/Image-fusion-with-guided-filtering
# Li S, Kang X, Hu J. Image fusion with guided filtering[J]. IEEE Transactions on Image processing, 2013, 22(7): 2864-2875.

# Base-layer weight-refinement radius, expressed as a multiple of the averaging
# window's radius. The paper fixes r1=45 alongside a 31x31 averaging window
# (radius 15), i.e. a 3:1 ratio; kernel_size=31 therefore still reproduces the
# published r1=45 exactly. Deriving r1 rather than pinning it keeps the weight
# smoothing matched to the decomposition scale at every slider position, which
# removes the small-kernel degradation (42.7 dB at kernel 7 against 43.4 at the
# default). It does not make the slider matter more - see item 9 of
# docs/ALGORITHM_IMPROVEMENTS.md for why this method is near-invariant to it.
R1_TO_BLUR_RADIUS = 3


def base_weight_radius(average_filter_size):
    """Guided-filter radius used to refine the base-layer weights."""
    return max(1, R1_TO_BLUR_RADIUS * (average_filter_size // 2))


def _map_in_order(fn, count, max_workers):
    """Run fn(0..count-1) on a thread pool, yielding (index, result) in index order.

    A sliding window keeps exactly max_workers tasks outstanding: peak memory is
    then set by the pool size rather than by the stack depth, so per-frame
    float32 buffers are freed as they are consumed instead of piling up for the
    whole stack. Yielding in index order fixes the accumulation order, so repeat
    runs are bit-identical. Submitting the replacement before yielding keeps the
    pool fed while the caller consumes, which a batch-at-a-time loop would stall.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = deque()
        queued = 0
        while queued < min(max_workers, count):
            pending.append(executor.submit(fn, queued))
            queued += 1

        for k in range(count):
            result = pending.popleft().result()
            if queued < count:
                pending.append(executor.submit(fn, queued))
                queued += 1
            yield k, result


def gff_impl(input_source, img_resize, kernel_size=31, thread_count: int = None):
    """
    Guided-filter-based multi-focus image-stack fusion algorithm implementation (CPU version)
    Does not depend on opencv-contrib (ximgproc); includes a built-in guided-filter implementation.
    """
    # Determine thread pool size. Capped at 8 by default: the OpenCV filters are
    # already internally parallel, and the cap bounds how many frames are held in
    # float32 at once.
    if thread_count is None:
        max_workers = min(8, os.cpu_count() or 4)
    else:
        try:
            max_workers = max(1, int(thread_count))
        except Exception:
            max_workers = min(8, os.cpu_count() or 4)

    # ========== Parameter settings ==========
    # Default parameters (referencing the original script)
    DEFAULT_R2 = 7
    DEFAULT_EPS1 = 0.3
    DEFAULT_EPS2 = 10e-6
    DEFAULT_SIGMA_R = 5

    # If a valid kernel_size is passed in, use it as the mean-filter size
    average_filter_size = kernel_size if kernel_size is not None and kernel_size > 0 else 31

    # Ensure the filter size is odd
    if average_filter_size % 2 == 0:
        average_filter_size += 1

    r1 = base_weight_radius(average_filter_size)

    # ========== Data loading ==========
    if isinstance(input_source, str):
        def get_image_suffix(input_stack_path):
            filenames = os.listdir(input_stack_path)
            if len(filenames) == 0:
                return None
            suffixes = [os.path.splitext(filename)[1] for filename in filenames]
            return suffixes[0]

        img_ext = get_image_suffix(input_source)
        if img_ext is None:
            raise ValueError("Input folder is empty or no images were found")

        glob_format = '*' + img_ext
        img_stack_path_list = glob.glob(os.path.join(input_source, glob_format))
        # Try to sort numerically
        try:
            img_stack_path_list.sort(key=lambda x: int(str(re.findall(r"\d+", x.split(os.sep)[-1])[-1])))
        except IndexError:
            img_stack_path_list.sort() # fall back to lexicographic order

        stack_ori = [read_image_any_depth(img_path) for img_path in img_stack_path_list]
        stack_ori = [img for img in stack_ori if img is not None]
    else:
        stack_ori = input_source

    if not stack_ori:
        raise ValueError("No image data was loaded")

    # The result is written back at the depth the stack arrived in, so a 16-bit
    # stack stays 16-bit end to end.
    out_dtype = bitdepth.stack_dtype(stack_ori)

    num_images = len(stack_ori)
    if img_resize:
        cols, rows = int(img_resize[0]), int(img_resize[1])
    else:
        rows, cols = stack_ori[0].shape[:2]
    channels = stack_ori[0].shape[2]

    # Resize and float32 conversion happen per frame, on demand. Holding the
    # whole stack as float32 BGR costs 12 bytes per pixel per frame - 2.9 GB for
    # ten 24-megapixel frames before any working buffers - and the two passes
    # below only ever need a handful of frames at a time.
    def load_float(k):
        img = stack_ori[k]
        if img_resize and (img.shape[1], img.shape[0]) != (cols, rows):
            img = cv2.resize(img, (cols, rows))
        # Normalised against the frame's own full scale, so 8- and 16-bit input
        # both land in [0, 1] and the algorithm below is depth-agnostic.
        return bitdepth.to_float01(img)

    # ========== Core algorithm implementation ==========

    # 1. Initial decision map: at each pixel, the index of the image with the
    #    highest saliency, where Saliency = Gaussian(abs(Laplacian(Sum_Channels)))
    #    Reduced with a running maximum so the N saliency maps are never all
    #    resident at once.

    def process_saliency(k):
        img = load_float(k)
        # Sum the three channels for the Laplacian computation
        img_sum = np.sum(img, axis=2)
        lap = np.abs(cv2.Laplacian(img_sum, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT))
        return cv2.GaussianBlur(lap, (0, 0), sigmaX=DEFAULT_SIGMA_R, sigmaY=DEFAULT_SIGMA_R,
                                borderType=cv2.BORDER_REFLECT)

    best_saliency = np.full((rows, cols), -np.inf, dtype=np.float32)
    max_indices = np.zeros((rows, cols), dtype=np.int32)

    # Strict '>' keeps the earliest index on ties, matching np.argmax over the
    # stacked saliency maps.
    for k, sal in _map_in_order(process_saliency, num_images, max_workers):
        better = sal > best_saliency
        np.copyto(best_saliency, sal, where=better)
        np.copyto(max_indices, k, where=better)

    del best_saliency

    # 2. Weight-map refinement and fusion (Weight Refinement & Fusion)
    # Initialize the accumulators
    fused_base_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
    fused_base_denominator = np.zeros((rows, cols), dtype=np.float32) # optimized to single channel

    fused_detail_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
    fused_detail_denominator = np.zeros((rows, cols), dtype=np.float32) # optimized to single channel

    def process_weight_fusion(k):
        img_k = load_float(k)

        # Image decomposition (Base Layer & Detail Layer)
        # cv2.blur rather than scipy's uniform_filter, for speed
        base_k = cv2.blur(img_k, (average_filter_size, average_filter_size),
                          borderType=cv2.BORDER_REFLECT)
        detail_k = img_k - base_k

        # Generate the binary mask for the k-th image
        mask_k = (max_indices == k).astype(np.float32) # (H, W)

        # Refine the mask using the built-in guided filter
        img_k_gray = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)

        # Precompute I*p and I*I to avoid recomputing them across the two guided_filter calls
        I = img_k_gray
        p = mask_k
        Ip = I * p
        I2 = I * I

        # Guided filter (He et al.), reusing the precomputed products
        def run_gf(r, eps):
            ksize = (2 * r + 1, 2 * r + 1)
            mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
            mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
            mean_Ip = cv2.boxFilter(Ip, cv2.CV_32F, ksize)
            cov_Ip = mean_Ip - mean_I * mean_p

            mean_II = cv2.boxFilter(I2, cv2.CV_32F, ksize)
            var_I = mean_II - mean_I * mean_I

            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
            mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

            q = mean_a * I + mean_b
            return q

        # For the Base layer
        weight_base_k = run_gf(r1, DEFAULT_EPS1)

        # For the Detail layer
        weight_detail_k = run_gf(DEFAULT_R2, DEFAULT_EPS2)

        # Optimization: no longer use np.repeat to expand to 3 channels, use broadcasting instead
        # weight_base_k and weight_detail_k are both (H, W)

        # Compute the numerator term (H, W, 3) * (H, W, 1) -> (H, W, 3)
        base_num = base_k * weight_base_k[:, :, np.newaxis]
        detail_num = detail_k * weight_detail_k[:, :, np.newaxis]

        # The denominator term can just return the single-channel weight
        return base_num, weight_base_k, detail_num, weight_detail_k

    # Accumulate in index order so the float summation is reproducible run to run
    for _, (bn, bd, dn, dd) in _map_in_order(process_weight_fusion, num_images, max_workers):
        fused_base_numerator += bn
        fused_base_denominator += bd
        fused_detail_numerator += dn
        fused_detail_denominator += dd
        # Explicitly delete references to help garbage collection
        del bn, bd, dn, dd

    # 3. Reconstruct the image
    # Avoid division by zero
    fused_base_denominator[fused_base_denominator < 1e-6] = 1e-6
    fused_detail_denominator[fused_detail_denominator < 1e-6] = 1e-6

    # Broadcast division (H, W, 3) / (H, W, 1)
    fused_base = fused_base_numerator / fused_base_denominator[:, :, np.newaxis]
    fused_detail = fused_detail_numerator / fused_detail_denominator[:, :, np.newaxis]

    fused_img = fused_base + fused_detail

    # Clip and convert back to the stack's own depth
    return bitdepth.from_float01(fused_img, out_dtype)
