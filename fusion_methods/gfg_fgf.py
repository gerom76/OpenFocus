# Reference:
# Fu Hongyu, Gong Yan, Wang Luhan, et al.. Multi-focus microscopy image fusion algorithm[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022-0618022-9.


import os
import glob
import re
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import bitdepth
from utils.image_utils import read_image_any_depth

# -----------------------------------------------------------------------------
# Global helpers and core algorithm implementation
# -----------------------------------------------------------------------------

def _fast_guided_filter_impl(I, p, r, eps, s=4):
    """
    Python implementation of fast guided filtering (He et al. 2015).
    Significantly speeds up coefficient computation via downsampling (s).
    """
    # 1. Downsample
    h, w = I.shape[:2]
    h_sub = int(h / s)
    w_sub = int(w / s)
    
    # Avoid an overly small size
    if h_sub < 1 or w_sub < 1:
        I_sub = I
        p_sub = p
        r_sub = r
    else:
        I_sub = cv2.resize(I, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)
        p_sub = cv2.resize(p, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)
        r_sub = max(1, int(r / s))

    ksize = (2 * r_sub + 1, 2 * r_sub + 1)

    # 2. Compute statistics (Box Filter)
    # Explicitly specify depth with cv2.CV_32F to avoid unnecessary inference
    mean_I = cv2.boxFilter(I_sub, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p_sub, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I_sub * p_sub, cv2.CV_32F, ksize)
    mean_II = cv2.boxFilter(I_sub * I_sub, cv2.CV_32F, ksize)

    # 3. Compute the linear coefficients a, b
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 4. Coefficient smoothing
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    # 5. Upsample back to the original size
    if h_sub < 1 or w_sub < 1:
        q = mean_a * I + mean_b
    else:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
        q = mean_a * I + mean_b

    return q

# Try to get OpenCV's fast implementation, otherwise use the Python optimized version above
try:
    # Check whether the ximgproc module is available
    _cv_guided_filter = cv2.ximgproc.guidedFilter
    def _run_guided_filter(guide, src, radius, eps):
        return _cv_guided_filter(guide, src, radius, eps)
except AttributeError:
    # Use the Python optimized version, with s=4 by default for speedup
    def _run_guided_filter(guide, src, radius, eps):
        return _fast_guided_filter_impl(guide, src, radius, eps, s=4)

def gfgfgf_impl(input_source, img_resize=None, kernel_size=7, thread_count: int = None):
    """
    GFG-FGF optimized implementation.
    """
    # -------------------------------------------------------------------------
    # 1. Image loading and preprocessing
    # -------------------------------------------------------------------------
    if isinstance(input_source, str):
        # Folder-reading logic
        if not os.path.exists(input_source):
             raise ValueError(f"Path not found: {input_source}")
             
        files = os.listdir(input_source)
        # Simple filtering by image extension
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        img_paths = [
            os.path.join(input_source, f) for f in files 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        if not img_paths:
            raise ValueError("No images found in folder")

        # Try to sort numerically
        try:
            img_paths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]))
        except Exception:
            img_paths.sort()

        stack_ori = [read_image_any_depth(p) for p in img_paths]
        stack_ori = [img for img in stack_ori if img is not None]
    else:
        # List input
        stack_ori = input_source

    if not stack_ori:
        raise ValueError("Input stack is empty")

    # Resize handling
    if img_resize is not None:
        # Check whether a resize is needed, to avoid useless work
        if (stack_ori[0].shape[1], stack_ori[0].shape[0]) != img_resize:
            stack_ori = [cv2.resize(img, img_resize) for img in stack_ori]

    # -------------------------------------------------------------------------
    # 2. Data preparation: convert everything to Float32 (0.0 - 1.0)
    #    This significantly reduces subsequent divisions and leverages SIMD optimization
    # -------------------------------------------------------------------------
    num_imgs = len(stack_ori)
    h, w = stack_ori[0].shape[:2]

    # The result is written back at the depth the stack arrived in, so a 16-bit
    # stack stays 16-bit end to end.
    out_dtype = bitdepth.stack_dtype(stack_ori)

    # imgs_f32: (N, H, W, 3) range [0, 1]
    # Store in a list to avoid allocating one huge numpy array and running out of memory
    imgs_f32 = []
    for img in stack_ori:
        if img.dtype.kind == "f":
            # Already float: taken as normalised, unless it is plainly in
            # display units, in which case fall back to the 8-bit scale.
            temp = img.astype(np.float32)
            if temp.max() > 1.0:
                temp /= 255.0
            imgs_f32.append(temp)
        else:
            # uint8 or uint16, each normalised by its own full scale
            imgs_f32.append(bitdepth.to_float01(img))

    # The original integer data is no longer needed; release the reference (if not held externally)
    del stack_ori

    # Guide image for the guided filter: luminance, not a single colour channel.
    # Detail is measured across all channels further down, so a structure that
    # only exists in one channel (e.g. red text on a dark background) is not lost.
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs_f32]

    # -------------------------------------------------------------------------
    # 3. Compute focus measure (Focus Measure) - stage 1
    # -------------------------------------------------------------------------
    # Define the convolution kernel (keep float32)
    kx = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
    ky = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)

    focus_vals = np.zeros(num_imgs, dtype=np.float32)

    def compute_focus_score(idx):
        # Measure gradient energy on every colour channel and keep the strongest
        # response per pixel, so detail carried by a single channel still counts.
        img = imgs_f32[idx]
        # filter2D is well-parallelized internally, but a ThreadPool still helps with many small images
        gx = cv2.filter2D(img, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT)
        gy = cv2.filter2D(img, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT)

        # Ignore a 1-pixel border to avoid boundary artifacts
        energy = (gx * gx + gy * gy)[1:-1, 1:-1]

        # Vectorized computation of the mean of the per-pixel channel maximum
        score = np.mean(energy.max(axis=2))
        return idx, score

    # Decide the thread-pool size: prefer the passed-in thread_count, otherwise use the original strategy (max 8)
    if thread_count is None:
        max_workers = min(8, os.cpu_count() or 4)
    else:
        try:
            max_workers = max(1, int(thread_count))
        except Exception:
            max_workers = min(8, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_focus_score, i) for i in range(num_imgs)]
        for fut in as_completed(futures):
            i, val = fut.result()
            focus_vals[i] = val

    max_focus = np.max(focus_vals) if num_imgs > 0 else 1.0
    if max_focus == 0: max_focus = 1.0 # avoid division by zero

    # -------------------------------------------------------------------------
    # 4. Compute the initial decision maps (AFMs) - stage 2
    # -------------------------------------------------------------------------
    g_msz = kernel_size
    g_gsz = 5
    g_eps = 0.3
    threshold = 0.005

    # A frame is only dropped when it carries no gradient energy at all (a blank
    # or dead frame). A global sharpness quota would throw away frames that are
    # mostly defocused yet hold the only sharp region for part of the field --
    # the per-pixel argmax below is what decides which frame wins where.
    active = focus_vals > 1e-6 * max_focus

    # Preallocate the stack space (N, H, W)
    afms_stack = np.zeros((num_imgs, h, w), dtype=np.float32)

    def compute_afm_map(i):
        # Skip degenerate frames, keeping all zeros
        if not active[i]:
            return i, None

        img = imgs_f32[i]
        # Local averaging
        src_blur = cv2.blur(img, (g_msz, g_msz))
        # Difference
        src_diff = cv2.absdiff(img, src_blur)

        # Collapse the per-channel local contrast by taking the strongest channel
        activity = src_diff.max(axis=2)

        # Thresholding: in-place operation optimization
        # gfg_map = activity if activity > threshold else 0
        _, gfg_map = cv2.threshold(activity, threshold, 0, cv2.THRESH_TOZERO)

        # Guided filtering
        afm = _run_guided_filter(grays[i], gfg_map, g_gsz, g_eps)
        return i, afm

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_afm_map, i) for i in range(num_imgs)]
        for fut in as_completed(futures):
            i, res = fut.result()
            if res is not None:
                afms_stack[i] = res

    # -------------------------------------------------------------------------
    # 5. Decision and fusion (Fusion) - stage 3
    # -------------------------------------------------------------------------
    # Compute the IDM (Argmax)
    # idm_map: (H, W) the index of the image with the maximum response at each pixel
    idm_map = np.argmax(afms_stack, axis=0).astype(np.int32)
    
    # Free the afms_stack memory
    del afms_stack

    # Accumulator
    sum_fdms = np.zeros((h, w), dtype=np.float32)
    imfu_result = np.zeros((h, w, 3), dtype=np.float32)

    # Lock used for accumulation (even though the Python GIL exists, += on a numpy array is not atomic, so mind thread safety)
    # But for performance, we let each thread return its result and accumulate in the main thread, or allocate separate buffers
    # Considering memory, we accumulate serially or in chunks.
    # Since GuidedFilter is time-consuming, we still compute the filter in parallel and accumulate in the main thread.
    
    def compute_fusion_component(i):
        if not active[i]:
            return None
            
        # Generate a binary mask (float)
        # This step is fairly fast
        maxfm_binary = (idm_map == i).astype(np.float32)
        
        # Guided-filter the weights again to smooth them
        # Here maxfm_binary is also 0-1, and the guided-filter output is also 0-1
        fdm = _run_guided_filter(grays[i], maxfm_binary, g_gsz, g_eps)
        
        return i, fdm

    # Launch parallel weight computation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_fusion_component, i) for i in range(num_imgs)]
        
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            
            i, fdm_weight = res
            
            # Accumulate the weights (H, W)
            sum_fdms += fdm_weight
            
            # Accumulate the weighted image (H, W, 3)
            # Use broadcasting: (H,W,3) * (H,W,1) -> (H,W,3)
            # This is one of the main computational costs
            imfu_result += imgs_f32[i] * fdm_weight[:, :, None]

    # -------------------------------------------------------------------------
    # 6. Normalization and output
    # -------------------------------------------------------------------------
    # Avoid division by zero
    # Create a mask, processing only the regions where the weight is non-zero
    nonzero_mask = sum_fdms > 1e-6
    
    # In-place normalization
    # Only process the nonzero regions, keeping the others at 0
    # Since imfu_result is already float32, operate on it directly
    
    # Expand the dimensions of sum_fdms for broadcasting
    sum_fdms_expanded = sum_fdms[:, :, None]
    
    # Use np.divide's where parameter or boolean indexing
    # Boolean indexing may create temporary copies for large arrays, but is faster than iterating
    # For memory efficiency, process channel by channel
    
    out = np.zeros((h, w, 3), dtype=out_dtype)
    full_scale = bitdepth.max_value(out_dtype)

    for c in range(3):
        # Extract the channel
        ch_data = imfu_result[:, :, c]
        # Division
        # Use np.divide's out parameter
        np.divide(ch_data, sum_fdms, out=ch_data, where=nonzero_mask)
        # Clip and convert to the stack's own depth
        np.clip(ch_data * full_scale, 0, full_scale, out=ch_data)
        out[:, :, c] = np.rint(ch_data).astype(out_dtype)

    return out

# -----------------------------------------------------------------------------
# Test entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        t0 = time.time()
        out = gfgfgf_impl(folder, None)
        t1 = time.time()
        print(f"Fusion done in {t1 - t0:.4f}s")
        cv2.imwrite('gfg_fused_opt.png', out)
    else:
        # Generate larger test data
        print("Running synthesis test...")
        h, w = 1024, 1024
        img1 = np.zeros((h, w, 3), dtype=np.uint8)
        img2 = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw the pattern
        cv2.circle(img1, (300, 300), 100, (255, 255, 255), -1)
        cv2.rectangle(img2, (600, 600), (900, 900), (0, 255, 0), -1)
        
        # Blur to simulate defocus
        blur1 = cv2.GaussianBlur(img1, (51, 51), 0)
        blur2 = cv2.GaussianBlur(img2, (51, 51), 0)
        
        # Input sources: left sharp / right blurred vs left blurred / right sharp
        src1 = img1.copy()
        src1[:, 512:] = blur1[:, 512:]
        
        src2 = img2.copy()
        src2[:, :512] = blur2[:, :512]
        
        input_list = [src1, src2]
        
        t0 = time.time()
        fused = gfgfgf_impl(input_list, None)
        t1 = time.time()
        
        print(f"Optimization Fusion Time: {t1 - t0:.4f}s for size {w}x{h}")
        # cv2.imshow("Fused", fused)
        # cv2.waitKey(0)
        cv2.imwrite("test_fused.png", fused)