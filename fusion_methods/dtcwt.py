# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import os
import re
import glob
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

from utils import bitdepth
from utils.image_utils import read_image_any_depth

# NumPy 2.0 compatibility shims
if not hasattr(np, "asfarray"):
    def _asfarray_compat(arr, dtype=None):
        target_dtype = dtype or np.float_
        kind = np.dtype(target_dtype).kind
        if kind not in ("f", "c"):
            target_dtype = np.float_
        return np.asarray(arr, dtype=target_dtype)
    np.asfarray = _asfarray_compat

if not hasattr(np, "issubsctype"):
    def _issubsctype_compat(arg1, arg2):
        try:
            dtype1 = np.dtype(arg1) if not isinstance(arg1, np.dtype) else arg1
        except TypeError:
            dtype1 = np.asarray(arg1).dtype
        dtype2 = np.dtype(arg2) if not isinstance(arg2, np.dtype) else arg2
        return np.issubdtype(dtype1, dtype2)
    np.issubsctype = _issubsctype_compat

# Optional imports with checks
try:
    import dtcwt
except ImportError:
    dtcwt = None

# Added to each frame's per-pixel lowpass weight (its aggregate highpass
# activity) before the weighted average of the lowpass band. Where every frame
# is flat the activities vanish and the epsilon takes over, degrading
# gracefully to the plain mean used before; where any frame carries detail the
# epsilon is negligible against real coefficient magnitudes.
_LOWPASS_ACTIVITY_EPS = 1e-6


def _lowpass_activity(pyramid, lowpass_shape):
    """Aggregate detail activity of one frame on the lowpass grid.

    Sums the complex magnitudes of every highpass level over the six
    orientations, area-resampling each level's map to the lowpass resolution so
    all scales contribute. Used to weight the frame's lowpass band so the
    frames that win the detail bands also dominate the coarse band (item 16 in
    docs/ALGORITHM_IMPROVEMENTS.md).
    """
    acc = np.zeros(lowpass_shape, dtype=np.float32)
    for hp in pyramid.highpasses:
        mag = np.abs(hp).sum(axis=2).astype(np.float32)
        acc += cv2.resize(mag, (lowpass_shape[1], lowpass_shape[0]),
                          interpolation=cv2.INTER_AREA)
    return acc


def _dtcwt_impl(input_source, img_resize, N, use_gpu):
    """
    DTCWT fusion implementation (Optimized CPU version).
    
    Args:
        input_source: List of images or path to image directory.
        img_resize: Tuple (width, height) or None.
        N: Number of wavelet decomposition levels.
        use_gpu: Ignored in this optimized NumPy version (CPU is used).
    """
    if dtcwt is None:
        raise RuntimeError(
            "DTCWT fusion requires 'dtcwt'. "
            "Please install it via: pip install dtcwt"
        )

    # 1. Load and Preprocess Images
    images = _load_images(input_source, img_resize)
    
    if len(images) < 2:
        raise ValueError("At least two images are required for fusion.")

    # The result is written back at the depth the stack arrived in.
    out_dtype = bitdepth.stack_dtype(images)

    # Convert to float32 RGB [0, 1]
    # Processing images as a batch is not easily possible with standard dtcwt library
    # (which expects 2D inputs), so we prepare them for channel-wise processing.
    images_rgb = [
        bitdepth.to_float01(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for img in images
    ]

    # 2. Define Optimized Fusion Function (Vectorized)
    def fuse_highfreq_vectorized(coeffs_list, window_size=3):
        """
        Optimized high-frequency fusion.
        Vectorizes operations over the 6 wavelet directions to avoid Python loops.

        Each entry of coeffs_list is one frame's coefficients for a level with
        the colour channels stacked on a trailing axis, shape (H, W, 6, C).
        A single decision mask - built from the strongest channel's activity -
        selects all C channels together, so a pixel's colour cannot split
        across source frames (item 12 in docs/ALGORITHM_IMPROVEMENTS.md).
        """
        num_imgs = len(coeffs_list)
        if num_imgs == 1:
            return coeffs_list[0]

        # Recursive pairwise fusion if more than 2 images
        if num_imgs > 2:
            fused = coeffs_list[0]
            for i in range(1, num_imgs):
                fused = fuse_highfreq_vectorized([fused, coeffs_list[i]], window_size)
            return fused

        # Pairwise Fusion
        c1, c2 = coeffs_list[0], coeffs_list[1]

        # c1 shape is (H, W, 6, C).
        # We want to filter spatially (H, W) but independently for each direction (6).
        # OpenCV filters work on 2D planes, so the 6 slices are looped.
        dilate_kernel = np.ones((window_size, window_size), dtype=np.uint8)

        # 1. Compute Magnitudes - keep the strongest channel response at each
        # coefficient, so structure present in any channel drives the choice
        # for all of them.
        mag1 = np.abs(c1).max(axis=-1)
        mag2 = np.abs(c2).max(axis=-1)

        # 2. Activity Level Measurement (Max Filter)
        # cv2.dilate is a sliding maximum; its default border ignores
        # out-of-bounds pixels, which for a max filter is bit-identical to
        # scipy's reflect mode (reflection only duplicates in-window values).
        A1 = np.empty_like(mag1)
        A2 = np.empty_like(mag2)
        for d in range(mag1.shape[2]):
            A1[:, :, d] = cv2.dilate(np.ascontiguousarray(mag1[:, :, d]), dilate_kernel)
            A2[:, :, d] = cv2.dilate(np.ascontiguousarray(mag2[:, :, d]), dilate_kernel)

        # 3. Initial Mask Generation
        initial_mask = A1 > A2  # Boolean array (H, W, 6)

        # 4. Consistency Verification (Majority Filter)
        # Unnormalized box filter counts each pixel's agreeing neighbours.
        # BORDER_CONSTANT zero-pads, keeping the border behaviour of the
        # previous mode='constant', cval=0.0 convolution (the border bias of
        # item 14 in docs/ALGORITHM_IMPROVEMENTS.md is preserved deliberately
        # so this swap stays bit-identical).
        mask_f32 = initial_mask.astype(np.float32)
        count_map = np.empty_like(mask_f32)
        for d in range(mask_f32.shape[2]):
            count_map[:, :, d] = cv2.boxFilter(
                np.ascontiguousarray(mask_f32[:, :, d]), -1,
                (window_size, window_size),
                normalize=False, borderType=cv2.BORDER_CONSTANT,
            )

        # Threshold: if more than half the window supports source 1, use source 1
        threshold = (window_size * window_size) / 2.0
        W = count_map > threshold  # Final Boolean Mask (H, W, 6)

        # 5. Final Blending
        # W is boolean: True -> c1, False -> c2; the same mask picks every
        # channel of a coefficient.
        fused = np.where(W[..., None], c1, c2)

        return fused

    # 3. Perform DTCWT and Fusion
    transform = dtcwt.Transform2d()

    # The stack is folded into a running result one frame at a time - the same
    # sequential pairwise order as before (item 7 in
    # docs/ALGORITHM_IMPROVEMENTS.md still applies) - but all three channels of
    # a frame travel through fusion together, so each pairwise step selects
    # them with one shared decision mask instead of three independent ones
    # (item 12). Streaming also bounds memory to about two frames' worth of
    # coefficients regardless of stack depth.
    lowpass_sum = None       # (h', w', 3) activity-weighted lowpass sum
    weight_sum = None        # (h', w', 3) sum of those weights
    fused_highpasses = None  # per level: (H, W, 6, 3) complex

    for img in images_rgb:
        pyramids = [
            transform.forward(img[:, :, ch], nlevels=N) for ch in range(3)
        ]

        # The lowpass band is averaged with each frame weighted by its
        # aggregate highpass activity so the sharpest frames dominate the
        # coarse band too, instead of a plain mean ghosting in frames that
        # disagree at large scale (exposure drift, focus breathing).
        lowpass = np.stack([p.lowpass for p in pyramids], axis=-1)
        weight = np.stack(
            [_lowpass_activity(p, lowpass.shape[:2]) for p in pyramids],
            axis=-1) + _LOWPASS_ACTIVITY_EPS
        highpasses = [
            np.stack([p.highpasses[level] for p in pyramids], axis=-1)
            for level in range(N)
        ]

        if lowpass_sum is None:
            lowpass_sum = lowpass * weight
            weight_sum = weight
            fused_highpasses = highpasses
        else:
            lowpass_sum += lowpass * weight
            weight_sum += weight
            fused_highpasses = [
                fuse_highfreq_vectorized([fused, new])
                for fused, new in zip(fused_highpasses, highpasses)
            ]

    fused_lowpass = lowpass_sum / weight_sum

    # 4. Reconstruct Final Image
    # The inverse transform runs per channel; the coupling above only decides
    # which frame each coefficient comes from.
    fused_channels = [
        transform.inverse(dtcwt.Pyramid(
            fused_lowpass[..., ch],
            tuple(level[..., ch] for level in fused_highpasses)))
        for ch in range(3)
    ]
    fused_img = np.stack(fused_channels, axis=-1)
    
    # Clip and convert back to the stack's own depth
    fused_img = bitdepth.from_float01(fused_img, out_dtype)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)

    # dtcwt duplicates the bottom row and rightmost column of an odd-sized image
    # before decomposing, so the reconstruction comes back one pixel larger.
    # Drop the padding again to return the geometry we were handed.
    src_h, src_w = images[0].shape[:2]
    if fused_img.shape[:2] != (src_h, src_w):
        fused_img = fused_img[:src_h, :src_w]

    return fused_img


def _load_images(input_source: Union[str, List[np.ndarray]], img_resize: Optional[Tuple[int, int]]) -> List[np.ndarray]:
    """Helper to load images from a path or list."""
    images = []
    if isinstance(input_source, str):
        if not os.path.exists(input_source):
             raise ValueError(f"Input path does not exist: {input_source}")
             
        filenames = os.listdir(input_source)
        if not filenames:
            return []
            
        # Infer extension from first file
        suffixes = [os.path.splitext(f)[1] for f in filenames if os.path.splitext(f)[1]]
        img_ext = suffixes[0] if suffixes else ''
        
        glob_pattern = os.path.join(input_source, '*' + img_ext)
        img_paths = glob.glob(glob_pattern)
        
        # Sort numerically if possible
        def sort_key(x):
            nums = re.findall(r"\d+", os.path.basename(x))
            return int(nums[-1]) if nums else x
            
        img_paths.sort(key=sort_key)

        for p in img_paths:
            img = read_image_any_depth(p)
            if img is not None:
                if img_resize:
                    img = cv2.resize(img, img_resize)
                images.append(img)
    else:
        for img in input_source:
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)
            
    return images