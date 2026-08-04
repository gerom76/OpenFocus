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

# Window, in coefficients, that a frame's activity is measured and pooled over
# before the frames are compared. 3x3 is the window the pairwise consistency
# vote used to count over (item 7 in docs/ALGORITHM_IMPROVEMENTS.md).
_WINDOW = 3


def _coefficient_activity(coeffs, window_size=_WINDOW):
    """How much detail one frame carries at each coefficient of one level.

    ``coeffs`` is (H, W, 6, C) complex - one level, the colour channels on the
    trailing axis. Returns two (H, W, 6) float32 maps:

    * ``magnitude`` - the strongest channel's coefficient magnitude. Reducing
      the channels here is what lets one decision serve all of them, so a
      pixel's colour cannot split across source frames (item 12), and
      structure that lives in a single channel still drives the choice at full
      strength rather than being diluted by two flat channels.
    * ``score`` - that magnitude maximum-filtered over the window, which is
      Lewis's activity measure, then pooled over the same window. Frames are
      compared on the pooled measure, so the decision is regional and grain
      cannot flip a lone coefficient; that is what the pairwise consistency
      vote was there to do, and what `pyramid.py` and `dct.py` already do.

    OpenCV filters work on 2D planes, so the six orientations are looped.
    ``cv2.dilate`` is a sliding maximum whose default border ignores
    out-of-bounds pixels - for a maximum that is the same set of values
    reflection gives - and the pooling reflects, so neither map is biased at
    the frame edge.
    """
    magnitude = np.abs(coeffs).max(axis=-1).astype(np.float32)
    score = np.empty_like(magnitude)
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    for d in range(magnitude.shape[2]):
        plane = cv2.dilate(np.ascontiguousarray(magnitude[:, :, d]), kernel)
        score[:, :, d] = cv2.boxFilter(plane, -1, (window_size, window_size),
                                       normalize=True,
                                       borderType=cv2.BORDER_REFLECT_101)
    return score, magnitude


def _keep_stronger(fused, best_score, best_magnitude, coeffs, score, magnitude):
    """Fold one frame's coefficients into the running per-coefficient winner.

    Everything here is one level: ``fused`` and ``coeffs`` are (H, W, 6, C)
    complex, the three maps (H, W, 6) float32. ``fused``, ``best_score`` and
    ``best_magnitude`` are updated in place wherever this frame wins.

    A frame wins a coefficient by carrying more regional activity there, so
    every frame is judged against every other on the same measure rather than
    against a running fusion of the frames before it. Exact ties do occur - a
    strong structure two frames share fills the maximum filter for several
    coefficients around it, leaving their scores equal where their own
    coefficients are not - and are settled on the coefficient's own magnitude
    rather than on which frame arrived first, which is what keeps the answer
    independent of the order the stack is passed in.
    """
    takes = (score > best_score) | ((score == best_score) &
                                    (magnitude > best_magnitude))
    np.copyto(best_score, score, where=takes)
    np.copyto(best_magnitude, magnitude, where=takes)
    np.copyto(fused, coeffs, where=takes[..., None])


def _lowpass_weight(pyramids, lowpass_shape):
    """Aggregate detail activity of one frame on the lowpass grid.

    ``pyramids`` is the frame's three per-channel transforms. Sums the complex
    magnitudes of every highpass level over the six orientations,
    area-resampling each level's map to the lowpass resolution so all scales
    contribute. Used to weight the frame's lowpass band so the frames that win
    the detail bands also dominate the coarse band (item 16 in
    docs/ALGORITHM_IMPROVEMENTS.md).

    The channels are summed into one grey map before the weight is formed, so
    all three channels of a pixel are mixed over the stack in the same
    proportions and no colour can appear in the coarse band that no frame
    carried (item 12). That is the reduction `pyramid.py` and `depthmap.py`
    already use for the same reason; a maximum, which is what
    `_coefficient_activity` uses to *select* a coefficient, was measured here
    too and is 0.06-0.14 dB worse on `deep_stack` and level everywhere else -
    selecting wants the strongest channel, weighing wants all of them. Summing
    also commutes with the area resampling below, so unlike a maximum it cannot
    matter whether the channels are reduced before or after the resize.
    """
    acc = np.zeros(lowpass_shape, dtype=np.float32)
    for level in range(len(pyramids[0].highpasses)):
        mag = sum(np.abs(p.highpasses[level]).sum(axis=2).astype(np.float32)
                  for p in pyramids)
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

    # 2. Perform DTCWT and Fusion
    transform = dtcwt.Transform2d()

    # Every frame is compared against every other on the same measure: each
    # coefficient goes to the frame carrying the most activity around it, taken
    # in one pass over the stack rather than by folding the frames together two
    # at a time (item 7 in docs/ALGORITHM_IMPROVEMENTS.md). All three channels
    # of a frame travel through the decision together, so one shared mask
    # selects them and a pixel's colour cannot split across frames (item 12).
    # The pass streams, so memory stays at about two frames' worth of
    # coefficients regardless of stack depth.
    lowpass_sum = None       # (h', w', 3) activity-weighted lowpass sum
    weight_sum = None        # (h', w') sum of those weights, shared by channels
    fused_highpasses = None  # per level: (H, W, 6, 3) complex - winners so far
    best_score = None        # per level: (H, W, 6) float32 - their activity
    best_magnitude = None    # per level: (H, W, 6) float32 - their magnitude

    for img in images_rgb:
        pyramids = [
            transform.forward(img[:, :, ch], nlevels=N) for ch in range(3)
        ]

        # The lowpass band is averaged with each frame weighted by its
        # aggregate highpass activity so the sharpest frames dominate the
        # coarse band too, instead of a plain mean ghosting in frames that
        # disagree at large scale (exposure drift, focus breathing). One weight
        # per pixel serves all three channels, so the coarse band mixes the
        # stack in the same proportions in each of them (item 12).
        lowpass = np.stack([p.lowpass for p in pyramids], axis=-1)
        weight = (_lowpass_weight(pyramids, lowpass.shape[:2])
                  + _LOWPASS_ACTIVITY_EPS)
        highpasses = [
            np.stack([p.highpasses[level] for p in pyramids], axis=-1)
            for level in range(N)
        ]
        activity = [_coefficient_activity(hp) for hp in highpasses]

        if lowpass_sum is None:
            # float64, because this is the one running quantity whose value
            # depends on the order it is summed in - and a stack handed over
            # backwards should render the identical picture.
            lowpass_sum = (lowpass * weight[..., None]).astype(np.float64)
            weight_sum = weight.astype(np.float64)
            fused_highpasses = highpasses
            best_score = [score for score, _ in activity]
            best_magnitude = [magnitude for _, magnitude in activity]
        else:
            lowpass_sum += lowpass * weight[..., None]
            weight_sum += weight
            for level, (score, magnitude) in enumerate(activity):
                _keep_stronger(fused_highpasses[level], best_score[level],
                               best_magnitude[level], highpasses[level],
                               score, magnitude)

    fused_lowpass = (lowpass_sum / weight_sum[..., None]).astype(np.float32)

    # 3. Reconstruct Final Image
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