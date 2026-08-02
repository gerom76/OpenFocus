"""
Unified interface for image-sequence registration

Provides a unified calling interface for four registration methods:
1. Scale (focus-breathing / magnification correction, similarity transform)
2. Homography (homography alignment)
3. ECC (ECC alignment)
4. Both (combined registration: Homography + ECC)

All algorithm implementations are contained in this script, with no external dependencies

# Scale / focus-breathing correction only
python Registration.py --mode scale

# Homography alignment only
python Registration.py --mode homography

# ECC registration only
python Registration.py --mode ecc

# Combined registration (homography + ecc)
python Registration.py --mode both

"""


import argparse
import re
import time
import cv2
import numpy as np
import os
import glob
import sys
from typing import Union, List, Optional

from utils import bitdepth
from utils.image_utils import read_image_any_depth

# Set standard output encoding to utf-8 when a console stream exists.
#
# Reconfigure in place rather than assigning a fresh TextIOWrapper around
# sys.stdout.buffer. A replacement wrapper owns the buffer it was handed, so
# when the wrapper is garbage collected it closes that buffer - which, if
# anything else is holding the original stream (pytest's output capture does
# exactly this), leaves it closed underneath its owner and every later write
# fails with "I/O operation on closed file".
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        # Already detached, or a stream that cannot be reconfigured - the
        # encoding is cosmetic, so carry on rather than failing the import.
        pass


def resolve_reference_index(reference_mode, num_images: int) -> int:
    """Map a reference-frame mode onto a concrete 0-based frame index.

    'middle' resolves to the centre frame (num_images // 2), 'last' to the final
    frame; anything else - including the default 'first' - resolves to frame 0,
    preserving the historical behaviour. Kept next to the registration code so
    the UI, the render worker and the batch worker all agree on the mapping.
    """
    if not num_images or num_images <= 0:
        return 0
    if reference_mode == "middle":
        return num_images // 2
    if reference_mode == "last":
        return num_images - 1
    return 0


# ========== Warping ==========

# Every stage resamples with the same kernel, on every machine. Lanczos4 is the
# widest kernel cv2 offers for a perspective warp and the one a focus stack
# wants: each frame is resampled before the focus measure ever reads it, so a
# kernel that loses high-frequency detail loses exactly the signal the fusion
# stage exists to find. Bilinear keeps 72% of what Lanczos4 does on real frames,
# and the fusion then reads that loss as an absence of focus.
WARP_INTERPOLATION = cv2.INTER_LANCZOS4


def _warp_frame(img, H_final, target_w, target_h):
    """Warp one frame into the shared canvas.

    H_final is the forward (source -> destination) map, which is the convention
    cv2.warpPerspective takes: it samples through the inverse itself.

    There used to be a second, CuPy implementation of this that
    map_coordinates sampled at order=1, taken on the assumption that a GPU warp
    was worth the interpolation it gave up. It was not - measured on a 2560x1430
    frame, bilinear on the GPU is 0.007s against 0.012s for Lanczos4 on the CPU,
    while the spline order that matches Lanczos4's sharpness costs 0.016-0.025s.
    The GPU path was only ever faster because it was doing less, and it was the
    source of every CPU-versus-GPU difference in this codebase.
    """
    return cv2.warpPerspective(img, H_final, (target_w, target_h),
                               flags=WARP_INTERPOLATION,
                               borderMode=cv2.BORDER_CONSTANT)


# ========== Detection resolution ==========

# Width the frames are downsampled to before a transform is measured on them.
# Only the *measurement* happens at this size - the transform is rescaled and
# applied to the full-resolution frame - so this is a straight trade of
# detection accuracy against time, which is why it is exposed as a setting.
#
# The three stages used to assign this value themselves: any frame whose longer
# side reached 2048 px had the requested width replaced by 1024, unconditionally
# rather than as a cap. Since that fires on every frame from any camera made
# this century, the setting was inert on real input - raising it to 2048 for a
# difficult stack and lowering it to 512 for speed both produced byte-identical
# output. The override is gone; the value the caller passes is the value used.
# 1024 is what the app, its config file and ImageRegistration have always
# defaulted to, so a user who has not touched the setting sees no change.
DEFAULT_DETECTION_WIDTH = 1024


def _resolve_detection_width(downscale_width, h_orig, w_orig, tag):
    """Detection width for one stage, as asked for rather than as overridden.

    Args:
        downscale_width: the value the caller passed, possibly None or invalid
        h_orig, w_orig: frame size, for the log line only
        tag: stage name for the log line

    Returns:
        int width to downsample to before detecting features.
    """
    try:
        width = int(downscale_width)
    except (TypeError, ValueError):
        width = 0
    if width <= 0:
        width = DEFAULT_DETECTION_WIDTH
    print(f"[Registration][{tag}] Frame {w_orig}x{h_orig}, detecting at {width} px")
    return width


def _detection_scale(width, downscale_width):
    """Factor a frame of this width is resampled by for detection (<= 1.0)."""
    return downscale_width / float(width) if width > downscale_width else 1.0


# ========== Stack loading ==========

# The stages have never agreed on which extensions they accept - ECC's list is
# missing '.tiff', so a folder of .tiff frames aligns under scale and
# homography and silently loses every frame under ECC. That disagreement is
# item 12 of docs/REGISTRATION_IMPROVEMENTS.md and is deliberately left
# standing here: the loader itself is now written once, but the set it is
# handed is still the set the stage in front of the pipeline used, so nothing
# about the behaviour changes until item 12 is taken on its own terms.
_FEATURE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
_ECC_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}

_STAGE_EXTENSIONS = {
    'scale': _FEATURE_EXTENSIONS,
    'homography': _FEATURE_EXTENSIONS,
    'ecc': _ECC_EXTENSIONS,
}


def _load_stack(input_source, img_filenames, valid_exts):
    """Resolve a directory path or a preloaded list into (images, filenames).

    A caller that passes images (which every in-app caller does) gets them back
    untouched; only the CLI and library entry points reach the directory walk.
    """
    if img_filenames is not None or not isinstance(input_source, str):
        return input_source, img_filenames

    num_pattern = re.compile(r"\d+")
    img_paths = sorted(
        (os.path.join(input_source, f) for f in os.listdir(input_source)
         if os.path.splitext(f)[1].lower() in valid_exts),
        key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1]) if num_pattern.findall(os.path.basename(x)) else x
    )
    images = [read_image_any_depth(path) for path in img_paths]
    images = [img for img in images if img is not None]
    return images, [os.path.basename(path) for path in img_paths]


# ========== Scale / focus-breathing correction (similarity) ==========

def _estimate_scale(images, downscale_width, thread_count, parallel_ecc=True):
    """Estimate the focus-breathing similarity of every frame.

    See _align_scale_impl for what the model is and why it is constrained.

    Returns:
        list of 3x3 forward maps placing frame i onto the frame-0 canvas.
    """
    import concurrent.futures

    num_images = len(images)

    # --- 1. Initialization ---
    h_orig, w_orig = images[0].shape[:2]
    downscale_width = _resolve_detection_width(downscale_width, h_orig, w_orig, "Scale")

    # Global accumulated matrix maps the current frame back to frame 0.
    H_global = np.eye(3, dtype=np.float32)
    # Collect all forward transforms (frame -> frame 0) for precise cropping.
    H_matrices = [np.eye(3, dtype=np.float32)]  # frame 0 is the reference

    print(f"Aligning {num_images} images using Scale / focus-breathing correction (similarity)...")

    # --- 2. Parallel feature extraction (SIFT on 8-bit downscaled frames) ---
    def get_features_task(img):
        # An independent detector per thread keeps this thread-safe.
        local_detector = cv2.SIFT_create()
        h, w = img.shape[:2]
        scale = _detection_scale(w, downscale_width)
        if scale < 1.0:
            img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            img_small = img
        # SIFT only accepts 8-bit input; this only measures a transform that is
        # then applied to the full-depth frame, so narrowing costs nothing here.
        kps, des = local_detector.detectAndCompute(bitdepth.to_analysis8(img_small), None)
        return kps, des, scale

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        features_list = list(executor.map(get_features_task, images))

    # --- 3. Sequential similarity estimation (chain must stay serial) ---
    bf = cv2.BFMatcher(cv2.NORM_L2)
    last_kps, last_des, last_scale = features_list[0]

    for idx in range(1, num_images):
        curr_kps, curr_des, curr_scale = features_list[idx]

        # Not enough features to fit anything - keep the running trajectory.
        if curr_des is None or len(curr_kps) < 4 or last_des is None:
            print(f"Warning: Frame {idx} features insufficient. Keeping original position.")
            H_matrices.append(H_global.copy())
            continue

        matches = bf.knnMatch(curr_des, last_des, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.70 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 6:
            print(f"Warning: Frame {idx} poor matches ({len(good_matches)}). Keeping previous trajectory.")
            H_matrices.append(H_global.copy())
            continue

        # Bring the matched points back to full resolution before fitting.
        pts_curr = np.float32([curr_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / curr_scale
        pts_last = np.float32([last_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / last_scale

        # Partial affine == similarity: uniform scale + rotation + translation,
        # no shear or perspective. This is the focus-breathing model.
        M_local, _mask = cv2.estimateAffinePartial2D(
            pts_curr, pts_last, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if M_local is None:
            print(f"Frame {idx} scale alignment failed. Assuming no breathing.")
            H_local = np.eye(3, dtype=np.float32)
        else:
            # Lift the 2x3 similarity to a 3x3 homogeneous matrix for chaining.
            H_local = np.vstack([M_local, [0.0, 0.0, 1.0]]).astype(np.float32)

        # Chain: current -> previous -> ... -> frame 0.
        H_global = np.matmul(H_global, H_local)
        H_matrices.append(H_global.copy())

        last_kps = curr_kps
        last_des = curr_des
        last_scale = curr_scale

    return H_matrices


def _align_scale_impl(input_source, output_path=None, img_filenames=None,
                      downscale_width=DEFAULT_DETECTION_WIDTH, thread_count: int = 4,
                      reference_index: int = 0):
    """
    Focus-breathing (magnification) correction via a similarity transform.

    Focus stacking changes the optical magnification slightly from frame to frame
    as the focus plane moves - the effect photographers call "focus breathing".
    Geometrically it is a similarity: a uniform scale about the optical axis plus
    a small rotation and recentring translation (4 DOF). Fitting a full 8-DOF
    homography to it - as the other methods here do - overfits on frames that
    differ in blur, so this stage estimates a constrained similarity instead.

    For each consecutive pair the transform is recovered from SIFT matches with
    cv2.estimateAffinePartial2D under RANSAC, which yields exactly a scaled
    rotation plus translation and no shear or perspective. The pairwise transforms
    are chained back to the first frame; every frame is then warped into that
    shared frame and cropped to the common valid region, matching the homography
    and ECC methods so the stages compose cleanly.

    Args:
        input_source: directory path or a preloaded list of images
        output_path: optional directory to save results (None = return only)
        img_filenames: optional filenames matching a preloaded image list
        downscale_width: width the frames are downsampled to for feature detection
        thread_count: worker threads for feature extraction and warping
        reference_index: frame held fixed (0 = first frame, the default)

    Returns:
        list of aligned images
    """
    return align_stack(input_source, ('scale',), output_path=output_path,
                       img_filenames=img_filenames, downscale_width=downscale_width,
                       thread_count=thread_count, reference_index=reference_index)



# ========== Precise cropping function based on transform matrices ==========

def _compute_valid_region_from_transforms(H_matrices, img_shape, margin=2):
    """
    Compute the common valid region of all images via the transform matrices
    
    Args:
        H_matrices: list of transform matrices (3x3)
        img_shape: image size (h, w)
        margin: safety margin, used to handle edge inaccuracy caused by perspective distortion
    
    Returns:
        (top, bottom, left, right): bounds of the common valid region
    """
    h, w = img_shape
    
    # The four corner points of the image
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ], dtype=np.float32).T  # 3x4 matrix
    
    # Initialize the valid region as the whole image
    top, bottom, left, right = 0.0, float(h), 0.0, float(w)
    
    for H in H_matrices:
        if H is None:
            continue
        
        # Transform the corner points
        transformed = H @ corners  # 3x4
        # Homogeneous-coordinate conversion
        w_coords = transformed[2:3, :]
        # Prevent division by zero
        w_coords = np.where(np.abs(w_coords) < 1e-10, 1e-10, w_coords)
        transformed = transformed[:2, :] / w_coords  # 2x4
        
        x_coords = transformed[0, :]
        y_coords = transformed[1, :]
        
        # For perspective transforms, we need to consider each edge of the quadrilateral
        # Top edge: connect the top-left and top-right corners, find the max y value (the region to crop)
        # Bottom edge: connect the bottom-left and bottom-right corners, find the min y value
        # Left edge: connect the top-left and bottom-left corners, find the max x value
        # Right edge: connect the top-right and bottom-right corners, find the min x value
        
        # Corner indices: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
        # Valid y for the top edge: max(y_top-left, y_top-right)
        valid_top_edge = max(y_coords[0], y_coords[1])
        # Valid y for the bottom edge: min(y_bottom-left, y_bottom-right)
        valid_bottom_edge = min(y_coords[2], y_coords[3])
        # Valid x for the left edge: max(x_top-left, x_bottom-left)
        valid_left_edge = max(x_coords[0], x_coords[3])
        # Valid x for the right edge: min(x_top-right, x_bottom-right)
        valid_right_edge = min(x_coords[1], x_coords[2])
        
        # Intersect with the canvas bounds
        valid_left_edge = max(0, valid_left_edge)
        valid_right_edge = min(w, valid_right_edge)
        valid_top_edge = max(0, valid_top_edge)
        valid_bottom_edge = min(h, valid_bottom_edge)
        
        # Update the common valid region (intersection of all images)
        left = max(left, valid_left_edge)
        right = min(right, valid_right_edge)
        top = max(top, valid_top_edge)
        bottom = min(bottom, valid_bottom_edge)
    
    # Add the safety margin
    top += margin
    bottom -= margin
    left += margin
    right -= margin
    
    # Convert to integers, rounding inward to stay safe
    top = int(np.ceil(top))
    bottom = int(np.floor(bottom))
    left = int(np.ceil(left))
    right = int(np.floor(right))
    
    return top, bottom, left, right


def _rereference_transforms(H_matrices, reference_index):
    """Re-express every frame's transform relative to a chosen reference frame.

    The pairwise chain is always accumulated against frame 0, so H_matrices[i]
    is the forward map placing frame i onto the frame-0 canvas (and
    H_matrices[0] is the identity). Post-composing every map with the inverse
    of the reference frame's map re-expresses the whole stack on the frame-ref
    canvas: the reference frame collapses to identity - it stays fixed, only
    cropped - and all other frames align onto it.

    Choosing the middle frame as the reference halves the maximum chain length,
    so accumulated drift is spread symmetrically across the stack instead of
    piling up at the far end. reference_index 0 (or out of range) returns the
    list unchanged, preserving the original frame-0 behaviour.

    Args:
        H_matrices: list of 3x3 forward transforms (frame i -> frame 0)
        reference_index: index of the frame to hold fixed

    Returns:
        list of transforms re-referenced to reference_index
    """
    if reference_index is None or reference_index <= 0 or reference_index >= len(H_matrices):
        return H_matrices

    ref = H_matrices[reference_index]
    if ref is None:
        return H_matrices

    try:
        ref_inv = np.linalg.inv(ref)
    except np.linalg.LinAlgError:
        # A singular reference transform cannot be inverted; fall back to
        # frame 0 rather than corrupting the whole stack.
        print(f"Warning: reference frame {reference_index} transform is singular. "
              f"Falling back to frame 0.")
        return H_matrices

    return [ref_inv @ H if H is not None else None for H in H_matrices]


# --- Reference-frame resampling (docs/REGISTRATION_IMPROVEMENTS.md item 3) ---
#
# After _rereference_transforms the reference frame's transform is exactly the
# identity, and the crop translation folded in behind it is an integer offset.
# A warp by an integer translation is a copy - every interpolation weight
# collapses to 1 - so the reference frame alone left registration unresampled
# while every other frame was softened once. Selection-based fusion decides
# everything by asking which frame is locally sharpest, and it read that
# artificial advantage as focus: on a defocused strip of handheld_drift the
# anchor frame measured 25% sharper than the rest on the Lanczos warp path and
# 187% sharper on the bilinear CuPy one this file used to carry, and up to two
# thirds of the output was sourced from it on a scene where it is genuinely
# sharpest nowhere. (That second path is gone as of item 4; the two
# interpolators are kept in the reasoning below because the offset was chosen to
# hold for both, which is what makes it a property of the phase and not of the
# kernel.)
#
# Folding a common sub-pixel offset into the crop translation puts the
# reference through the same interpolator as everything else. The offset is
# shared by every frame, so relative alignment - the thing registration is
# actually measured on - is untouched, and it costs no extra warp: the same
# single warpPerspective per frame simply receives a matrix whose translation
# is no longer integral.
#
# A quarter pixel, not a half. Interpolation loss is worst at the half-pixel
# phase and zero on the grid, so half a pixel does not equalise the anchor, it
# over-corrects: measured on a stack whose frames carry identical content, half
# a pixel leaves the anchor 22% softer than the frames it anchors on the Lanczos
# path and 72% softer on the bilinear one, which merely inverts the bias the
# fix exists to remove. A quarter pixel is where the anchor's sharpness matches
# the stack mean to within a few percent on both interpolators and wherever the
# anchor sits - close to the 0.211 phase at which bilinear attenuation equals
# its average over a uniformly distributed sub-pixel offset, which is the
# treatment an arbitrary frame's residual translation actually gets.
_REFERENCE_RESAMPLE_OFFSET = 0.25


def _build_crop_transform(offset_x, offset_y):
    """Crop translation, shifted so that no frame escapes being resampled.

    Args:
        offset_x, offset_y: integer crop offsets (-left, -top), or 0 when the
            valid region is degenerate and the crop is skipped.

    Returns:
        3x3 forward translation to post-multiply onto each frame's transform.
    """
    return np.array([
        [1, 0, offset_x + _REFERENCE_RESAMPLE_OFFSET],
        [0, 1, offset_y + _REFERENCE_RESAMPLE_OFFSET],
        [0, 0, 1],
    ], dtype=np.float32)


def _crop_with_transforms(images, H_matrices):
    """
    Precisely crop the image based on the transform matrices
    
    Args:
        images: list of images
        H_matrices: list of transform matrices
    
    Returns:
        list of cropped images
    """
    if not images or len(images) == 0:
        return images
    
    h, w = images[0].shape[:2]
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h, w))
    
    # Ensure the crop region is valid
    if top >= bottom or left >= right:
        print("Warning: Invalid crop region, returning original images.")
        return images
    
    # Crop all images
    cropped_images = []
    for img in images:
        cropped = img[top:bottom, left:right].copy()
        cropped_images.append(cropped)
    
    print(f"Cropped by transform matrices: ({left}, {top}) to ({right}, {bottom}), new size: {right-left}x{bottom-top}")
    
    return cropped_images


# ========== Homography alignment algorithm implementation (non-linear) ==========

# --- Per-pair model selection (docs/REGISTRATION_IMPROVEMENTS.md item 2) -----
#
# An 8-DOF homography fitted to SIFT matches spends its two perspective terms on
# whatever is left in the residual, and on a focus stack that is match noise: a
# rail moves the focal plane along the optical axis, so the frame-to-frame motion
# is a magnification change plus small rotation and recentring - a 4-DOF
# similarity. Genuine perspective needs the camera to tilt between frames.
#
# Which model a pair deserves is decided per pair, by held-out reprojection
# error: fit both on part of the matches, score both on the rest, and keep the
# homography only when its extra freedom predicts matches it has not seen. Noise
# does not survive that; a real tilt does.

_CV_SPLITS = 9             # train/test splits averaged per pair
_CV_TRAIN_FRACTION = 0.7   # share of matches used to fit within a split
_CV_MARGIN = 0.8           # the homography must beat the similarity by 20%
_CV_MIN_MATCHES = 24       # three inliers per degree of freedom, or 4 DOF wins


def _project_points(M, pts):
    """Apply a 3x3 projective matrix to an (N, 2) array of points."""
    q = np.asarray(M, dtype=np.float64) @ np.vstack([pts.T, np.ones(len(pts))])
    w = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    return (q[:2] / w).T


def _fit_similarity_ls(src, dst):
    """Least-squares similarity (uniform scale + rotation + translation).

    Written out rather than taken from estimateAffinePartial2D because the
    cross-validation below has to be deterministic, and OpenCV's estimator is
    RANSAC-driven. Solves x' = a*x - b*y + tx, y' = b*x + a*y + ty.
    """
    n = len(src)
    if n < 2:
        return None
    A = np.zeros((2 * n, 4), dtype=np.float64)
    A[0::2, 0] = src[:, 0]
    A[0::2, 1] = -src[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 0] = src[:, 1]
    A[1::2, 1] = src[:, 0]
    A[1::2, 3] = 1.0
    try:
        sol, *_ = np.linalg.lstsq(A, dst.reshape(-1).astype(np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, tx, ty = sol
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _fit_homography_ls(src, dst):
    """Least-squares (DLT) homography - deterministic, unlike the RANSAC call."""
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), 0)
    return None if H is None else np.asarray(H, dtype=np.float64)


def _held_out_error(fit, src, dst, splits=None):
    """Median reprojection error of `fit` on matches it was not fitted to.

    The same splits are used for every model - the RNG is seeded per call - so
    the comparison is paired and the whole stage stays reproducible.
    """
    splits = _CV_SPLITS if splits is None else splits
    n = len(src)
    train_n = int(round(n * _CV_TRAIN_FRACTION))
    if n - train_n < 2 or train_n < 4:
        return float('inf')

    rng = np.random.default_rng(0)
    errors = []
    for _ in range(splits):
        order = rng.permutation(n)
        train, test = order[:train_n], order[train_n:]
        M = fit(src[train], dst[train])
        if M is None:
            return float('inf')
        # Median over the held-out matches: a ratio test at 0.70 still lets a
        # few false matches through, and one of those in a test fold would
        # otherwise decide the model.
        errors.append(float(np.median(
            np.linalg.norm(_project_points(M, src[test]) - dst[test], axis=1))))
    return float(np.median(errors))


def _select_pair_transform(pts_curr, pts_last):
    """Fit both models to one pair's matches and return (matrix, model name).

    Returns (None, "failed") when neither model can be estimated.
    """
    src = np.asarray(pts_curr, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(pts_last, dtype=np.float64).reshape(-1, 2)

    H_hom, mask_hom = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                         cv2.RANSAC, 5.0)
    M_sim, mask_sim = cv2.estimateAffinePartial2D(
        src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=5.0)
    H_sim = (None if M_sim is None
             else np.vstack([M_sim, [0.0, 0.0, 1.0]]).astype(np.float64))

    if H_sim is None:
        return (H_hom, "homography") if H_hom is not None else (None, "failed")
    if H_hom is None:
        return H_sim, "similarity"

    # Validate on the matches RANSAC believed, not on everything the ratio test
    # let through: the folds are fitted by least squares, which one false match
    # is enough to wreck. The union of the two inlier sets is used so that a
    # match only the homography accepted - which is what a real perspective
    # looks like out at the frame edges - still counts as evidence for it.
    keep = np.zeros(len(src), dtype=bool)
    if mask_hom is not None:
        keep |= mask_hom.ravel().astype(bool)
    if mask_sim is not None:
        keep |= mask_sim.ravel().astype(bool)
    if keep.any():
        src, dst = src[keep], dst[keep]

    if len(src) < _CV_MIN_MATCHES:
        # Too few matches to tell the models apart: 8 DOF through ~10 points
        # reproduces them almost exactly however wrong it is, so validation
        # cannot see the overfit. The constrained model is the safe answer.
        return H_sim, "similarity"

    err_hom = _held_out_error(_fit_homography_ls, src, dst)
    err_sim = _held_out_error(_fit_similarity_ls, src, dst)
    if err_hom < _CV_MARGIN * err_sim:
        return H_hom, "homography"
    return H_sim, "similarity"


def _estimate_homography(images, downscale_width, thread_count, parallel_ecc=True):
    """Estimate the per-frame homography (or similarity) of every frame.

    See _align_homography_impl for the stage's properties and
    _select_pair_transform for how the model is chosen per pair.

    Returns:
        list of 3x3 forward maps placing frame i onto the frame-0 canvas.
    """
    import concurrent.futures

    num_images = len(images)

    # --- Initialization ---
    h_orig, w_orig = images[0].shape[:2]
    downscale_width = _resolve_detection_width(downscale_width, h_orig, w_orig, "Homography")

    # Global accumulated matrix (used to map the current frame directly back to frame 0)
    H_global = np.eye(3, dtype=np.float32)

    # Collect all transform matrices for precise cropping
    H_matrices = [np.eye(3, dtype=np.float32)]  # The first image is the reference, identity matrix

    print(f"Aligning {num_images} images using Sequential Homography (Parallel Optimized)...")

    # --- 1. Parallel feature extraction ---
    print("  - Step 1/2: Extracting features concurrently...")

    def get_features_task(img):
        # Create an independent detector in each thread to ensure thread safety
        local_detector = cv2.SIFT_create()
        h, w = img.shape[:2]
        scale = _detection_scale(w, downscale_width)
        if scale < 1.0:
            # INTER_LINEAR is faster and usually good enough for feature detection
            img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            img_small = img
        # SIFT only accepts 8-bit input. This measures a transform which is then
        # applied to the full-depth frame, so narrowing here costs nothing in
        # the output - keypoint positions do not get more accurate with more
        # bits, and the frames are already downscaled for detection anyway.
        kps, des = local_detector.detectAndCompute(bitdepth.to_analysis8(img_small), None)
        return kps, des, scale

    # Use a thread pool to extract features in parallel
    # Most OpenCV operations release the GIL, so multithreading can effectively speed things up
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        features_list = list(executor.map(get_features_task, images))

    # --- 2. Sequential matrix computation (must be serial) ---
    print("  - Step 2/2: Calculating transform matrices...")

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Get the features of the first frame
    last_kps, last_des, last_scale = features_list[0]

    constrained_pairs = 0

    for idx in range(1, num_images):
        curr_kps, curr_des, curr_scale = features_list[idx]

        # Exception handling: insufficient features
        if curr_des is None or len(curr_kps) < 4 or last_des is None:
            print(f"Warning: Frame {idx} features insufficient. Keeping original position.")
            H_matrices.append(H_global.copy())
            continue

        # Feature matching (Current vs Last)
        matches = bf.knnMatch(curr_des, last_des, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.70 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 6:
            print(f"Warning: Frame {idx} poor matches ({len(good_matches)}). Keeping previous trajectory.")
            H_matrices.append(H_global.copy())
            continue

        # Extract coordinates
        pts_curr = np.float32([curr_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / curr_scale
        pts_last = np.float32([last_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / last_scale

        # Fit a homography and a similarity to the same matches and keep
        # whichever earns its degrees of freedom on held-out matches.
        H_local, model = _select_pair_transform(pts_curr, pts_last)

        if H_local is None:
            print(f"Frame {idx} alignment failed.")
            H_local = np.eye(3)
        elif model == "similarity":
            constrained_pairs += 1

        # Matrix chain multiplication
        H_global = np.matmul(H_global, H_local)
        H_matrices.append(H_global.copy())

        # Update the reference
        last_kps = curr_kps
        last_des = curr_des
        last_scale = curr_scale

    if constrained_pairs:
        print(f"    {constrained_pairs}/{num_images - 1} pairs took the constrained "
              f"similarity - their perspective terms did not survive validation.")

    return H_matrices


def _align_homography_impl(input_source, output_path=None, img_filenames=None,
                           downscale_width=DEFAULT_DETECTION_WIDTH, thread_count: int = 4,
                           reference_index: int = 0):
    """
    Optimized commercial-grade image-alignment algorithm
    Features:
    1. Sequential Alignment: solves feature loss under large depth of field
    2. Pyramid speedup (Downscale Processing): greatly improves feature-detection speed
    3. Matrix Chaining: reduces accumulated error
    4. Lanczos interpolation: preserves image sharpness
    5. Parallel computation optimization (Parallel Processing): uses multiple cores to speed up feature extraction and image warping
    6. Per-pair model selection: the 8-DOF homography is kept only where its two
       perspective terms predict held-out matches better than a 4-DOF
       similarity does, so a pair whose motion carries no perspective - which
       on a focus rail is every pair - is fitted with the constrained model
       instead of bending the frame to fit match noise. See
       _select_pair_transform.
    """
    return align_stack(input_source, ('homography',), output_path=output_path,
                       img_filenames=img_filenames, downscale_width=downscale_width,
                       thread_count=thread_count, reference_index=reference_index)


# ========== ECC alignment algorithm implementation (high precision) ==========

def _estimate_ecc(images, downscale_width, thread_count, parallel_ecc=True):
    """Estimate the per-frame ECC homography of every frame.

    See _align_ecc_impl for what the stage is for and where it is weak.

    Returns:
        list of 3x3 forward maps placing frame i onto the frame-0 canvas.
    """
    import concurrent.futures

    # --- 1. Initialization ---
    h_orig, w_orig = images[0].shape[:2]
    downscale_width = _resolve_detection_width(downscale_width, h_orig, w_orig, "ECC")

    # Global transform matrix (3x3 identity matrix)
    H_global = np.eye(3, dtype=np.float32)

    # Collect all transform matrices for precise cropping
    H_matrices = [np.eye(3, dtype=np.float32)]  # The first image is the reference

    # Define the ECC transform type
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # ECC termination criteria
    number_of_iterations = 50
    termination_eps = 1e-4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Preprocessing function: to grayscale + downsampling + Gaussian blur
    def preprocess(img):
        h, w = img.shape[:2]
        scale = _detection_scale(w, downscale_width)
        if scale < 1.0:
            small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_img = img # just reference, no copy needed

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # findTransformECC takes 8-bit or float32 only. 8-bit is used here for
        # the same reason as SIFT above: this measures a transform that is then
        # applied to the full-depth frame, and the image it measures on has
        # already been downscaled and Gaussian-blurred, so the extra bits carry
        # no alignment information. Narrowing also keeps the 8-bit path
        # bit-identical to previous releases.
        return bitdepth.to_analysis8(gray), scale

    print(f"Aligning {len(images)} images using ECC (Parallel Optimized)...")

    # --- 2. Parallel preprocessing ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        # map guarantees the result order matches the input
        preprocessed_data = list(executor.map(preprocess, images))

    # All images share the same width, so the first image's scale applies to all pairs
    _, scale_factor = preprocessed_data[0]

    # --- 3. Pairwise matrix computation ---
    # Each pair (frame idx-1 vs idx) is independent; only the chain accumulation
    # below must stay sequential. findTransformECC releases the GIL, so the
    # pairs can run concurrently when parallel_ecc is enabled.
    def compute_pair_matrix(idx):
        curr_gray, _ = preprocessed_data[idx]
        prev_gray, _ = preprocessed_data[idx - 1]

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                prev_gray,  # template
                curr_gray,  # input
                warp_matrix,
                warp_mode,
                criteria,
                None,
                1
            )

            # Restore scale to full resolution
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix[0, 2] /= scale_factor
                warp_matrix[1, 2] /= scale_factor
                warp_matrix[2, 0] *= scale_factor
                warp_matrix[2, 1] *= scale_factor
            else:
                warp_matrix[0, 2] /= scale_factor
                warp_matrix[1, 2] /= scale_factor

        except cv2.error:
            print(f"Warning: ECC failed to converge at frame {idx}. Assuming no motion.")
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

        return warp_matrix

    pair_indices = range(1, len(images))
    if parallel_ecc and len(images) > 2:
        print(f"    ECC: computing {len(images) - 1} pair matrices in parallel ({thread_count} workers)", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            pair_matrices = list(executor.map(compute_pair_matrix, pair_indices))
    else:
        pair_matrices = [compute_pair_matrix(idx) for idx in pair_indices]

    # --- Sequential chain accumulation (order-dependent) ---
    for warp_matrix in pair_matrices:
        if warp_mode == cv2.MOTION_AFFINE:
            row = np.array([[0, 0, 1]], dtype=np.float32)
            H_local_3x3 = np.vstack([warp_matrix, row])
            H_global = np.matmul(H_global, H_local_3x3)
        else:
            H_global = np.matmul(H_global, warp_matrix)

        # Record the inverse transform (maps target back to source)
        H_inv = np.linalg.inv(H_global)
        H_matrices.append(H_inv.copy())

    return H_matrices


def _align_ecc_impl(input_source, output_path=None, img_filenames=None,
                    downscale_width=DEFAULT_DETECTION_WIDTH, thread_count: int = 4,
                    parallel_ecc: bool = True, reference_index: int = 0):
    """
    High-precision image-stack alignment algorithm based on ECC (Enhanced Correlation Coefficient)
    Suitable for: fine, sub-pixel refinement of global drift when feature detection is unreliable
    Advantages: sub-pixel precision, does not rely on feature points
    Optimizations: parallel preprocessing, parallel warping, merged cropping operation

    Note: this stage fits an 8-DOF homography. For focus-breathing (magnification
    change) specifically, prefer the dedicated 'scale' method (_align_scale_impl),
    which fits a constrained 4-DOF similarity and does not overfit on frames that
    differ in blur.
    """
    return align_stack(input_source, ('ecc',), output_path=output_path,
                       img_filenames=img_filenames, downscale_width=downscale_width,
                       thread_count=thread_count, parallel_ecc=parallel_ecc,
                       reference_index=reference_index)


# ========== The pipeline: one resample, one crop ==========
#
# (docs/REGISTRATION_IMPROVEMENTS.md item 7)
#
# Each stage used to be a complete registration: estimate, warp, crop, hand the
# *pixels* to the next stage. Resampling is not idempotent, so a three-stage
# pipeline interpolated every frame three times - 31% of the high-frequency
# energy gone where one stage costs 20% - and every stage took the intersection
# of the valid regions again, so the crops composed and a fifth of the frame was
# thrown away. In a focus stacker both are paid in the wrong currency: the
# fusion stage decides which frame to source each pixel from by asking which is
# locally sharpest, and it cannot see detail an earlier warp has already
# discarded.
#
# Every stage already produces a 3x3 matrix per frame, so the stages compose
# with a matrix product. The pipeline below estimates each stage in turn,
# accumulates the maps, and warps the *original* frames exactly once, into a
# single crop computed from the composed maps. Output frames are therefore
# resampled once and cropped once for any number of stages.
#
# The intermediate frames still exist, because a stage measures on pixels: ECC
# correlates them and SIFT detects on them. They are built the same way the
# previous stage's output used to be built - same warp, same kernel, same crop -
# so every estimate is what it was before, and they are then thrown away rather
# than carried into the result. That keeps the warp count unchanged (n stages
# still means n warps per frame) and moves the interpolation off the picture and
# onto a measurement.
#
# The bookkeeping is a conjugation. `composed[i]` maps original frame i onto the
# reference frame's own coordinates; `canvas` maps those coordinates onto the
# intermediate canvas the current pixels live in. A stage measures `S[i]` in
# that canvas, so the map in original coordinates is inv(canvas) . S[i] . canvas,
# applied in front of what is already composed. With one stage `canvas` is the
# identity and the composition is a no-op, which is what keeps every
# single-stage result bit-identical to the release before this one.

_STAGE_ESTIMATORS = {
    'scale': _estimate_scale,
    'homography': _estimate_homography,
    'ecc': _estimate_ecc,
}

# Pipeline order, and what each public method name expands to. 'both' has always
# meant homography followed by ECC.
_STAGE_ORDER = ('scale', 'homography', 'ecc')
_METHOD_STAGES = {
    'scale': ('scale',),
    'homography': ('homography',),
    'ecc': ('ecc',),
    'both': ('homography', 'ecc'),
}


def resolve_stages(method) -> tuple:
    """Normalise a method into the stages it runs, in pipeline order.

    Accepts a single name ('scale', 'homography', 'ecc', 'both'), a '+'-joined
    string ('scale+ecc'), or any sequence of those. Duplicates collapse, so
    ('scale', 'both', 'ecc') is scale -> homography -> ECC.

    Raises:
        ValueError: on a name that is not a registration method.
    """
    if method is None:
        return ()
    names = (method.split('+') if isinstance(method, str) else list(method))

    stages = []
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if name not in _METHOD_STAGES:
            raise ValueError(
                f"Unsupported registration method: {name}. "
                f"Supported methods: {', '.join(_METHOD_STAGES)}"
            )
        for stage in _METHOD_STAGES[name]:
            if stage not in stages:
                stages.append(stage)
    return tuple(sorted(stages, key=_STAGE_ORDER.index))


def _plan_canvas(H_matrices, img_shape):
    """Output size and crop translation for one set of composed maps.

    Returns:
        (target_w, target_h, T_crop). A degenerate valid region - which means
        the frames have no overlap worth cropping to - falls back to the input
        size and no crop, as it always has.
    """
    h_orig, w_orig = img_shape
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, img_shape)

    if top >= bottom or left >= right:
        print("Warning: Invalid crop region, skipping crop.")
        return w_orig, h_orig, _build_crop_transform(0, 0)

    # The crop translation carries the sub-pixel offset that keeps the reference
    # frame from being copied through unresampled (item 3), so it is built the
    # same way whether or not the crop itself is real.
    return right - left, bottom - top, _build_crop_transform(-left, -top)


def _warp_stack(images, H_matrices, T_crop, target_w, target_h, thread_count):
    """Warp every frame into the shared canvas, once, in parallel."""
    import concurrent.futures

    def warp_task(args):
        img, H = args
        return _warp_frame(img, T_crop @ H, target_w, target_h)

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        return list(executor.map(warp_task, zip(images, H_matrices)))


def align_stack(input_source, stages, output_path=None, img_filenames=None,
                downscale_width=DEFAULT_DETECTION_WIDTH, thread_count: int = 4,
                parallel_ecc: bool = True, reference_index: int = 0):
    """Register a stack with any number of stages, resampling the frames once.

    Args:
        input_source: directory path or a preloaded list of images
        stages: stage names in pipeline order, as resolve_stages returns them
        output_path: optional directory to save the result to (None = return only)
        img_filenames: optional filenames matching a preloaded image list
        downscale_width: width the frames are downsampled to before a transform
            is measured on them - detection accuracy against time
        thread_count: worker threads for detection and warping
        parallel_ecc: compute ECC pair matrices concurrently (identical results)
        reference_index: frame held fixed (0 = first frame, the default)

    Returns:
        list of aligned images
    """
    stages = tuple(stages)
    images, img_filenames = _load_stack(
        input_source, img_filenames,
        _STAGE_EXTENSIONS[stages[0]] if stages else _FEATURE_EXTENSIONS)

    if not stages or len(images) < 2:
        return images

    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    h_orig, w_orig = images[0].shape[:2]

    # Original frame i -> reference-frame coordinates. None until the first
    # stage has run, so that a single-stage pipeline uses that stage's own
    # matrices untouched rather than multiplying them by an identity.
    composed = None
    # Reference-frame coordinates -> the canvas the current pixels live in.
    canvas = None
    current = images

    for position, stage in enumerate(stages):
        H_stage = _STAGE_ESTIMATORS[stage](
            current, downscale_width, max_workers, parallel_ecc)
        # Re-reference the chain onto the chosen frame before it is composed:
        # the stage measured against frame 0 of whatever it was handed.
        H_stage = _rereference_transforms(H_stage, reference_index)

        if composed is None:
            composed = H_stage
        else:
            canvas_inv = np.linalg.inv(canvas)
            composed = [canvas_inv @ np.asarray(S, dtype=np.float64) @ canvas
                        @ np.asarray(C, dtype=np.float64)
                        for S, C in zip(H_stage, composed)]

        target_w, target_h, T_crop = _plan_canvas(composed, (h_orig, w_orig))

        if position + 1 == len(stages):
            break

        # Not the last stage: rebuild the canvas the next estimator measures on.
        # These pixels are the previous release's stage output exactly, and they
        # go no further than the next estimate.
        print(f"    Composing {stage} into the pipeline; "
              f"measuring the next stage on a {target_w}x{target_h} canvas...")
        current = _warp_stack(images, composed, T_crop, target_w, target_h, max_workers)
        canvas = np.asarray(T_crop, dtype=np.float64)

    print(f"    Warping {len(images)} frames once, "
          f"through {len(stages)} composed stage(s), to {target_w}x{target_h}...")
    aligned_images = _warp_stack(images, composed, T_crop, target_w, target_h, max_workers)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        for idx, img in enumerate(aligned_images):
            fname = img_filenames[idx] if img_filenames else f'frame_{idx:04d}.png'
            cv2.imwrite(os.path.join(output_path, fname),
                        bitdepth.prepare_for_write(img, os.path.splitext(fname)[1]))

    return aligned_images


# ========== Stable registration algorithm implementation ==========

def _stabilisation_impl(input_source, output_path=None, filenames=None):
    """
    Stabilize images from directory path or image list
    Args:
        input_source: string (directory path) or list of images
        output_path: string, directory path to save results (optional, None means no saving)
        filenames: list of original filenames (optional)
    Returns:
        list of stabilized images (always returns processed images regardless of output_path)
    """
    # Process input source
    img_filenames = filenames  # Use passed filenames or detect from input
    if isinstance(input_source, str):
        img_paths = sorted(
            [os.path.join(input_source, file) for file in os.listdir(input_source)
             if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']],
            key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1])
        )
        images = [read_image_any_depth(path) for path in img_paths]
        images = [img for img in images if img is not None]
        # Store original filenames with extensions
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source

    n_frames = len(images)
    img_first = images[0]
    h, w = img_first.shape[:2]

    prev_gray = cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames, 3), np.float32)

    # Feature detection parameters
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate transforms (frame i to frame i+1)
    for i in range(n_frames - 1):
        curr = images[i + 1]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
        if prev_pts is None:
            # If no feature points are found, use the previous transform
            if i > 0:
                transforms[i] = transforms[i-1]
            continue
            
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        idx = status.ravel() == 1
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if len(prev_pts) < 4:
            # If there are too few matching points, use the previous transform
            if i > 0:
                transforms[i] = transforms[i-1]
            continue

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is None:
            # If estimation fails, use the previous transform
            if i > 0:
                transforms[i] = transforms[i-1]
            continue
            
        dx, dy = m[0, 2], m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    # The last frame uses the same transform as the second-to-last frame
    transforms[-1] = transforms[-2]

    # Smooth trajectory
    def smooth(trajectory, radius=30):
        smoothed_trajectory = np.copy(trajectory)
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        padding = np.pad(trajectory, ((radius, radius), (0, 0)), 'edge')
        for i in range(3):
            smoothed_trajectory[:, i] = np.convolve(padding[:, i], kernel, mode='valid')
        return smoothed_trajectory

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory

    # Apply transforms
    stabilized_images = []
    for i, frame in enumerate(images):
        # Apply the corresponding difference correction to each frame
        dx, dy, da = difference[i]
        
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy]
        ], dtype=np.float32)

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
        frame_stabilized = cv2.warpAffine(frame_stabilized, T, (w, h))
        stabilized_images.append(frame_stabilized)

        # Save results if output path is provided
        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # Use original filename if available, otherwise use default naming
            if img_filenames:
                output_file = os.path.join(output_path, img_filenames[i])
            else:
                output_file = os.path.join(output_path, f'frame_{i:04d}.png')
            cv2.imwrite(output_file, frame_stabilized)

    return stabilized_images


# # ========== Combined registration algorithm implementation ==========

# def _registration_impl(input_source, output_path=None):
#     """
#     Internal implementation of combined registration
#     Args:
#         input_source: string (directory path) or list of images
#         output_path: string, directory path to save results (optional, None means no saving)
#     Returns:
#         list of registered images (always returns processed images regardless of output_path)
#     """
#     # Store original filenames if input is a directory
#     img_filenames = None
#     if isinstance(input_source, str):
#         num_pattern = re.compile(r"\d+")
#         img_paths = sorted(
#             (os.path.join(input_source, f) for f in os.listdir(input_source)
#              if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}),
#             key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1])
#         )
#         img_filenames = [os.path.basename(path) for path in img_paths]
    
#     # Step 1: Coarse alignment using zoom
#     zoom_aligned_images = _align_zoom_impl(input_source)
    
#     # Step 2: Fine registration using stabilisation
#     final_images = _stabilisation_impl(zoom_aligned_images, output_path, img_filenames)
    
#     return final_images


# ========== Unified interface class ==========

class ImageRegistration:
    """
    Unified interface class for image-sequence registration
    
    Supported methods:
    - 'scale': scale / focus-breathing correction (similarity: uniform scale +
      rotation + translation)
    - 'homography': homography alignment registration (non-linear)
    - 'ecc': ECC alignment registration (high precision, sub-pixel level)
    - 'both': combined registration (Homography first, then ECC)

    Several stages can also be asked for in one call, as a '+'-joined string or
    a sequence - 'scale+ecc', ['scale', 'both']. That is how they should be run:
    the stages are composed into a single transform per frame and the frames are
    resampled and cropped once, where chaining separate calls resamples and
    crops once per stage (docs/REGISTRATION_IMPROVEMENTS.md item 7).

    Example:
        # Correct focus breathing (magnification change)
        registration = ImageRegistration(method='scale')
        result = registration.process('./images', './output')

        # Use homography alignment
        registration = ImageRegistration(method='homography')
        result = registration.process('./images', './output')
        
        # Use ECC alignment
        registration = ImageRegistration(method='ecc')
        result = registration.process(image_list, './output')

        # Use combined alignment
        registration = ImageRegistration(method='both')
        result = registration.process(image_list, './output')

        # The app's default pipeline, composed into one warp
        registration = ImageRegistration(method='scale+homography')
        result = registration.process(image_list, './output')
    """

    SUPPORTED_METHODS = ['scale', 'homography', 'ecc', 'both']

    def __init__(self, method='homography', downscale_width: int = DEFAULT_DETECTION_WIDTH,
                 ecc_parallel: bool = True,
                 reference_index: int = 0):
        """
        Initialize the registrar

        Args:
            method (str or sequence): registration method - one of 'scale',
                'homography', 'ecc', 'both', or several of them together as a
                '+'-joined string or a sequence. Stages asked for together are
                composed and applied as one warp; chaining separate calls costs
                a resample and a crop per stage.
            downscale_width (int): width the frames are downsampled to before a
                transform is measured on them. Honoured at any frame size - the
                stages no longer replace it with 1024 on large input - so it is
                a real trade of detection accuracy against time.
            ecc_parallel (bool): Compute ECC pair matrices concurrently (identical results, faster)
            reference_index (int): index of the frame held fixed during alignment.
                0 (the default) keeps the historical behaviour of referencing the
                first frame; the middle frame minimises accumulated chain drift.
        """
        # Raises ValueError on anything that is not a registration method.
        self.stages = resolve_stages(method)

        self.method = method
        # User-configurable downsampling width, used in preprocessing stages such as feature extraction
        self.downscale_width = (int(downscale_width) if downscale_width is not None
                                else DEFAULT_DETECTION_WIDTH)
        self.ecc_parallel = bool(ecc_parallel)
        # Frame that stays fixed while the others align onto it (0 = first frame).
        self.reference_index = int(reference_index) if reference_index is not None else 0

    def process(self,
                input_source: Union[str, List[np.ndarray]],
                output_path: Optional[str] = None,
                thread_count: int = 4) -> List[np.ndarray]:
        """
        Perform image registration

        Args:
            input_source (str or list): image directory path or a preloaded list of images
            output_path (str, optional): output directory path.
                - If a path is provided, the registered images are saved to disk
                - If None, only the image list is returned without saving, saving disk space and I/O overhead

        Returns:
            list: list of registered images (always returned, whether or not saved to disk)
        """
        if len(self.stages) > 1:
            print(f"=== Composed registration: {' -> '.join(self.stages)} ===")
        return align_stack(input_source, self.stages, output_path=output_path,
                           downscale_width=self.downscale_width,
                           thread_count=thread_count,
                           parallel_ecc=self.ecc_parallel,
                           reference_index=self.reference_index)

    # The three _process_<stage> methods that used to sit here were the dispatch
    # table `process` walked; the pipeline dispatches on the stage list instead,
    # so they became three unreachable one-line delegations and went. The module
    # functions they called - _align_scale_impl and its two neighbours - are
    # still the single-stage entry points, and still what the CLI uses.

    def set_method(self, method):
        """
        Switch the registration method

        Args:
            method (str or sequence): the new registration method, in any of the
                forms __init__ accepts
        """
        self.stages = resolve_stages(method)
        self.method = method
    
    def get_info(self) -> dict:
        """
        Get the current registrar info
        
        Returns:
            dict: contains info such as the registration method
        """
        return {
            'method': self.method
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ImageRegistration(method='{self.method}')"


# ========== Convenience function ==========

def register_images(input_source: Union[str, List[np.ndarray]],
                    method: str = 'homography',
                    output_path: Optional[str] = None) -> List[np.ndarray]:
    """
    Convenience function: perform image registration in one call
    
    Args:
        input_source: image source (directory path or list of images)
        method: registration method ('scale', 'homography', 'ecc', 'both')
        output_path: output path (optional, None means return the list only without saving)
    
    Returns:
        list of registered images
    
    Example:
        # Save to disk and return the list
        result = register_images('./images', method='homography', output_path='./output')
        
        # Return the list only without saving, to save disk space
        result = register_images('./images', method='ecc')
        
        # Combined alignment
        result = register_images(image_list, method='both')
    """
    registration = ImageRegistration(method=method)
    return registration.process(input_source, output_path=output_path)


# ========== Backward-compatible function aliases ==========

# def image_stack_align_zoom(input_source, output_path=None):
#     """Backward-compatible function aliases"""
#     return _align_zoom_impl(input_source, output_path)

# def image_stack_stabilisation(input_source, output_path=None, filenames=None):
#     """Backward-compatible function aliases"""
#     return _stabilisation_impl(input_source, output_path, filenames)

# def image_stack_registration(input_source, output_path=None):
#     """Backward-compatible function aliases"""
#     return _registration_impl(input_source, output_path)

# def process_image_stack(input_path, output_path=None):
#     """Backward-compatible function aliases"""
#     return _registration_impl(input_path, output_path)

def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description='Image Stack Registration Tool')
    parser.add_argument('--input_path', default='./coral_best_zoom', help='Input image directory path')
    parser.add_argument('--output', default=r'E:\FinishedProjects\XuChuang\regi_test', help='Output directory path')
    parser.add_argument('--mode', default='homography', choices=['scale', 'homography', 'ecc', 'both'],
                        help='Processing mode: scale (focus breathing), homography, ecc, both (homography + ecc)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        print(f"\n=== {args.mode.upper()} Alignment ===")
        output_dir = os.path.join(args.output, f'{args.mode}_result')
        registration = ImageRegistration(method=args.mode)
        processed_images = registration.process(args.input_path, output_dir)
        print(f"Alignment completed. Results saved to: {output_dir}")
        print(f"Processed {len(processed_images)} images")
        
        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())