"""
Unified interface for image-sequence registration

Provides a unified calling interface for three registration methods:
1. Homography (homography alignment)
2. ECC (ECC alignment)
3. Both (combined registration: Homography + ECC)

All algorithm implementations are contained in this script, with no external dependencies

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
import io
from typing import Union, List, Optional

# Set standard output encoding to utf-8 when a console stream exists
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# # ========== Scale-alignment algorithm implementation (linear) ==========



# def _align_zoom_impl(input_source, output_path=None, img_filenames=None):
#     """
#     Align images from directory path or image list
#     Args:
#         input_source: string (directory path) or list of images
#         output_path: string, directory path to save results (optional, None means no saving)
#     Returns:
#         list of aligned images (always returns processed images regardless of output_path)
#     """
#     # Process input source
#     if img_filenames is None and isinstance(input_source, str):
#         # Pre-compile regular expression
#         num_pattern = re.compile(r"\d+")
#         # Use generator expression and list comprehension for optimized file filtering and sorting
#         img_paths = sorted(
#             (os.path.join(input_source, f) for f in os.listdir(input_source)
#              if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}),
#             key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1])
#         )
#         # Use list comprehension to read all images at once
#         images = [cv2.imread(path) for path in img_paths]
#         # Store original filenames with extensions
#         img_filenames = [os.path.basename(path) for path in img_paths]
#     else:
#         images = input_source

#     # Cache image dimensions and feature detector
#     img_first = images[0]
#     img_last = images[-1]
#     h_first, w_first = img_first.shape[:2]
#     h_last, w_last = img_last.shape[:2]

#     # SIFT feature detection and matching
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img_first, None)
#     kp2, des2 = sift.detectAndCompute(img_last, None)

#     # Use BFMatcher for feature matching (SIFT uses L2 norm)
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

#     # Find valid matching points
#     min_distance_threshold = w_first / 2
#     p1 = p2 = q1 = q2 = None

#     # Use numpy array operations to optimize distance calculation
#     for i in range(len(matches) - 1):
#         p1 = np.array(kp1[matches[i].queryIdx].pt)
#         p2 = np.array(kp1[matches[i + 1].queryIdx].pt)
#         q1 = np.array(kp2[matches[i].trainIdx].pt)
#         q2 = np.array(kp2[matches[i + 1].trainIdx].pt)

#         if np.linalg.norm(p1 - p2) > min_distance_threshold:
#             break
#     else:
#         raise ValueError("Unable to find matching point pairs that meet the criteria")

#     # Calculate scaling factor
#     dist = np.linalg.norm(p1 - p2)
#     dist2 = np.linalg.norm(q1 - q2)
#     c = max(dist2 / dist, dist / dist2)

#     # Pre-calculate all scaling factors
#     aug_list = np.linspace(1, c, len(images))
#     is_forward = dist2 / dist > 1
#     target_shape = (h_last, w_last) if is_forward else (h_first, w_first)
#     aug_factors = aug_list[::-1] if is_forward else aug_list

#     # Pre-create output directory
#     if output_path:
#         os.makedirs(output_path, exist_ok=True)

#     # Process images
#     aligned_images = []
#     target_h, target_w = target_shape

#     for idx, (img, aug_factor) in enumerate(zip(images, aug_factors)):
#         # Use cv2.INTER_AREA for scaling
#         img_resized = cv2.resize(img, None, fx=aug_factor, fy=aug_factor, interpolation=cv2.INTER_AREA)

#         # Calculate crop position
#         h, w = img_resized.shape[:2]
#         x = (w - target_w) // 2
#         y = (h - target_h) // 2

#         # Crop image
#         img_crop = img_resized[y:y + target_h, x:x + target_w]
#         aligned_images.append(img_crop)

#         # Save results
#         if output_path:
#             # Use original filename if available, otherwise use default naming
#             if img_filenames:
#                 output_file = os.path.join(output_path, img_filenames[idx])
#             else:
#                 output_file = os.path.join(output_path, f'frame_{idx:04d}.png')
#             cv2.imwrite(output_file, img_crop)

#     return aligned_images


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

def _align_homography_impl(input_source, output_path=None, img_filenames=None, downscale_width=1600, thread_count: int = 4):
    """
    Optimized commercial-grade image-alignment algorithm
    Features:
    1. Sequential Alignment: solves feature loss under large depth of field
    2. Pyramid speedup (Downscale Processing): greatly improves feature-detection speed
    3. Matrix Chaining: reduces accumulated error
    4. Lanczos interpolation: preserves image sharpness
    5. Parallel computation optimization (Parallel Processing): uses multiple cores to speed up feature extraction and image warping
    """
    import concurrent.futures

    # --- 1. Data loading and preprocessing ---
    if img_filenames is None and isinstance(input_source, str):
        num_pattern = re.compile(r"\d+")
        # Support common formats, filter out non-images
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in valid_exts),
            key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1]) if num_pattern.findall(os.path.basename(x)) else x
        )
        # Note: for memory reasons, commercial software usually does not read all large images at once
        # But to keep the interface consistent, we read them all in here. A better approach would be to build a generator.
        images = [cv2.imread(path) for path in img_paths]
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source

    num_images = len(images)
    if num_images < 2:
        return images

    # --- Initialization ---
    h_orig, w_orig = images[0].shape[:2]
    # If a single image is too large (any side >= 2048), force the target width used for downsampling to 1024
    max_dim = max(h_orig, w_orig)
    if max_dim >= 2048:
        try:
            # Record the previous value for debugging
            prev_down = downscale_width
        except NameError:
            prev_down = None
        downscale_width = 1024
        print(f"[Registration] Large image detected ({h_orig}x{w_orig}), setting downscale_width {prev_down} -> {downscale_width}")
    
    # Global accumulated matrix (used to map the current frame directly back to frame 0)
    H_global = np.eye(3, dtype=np.float32)
    
    # Collect all transform matrices for precise cropping
    H_matrices = [np.eye(3, dtype=np.float32)]  # The first image is the reference, identity matrix

    print(f"Aligning {num_images} images using Sequential Homography (Parallel Optimized)...")

    # --- 2. Parallel feature extraction ---
    print("  - Step 1/3: Extracting features concurrently...")

    def get_features_task(img):
        # Create an independent detector in each thread to ensure thread safety
        local_detector = cv2.SIFT_create()
        h, w = img.shape[:2]
        scale = downscale_width / float(w) if w > downscale_width else 1.0
        if scale < 1.0:
            # INTER_LINEAR is faster and usually good enough for feature detection
            img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            img_small = img
        kps, des = local_detector.detectAndCompute(img_small, None)
        return kps, des, scale

    # Use a thread pool to extract features in parallel
    # Most OpenCV operations release the GIL, so multithreading can effectively speed things up
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        features_list = list(executor.map(get_features_task, images))

    # --- 3. Sequential matrix computation (must be serial) ---
    print("  - Step 2/3: Calculating transform matrices...")
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # Get the features of the first frame
    last_kps, last_des, last_scale = features_list[0]

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

        # Compute the homography matrix with RANSAC
        H_local, mask = cv2.findHomography(pts_curr, pts_last, cv2.RANSAC, 5.0)

        if H_local is None:
            print(f"Frame {idx} alignment failed.")
            H_local = np.eye(3)

        # Matrix chain multiplication
        H_global = np.matmul(H_global, H_local)
        H_matrices.append(H_global.copy())

        # Update the reference
        last_kps = curr_kps
        last_des = curr_des
        last_scale = curr_scale

    # --- 4. Apply transforms and crop in parallel ---
    print("  - Step 3/3: Warping images concurrently...")
    
    # Precompute the crop region and warp directly into the target region, avoiding the waste of warping first and cropping later
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h_orig, w_orig))
    
    do_crop = True
    if top >= bottom or left >= right:
        print("Warning: Invalid crop region, skipping crop.")
        do_crop = False
        target_w, target_h = w_orig, h_orig
        offset_x, offset_y = 0, 0
    else:
        target_w = right - left
        target_h = bottom - top
        offset_x = -left
        offset_y = -top
        print(f"    Optimized: Warping directly to cropped region ({target_w}x{target_h})...")

    # Build the crop translation matrix
    T_crop = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    def warp_task(args):
        img, H = args
        # Merge the crop transform
        if do_crop:
            H_final = T_crop @ H
        else:
            H_final = H
            
        return cv2.warpPerspective(img, H_final, (target_w, target_h), 
                                 flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    # Prepare the parameters
    warp_args = zip(images, H_matrices)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        aligned_images = list(executor.map(warp_task, warp_args))

    # If an output path is provided, save the image
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        print("  - Saving results...")
        for idx, img in enumerate(aligned_images):
            fname = img_filenames[idx] if img_filenames else f'frame_{idx:04d}.png'
            cv2.imwrite(os.path.join(output_path, fname), img)

    return aligned_images


# ========== ECC alignment algorithm implementation (high precision) ==========

def _align_ecc_impl(input_source, output_path=None, img_filenames=None, downscale_width=1000, thread_count: int = 4,
                    parallel_ecc: bool = True):
    """
    High-precision image-stack alignment algorithm based on ECC (Enhanced Correlation Coefficient)
    Suitable for: image stacks with focus-breathing in microscopy and macro photography
    Advantages: sub-pixel precision, automatically handles scaling-center offset, does not rely on feature points
    Optimizations: parallel preprocessing, parallel warping, merged cropping operation
    """
    import concurrent.futures

    # --- 1. Data loading ---
    if img_filenames is None and isinstance(input_source, str):
        num_pattern = re.compile(r"\d+")
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}),
            key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1]) if num_pattern.findall(os.path.basename(x)) else x
        )
        images = [cv2.imread(path) for path in img_paths]
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source

    if len(images) < 2:
        return images

    # --- 2. Initialization ---
    h_orig, w_orig = images[0].shape[:2]
    # If a single image is too large (any side >= 2048), force the target width used for downsampling to 1024
    max_dim = max(h_orig, w_orig)
    if max_dim >= 2048:
        try:
            prev_down = downscale_width
        except NameError:
            prev_down = None
        downscale_width = 1024
        print(f"[Registration][ECC] Large image detected ({h_orig}x{w_orig}), setting downscale_width {prev_down} -> {downscale_width}")
    
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
        scale = downscale_width / float(w) if w > downscale_width else 1.0
        if scale < 1.0:
            small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_img = img # just reference, no copy needed
        
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, scale

    print(f"Aligning {len(images)} images using ECC (Parallel Optimized)...")

    # --- 3. Parallel preprocessing ---
    # print("  - Step 1/3: Preprocessing images concurrently...")
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map guarantees the result order matches the input
        preprocessed_data = list(executor.map(preprocess, images))

    # All images share the same width, so the first image's scale applies to all pairs
    _, scale_factor = preprocessed_data[0]

    # Create the output directory in advance
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # --- 4. Pairwise matrix computation ---
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
        print(f"    ECC: computing {len(images) - 1} pair matrices in parallel ({max_workers} workers)", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    # --- 5. Apply transforms and crop in parallel ---
    # print("  - Step 3/3: Warping and saving concurrently...")

    # Compute the common valid region
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h_orig, w_orig))
    
    do_crop = True
    if top >= bottom or left >= right:
        print("Warning: Invalid crop region, skipping crop.")
        do_crop = False
        target_w, target_h = w_orig, h_orig
        offset_x, offset_y = 0, 0
    else:
        target_w = right - left
        target_h = bottom - top
        offset_x = -left
        offset_y = -top
        # print(f"    Optimized: Warping directly to cropped region ({target_w}x{target_h})...")

    # Build the crop translation matrix
    T_crop = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    # Try to import cupy
    try:
        import cupy as cp
        import cupyx.scipy.ndimage
        HAS_CUPY = True
        print("    [Info] Cupy detected. GPU acceleration enabled for warping.")
    except ImportError:
        HAS_CUPY = False
        print("    [Info] Cupy not found. Using CPU for warping.")

    def warp_task(args):
        idx, img, H = args
        
        # Merge the crop transform: first apply H, then translate by T_crop
        if do_crop:
            H_final = T_crop @ H
        else:
            H_final = H
            
        # If Cupy is available, use GPU acceleration
        if HAS_CUPY:
            # Transfer the image to the GPU
            img_gpu = cp.asarray(img)
            
            # Cupy's affine_transform requires the inverse transform matrix
            # cv2.warpPerspective uses H_final (the inverse of the forward mapping matrix, i.e. from destination to source)
            # but ndimage.affine_transform also needs a matrix mapping from output coordinates back to input coordinates
            # Note: ndimage.affine_transform may define the matrix differently from OpenCV
            # OpenCV: dst(x,y) = src(M * [x,y,1])
            # ndimage: output[i, j] = input[matrix @ [i,j] + offset]
            
            # For perspective transforms (Homography), affine_transform is insufficient because it only supports affine transforms
            # We need to build the coordinate grid manually and use map_coordinates
            
            # Create the target grid
            y_grid, x_grid = cp.meshgrid(cp.arange(target_h), cp.arange(target_w), indexing='ij')
            
            # Flatten the grid
            ones = cp.ones_like(x_grid)
            coords = cp.stack([x_grid, y_grid, ones]) # 3 x N
            coords = coords.reshape(3, -1)
            
            # Apply the transform matrix (H_final is already H_inv, i.e. the mapping from destination to source)
            # src_coords = H_final @ dst_coords
            H_gpu = cp.asarray(H_final)
            src_coords_homo = cp.matmul(H_gpu, coords)
            
            # Normalize the homogeneous coordinates
            w_coords = src_coords_homo[2, :]
            w_coords = cp.where(cp.abs(w_coords) < 1e-10, 1e-10, w_coords)
            src_x = src_coords_homo[0, :] / w_coords
            src_y = src_coords_homo[1, :] / w_coords
            
            # Reshape back to the image shape
            src_x = src_x.reshape(target_h, target_w)
            src_y = src_y.reshape(target_h, target_w)
            
            # Interpolate each channel
            channels = []
            for c in range(img.shape[2]):
                # order=1 (linear) is fastest, order=3 (cubic) gives better quality
                # Here we use order=1 for maximum speed; use order=3 if quality is preferred
                channel_out = cupyx.scipy.ndimage.map_coordinates(
                    img_gpu[:, :, c], 
                    cp.stack([src_y, src_x]), 
                    order=1, 
                    mode='constant', 
                    cval=0
                )
                channels.append(channel_out)
            
            aligned_img_gpu = cp.stack(channels, axis=2)
            
            # Transfer back to the CPU
            aligned_img = cp.asnumpy(aligned_img_gpu).astype(np.uint8)
            
        else:
            # CPU version (OpenCV)
            aligned_img = cv2.warpPerspective(
                img,
                H_final,
                (target_w, target_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT
            )
        
        # If an output path is provided, save directly in the thread
        if output_path:
            fname = img_filenames[idx] if img_filenames else f'frame_{idx:04d}.png'
            cv2.imwrite(os.path.join(output_path, fname), aligned_img)
            
        return aligned_img

    # Prepare the parameters
    task_args = []
    for i in range(len(images)):
        task_args.append((i, images[i], H_matrices[i]))

    # If Cupy is available, do not use multithreading, since GPU operations are already parallel and limited by PCIe bandwidth
    # Multiple threads transferring data to the GPU simultaneously may cause contention
    if HAS_CUPY:
        aligned_images = []
        for args in task_args:
            aligned_images.append(warp_task(args))
    else:
        # Keep using multithreading in CPU mode
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            aligned_images = list(executor.map(warp_task, task_args))

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
        images = [cv2.imread(path) for path in img_paths]
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
    - 'homography': homography alignment registration (non-linear)
    - 'ecc': ECC alignment registration (high precision, sub-pixel level)
    - 'both': combined registration (Homography first, then ECC)
    
    Example:
        # Use homography alignment
        registration = ImageRegistration(method='homography')
        result = registration.process('./images', './output')
        
        # Use ECC alignment
        registration = ImageRegistration(method='ecc')
        result = registration.process(image_list, './output')

        # Use combined alignment
        registration = ImageRegistration(method='both')
        result = registration.process(image_list, './output')
    """
    
    SUPPORTED_METHODS = ['homography', 'ecc', 'both']
    
    def __init__(self, method: str = 'homography', downscale_width: int = 1024, ecc_parallel: bool = True):
        """
        Initialize the registrar

        Args:
            method (str): registration method name, one of 'homography', 'ecc', 'both'
            ecc_parallel (bool): Compute ECC pair matrices concurrently (identical results, faster)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported registration method: {method}. "
                f"Supported methods: {', '.join(self.SUPPORTED_METHODS)}"
            )

        self.method = method
        # User-configurable downsampling width, used in preprocessing stages such as feature extraction
        self.downscale_width = int(downscale_width) if downscale_width is not None else 1024
        self.ecc_parallel = bool(ecc_parallel)
    
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
        if self.method == 'homography':
            return self._process_homography(input_source, output_path, thread_count=thread_count)
        elif self.method == 'ecc':
            return self._process_ecc(input_source, output_path, thread_count=thread_count)
        elif self.method == 'both':
            # Combined mode: Homography first, then ECC
            # Step 1: Homography (do not save intermediate results, unless this is the only step)
            print("=== Step 1: Homography Alignment ===")
            # In both mode, step 1 does not need to save to output_path, only passing in memory
            homography_result = self._process_homography(input_source, output_path=None, thread_count=thread_count)
            
            print("\n=== Step 2: ECC Alignment ===")
            # Step 2: ECC (save the final result)
            return self._process_ecc(homography_result, output_path, thread_count=thread_count)
    
    def _process_homography(self, 
                           input_source: Union[str, List[np.ndarray]], 
                           output_path: Optional[str] = None,
                           thread_count: int = 4) -> List[np.ndarray]:
        """
        Homography alignment registration (non-linear)
        
        Args:
            input_source: image source
            output_path: output path
        
        Returns:
            list of registered images
        """
        return _align_homography_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count)
    
    def _process_ecc(self, 
                    input_source: Union[str, List[np.ndarray]], 
                    output_path: Optional[str] = None,
                    thread_count: int = 4) -> List[np.ndarray]:
        """
        ECC alignment registration (high precision, sub-pixel level)
        
        Args:
            input_source: image source
            output_path: output path
        
        Returns:
            list of registered images
        """
        return _align_ecc_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count,
                               parallel_ecc=self.ecc_parallel)
    
    def set_method(self, method: str):
        """
        Switch the registration method
        
        Args:
            method (str): the new registration method name
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported registration method: {method}. "
                f"Supported methods: {', '.join(self.SUPPORTED_METHODS)}"
            )
        
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
        method: registration method ('homography', 'ecc', 'both')
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
    parser.add_argument('--mode', default='homography', choices=['homography', 'ecc', 'both'],
                        help='Processing mode: homography, ecc, both (homography + ecc)')
    
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