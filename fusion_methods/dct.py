# Reference:
# Haghighat M B A, Aghagolzadeh A, Seyedarabi H. Multi-focus image fusion for visual sensor networks in DCT domain[J]. Computers & Electrical Engineering, 2011, 37(5): 789-797.
import glob
import os
import time
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np

ArraySource = Sequence[np.ndarray]

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
        img = cv2.imread(path, cv2.IMREAD_COLOR)
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

def dct_focus_stack_fusion(
    source: Union[str, ArraySource],
    output_path: str = None,
    block_size: int = 8,
    kernel_size: int = 7,
) -> np.ndarray:
    """
    Highly optimized DCT/variance image-fusion algorithm.
    
    Optimization notes:
    By Parseval's theorem, the high-frequency energy (variance) in the DCT domain is equivalent to the pixel variance in the spatial domain.
    cv2.resize(INTER_AREA) is used to quickly compute block means and squared means, replacing the originally extremely slow
    per-block DCT loop.
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

    # --- 2. Quickly compute the variance map (core optimization) ---
    # Preallocate space
    max_variance_map = np.full((map_h, map_w), -1.0, dtype=np.float32)
    # Use a smaller data type to store indices, saving memory
    idx_dtype = np.uint8 if len(images) < 256 else np.int32
    best_index_map = np.zeros((map_h, map_w), dtype=idx_dtype)

    for idx, bgr_img in enumerate(normalized_images):
        # Crop the edges to match the block tiling
        img_trim = bgr_img[:h_trim, :w_trim]
        
        # Convert to grayscale and to float32 to prevent squaring overflow
        gray = cv2.cvtColor(img_trim, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 1. Compute E[X^2] (mean of squares)
        # cv2.resize with INTER_AREA effectively does block averaging, which is very fast
        mean_sq = cv2.resize(gray ** 2, (map_w, map_h), interpolation=cv2.INTER_AREA)
        
        # 2. Compute (E[X])^2 (square of the mean)
        mean_val = cv2.resize(gray, (map_w, map_h), interpolation=cv2.INTER_AREA)
        sq_mean = mean_val ** 2
        
        # 3. Variance Var(X) = E[X^2] - (E[X])^2
        # This is mathematically strictly equivalent to the energy sum of the DCT AC components
        var_map = mean_sq - sq_mean
        
        # Update the maximum-variance map
        mask = var_map > max_variance_map
        max_variance_map[mask] = var_map[mask]
        best_index_map[mask] = idx

    # --- 3. Consistency verification (median filtering) ---
    # Must convert back to a type suitable for filtering; uint8 would also work, but convert for robustness
    if idx_dtype == np.uint8:
        map_to_filter = best_index_map
    else:
        map_to_filter = best_index_map.astype(np.float32)

    # Two passes of median filtering to remove noise
    filtered_map = cv2.medianBlur(map_to_filter, kernel_size)
    filtered_map = cv2.medianBlur(filtered_map, kernel_size)
    
    # Convert back to integer indices
    if filtered_map.dtype != np.int32 and filtered_map.dtype != np.uint8:
        final_index_map = filtered_map.astype(np.int32)
    else:
        final_index_map = filtered_map

    # --- 4. Fast reconstruction ---
    # Scale the small index map back up to the original size in one go (Nearest Neighbor)
    full_size_indices = cv2.resize(
        final_index_map.astype(np.uint8), # resize is fastest on uint8
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

    fused_image = np.zeros((h, w, 3), dtype=np.uint8)
    
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