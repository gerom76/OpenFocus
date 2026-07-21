import os
import glob
import re
import time
import cv2
import numpy as np
import concurrent.futures

# Reference:
# https://github.com/RCharradi/Image-fusion-with-guided-filtering
# Li S, Kang X, Hu J. Image fusion with guided filtering[J]. IEEE Transactions on Image processing, 2013, 22(7): 2864-2875.

def gff_impl(input_source, img_resize, kernel_size=31, thread_count: int = None):
    """
    Guided-filter-based multi-focus image-stack fusion algorithm implementation (CPU version)
    Does not depend on opencv-contrib (ximgproc); includes a built-in guided-filter implementation.
    """
    # Determine thread pool size: if thread_count is provided use it, else use default
    if thread_count is None:
        max_workers = None
    else:
        try:
            max_workers = max(1, int(thread_count))
        except Exception:
            max_workers = None
    
    # ========== Parameter settings ==========
    # Default parameters (referencing the original script)
    DEFAULT_R1 = 45
    DEFAULT_R2 = 7
    DEFAULT_EPS1 = 0.3
    DEFAULT_EPS2 = 10e-6
    DEFAULT_SIGMA_R = 5
    
    # If a valid kernel_size is passed in, use it as the mean-filter size
    average_filter_size = kernel_size if kernel_size is not None and kernel_size > 0 else 31
    
    # Ensure the filter size is odd
    if average_filter_size % 2 == 0:
        average_filter_size += 1

    # ========== Built-in guided-filter implementation ==========
    def guided_filter(I, p, r, eps):
        """
        Fast guided-filter implementation (based on Box Filter)
        I: guide image (Guide Image), single- or three-channel
        p: input image (Input Image), single-channel
        r: filter radius
        eps: regularization parameter
        """
        # Ensure I and p have the same type
        if I.dtype != np.float32:
            I = I.astype(np.float32)
        if p.dtype != np.float32:
            p = p.astype(np.float32)
            
        # Filter diameter
        ksize = (2 * r + 1, 2 * r + 1)
        
        # Mean computation
        mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
        
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
        
        q = mean_a * I + mean_b
        return q

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
            
        stack_ori = [cv2.imread(img_path) for img_path in img_stack_path_list]
    else:
        stack_ori = input_source
    
    if not stack_ori:
        raise ValueError("No image data was loaded")

    if img_resize:
        stack_ori = [cv2.resize(img, img_resize) for img in stack_ori]

    # Convert to float32 and normalize to [0, 1]
    stack_flt = [img.astype(np.float32) / 255.0 for img in stack_ori]
    
    # ========== Core algorithm implementation ==========
    
    def guided_filter_fusion_stack(images):
        """
        Perform guided-filter-based fusion on the image stack
        """
        num_images = len(images)
        if num_images == 0:
            return None
        
        rows, cols, channels = images[0].shape
        
        # 1. Image decomposition (Base Layer & Detail Layer)
        # Use uniform_filter to extract the Base layer
        
        def process_decompose(img):
            # Use cv2.blur instead of uniform_filter for better speed
            base = cv2.blur(img, (average_filter_size, average_filter_size), borderType=cv2.BORDER_REFLECT)
            detail = img - base
            return base, detail

        # Process image decomposition in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            decompose_results = list(executor.map(process_decompose, images))
        
        base_layers = [res[0] for res in decompose_results]
        detail_layers = [res[1] for res in decompose_results]
            
        # 2. Compute the saliency map (Saliency Map)
        # Saliency = Gaussian(abs(Laplacian(Sum_Channels)))
        
        def process_saliency(img):
            # Sum the three channels for the Laplacian computation
            img_sum = np.sum(img, axis=2)
            # Use cv2.Laplacian instead of scipy.ndimage.laplace
            lap = np.abs(cv2.Laplacian(img_sum, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT))
            # Use cv2.GaussianBlur instead of scipy.ndimage.gaussian_filter
            sal = cv2.GaussianBlur(lap, (0, 0), sigmaX=DEFAULT_SIGMA_R, sigmaY=DEFAULT_SIGMA_R, borderType=cv2.BORDER_REFLECT)
            return sal

        # Compute the saliency map in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            saliency_maps = list(executor.map(process_saliency, images))
            
        # 3. Generate the initial decision map (Decision Map)
        # At each pixel, select the index of the image with the highest saliency
        saliency_stack = np.stack(saliency_maps, axis=0) # (N, H, W)
        max_indices = np.argmax(saliency_stack, axis=0)  # (H, W)
        
        # 4. Weight-map refinement and fusion (Weight Refinement & Fusion)
        # Initialize the accumulators
        fused_base_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
        fused_base_denominator = np.zeros((rows, cols), dtype=np.float32) # optimized to single channel
        
        fused_detail_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
        fused_detail_denominator = np.zeros((rows, cols), dtype=np.float32) # optimized to single channel
        
        def process_weight_fusion(k):
            # Generate the binary mask for the k-th image
            mask_k = (max_indices == k).astype(np.float32) # (H, W)
            
            img_k = images[k] # Guide image (H, W, 3)
            
            # Refine the mask using the built-in guided filter
            img_k_gray = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)
            
            # Precompute I*p and I*I to avoid recomputing them across the two guided_filter calls
            I = img_k_gray
            p = mask_k
            Ip = I * p
            I2 = I * I
            
            # Internal optimized guided_filter that reuses the precomputed results
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
            weight_base_k = run_gf(DEFAULT_R1, DEFAULT_EPS1)
            
            # For the Detail layer
            weight_detail_k = run_gf(DEFAULT_R2, DEFAULT_EPS2)

            # Optimization: no longer use np.repeat to expand to 3 channels, use broadcasting instead
            # weight_base_k and weight_detail_k are both (H, W)
            
            # Compute the numerator term (H, W, 3) * (H, W, 1) -> (H, W, 3)
            base_num = base_layers[k] * weight_base_k[:, :, np.newaxis]
            detail_num = detail_layers[k] * weight_detail_k[:, :, np.newaxis]
            
            # The denominator term can just return the single-channel weight
            return base_num, weight_base_k, detail_num, weight_detail_k

        # Process weight computation and fusion in parallel
        # Use the as_completed pattern, accumulating each result as it finishes, to avoid holding all results at once and blowing up memory
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_weight_fusion, k): k for k in range(num_images)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    bn, bd, dn, dd = future.result()
                    fused_base_numerator += bn
                    fused_base_denominator += bd
                    fused_detail_numerator += dn
                    fused_detail_denominator += dd
                    # Explicitly delete references to help garbage collection
                    del bn, bd, dn, dd
                except Exception as exc:
                    print(f'Image {futures[future]} generated an exception: {exc}')

            
        # 5. Reconstruct the image
        # Avoid division by zero
        fused_base_denominator[fused_base_denominator < 1e-6] = 1e-6
        fused_detail_denominator[fused_detail_denominator < 1e-6] = 1e-6
        
        # Broadcast division (H, W, 3) / (H, W, 1)
        fused_base = fused_base_numerator / fused_base_denominator[:, :, np.newaxis]
        fused_detail = fused_detail_numerator / fused_detail_denominator[:, :, np.newaxis]
        
        fused_img = fused_base + fused_detail
        
        # Clip and convert back to uint8
        fused_img = np.rint(np.clip(fused_img * 255, 0, 255)).astype(np.uint8)
        
        return fused_img

    return guided_filter_fusion_stack(stack_flt)