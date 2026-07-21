# Conditional import, to avoid errors when it is not needed
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

import os
import re
import glob
import numpy as np
import cv2

# ================= Global cache variables =================
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None

# Cache the available accelerator type (detected once at module load)
_MPS_AVAILABLE = None
_CUDA_AVAILABLE = None

# The encoder pools its input repeatedly, so a small enough image collapses to a
# zero-sized feature map partway down and torch raises from max_pool2d with a
# tensor shape rather than anything actionable. 112 px is the smallest edge that
# survives; below it we say so ourselves.
MIN_INPUT_EDGE = 112


def _check_input_size(height, width):
    """Reject inputs the network cannot pool down, with an explanation."""
    if min(height, width) < MIN_INPUT_EDGE:
        raise ValueError(
            f"StackMFF-V4 needs at least {MIN_INPUT_EDGE} px on each side, "
            f"got {width}x{height}. Increase the output size, raise the tile "
            f"block size, or pick another fusion method for images this small."
        )


def _detect_accelerators():
    """Detect the available accelerator type on the system, run once at module load"""
    global _MPS_AVAILABLE, _CUDA_AVAILABLE
    try:
        import torch
        _MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        _CUDA_AVAILABLE = torch.cuda.is_available()
    except Exception:
        _MPS_AVAILABLE = False
        _CUDA_AVAILABLE = False


_detect_accelerators()


def _get_model_and_device(model_path, use_gpu):
    """
    Get or initialize the global model and device
    
    Args:
        model_path: path to the model weights file
        use_gpu: whether to use the GPU
    
    Returns:
        (model, device) tuple
    """
    if torch is None or F is None:
        raise ImportError("PyTorch not installed")
    from core.models.stackmffv4_network import StackMFF_V4

    if use_gpu and _MPS_AVAILABLE:
        device = torch.device('mps')
    elif use_gpu and _CUDA_AVAILABLE:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    global _GLOBAL_MODEL, _GLOBAL_DEVICE

    if _GLOBAL_MODEL is None:
        print("Loading StackMFF-V4 model (once)...")
        model = StackMFF_V4()
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _GLOBAL_MODEL = model
        _GLOBAL_DEVICE = device
    else:
        model = _GLOBAL_MODEL
        if _GLOBAL_DEVICE != device:
            model.to(device)
            _GLOBAL_DEVICE = device
    
    return model, device


def _resize_to_multiple_of_32(image):
    """Resize the image to a multiple of 32"""
    h, w = image.shape[-2:]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    if new_h == h and new_w == w:
        return image, (h, w)
    resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized, (h, w)


def _stackmffv4_batch_impl(tiles_list, model_path, use_gpu):
    """
    Batch StackMFF-V4 fusion over multiple tiles
    
    Args:
        tiles_list: list where each element is the image list of one tile [(img1, img2, ...), (img1, img2, ...), ...]
                    each image is a BGR-format numpy array
        model_path: path to the model weights file
        use_gpu: whether to use the GPU
    
    Returns:
        list of fused images, each a BGR-format uint8 numpy array
    """
    if torch is None or F is None:
        raise ImportError("PyTorch not installed")
    
    if not tiles_list:
        return []
    
    model, device = _get_model_and_device(model_path, use_gpu)
    
    batch_size = len(tiles_list)
    
    # Prepare the data for all tiles
    all_color_images = []  # RGB images of shape [batch][num_images]
    all_gray_stacks = []   # torch tensor stacks of shape [batch]
    all_original_sizes = []
    num_images_per_tile = None
    
    for tile_idx, tile_images in enumerate(tiles_list):
        bgr_images = list(tile_images)
        if not bgr_images:
            raise ValueError(f"Empty image list for tile {tile_idx}")
        
        if num_images_per_tile is None:
            num_images_per_tile = len(bgr_images)
        elif len(bgr_images) != num_images_per_tile:
            raise ValueError(f"Inconsistent number of images: tile {tile_idx} has {len(bgr_images)}, expected {num_images_per_tile}")
        
        color_images = []
        gray_tensors = []
        
        for img in bgr_images:
            color_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
            gray_tensors.append(gray_tensor)
        
        all_color_images.append(color_images)
        image_stack = torch.stack(gray_tensors)  # [num_images, H, W]
        all_original_sizes.append(image_stack.shape[-2:])
        all_gray_stacks.append(image_stack)
    
    # Find the maximum size and pad all tiles to the same size
    max_h = max(s[0] for s in all_original_sizes)
    max_w = max(s[1] for s in all_original_sizes)

    # Every tile is padded up to this, so the batch stands or falls together
    _check_input_size(int(max_h), int(max_w))

    # Pad to a multiple of 32
    padded_h = ((max_h - 1) // 32 + 1) * 32
    padded_w = ((max_w - 1) // 32 + 1) * 32
    
    # Create the batched input tensor [batch, num_images, padded_h, padded_w]
    batch_input = torch.zeros(batch_size, num_images_per_tile, padded_h, padded_w)
    
    for i, (stack, (orig_h, orig_w)) in enumerate(zip(all_gray_stacks, all_original_sizes)):
        batch_input[i, :, :orig_h, :orig_w] = stack
    
    # Run batched inference
    with torch.no_grad():
        batch_input = batch_input.to(device)
        _, focus_indices_batch = model(batch_input)
        focus_indices_batch = focus_indices_batch.cpu().numpy()  # [batch, padded_h, padded_w]
        del batch_input
    
    # Process the output of each tile
    results = []
    for tile_idx in range(batch_size):
        focus_indices = focus_indices_batch[tile_idx]
        orig_h, orig_w = all_original_sizes[tile_idx]
        
        # Crop back to the original size
        focus_map = focus_indices[:orig_h, :orig_w]
        focus_map = np.clip(focus_map.astype(int), 0, num_images_per_tile - 1)
        
        color_images = all_color_images[tile_idx]
        color_array = np.stack(color_images, axis=0)
        fused_color = color_array[focus_map, np.arange(orig_h)[:, None], np.arange(orig_w)]
        fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)
        results.append(fused_color_bgr)
    
    return results


def _stackmffv4_impl(input_source, img_resize, model_path, use_gpu):
    """
    Image-fusion algorithm based on the StackMFF-V4 neural network
    
    Args:
        input_source: image source (directory path or list of images)
        img_resize: target size (width, height)
        model_path: path to the model weights file
        use_gpu: whether to use the GPU
    
    Returns:
        the fused image (BGR format, uint8)
    """
    if torch is None or F is None:
        raise ImportError("PyTorch not installed")

    model, device = _get_model_and_device(model_path, use_gpu)

    device_name = device.type.upper()
    if device.type == 'cuda':
        device_name = 'CUDA'

    print(f"Running AI fusion on {device_name}...")

    color_images = []
    gray_tensors = []

    if isinstance(input_source, str):
        def get_image_suffix(input_stack_path):
            filenames = os.listdir(input_stack_path)
            if len(filenames) == 0:
                return None
            suffixes = [os.path.splitext(filename)[1] for filename in filenames]
            return suffixes[0]

        img_ext = get_image_suffix(input_source)
        glob_format = '*' + img_ext
        img_stack_path_list = glob.glob(os.path.join(input_source, glob_format))
        img_stack_path_list.sort(
            key=lambda x: int(str(re.findall(r"\d+", x.split(os.sep)[-1])[-1])))

        for img_path in img_stack_path_list:
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                raise ValueError(f"Failed to read image: {img_path}")
            if img_resize:
                bgr_img = cv2.resize(bgr_img, img_resize)
            color_images.append(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
            gray_tensors.append(gray_tensor)
    else:
        bgr_images = list(input_source)
        if not bgr_images:
            raise ValueError("Empty image list")
        if img_resize:
            bgr_images = [cv2.resize(img, img_resize) for img in bgr_images]
        for img in bgr_images:
            color_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
            gray_tensors.append(gray_tensor)
    
    num_images = len(color_images)
    if num_images < 2:
        raise ValueError("At least two images are required for fusion")

    print(f"Loaded {num_images} images")

    image_stack = torch.stack(gray_tensors)
    original_size = image_stack.shape[-2:]
    _check_input_size(int(original_size[0]), int(original_size[1]))

    with torch.no_grad():
        input_tensor = image_stack.unsqueeze(0).to(device)
        resized_input, _ = _resize_to_multiple_of_32(input_tensor)

        _, focus_indices = model(resized_input)

        focus_indices = focus_indices.squeeze().cpu().numpy()

        del input_tensor, resized_input

    h, w = original_size
    focus_map = cv2.resize(
        focus_indices.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    ).astype(int)

    focus_map = np.clip(focus_map, 0, num_images - 1)
    color_array = np.stack(color_images, axis=0)
    fused_color = color_array[focus_map, np.arange(h)[:, None], np.arange(w)]
    fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return fused_color_bgr