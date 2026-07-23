import concurrent.futures
import importlib
import importlib.util
import os
import numpy as np
from typing import Union, List, Tuple, Optional
from fusion_methods.dct import dct_focus_stack_fusion
from fusion_methods.gff import gff_impl
from fusion_methods.pyramid import pyramid_impl
from fusion_methods.depthmap import depthmap_impl, MODE_MAX, MODE_AVERAGE
from fusion_methods.stackmffv4 import _stackmffv4_impl, _stackmffv4_batch_impl
from fusion_methods.dtcwt import _dtcwt_impl
from utils import resource_path, bitdepth
from utils.image_utils import read_image_any_depth


# Tile parameters are now instance attributes of MultiFocusFusion; see the
# constructor defaults and accessor methods.

# Module-level defaults for backwards compatibility and convenience
_DEFAULT_TILE_ENABLED = True
_DEFAULT_TILE_BLOCK_SIZE = 1024
_DEFAULT_TILE_OVERLAP = 256
_DEFAULT_TILE_THRESHOLD = 2048
_DEFAULT_STACKMFFV4_BATCH_SIZE = 2


def set_tile_mode(enabled: bool):
    """Compatibility helper: set module-level default tile enabled flag.

    This was previously a module-level API in older versions. Prefer using
    MultiFocusFusion.set_tile_mode or passing tile parameters to fuse_images.
    """
    global _DEFAULT_TILE_ENABLED
    _DEFAULT_TILE_ENABLED = bool(enabled)


def set_tile_params(block_size: Optional[int] = None,
                    overlap: Optional[int] = None,
                    threshold: Optional[int] = None):
    """Compatibility helper: set module-level default tile parameters.

    Passing None leaves the parameter unchanged.
    """
    global _DEFAULT_TILE_BLOCK_SIZE, _DEFAULT_TILE_OVERLAP, _DEFAULT_TILE_THRESHOLD
    if block_size is not None:
        _DEFAULT_TILE_BLOCK_SIZE = max(1, int(block_size))
    if overlap is not None:
        _DEFAULT_TILE_OVERLAP = max(0, int(overlap))
    if threshold is not None:
        _DEFAULT_TILE_THRESHOLD = max(1, int(threshold))


def get_tile_params() -> dict:
    return {
        'tile_enabled': bool(_DEFAULT_TILE_ENABLED),
        'tile_block_size': int(_DEFAULT_TILE_BLOCK_SIZE),
        'tile_overlap': int(_DEFAULT_TILE_OVERLAP),
        'tile_threshold': int(_DEFAULT_TILE_THRESHOLD),
    }


def is_stackmffv4_available() -> bool:
    """Return True when PyTorch is importable for the StackMFF-V4 fusion."""
    torch_spec = importlib.util.find_spec("torch")
    if not torch_spec:
        return False

    try:
        importlib.import_module("torch")
    except ImportError:
        return False

    return True

class MultiFocusFusion:
    """
    Unified interface for multi-focus image fusion.

    Supported algorithms:
    - 'guided_filter': guided-filter fusion
    - 'dct': DCT-variance consistency fusion
    - 'dtcwt': dual-tree complex wavelet fusion
    - 'stackmffv4': StackMFF-V4 neural network fusion
    """
    
    SUPPORTED_ALGORITHMS = ['guided_filter', 'dct', 'dtcwt', 'gfgfgf', 'pyramid',
                            'depthmap_max', 'depthmap_average', 'stackmffv4']
    
    def __init__(self, algorithm: str = 'guided_filter', use_gpu: bool = False,
                 tile_enabled: bool = True, tile_block_size: int = 1024,
                 tile_overlap: int = 256, tile_threshold: int = 2048,
                 stackmffv4_batch_size: int = 2):
        """
        Initialize the fusion engine.

        Args:
            algorithm (str): Fusion algorithm name; one of 'guided_filter', 'dct', 'dtcwt', 'stackmffv4'
            use_gpu (bool): Whether to use GPU acceleration (CUDA/MPS), default False
            stackmffv4_batch_size (int): StackMFF V4 batch size, default 2
        """
        self._ensure_supported_algorithm(algorithm)
        self.algorithm = algorithm
        self.use_gpu = bool(use_gpu)
        # Tile (tiled fusion) related instance-level settings
        # Tiled fusion is used when tile_enabled is True and the image's longest side exceeds tile_threshold
        self.tile_enabled = bool(tile_enabled)
        self.tile_block_size = int(tile_block_size)
        self.tile_overlap = int(tile_overlap)
        self.tile_threshold = int(tile_threshold)
        # StackMFF V4 batch size
        self.stackmffv4_batch_size = max(1, int(stackmffv4_batch_size))
        self._validate_environment()
    
    def _ensure_supported_algorithm(self, algorithm: str) -> None:
        """Ensure the algorithm is supported."""
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

    def _validate_environment(self):
        """Validate the runtime environment."""
        if self.algorithm == 'dtcwt':
            self._validate_transform_environment()
        elif self.algorithm == 'dct':
            self._validate_dct_environment()
        elif self.algorithm == 'guided_filter':
            self._validate_spatial_environment()
        elif self.algorithm == 'stackmffv4':
            self._validate_ai_environment()

    def _fuse_gfgfgf(self,
                     input_source: Union[str, List[np.ndarray]],
                     img_resize: Optional[Tuple[int, int]] = None,
                     kernel_size: int = 7,
                     **kwargs) -> np.ndarray:
        """
        GFG-FGF fusion interface.

        Args:
            input_source: Image source
            img_resize: Target size
            kernel_size: Kernel size for the initial mean/blur filter (odd preferred), default 7

        Returns:
            Fused image
        """
        try:
            from fusion_methods.gfg_fgf import gfgfgf_impl
        except Exception as exc:
            raise RuntimeError("GFG-FGF fusion requires the gfg_fgf module.") from exc

        kernel_size = max(1, int(kernel_size or 31))
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)

        if self.use_gpu:
            try:
                from fusion_methods.gfg_fgf_torch import gfgfgf_torch_impl
                return gfgfgf_torch_impl(input_source, img_resize, kernel_size=kernel_size)
            except Exception as exc:
                print(f"Warning: GPU GFG-FGF fusion failed ({exc}); falling back to CPU.")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        thread_count = kwargs.get('thread_count', None)
        return gfgfgf_impl(input_source, img_resize, kernel_size=kernel_size, thread_count=thread_count)

    def _fuse_pyramid(self,
                      input_source: Union[str, List[np.ndarray]],
                      img_resize: Optional[Tuple[int, int]] = None,
                      levels: Optional[int] = None,
                      **kwargs) -> np.ndarray:
        """
        Laplacian-pyramid fusion.

        Args:
            input_source: Image source
            img_resize: Target size
            levels: Number of pyramid decomposition levels (None -> method default)

        Returns:
            Fused image
        """
        # CPU-only; there is no GPU implementation, so use_gpu is ignored here.
        # kernel_size may arrive from the shared worker call path - the pyramid
        # has no kernel, so it is quietly dropped along with any other extras.
        thread_count = kwargs.get('thread_count', None)
        return pyramid_impl(input_source, img_resize, levels=levels, thread_count=thread_count)

    def _fuse_depthmap(self,
                       input_source: Union[str, List[np.ndarray]],
                       img_resize: Optional[Tuple[int, int]] = None,
                       mode: str = MODE_MAX,
                       kernel_size: Optional[int] = None,
                       **kwargs) -> np.ndarray:
        """
        Depth-map fusion (per-pixel select or contrast-weighted average).

        Args:
            input_source: Image source
            img_resize: Target size
            mode: 'max' for hard per-pixel select, 'average' for the
                  contrast-weighted average
            kernel_size: Side of the focus-measure pooling window (odd)

        Returns:
            Fused image
        """
        # CPU-only; there is no GPU implementation, so use_gpu is ignored here.
        thread_count = kwargs.get('thread_count', None)
        return depthmap_impl(input_source, img_resize, mode=mode,
                             kernel_size=kernel_size, thread_count=thread_count)

    def _validate_dct_environment(self) -> None:
        """Validate DCT fusion dependencies."""
        try:
            import cv2  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependency
            raise RuntimeError(
                "DCT fusion requires OpenCV. Install it with: pip install opencv-python"
            ) from exc

        if self.use_gpu:
            try:
                import torch
                has_gpu = torch.cuda.is_available() or (
                    hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                )
                if not has_gpu:
                    print("Note: No GPU acceleration available (CUDA/MPS); DCT fusion will run on CPU.")
            except ImportError:
                has_gpu = False
                print("Note: PyTorch not installed; DCT fusion will run on CPU.")
            if not has_gpu:
                self.use_gpu = False

    def _validate_transform_environment(self) -> None:
        """Validate transform-domain fusion dependencies."""
        gpu_ready = False
        if self.use_gpu:
            # Check availability without executing the imports (pytorch_wavelets
            # pulls in pkg_resources, which is shimmed only inside dtcwt_torch).
            have_wavelets = (
                importlib.util.find_spec("pytorch_wavelets") is not None
                and importlib.util.find_spec("pywt") is not None
            )
            try:
                import torch
                has_device = torch.cuda.is_available() or (
                    hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                )
            except ImportError:
                has_device = False

            gpu_ready = have_wavelets and has_device
            if not gpu_ready:
                if not have_wavelets:
                    print("Note: GPU DTCWT fusion requires pytorch_wavelets and PyWavelets; falling back to CPU.")
                else:
                    print("Note: No GPU acceleration available (CUDA/MPS); DTCWT fusion will run on CPU.")
                self.use_gpu = False

        if not gpu_ready:
            # The CPU path requires the dtcwt package
            dtcwt_spec = importlib.util.find_spec("dtcwt")
            if dtcwt_spec:
                importlib.import_module("dtcwt")
            else:
                raise RuntimeError(
                    "DTCWT fusion requires the dtcwt package on CPU. Install it with: pip install dtcwt scipy"
                )

    def _validate_spatial_environment(self) -> None:
        """Validate spatial-domain fusion dependencies."""
        if self.use_gpu:
            try:
                import torch
                has_gpu = torch.cuda.is_available() or (
                    hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                )
                if not has_gpu:
                    print("Note: No GPU acceleration available (CUDA/MPS); guided-filter fusion will run on CPU.")
            except ImportError:
                has_gpu = False
                print("Note: PyTorch not installed; guided-filter fusion will run on CPU.")
            if not has_gpu:
                self.use_gpu = False

    def _validate_ai_environment(self) -> None:
        """Validate AI fusion dependencies."""
        if not is_stackmffv4_available():
            raise RuntimeError(
                "StackMFF-V4 fusion requires PyTorch. Install it with: pip install torch torchvision"
            )

        import torch

        if self.use_gpu:
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                print("Warning: No GPU acceleration available (CUDA/MPS). Running StackMFF-V4 on CPU (slower).")
                self.use_gpu = False
    
    def fuse(self, 
             input_source: Union[str, List[np.ndarray]], 
             img_resize: Optional[Tuple[int, int]] = None,
             **kwargs) -> np.ndarray:
        """
        Run image fusion.

        Args:
            input_source (str or list): Image directory path or list of preloaded images
            img_resize (tuple, optional): Target size (width, height)
            **kwargs: Algorithm-specific parameters

                guided_filter parameters:
                    - kernel_size (int): Mean filter kernel size for guided filtering, default 31 (must be odd)
                dct parameters:
                    - block_size (int): DCT block size, default 8
                    - kernel_size (int): Median filter kernel size, default 7 (must be odd)

                dtcwt parameters:
                    - N (int): Number of DTCWT decomposition levels, default 4

                stackmffv4 parameters:
                    - model_path (str): Model weights file path, default './weights/stackmffv4.pth'

        Returns:
            numpy.ndarray: Fused image (uint8)
        """
        # Use tiled fusion when given a list of preloaded images whose size is too large
        try:
            is_list_of_arrays = (
                isinstance(input_source, list)
                and len(input_source) > 0
                and isinstance(input_source[0], np.ndarray)
            )
        except Exception:
            is_list_of_arrays = False

        # Case 1: list of preloaded images
        if is_list_of_arrays:
            h, w = input_source[0].shape[:2]
            if self.tile_enabled and max(h, w) > self.tile_threshold:
                # Use tiled fusion; block size and overlap come from instance attributes
                print(f"Info: Large image size detected ({w}x{h}). Using tiled fusion mode (block={self.tile_block_size}, overlap={self.tile_overlap}).")
                kws = dict(kwargs)
                # Drop caller kwargs that would conflict with the internally specified values
                kws.pop('block_size', None)
                kws.pop('overlap', None)
                return self._fuse_tiled(
                    input_source,
                    algorithm=self.algorithm,
                    img_resize=img_resize,
                    block_size=self.tile_block_size,
                    overlap=self.tile_overlap,
                    **kws,
                )

        # Case 2: directory path input; lazily read only the first image to determine size
        if isinstance(input_source, str) and os.path.isdir(input_source):
            try:
                import cv2  # deferred import
            except Exception as exc:
                raise RuntimeError("OpenCV is required to read image files for tiled fusion. Install with: pip install opencv-python") from exc

            # Common image extensions
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(input_source)) if f.lower().endswith(exts)]
            if len(files) > 0:
                first_path = os.path.join(input_source, files[0])
                first_img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
                if first_img is None:
                    raise RuntimeError(f"Unable to read first image: {first_path}")
                fh, fw = first_img.shape[:2]
                if self.tile_enabled and max(fh, fw) > self.tile_threshold:
                    # Directory input: tiled fusion with lazy per-file loading (only when tile_enabled is on)
                    kws = dict(kwargs)
                    kws.pop('block_size', None)
                    kws.pop('overlap', None)
                    return self._fuse_tiled(
                        input_source,
                        algorithm=self.algorithm,
                        img_resize=img_resize,
                        block_size=self.tile_block_size,
                        overlap=self.tile_overlap,
                        **kws,
                    )

        if self.algorithm == 'guided_filter':
            return self._fuse_guided_filter(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dct':
            return self._fuse_dct(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dtcwt':
            return self._fuse_dtcwt(input_source, img_resize, **kwargs)
        elif self.algorithm == 'gfgfgf':
            return self._fuse_gfgfgf(input_source, img_resize, **kwargs)
        elif self.algorithm == 'pyramid':
            return self._fuse_pyramid(input_source, img_resize, **kwargs)
        elif self.algorithm == 'depthmap_max':
            return self._fuse_depthmap(input_source, img_resize, mode=MODE_MAX, **kwargs)
        elif self.algorithm == 'depthmap_average':
            return self._fuse_depthmap(input_source, img_resize, mode=MODE_AVERAGE, **kwargs)
        elif self.algorithm == 'stackmffv4':
            return self._fuse_stackmffv4(input_source, img_resize, **kwargs)
    
    def _fuse_guided_filter(self, 
                            input_source: Union[str, List[np.ndarray]], 
                            img_resize: Optional[Tuple[int, int]] = None,
                            kernel_size: int = 31,
                            **kwargs) -> np.ndarray:
        """
        Guided-filter fusion.

        Args:
            input_source: Image source
            img_resize: Target size
            kernel_size: Mean filter kernel size used in guided filtering (must be odd)

        Returns:
            Fused image
        """
        kernel_size = max(1, int(kernel_size or 31))
        if kernel_size % 2 == 0:
            kernel_size += 1

        if self.use_gpu:
            try:
                from fusion_methods.gff_torch import gff_torch_impl
                return gff_torch_impl(input_source, img_resize, kernel_size=kernel_size)
            except Exception as exc:
                print(f"Warning: GPU guided-filter fusion failed ({exc}); falling back to CPU.")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        # allow optional thread_count forwarded via kwargs in fuse
        # (handled by fuse caller). Here simply pass through if present in kwargs
        # Note: _fuse_guided_filter is invoked with explicit kernel_size only
        thread_count = kwargs.get('thread_count', None)
        return gff_impl(
            input_source,
            img_resize,
            kernel_size=kernel_size,
            thread_count=thread_count
        )

    def _fuse_dct(self,
                  input_source: Union[str, List[np.ndarray]],
                  img_resize: Optional[Tuple[int, int]] = None,
                  block_size: int = 8,
                  kernel_size: int = 7,
                  **kwargs) -> np.ndarray:
        """
        DCT-variance fusion.

        Args:
            input_source: Image source
            img_resize: Target size (currently unsupported; raises if specified)
            block_size: DCT block size
            kernel_size: Median filter kernel size for consistency verification

        Returns:
            Fused image
        """
        if img_resize is not None:
            raise ValueError("DCT fusion does not support dynamic resizing. Resize images before processing.")

        if self.use_gpu:
            try:
                from fusion_methods.dct_torch import dct_torch_impl
                return dct_torch_impl(input_source, block_size=block_size, kernel_size=kernel_size)
            except Exception as exc:
                print(f"Warning: GPU DCT fusion failed ({exc}); falling back to CPU.")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        # thread_count (if passed) is ignored by this implementation
        return dct_focus_stack_fusion(
            input_source,
            output_path=None,
            block_size=block_size,
            kernel_size=kernel_size
        )
    
    def _fuse_dtcwt(self,
                    input_source: Union[str, List[np.ndarray]],
                    img_resize: Optional[Tuple[int, int]] = None,
                    N: int = 4,
                    **kwargs) -> np.ndarray:
        """
        DTCWT transform-domain fusion.

        Args:
            input_source: Image source
            img_resize: Target size
            N: Number of DTCWT decomposition levels

        Returns:
            Fused image
        """
        if self.use_gpu:
            try:
                from fusion_methods.dtcwt_torch import dtcwt_torch_impl
                return dtcwt_torch_impl(input_source, img_resize, N=N)
            except Exception as exc:
                print(f"Warning: GPU DTCWT fusion failed ({exc}); falling back to CPU.")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        return _dtcwt_impl(
            input_source,
            img_resize,
            N,
            False
        )
    
    def _fuse_stackmffv4(self,
                         input_source: Union[str, List[np.ndarray]],
                         img_resize: Optional[Tuple[int, int]] = None,
                         model_path: Optional[str] = 'weights/stackmffv4.pth',
                         **kwargs) -> np.ndarray:
        """
        StackMFF-V4 fusion.

        Args:
            input_source: Image source
            img_resize: Target size
            model_path: Model weights file path

        Returns:
            Fused image
        """
        if not model_path:
            model_path = 'weights/stackmffv4.pth'
        if not os.path.isabs(model_path):
            model_path = resource_path(model_path)

        return _stackmffv4_impl(
            input_source,
            img_resize,
            model_path,
            self.use_gpu
        )

    
    def set_algorithm(self, algorithm: str):
        """
        Switch the fusion algorithm.

        Args:
            algorithm (str): New algorithm name ('guided_filter', 'dct', 'dtcwt', 'stackmffv4')
        """
        self._ensure_supported_algorithm(algorithm)
        self.algorithm = algorithm
        self._validate_environment()

    # Tile control API (instance-level)
    def set_tile_mode(self, enabled: bool):
        """Enable or disable tiled fusion (instance-level)."""
        self.tile_enabled = bool(enabled)

    def get_tile_mode(self) -> bool:
        """Return this instance's tiled fusion flag."""
        return bool(self.tile_enabled)

    def set_tile_params(self, block_size: Optional[int] = None, overlap: Optional[int] = None, threshold: Optional[int] = None):
        """Set tile parameters. Passing None leaves the corresponding value unchanged.

        Args:
            block_size: Tile block size (pixels)
            overlap: Overlap size (pixels)
            threshold: Longest-side threshold above which tiling is used (pixels)
        """
        if block_size is not None:
            self.tile_block_size = max(1, int(block_size))
        if overlap is not None:
            self.tile_overlap = max(0, int(overlap))
        if threshold is not None:
            self.tile_threshold = max(1, int(threshold))

    def get_tile_params(self) -> dict:
        """Return this instance's tile parameters as a dict."""
        return {
            'block_size': int(self.tile_block_size),
            'overlap': int(self.tile_overlap),
            'threshold': int(self.tile_threshold),
        }

    def _calculate_optimal_thread_count(
        self,
        h: int, w: int, channels: int,
        num_images: int,
        block_size: int,
        user_thread_count: int = None
    ) -> int:
        """
        Compute the optimal thread count from system memory and image properties (conservative).

        Memory estimate:
        - crops for one tile: num_images * block_size * block_size * channels * 4 bytes
        - fused_tile: block_size * block_size * channels * 4 bytes
        - weight2d: block_size * block_size * 4 bytes
        """
        try:
            import psutil
            avail_memory_mb = psutil.virtual_memory().available // (1024**2)
        except Exception:
            avail_memory_mb = 4096

        tile_memory = (num_images + 2) * (block_size ** 2) * channels * 4 / (1024**2)

        reserved_memory_mb = 2048
        usable_memory_mb = max(tile_memory * 2, avail_memory_mb - reserved_memory_mb)

        max_by_memory = max(1, int(usable_memory_mb / tile_memory))

        max_workers = min(max_by_memory, user_thread_count or 8, os.cpu_count() or 4)

        return max(1, max_workers)

    def _compute_tile_weights(self, x0: int, y0: int, x1: int, y1: int,
                               fw: int, fh: int, w: int, h: int,
                               overlap: int) -> np.ndarray:
        """
        Compute the feathering weights for a single tile (separable 2D weights).

        Args:
            x0, y0, x1, y1: Tile boundaries
            fw, fh: Width and height of the fused tile
            w, h: Width and height of the full image
            overlap: Overlap region size

        Returns:
            2D weight array (fh, fw), dtype=np.float32
        """
        left_exists = x0 > 0
        right_exists = x1 < w
        top_exists = y0 > 0
        bottom_exists = y1 < h

        left_o = overlap if left_exists else 0
        right_o = overlap if right_exists else 0
        top_o = overlap if top_exists else 0
        bottom_o = overlap if bottom_exists else 0

        left_o = min(left_o, fw - 1) if fw > 1 else 0
        right_o = min(right_o, fw - 1) if fw > 1 else 0
        top_o = min(top_o, fh - 1) if fh > 1 else 0
        bottom_o = min(bottom_o, fh - 1) if fh > 1 else 0

        if fw == 1:
            wx = np.ones((1,), dtype=np.float32)
        else:
            ix = np.arange(fw, dtype=np.float32)
            wx = np.ones((fw,), dtype=np.float32)
            if left_o > 0:
                wx_left = np.clip(ix / float(left_o), 0.0, 1.0)
                wx = np.minimum(wx, wx_left)
            if right_o > 0:
                wx_right = np.clip((fw - 1 - ix) / float(right_o), 0.0, 1.0)
                wx = np.minimum(wx, wx_right)

        if fh == 1:
            wy = np.ones((1,), dtype=np.float32)
        else:
            iy = np.arange(fh, dtype=np.float32)
            wy = np.ones((fh,), dtype=np.float32)
            if top_o > 0:
                wy_top = np.clip(iy / float(top_o), 0.0, 1.0)
                wy = np.minimum(wy, wy_top)
            if bottom_o > 0:
                wy_bottom = np.clip((fh - 1 - iy) / float(bottom_o), 0.0, 1.0)
                wy = np.minimum(wy, wy_bottom)

        return np.outer(wy, wx).astype(np.float32)

    def _fuse_tiled(self,
                    input_source: Union[List[np.ndarray], str],
                    algorithm: str,
                    img_resize: Optional[Tuple[int, int]] = None,
                    block_size: int = 1024,
                    overlap: int = 256,
                    thread_count: int = None,
                    **kwargs) -> np.ndarray:
        """
        Tiled (sliding-window) fusion, used when a single image is too large.

        - Splits the image into `block_size` tiles with `overlap` between them.
        - Runs the selected algorithm on each tile, then blends overlap regions
          by simple averaging to remove seam artifacts.
        - Tiles are processed in parallel threads; the thread count is derived
          from available system memory (conservative).
        - StackMFF V4 uses batched processing to exploit GPU parallelism.
        """

        imgs = None
        img_dir = None
        if isinstance(input_source, list):
            imgs = input_source
            if len(imgs) == 0:
                raise ValueError("_fuse_tiled requires a non-empty list of numpy arrays as input_source")
            h, w = imgs[0].shape[:2]
            channels = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
            out_dtype = bitdepth.stack_dtype(imgs)
        elif isinstance(input_source, str):
            img_dir = input_source
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(exts)]
            if len(files) == 0:
                raise ValueError(f"No image files found in directory: {img_dir}")
            first_img = read_image_any_depth(os.path.join(img_dir, files[0]))
            if first_img is None:
                raise RuntimeError(f"Unable to read image: {os.path.join(img_dir, files[0])}")
            h, w = first_img.shape[:2]
            channels = first_img.shape[2] if first_img.ndim == 3 else 1
            out_dtype = first_img.dtype
        else:
            raise ValueError("_fuse_tiled input_source must be a list of arrays or a directory path string")

        acc = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        step = max(1, block_size - overlap)
        num_images = len(imgs) if imgs is not None else len(files)

        # Compute tile coordinates
        tile_coords = []
        max_start_y = h - block_size
        max_start_x = w - block_size
        for y in range(0, h, step):
            y0 = min(y, max_start_y)
            y1 = y0 + block_size
            for x in range(0, w, step):
                x0 = min(x, max_start_x)
                x1 = x0 + block_size
                tile_coords.append((x0, y0, x1, y1))

        # StackMFF V4 uses batched processing
        if algorithm == 'stackmffv4':
            return self._fuse_tiled_stackmffv4_batched(
                imgs, img_dir, tile_coords, h, w, channels,
                block_size, overlap, out_dtype, **kwargs
            )

        # Other algorithms use the original multi-threaded processing
        optimal_threads = self._calculate_optimal_thread_count(
            h, w, channels, num_images, block_size, thread_count
        )

        # GPU fusion paths are already parallel internally; running tiles
        # concurrently would only contend for the device and multiply GPU memory use
        if self.use_gpu and algorithm in ('guided_filter', 'dct', 'dtcwt', 'gfgfgf'):
            optimal_threads = 1

        print(f"Tiled fusion: {len(tile_coords)} tiles, {optimal_threads} parallel workers (memory-optimized)", flush=True)

        def call_algo(crops):
            if algorithm == 'guided_filter':
                return self._fuse_guided_filter(crops, img_resize, **kwargs)
            elif algorithm == 'dct':
                return self._fuse_dct(crops, img_resize, **kwargs)
            elif algorithm == 'dtcwt':
                return self._fuse_dtcwt(crops, img_resize, **kwargs)
            elif algorithm == 'gfgfgf':
                return self._fuse_gfgfgf(crops, img_resize, **kwargs)
            elif algorithm == 'pyramid':
                return self._fuse_pyramid(crops, img_resize, **kwargs)
            elif algorithm == 'depthmap_max':
                return self._fuse_depthmap(crops, img_resize, mode=MODE_MAX, **kwargs)
            elif algorithm == 'depthmap_average':
                return self._fuse_depthmap(crops, img_resize, mode=MODE_AVERAGE, **kwargs)
            else:
                method_name = f"_fuse_{algorithm}"
                method = getattr(self, method_name, None)
                if callable(method):
                    return method(crops, img_resize, **kwargs)
                raise ValueError(f"Unsupported algorithm for tiled fusion: {algorithm}")

        def process_single_tile(coords):
            x0, y0, x1, y1 = coords

            if imgs is not None:
                crops = [img[y0:y1, x0:x1].copy() for img in imgs]
            else:
                crops = []
                for fname in files:
                    fp = os.path.join(img_dir, fname)
                    full = read_image_any_depth(fp)
                    if full is None:
                        raise RuntimeError(f"Unable to read image: {fp}")
                    crops.append(full[y0:y1, x0:x1].copy())

            fused_tile = call_algo(crops)
            if fused_tile is None:
                raise RuntimeError("Fusion returned None for a tile")

            if fused_tile.ndim == 2:
                fused_tile = fused_tile[:, :, np.newaxis]
            fh, fw = fused_tile.shape[:2]

            weight2d = self._compute_tile_weights(x0, y0, x1, y1, fw, fh, w, h, overlap)

            return (x0, y0, fh, fw, fused_tile, weight2d)

        results = {}
        total_tiles = len(tile_coords)
        completed_tiles = 0
        # Print at most ~10 progress lines to avoid flooding the terminal
        progress_step = max(1, total_tiles // 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
            futures = {executor.submit(process_single_tile, coords): coords
                       for coords in tile_coords}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                x0, y0, fh, fw, fused_tile, weight2d = result
                results[(x0, y0)] = result
                completed_tiles += 1
                if completed_tiles % progress_step == 0 or completed_tiles == total_tiles:
                    print(f"  Tiled fusion progress: {completed_tiles}/{total_tiles} tiles", flush=True)

        for x0, y0, fh, fw, fused_tile, weight2d in results.values():
            w_exp = weight2d[:, :, np.newaxis]
            acc[y0:y0+fh, x0:x0+fw, :channels] += fused_tile.astype(np.float32) * w_exp
            weight[y0:y0+fh, x0:x0+fw, 0] += weight2d

        weight[weight == 0] = 1.0
        fused = acc / weight
        # The accumulator holds pixel levels, not normalised values, so it is
        # clipped against the stack's own full scale. Clipping at 255 here would
        # drive every 16-bit level above 255 to white.
        fused = self._finish_tile_accumulator(fused, out_dtype)

        if channels == 1:
            return fused[:, :, 0]
        return fused

    @staticmethod
    def _finish_tile_accumulator(fused: np.ndarray, out_dtype) -> np.ndarray:
        """Round and clip a feathered tile accumulator to the stack's depth."""
        full_scale = bitdepth.max_value(out_dtype)
        return np.rint(np.clip(fused, 0, full_scale)).astype(out_dtype)

    def _fuse_tiled_stackmffv4_batched(self,
                                        imgs: Optional[List[np.ndarray]],
                                        img_dir: Optional[str],
                                        tile_coords: List[Tuple[int, int, int, int]],
                                        h: int, w: int, channels: int,
                                        block_size: int, overlap: int,
                                        out_dtype,
                                        **kwargs) -> np.ndarray:
        """
        Tiled StackMFF V4 fusion with batched processing.

        Packs multiple tiles into one batch for GPU inference to improve efficiency.
        """
        batch_size = self.stackmffv4_batch_size
        total_tiles = len(tile_coords)
        
        print(f"StackMFF V4 batched tiled fusion: {total_tiles} tiles, batch_size={batch_size}")
        
        # Resolve model path
        model_path = kwargs.get('model_path', 'weights/stackmffv4.pth')
        if not model_path:
            model_path = 'weights/stackmffv4.pth'
        if not os.path.isabs(model_path):
            model_path = resource_path(model_path)
        
        acc = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)
        
        # Get file list (for directory input)
        files = None
        if img_dir is not None:
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(exts)]
        
        # Process in batches
        for batch_start in range(0, total_tiles, batch_size):
            batch_end = min(batch_start + batch_size, total_tiles)
            batch_coords = tile_coords[batch_start:batch_end]
            current_batch_size = len(batch_coords)
            
            print(f"  Processing batch {batch_start // batch_size + 1}/{(total_tiles + batch_size - 1) // batch_size} ({current_batch_size} tiles)")
            
            # Prepare all tile data for this batch
            tiles_list = []
            for (x0, y0, x1, y1) in batch_coords:
                if imgs is not None:
                    crops = [img[y0:y1, x0:x1].copy() for img in imgs]
                else:
                    crops = []
                    for fname in files:
                        fp = os.path.join(img_dir, fname)
                        full = read_image_any_depth(fp)
                        if full is None:
                            raise RuntimeError(f"Unable to read image: {fp}")
                        crops.append(full[y0:y1, x0:x1].copy())
                tiles_list.append(crops)
            
            # Batched inference
            fused_tiles = self._run_stackmffv4_batch_with_fallback(tiles_list, model_path)

            # Merge results into the accumulator
            for idx, (x0, y0, x1, y1) in enumerate(batch_coords):
                fused_tile = fused_tiles[idx]
                if fused_tile is None:
                    raise RuntimeError("Fusion returned None for a tile")
                
                if fused_tile.ndim == 2:
                    fused_tile = fused_tile[:, :, np.newaxis]
                fh, fw = fused_tile.shape[:2]
                
                weight2d = self._compute_tile_weights(x0, y0, x1, y1, fw, fh, w, h, overlap)
                w_exp = weight2d[:, :, np.newaxis]
                
                acc[y0:y0+fh, x0:x0+fw, :channels] += fused_tile.astype(np.float32) * w_exp
                weight[y0:y0+fh, x0:x0+fw, 0] += weight2d
        
        weight[weight == 0] = 1.0
        fused = acc / weight
        fused = self._finish_tile_accumulator(fused, out_dtype)

        if channels == 1:
            return fused[:, :, 0]
        return fused

    def _run_stackmffv4_batch_with_fallback(self,
                                            tiles_list: list,
                                            model_path: str) -> list:
        """
        Run StackMFF V4 batch inference, degrading automatically when VRAM
        runs out: halve the batch progressively, then fall back to CPU if a
        single tile still does not fit.
        """
        if not self.use_gpu:
            return self._run_stackmffv4_cpu(tiles_list, model_path)

        import torch

        # Largest GPU batch confirmed to work during this run; avoids re-triggering slow OOM retries
        cap = getattr(self, '_stackmffv4_gpu_batch_cap', None)
        if cap is not None and len(tiles_list) > cap:
            results = []
            for i in range(0, len(tiles_list), cap):
                results.extend(self._run_stackmffv4_batch_with_fallback(
                    tiles_list[i:i + cap], model_path))
            return results

        if getattr(self, '_stackmffv4_force_cpu', False):
            return self._run_stackmffv4_cpu(tiles_list, model_path)

        try:
            return _stackmffv4_batch_impl(tiles_list, model_path, True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(tiles_list) > 1:
                half = max(1, len(tiles_list) // 2)
                self._stackmffv4_gpu_batch_cap = half
                print(f"Warning: CUDA out of memory with batch_size={len(tiles_list)}; "
                      f"retrying with batch_size={half}.")
                return (self._run_stackmffv4_batch_with_fallback(tiles_list[:half], model_path)
                        + self._run_stackmffv4_batch_with_fallback(tiles_list[half:], model_path))
            self._stackmffv4_force_cpu = True
            print("Warning: CUDA out of memory even for a single tile; "
                  "falling back to CPU for the rest of this render. "
                  "Reduce Tile Block Size in Settings to keep GPU acceleration.")
            return self._run_stackmffv4_cpu(tiles_list, model_path)

    def _run_stackmffv4_cpu(self, tiles_list: list, model_path: str) -> list:
        """
        Run StackMFF V4 batch inference on the CPU, converting an allocator
        out-of-memory failure into an actionable error message.
        """
        try:
            return _stackmffv4_batch_impl(tiles_list, model_path, False)
        except RuntimeError as exc:
            msg = str(exc)
            if 'not enough memory' in msg or 'DefaultCPUAllocator' in msg or 'bad allocation' in msg:
                raise RuntimeError(
                    "Out of memory during StackMFF-V4 fusion. The model's memory use "
                    "grows with tile area and with the square of the number of images "
                    "in the stack. Reduce Tile Block Size in Settings (e.g. 512 or 256) "
                    "and/or fuse fewer images at once, then try again."
                ) from exc
            raise

    def set_device(self, use_gpu: bool):
        """
        Switch the compute device.

        Args:
            use_gpu (bool): Whether to use the GPU
        """
        self.use_gpu = bool(use_gpu)
        self._validate_environment()
    
    def get_info(self) -> dict:
        """
        Get information about the current fusion engine.

        Returns:
            dict: Algorithm name, device type, etc.
        """
        import torch
        if self.use_gpu:
            if torch.cuda.is_available():
                device_name = 'CUDA'
            elif torch.backends.mps.is_available():
                device_name = 'MPS'
            else:
                device_name = 'CPU (GPU unavailable)'
        else:
            device_name = 'CPU'
        return {
            'algorithm': self.algorithm,
            'use_gpu': self.use_gpu,
            'device': device_name
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MultiFocusFusion(algorithm='{self.algorithm}', "
                f"use_gpu={self.use_gpu})")


# Convenience function
def fuse_images(input_source: Union[str, List[np.ndarray]],
                algorithm: str = 'guided_filter',
                use_gpu: bool = False,
                img_resize: Optional[Tuple[int, int]] = None,
                tile_enabled: Optional[bool] = None,
                tile_block_size: Optional[int] = None,
                tile_overlap: Optional[int] = None,
                tile_threshold: Optional[int] = None,
                stackmffv4_batch_size: Optional[int] = None,
                **kwargs) -> np.ndarray:
    """
    Convenience function: run image fusion in one call.

    Args:
        input_source: Image source (directory path or list of images)
        algorithm: Fusion algorithm ('guided_filter', 'dct', 'dtcwt', 'stackmffv4')
        use_gpu: Whether to use the GPU (CPU-only algorithms switch to CPU automatically)
        img_resize: Target size
        stackmffv4_batch_size: StackMFF V4 batch size (default 2)
        **kwargs: Algorithm-specific parameters

    Returns:
        Fused image

    Examples:
        # DCT algorithm
        result = fuse_images(image_list, algorithm='dct', block_size=8, kernel_size=7)

        # DTCWT algorithm
        result = fuse_images('./images', algorithm='dtcwt', use_gpu=False, N=4)

        # StackMFF-V4 algorithm
        result = fuse_images(image_list, algorithm='stackmffv4', use_gpu=False,
                           model_path='./weights/stackmffv4.pth')
    """
    # Resolve tile parameters: explicit args take precedence, then module defaults
    te = _DEFAULT_TILE_ENABLED if tile_enabled is None else bool(tile_enabled)
    tbs = _DEFAULT_TILE_BLOCK_SIZE if tile_block_size is None else int(tile_block_size)
    to = _DEFAULT_TILE_OVERLAP if tile_overlap is None else int(tile_overlap)
    tt = _DEFAULT_TILE_THRESHOLD if tile_threshold is None else int(tile_threshold)
    sbs = _DEFAULT_STACKMFFV4_BATCH_SIZE if stackmffv4_batch_size is None else int(stackmffv4_batch_size)

    fusion = MultiFocusFusion(
        algorithm=algorithm,
        use_gpu=use_gpu,
        tile_enabled=te,
        tile_block_size=tbs,
        tile_overlap=to,
        tile_threshold=tt,
        stackmffv4_batch_size=sbs,
    )

    return fusion.fuse(input_source, img_resize=img_resize, **kwargs)