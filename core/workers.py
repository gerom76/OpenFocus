from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import cv2
import numpy as np
import imageio.v2 as imageio
from core.registration import ImageRegistration
from core.multi_focus_fusion import MultiFocusFusion
from fusion_methods.ifcnn import _ifcnn_refine_impl, get_ifcnn_model_path, is_ifcnn_available
from utils import resource_path, normalize_kernel_size, get_imwrite_params
from constants import (
    TILE_BLOCK_SIZE, TILE_OVERLAP, TILE_THRESHOLD,
    DEFAULT_THREAD_COUNT
)


def refine_with_ifcnn(fusion_result, source_images, tile_enabled=None, tile_block_size=None,
                      tile_overlap=None, tile_threshold=None):
    """Run the IFCNN refinement stage on a fusion result.

    Returns the unrefined result unchanged when the stage cannot run, so a
    missing checkpoint or a model failure never costs the user their render.
    """
    if fusion_result is None:
        return fusion_result

    if not is_ifcnn_available():
        print(f"IFCNN refinement skipped: weights not found at {get_ifcnn_model_path()}", flush=True)
        return fusion_result

    try:
        return _ifcnn_refine_impl(
            fusion_result,
            source_images,
            get_ifcnn_model_path(),
            use_gpu=True,
            tile_enabled=(tile_enabled if tile_enabled is not None else True),
            tile_block_size=(tile_block_size if tile_block_size is not None else TILE_BLOCK_SIZE),
            tile_overlap=(tile_overlap if tile_overlap is not None else TILE_OVERLAP),
            tile_threshold=(tile_threshold if tile_threshold is not None else TILE_THRESHOLD),
        )
    except Exception as e:
        print(f"IFCNN refinement failed, keeping the unrefined result: {e}", flush=True)
        return fusion_result


class ROIAlignmentWorker(QThread):
    """Background thread that runs ECC registration in ROI mode, used to align the image stack before ROI selection"""

    finished_signal = pyqtSignal(object, float)  # aligned_images, alignment_time
    error_signal = pyqtSignal(str)

    def __init__(self, raw_images, reg_downscale_width=None, thread_count: int = DEFAULT_THREAD_COUNT,
                 ecc_parallel: bool = True):
        super().__init__()
        self.raw_images = raw_images
        self.reg_downscale_width = reg_downscale_width
        self.ecc_parallel = bool(ecc_parallel)
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4

    def run(self):
        """Run ECC registration in the thread"""
        try:
            alignment_start_time = time.time()

            # Register using the ECC method
            if self.reg_downscale_width is not None:
                registration = ImageRegistration(method="ecc", downscale_width=self.reg_downscale_width,
                                                 ecc_parallel=self.ecc_parallel)
            else:
                registration = ImageRegistration(method="ecc", ecc_parallel=self.ecc_parallel)

            aligned_images = registration.process(self.raw_images, output_path=None, thread_count=self.thread_count)
            alignment_time = time.time() - alignment_start_time

            self.finished_signal.emit(aligned_images, alignment_time)
        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()


class RenderWorker(QThread):
    """Background thread that performs image registration and fusion (extracted from main.py)"""

    finished_signal = pyqtSignal(object, object, bool, float, float, str)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        raw_images,
        aligned_images,
        is_images_aligned,
        last_alignment_options,
        need_align_homography,
        need_align_ecc,
        need_fusion,
        rb_a_checked,
        rb_b_checked,
        rb_c_checked,
        rb_gfg_checked,
        rb_d_checked,
        kernel_slider_value,
        tile_enabled=None,
        tile_block_size=None,
        tile_overlap=None,
        tile_threshold=None,
        reg_downscale_width=None,
        thread_count: int = DEFAULT_THREAD_COUNT,
        stackmffv4_batch_size: int = 2,
        roi_rect=None,
        roi_mode="crop", # 'crop' or 'paste'
        roi_base_index=0,
        ecc_parallel: bool = True,
        ifcnn_refine: bool = False,
    ):
        super().__init__()
        self.raw_images = raw_images
        self.aligned_images = aligned_images
        self.is_images_aligned = is_images_aligned
        self.last_alignment_options = last_alignment_options
        self.roi_rect = roi_rect
        self.roi_mode = roi_mode
        self.roi_base_index = roi_base_index

        # Registration options
        self.need_align_homography = need_align_homography
        self.need_align_ecc = need_align_ecc

        # Fusion options
        self.need_fusion = need_fusion
        self.rb_a_checked = rb_a_checked
        self.rb_b_checked = rb_b_checked
        self.rb_c_checked = rb_c_checked
        self.rb_gfg_checked = rb_gfg_checked
        self.rb_d_checked = rb_d_checked
        self.kernel_slider_value = kernel_slider_value
        # Optional IFCNN refinement stage, applied to the fusion result
        self.ifcnn_refine = bool(ifcnn_refine)
        # Tile params passed from UI (may be None -> use fusion defaults)
        self.tile_enabled = tile_enabled
        self.tile_block_size = tile_block_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        # StackMFF V4 batch size
        self.stackmffv4_batch_size = max(1, int(stackmffv4_batch_size)) if stackmffv4_batch_size else 2
        # Registration downscale width passed from UI (optional)
        self.reg_downscale_width = reg_downscale_width
        # Compute ECC pair matrices concurrently (identical results, faster)
        self.ecc_parallel = bool(ecc_parallel)
        # User-configured thread count (controls the internal ThreadPool size)
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4

    def run(self):
        """Run the image-processing pipeline in the thread"""
        try:
            alignment_time = 0
            fusion_time = 0
            device_name = "CPU"

            if self.raw_images:
                h, w = self.raw_images[0].shape[:2]
                print(f"Render started: {len(self.raw_images)} images ({w}x{h})", flush=True)

            current_alignment_options = (
                self.need_align_homography,
                self.need_align_ecc,
            )
            need_registration = (
                self.need_align_homography
                or self.need_align_ecc
            )
            can_reuse_aligned_images = (
                self.is_images_aligned
                and len(self.aligned_images) > 0
                and self.last_alignment_options == current_alignment_options
            )

            # 1. Registration stage
            if need_registration and can_reuse_aligned_images:
                processed_images = self.aligned_images
                registration_performed = True
                print("Registration: reusing cached aligned images", flush=True)
            else:
                processed_images = self.raw_images.copy()
                registration_performed = False

                if need_registration:
                    alignment_start_time = time.time()
                    processed_images, alignment_time = self._run_registration(processed_images)
                    registration_performed = True
                    print(f"Registration completed in {alignment_time:.2f}s", flush=True)

            # 2. ROI cropping stage
            cropped_images, base_full_image, roi_rect_int = self._apply_roi_cropping(processed_images)
            if roi_rect_int is not None:
                rx, ry, rw, rh = roi_rect_int
                print(f"ROI crop: {rw}x{rh} at ({rx}, {ry}), mode={self.roi_mode}", flush=True)

            # 3. Fusion stage
            fusion_result = None
            if self.need_fusion:
                fusion_start_time = time.time()
                # Fuse using the cropped images (if ROI is enabled)
                fusion_images = cropped_images if cropped_images is not None else processed_images
                fusion_result, device_name = self._run_fusion(fusion_images)

                # 4. IFCNN refinement stage (repairs detail the fusion step missed)
                if self.ifcnn_refine and fusion_result is not None:
                    refine_start_time = time.time()
                    fusion_result = self._run_ifcnn_refine(fusion_result, fusion_images)
                    print(f"IFCNN refinement completed in {time.time() - refine_start_time:.2f}s", flush=True)

                # ROI paste stage
                if self.roi_mode == "paste" and base_full_image is not None and fusion_result is not None and roi_rect_int is not None:
                    fusion_result = self._apply_roi_pasting(fusion_result, base_full_image, roi_rect_int)

                fusion_time = time.time() - fusion_start_time
                print(f"Fusion completed in {fusion_time:.2f}s", flush=True)

            print(f"Render finished in {alignment_time + fusion_time:.2f}s total", flush=True)

            self.finished_signal.emit(
                processed_images,
                fusion_result,
                registration_performed,
                alignment_time,
                fusion_time,
                device_name,
            )
        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()

    def _run_registration(self, images):
        """Perform image registration"""
        alignment_start_time = time.time()

        if self.need_align_homography and self.need_align_ecc:
            mode = "both"
        elif self.need_align_homography:
            mode = "homography"
        elif self.need_align_ecc:
            mode = "ecc"
        else:
            return images, 0

        print(f"Registration started: mode={mode}, {len(images)} images", flush=True)

        if self.reg_downscale_width is not None:
            registration = ImageRegistration(method=mode, downscale_width=self.reg_downscale_width,
                                             ecc_parallel=self.ecc_parallel)
        else:
            registration = ImageRegistration(method=mode, ecc_parallel=self.ecc_parallel)

        processed = registration.process(images, output_path=None, thread_count=self.thread_count)
        alignment_time = time.time() - alignment_start_time

        return processed, alignment_time

    def _apply_roi_cropping(self, images):
        """Apply ROI cropping, returns (cropped image stack, base image, ROI rectangle)"""
        base_full_image = None
        roi_rect_int = None

        if self.roi_rect is not None and len(images) > 0:
            rx, ry, rw, rh = self._normalize_roi_rect(images[0].shape)

            roi_rect_int = (rx, ry, rw, rh)

            if self.roi_mode == "paste":
                idx = max(0, min(self.roi_base_index, len(images) - 1))
                base_full_image = images[idx].copy()

            cropped_stack = []
            for img in images:
                if img.ndim == 3:
                    crop = img[ry:ry+rh, rx:rx+rw, :]
                else:
                    crop = img[ry:ry+rh, rx:rx+rw]
                cropped_stack.append(crop)

            return cropped_stack, base_full_image, roi_rect_int

        return None, None, None

    def _normalize_roi_rect(self, image_shape):
        """Normalize the ROI rectangle bounds"""
        rx = int(self.roi_rect.x())
        ry = int(self.roi_rect.y())
        rw = int(self.roi_rect.width())
        rh = int(self.roi_rect.height())

        h, w = image_shape[:2]
        rx = max(0, min(rx, w))
        ry = max(0, min(ry, h))
        rw = max(1, min(rw, w - rx))
        rh = max(1, min(rh, h - ry))

        return rx, ry, rw, rh

    def _run_fusion(self, images):
        """Perform image fusion, returns (fusion result, device name)"""
        algorithm = self._get_fusion_algorithm()

        fusion = MultiFocusFusion(
            algorithm=algorithm,
            use_gpu=True,
            tile_enabled=(self.tile_enabled if self.tile_enabled is not None else True),
            tile_block_size=(self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE),
            tile_overlap=(self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP),
            tile_threshold=(self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD),
            stackmffv4_batch_size=self.stackmffv4_batch_size,
        )

        info = fusion.get_info()
        device_name = info['device']

        print(f"Fusion started: {algorithm} on {device_name}, {len(images)} images", flush=True)

        kernel_size = normalize_kernel_size(self.kernel_slider_value)

        if algorithm == "guided_filter":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
            )
        elif algorithm == "dct":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                block_size=8,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
            )
        elif algorithm == "dtcwt":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                thread_count=self.thread_count,
            )
        elif algorithm == "gfgfgf":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
            )
        elif algorithm == "stackmffv4":
            model_path = resource_path("weights", "stackmffv4.pth")
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                model_path=model_path,
                thread_count=self.thread_count,
            )
        else:
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
            )

        return result, device_name

    def _run_ifcnn_refine(self, fusion_result, source_images):
        """Refine the fusion result with IFCNN using this worker's tile settings."""
        return refine_with_ifcnn(
            fusion_result,
            source_images,
            tile_enabled=self.tile_enabled,
            tile_block_size=self.tile_block_size,
            tile_overlap=self.tile_overlap,
            tile_threshold=self.tile_threshold,
        )

    def _get_fusion_algorithm(self):
        """Get the fusion-algorithm name based on the UI selection"""
        if self.rb_a_checked:
            return "guided_filter"
        elif self.rb_b_checked:
            return "dct"
        elif self.rb_c_checked:
            return "dtcwt"
        elif self.rb_gfg_checked:
            return "gfgfgf"
        elif self.rb_d_checked:
            return "stackmffv4"
        else:
            return "guided_filter"

    def _apply_roi_pasting(self, fusion_result, base_image, roi_rect_int):
        """Paste the fusion result back onto the base image"""
        rx, ry, rw, rh = roi_rect_int

        # Ensure the fusion-result size matches (sizes may not be exactly equal in tiled fusion)
        fr_h, fr_w = fusion_result.shape[:2]
        copy_h = min(fr_h, rh)
        copy_w = min(fr_w, rw)

        if base_image.ndim == 3 and fusion_result.ndim == 3:
            base_image[ry:ry+copy_h, rx:rx+copy_w, :] = fusion_result[:copy_h, :copy_w, :]
        elif base_image.ndim == 2 and fusion_result.ndim == 2:
            base_image[ry:ry+copy_h, rx:rx+copy_w] = fusion_result[:copy_h, :copy_w]
        elif base_image.ndim == 3 and fusion_result.ndim == 2:
            for c in range(3):
                base_image[ry:ry+copy_h, rx:rx+copy_w, c] = fusion_result[:copy_h, :copy_w]

        return base_image


class BatchWorker(QThread):
    """Batch-processing worker thread"""

    progress_updated = pyqtSignal(int, int, str)  # current progress, total, message
    finished = pyqtSignal(dict)  # processing result
    error = pyqtSignal(str)  # error message

    def __init__(self, folder_paths, output_type, output_path, processing_settings, reg_downscale_width=None,
                 tile_enabled=None, tile_block_size=None, tile_overlap=None, tile_threshold=None, thread_count: int = 4,
                 stackmffv4_batch_size: int = 2,
                 import_mode="multiple_folders", split_method=None, split_param=None,
                 single_folder_images_with_times=None, ecc_parallel: bool = True):
        super().__init__()
        self.folder_paths = folder_paths
        self.output_type = output_type
        self.output_path = output_path
        self.processing_settings = processing_settings
        self.is_cancelled = False
        self.import_mode = import_mode
        self.split_method = split_method
        self.split_param = split_param
        self.single_folder_images_with_times = single_folder_images_with_times or []

        from core.image_loader import ImageStackLoader
        self.image_loader = ImageStackLoader()
        from core.registration import ImageRegistration
        self.reg_downscale_width = reg_downscale_width
        self.ecc_parallel = bool(ecc_parallel)
        self.tile_enabled = tile_enabled
        self.tile_block_size = tile_block_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        self.stackmffv4_batch_size = max(1, int(stackmffv4_batch_size)) if stackmffv4_batch_size else 2
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4
    
    def run(self):
        """Run batch processing"""
        try:
            if self.import_mode == "single_folder" and self.single_folder_images_with_times:
                stacks = self._split_images_for_processing()
                total_count = len(stacks)
                success_count = 0
                failed_stacks = []

                original_paths = [item[0] for item in self.single_folder_images_with_times]
                source_folder = os.path.dirname(original_paths[0]) if original_paths else ""
                source_folder_name = os.path.basename(source_folder) if source_folder else "Output"

                output_dir = self._get_output_path_for_single_folder(source_folder)
                os.makedirs(output_dir, exist_ok=True)

                for i, stack_images_with_times in enumerate(stacks):
                    if self.is_cancelled:
                        break

                    stack_name = f"{source_folder_name}_{i + 1:03d}"
                    self.progress_updated.emit(i, total_count, f"Processing {stack_name} ({i + 1}/{total_count})")

                    try:
                        self._process_single_stack(stack_images_with_times, output_dir, stack_name, i)
                        success_count += 1
                    except Exception as e:
                        failed_stacks.append(f"{stack_name}: {str(e)}")
                        print(f"Error processing {stack_name}: {str(e)}")

                results = {
                    'success': success_count,
                    'total': total_count,
                    'failed': failed_stacks,
                    'cancelled': self.is_cancelled
                }
                self.finished.emit(results)
            else:
                success_count = 0
                total_count = len(self.folder_paths)
                failed_folders = []

                for i, folder_path in enumerate(self.folder_paths):
                    if self.is_cancelled:
                        break

                    folder_name = os.path.basename(folder_path)
                    self.progress_updated.emit(i, total_count, f"Processing folder {i + 1}/{total_count}: {folder_name}")

                    try:
                        self.process_single_folder(folder_path)
                        success_count += 1
                    except Exception as e:
                        failed_folders.append(f"{folder_name}: {str(e)}")
                        print(f"Error processing {folder_name}: {str(e)}")

                results = {
                    'success': success_count,
                    'total': total_count,
                    'failed': failed_folders,
                    'cancelled': self.is_cancelled
                }
                self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Batch processing failed: {str(e)}")

    def _split_images_for_processing(self):
        """Split the images in a single folder"""
        if not self.single_folder_images_with_times:
            return []

        if self.split_method == "count":
            return self.image_loader.split_by_count(
                self.single_folder_images_with_times,
                self.split_param or 5
            )
        elif self.split_method == "time_threshold":
            return self.image_loader.split_by_time_threshold(
                self.single_folder_images_with_times,
                self.split_param or 5.0
            )
        else:
            return [self.single_folder_images_with_times]

    def _process_single_stack(self, stack_images_with_times, output_dir, stack_name, stack_index):
        """Process a single image stack (from single-folder splitting)"""
        images = [item[1] for item in stack_images_with_times]
        original_paths = [item[0] for item in stack_images_with_times]

        aligned_images = images.copy()
        reg_methods = self.processing_settings.get('reg_methods', [])

        if reg_methods:
            align_homography = "homography" in reg_methods
            align_ecc = "ecc" in reg_methods

            if align_homography and align_ecc:
                mode = "both"
            elif align_homography:
                mode = "homography"
            elif align_ecc:
                mode = "ecc"
            else:
                mode = None

            if mode:
                from core.registration import ImageRegistration
                if self.reg_downscale_width is not None:
                    registration = ImageRegistration(method=mode, downscale_width=self.reg_downscale_width,
                                                     ecc_parallel=self.ecc_parallel)
                else:
                    registration = ImageRegistration(method=mode, ecc_parallel=self.ecc_parallel)
                aligned_images = registration.process(images, output_path=None, thread_count=self.thread_count)

        fusion_method = self.processing_settings.get('fusion_method')
        if fusion_method:
            fusion_params = self.processing_settings.get('fusion_params', {}).copy()
            if fusion_method == "stackmffv4" and "model_path" not in fusion_params:
                fusion_params["model_path"] = resource_path("weights", "stackmffv4.pth")

            tile_kwargs = {}
            tile_kwargs.setdefault('tile_enabled', self.tile_enabled if self.tile_enabled is not None else True)
            tile_kwargs.setdefault('tile_block_size', self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE)
            tile_kwargs.setdefault('tile_overlap', self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP)
            tile_kwargs.setdefault('tile_threshold', self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD)
            tile_kwargs.setdefault('stackmffv4_batch_size', self.stackmffv4_batch_size)

            fusion = MultiFocusFusion(algorithm=fusion_method, use_gpu=True, **tile_kwargs)

            if fusion_method == "guided_filter":
                kernel_size = fusion_params.get('kernel_size', 31)
                if kernel_size % 2 == 0:
                    kernel_size = max(1, kernel_size - 1)
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    kernel_size=kernel_size,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "dct":
                kernel_size = fusion_params.get('kernel_size', 7)
                if kernel_size % 2 == 0:
                    kernel_size = max(1, kernel_size - 1)
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    block_size=8,
                    kernel_size=kernel_size,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "dtcwt":
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "gfgfgf":
                kernel_size = fusion_params.get('kernel_size', 7)
                if kernel_size % 2 == 0:
                    kernel_size = max(1, kernel_size - 1)
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    kernel_size=kernel_size,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "stackmffv4":
                model_path = fusion_params.get('model_path', resource_path("weights", "stackmffv4.pth"))
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    model_path=model_path,
                    thread_count=self.thread_count,
                )
            else:
                result = None

            if result is not None and self.processing_settings.get('ifcnn_refine'):
                result = refine_with_ifcnn(
                    result,
                    aligned_images,
                    tile_enabled=self.tile_enabled,
                    tile_block_size=self.tile_block_size,
                    tile_overlap=self.tile_overlap,
                    tile_threshold=self.tile_threshold,
                )

            output_format = self.processing_settings.get('format', 'jpg')
            imwrite_params = get_imwrite_params(output_format)

            if result is not None:
                output_path = os.path.join(output_dir, f"{stack_name}.{output_format}")
                cv2.imwrite(output_path, result, imwrite_params)

            if self.processing_settings.get('save_aligned'):
                for idx, img in enumerate(aligned_images):
                    aligned_filename = f"{stack_name}_aligned_{idx + 1:03d}.{output_format}"
                    aligned_path = os.path.join(output_dir, aligned_filename)
                    cv2.imwrite(aligned_path, img, imwrite_params)
    
    def process_single_folder(self, folder_path):
        """Process a single folder"""
        # 1. Load images
        success, message, images, filenames = self.image_loader.load_from_folder(folder_path)
        if not success or not images:
            raise Exception(f"Failed to load images: {message}")
        
        # 2. Image registration (if needed)
        aligned_images = images.copy()
        reg_methods = self.processing_settings.get('reg_methods', [])
        
        if reg_methods:
            # Registration options
            align_homography = "homography" in reg_methods
            align_ecc = "ecc" in reg_methods
            
            # Perform registration
            aligned_images = []
            
            # Determine the registration mode
            if align_homography and align_ecc:
                mode = "both"
            elif align_homography:
                mode = "homography"
            elif align_ecc:
                mode = "ecc"
            else:
                mode = None

            if mode:
                if getattr(self, 'reg_downscale_width', None) is not None:
                    registration = ImageRegistration(method=mode, downscale_width=self.reg_downscale_width,
                                                     ecc_parallel=self.ecc_parallel)
                else:
                    registration = ImageRegistration(method=mode, ecc_parallel=self.ecc_parallel)
                aligned_images = registration.process(images, output_path=None, thread_count=self.thread_count)
            else:
                # If no registration method is selected, use the original images directly
                aligned_images = images.copy()
        
        # 3. Image fusion
        fusion_method = self.processing_settings.get('fusion_method')
        if fusion_method:
            fusion_params = self.processing_settings.get('fusion_params', {})
            if fusion_method == "stackmffv4" and "model_path" not in fusion_params:
                fusion_params = dict(fusion_params)
                fusion_params["model_path"] = resource_path("weights", "stackmffv4.pth")
            
            # Create a fuser instance for the corresponding algorithm
            # Prefer the tile override value that may be included in fusion_params within processing_settings
            tile_kwargs = {}
            if isinstance(fusion_params, dict):
                # allow explicit per-batch overrides
                if 'tile_enabled' in fusion_params:
                    tile_kwargs['tile_enabled'] = fusion_params.pop('tile_enabled')
                if 'tile_block_size' in fusion_params:
                    tile_kwargs['tile_block_size'] = fusion_params.pop('tile_block_size')
                if 'tile_overlap' in fusion_params:
                    tile_kwargs['tile_overlap'] = fusion_params.pop('tile_overlap')
                if 'tile_threshold' in fusion_params:
                    tile_kwargs['tile_threshold'] = fusion_params.pop('tile_threshold')

            # Fallback to worker-level (window) settings if not provided
            tile_kwargs.setdefault('tile_enabled', self.tile_enabled if self.tile_enabled is not None else True)
            tile_kwargs.setdefault('tile_block_size', self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE)
            tile_kwargs.setdefault('tile_overlap', self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP)
            tile_kwargs.setdefault('tile_threshold', self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD)

            fusion = MultiFocusFusion(algorithm=fusion_method, use_gpu=True, **tile_kwargs)
            
            # Call the fuse method to perform fusion
            fusion_result = fusion.fuse(aligned_images, thread_count=self.thread_count, **fusion_params)

            if fusion_result is not None and self.processing_settings.get('ifcnn_refine'):
                fusion_result = refine_with_ifcnn(
                    fusion_result,
                    aligned_images,
                    tile_enabled=self.tile_enabled,
                    tile_block_size=self.tile_block_size,
                    tile_overlap=self.tile_overlap,
                    tile_threshold=self.tile_threshold,
                )
        else:
            fusion_result = None
        
        # 4. Save the result
        if fusion_result is not None:
            self.save_fusion_result(folder_path, fusion_result)
        
        # 5. If needed, save the registered image stack
        save_aligned = self.processing_settings.get('save_aligned', False)
        if save_aligned:
            self.save_registered_stack(folder_path, aligned_images, filenames)
    
    def save_fusion_result(self, folder_path, fusion_result):
        """Save the fusion result"""
        # Determine the output path
        if self.output_type == "subfolder":
            output_dir = os.path.join(folder_path, self.output_path)
            os.makedirs(output_dir, exist_ok=True)
        elif self.output_type == "same":
            output_dir = folder_path
        else:  # custom
            output_dir = self.output_path
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate the file name
        folder_name = os.path.basename(folder_path)
        extension = self.processing_settings.get('format', 'png')
        filename = f"{folder_name}.{extension}"
        output_path = os.path.join(output_dir, filename)

        # Save the image
        cv2.imwrite(output_path, fusion_result, get_imwrite_params(extension))
    
    def save_registered_stack(self, folder_path, images, filenames):
        """Save the registered image stack"""
        # Determine the output path
        if self.output_type == "custom":
            output_dir = self.output_path
            folder_name = os.path.basename(folder_path)
            output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)
        else:  # same as source
            output_dir = folder_path
        
        # Save each image
        extension = self.processing_settings.get('format', 'png')
        imwrite_params = get_imwrite_params(extension)

        for i, image in enumerate(images):
            if i < len(filenames):
                # Use the original file name
                filename = filenames[i]
                if not filename.lower().endswith(f".{extension}"):
                    # Change the extension
                    base_name = os.path.splitext(filename)[0]
                    filename = f"{base_name}.{extension}"
            else:
                # Generate the default file name
                filename = f"registered_{i+1:04d}.{extension}"
            
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image, imwrite_params)

    def _get_output_path_for_single_folder(self, source_folder_path):
        """Get the output path in single-folder mode"""
        if self.output_type == "subfolder":
            output_dir = os.path.join(source_folder_path, self.output_path)
        elif self.output_type == "same":
            output_dir = source_folder_path
        else:
            output_dir = self.output_path
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _get_output_path(self, folder_path, stack_name=None):
        """Get the output path"""
        if self.output_type == "subfolder":
            output_dir = os.path.join(folder_path, self.output_path)
        elif self.output_type == "same":
            output_dir = folder_path
        else:
            output_dir = self.output_path
            if stack_name:
                output_dir = os.path.join(output_dir, stack_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def cancel(self):
        """Cancel batch processing"""
        self.is_cancelled = True


class GifSaverWorker(QThread):
    """Background thread that saves a GIF"""
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, images, file_path, duration_sec, label_manager, target_type):
        super().__init__()
        self.images = images
        self.file_path = file_path
        self.duration_sec = duration_sec
        self.label_manager = label_manager
        self.target_type = target_type

    def run(self):
        try:
            normalized_images = []
            for i, img in enumerate(self.images):
                # Create a copy of the image and overlay a label when needed
                img_copy = self.label_manager.prepare_bgr_image(self.target_type, img, i)
                
                # If the image is grayscale, convert it to RGB
                if len(img_copy.shape) == 2:
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
                # If it is in BGR format (OpenCV format), convert it to RGB
                elif len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                
                # Ensure the image is in uint8 format
                if img_copy.dtype != np.uint8:
                    img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
                
                normalized_images.append(img_copy)
            
            # Save the GIF using imageio
            imageio.mimsave(
                self.file_path,
                normalized_images,
                duration=self.duration_sec,
                loop=0  # loop playback, 0 means infinite loop
            )
            self.finished_signal.emit(True, f"GIF animation saved to:\n{self.file_path}")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


