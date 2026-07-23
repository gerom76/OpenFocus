"""
Image stack loader module
Responsible for loading image stacks from folders and generating thumbnails
"""

import os
import cv2
import numpy as np
import concurrent.futures
import time
from datetime import datetime
from typing import Callable, List, Tuple, Optional, Dict, Any

# Called as (completed, total) while a stack is being decoded.
ProgressCallback = Callable[[int, int], None]

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

from PyQt6.QtGui import QPixmap, QImage

from utils import bitdepth
from utils.image_utils import read_image_any_depth


class ImageStackLoader:
    """Image stack loader"""

    RAW_FORMATS = {'.nef', '.nrw'}  # Nikon RAW, requires rawpy (LibRaw)
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'} | (RAW_FORMATS if RAWPY_AVAILABLE else set())
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self):
        self.image_paths = []
        self.images = []
        self.thumbnail_size = (600, 400)

    @classmethod
    def read_image_bgr(cls, full_path: str) -> Optional[np.ndarray]:
        """Read an image as a BGR array, then bring it to the active depth mode.

        The decode itself is always done at the source's native depth - RAW at
        16 bits, and IMREAD_UNCHANGED for everything else so a 16-bit PNG or
        TIFF arrives intact. bitdepth.apply_load_mode then narrows or widens the
        result according to the mode, so in the default auto mode a >8-bit file
        keeps its extra bits and an 8-bit file stays 8-bit.

        Returns None on failure.
        """
        ext = os.path.splitext(full_path)[1].lower()
        if ext in cls.RAW_FORMATS:
            if not RAWPY_AVAILABLE:
                return None
            # 8-bit output is only ever requested when the mode forces it; the
            # RAW postprocess is where the extra bits would be lost for good.
            output_bps = 8 if bitdepth.get_mode() == bitdepth.MODE_8 else 16
            # Pass a file object to support paths with non-ASCII characters
            with open(full_path, 'rb') as f:
                with rawpy.imread(f) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, output_bps=output_bps)
            return bitdepth.apply_load_mode(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        return read_image_any_depth(full_path)

    def _load_files_parallel(
        self,
        entries: List[Tuple[str, str]],
        scale_factor: float = 1.0,
        max_workers: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Optional[np.ndarray]]:
        """Decode a list of (filename, full_path) entries in parallel.

        Decoding (cv2/rawpy) releases the GIL, so threads give near-linear speedup.
        Returns decoded images in input order (None for entries that failed);
        progress is printed as files complete and reported to progress_callback,
        followed by a summary line with elapsed time and throughput.
        """
        if max_workers is None:
            max_workers = min(8, os.cpu_count() or 4)
        max_workers = max(1, min(max_workers, len(entries)))

        total = len(entries)
        results: List[Optional[np.ndarray]] = [None] * total

        def decode(index: int, filename: str, full_path: str):
            if not os.path.exists(full_path):
                return index, filename, None, "not_found"
            try:
                img = self.read_image_bgr(full_path)
                if img is not None and scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(img.shape[1] * scale_factor)
                    height = int(img.shape[0] * scale_factor)
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                return index, filename, img, None
            except Exception as e:
                return index, filename, None, str(e)

        start_time = time.perf_counter()
        failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(decode, i, filename, full_path)
                for i, (filename, full_path) in enumerate(entries)
            ]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                index, filename, img, error = future.result()
                results[index] = img
                completed += 1
                if error == "not_found":
                    failed += 1
                    print(f"[{completed}/{total}] File not found: {filename}", flush=True)
                elif error is not None:
                    failed += 1
                    print(f"[{completed}/{total}] Failed to load {filename}: {error}", flush=True)
                elif img is None:
                    failed += 1
                    print(f"[{completed}/{total}] Failed to load {filename}", flush=True)
                else:
                    print(f"[{completed}/{total}] Loaded {filename} ({img.shape[1]}x{img.shape[0]})", flush=True)

                if progress_callback is not None:
                    try:
                        progress_callback(completed, total)
                    except Exception:
                        pass

        print(self._format_load_stats(total - failed, failed, time.perf_counter() - start_time), flush=True)

        return results

    @staticmethod
    def _unify_stack_depth(images: List[np.ndarray]) -> List[np.ndarray]:
        """Bring every frame in a stack to one depth, and say so on the console.

        A folder can legitimately hold mixed depths - 16-bit TIFFs beside
        JPEGs - and the fusion methods all assume a single full scale across the
        stack, so the odd ones out are promoted rather than left to skew the
        focus measures.
        """
        if not images:
            return images
        target = bitdepth.stack_dtype(images)
        mixed = any(np.dtype(img.dtype) != target for img in images)
        if mixed:
            print(f"[Depth] Mixed-depth stack, promoting all frames to "
                  f"{bitdepth.describe(target)}", flush=True)
            images = bitdepth.unify(images, target)
        print(f"[Depth] {bitdepth.stack_summary(images)} "
              f"(mode: {bitdepth.get_mode()})", flush=True)
        return images

    @staticmethod
    def _format_load_stats(loaded: int, failed: int, elapsed: float) -> str:
        """One-line summary: how many images, how long it took, and the mean rate."""
        rate = loaded / elapsed if elapsed > 0 else 0.0
        message = f"Loaded {loaded} image(s) in {elapsed:.2f} s ({rate:.1f} images/s)"
        if failed > 0:
            message += f" - {failed} failed"
        return message

    def load_from_folder(
        self,
        folder_path: str,
        scale_factor: float = 1.0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not os.path.isdir(folder_path):
            return False, "Selected path is not a valid directory", [], []

        image_files = []
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_FORMATS:
                full_path = os.path.join(folder_path, filename)
                image_files.append((filename, full_path))

        if not image_files:
            return False, "No supported image files found in the folder", [], []

        image_files.sort(key=lambda x: x[0])

        decoded = self._load_files_parallel(image_files, scale_factor, progress_callback=progress_callback)

        loaded_images = []
        filenames = []
        loaded_paths = []
        failed_count = 0
        for (filename, full_path), img in zip(image_files, decoded):
            if img is None:
                failed_count += 1
            else:
                loaded_images.append(img)
                filenames.append(filename)
                loaded_paths.append(full_path)

        if not loaded_images:
            return False, "Could not load any image files", [], []

        loaded_images = self._unify_stack_depth(loaded_images)

        self.images = loaded_images
        self.image_paths = loaded_paths

        message = f"Loaded {len(loaded_images)} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        return True, message, loaded_images, filenames

    def load_from_video(
        self,
        video_path: str,
        scale_factor: float = 1.0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not os.path.isfile(video_path):
            return False, "Video file does not exist", [], []

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_VIDEO_FORMATS:
            return False, f"Unsupported video format: {ext}", [], []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file", [], []

        loaded_images = []
        filenames = []
        frame_index = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        start_time = time.perf_counter()

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(frame.shape[1] * scale_factor)
                    height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Video frames always decode as 8-bit; honour a forced 16-bit
                # mode so a video stack matches the rest of the pipeline.
                loaded_images.append(bitdepth.apply_load_mode(frame))
                filenames.append(f"{video_name}_frame_{frame_index:04d}.png")
                frame_index += 1

                if total_frames > 0:
                    print(f"[{frame_index}/{total_frames}] Extracted frame "
                          f"({frame.shape[1]}x{frame.shape[0]})", flush=True)
                if progress_callback is not None:
                    try:
                        progress_callback(frame_index, total_frames)
                    except Exception:
                        pass
        finally:
            cap.release()

        elapsed = time.perf_counter() - start_time
        rate = len(loaded_images) / elapsed if elapsed > 0 else 0.0
        print(f"Extracted {len(loaded_images)} frame(s) in {elapsed:.2f} s "
              f"({rate:.1f} frames/s)", flush=True)

        if not loaded_images:
            return False, "No frames could be extracted from the video", [], []

        self.images = loaded_images
        self.image_paths = [video_path] * len(loaded_images)

        message = f"Loaded {len(loaded_images)} frame(s) from video"
        if total_frames > 0 and len(loaded_images) < total_frames:
            message += f" (expected: {total_frames})"

        return True, message, loaded_images, filenames

    def is_video_file(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.SUPPORTED_VIDEO_FORMATS

    def load_from_filepaths(
        self,
        filepaths: list[str],
        scale_factor: float = 1.0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not filepaths:
            return False, "No file paths provided", [], []

        entries = [(os.path.basename(p), p) for p in filepaths]
        decoded = self._load_files_parallel(entries, scale_factor, progress_callback=progress_callback)

        loaded_images = []
        filenames = []
        loaded_paths = []
        failed_count = 0
        for (filename, full_path), img in zip(entries, decoded):
            if img is None:
                failed_count += 1
            else:
                loaded_images.append(img)
                filenames.append(filename)
                loaded_paths.append(full_path)

        if not loaded_images:
            return False, "Could not load any image files", [], []

        loaded_images = self._unify_stack_depth(loaded_images)

        self.images = loaded_images
        self.image_paths = loaded_paths

        message = f"Loaded {len(loaded_images)} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        return True, message, loaded_images, filenames

    def create_pixmaps(self, images: List[np.ndarray], max_size: Tuple[int, int] = (800, 600)) -> List[QPixmap]:
        pixmaps = []
        for img in images:
            pixmap = self._cv_to_pixmap(img, max_size)
            pixmaps.append(pixmap)
        return pixmaps

    def create_thumbnails(self, images: List[np.ndarray], thumb_size: int = 40) -> List[QPixmap]:
        thumbnails = []
        for img in images:
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid image dimensions: {w}x{h}")
            scale = thumb_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pixmap = self._cv_to_pixmap(resized, (thumb_size, thumb_size))
            thumbnails.append(pixmap)
        return thumbnails

    def _cv_to_pixmap(self, cv_img: np.ndarray, max_size: Optional[Tuple[int, int]] = None) -> QPixmap:
        # Qt has no 16-bit-per-channel RGB format, so previews are always shown
        # from an 8-bit copy. Only the display path is narrowed; the frame kept
        # for processing keeps its full depth.
        rgb_img = cv2.cvtColor(bitdepth.to_display8(cv_img), cv2.COLOR_BGR2RGB)

        if max_size is not None:
            h, w = rgb_img.shape[:2]
            max_w, max_h = max_size

            scale = min(max_w / w, max_h / h)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # QImage reads the raw buffer against the stride it is given, so it has
        # to be handed 8-bit interleaved data in contiguous memory. It cannot
        # detect a mismatch: a wrong depth or layout renders as scrambled colour
        # over torn geometry rather than raising.
        rgb_img = np.ascontiguousarray(rgb_img)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w

        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        return QPixmap.fromImage(q_img)

    def get_image_info(self, index: int) -> dict:
        if not self.images or index < 0 or index >= len(self.images):
            return {}

        img = self.images[index]
        path = self.image_paths[index] if index < len(self.image_paths) else ""

        return {
            'index': index,
            'path': path,
            'filename': os.path.basename(path) if path else "",
            'shape': img.shape,
            'size': f"{img.shape[1]}x{img.shape[0]}",
            'channels': img.shape[2] if len(img.shape) > 2 else 1,
            'dtype': img.dtype
        }

    def get_image_timestamp(self, filepath: str) -> Optional[float]:
        if not PIL_AVAILABLE:
            return None

        try:
            with PILImage.open(filepath) as img:
                exif_data = img._getexif()
                if exif_data is None:
                    return None

                date_time_original = None
                for tag_id, value in exif_data.items():
                    tag_name = PILImage.ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == 'DateTimeOriginal':
                        date_time_original = value
                        break

                if date_time_original:
                    dt = datetime.strptime(date_time_original, '%Y:%m:%d %H:%M:%S')
                    return dt.timestamp()
        except Exception as e:
            pass

        return None

    def load_images_with_timestamps(
        self,
        folder_path: str,
        sort_by: str = 'timestamp',
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[bool, str, List[Tuple[str, np.ndarray, Optional[float]]], List[str]]:
        if not os.path.isdir(folder_path):
            return False, "Selected path is not a valid directory", [], []

        image_files = []
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_FORMATS:
                full_path = os.path.join(folder_path, filename)
                image_files.append((filename, full_path))

        if not image_files:
            return False, "No supported image files found in the folder", [], []

        decoded = self._load_files_parallel(image_files, progress_callback=progress_callback)

        loaded_data = []
        failed_count = 0
        for (filename, full_path), img in zip(image_files, decoded):
            if img is None:
                failed_count += 1
            else:
                timestamp = self.get_image_timestamp(full_path)
                loaded_data.append((filename, full_path, img, timestamp))

        if not loaded_data:
            return False, "Could not load any image files", [], []

        def get_sort_key(item):
            if sort_by == 'timestamp':
                ts = item[3]
                if ts is not None:
                    return ts
                return float('inf')
            return item[0]

        loaded_data.sort(key=get_sort_key)

        successful_count = len(loaded_data)
        message = f"Loaded {successful_count} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        result = [(item[1], item[2], item[3]) for item in loaded_data]
        filenames = [item[0] for item in loaded_data]

        return True, message, result, filenames

    def split_by_count(
        self,
        images_with_times: List[Tuple[str, np.ndarray, Optional[float]]],
        count_per_stack: int
    ) -> List[List[Tuple[str, np.ndarray, Optional[float]]]]:
        if count_per_stack <= 0:
            count_per_stack = 1

        stacks = []
        for i in range(0, len(images_with_times), count_per_stack):
            stack = images_with_times[i:i + count_per_stack]
            if stack:
                stacks.append(stack)

        return stacks

    def split_by_time_threshold(
        self,
        images_with_times: list,
        threshold_seconds: float
    ) -> list:
        if threshold_seconds <= 0:
            threshold_seconds = 1.0

        stacks = []
        current_stack = []

        for i, (path, img, timestamp) in enumerate(images_with_times):
            if i == 0:
                current_stack.append((path, img, timestamp))
                continue

            if timestamp is None:
                current_stack.append((path, img, timestamp))
                continue

            prev_timestamp = images_with_times[i - 1][2]
            if prev_timestamp is None:
                current_stack.append((path, img, timestamp))
                continue

            time_diff = timestamp - prev_timestamp
            if time_diff > threshold_seconds:
                if current_stack:
                    stacks.append(current_stack)
                current_stack = [(path, img, timestamp)]
            else:
                current_stack.append((path, img, timestamp))

        if current_stack:
            stacks.append(current_stack)

        return stacks

    def get_stack_info(
        self,
        stacks: List[List[Tuple[str, np.ndarray, Optional[float]]]]
    ) -> List[Dict[str, Any]]:
        info_list = []
        for i, stack in enumerate(stacks):
            count = len(stack)
            first_img = stack[0][1]
            height, width = first_img.shape[:2]

            timestamps = [item[2] for item in stack if item[2] is not None]
            if timestamps:
                first_time = datetime.fromtimestamp(min(timestamps)).strftime('%H:%M:%S')
                last_time = datetime.fromtimestamp(max(timestamps)).strftime('%H:%M:%S')
                time_range = f"{first_time}-{last_time}"
            else:
                time_range = "Unknown"

            info_list.append({
                'stack_index': i,
                'name': f"Stack {i + 1}",
                'count': count,
                'resolution': f"{width}x{height}",
                'time_range': time_range
            })

        return info_list
