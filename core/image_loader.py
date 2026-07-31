"""
Image stack loader module
Responsible for loading image stacks from folders and generating thumbnails
"""

import os
import cv2
import numpy as np
import concurrent.futures
import time
from collections import OrderedDict
from datetime import datetime
from typing import Callable, List, Tuple, Optional, Dict, Any, Sequence

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

from utils import bitdepth, dng, jxl
from utils.image_utils import ensure_bgr, read_image_any_depth
from core import downsample, gpu_decode, memory

# Progress lines are throttled to this interval so a large stack does not
# flood the Qt console redirect, which is far slower than the decode itself.
_PRINT_INTERVAL = 0.1


def _bgra_to_pixmap(bgra: np.ndarray) -> QPixmap:
    """Wrap a contiguous BGRA buffer as a QPixmap.

    BGRA maps straight onto QImage.Format_RGB32, so this is a copy without a
    per-pixel conversion. The .copy() is essential: fromImage() on an already
    RGB32 image may share the buffer instead of copying, and this buffer is
    numpy memory that dies with `bgra` - the pixmap would dangle. The copy is
    Qt-owned and still a straight memcpy, no swizzle.
    """
    h, w = bgra.shape[:2]
    qimg = QImage(bgra.data, w, h, 4 * w, QImage.Format.Format_RGB32)
    return QPixmap.fromImage(qimg.copy())


def _to_display_bgra(img: np.ndarray) -> np.ndarray:
    """Full-resolution 8-bit BGRA view of a frame, ready for _bgra_to_pixmap."""
    disp8 = bitdepth.to_display8(img)
    h, w = disp8.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    return np.ascontiguousarray(cv2.cvtColor(disp8, cv2.COLOR_BGR2BGRA))


class LazyPixmapStack:
    """Full-resolution display pixmaps, built on access and cached briefly.

    A stack used to be converted to one QPixmap per frame up front. Qt stores
    those at 32 bits per pixel, so a 24 MP frame costs ~98 MB and a 333-frame
    stack ~33 GB - all of it to display a single frame at a time. Frames are
    converted here on demand instead, and only the few most recently viewed
    are kept, which bounds the display cost at a few hundred MB however large
    the stack is.

    Nothing on screen changes: the pixmap handed out is still full resolution,
    so the magnifier zooms into the same pixels it always did. Scrubbing the
    slider now pays a conversion per frame (tens of ms at 24 MP), which is the
    same work that used to be done for every frame before the stack opened.

    Holds `images` by reference and caches by index, so anything that mutates
    that list - the delete and transform paths - has to reload the stack
    afterwards, which replaces this object. They all already do.
    """

    # Enough to cover a frame plus its neighbours while stepping through the
    # stack, without the cache itself becoming the memory problem: 4 frames of
    # 24 MP is ~390 MB.
    MAX_CACHED = 4

    def __init__(self, images: Sequence[np.ndarray]):
        self._images = images
        self._cache: "OrderedDict[int, QPixmap]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._images)

    def __bool__(self) -> bool:
        return len(self._images) > 0

    def __getitem__(self, index: int) -> QPixmap:
        pixmap = self._cache.get(index)
        if pixmap is not None:
            self._cache.move_to_end(index)
            return pixmap

        pixmap = _bgra_to_pixmap(_to_display_bgra(self._images[index]))
        self._cache[index] = pixmap
        while len(self._cache) > self.MAX_CACHED:
            self._cache.popitem(last=False)
        return pixmap

    def clear_cache(self) -> None:
        """Drop the cached pixmaps; the next access rebuilds what it needs."""
        self._cache.clear()


class ImageStackLoader:
    """Image stack loader"""

    # Nikon RAW and DNG, all developed by rawpy (LibRaw). A DNG that OpenFocus
    # wrote itself is already demosaiced and is read verbatim instead - see
    # read_image_bgr - but it shares the extension, so it shares the entry here.
    RAW_FORMATS = {'.nef', '.nrw', '.dng'}
    # JPEG XL, requires imagecodecs (libjxl); absent it, the format is simply
    # not a supported input, the same way RAW is not without rawpy.
    JXL_FORMATS = set(jxl.extensions())
    SUPPORTED_FORMATS = ({'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
                         | (RAW_FORMATS if RAWPY_AVAILABLE else set())
                         | JXL_FORMATS)
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    # Formats nvJPEG can decode on the GPU; everything else stays on OpenCV.
    # JPEG XL is absent on purpose - no GPU JPEG XL decoder exists, see utils.jxl.
    GPU_DECODE_FORMATS = {'.jpg', '.jpeg'}

    # How many GPU JPEGs to decode per burst. Progress and the stop flag are
    # only serviced between bursts, so this bounds Stop-button latency.
    GPU_CHUNK = 16

    # Full frames a single RAW worker holds at its peak: LibRaw's internal
    # 4-channel image, the RGB result it hands back, and numpy's copy of it.
    # Used to size the decode thread pool against free memory.
    RAW_DECODE_FRAMES = 4

    # The same figure for JPEG XL: libjxl decodes into float internally, so a
    # decode peaks at about three times the frame it hands back. Measured on
    # 12 MP frames at both 8 and 16 bits.
    JXL_DECODE_FRAMES = 3

    # Warn once the projected stack passes this share of free memory. Below it
    # the load is merely large; above it the machine will start swapping.
    MEMORY_WARN_SHARE = 0.8

    def __init__(self):
        self.image_paths = []
        self.images = []
        self.thumbnail_size = (600, 400)
        # Cooperative stop: request_stop() may be called (from a Stop button
        # handler running inside the progress callback's processEvents) while
        # a load is under way; the load loops notice it at the next file or
        # GPU-chunk boundary. `cancelled` reports how the last load ended.
        self._cancel_requested = False
        self.cancelled = False

    def request_stop(self) -> None:
        """Ask the running load to stop at its next checkpoint."""
        self._cancel_requested = True
        print("[Loader] Stop requested - finishing files already in flight...", flush=True)

    @classmethod
    def read_image_bgr(cls, full_path: str) -> Optional[np.ndarray]:
        """Read an image as a BGR array, then bring it to the active depth mode.

        The decode itself is always done at the source's native depth - RAW at
        16 bits, libjxl for JPEG XL, and IMREAD_UNCHANGED for everything else so
        a 16-bit PNG or TIFF arrives intact. bitdepth.apply_load_mode then
        narrows or widens the result according to the mode, so in the default
        auto mode a >8-bit file keeps its extra bits and an 8-bit file stays
        8-bit.

        Returns None on failure.
        """
        ext = os.path.splitext(full_path)[1].lower()
        if dng.is_dng(ext):
            # A DNG OpenFocus wrote holds demosaiced pixels, so developing it
            # would put a second tone curve and white balance on a frame that
            # already has one; it is read straight from its strips instead. A
            # camera DNG returns None here and falls through to LibRaw below.
            linear = dng.read_linear(full_path)
            if linear is not None:
                return bitdepth.apply_load_mode(ensure_bgr(linear))
        if ext in cls.RAW_FORMATS:
            if not RAWPY_AVAILABLE:
                return None
            # 8-bit output is only ever requested when the mode forces it; the
            # RAW postprocess is where the extra bits would be lost for good.
            output_bps = 8 if bitdepth.get_mode() == bitdepth.MODE_8 else 16
            # Pass a file object to support paths with non-ASCII characters
            with open(full_path, 'rb') as f:
                with rawpy.imread(f) as raw:
                    # The GPU develop and its LibRaw fallback live in one place,
                    # so a RAW read through utils.dng gets the same treatment.
                    bgr = gpu_decode.develop_raw(raw, output_bps)
            return bitdepth.apply_load_mode(bgr)

        return read_image_any_depth(full_path)

    @classmethod
    def _probe_frame_bytes(cls, full_path: str, scale_factor: float = 1.0,
                           target_long_edge: Optional[int] = None) -> Optional[int]:
        """Decoded size of one frame in bytes, read from headers only.

        Lets the loader size its thread pool and warn about the stack it is
        about to build before it allocates any of it. Returns None when the
        dimensions cannot be read cheaply, in which case callers carry on
        without the estimate rather than paying a full decode for it.
        """
        ext = os.path.splitext(full_path)[1].lower()
        try:
            own_dng = dng.probe(full_path) if dng.is_dng(ext) else None
            jxl_header = jxl.probe(full_path) if jxl.is_jxl(ext) else None
            if jxl_header is not None:
                # imagecodecs exposes no header reader, so utils.jxl parses the
                # codestream header itself - the depth it reports is the one the
                # frame decodes at, before the mode narrows or widens it.
                width, height, bits = jxl_header
                native = bitdepth.UINT16 if bits > 8 else bitdepth.UINT8
            elif own_dng is not None:
                # An OpenFocus DNG is read verbatim, so its stored depth is the
                # depth the frame arrives at - LibRaw would report 16 either way.
                width, height, bits = own_dng
                native = bitdepth.UINT16 if bits > 8 else bitdepth.UINT8
            elif ext in cls.RAW_FORMATS:
                if not RAWPY_AVAILABLE:
                    return None
                with open(full_path, 'rb') as f:
                    with rawpy.imread(f) as raw:
                        width, height = raw.sizes.width, raw.sizes.height
                # RAW always develops to 16 bits unless the mode forces 8.
                native = bitdepth.UINT8 if bitdepth.get_mode() == bitdepth.MODE_8 else bitdepth.UINT16
            elif PIL_AVAILABLE:
                with PILImage.open(full_path) as im:
                    width, height = im.size
                    native = bitdepth.UINT16 if im.mode in ("I", "I;16", "I;16B", "I;16L", "F") else bitdepth.UINT8
            else:
                return None
        except Exception:
            return None

        resized = downsample.target_size(width, height, scale_factor, target_long_edge)
        if resized is not None:
            width, height = resized
        # Frames are stored as 3-channel BGR at the depth the mode resolves to.
        return width * height * 3 * np.dtype(bitdepth.resolve_load_dtype(native)).itemsize

    def _report_stack_memory(self, frame_bytes: Optional[int], total: int) -> None:
        """Say up front how much the stack will occupy, and warn if it is too much."""
        if not frame_bytes:
            return
        projected = frame_bytes * total
        available = memory.available_bytes()
        line = (f"[Memory] Stack needs about {memory.format_bytes(projected)} "
                f"({memory.format_bytes(frame_bytes)} x {total} frame(s))")
        if available is not None:
            line += f", {memory.format_bytes(available)} free"
        print(line, flush=True)
        if available is not None and projected > available * self.MEMORY_WARN_SHARE:
            print("[Memory] That is more than this machine can comfortably hold. "
                  "Load fewer frames, or pick a smaller scale (or long edge) in the "
                  "resize dialog so frames are downsampled as they are decoded.", flush=True)

    def _load_files_parallel(
        self,
        entries: List[Tuple[str, str]],
        scale_factor: float = 1.0,
        target_long_edge: Optional[int] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Optional[np.ndarray]]:
        """Decode a list of (filename, full_path) entries in parallel.

        Decoding (cv2/rawpy) releases the GIL, so threads give near-linear
        speedup. JPEG stacks are additionally routed through nvJPEG on the GPU
        when one is available; files nvJPEG refuses (progressive JPEGs, bad
        data) silently fall back to the OpenCV path. Returns decoded images in
        input order (None for entries that failed); progress is printed as
        files complete and reported to progress_callback, followed by a
        summary line with elapsed time and throughput.
        """
        if max_workers is None:
            max_workers = min(32, os.cpu_count() or 4)
        max_workers = max(1, min(max_workers, len(entries)))

        self._cancel_requested = False
        self.cancelled = False

        total = len(entries)
        results: List[Optional[np.ndarray]] = [None] * total

        frame_bytes = (self._probe_frame_bytes(entries[0][1], scale_factor, target_long_edge)
                       if entries else None)
        self._report_stack_memory(frame_bytes, total)

        # RAW and JPEG XL are the formats whose decode holds several full frames
        # per thread, so they are the ones whose thread count has to answer to
        # free memory; JPEG and PNG decoders allocate one output buffer.
        stack_formats = {os.path.splitext(filename)[1].lower() for filename, _ in entries}
        heavy = (('RAW', self.RAW_DECODE_FRAMES) if stack_formats & self.RAW_FORMATS
                 else ('JPEG XL', self.JXL_DECODE_FRAMES) if stack_formats & self.JXL_FORMATS
                 else None)
        if heavy is not None:
            label, frames_in_flight = heavy
            bounded = memory.decode_worker_limit(frame_bytes, frames_in_flight, max_workers)
            if bounded < max_workers:
                print(f"[Memory] Limiting {label} decode to {bounded} thread(s) "
                      f"(was {max_workers}) to keep decode buffers in memory", flush=True)
                max_workers = bounded

        # The count check comes first so tiny stacks never pay for CUDA
        # context creation inside is_available().
        gpu_indices = [
            i for i, (filename, _) in enumerate(entries)
            if os.path.splitext(filename)[1].lower() in self.GPU_DECODE_FORMATS
        ]
        use_gpu = len(gpu_indices) >= gpu_decode.MIN_IMAGES and gpu_decode.is_available()
        gpu_set = set(gpu_indices) if use_gpu else set()
        if use_gpu:
            print(f"[Loader] Decoding {len(gpu_set)} JPEG(s) on GPU "
                  f"({gpu_decode.device_name()})", flush=True)

        # libjxl decodes on a thread pool of its own, so the two levels of
        # parallelism have to be divided between the cores rather than
        # multiplied over them - see utils.jxl. A stack with fewer frames than
        # cores gets the whole machine per frame.
        jxl_threads = jxl.threads_for_workers(max_workers)
        if stack_formats & self.JXL_FORMATS:
            print(f"[Loader] Decoding JPEG XL on {max_workers} worker(s) x "
                  f"{jxl_threads or os.cpu_count() or 1} libjxl thread(s)", flush=True)

        def decode(index: int, filename: str, full_path: str):
            if not os.path.exists(full_path):
                return index, filename, None, "not_found"
            try:
                img = self.read_image_bgr(full_path)
                if img is not None:
                    size = downsample.target_size(
                        img.shape[1], img.shape[0], scale_factor, target_long_edge)
                    if size is not None:
                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                return index, filename, img, None
            except Exception as e:
                return index, filename, None, str(e)

        def read_bytes(index: int, full_path: str):
            try:
                return index, np.fromfile(full_path, dtype=np.uint8)
            except Exception:
                return index, None

        start_time = time.perf_counter()
        failed = 0
        completed = 0
        last_print = 0.0

        def note(filename: str, img: Optional[np.ndarray], error: Optional[str]):
            """Record one finished file: failures always print, successes are throttled."""
            nonlocal failed, completed, last_print
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
                now = time.perf_counter()
                if completed == total or now - last_print >= _PRINT_INTERVAL:
                    last_print = now
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print(f"[{completed}/{total}] Loaded {filename} "
                          f"({img.shape[1]}x{img.shape[0]}) - avg {rate:.1f} images/s", flush=True)
            if progress_callback is not None:
                try:
                    progress_callback(completed, total)
                except Exception:
                    pass

        def drain(futures):
            """Consume decode futures; on a stop request, drop what remains."""
            for future in concurrent.futures.as_completed(futures):
                if self._cancel_requested:
                    for pending in futures:
                        pending.cancel()
                    return
                index, filename, img, error = future.result()
                results[index] = img
                note(filename, img, error)

        with jxl.decode_thread_budget(jxl_threads), \
                concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            cpu_futures = [
                executor.submit(decode, i, filename, full_path)
                for i, (filename, full_path) in enumerate(entries) if i not in gpu_set
            ]
            read_futures = [executor.submit(read_bytes, i, entries[i][1]) for i in sorted(gpu_set)]

            buffers: Dict[int, Optional[np.ndarray]] = {}
            for future in concurrent.futures.as_completed(read_futures):
                index, data = future.result()
                buffers[index] = data

            drain(cpu_futures)

            if gpu_set and not self._cancel_requested:
                order = sorted(gpu_set)
                fallback = []
                # Decode in bursts so progress (and with it the event loop,
                # where the Stop button lives) is serviced between them.
                for chunk_start in range(0, len(order), self.GPU_CHUNK):
                    if self._cancel_requested:
                        break
                    chunk = order[chunk_start:chunk_start + self.GPU_CHUNK]
                    decoded = gpu_decode.decode_jpegs(
                        [buffers.get(i) for i in chunk], scale_factor, target_long_edge)
                    for index, img in zip(chunk, decoded):
                        if img is None:
                            fallback.append(index)
                        else:
                            results[index] = bitdepth.apply_load_mode(img)
                            note(entries[index][0], results[index], None)
                if not self._cancel_requested:
                    drain([executor.submit(decode, i, *entries[i]) for i in fallback])

        self.cancelled = self._cancel_requested
        if self.cancelled:
            print(f"Loading cancelled by user after {completed} of {total} image(s)", flush=True)
        else:
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
        target_long_edge: Optional[int] = None,
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

        decoded = self._load_files_parallel(image_files, scale_factor, target_long_edge,
                                            progress_callback=progress_callback)
        if self.cancelled:
            return False, "Loading cancelled", [], []

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
        target_long_edge: Optional[int] = None,
    ) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not os.path.isfile(video_path):
            return False, "Video file does not exist", [], []

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_VIDEO_FORMATS:
            return False, f"Unsupported video format: {ext}", [], []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file", [], []

        self._cancel_requested = False
        self.cancelled = False

        loaded_images = []
        filenames = []
        frame_index = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        start_time = time.perf_counter()
        last_print = 0.0

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                if self._cancel_requested:
                    self.cancelled = True
                    print(f"Loading cancelled by user after {frame_index} frame(s)", flush=True)
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                size = downsample.target_size(
                    frame.shape[1], frame.shape[0], scale_factor, target_long_edge)
                if size is not None:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

                # Video frames always decode as 8-bit; honour a forced 16-bit
                # mode so a video stack matches the rest of the pipeline.
                loaded_images.append(bitdepth.apply_load_mode(frame))
                filenames.append(f"{video_name}_frame_{frame_index:04d}.png")
                frame_index += 1

                now = time.perf_counter()
                if total_frames > 0 and (frame_index == total_frames
                                         or now - last_print >= _PRINT_INTERVAL):
                    last_print = now
                    elapsed = now - start_time
                    rate = frame_index / elapsed if elapsed > 0 else 0.0
                    print(f"[{frame_index}/{total_frames}] Extracted frame "
                          f"({frame.shape[1]}x{frame.shape[0]}) - avg {rate:.1f} frames/s", flush=True)
                if progress_callback is not None:
                    try:
                        progress_callback(frame_index, total_frames)
                    except Exception:
                        pass
        finally:
            cap.release()

        if self.cancelled:
            return False, "Loading cancelled", [], []

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
        target_long_edge: Optional[int] = None,
    ) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not filepaths:
            return False, "No file paths provided", [], []

        entries = [(os.path.basename(p), p) for p in filepaths]
        decoded = self._load_files_parallel(entries, scale_factor, target_long_edge,
                                            progress_callback=progress_callback)
        if self.cancelled:
            return False, "Loading cancelled", [], []

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

    def create_stack_previews(
        self,
        images: List[np.ndarray],
        thumb_size: int = 40,
    ) -> Tuple[LazyPixmapStack, List[QPixmap]]:
        """Source-list thumbnails, plus a lazy view onto the display pixmaps.

        Only the thumbnails are built here. The full-resolution display pixmap
        of a frame is produced by LazyPixmapStack when that frame is actually
        shown, because materialising all of them costs ~98 MB each and only
        one is ever on screen.

        Thumbnails are downscaled at the frame's own depth and narrowed to 8
        bits afterwards, so a 16-bit stack never allocates a full-resolution
        8-bit copy just to produce a 64 px icon. The cv2/numpy work releases
        the GIL and runs across threads; only the QPixmap wrapping, which must
        happen on the GUI thread, stays serial.
        """
        def prepare(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid image dimensions: {w}x{h}")
            scale = thumb_size / max(h, w)
            small = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            return np.ascontiguousarray(cv2.cvtColor(bitdepth.to_display8(small), cv2.COLOR_BGR2BGRA))

        thumbnails: List[QPixmap] = []
        if not images:
            return LazyPixmapStack([]), thumbnails

        start_time = time.perf_counter()
        max_workers = max(1, min(8, os.cpu_count() or 4, len(images)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for thumb in executor.map(prepare, images):
                thumbnails.append(_bgra_to_pixmap(thumb))

        print(f"[Stack] Prepared {len(thumbnails)} thumbnail(s) in "
              f"{time.perf_counter() - start_time:.2f} s", flush=True)
        return LazyPixmapStack(images), thumbnails

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
        if self.cancelled:
            return False, "Loading cancelled", [], []

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
