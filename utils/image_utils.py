import os

import cv2
import numpy as np
from typing import Optional

from PyQt6.QtGui import QPixmap, QImage

from utils import bitdepth
from utils import dng
from utils import jxl
from utils.metadata import (
    RenderMetadata,
    copy_exif,
    embed as embed_metadata,
    read_source_exif,
)


# WebP is written losslessly, like every other container here: OpenCV selects
# lossless encoding for any quality above 100.
WEBP_LOSSLESS_QUALITY = 101

# libwebp cannot address more than 16383 px on either side, whatever the encoder
# settings are. Past that the encode just fails, so the limit is checked before
# the write and reported as itself rather than as a bare "could not write".
WEBP_MAX_DIMENSION = 16383


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Force a decoded frame to 3-channel BGR without changing its bit depth.

    IMREAD_UNCHANGED preserves depth but also preserves channel count, so
    greyscale and alpha have to be normalised here. IMREAD_COLOR used to do this
    implicitly, at the cost of forcing everything down to 8-bit.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def read_image_any_depth(path: str, apply_mode: bool = True) -> Optional[np.ndarray]:
    """Read an image file as BGR, keeping 16-bit sources at 16 bits.

    Used by the folder-input paths of the fusion methods, which would otherwise
    silently narrow a 16-bit stack the moment it was passed as a directory
    rather than as preloaded arrays. Returns None if the file cannot be decoded.

    JPEG XL is decoded by utils.jxl rather than OpenCV, whose wheels are not
    built with libjxl - the mirror of how write_image encodes it. DNG goes to
    utils.dng, which reads its own linear output verbatim and develops a camera
    DNG through LibRaw. Everything else goes through cv2.imdecode.
    """
    ext = os.path.splitext(path)[1]
    if jxl.is_jxl(ext):
        img = jxl.read(path)
    elif dng.is_dng(ext):
        # A camera DNG has to be developed, and the depth mode decides at what
        # depth; an OpenFocus linear DNG ignores this and comes back as stored.
        img = dng.read(path, output_bps=8 if bitdepth.get_mode() == bitdepth.MODE_8 else 16)
    else:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = ensure_bgr(img)
    return bitdepth.apply_load_mode(img) if apply_mode else img


def get_imwrite_params(extension: str) -> list:
    """Get OpenCV imwrite parameters for maximum quality based on file extension.

    Args:
        extension: File extension, with or without a leading dot (e.g. '.jpg', 'jpg')

    Returns:
        List of parameter tuples for cv2.imwrite, or empty list if no special params needed
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        # JPG: 100 quality (highest, default is ~95)
        return [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif ext in (".png",):
        # PNG: 0 compression (no compression, default is 3)
        return [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif ext in (".tif", ".tiff"):
        # TIFF: LZW compression disabled (compression flag 1 = no compression)
        return [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    elif ext in (".webp",):
        # WebP: above 100 the encoder switches to lossless (default is 100, lossy)
        return [cv2.IMWRITE_WEBP_QUALITY, WEBP_LOSSLESS_QUALITY]
    elif ext in (".bmp",):
        # BMP: No quality parameters needed (always lossless)
        return []
    else:
        # JPEG XL and DNG land here too: neither is encoded by OpenCV at all, so
        # they have no imwrite parameters - see utils.jxl for JPEG XL's quality
        # settings, and utils.dng for DNG's compression mode and preview, which
        # are module settings rather than per-call parameters.
        return []


def webp_size_error(image: np.ndarray) -> Optional[str]:
    """Why libwebp will refuse this image, or None if it will accept it.

    WebP's canvas is capped at 16383 px per side by the format itself, so a
    fused panorama wider than that cannot be written however it is encoded. The
    encoder only reports a generic failure, hence the check up front.
    """
    if image is None:
        return None
    height, width = image.shape[:2]
    if max(width, height) > WEBP_MAX_DIMENSION:
        return (f"WebP cannot store images larger than {WEBP_MAX_DIMENSION} px "
                f"on a side; this one is {width}x{height}. "
                f"Save as PNG, TIFF, JPEG XL or DNG instead.")
    return None


def write_image(
    file_path: str,
    image: np.ndarray,
    announce: bool = False,
    metadata: Optional[RenderMetadata] = None,
    source_path: Optional[str] = None,
) -> bool:
    """Write an image, narrowing it first if the container cannot hold its depth.

    PNG, TIFF, JPEG XL and DNG store 16 bits per channel; JPEG, BMP and WebP do
    not. Handing 16-bit data to a JPEG encoder does not produce a 16-bit JPEG, so
    the narrowing has to be explicit. `announce` prints one line when it happens,
    which the single-image save paths use so the loss is never silent; stack
    exports log once around the loop instead of once per frame.

    JPEG XL is encoded by utils.jxl rather than OpenCV, whose wheels are not
    built with libjxl, and DNG by utils.dng, which OpenCV cannot write at all;
    everything else, WebP included, goes through cv2.imwrite. Both of those
    modules carry their own settings - DNG's compression mode and fast-load
    preview among them - so nothing about the codec has to be threaded through
    here from the save paths that share this function.

    `metadata` describes the render behind the image. When given, and when the
    container is JPEG, PNG or JPEG XL, the source EXIF and OpenFocus' XMP group
    are added to the encoded file afterwards - see utils.metadata. Metadata
    failures never fail the save: the image is already on disk by then. WebP,
    like TIFF and BMP, is written without them.

    `source_path` is the file this image came from, for the saves that are not
    renders - a processed input frame, above all. Its EXIF block is carried into
    the written file, and nothing else is: there is no render to describe, so no
    XMP packet is added. A `metadata` record supersedes it, since that already
    names the source it should inherit from.

    DNG takes its EXIF a different way. A TIFF cannot have tags spliced in after
    the fact without moving every offset behind them, so the block is handed to
    the writer and laid out with the rest of the file.
    """
    ext = os.path.splitext(file_path)[1]
    if announce and bitdepth.is_high_depth(image) and not bitdepth.supports_16bit(ext):
        print(f"[Depth] {ext or 'this format'} cannot store 16-bit; saving 8-bit. "
              f"Use PNG, TIFF, JPEG XL or DNG to keep the full depth.", flush=True)
    if ext.lower() == ".webp":
        reason = webp_size_error(image)
        if reason is not None:
            print(f"[WebP] {reason}", flush=True)
            return False
    image = bitdepth.prepare_for_write(image, ext)
    if jxl.is_jxl(ext):
        if not jxl.write(file_path, image):
            return False
    elif dng.is_dng(ext):
        source = metadata.source_path if metadata is not None else source_path
        if not dng.write(file_path, image, exif=read_source_exif(source)):
            return False
    elif not cv2.imwrite(file_path, image, get_imwrite_params(ext)):
        return False

    if metadata is not None:
        embed_metadata(file_path, metadata)
    elif source_path:
        copy_exif(file_path, source_path)
    return True


def pixmap_to_cv2(pixmap: QPixmap) -> Optional[np.ndarray]:
    try:
        qimage = pixmap.toImage()
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)

        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()

        ptr = qimage.bits()
        ptr.setsize(bytes_per_line * height)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        return bgr_image
    except Exception:
        return None


def cv2_to_pixmap(cv2_img: np.ndarray) -> QPixmap:
    try:
        # Qt has no 16-bit-per-channel RGB format, so anything headed for the
        # screen is narrowed here. Processing buffers keep their full depth.
        cv2_img = bitdepth.to_display8(cv2_img)
        if len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
            rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)

        rgb_image = np.ascontiguousarray(rgb_image)

        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)

        return pixmap
    except Exception:
        return QPixmap()
