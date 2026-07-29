import os

import cv2
import numpy as np
from typing import Optional

from PyQt6.QtGui import QPixmap, QImage

from utils import bitdepth
from utils import jxl
from utils.metadata import RenderMetadata, embed as embed_metadata


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
    built with libjxl - the mirror of how write_image encodes it. Everything
    else goes through cv2.imdecode.
    """
    if jxl.is_jxl(os.path.splitext(path)[1]):
        img = jxl.read(path)
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
        # JPEG XL lands here too: it is not encoded by OpenCV at all, so it has
        # no imwrite parameters - see utils.jxl for its quality settings.
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
                f"Save as PNG, TIFF or JPEG XL instead.")
    return None


def write_image(
    file_path: str,
    image: np.ndarray,
    announce: bool = False,
    metadata: Optional[RenderMetadata] = None,
) -> bool:
    """Write an image, narrowing it first if the container cannot hold its depth.

    PNG, TIFF and JPEG XL store 16 bits per channel; JPEG, BMP and WebP do not.
    Handing 16-bit data to a JPEG encoder does not produce a 16-bit JPEG, so the
    narrowing has to be explicit. `announce` prints one line when it happens,
    which the single-image save paths use so the loss is never silent; stack
    exports log once around the loop instead of once per frame.

    JPEG XL is encoded by utils.jxl rather than OpenCV, whose wheels are not
    built with libjxl; everything else, WebP included, goes through cv2.imwrite.

    `metadata` describes the render behind the image. When given, and when the
    container is JPEG, PNG or JPEG XL, the source EXIF and OpenFocus' XMP group
    are added to the encoded file afterwards - see utils.metadata. Metadata
    failures never fail the save: the image is already on disk by then. WebP,
    like TIFF and BMP, is written without them.
    """
    ext = os.path.splitext(file_path)[1]
    if announce and bitdepth.is_high_depth(image) and not bitdepth.supports_16bit(ext):
        print(f"[Depth] {ext or 'this format'} cannot store 16-bit; saving 8-bit. "
              f"Use PNG, TIFF or JPEG XL to keep the full depth.", flush=True)
    if ext.lower() == ".webp":
        reason = webp_size_error(image)
        if reason is not None:
            print(f"[WebP] {reason}", flush=True)
            return False
    image = bitdepth.prepare_for_write(image, ext)
    if jxl.is_jxl(ext):
        if not jxl.write(file_path, image):
            return False
    elif not cv2.imwrite(file_path, image, get_imwrite_params(ext)):
        return False

    if metadata is not None:
        embed_metadata(file_path, metadata)
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
