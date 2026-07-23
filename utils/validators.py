from typing import List, Tuple, Any, Dict
import cv2
import numpy as np

from utils import bitdepth


def _scale_colour(colour, dtype):
    """Map an 0-255 BGR colour onto the full scale of `dtype`.

    Overlay colours are authored in 8-bit units throughout the UI; a 16-bit
    frame needs them scaled or every overlay renders near-black.
    """
    if colour is None:
        return colour
    scale = bitdepth.max_value(dtype) / 255.0
    if scale == 1.0:
        return colour
    return tuple(float(c) * scale for c in colour)


def _put_text_any_depth(image, text, org, font, font_scale, colour, thickness):
    """Draw text on an image of any supported depth.

    cv2.putText asserts CV_8U, so it cannot write to a 16-bit frame at all -
    unlike cv2.rectangle, which handles both. For 16-bit the glyphs are rendered
    into an 8-bit coverage mask and composited at full depth, which also keeps
    the edge shading the mask carries.

    The 8-bit path is left as a direct putText call so existing renders stay
    byte-identical.
    """
    if image.dtype == np.uint8:
        cv2.putText(image, text, org, font, font_scale, colour, thickness)
        return image

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.putText(mask, text, org, font, font_scale, 255, thickness)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    colour_layer = np.asarray(colour[:image.shape[2]], dtype=np.float32).reshape(1, 1, -1)
    blended = image.astype(np.float32) * (1.0 - alpha) + colour_layer * alpha
    np.copyto(image, np.rint(blended).astype(image.dtype))
    return image


def normalize_kernel_size(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value = max(1, value - 1)
    return value


def get_algorithm_from_checkboxes(
    rb_a: bool,
    rb_b: bool,
    rb_c: bool,
    rb_gfg: bool,
    rb_d: bool,
    default: str = "guided_filter"
) -> str:
    if rb_a:
        return "guided_filter"
    elif rb_b:
        return "dct"
    elif rb_c:
        return "dtcwt"
    elif rb_gfg:
        return "gfgfgf"
    elif rb_d:
        return "stackmffv4"
    return default


def crop_roi(
    images: List[Any],
    roi_rect: Tuple[float, float, float, float]
) -> Tuple[List[Any], None]:
    rx, ry, rw, rh = (
        int(roi_rect[0]),
        int(roi_rect[1]),
        int(roi_rect[2]),
        int(roi_rect[3])
    )

    if len(images) == 0:
        return images, None

    h, w = images[0].shape[:2]
    rx = max(0, min(rx, w))
    ry = max(0, min(ry, h))
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))

    cropped = []
    for img in images:
        if img.ndim == 3:
            crop = img[ry:ry+rh, rx:rx+rw, :]
        else:
            crop = img[ry:ry+rh, rx:rx+rw]
        cropped.append(crop)

    return cropped, None


def paste_roi(
    result: Any,
    base: Any,
    roi_rect: Tuple[int, int, int, int]
) -> Any:
    rx, ry, rw, rh = roi_rect

    if base.ndim == 3 and result.ndim == 3:
        base[ry:ry+rh, rx:rx+rw, :] = result[:rh, :rw, :]
    elif base.ndim == 2 and result.ndim == 2:
        base[ry:ry+rh, rx:rx+rw] = result[:rh, :rw]
    elif base.ndim == 3 and result.ndim == 2:
        for c in range(3):
            base[ry:ry+rh, rx:rx+rw, c] = result[:rh, :rw]

    return base


class LabelConfig:
    def __init__(self) -> None:
        self.target_stack: int = 1
        self.format: str = "{value}"
        self.starting_value: int = 1
        self.interval: int = 1
        self.x_location: int = 20
        self.y_location: int = 80
        self.font_size: int = 80
        self.font_family: str = "Arial"
        self.text: str = ""
        self.range: str = "All"
        self.transparent_bg: bool = True
        self.bg_color: Tuple[int, int, int] = (0, 0, 0)
        self.font_color: Tuple[int, int, int] = (255, 255, 255)

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LabelAdder:
    def __init__(self) -> None:
        self.config = LabelConfig()
        self.font_mapping: Dict[str, int] = {
            'Arial': cv2.FONT_HERSHEY_SIMPLEX,
            'Times New Roman': cv2.FONT_HERSHEY_SIMPLEX,
            'Courier New': cv2.FONT_HERSHEY_TRIPLEX,
            'Calibri': cv2.FONT_HERSHEY_SIMPLEX,
            'Verdana': cv2.FONT_HERSHEY_SIMPLEX,
            'Georgia': cv2.FONT_HERSHEY_SIMPLEX,
            'Helvetica': cv2.FONT_HERSHEY_SIMPLEX,
            'Comic Sans MS': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            'Impact': cv2.FONT_HERSHEY_SIMPLEX,
            'Lucida Console': cv2.FONT_HERSHEY_TRIPLEX,
            'Tahoma': cv2.FONT_HERSHEY_SIMPLEX,
            'Trebuchet MS': cv2.FONT_HERSHEY_SIMPLEX,
            'Palatino': cv2.FONT_HERSHEY_SIMPLEX,
            'Garamond': cv2.FONT_HERSHEY_SIMPLEX,
            'Bookman': cv2.FONT_HERSHEY_SIMPLEX
        }

    def add_label_to_image(self, image: np.ndarray, index: int) -> np.ndarray:
        try:
            format_str = self.config.format
            starting_value = self.config.starting_value
            interval = self.config.interval
            x_location = self.config.x_location
            y_location = self.config.y_location
            font_size = self.config.font_size
            font_family = self.config.font_family
            text = self.config.text
            transparent_bg = self.config.transparent_bg
            bg_color = self.config.bg_color
            font_color = self.config.font_color

            current_value = starting_value + index * interval

            if '{value}' in format_str:
                label_text = format_str.replace('{value}', str(current_value))
            else:
                label_text = text

            # Label colours are authored as 0-255 BGR. Drawing them onto a
            # 16-bit frame unscaled would put "white" at level 255 out of 65535,
            # i.e. all but black, so they are scaled to the frame's full scale.
            bg_color = _scale_colour(bg_color, image.dtype)
            font_color = _scale_colour(font_color, image.dtype)

            font = self.font_mapping.get(font_family, cv2.FONT_HERSHEY_SIMPLEX)

            font_scale = font_size / 30.0
            thickness = max(1, int(font_size / 15))

            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            if not transparent_bg:
                cv2.rectangle(
                    image,
                    (x_location, y_location - text_height - 10),
                    (x_location + text_width + 10, y_location + baseline + 5),
                    bg_color,
                    -1
                )

            _put_text_any_depth(
                image,
                label_text,
                (x_location + 5, y_location),
                font,
                font_scale,
                font_color,
                thickness
            )

            return image
        except Exception:
            return image

