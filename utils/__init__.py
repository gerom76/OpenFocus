"""
Utility modules for OpenFocus.

Modules:
- image_utils: Image conversion functions (pixmap <-> cv2)
- auto_params: What a setting left on 'Auto' resolved to during a render
- bitdepth: 8/16-bit depth policy and conversion
- jxl: JPEG XL encoding and decoding, optional and off when imagecodecs is absent
- dng: DNG reading (LibRaw for camera files, verbatim for our own) and writing
- ljpeg: lossless JPEG encoding, the codec DNG's lossless compression is built on
- ui_utils: UI-related functions (message boxes, dialogs)
- metadata: EXIF passthrough and OpenFocus XMP for saved results
- validators: Validation and utility functions
- platform_utils: Cross-platform utilities (font detection, OS detection)
- torch_env: PyTorch availability checks, including frozen-build stub handling
"""

# Imported first: it disarms a code-less torch/ directory left in a frozen
# bundle, and must run before anything else can import PyTorch.
from utils import torch_env  # noqa: F401

from utils.image_utils import (
    pixmap_to_cv2,
    cv2_to_pixmap,
    get_imwrite_params,
    ensure_bgr,
    read_image_any_depth,
    write_image,
)

from utils import auto_params
from utils import dng
from utils import jxl
from utils import metadata
from utils.metadata import RenderMetadata

from utils import bitdepth
from utils.bitdepth import (
    MODE_AUTO,
    MODE_8,
    MODE_16,
    VALID_MODES,
    apply_load_mode,
    bits_for,
    convert,
    depth_label,
    describe,
    from_float01,
    get_mode,
    is_high_depth,
    max_value,
    prepare_for_write,
    set_mode,
    stack_dtype,
    stack_summary,
    supports_16bit,
    to_analysis8,
    to_display8,
    to_float01,
    unify,
)

from utils.ui_utils import (
    fit_list_rows_to_thumbnails,
    log_message_box,
    exec_message_box,
    show_message_box,
    show_warning_box,
    show_error_box,
    show_success_box,
    show_custom_message_box,
    resource_path,
)

from utils.platform_utils import (
    get_os_type,
    get_ui_font_family,
    get_monospace_font_family,
    get_default_font,
    is_windows,
    is_macos,
    is_linux,
)

from utils.validators import (
    normalize_kernel_size,
    get_algorithm_from_checkboxes,
    LabelAdder,
    LabelConfig,
)

__all__ = [
    # platform_utils
    'get_os_type',
    'get_ui_font_family',
    'get_monospace_font_family',
    'get_default_font',
    'is_windows',
    'is_macos',
    'is_linux',
    # image_utils
    'pixmap_to_cv2',
    'cv2_to_pixmap',
    'get_imwrite_params',
    'ensure_bgr',
    'read_image_any_depth',
    'write_image',
    # auto_params
    'auto_params',
    # jxl
    'jxl',
    # dng
    'dng',
    # metadata
    'metadata',
    'RenderMetadata',
    # bitdepth
    'bitdepth',
    'MODE_AUTO',
    'MODE_8',
    'MODE_16',
    'VALID_MODES',
    'apply_load_mode',
    'bits_for',
    'convert',
    'depth_label',
    'describe',
    'from_float01',
    'get_mode',
    'is_high_depth',
    'max_value',
    'prepare_for_write',
    'set_mode',
    'stack_dtype',
    'stack_summary',
    'supports_16bit',
    'to_analysis8',
    'to_display8',
    'to_float01',
    'unify',
    # ui_utils
    'fit_list_rows_to_thumbnails',
    'log_message_box',
    'exec_message_box',
    'show_message_box',
    'show_warning_box',
    'show_error_box',
    'show_success_box',
    'show_custom_message_box',
    'resource_path',
    # validators
    'normalize_kernel_size',
    'get_algorithm_from_checkboxes',
    'LabelAdder',
    'LabelConfig',
]
