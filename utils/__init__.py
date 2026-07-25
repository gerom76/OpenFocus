"""
Utility modules for OpenFocus.

Modules:
- image_utils: Image conversion functions (pixmap <-> cv2)
- bitdepth: 8/16-bit depth policy and conversion
- ui_utils: UI-related functions (message boxes, dialogs)
- validators: Validation and utility functions
- platform_utils: Cross-platform utilities (font detection, OS detection)
"""

from utils.image_utils import (
    pixmap_to_cv2,
    cv2_to_pixmap,
    get_imwrite_params,
    ensure_bgr,
    read_image_any_depth,
    write_image,
)

from utils import bitdepth
from utils.bitdepth import (
    MODE_AUTO,
    MODE_8,
    MODE_16,
    VALID_MODES,
    apply_load_mode,
    bits_for,
    convert,
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
    # bitdepth
    'bitdepth',
    'MODE_AUTO',
    'MODE_8',
    'MODE_16',
    'VALID_MODES',
    'apply_load_mode',
    'bits_for',
    'convert',
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
