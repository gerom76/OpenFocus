import datetime
import os
from typing import Any, Optional

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from constants import (
    DCT_BLOCK_SIZE_DEFAULT, DCT_PLATEAU_DEFAULT,
    DEPTH_SMOOTHING_DEFAULT,
    PYRAMID_BASE_DEFAULT, PYRAMID_COHERENCE_DEFAULT, PYRAMID_SELECTIVITY_DEFAULT,
)
from core import contrast
from core.app import OpenFocusApplication
from dialogs import DurationDialog, ExportFormatDialog
from ui.styles import PROGRESS_DIALOG_STYLE
from utils import (
    bitdepth,
    carries_exif,
    dng,
    jxl,
    write_image,
    show_error_box,
    show_message_box,
    show_success_box,
    show_warning_box,
)
from core.workers import GifSaverWorker
from locales import trans

ALLOWED_EXPORT_EXTENSION_MAP = {
    ".png": ".png",
    ".jpg": ".jpg",
    ".bmp": ".bmp",
    ".tif": ".tif",
    ".tiff": ".tiff",
    ".webp": ".webp",
}

# JPEG XL needs an encoder this build may not have; offering it in the dialogs
# when it cannot be written would only produce failed saves, so it is added to
# the accepted extensions only once utils.jxl reports a backend.
if jxl.is_available():
    ALLOWED_EXPORT_EXTENSION_MAP[".jxl"] = ".jxl"

# DNG is written by utils.dng without help from any optional package, but a build
# that cannot read one back has no business offering it as a save format, so it
# follows the same availability gate - see utils.dng.is_available.
if dng.is_available():
    ALLOWED_EXPORT_EXTENSION_MAP[".dng"] = ".dng"

EXPORT_EXTENSION_ALIASES = {
    ".jpeg": ".jpg",
    ".jpe": ".jpg",
    ".jfif": ".jpg",
    ".jp2": ".jpg",
}

DEFAULT_EXPORT_EXTENSION = ".png"

# Formats offered when a stack is exported to a folder, in the same order as the
# save-dialog entries. ".tiff" is left out because it writes the same container
# as ".tif"; JPEG XL is appended only where it can be encoded.
EXPORT_FORMAT_CHOICES = [
    (".jpg", "JPG"),
    (".png", "PNG"),
    (".bmp", "Bitmap"),
    (".tif", "TIFF"),
    (".webp", "WebP"),
]

# Which save-dialog entry each extension belongs to, so a remembered format can
# be pre-selected in the dialog's format dropdown.
EXPORT_FILTER_LABELS = {
    ".jpg": "JPG Files (*.jpg)",
    ".png": "PNG Files (*.png)",
    ".bmp": "Bitmap Files (*.bmp)",
    ".tif": "TIFF Files (*.tif *.tiff)",
    ".tiff": "TIFF Files (*.tif *.tiff)",
    ".webp": "WebP Files (*.webp)",
    ".jxl": "JPEG XL Files (*.jxl)",
    ".dng": "DNG Files (*.dng)",
}


def export_format_choices() -> list[tuple[str, str]]:
    """(extension, name) pairs the folder-export dialog offers.

    JPEG XL and DNG are appended only where they can be written, so the list
    matches what ALLOWED_EXPORT_EXTENSION_MAP will actually accept.
    """
    choices = list(EXPORT_FORMAT_CHOICES)
    if ".jxl" in ALLOWED_EXPORT_EXTENSION_MAP:
        choices.append((".jxl", "JPEG XL"))
    if ".dng" in ALLOWED_EXPORT_EXTENSION_MAP:
        choices.append((".dng", "DNG"))
    return choices


def save_dialog_filter() -> str:
    """Filter string for the save dialogs, with JPEG XL and DNG where writable."""
    supported = "*.png *.jpg *.bmp *.tif *.tiff *.webp"
    entries = [
        "JPG Files (*.jpg)",
        "PNG Files (*.png)",
        "Bitmap Files (*.bmp)",
        "TIFF Files (*.tif *.tiff)",
        "WebP Files (*.webp)",
    ]
    if jxl.is_available():
        supported += " *.jxl"
        entries.append("JPEG XL Files (*.jxl)")
    if dng.is_available():
        supported += " *.dng"
        entries.append("DNG Files (*.dng)")
    entries.append("All Files (*)")
    return f"All Supported Formats ({supported});;" + ";;".join(entries)


def save_dialog_selected_filter(extension: str) -> str:
    """Entry to pre-select in the save dialog for a given extension.

    An empty string leaves Qt on the first ("All Supported Formats") entry,
    which is also what an extension this build cannot write falls back to.
    """
    ext = (extension or "").lower()
    if ext not in ALLOWED_EXPORT_EXTENSION_MAP:
        return ""
    return EXPORT_FILTER_LABELS.get(ext, "")


class ExportManager:
    """Handles save and export workflows for the main window."""

    def __init__(self, window: Any) -> None:
        self.window = window
        self.gif_progress_dialog: Optional[QProgressDialog] = None
        self.gif_worker: Optional[GifSaverWorker] = None

    def normalize_export_path(self, file_path: str, fallback_extension: str = DEFAULT_EXPORT_EXTENSION) -> str:
        """Map user supplied paths onto the supported export extensions."""
        root, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()

        fallback = (fallback_extension or DEFAULT_EXPORT_EXTENSION).lower()
        if fallback not in ALLOWED_EXPORT_EXTENSION_MAP:
            fallback = DEFAULT_EXPORT_EXTENSION

        if not ext:
            return root + fallback

        mapped = ALLOWED_EXPORT_EXTENSION_MAP.get(ext_lower)
        if mapped:
            return root + mapped

        alias_target = EXPORT_EXTENSION_ALIASES.get(ext_lower)
        if alias_target:
            mapped_alias = ALLOWED_EXPORT_EXTENSION_MAP.get(alias_target)
            if mapped_alias:
                return root + mapped_alias

        return root + fallback

    def preferred_export_extension(self) -> str:
        """Extension the save dialogs should default to.

        The format of the last export is remembered in the settings; a format
        this build cannot write (a hand-edited config, or a JPEG XL encoder that
        is no longer installed) falls back to the default.
        """
        remembered = getattr(getattr(self.window, "settings_manager", None), "output_format", "")
        ext = (remembered or "").lower()
        return ext if ext in ALLOWED_EXPORT_EXTENSION_MAP else DEFAULT_EXPORT_EXTENSION

    def with_export_extension(self, name: str, extension: str) -> str:
        """Put `extension` on a suggested filename, replacing an image one if present.

        Output names are extensionless, but they come from the output list where a
        user-renamed entry may carry one; anything else (a dotted name such as
        "2.5x") is left intact so only the extension is appended.
        """
        root, ext = os.path.splitext(name)
        ext_lower = ext.lower()
        if ext_lower in ALLOWED_EXPORT_EXTENSION_MAP or ext_lower in EXPORT_EXTENSION_ALIASES:
            return root + extension
        return name + extension

    def ask_export_format(self) -> Optional[str]:
        """Ask which format a folder export writes, or None if the user cancels.

        The choice is remembered in the settings straight away, so it is also the
        format the next stack export and the next save dialog start on.
        """
        dialog = ExportFormatDialog(
            self.window,
            formats=export_format_choices(),
            initial=self.preferred_export_extension(),
        )
        if not dialog.exec():
            return None

        extension = dialog.selected_extension() or DEFAULT_EXPORT_EXTENSION
        settings = getattr(self.window, "settings_manager", None)
        if settings is not None:
            settings.set_output_format(extension)
        return extension

    def force_export_extension(self, name: str, extension: str) -> str:
        """Put `extension` on a source filename, dropping the one it arrived with.

        Source names carry their own extension - ".jpg", but also ".CR2" or
        ".ARW" - and a stack exported in a chosen format must not keep any of
        them. A trailing dot group that does not look like a file extension (a
        name such as "2.5x") is left alone, so only the extension is appended.
        """
        root, ext = os.path.splitext(name)
        suffix = ext[1:]
        if suffix and suffix.isalnum() and len(suffix) <= 5:
            return root + extension
        return name + extension

    def remember_export_format(self, file_path: str) -> None:
        """Record the format just exported so the next save dialog offers it again."""
        settings = getattr(self.window, "settings_manager", None)
        if settings is not None:
            settings.set_output_format(os.path.splitext(file_path)[1])

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------
    def _kernel_suffix(self) -> str:
        """Return a '_k<size>' suffix for methods that use the kernel slider.

        GuidedFilter (rb_a), DCT (rb_b), GFG-FGF (rb_gfg), Pyramid and both
        Depth Map modes rely on the kernel-size slider; DTCWT and StackMFF-V4
        ignore it, so no suffix is emitted for those.
        """
        window = self.window
        if (window.rb_a.isChecked() or window.rb_b.isChecked()
                or window.rb_gfg.isChecked() or window.rb_pyramid.isChecked()
                or window.rb_dmap_max.isChecked() or window.rb_dmap_avg.isChecked()):
            try:
                return f"_k{int(window.slider_smooth.value())}"
            except Exception:
                return ""
        return ""

    def _dct_suffix(self) -> str:
        """Return the DCT tuning that changes the result, for the filename.

        The block size and the focus-tolerance preset both change what comes
        out, so two renders that differ only in those must not land on the same
        name. Only non-default values are emitted, so existing filenames are
        unchanged for anyone who leaves the controls alone.
        """
        window = self.window
        if not window.rb_b.isChecked():
            return ""
        parts = []
        try:
            block = int(window.combo_dct_block.currentData())
            if block != DCT_BLOCK_SIZE_DEFAULT:
                parts.append(f"b{block}")
            preset = window.combo_dct_plateau.currentData()
            if preset and preset != DCT_PLATEAU_DEFAULT:
                parts.append(str(preset))
            if not window.cb_dct_blend.isChecked():
                parts.append("hard")
        except Exception:
            return ""
        return f"_{'+'.join(parts)}" if parts else ""

    def _pyramid_suffix(self) -> str:
        """Return the pyramid tuning that changes the result, for the filename.

        Same rule as the DCT suffix: only what departs from the defaults is
        named, so a render left on the defaults keeps the filename it always
        had, and two renders that differ only in tuning cannot collide.
        """
        window = self.window
        if not window.rb_pyramid.isChecked():
            return ""
        parts = []
        try:
            levels = window.combo_pyr_levels.currentData()
            if levels:
                parts.append(f"l{int(levels)}")
            # Prefixed by which control they came from: three of the presets
            # share names ('strong' is both a coherence and a base setting), and
            # a filename has to say which one moved.
            for tag, combo, default in (
                    ("sel", window.combo_pyr_selectivity, PYRAMID_SELECTIVITY_DEFAULT),
                    ("coh", window.combo_pyr_coherence, PYRAMID_COHERENCE_DEFAULT),
                    ("base", window.combo_pyr_base, PYRAMID_BASE_DEFAULT)):
                preset = combo.currentData()
                if preset and preset != default:
                    parts.append(f"{tag}-{preset}")
            if not window.cb_pyr_noise_gate.isChecked():
                parts.append("nogate")
            if not window.cb_pyr_envelope.isChecked():
                parts.append("unclamped")
        except Exception:
            return ""
        return f"_{'+'.join(parts)}" if parts else ""

    def _halo_suffix(self) -> str:
        """Return a '_h<radius>' suffix when depth-map halo suppression is on."""
        window = self.window
        if window.rb_dmap_max.isChecked() or window.rb_dmap_avg.isChecked():
            try:
                radius = int(window.slider_halo.value())
            except Exception:
                return ""
            if radius > 0:
                return f"_h{radius}"
        return ""

    def _coherent_suffix(self) -> str:
        """Return a '_ds<n>' suffix for the Depth Map (Max) coherence dial.

        Only when it differs from the default: it is on by default, so naming it
        every time would push the setting that was actually varied off the end
        of an already long filename.
        """
        window = self.window
        if not window.rb_dmap_max.isChecked():
            return ""
        try:
            smoothing = int(window.slider_depth_smooth.value())
        except Exception:
            return ""
        return "" if smoothing == DEPTH_SMOOTHING_DEFAULT else f"_ds{smoothing}"

    def _fusion_suffix(self) -> str:
        """Describe the fusion stages in the name: method, kernel, refinement."""
        window = self.window

        if window.rb_a.isChecked():
            fusion_method = "GuidedFilter"
        elif window.rb_b.isChecked():
            fusion_method = "DCT"
        elif window.rb_c.isChecked():
            fusion_method = "DTCWT"
        elif window.rb_gfg.isChecked():
            fusion_method = "GFGFGF"
        elif window.rb_pyramid.isChecked():
            fusion_method = "Pyramid"
        elif window.rb_dmap_max.isChecked():
            fusion_method = "DepthMapMax"
        elif window.rb_dmap_avg.isChecked():
            fusion_method = "DepthMapAvg"
        elif window.rb_d.isChecked():
            fusion_method = "StackMFFV4"
        else:
            fusion_method = "None"

        fusion_method += self._kernel_suffix()
        fusion_method += self._dct_suffix()
        fusion_method += self._pyramid_suffix()
        fusion_method += self._halo_suffix()
        fusion_method += self._coherent_suffix()

        # The IFCNN stage runs on top of the method above, so it reads as an addition
        if getattr(window, "cb_ifcnn", None) is not None and window.cb_ifcnn.isChecked():
            fusion_method += "+IFCNN"

        # Contrast is a post-fusion output step, so it reads as a further addition
        contrast_tag = contrast.describe(
            getattr(window, "contrast_method", contrast.METHOD_OFF),
            getattr(window, "contrast_strength", 0) / 100.0,
        )
        if contrast_tag:
            fusion_method += "+" + contrast_tag

        return fusion_method

    def _registration_suffix(self) -> str:
        """Describe the registration stages in the name."""
        window = self.window
        reg_methods = []
        if getattr(window, "cb_align_scale", None) and window.cb_align_scale.isChecked():
            reg_methods.append("Scale")
        if window.cb_align_homography.isChecked():
            reg_methods.append("Homography")
        if window.cb_align_ecc.isChecked():
            reg_methods.append("ECC")
        return "+".join(reg_methods) if reg_methods else "NoAlign"

    def generate_default_filename(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = OpenFocusApplication.VERSION
        return f"OpenFocus_{version}_{timestamp}_{self._fusion_suffix()}_{self._registration_suffix()}"

    def _suggested_output_name(self) -> str:
        """Prefer the name of the selected output entry, so saving matches the list."""
        output_list = getattr(self.window, "output_list", None)
        if output_list is not None:
            item = output_list.currentItem()
            if item is not None and item.text().strip():
                return item.text().strip()
        return self.generate_default_filename()

    def generate_default_foldername(self) -> str:
        window = self.window
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = OpenFocusApplication.VERSION

        folder_basename = "OpenFocus_Stack"
        if getattr(window, "current_folder_path", None):
            folder_basename = os.path.basename(window.current_folder_path)

        return f"{folder_basename}_{version}_{timestamp}_{self._fusion_suffix()}_{self._registration_suffix()}"

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def save_result(self) -> None:
        window = self.window
        result_to_save = None
        title = ""

        if window.fusion_result is not None:
            result_to_save = window.fusion_result
            title = trans.t("action_save")
        elif window.registration_results:
            index = window.current_result_index if window.current_result_index >= 0 else 0
            result_to_save = window.registration_results[index]
            title = trans.t("action_save")
        else:
            show_warning_box(window, trans.t("msg_no_result_title"), trans.t("msg_no_result_text"))
            return

        preferred_ext = self.preferred_export_extension()
        default_filename = self.with_export_extension(self._suggested_output_name(), preferred_ext)
        file_path, _ = QFileDialog.getSaveFileName(
            window,
            title,
            window.settings_manager.default_output_path(default_filename),
            save_dialog_filter(),
            save_dialog_selected_filter(preferred_ext),
        )

        if not file_path:
            return

        file_path = self.normalize_export_path(file_path, preferred_ext)
        window.settings_manager.set_output_dir(os.path.dirname(file_path))
        self.remember_export_format(file_path)

        try:
            index = 0 if window.fusion_result is not None else window.current_result_index
            if index < 0:
                index = 0
            # Contrast is a fusion-output setting, so it applies only when the
            # thing being saved is the fused result - a bare registration result
            # (no fusion run) is saved as aligned.
            content = result_to_save
            metadata = None
            if window.fusion_result is not None:
                content = window.apply_output_contrast(result_to_save)
                # Only a fused result carries render metadata; a bare aligned
                # frame is saved the way it always was.
                metadata = window.output_manager.metadata_for_current()
            image_to_save = window.label_manager.prepare_bgr_image("registered", content, index)
            if write_image(file_path, image_to_save, announce=True, metadata=metadata):
                show_message_box(
                    window,
                    trans.t("msg_success"),
                    trans.t("msg_image_saved_text"),
                    trans.t("msg_image_saved_info").format(path=file_path),
                    QMessageBox.Icon.Information,
                )
            else:
                show_message_box(
                    window,
                    trans.t("msg_save_failed_title"),
                    trans.t("msg_save_failed_text"),
                    trans.t("msg_save_failed_info_write"),
                    QMessageBox.Icon.Critical,
                )
        except cv2.error as exc:
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_opencv").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_unexpected").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )

    def save_result_stack(self) -> None:
        window = self.window
        if not window.registration_results:
            show_warning_box(window, trans.t("msg_no_stack_title"), trans.t("msg_no_stack_text"))
            return

        default_foldername = self.generate_default_foldername()
        folder_path = QFileDialog.getExistingDirectory(
            window,
            "Select Folder to Save Registration Stack",
            window.settings_manager.default_output_path(default_foldername),
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        # Remember the parent so the next export starts beside the created stack folder.
        window.settings_manager.set_output_dir(os.path.dirname(folder_path))

        # Source names keep their own writable extension; everything else (an
        # unwritable source format, or a generated name) uses the chosen format.
        preferred_ext = self.preferred_export_extension()

        try:
            saved_count = 0
            for index, image in enumerate(window.registration_results):
                image_to_save = window.label_manager.prepare_bgr_image("registered", image, index)
                if index < len(window.image_filenames):
                    filename = window.image_filenames[index]
                else:
                    filename = f"registered_{index + 1:04d}{preferred_ext}"
                file_path = os.path.join(folder_path, filename)
                file_path = self.normalize_export_path(file_path, preferred_ext)
                if write_image(file_path, image_to_save):
                    saved_count += 1

            show_message_box(
                window,
                trans.t("msg_success"),
                trans.t("msg_stack_saved_text"),
                trans.t("msg_stack_saved_info").format(
                    saved=saved_count,
                    total=len(window.registration_results),
                    folder=folder_path,
                ),
                QMessageBox.Icon.Information,
            )
        except cv2.error as exc:
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_opencv_info").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_unexpected_info").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )

    def save_as_gif(self, target_type: str = "registered") -> None:
        window = self.window

        if target_type == "registered":
            if not window.registration_results:
                show_warning_box(window, trans.t("msg_no_registered_images_title"), trans.t("msg_no_registered_images_text"))
                return
            images_to_save = window.registration_results
        else:
            if not window.raw_images:
                show_warning_box(window, trans.t("msg_no_input_images_title"), trans.t("msg_no_input_images_text"))
                return
            images_to_save = window.raw_images

        duration_dialog = DurationDialog(window)
        if not duration_dialog.exec():
            return

        duration_ms = duration_dialog.get_duration()
        default_filename = self._suggested_output_name() + ".gif"
        file_path, _ = QFileDialog.getSaveFileName(
            window,
            "Save as GIF",
            window.settings_manager.default_output_path(default_filename),
            "GIF Files (*.gif);;All Files (*)",
        )

        if not file_path:
            return

        window.settings_manager.set_output_dir(os.path.dirname(file_path))

        self.gif_progress_dialog = QProgressDialog(
            trans.t("msg_gif_saving_text"),
            trans.t("btn_cancel"),
            0,
            0,
            window,
        )
        self.gif_progress_dialog.setWindowTitle(trans.t("msg_gif_processing_title"))
        self.gif_progress_dialog.setStyleSheet(PROGRESS_DIALOG_STYLE)
        self.gif_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.gif_progress_dialog.setMinimumDuration(0)
        self.gif_progress_dialog.setCancelButton(None)
        self.gif_progress_dialog.show()

        duration_sec = duration_ms / 1000.0
        self.gif_worker = GifSaverWorker(images_to_save, file_path, duration_sec, window.label_manager, target_type)
        self.gif_worker.finished_signal.connect(
            lambda success, msg: self.on_gif_saved(success, msg, duration_ms)
        )
        self.gif_worker.start()

    def on_gif_saved(self, success: bool, message: str, duration_ms: int) -> None:
        window = self.window

        if self.gif_progress_dialog:
            self.gif_progress_dialog.close()
            self.gif_progress_dialog = None

        self.gif_worker = None

        if success:
            show_success_box(
                window,
                trans.t("msg_success"),
                trans.t("msg_gif_saved_text"),
                trans.t("msg_gif_saved_info").format(message=message, duration=duration_ms),
            )
        else:
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_gif_save_failed_text"),
                trans.t("msg_gif_save_failed_info").format(message=message),
                QMessageBox.Icon.Critical,
            )

    def save_processed_input_stack(self) -> None:
        window = self.window
        if not window.raw_images:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_no_images_save_stack_text"))
            return

        # The format is asked for before the folder: a folder dialog has no
        # filter to carry it, and cancelling here costs the user nothing.
        extension = self.ask_export_format()
        if not extension:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_foldername = f"Processed_Input_Stack_{timestamp}"
        folder_path = QFileDialog.getExistingDirectory(
            window,
            "Select Folder to Save Processed Input Stack",
            window.settings_manager.default_output_path(default_foldername),
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        # Remember the parent so the next export starts beside the created stack folder.
        window.settings_manager.set_output_dir(os.path.dirname(folder_path))

        # One line for the whole stack when the chosen container cannot hold the
        # source depth, rather than one per frame from write_image.
        if bitdepth.is_high_depth(window.raw_images) and not bitdepth.supports_16bit(extension):
            print(f"[Depth] {extension} cannot store 16-bit; the stack is saved 8-bit. "
                  f"Use PNG, TIFF or JPEG XL to keep the full depth.", flush=True)

        # Each frame keeps the EXIF of the file it was loaded from, so the
        # camera, lens and exposure of the shot survive the processing. Said
        # once here for the whole stack when the chosen container has nowhere to
        # put them, rather than once per frame.
        source_paths = getattr(window, "image_paths", None) or []
        if source_paths and not carries_exif(extension):
            print(f"[Metadata] {extension} cannot carry EXIF; the stack is saved without it. "
                  f"Use JPG, PNG, JPEG XL or DNG to keep the camera tags.", flush=True)

        try:
            saved_count = 0
            for index, image in enumerate(window.raw_images):
                image_to_save = window.label_manager.prepare_bgr_image("input", image, index)
                if index < len(window.image_filenames):
                    filename = self.force_export_extension(window.image_filenames[index], extension)
                else:
                    filename = f"processed_{index + 1:04d}{extension}"
                file_path = os.path.join(folder_path, filename)
                source = source_paths[index] if index < len(source_paths) else None
                if write_image(file_path, image_to_save, source_path=source):
                    saved_count += 1

            show_success_box(
                window,
                trans.t("msg_success"),
                trans.t("msg_processed_stack_saved_text"),
                trans.t("msg_processed_stack_saved_info").format(
                    saved=saved_count,
                    total=len(window.raw_images),
                    folder=folder_path,
                ),
            )
        except cv2.error as exc:
            show_error_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_opencv_info").format(error=str(exc)),
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_unexpected_info").format(error=str(exc)),
            )
