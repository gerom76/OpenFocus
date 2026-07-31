from collections import deque
from dataclasses import dataclass
from typing import Any

import os
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QDialog,
)

from dialogs import DownsampleDialog
from utils import (
    exec_message_box,
    fit_list_rows_to_thumbnails,
    show_message_box,
    show_warning_box,
)
from ui.styles import MESSAGE_BOX_STYLE
from locales import trans


@dataclass
class LoadOptions:
    scale_factor: float
    filenames: list[str]
    paths: list[str]
    base_images: list[Any]
    working_images: list[Any]


class SourceManager:
    """Handles source list interactions and bookkeeping for the main window."""

    def __init__(self, window: Any):
        self.window = window
        # Live Stop-button connection while a load is running, None otherwise.
        self._stop_connection = None

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_source_images_count(self) -> None:
        count = self.window.file_list.count()
        checked = len(self.checked_source_indices())
        if checked == count:
            self.window.source_images_label.setText(trans.t("label_source_images").format(count))
        else:
            self.window.source_images_label.setText(
                trans.t("label_source_images_sel").format(count, checked)
            )

    # ------------------------------------------------------------------
    # Check-state handling (which frames take part in processing)
    # ------------------------------------------------------------------
    def checked_source_indices(self) -> list[int]:
        """Rows whose checkbox is ticked, i.e. the frames to be processed."""
        file_list = getattr(self.window, "file_list", None)
        if file_list is None:
            return []

        indices = []
        for row in range(file_list.count()):
            item = file_list.item(row)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                indices.append(row)
        return indices

    def _set_check_states(self, predicate) -> None:
        """Apply predicate(row) -> bool to every row, refreshing the count once."""
        file_list = getattr(self.window, "file_list", None)
        if file_list is None:
            return

        file_list.blockSignals(True)
        try:
            for row in range(file_list.count()):
                item = file_list.item(row)
                if item is None:
                    continue
                item.setCheckState(
                    Qt.CheckState.Checked if predicate(row) else Qt.CheckState.Unchecked
                )
        finally:
            file_list.blockSignals(False)

        self.update_source_images_count()

    def check_all_sources(self) -> None:
        self._set_check_states(lambda row: True)

    def uncheck_all_sources(self) -> None:
        self._set_check_states(lambda row: False)

    def invert_source_checks(self) -> None:
        checked = set(self.checked_source_indices())
        self._set_check_states(lambda row: row not in checked)

    def mark_selected_sources(self) -> None:
        """Tick the highlighted rows, leaving the rest of the list untouched."""
        self._set_selected_check_state(True)

    def unmark_selected_sources(self) -> None:
        self._set_selected_check_state(False)

    def _set_selected_check_state(self, checked: bool) -> None:
        file_list = getattr(self.window, "file_list", None)
        if file_list is None:
            return

        selected_items = file_list.selectedItems()
        if not selected_items:
            return

        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        file_list.blockSignals(True)
        try:
            for item in selected_items:
                item.setCheckState(state)
        finally:
            file_list.blockSignals(False)

        self.update_source_images_count()

    def check_every_nth_source(self) -> None:
        """Clear the selection, then tick the 1st, (n+1)-th, (2n+1)-th ... rows."""
        spin = getattr(self.window, "spin_select_nth", None)
        step = spin.value() if spin is not None else 2
        step = max(1, int(step))
        self._set_check_states(lambda row: row % step == 0)

    def handle_source_item_changed(self, item: QListWidgetItem) -> None:
        self.update_source_images_count()

    # ------------------------------------------------------------------
    # Load progress
    # ------------------------------------------------------------------
    def _begin_load_progress(self, loader: Any = None):
        """Show the console progress bar and return a callback for the loader.

        Loading runs on the GUI thread, so the callback pumps the event loop to
        keep the bar and the console output moving - which is also what lets a
        Stop-button click through: the button is enabled here and routed to the
        loader's cooperative stop for the duration of the load. Returns None
        when there is no console to drive.
        """
        loader = loader if loader is not None else getattr(self.window, "image_loader", None)
        button = getattr(self.window, "btn_stop", None)
        if button is not None and loader is not None and hasattr(loader, "request_stop"):
            self._stop_connection = button.clicked.connect(loader.request_stop)
            button.setEnabled(True)

        console = getattr(self.window, "status_console", None)
        if console is None:
            return None

        console.start_progress()
        QApplication.processEvents()

        def report(current: int, total: int) -> None:
            console.set_progress(current, total)
            QApplication.processEvents()

        return report

    def _end_load_progress(self) -> None:
        # Idempotent: some load paths reach this twice (explicitly and via
        # finally), so the Stop wiring is only undone while it exists.
        button = getattr(self.window, "btn_stop", None)
        if button is not None and self._stop_connection is not None:
            try:
                button.clicked.disconnect(self._stop_connection)
            except TypeError:
                pass
            self._stop_connection = None
            button.setEnabled(False)

        console = getattr(self.window, "status_console", None)
        if console is not None:
            console.stop_progress()

    def load_image_stack(self, folder_path: str, append: bool = False) -> None:
        window = self.window

        current_scale = getattr(window, "current_scale_factor", 1.0)
        dialog = DownsampleDialog(window, initial_scale=current_scale)
        if not dialog.exec():
            return

        scale_factor = dialog.get_scale_factor()
        long_edge = dialog.get_target_long_edge()

        window.current_folder_path = folder_path

        try:
            success, message, full_res_images, filenames = window.image_loader.load_from_folder(
                folder_path, scale_factor=scale_factor, progress_callback=self._begin_load_progress(),
                target_long_edge=long_edge
            )

            if not success:
                if getattr(window.image_loader, "cancelled", False):
                    return  # stopped on purpose, not an error
                show_warning_box(window, trans.t("msg_load_failed"), trans.t("msg_load_stack_failed_text"), message)
                return

            if append and not self._confirm_append_dimensions(full_res_images):
                return
            load_options = self._build_load_options(
                full_res_images, filenames, scale_factor, window.image_loader.image_paths
            )
            self._apply_load_options(load_options, append=append)
            window.settings_manager.add_recent_folder(folder_path)
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_load_error"),
                trans.t("msg_load_stack_error_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
        finally:
            self._end_load_progress()

    def load_video_stack(self, video_path: str, append: bool = False) -> None:
        """Load image stack from a video file."""
        window = self.window

        current_scale = getattr(window, "current_scale_factor", 1.0)
        dialog = DownsampleDialog(window, initial_scale=current_scale)
        if not dialog.exec():
            return

        scale_factor = dialog.get_scale_factor()
        long_edge = dialog.get_target_long_edge()

        window.current_folder_path = os.path.dirname(video_path)

        try:
            success, message, full_res_images, filenames = window.image_loader.load_from_video(
                video_path, scale_factor=scale_factor, progress_callback=self._begin_load_progress(),
                target_long_edge=long_edge
            )

            if not success:
                if getattr(window.image_loader, "cancelled", False):
                    return  # stopped on purpose, not an error
                show_warning_box(window, trans.t("msg_load_failed"), trans.t("msg_load_video_failed_text"), message)
                return

            if append and not self._confirm_append_dimensions(full_res_images):
                return
            load_options = self._build_load_options(
                full_res_images, filenames, scale_factor, window.image_loader.image_paths
            )
            self._apply_load_options(load_options, append=append)
            window.settings_manager.add_recent_video(video_path)
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_load_error"),
                trans.t("msg_load_video_error_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
        finally:
            self._end_load_progress()

    def prompt_and_load_stack(self) -> None:
        window = self.window

        folder_path = QFileDialog.getExistingDirectory(
            window,
            "Select Image Stack Folder",
            window.settings_manager.last_folder_dir(),
            QFileDialog.Option.ShowDirsOnly,
        )

        if folder_path:
            self.load_image_stack(folder_path)

    def prompt_and_load_video(self) -> None:
        """Open a file dialog to select a video file and load it as image stack."""
        window = self.window
        from core.image_loader import ImageStackLoader

        # Build video filter string
        video_exts = " ".join([f"*{ext}" for ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS])
        
        video_path, _ = QFileDialog.getOpenFileName(
            window,
            "Select Video File",
            window.settings_manager.last_video_dir(),
            f"Video Files ({video_exts});;All Files (*)",
        )

        if video_path:
            self.load_video_stack(video_path)

    def can_accept_drag(self, event: QDragEnterEvent) -> bool:
        if not event.mimeData().hasUrls():
            return False

        urls = event.mimeData().urls()
        if not urls:
            return False

        # Accept if single directory, or one-or-more files
        # We will further validate on drop
        return True

    def handle_drop_event(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [u.toLocalFile() for u in urls]

        # If a single directory was dropped, ask user how to import
        if len(paths) == 1 and os.path.isdir(paths[0]):
            from dialogs import FolderImportDialog
            dialog = FolderImportDialog(paths[0], self.window)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.is_single_stack():
                    self.load_image_stack(paths[0], append=True)
                    event.acceptProposedAction()
                else:
                    # Multiple image stacks - first pop up the downsampling settings
                    current_scale = getattr(self.window, "current_scale_factor", 1.0)
                    dlg = DownsampleDialog(self.window, initial_scale=current_scale)
                    if not dlg.exec():
                        event.ignore()
                        return
                    scale = dlg.get_scale_factor()

                    # Open the batch dialog and preload the folder
                    self.window.show_batch_processing_dialog(
                        preload_folder_paths=[paths[0]], scale_factor=scale,
                        target_long_edge=dlg.get_target_long_edge())
                    event.acceptProposedAction()
            else:
                event.ignore()
            return

        # If multiple directories were dropped, ask user how to import
        elif all(os.path.isdir(p) for p in paths):
            current_scale = getattr(self.window, "current_scale_factor", 1.0)
            dlg = DownsampleDialog(self.window, initial_scale=current_scale)
            if not dlg.exec():
                event.ignore()
                return
            scale = dlg.get_scale_factor()

            self.window.show_batch_processing_dialog(
                preload_folder_paths=paths, scale_factor=scale,
                target_long_edge=dlg.get_target_long_edge())
            event.acceptProposedAction()
            return

        # Check if a single video file was dropped
        from core.image_loader import ImageStackLoader
        if len(paths) == 1 and os.path.isfile(paths[0]):
            ext = os.path.splitext(paths[0])[1].lower()
            if ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS:
                self.load_video_stack(paths[0], append=True)
                event.acceptProposedAction()
                return

        # Otherwise treat dropped items as a list of files
        filepaths = [p for p in paths if os.path.isfile(p)]
        if not filepaths:
            show_warning_box(self.window, trans.t("msg_warning"), trans.t("msg_drop_no_valid_files_text"))
            event.ignore()
            return

        # Filter by supported image extensions (use ImageStackLoader.SUPPORTED_FORMATS)
        from core.image_loader import ImageStackLoader

        loader = self.window.image_loader if hasattr(self.window, "image_loader") else ImageStackLoader()

        valid_paths = []
        for p in filepaths:
            ext = os.path.splitext(p)[1].lower()
            if ext in ImageStackLoader.SUPPORTED_FORMATS:
                valid_paths.append(p)

        if not valid_paths:
            show_warning_box(self.window, trans.t("msg_warning"), trans.t("msg_drop_no_supported_images_text"))
            event.ignore()
            return

        # Prompt for downsample (same UX as folder loading)
        try:
            current_scale = getattr(self.window, "current_scale_factor", 1.0)
            dlg = DownsampleDialog(self.window, initial_scale=current_scale)
            if not dlg.exec():
                # user cancelled
                event.ignore()
                return
            scale = dlg.get_scale_factor()

            success, message, full_res_images, filenames = loader.load_from_filepaths(
                valid_paths, scale_factor=scale, progress_callback=self._begin_load_progress(loader),
                target_long_edge=dlg.get_target_long_edge()
            )
            self._end_load_progress()
            if not success:
                if not getattr(loader, "cancelled", False):
                    show_warning_box(self.window, trans.t("msg_load_failed"), trans.t("msg_load_dropped_failed_text"), message)
                event.ignore()
                return
            # Check image sizes (after loading / downsampling)
            shapes = {(img.shape[0], img.shape[1]) for img in full_res_images}
            if len(shapes) > 1:
                # Ask user to continue or cancel (Continue/Cancel)
                msg = QMessageBox(self.window)
                msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
                msg.setText(trans.t("msg_size_mismatch_stack_text"))
                msg.setInformativeText(trans.t("msg_size_mismatch_open_info"))
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                # relabel buttons to Continue / Cancel
                yes_btn = msg.button(QMessageBox.StandardButton.Yes)
                no_btn = msg.button(QMessageBox.StandardButton.No)
                if yes_btn:
                    yes_btn.setText(trans.t("btn_continue"))
                if no_btn:
                    no_btn.setText(trans.t("btn_cancel_generic"))
                msg.setStyleSheet(MESSAGE_BOX_STYLE)
                ret = exec_message_box(msg)
                if ret != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            if not self._confirm_append_dimensions(full_res_images):
                event.ignore()
                return
            load_options = self._build_load_options(
                full_res_images, filenames, scale, loader.image_paths
            )
            self._apply_load_options(load_options, append=True)
            event.acceptProposedAction()
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                self.window,
                trans.t("msg_load_error"),
                trans.t("msg_load_dropped_error_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
            event.ignore()
        finally:
            self._end_load_progress()

    def _build_load_options(
        self,
        full_res_images: list[Any],
        filenames: list[str],
        scale_factor: float,
        loader_paths: list[str] | None = None,
    ) -> LoadOptions:
        return LoadOptions(
            scale_factor=1.0,
            filenames=list(filenames),
            paths=self._align_paths(loader_paths, filenames),
            base_images=full_res_images,
            working_images=full_res_images,
        )

    @staticmethod
    def _align_paths(loader_paths: list[str] | None, filenames: list[str]) -> list[str]:
        """Source paths for the frames just loaded, one per filename.

        The loader fills both lists in the same pass, so they line up by index.
        A mismatched count cannot be aligned by guessing, so the paths are
        dropped instead - the frames still load, they just carry no source to
        read EXIF from later.
        """
        paths = list(loader_paths or [])
        if len(paths) != len(filenames):
            return [""] * len(filenames)
        return paths

    def _apply_load_options(self, options: LoadOptions, append: bool = False) -> None:
        window = self.window

        if append and window.raw_images:
            window.base_images = (window.base_images or []) + options.base_images
            window.raw_images = (window.raw_images or []) + options.working_images
            window.image_filenames = (window.image_filenames or []) + options.filenames
            window.image_paths = (getattr(window, "image_paths", None) or []) + options.paths
            initial_index = window.current_display_index if window.current_display_index >= 0 else 0
        else:
            window.base_images = options.base_images
            window.current_scale_factor = options.scale_factor
            window.raw_images = options.working_images
            window.image_filenames = options.filenames
            window.image_paths = options.paths
            initial_index = 0

        window.label_manager.reset_labels()
        window.transform_manager.invalidate_processing_results(
            clear_output_view=True,
            preserve_outputs=True,
        )
        window.transform_manager.reload_image_stack(initial_index=initial_index)

    def _confirm_append_dimensions(self, new_images: list[Any]) -> bool:
        window = self.window
        if not window.raw_images:
            return True

        try:
            existing_shape = window.raw_images[0].shape[:2]
            new_shapes = {(img.shape[0], img.shape[1]) for img in new_images}
            if len(new_shapes) == 1 and existing_shape in new_shapes:
                return True
        except Exception:
            return True

        msg = QMessageBox(window)
        msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
        msg.setText(trans.t("msg_size_mismatch_text"))
        msg.setInformativeText(trans.t("msg_size_mismatch_info"))
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        yes_btn = msg.button(QMessageBox.StandardButton.Yes)
        no_btn = msg.button(QMessageBox.StandardButton.No)
        if yes_btn:
            yes_btn.setText(trans.t("btn_continue"))
        if no_btn:
            no_btn.setText(trans.t("btn_cancel_generic"))
        msg.setStyleSheet(MESSAGE_BOX_STYLE)
        ret = exec_message_box(msg)
        return ret == QMessageBox.StandardButton.Yes

    def refresh_current_source_view(self) -> None:
        index = getattr(self.window, "current_display_index", -1)
        if index >= 0:
            self.window.update_source_view(index)

    def clear_image_stack(self) -> None:
        window = self.window

        window.transform_manager.invalidate_processing_results(clear_output_view=True)

        window.raw_images = []
        window.base_images = []
        window.stack_images = []
        window.image_filenames = []
        window.image_paths = []
        window.current_display_index = -1
        
        # Clear the ROI alignment cache
        window.roi_aligned_images = []
        window.roi_aligned_raw_count = 0
        window.roi_mode_active = False

        window.label_manager.reset_labels()

        window.transform_manager.reload_image_stack(initial_index=None)

    def update_slider_range(self) -> None:
        window = self.window
        if window.stack_images:
            window.stack_slider.setEnabled(True)
            window.stack_slider.setRange(0, len(window.stack_images) - 1)
        else:
            window.stack_slider.setEnabled(False)
            window.stack_slider.setRange(0, 0)

    def update_file_list(self, filenames, thumbnails) -> None:
        window = self.window
        # The list is rebuilt from scratch on every stack mutation (delete,
        # append, resize), so carry the check states over per filename: frames
        # that survive keep their state, anything new starts checked.
        previous_states = self._collect_check_states()

        # Signals stay blocked while filling so the per-item check state does not
        # fire a label refresh for every row.
        window.file_list.blockSignals(True)
        window.file_list.clear()

        try:
            for filename, thumbnail in zip(filenames, thumbnails):
                item = QListWidgetItem(QIcon(thumbnail), filename)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # Newly loaded images all take part in processing by default.
                states = previous_states.get(filename)
                checked = states.popleft() if states else True
                item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                window.file_list.addItem(item)
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_update_error_title"),
                trans.t("msg_update_file_list_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
        finally:
            fit_list_rows_to_thumbnails(window.file_list)
            window.file_list.blockSignals(False)

        self.update_source_images_count()

    def _collect_check_states(self) -> dict[str, deque]:
        """Current check states grouped by filename, in row order."""
        file_list = getattr(self.window, "file_list", None)
        states: dict[str, deque] = {}
        if file_list is None:
            return states

        for row in range(file_list.count()):
            item = file_list.item(row)
            if item is None:
                continue
            states.setdefault(item.text(), deque()).append(
                item.checkState() == Qt.CheckState.Checked
            )
        return states

    def sync_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.stack_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & deletions
    # ------------------------------------------------------------------
    def show_source_context_menu(self, position: QPoint) -> None:
        window = self.window
        menu = QMenu(window)

        delete_action = QAction("Delete", window)
        delete_action.triggered.connect(self.delete_selected_source_images)
        menu.addAction(delete_action)

        has_selection = bool(window.file_list.selectedItems())

        mark_action = QAction(trans.t("menu_mark"), window)
        mark_action.triggered.connect(self.mark_selected_sources)
        mark_action.setEnabled(has_selection)
        menu.addAction(mark_action)

        unmark_action = QAction(trans.t("menu_unmark"), window)
        unmark_action.triggered.connect(self.unmark_selected_sources)
        unmark_action.setEnabled(has_selection)
        menu.addAction(unmark_action)

        menu.exec(window.file_list.mapToGlobal(position))

    def delete_source_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.file_list.row(item)
        if row < 0:
            return

        if len(window.raw_images) <= 1:
            self.clear_image_stack()
            return

        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        self._pop_sequence(window.image_filenames, row)
        self._pop_sequence(getattr(window, "image_paths", None), row)
        self._pop_sequence(window.raw_images, row)
        self._pop_sequence(getattr(window, "base_images", None), row)

        if window.raw_images:
            new_index = min(row, len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=new_index)
        else:
            self.clear_image_stack()

    def delete_selected_source_images(self) -> None:
        window = self.window
        selected_items = window.file_list.selectedItems()
        if not selected_items:
            return

        rows = [window.file_list.row(item) for item in selected_items]
        rows = [row for row in rows if row >= 0]
        if not rows:
            return

        remaining = len(window.raw_images) - len(rows)
        if remaining <= 0:
            self.clear_image_stack()
            return

        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        rows.sort(reverse=True)

        for row in rows:
            self._pop_sequence(window.image_filenames, row)
            self._pop_sequence(getattr(window, "image_paths", None), row)
            self._pop_sequence(window.raw_images, row)
            self._pop_sequence(getattr(window, "base_images", None), row)

        if window.raw_images:
            target_index = min(min(rows), len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=target_index)
        else:
            self.clear_image_stack()

    def _pop_sequence(self, sequence: list[Any] | None, index: int) -> None:
        if sequence is not None and 0 <= index < len(sequence):
            sequence.pop(index)

    # === Icon Drag-and-Drop Support Methods ===
    # These methods handle files dropped on app icon/taskbar/dock

    def load_image_stack_from_icon(self, folder_path: str) -> None:
        """Load image stack from folder dropped on app icon."""
        from dialogs import FolderImportDialog
        dialog = FolderImportDialog(folder_path, self.window)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.is_single_stack():
                self.load_image_stack(folder_path, append=True)
            else:
                current_scale = getattr(self.window, "current_scale_factor", 1.0)
                dlg = DownsampleDialog(self.window, initial_scale=current_scale)
                if dlg.exec():
                    self.window.show_batch_processing_dialog(
                        preload_folder_paths=[folder_path],
                        scale_factor=dlg.get_scale_factor(),
                        target_long_edge=dlg.get_target_long_edge()
                    )

    def load_multiple_folders_from_icon(self, folder_paths: list[str]) -> None:
        """Load multiple folders dropped on app icon."""
        current_scale = getattr(self.window, "current_scale_factor", 1.0)
        dlg = DownsampleDialog(self.window, initial_scale=current_scale)
        if dlg.exec():
            self.window.show_batch_processing_dialog(
                preload_folder_paths=folder_paths,
                scale_factor=dlg.get_scale_factor(),
                target_long_edge=dlg.get_target_long_edge()
            )

    def load_video_stack_from_icon(self, video_path: str) -> None:
        """Load video file dropped on app icon."""
        self.load_video_stack(video_path)

    def load_image_files_from_icon(self, file_paths: list[str]) -> None:
        """Load image files dropped on app icon."""
        from core.image_loader import ImageStackLoader

        current_scale = getattr(self.window, "current_scale_factor", 1.0)
        dlg = DownsampleDialog(self.window, initial_scale=current_scale)
        if not dlg.exec():
            return
        scale = dlg.get_scale_factor()

        loader = self.window.image_loader if hasattr(self.window, "image_loader") else ImageStackLoader()
        try:
            success, message, full_res_images, filenames = loader.load_from_filepaths(
                file_paths, scale_factor=scale, progress_callback=self._begin_load_progress(loader),
                target_long_edge=dlg.get_target_long_edge()
            )
        finally:
            self._end_load_progress()

        if not success:
            if getattr(loader, "cancelled", False):
                return  # stopped on purpose, not an error
            show_warning_box(self.window, trans.t("msg_load_failed"), trans.t("msg_load_dropped_failed_text"), message)
            return

        if not self._confirm_append_dimensions(full_res_images):
            return

        # Check image sizes
        shapes = {(img.shape[0], img.shape[1]) for img in full_res_images}
        if len(shapes) > 1:
            msg = QMessageBox(self.window)
            msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
            msg.setText(trans.t("msg_size_mismatch_stack_text"))
            msg.setInformativeText(trans.t("msg_size_mismatch_open_info"))
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            yes_btn = msg.button(QMessageBox.StandardButton.Yes)
            no_btn = msg.button(QMessageBox.StandardButton.No)
            if yes_btn:
                yes_btn.setText(trans.t("btn_continue"))
            if no_btn:
                no_btn.setText(trans.t("btn_cancel_generic"))
            msg.setStyleSheet(MESSAGE_BOX_STYLE)
            if exec_message_box(msg) != QMessageBox.StandardButton.Yes:
                return

        load_options = self._build_load_options(
            full_res_images, filenames, scale, loader.image_paths
        )
        self._apply_load_options(load_options, append=True)