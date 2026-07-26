import os
from dataclasses import replace
from typing import Any

import cv2
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMenu, QMessageBox

from utils import write_image, cv2_to_pixmap, fit_list_rows_to_thumbnails
from utils import show_error_box, show_message_box, show_success_box, show_warning_box
from controllers.export_manager import save_dialog_filter
from core import render_options
from locales import trans


class OutputManager:
    """Encapsulates output panel interactions and state updates."""

    def __init__(self, window: Any):
        self.window = window

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_output_count(self) -> None:
        count = self.window.output_list.count()
        self.window.output_label.setText(f"Output: {count}")

    def update_output_list_for_fusion(self) -> None:
        window = self.window

        fusion_filename = window.export_manager.generate_default_filename()

        if window.fusion_result is not None:
            window.fusion_results.insert(0, window.fusion_result.copy())
            # Kept in step with fusion_results so row N of the output list always
            # describes the render behind fusion_results[N].
            window.fusion_metadata.insert(0, getattr(window, "fusion_result_metadata", None))

        item = QListWidgetItem(fusion_filename)

        if window.fusion_result is not None:
            try:
                pixmap = cv2_to_pixmap(window.apply_output_contrast(window.fusion_result))
                # 64px tall keeps the thumbnail sharp once the icon box scales it
                # down to the row height, whatever the result's aspect ratio.
                icon = QIcon(pixmap.scaledToHeight(64, Qt.TransformationMode.SmoothTransformation))
                item.setIcon(icon)
            except Exception:
                pass

        window.output_list.insertItem(0, item)
        window.output_list.setCurrentRow(0)
        fit_list_rows_to_thumbnails(window.output_list)
        self.update_output_count()

    def metadata_for_row(self, row: int) -> Any:
        """Render metadata of an output row, or None if it has none.

        Results produced before the stack was reloaded, or by a code path that
        did not record anything, simply save without the extra metadata.
        """
        records = getattr(self.window, "fusion_metadata", None) or []
        if 0 <= row < len(records):
            return self._with_current_contrast(records[row])
        return None

    def metadata_for_current(self) -> Any:
        """Render metadata of the result `window.fusion_result` holds."""
        return self._with_current_contrast(getattr(self.window, "fusion_result_metadata", None))

    def _with_current_contrast(self, record: Any) -> Any:
        """Restate the Contrast option as it is at this moment.

        Contrast is a post-fusion output setting: it is applied when the result
        is displayed or saved, from whatever the slider says then, so the value
        frozen at render time would describe the wrong file. The stored record is
        left untouched - the refreshed copy exists only for this save.
        """
        if record is None or not record.options:
            return record

        window = self.window
        options = dict(record.options)
        options["Contrast"] = render_options.describe_contrast(
            getattr(window, "contrast_method", "off"),
            getattr(window, "contrast_strength", 0),
        )
        return replace(record, options=options)

    def sync_output_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.result_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & list operations
    # ------------------------------------------------------------------
    def show_output_context_menu(self, position: QPoint) -> None:
        window = self.window
        menu = QMenu(window)

        delete_action = QAction("Delete", window)
        delete_action.triggered.connect(self.delete_selected_output_images)
        menu.addAction(delete_action)

        save_as_action = QAction("Save as", window)
        save_as_action.triggered.connect(lambda: self.save_output_image_as(window.output_list.currentItem()))
        save_as_action.setEnabled(len(window.output_list.selectedItems()) == 1)
        menu.addAction(save_as_action)

        menu.exec(window.output_list.mapToGlobal(position))

    def delete_output_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return

        window.output_list.takeItem(row)

        if 0 <= row < len(window.fusion_results):
            window.fusion_results.pop(row)
        if 0 <= row < len(window.fusion_metadata):
            window.fusion_metadata.pop(row)

        self.update_output_count()

        if window.fusion_results:
            new_index = min(row, len(window.fusion_results) - 1)
            window.fusion_result = window.fusion_results[new_index]
            window.fusion_result_metadata = self.metadata_for_row(new_index)
            window.output_list.setCurrentRow(new_index)
            self.display_specific_fusion_result(window.fusion_result)
        else:
            window.fusion_result = None
            window.fusion_result_metadata = None
            window.lbl_result_img.clear()
            window.result_control_bar.setVisible(False)

    def save_output_image_as(self, item: QListWidgetItem | None) -> None:
        window = self.window
        if item is None:
            return

        row = window.output_list.row(item)
        if row < 0:
            return

        # Propose the name shown in the output list so the saved file matches the entry.
        default_filename = item.text().strip() or window.export_manager.generate_default_filename()
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            window,
            trans.t("action_save"),
            window.settings_manager.default_output_path(default_filename),
            save_dialog_filter(),
        )

        if not file_path:
            return

        fallback_ext = os.path.splitext(default_filename)[1]
        file_path = window.export_manager.normalize_export_path(
            file_path,
            fallback_extension=fallback_ext if fallback_ext else None,
        )
        window.settings_manager.set_output_dir(os.path.dirname(file_path))

        try:
            if row < len(window.fusion_results):
                image_to_save = window.label_manager.prepare_bgr_image(
                    "registered", window.apply_output_contrast(window.fusion_results[row]), 0
                )

                success = write_image(
                    file_path, image_to_save, announce=True, metadata=self.metadata_for_row(row)
                )
                if success:
                    show_success_box(
                        window,
                        trans.t("msg_success"),
                        trans.t("msg_image_saved_text"),
                        trans.t("msg_image_saved_info").format(path=file_path),
                    )
                else:
                    show_error_box(
                        window,
                        trans.t("msg_save_failed_title"),
                        trans.t("msg_save_failed_text"),
                        trans.t("msg_save_failed_info_write"),
                    )
            elif window.fusion_result is not None and row == 0:
                image_to_save = window.label_manager.prepare_bgr_image(
                    "registered", window.apply_output_contrast(window.fusion_result), 0
                )

                success = write_image(
                    file_path, image_to_save, announce=True,
                    metadata=self.metadata_for_current(),
                )
                if success:
                    show_success_box(
                        window,
                        trans.t("msg_success"),
                        trans.t("msg_image_saved_text"),
                        trans.t("msg_image_saved_info").format(path=file_path),
                    )
                else:
                    show_error_box(
                        window,
                        trans.t("msg_save_failed_title"),
                        trans.t("msg_save_failed_text"),
                        trans.t("msg_save_failed_info_write"),
                    )
            else:
                show_warning_box(
                    window,
                    trans.t("msg_warning"),
                    trans.t("msg_no_valid_image_text"),
                    trans.t("msg_no_valid_image_info"),
                )
        except cv2.error as exc:
            show_error_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_opencv").format(error=str(exc)),
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_unexpected").format(error=str(exc)),
            )

    def delete_selected_output_images(self) -> None:
        window = self.window
        selected_items = window.output_list.selectedItems()
        if not selected_items:
            return

        rows = [window.output_list.row(item) for item in selected_items]
        rows.sort(reverse=True)

        for row in rows:
            if row >= 0:
                window.output_list.takeItem(row)
                if 0 <= row < len(window.fusion_results):
                    window.fusion_results.pop(row)
                if 0 <= row < len(window.fusion_metadata):
                    window.fusion_metadata.pop(row)

        self.update_output_count()

        if window.fusion_results:
            min_row = min(rows) if rows else 0
            new_index = min(min_row, len(window.fusion_results) - 1)
            window.fusion_result = window.fusion_results[new_index]
            window.fusion_result_metadata = self.metadata_for_row(new_index)
            window.output_list.setCurrentRow(new_index)
            self.display_specific_fusion_result(window.fusion_result)
        else:
            window.fusion_result = None
            window.fusion_result_metadata = None
            window.lbl_result_img.clear()
            window.result_control_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def display_specific_fusion_result(self, fusion_image) -> None:
        if fusion_image is None:
            return

        window = self.window
        # Remember the pristine result so the contrast slider can re-show it
        # without re-rendering (see OpenFocus.refresh_result_display).
        window._displayed_fusion_result = fusion_image

        try:
            # Contrast is applied to the image content first, then labels are
            # drawn on top at full strength so the text is never dimmed by it.
            adjusted = window.apply_output_contrast(fusion_image)
            display_image = window.label_manager.prepare_bgr_image("registered", adjusted, 0)
            pixmap = cv2_to_pixmap(display_image)

            window.lbl_result_img.set_display_pixmap(pixmap)
            window.result_control_bar.setVisible(False)
            window.lbl_result_info.setText("-- / --")
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_display_error_title"),
                trans.t("msg_display_fusion_failed_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

    def display_output_image_in_result_view(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return

        if row < len(window.fusion_results):
            self.display_specific_fusion_result(window.fusion_results[row])
        elif window.fusion_result is not None:
            self.show_fusion_result()

    def show_fusion_result(self) -> None:
        window = self.window
        if window.fusion_result is None:
            return

        self.display_specific_fusion_result(window.fusion_result)

    def show_registration_result(self, index: int) -> None:
        window = self.window

        if window.fusion_result is not None:
            return
        if not window.registration_results:
            return
        if not 0 <= index < len(window.registration_results):
            return

        window.current_result_index = index

        try:
            image = window.registration_results[index]
            image = window.label_manager.apply_labels_to_registered_image(image, index)

            pixmap = cv2_to_pixmap(image)

            window.lbl_result_img.set_display_pixmap(pixmap)
            window.lbl_result_info.setText(f"{index + 1} / {len(window.registration_results)}")
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_display_error_title"),
                trans.t("msg_display_registration_failed_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

    def refresh_current_result_view(self) -> None:
        window = self.window
        if window.fusion_result is not None:
            self.show_fusion_result()
        elif window.current_result_index >= 0:
            self.show_registration_result(window.current_result_index)