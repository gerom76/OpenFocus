import traceback
from datetime import datetime
from typing import Any, List, Optional

from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog

from utils import RenderMetadata, show_custom_message_box, show_message_box, show_warning_box
from constants import (
    DCT_PLATEAU_DEFAULT, DCT_PLATEAU_PRESETS,
    PYRAMID_BASE_DEFAULT, PYRAMID_BASE_PRESETS,
    PYRAMID_COHERENCE_DEFAULT, PYRAMID_COHERENCE_PRESETS,
    PYRAMID_SELECTIVITY_DEFAULT, PYRAMID_SELECTIVITY_PRESETS,
)
from core import render_options
from core.multi_focus_fusion import is_stackmffv4_available
from core.workers import RenderWorker
from dialogs import ROIRenderOptionsDialog  # Import the new dialog
from locales import trans

_PLATEAU_BY_KEY = dict(DCT_PLATEAU_PRESETS)


def dct_plateau_value(window) -> float:
    """The plateau width behind the selected focus-tolerance preset.

    The combo carries the preset key rather than the number, so a retuned preset
    changes here and not in every saved settings file.
    """
    key = window.combo_dct_plateau.currentData() or DCT_PLATEAU_DEFAULT
    return _PLATEAU_BY_KEY.get(key, _PLATEAU_BY_KEY[DCT_PLATEAU_DEFAULT])


def _preset_value(combo, presets, default_key):
    """The number behind a preset combo, or the default preset's number.

    Same contract as dct_plateau_value: what is stored and restored is the key,
    so retuning a preset moves every saved setting with it.
    """
    values = dict(presets)
    key = None if combo is None else combo.currentData()
    return values.get(key, values[default_key])


def pyramid_params(window) -> dict:
    """The pyramid tuning the window is showing, as method arguments.

    Kept next to dct_plateau_value because the batch dialog needs exactly the
    same reading of the same controls, and two readings would drift apart.
    """
    return {
        "levels": (window.combo_pyr_levels.currentData() or None),
        "selectivity": _preset_value(window.combo_pyr_selectivity,
                                     PYRAMID_SELECTIVITY_PRESETS,
                                     PYRAMID_SELECTIVITY_DEFAULT),
        "coherence": _preset_value(window.combo_pyr_coherence,
                                   PYRAMID_COHERENCE_PRESETS,
                                   PYRAMID_COHERENCE_DEFAULT),
        "base_selectivity": _preset_value(window.combo_pyr_base,
                                          PYRAMID_BASE_PRESETS,
                                          PYRAMID_BASE_DEFAULT),
        "noise_gate": window.cb_pyr_noise_gate.isChecked(),
        "envelope": window.cb_pyr_envelope.isChecked(),
    }


class RenderManager:
    """Encapsulates render pipeline orchestration for the main window."""

    def __init__(self, window: Any):
        self.window = window
        self.worker: Optional[RenderWorker] = None
        # True while the running render covers only the ticked subset of the
        # source list - its result must not be cached as the aligned full stack.
        self._subset_render = False
        # Source file the running render's result inherits its EXIF from, taken
        # when the render starts because the source list may change while it runs.
        self._render_source_path: Optional[str] = None

    def _set_stop_enabled(self, enabled: bool) -> None:
        """Enable the Stop button only while a render is interruptible."""
        btn = getattr(self.window, "btn_stop", None)
        if btn is not None:
            btn.setEnabled(enabled)

    def _set_console_progress(self, running: bool) -> None:
        """Show/hide the progress bar in the bottom status console."""
        console = getattr(self.window, "status_console", None)
        if console is None:
            return
        if running:
            console.start_progress()
        else:
            console.stop_progress()

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Lock the pipeline controls while a render is running, unlock afterwards."""
        window = self.window
        for attr in (
            "slider_smooth",
            "rb_a", "rb_b", "rb_c", "rb_gfg", "rb_pyramid",
            "rb_dmap_max", "rb_dmap_avg", "rb_d",
            "cb_ifcnn",
            "cb_align_scale", "cb_align_homography", "cb_align_ecc",
            "btn_reset",
        ):
            widget = getattr(window, attr, None)
            if widget is not None:
                widget.setEnabled(enabled)

        if enabled:
            # Methods and stages with optional dependencies stay disabled
            try:
                window._configure_fusion_method_availability()
            except Exception:
                pass

    def _abort_render(self) -> None:
        """Undo the pre-render UI state after bailing out before the worker starts."""
        self._set_controls_enabled(True)
        self.window.btn_render.setEnabled(True)
        self.window.btn_render.setText(trans.t('btn_render'))
        self._set_stop_enabled(False)
        self._set_console_progress(False)

    def stop_render(self) -> None:
        """Request cancellation of the running render.

        The worker unwinds cooperatively at its next checkpoint (stage boundary
        or tile/batch loop) and then emits ``cancelled_signal``, which restores
        the UI. We disable Stop and flag the pending stop here so the button
        cannot be clicked twice.
        """
        worker = self.worker
        if worker is None or not worker.isRunning():
            return

        worker.cancel()
        self._set_stop_enabled(False)
        self.window.btn_render.setText(trans.t('btn_render_stopping'))

    def _selected_source_rows(self) -> List[int]:
        """Rows ticked in the source list, or the whole stack when unavailable."""
        window = self.window
        total = len(window.raw_images or [])
        manager = getattr(window, "source_manager", None)
        file_list = getattr(window, "file_list", None)

        # The list widget mirrors raw_images; if they ever drift apart the
        # row indices cannot be trusted, so fall back to the full stack.
        if manager is None or file_list is None or file_list.count() != total:
            return list(range(total))

        return manager.checked_source_indices()

    def _first_source_path(self, selected_rows: List[int]) -> Optional[str]:
        """Path of the first source frame taking part in the render.

        That frame is where the result's EXIF comes from, so with only part of
        the stack ticked it is the first ticked frame rather than the first row
        of the list.
        """
        paths = getattr(self.window, "image_paths", None) or []
        for row in selected_rows:
            if 0 <= row < len(paths) and paths[row]:
                return paths[row]
        return None

    def _describe_render(self, fusion_result: Any, device_name: str) -> dict:
        """The settings the finished render ran with, for its saved metadata.

        Read off the worker rather than the widgets: a control the user moved
        while the render was running would otherwise be recorded as if it had
        taken part in it.
        """
        window = self.window
        worker = self.worker
        if worker is None:
            return {}

        return render_options.describe(
            algorithm=worker.fusion_algorithm(),
            kernel_size=worker.kernel_slider_value,
            halo_radius=worker.halo_radius_value,
            depth_smoothing=worker.depth_smoothing_value,
            average_selectivity=worker.average_selectivity_value,
            coherence_radius=worker.coherence_radius_value,
            slice_radius=worker.slice_radius_value,
            ifcnn_refine=worker.ifcnn_refine,
            align_scale=worker.need_align_scale,
            align_homography=worker.need_align_homography,
            align_ecc=worker.need_align_ecc,
            reference_mode=worker.reference_mode,
            reg_downscale_width=worker.reg_downscale_width,
            ecc_parallel=worker.ecc_parallel,
            contrast_method=getattr(window, "contrast_method", "off"),
            contrast_strength=getattr(window, "contrast_strength", 0),
            source_count=len(worker.raw_images or []),
            total_count=len(window.raw_images or []),
            roi_mode=worker.roi_mode if worker.roi_rect is not None else None,
            roi_base_index=worker.roi_base_index,
            tile_enabled=worker.tile_enabled,
            tile_block_size=worker.tile_block_size,
            tile_overlap=worker.tile_overlap,
            tile_threshold=worker.tile_threshold,
            stackmffv4_batch_size=worker.stackmffv4_batch_size,
            pyramid_levels=worker.pyramid_levels,
            pyramid_selectivity=worker.pyramid_selectivity,
            pyramid_coherence=worker.pyramid_coherence,
            pyramid_base=worker.pyramid_base,
            pyramid_noise_gate=worker.pyramid_noise_gate,
            pyramid_envelope=worker.pyramid_envelope,
            result_dtype=getattr(fusion_result, "dtype", None),
            device_name=device_name,
            thread_count=worker.thread_count,
            resolved_auto=getattr(worker, "resolved_auto", None),
        )

    def start_render(self) -> None:
        window = self.window

        if not window.raw_images or len(window.raw_images) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
            return

        selected_rows = self._selected_source_rows()
        if len(selected_rows) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_selected_text"))
            return

        self._render_source_path = self._first_source_path(selected_rows)

        window.btn_render.setEnabled(False)
        window.btn_render.setText(trans.t('btn_render_processing'))
        self._set_console_progress(True)
        QApplication.processEvents()

        # Disable UI controls that should not be modified during processing
        self._set_controls_enabled(False)

        need_align_scale = window.cb_align_scale.isChecked()
        need_align_homography = window.cb_align_homography.isChecked()
        need_align_ecc = window.cb_align_ecc.isChecked()

        need_fusion = (
            window.rb_a.isChecked()
            or window.rb_b.isChecked()
            or window.rb_c.isChecked()
            or window.rb_gfg.isChecked()
            or window.rb_pyramid.isChecked()
            or window.rb_dmap_max.isChecked()
            or window.rb_dmap_avg.isChecked()
            or window.rb_d.isChecked()
        )

        kernel_slider_value = window.slider_smooth.value()
        if kernel_slider_value <= 0:
            kernel_slider_value = 1
        if kernel_slider_value % 2 == 0:
            kernel_slider_value = max(1, kernel_slider_value - 1)

        if window.rb_d.isChecked() and not is_stackmffv4_available():
            show_warning_box(
                window,
                trans.t("msg_stackmff_unavailable_title"),
                trans.t("msg_stackmff_unavailable_text"),
            )
            window.rb_d.setChecked(False)
            self._abort_render()
            return

        # The IFCNN stage refines a fused image, so it cannot run on its own
        ifcnn_refine = bool(window.cb_ifcnn.isChecked())
        if ifcnn_refine and not need_fusion:
            show_warning_box(
                window,
                trans.t("msg_ifcnn_needs_fusion_title"),
                trans.t("msg_ifcnn_needs_fusion_text"),
            )
            self._abort_render()
            return

        # Handle ROI options - check if ROI mode is active and we have aligned images
        roi_rect = None
        roi_mode = "crop"
        roi_base_index = 0
        use_roi_aligned_images = False
        
        # The ROI stack is index-aligned with the raw stack, so the ticked rows
        # select from it the same way they do from raw_images.
        roi_source_images = window.roi_aligned_images
        if len(roi_source_images or []) == len(window.raw_images) and len(selected_rows) < len(window.raw_images):
            roi_source_images = [roi_source_images[i] for i in selected_rows]

        # Check whether ROI mode is active and get the ROI region from the right panel
        if getattr(window, 'roi_mode_active', False) and window.roi_aligned_images:
            # Get the ROI region from the right-side result panel (because the ROI is selected on the aligned image)
            roi_rect = window.lbl_result_img.get_roi_rect() if hasattr(window.lbl_result_img, 'get_roi_rect') else None
            if roi_rect is not None:
                use_roi_aligned_images = True
                dialog = ROIRenderOptionsDialog(len(roi_source_images), window)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    roi_mode = dialog.mode
                    roi_base_index = dialog.base_frame_index
                else:
                    # User cancelled the ROI dialog -> cancel render
                    self._abort_render()
                    return
        
        # Determine the image source to use
        self._subset_render = False
        if use_roi_aligned_images:
            # In ROI mode, use the already-aligned image stack and skip the extra registration
            source_images = roi_source_images
            # In ROI mode, the images are already aligned and do not need to be registered again
            effective_need_align_scale = False
            effective_need_align_homography = False
            effective_need_align_ecc = False
            # Tell the Worker that the images are already aligned
            effective_aligned_images = roi_source_images
            effective_is_aligned = True
            # ROI pre-alignment always references frame 0 (see ROIAlignmentWorker).
            effective_reference_mode = "first"
            effective_last_alignment_options = (False, False, True, effective_reference_mode)  # indicates ECC is done
        else:
            # Only the frames ticked in the source list take part in the render
            self._subset_render = len(selected_rows) < len(window.raw_images)
            if self._subset_render:
                source_images = [window.raw_images[i] for i in selected_rows]
                print(
                    f"Rendering {len(source_images)} of {len(window.raw_images)} source images",
                    flush=True,
                )
            else:
                source_images = window.raw_images

            effective_need_align_scale = need_align_scale
            effective_need_align_homography = need_align_homography
            effective_need_align_ecc = need_align_ecc
            effective_reference_mode = getattr(window, "reference_frame_mode", "first")
            effective_last_alignment_options = window.last_alignment_options

            cached_aligned = window.aligned_images or []
            if not self._subset_render:
                effective_aligned_images = window.aligned_images
                effective_is_aligned = window.is_images_aligned
            elif len(cached_aligned) == len(window.raw_images):
                # The cache is index-aligned with the raw stack, so the same
                # subset of it stays valid and the alignment can be reused.
                effective_aligned_images = [cached_aligned[i] for i in selected_rows]
                effective_is_aligned = window.is_images_aligned
            else:
                effective_aligned_images = []
                effective_is_aligned = False

        pyramid = pyramid_params(window)
        self.worker = RenderWorker(
            source_images,
            effective_aligned_images,
            effective_is_aligned,
            effective_last_alignment_options,
            effective_need_align_homography,
            effective_need_align_ecc,
            need_fusion,
            window.rb_a.isChecked(),
            window.rb_b.isChecked(),
            window.rb_c.isChecked(),
            window.rb_gfg.isChecked(),
            window.rb_d.isChecked(),
            kernel_slider_value,
            halo_radius_value=window.slider_halo.value(),
            depth_smoothing_value=window.slider_depth_smooth.value(),
            average_selectivity_value=window.slider_avg_selectivity.value(),
            coherence_radius_value=window.slider_avg_coherence.value(),
            slice_radius_value=window.slider_avg_slice.value(),
            dct_block_size=window.combo_dct_block.currentData(),
            dct_plateau=dct_plateau_value(window),
            dct_blend=window.cb_dct_blend.isChecked(),
            pyramid_levels=pyramid["levels"],
            pyramid_selectivity=pyramid["selectivity"],
            pyramid_coherence=pyramid["coherence"],
            pyramid_noise_gate=pyramid["noise_gate"],
            pyramid_base=pyramid["base_selectivity"],
            pyramid_envelope=pyramid["envelope"],
            rb_pyramid_checked=window.rb_pyramid.isChecked(),
            rb_dmap_max_checked=window.rb_dmap_max.isChecked(),
            rb_dmap_avg_checked=window.rb_dmap_avg.isChecked(),
            tile_enabled=getattr(window, "tile_enabled", None),
            tile_block_size=getattr(window, "tile_block_size", None),
            tile_overlap=getattr(window, "tile_overlap", None),
            tile_threshold=getattr(window, "tile_threshold", None),
            reg_downscale_width=getattr(window, "reg_downscale_width", None),
            thread_count=getattr(window, "thread_count", 4),
            stackmffv4_batch_size=getattr(window, "stackmffv4_batch_size", 2),
            roi_rect=roi_rect,
            roi_mode=roi_mode,
            roi_base_index=roi_base_index,
            ecc_parallel=getattr(window, "ecc_parallel", True),
            ifcnn_refine=ifcnn_refine,
            need_align_scale=effective_need_align_scale,
            reference_mode=effective_reference_mode,
        )

        self.worker.finished_signal.connect(self.on_render_finished)
        self.worker.error_signal.connect(self.on_render_error)
        self.worker.cancelled_signal.connect(self.on_render_cancelled)
        self._set_stop_enabled(True)
        self.worker.start()

    def on_render_finished(
        self,
        processed_images: List[Any],
        fusion_result: Optional[Any],
        registration_performed: bool,
        alignment_time: float,
        fusion_time: float,
        device_name: str,
    ) -> None:
        window = self.window

        try:
            if fusion_result is not None:
                window.fusion_result = fusion_result
                window.registration_results = processed_images
                # Describes this result for as long as it lives in the output
                # list; written into the file if it is ever saved. The date is
                # the moment the render finished, next to the time it took.
                window.fusion_result_metadata = RenderMetadata(
                    source_path=self._render_source_path,
                    rendered_at=datetime.now(),
                    duration_s=alignment_time + fusion_time,
                    options=self._describe_render(fusion_result, device_name),
                )

                # If this is fusion in ROI mode, exit ROI mode (but keep the aligned images for reuse)
                if getattr(window, 'roi_mode_active', False):
                    window.roi_mode_active = False
                    # Do not clear roi_aligned_images, keep it for the next reuse
                    if hasattr(window, 'lbl_result_img'):
                        window.lbl_result_img.roi_mode = False
                        window.lbl_result_img.set_roi_rect(None)
                    # Deselect the ROI button
                    if hasattr(window, 'btn_preview_roi'):
                        window.btn_preview_roi.blockSignals(True)
                        window.btn_preview_roi.setChecked(False)
                        window.btn_preview_roi.blockSignals(False)

                window.output_manager.show_fusion_result()
                window.output_manager.update_output_list_for_fusion()

                window.result_control_bar.setVisible(False)
                window.result_slider.setEnabled(False)
                window.current_result_index = -1
                window.add_label_action.setEnabled(True)

                print("Fusion completed successfully!")
            else:
                if registration_performed:
                    window.fusion_result = None
                    # No fused result to describe; the history keeps its own records.
                    window.fusion_result_metadata = None
                    window.registration_results = processed_images

                    window.result_slider.setEnabled(True)
                    window.result_slider.setRange(0, len(window.registration_results) - 1)
                    window.result_control_bar.setVisible(True)

                    window.current_result_index = 0
                    window.update_result_view(0)
                    window.add_label_action.setEnabled(True)

                    print("Registration completed successfully!")
                else:
                    print("No operation selected. Please select registration options or fusion method.")

            # Only cache aligned images if we performed a FULL registration (no ROI cropping)
            # Note: self.worker may be None if error occurred, so we check first
            # A subset render only aligns the ticked frames, so its output does
            # not describe the full stack and must not become the cache.
            worker = self.worker
            if (
                registration_performed
                and worker is not None
                and not getattr(worker, 'roi_rect', None)
                and not self._subset_render
            ):
                window.aligned_images = processed_images
                window.is_images_aligned = True
                window.last_alignment_options = (
                    worker.need_align_scale,
                    worker.need_align_homography,
                    worker.need_align_ecc,
                    worker.reference_mode,
                )

            total_time = alignment_time + fusion_time

            info_lines = []

            if registration_performed:
                align_methods = []
                if window.cb_align_scale.isChecked():
                    align_methods.append(trans.t("check_align_scale"))
                if window.cb_align_ecc.isChecked():
                    align_methods.append(trans.t("check_align_ecc"))
                if window.cb_align_homography.isChecked():
                    align_methods.append(trans.t("check_align_homography"))
                align_method_str = ", ".join(align_methods) if align_methods else trans.t("val_none")
                info_lines.append(trans.t("info_align_method").format(align_method_str))
                info_lines.append(trans.t("info_align_time").format(alignment_time))
            else:
                info_lines.append(trans.t("info_align_none_cached"))
                info_lines.append(trans.t("info_align_time").format(0.0))

            if (
                window.rb_a.isChecked()
                or window.rb_b.isChecked()
                or window.rb_c.isChecked()
                or window.rb_gfg.isChecked()
                or window.rb_pyramid.isChecked()
                or window.rb_dmap_max.isChecked()
                or window.rb_dmap_avg.isChecked()
                or window.rb_d.isChecked()
            ):
                if window.rb_a.isChecked():
                    method_name = trans.t("radio_guided_filter")
                elif window.rb_b.isChecked():
                    method_name = trans.t("radio_dct")
                elif window.rb_c.isChecked():
                    method_name = trans.t("radio_dtcwt")
                elif window.rb_gfg.isChecked():
                    method_name = trans.t("radio_gfg")
                elif window.rb_pyramid.isChecked():
                    method_name = trans.t("radio_pyramid")
                elif window.rb_dmap_max.isChecked():
                    method_name = trans.t("radio_depthmap_max")
                elif window.rb_dmap_avg.isChecked():
                    method_name = trans.t("radio_depthmap_avg")
                elif window.rb_d.isChecked():
                    method_name = trans.t("radio_stackmff")
                else:
                    method_name = trans.t("radio_guided_filter")

                if window.cb_ifcnn.isChecked():
                    method_name = trans.t("info_fusion_method_refined").format(method_name)

                info_lines.append(trans.t("info_fusion_method").format(method_name))
                info_lines.append(trans.t("info_fusion_time").format(fusion_time))
                info_lines.append(trans.t("info_proc_unit").format(device_name))
            else:
                info_lines.append(trans.t("info_fusion_none"))
                info_lines.append(trans.t("info_fusion_time").format(0.0))

            info_lines.append(trans.t("info_total_time").format(total_time))

            show_custom_message_box(
                window,
                trans.t("dialog_completed_title"),
                trans.t("dialog_completed_msg"),
                "\n".join(info_lines),
                QMessageBox.Icon.Information,
            )

        except Exception as exc:
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_proc_error"),
                str(exc),
                QMessageBox.Icon.Critical,
            )
            traceback.print_exc()

        finally:
            # Restore the UI controls
            self._set_controls_enabled(True)

            window.btn_render.setEnabled(True)
            window.btn_render.setText(trans.t('btn_render'))
            self._set_stop_enabled(False)
            self._set_console_progress(False)
            self.worker = None

    def on_render_cancelled(self) -> None:
        """Restore the UI after the worker unwinds from a user-requested stop."""
        window = self.window

        self._set_controls_enabled(True)
        window.btn_render.setEnabled(True)
        window.btn_render.setText(trans.t('btn_render'))
        self._set_stop_enabled(False)
        self._set_console_progress(False)
        self.worker = None

    def on_render_error(self, error_message: str) -> None:
        window = self.window

        self._set_controls_enabled(True)
        window.btn_render.setEnabled(True)
        window.btn_render.setText(trans.t('btn_render'))
        self._set_stop_enabled(False)
        self._set_console_progress(False)

        show_message_box(
            window,
            trans.t("msg_error"),
            trans.t("msg_proc_error"),
            error_message,
            QMessageBox.Icon.Critical,
        )

        traceback.print_exc()
        self.worker = None
