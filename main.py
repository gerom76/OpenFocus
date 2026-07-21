import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QMessageBox,
    QDialog,
)
from PyQt6.QtCore import Qt, QUrl, QEvent
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent, QImage, QPixmap
from core import ImageStackLoader
from core.app import OpenFocusApplication, process_command_line_args
from core.workers import ROIAlignmentWorker
import cv2
from ui.styles import GLOBAL_DARK_STYLE
from locales import trans
from dialogs import (
    EnvironmentInfoDialog,
    ContactInfoDialog,
    BatchProcessingDialog,
    TileSettingsDialog,
    ThreadSettingsDialog,
    StackMFFV4BatchSettingsDialog,
    ROIRenderOptionsDialog,
)
from dialogs.help import RenderMethodHelpDialog
from dialogs.settings import RegistrationSettingsDialog
from utils import (
    get_ui_font_family,
    get_monospace_font_family,
    get_default_font,
    show_message_box,
    show_warning_box,
    show_error_box,
    show_success_box,
    show_custom_message_box,
    resource_path,
)
from ui.image_panels import create_source_panel, create_result_panel
from widgets import StatusConsole
from ui.menus import setup_menus
from ui.right_panel import bind_right_panel, create_right_panel
from controllers.render_manager import RenderManager
from controllers.output_manager import OutputManager
from controllers.source_manager import SourceManager
from controllers.label_manager import LabelManager
from controllers.export_manager import ExportManager
from controllers.transform_manager import TransformManager
from controllers.batch_manager import BatchManager
from controllers.settings_manager import SettingsManager
from core import is_stackmffv4_available
from constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    TILE_BLOCK_SIZE, TILE_OVERLAP, TILE_THRESHOLD,
    REG_DOWNSCALE_WIDTH, DEFAULT_THREAD_COUNT,
    STACKMFFV4_BATCH_SIZE, ECC_PARALLEL,
)

class OpenFocus(QMainWindow):
    def __init__(self):
        super().__init__()

        app = QApplication.instance()
        version = getattr(app, 'version', '') or getattr(app, 'VERSION', '')
        self.setWindowTitle(f"OpenFocus v{version}" if version else "OpenFocus")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Set the window icon
        icon_path = resource_path("assets", "OpenFocus.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Initialize data as empty
        self.stack_images = []  # List of QPixmap
        self.current_img_index = -1
        self.image_filenames = []  # List of file names
        self.raw_images = []  # Store the original numpy-array images, used for registration and fusion
        self.base_images = []  # Store the base-size images from the first load, used for restoring and resize calculations
        self.fusion_result = None  # Store the latest fusion result
        self.fusion_results = []  # Store the history of all fusion results
        self.registration_results = []  # Store the registered image stack
        self.current_result_index = -1  # Index of the currently displayed result image
        self.aligned_images = []  # Store the aligned image stack to avoid re-aligning
        self.is_images_aligned = False  # Flag indicating whether the images are aligned
        self.last_alignment_options = None  # Store the alignment options used last time
        self.current_kernel_mode = None
        
        # ROI-mode-related attributes
        self.roi_aligned_images = []  # Store the aligned image stack in ROI mode
        self.roi_alignment_worker = None  # ROI registration worker thread
        self.roi_mode_active = False  # Whether ROI mode is active
        self.roi_aligned_raw_count = 0  # Record the number of original images at alignment time, used to decide whether it can be reused

        # Tile settings defaults
        self.tile_enabled = True
        self.tile_block_size = TILE_BLOCK_SIZE
        self.tile_overlap = TILE_OVERLAP
        self.tile_threshold = TILE_THRESHOLD
        # Registration downscale default (the user can change it in Settings -> Registration)
        self.reg_downscale_width = REG_DOWNSCALE_WIDTH
        # Parallel ECC pair computation (user-configurable in Settings -> Registration)
        self.ecc_parallel = ECC_PARALLEL
        # Global thread-count setting, default 4 (can be changed in Settings)
        self.thread_count = DEFAULT_THREAD_COUNT
        # StackMFF V4 batch-size setting, default 2 (can be changed in Settings)
        self.stackmffv4_batch_size = STACKMFFV4_BATCH_SIZE
        
        self.render_manager = RenderManager(self)
        self.output_manager = OutputManager(self)
        self.source_manager = SourceManager(self)
        self.label_manager = LabelManager(self)
        self.export_manager = ExportManager(self)
        self.transform_manager = TransformManager(self)
        self.batch_manager = BatchManager(self)
        self.settings_manager = SettingsManager(self)

        # Initialize the image loader
        self.image_loader = ImageStackLoader()
        
        # Enable drag-and-drop
        self.setAcceptDrops(True)
        
        # Index of the currently displayed image
        self.current_display_index = -1

        self.apply_dark_theme()
        self.init_ui()

        # Install global event filter to handle Space key for panning
        QApplication.instance().installEventFilter(self)

        self._mouse_in_source_preview = False
        self._mouse_in_result_preview = False
        
        # Connect language change signal
        trans.languageChanged.connect(self.update_ui_text)
        
        # Set initial text for drag hint if needed (it is set in image_panels or loaded later, 
        # but let's Ensure it matches current lang if empty)
        # Note: image_panels create_source_panel sets initial text. 
        # We might want to update it here or in update_ui_text called initially?
        # Let's call update_ui_text once to sync everything if needed, or just leave it.
        # Actually image_panels.py uses hardcoded text. 
        # I should probably update image_panels.py too, or just override text here.
        if hasattr(self, 'lbl_source_img'):
             self.lbl_source_img.setText(trans.t('drag_hint'))

        # Restore persisted settings (if openfocus.cfg.json exists). Done last so a
        # restored language triggers update_ui_text via the languageChanged signal.
        self.settings_manager.load_all_settings()

    def init_ui(self):
        setup_menus(self)

        # 2. Main container
        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === Core layout: main splitter ===
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ---------------------------------------------------------
        # A. Image display area (left)
        # ---------------------------------------------------------
        image_display_container = QWidget()
        image_display_layout = QVBoxLayout(image_display_container)
        image_display_layout.setContentsMargins(0, 0, 0, 0) # Flush to the edges
        image_display_layout.setSpacing(0)

        # Dual-view splitter (left: source image stack, right: fusion result)
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- A1/A2. Image view panels ---
        source_panel = create_source_panel()
        self.lbl_source_img = source_panel.image_label
        self.source_control_bar = source_panel.control_bar
        self.stack_slider = source_panel.slider
        self.lbl_stack_info = source_panel.info_label
        self.stack_slider.valueChanged.connect(self.update_source_view)

        # ROI Button
        self.btn_preview_roi = source_panel.roi_btn
        self.btn_preview_roi.toggled.connect(self.toggle_roi_mode)
        # Note: the ROI is now selected in the right panel, so lbl_source_img's roiDeleted no longer controls the button state
        # self.lbl_source_img.roiDeleted.connect(lambda: self.btn_preview_roi.setChecked(False))

        result_panel = create_result_panel()
        self.lbl_result_img = result_panel.image_label
        self.result_control_bar = result_panel.control_bar
        self.result_slider = result_panel.slider
        # Connect the right panel's ROI-mode exit-request signal
        self.lbl_result_img.roiModeExitRequested.connect(self._on_roi_mode_exit_requested)
        self.lbl_result_info = result_panel.info_label
        self.result_slider.valueChanged.connect(self.update_result_view)

        self.lbl_source_img.enterPreview.connect(self._on_enter_source_preview)
        self.lbl_source_img.leavePreview.connect(self._on_leave_source_preview)
        self.lbl_result_img.enterPreview.connect(self._on_enter_result_preview)
        self.lbl_result_img.leavePreview.connect(self._on_leave_result_preview)

        self.view_splitter.addWidget(source_panel.widget)
        self.view_splitter.addWidget(result_panel.widget)
        
        # Listen for splitter-move events to update the image size in real time
        self.view_splitter.splitterMoved.connect(self.on_splitter_moved)
        
        image_display_layout.addWidget(self.view_splitter)
        self.main_splitter.addWidget(image_display_container)

        # ---------------------------------------------------------
        # B. Right-side control panel
        # ---------------------------------------------------------
        right_panel_components = create_right_panel()
        self.right_panel_components = right_panel_components
        self.right_splitter = right_panel_components.splitter
        self.btn_reset = right_panel_components.btn_reset
        self.btn_render = right_panel_components.btn_render
        self.btn_method_help = right_panel_components.btn_method_help
        self.btn_reg_help = right_panel_components.btn_reg_help
        self.rb_a = right_panel_components.rb_a
        self.rb_b = right_panel_components.rb_b
        self.rb_c = right_panel_components.rb_c
        self.rb_gfg = right_panel_components.rb_gfg
        self.rb_d = right_panel_components.rb_d
        self.cb_align_homography = right_panel_components.cb_align_homography
        self.cb_align_ecc = right_panel_components.cb_align_ecc
        self.slider_smooth = right_panel_components.slider_smooth
        self.lbl_smooth_value = right_panel_components.smooth_value_label
        self.smooth_widget = right_panel_components.smooth_widget
        self.source_images_label = right_panel_components.source_images_label
        self.file_list = right_panel_components.file_list
        self.output_label = right_panel_components.output_label
        self.output_list = right_panel_components.output_list
        
        # Status labels
        self.lbl_status_loaded = right_panel_components.lbl_status_loaded
        self.lbl_status_resolution = right_panel_components.lbl_status_resolution
        self.lbl_status_gpu = right_panel_components.lbl_status_gpu
        self.lbl_status_memory = right_panel_components.lbl_status_memory

        self.main_splitter.addWidget(right_panel_components.widget)
        bind_right_panel(self, right_panel_components)

        self._configure_fusion_method_availability()

        # Make sure the splitter has added its child widgets before setting the collapsible property
        # Use a QTimer to delay setting the collapsible property, ensuring all child widgets have been added correctly
        from PyQt6.QtCore import QTimer
        
        # Timer for system status updates
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_dynamic_status)
        self.status_timer.start(2000) # Update every 2 seconds

        def set_splitter_properties():
            try:
                if self.view_splitter.count() >= 2:
                    self.view_splitter.setCollapsible(0, False)
                    self.view_splitter.setCollapsible(1, False)
                if self.right_splitter.count() >= 3:
                    self.right_splitter.setCollapsible(0, False)
                    self.right_splitter.setCollapsible(1, False)
                    self.right_splitter.setCollapsible(2, False)
                if self.main_splitter.count() >= 2:
                    self.main_splitter.setCollapsible(0, False)
                    self.main_splitter.setCollapsible(1, False)
            except IndexError:
                pass

        QTimer.singleShot(100, set_splitter_properties)
        
        # Ratio setting: the left image area takes most of the space, the right control panel is adjustable
        self.main_splitter.setStretchFactor(0, 3)  # Left image area stretch factor is 3
        self.main_splitter.setStretchFactor(1, 1)  # Right control panel stretch factor is 1
        
        # Set the initial split ratio (optional)
        # Get the window width and set the initial ratio to 75% : 25%
        total_width = self.width()
        self.main_splitter.setSizes([int(total_width * 0.75), int(total_width * 0.25)])

        # ---------------------------------------------------------
        # C. Status output area (bottom) — mirrors the terminal output
        # ---------------------------------------------------------
        self.status_console = StatusConsole()

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.addWidget(self.main_splitter)
        self.vertical_splitter.addWidget(self.status_console)
        self.vertical_splitter.setStretchFactor(0, 5)
        self.vertical_splitter.setStretchFactor(1, 0)
        self.vertical_splitter.setCollapsible(0, False)
        self.vertical_splitter.setCollapsible(1, True)

        total_height = self.height()
        console_height = max(120, int(total_height * 0.18))
        self.vertical_splitter.setSizes([total_height - console_height, console_height])

        main_layout.addWidget(self.vertical_splitter)

    def eventFilter(self, obj, event):
        """Global event filter to handle Space key and Delete key interactions."""
        # Handle Delete/Backspace for ROI
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
                # Check if mouse is over source image and ROI mode is active
                if hasattr(self, 'lbl_source_img') and self.lbl_source_img:
                    # Check if widget is visible and under mouse
                    if self.lbl_source_img.isVisible() and self.lbl_source_img.underMouse():
                         # If ROI exists, delete it and consume event
                         if self.lbl_source_img.get_roi_rect() is not None:
                             self.lbl_source_img.set_roi_rect(None)
                             self.lbl_source_img.roiDeleted.emit()
                             return True # Event handled, do not propagate to list widget

        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            # Check if mouse is over source or result image
            handled = False
            
            # Using underMouse() is safer than tracking enter/leave events which can be unreliable
            # during rapid movement or with other widgets involved
            if hasattr(self, 'lbl_source_img') and self.lbl_source_img and self.lbl_source_img.isVisible():
                if self.lbl_source_img.underMouse():
                    self.lbl_source_img.set_space_pressed(True)
                    handled = True
            
            if hasattr(self, 'lbl_result_img') and self.lbl_result_img and self.lbl_result_img.isVisible():
                if self.lbl_result_img.underMouse():
                    self.lbl_result_img.set_space_pressed(True)
                    handled = True
            
            if handled:
                return True
                
        elif event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                if hasattr(self, 'lbl_source_img') and self.lbl_source_img:
                    self.lbl_source_img.set_space_pressed(False)
                
                if hasattr(self, 'lbl_result_img') and self.lbl_result_img:
                    self.lbl_result_img.set_space_pressed(False)
        
        return super().eventFilter(obj, event)

    # --- Status Display ---
    def update_loaded_status(self):
        """Update loaded images count and resolution info."""
        count = len(self.stack_images)
        if count > 0:
            # Calculate average size (approximate from QPixmap or numpy array if available)
            # self.raw_images stores numpy arrays
            total_size_mb = 0
            if self.raw_images:
                # height * width * channels * 1 byte (uint8) / 1024 / 1024
                # assuming 3 channels uint8
                try:
                    total_size_bytes = sum(img.nbytes for img in self.raw_images)
                    avg_size_mb = (total_size_bytes / count) / (1024 * 1024)
                    self.lbl_status_loaded.setText(trans.t('status_loaded_fmt').format(count, avg_size_mb))
                    
                    h, w = self.raw_images[0].shape[:2]
                    self.lbl_status_resolution.setText(trans.t('status_res_fmt').format(w, h))
                except Exception:
                    self.lbl_status_loaded.setText(trans.t('status_loaded').format(count))
                    self.lbl_status_resolution.setText(trans.t('status_res').format('-'))
            else:
                self.lbl_status_loaded.setText(trans.t('status_loaded').format(count))
                self.lbl_status_resolution.setText(trans.t('status_res').format('-'))
        else:
            self.lbl_status_loaded.setText(trans.t('status_loaded').format(0))
            self.lbl_status_resolution.setText(trans.t('status_res').format('-'))

    def _update_dynamic_status(self):
        """Update GPU and Memory status."""
        # Memory
        try:
            import ctypes
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            
            used_gb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024**3)
            total_gb = stat.ullTotalPhys / (1024**3)
            self.lbl_status_memory.setText(trans.t('status_ram_fmt').format(used_gb, total_gb))
        except Exception:
            self.lbl_status_memory.setText(trans.t('status_ram').format('N/A'))

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.lbl_status_gpu.setText(trans.t('status_gpu').format('CUDA'))
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.lbl_status_gpu.setText(trans.t('status_gpu').format('MPS'))
            else:
                self.lbl_status_gpu.setText(trans.t('status_gpu').format('N/A'))
        except ImportError:
            self.lbl_status_gpu.setText(trans.t('status_gpu').format('N/A'))
        except Exception:
            self.lbl_status_gpu.setText(trans.t('status_gpu').format('Err'))

    # --- Logic control ---

    def reset_to_default(self):
        """Reset to the default state"""
        # Reset the render method - method A selected by default
        self.rb_a.setChecked(True)
        self.rb_b.setChecked(False)
        self.rb_c.setChecked(False)
        self.rb_gfg.setChecked(False)
        self.rb_d.setChecked(False)
        
        # Reset the registration options - ECC selected by default, Homography unselected
        self.cb_align_homography.setChecked(False)
        self.cb_align_ecc.setChecked(True)
        
        # Reset the slider value to the default
        self.slider_smooth.setValue(31)
        
        # Update slider availability
        self.update_slider_availability()

        # Reset the image-display zoom and magnifier state
        self.lbl_source_img.reset_view()
        self.lbl_result_img.reset_view()
        
        # Restore the empty-state hint (if no image is currently loaded)
        if hasattr(self, 'stack_images') and not self.stack_images:
            self.lbl_source_img.setText(trans.t('drag_hint'))
            self.lbl_source_img.setFont(QFont(get_default_font(), 16))
    


    def _configure_fusion_method_availability(self) -> None:
        """Disable fusion methods whose dependencies are not installed."""
        if not is_stackmffv4_available():
            self.rb_d.setEnabled(False)
            self.rb_d.setToolTip(trans.t("msg_stackmff_unavailable_text"))
        else:
            self.rb_d.setEnabled(True)
            self.rb_d.setToolTip("")


    def update_source_view(self, index):
        if not self.stack_images:
            return
            
        if 0 <= index < len(self.stack_images):
            self.current_display_index = index
            
            try:
                # Get the original image and overlay labels as needed
                original_pixmap = self.stack_images[index]
                display_pixmap = self.label_manager.apply_labels_to_source_pixmap(original_pixmap, index)

                # Use a custom label to handle zooming and zoom reset
                self.lbl_source_img.set_display_pixmap(display_pixmap)
                
                # Update the text
                self.lbl_stack_info.setText(f"{index + 1} / {len(self.stack_images)}")
                
                # Only set the list when the list selection and the current slider disagree, to prevent a signal loop
                if self.file_list.currentRow() != index:
                    self.file_list.setCurrentRow(index)
            except Exception as e:
                show_message_box(
                    self,
                    trans.t("msg_display_error_title"),
                    trans.t("msg_display_source_failed_text"),
                    f"Error: {str(e)}",
                    QMessageBox.Icon.Critical
                )
    
    
    def handle_method_selection(self, selected_button):
        """Handle fusion-method selection, mutually exclusive but cancelable"""
        # If the clicked button is already selected, deselect it
        if selected_button.isChecked():
            # Deselect the other buttons
            for btn in [self.rb_a, self.rb_b, self.rb_c, self.rb_gfg, self.rb_d]:
                if btn != selected_button:
                    btn.setChecked(False)
        # If it was not selected when clicked, do nothing (already auto-deselected)
    
    # handle_align_exclusive removed, since the mutual-exclusion logic is no longer needed

    def handle_kernel_slider_change(self, value):
        """Ensure the kernel size is always odd, and update the display label"""
        adjusted = int(value)
        if adjusted % 2 == 0:
            if adjusted >= self.slider_smooth.maximum():
                adjusted -= 1
            else:
                adjusted += 1
            self.slider_smooth.blockSignals(True)
            self.slider_smooth.setValue(adjusted)
            self.slider_smooth.blockSignals(False)

        self.lbl_smooth_value.setText(str(adjusted))

    def update_slider_availability(self):
        """Update slider availability and default value based on the selected fusion method"""
        # Guided Filter/DCT: shared kernel slider
        # DTCWT / StackMFF-V4: no slider used

        if self.rb_a.isChecked():
            self.smooth_widget.setEnabled(True)
            if self.current_kernel_mode != "guided":
                self.slider_smooth.setValue(31)
            self.current_kernel_mode = "guided"
        elif self.rb_b.isChecked():
            self.smooth_widget.setEnabled(True)
            if self.current_kernel_mode != "dct":
                self.slider_smooth.setValue(7)
            self.current_kernel_mode = "dct"
        elif self.rb_gfg.isChecked():
            # GFG-FGF uses the initial mean/blur kernel controlled by the same slider
            self.smooth_widget.setEnabled(True)
            if self.current_kernel_mode != "gfg":
                # GFG-FGF uses kernel=7 by default
                self.slider_smooth.setValue(7)
            self.current_kernel_mode = "gfg"
        else:
            self.smooth_widget.setEnabled(False)
            self.current_kernel_mode = None
    
    def update_result_view(self, index):
        """Update the image shown in the Output area (for registration results or aligned images in ROI mode)"""
        # If ROI mode is active, show the aligned image stack
        if getattr(self, 'roi_mode_active', False) and self.roi_aligned_images:
            self._display_roi_aligned_image(index)
        else:
            self.output_manager.show_registration_result(index)
    
    # --- File-loading features ---
    
    def open_folder_dialog(self):
        """Open the folder-selection dialog"""
        self.source_manager.prompt_and_load_stack()
    
    def open_video_dialog(self):
        """Open the video-file selection dialog"""
        self.source_manager.prompt_and_load_video()
    
    def show_environment_info(self):
        """Show the environment-info dialog"""
        dialog = EnvironmentInfoDialog(self)
        dialog.exec()
    
    def display_fusion_result(self):
        """Show the fusion result in the right-side preview area"""
        self.output_manager.show_fusion_result()
    
    # --- Drag-and-drop features ---
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Drag-enter event"""
        if self.source_manager.can_accept_drag(event):
            event.acceptProposedAction()
            return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Drop event"""
        self.source_manager.handle_drop_event(event)
    
    def on_splitter_moved(self, pos, index):
        """Rescale the image when the splitter moves"""
        self.source_manager.refresh_current_source_view()
        self.output_manager.refresh_current_result_view()
    
    def resizeEvent(self, event):
        """Rescale the image when the window size changes"""
        super().resizeEvent(event)
        self.source_manager.refresh_current_source_view()
        self.output_manager.refresh_current_result_view()

    def toggle_roi_mode(self, enabled: bool):
        """Toggle ROI selection mode. 
        When enabled, first align images using ECC, then display aligned stack on result panel for ROI selection.
        """
        if enabled:
            # Check whether there are enough images
            if not self.raw_images or len(self.raw_images) < 2:
                show_warning_box(self, trans.t("msg_warning"), trans.t("roi_need_images"))
                self.btn_preview_roi.blockSignals(True)
                self.btn_preview_roi.setChecked(False)
                self.btn_preview_roi.blockSignals(False)
                return
            
            # Check whether the existing aligned images can be reused
            can_reuse = (
                len(self.roi_aligned_images) > 0 and
                len(self.roi_aligned_images) == len(self.raw_images) and
                self.roi_aligned_raw_count == len(self.raw_images)
            )
            
            if can_reuse:
                # Use the existing aligned images directly
                self.roi_mode_active = True
                self._display_roi_aligned_image(0)
                
                # Enable the slider
                self.result_slider.setRange(0, len(self.roi_aligned_images) - 1)
                self.result_slider.setValue(0)
                self.result_slider.setEnabled(True)
                self.result_control_bar.setVisible(True)
                
                # Enable the ROI-selection mode in the right panel
                if hasattr(self, 'lbl_result_img'):
                    self.lbl_result_img.roi_mode = True
                return
            
            # Re-alignment needed: disable the button and show a processing state
            self.btn_preview_roi.setEnabled(False)
            self.btn_preview_roi.setText(trans.t("roi_aligning"))
            self.status_console.start_progress()
            QApplication.processEvents()
            
            # Start the ECC registration thread
            self.roi_alignment_worker = ROIAlignmentWorker(
                self.raw_images,
                reg_downscale_width=self.reg_downscale_width,
                thread_count=self.thread_count,
                ecc_parallel=self.ecc_parallel
            )
            self.roi_alignment_worker.finished_signal.connect(self._on_roi_alignment_finished)
            self.roi_alignment_worker.error_signal.connect(self._on_roi_alignment_error)
            self.roi_alignment_worker.start()
        else:
            # Exit ROI mode
            self.roi_mode_active = False
            if hasattr(self, 'lbl_result_img'):
                self.lbl_result_img.roi_mode = False
            # Restore the right-panel display
            if self.fusion_result is not None:
                self.output_manager.show_fusion_result()
            elif self.registration_results:
                self.output_manager.show_registration_result(self.current_result_index)
            else:
                self.lbl_result_img.clear()
                self.lbl_result_img.setText(trans.t("result_hint"))
            # Clear the ROI aligned images
            self.roi_aligned_images = []

    def _on_roi_alignment_finished(self, aligned_images, alignment_time):
        """Callback for when ROI registration completes"""
        self.roi_aligned_images = aligned_images
        self.roi_aligned_raw_count = len(self.raw_images)  # Record the number of original images at alignment time
        self.roi_mode_active = True
        
        # Restore the button state
        self.btn_preview_roi.setEnabled(True)
        self.btn_preview_roi.setText(trans.t("btn_roi"))
        self.status_console.stop_progress()

        # Show the first aligned image in the right-side result panel
        if aligned_images and len(aligned_images) > 0:
            self._display_roi_aligned_image(0)
            
            # Enable the slider so the user can browse the aligned image stack
            self.result_slider.setRange(0, len(aligned_images) - 1)
            self.result_slider.setValue(0)
            self.result_slider.setEnabled(True)
            self.result_control_bar.setVisible(True)
            
            # Enable the ROI-selection mode in the right panel
            if hasattr(self, 'lbl_result_img'):
                self.lbl_result_img.roi_mode = True
                
            # Show a hint message
            show_message_box(
                self, 
                trans.t("roi_ready_title"),
                trans.t("roi_ready_msg"),
                trans.t("roi_ready_detail").format(alignment_time),
                QMessageBox.Icon.Information
            )
        
        self.roi_alignment_worker = None

    def _on_roi_alignment_error(self, error_message):
        """Callback for a ROI registration error"""
        self.btn_preview_roi.setEnabled(True)
        self.btn_preview_roi.setText(trans.t("btn_roi"))
        self.btn_preview_roi.blockSignals(True)
        self.btn_preview_roi.setChecked(False)
        self.btn_preview_roi.blockSignals(False)
        self.status_console.stop_progress()

        show_error_box(
            self,
            trans.t("msg_error"),
            trans.t("roi_align_failed"),
            error_message
        )
        
        self.roi_alignment_worker = None

    def _display_roi_aligned_image(self, index: int):
        """Show the ROI-mode aligned image in the right-side result panel"""
        if not self.roi_aligned_images or index < 0 or index >= len(self.roi_aligned_images):
            return
        
        # Save the current ROI region so the ROI selection is kept when switching images
        current_roi = None
        if hasattr(self, 'lbl_result_img') and self.lbl_result_img.get_roi_rect() is not None:
            current_roi = self.lbl_result_img.get_roi_rect()
        
        try:
            image = self.roi_aligned_images[index]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _channels = rgb_image.shape
            bytes_per_line = 3 * width
            
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.lbl_result_img.set_display_pixmap(pixmap)
            self.lbl_result_info.setText(f"{index + 1} / {len(self.roi_aligned_images)}")
            # Keep the control bar visible in ROI mode so the user can switch images
            self.result_control_bar.setVisible(True)
            
            # Restore the previously selected ROI region
            if current_roi is not None:
                self.lbl_result_img.set_roi_rect(current_roi)
        except Exception as exc:
            show_error_box(self, trans.t("msg_error"), trans.t("roi_display_failed"), str(exc))

    def _on_roi_mode_exit_requested(self):
        """Handle a ROI-mode exit request (click the X button or right-click outside the ROI area)"""
        if self.roi_mode_active:
            # Deselect the ROI button, which triggers toggle_roi_mode(False)
            self.btn_preview_roi.setChecked(False)

    def apply_dark_theme(self):
        # Dynamically replace with the platform-specific font
        ui_font = get_ui_font_family()
        mono_font = get_monospace_font_family()
        style_sheet = GLOBAL_DARK_STYLE.replace('"Segoe UI", "Microsoft YaHei"', ui_font).replace('Consolas, "Segoe UI", monospace', mono_font)
        self.setStyleSheet(style_sheet)

    def show_contact_info(self):
        """Show the contact info"""
        # Create the contact-info dialog
        dialog = ContactInfoDialog(self)
        dialog.exec()
    
    def show_batch_processing_dialog(self, preload_folder_paths: list[str] = None, scale_factor: float = 1.0):
        """Show the batch-processing settings dialog

        Args:
            preload_folder_paths: optional, list of folder paths to preload (for drag-and-drop scenarios)
            scale_factor: scale factor, used to scale images during preloading
        """
        from dialogs import BatchProcessingDialog

        dialog = BatchProcessingDialog(self)

        if preload_folder_paths:
            if len(preload_folder_paths) == 1:
                dialog.preload_single_folder(preload_folder_paths[0], scale_factor)
            else:
                dialog.preload_multiple_folders(preload_folder_paths, scale_factor)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            folder_paths = dialog.folder_paths
            output_type, output_path = dialog.get_output_settings()
            processing_settings = dialog.get_processing_settings()
            import_mode = dialog.get_import_mode()
            split_method, split_param = dialog.get_split_settings()

            if import_mode == "single_folder":
                self.batch_manager.start_batch_processing(
                    folder_paths=[],
                    output_type=output_type,
                    output_path=output_path,
                    processing_settings=processing_settings,
                    import_mode=import_mode,
                    split_method=split_method,
                    split_param=split_param,
                    single_folder_images_with_times=dialog.single_folder_images_with_times,
                )
            else:
                self.batch_manager.start_batch_processing(
                    folder_paths=folder_paths,
                    output_type=output_type,
                    output_path=output_path,
                    processing_settings=processing_settings,
                    import_mode=import_mode,
                )

    def show_tile_settings(self):
        """Show the Tile settings dialog"""
        dialog = TileSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Settings have been written back to self attributes by the dialog; trigger any necessary refresh here
            self.update_slider_availability()
            # If needed, reflect the setting changes in the right panel or elsewhere
            return True
        return False

    def show_registration_settings(self):
        """Show the registration settings dialog, allowing the user to change downscale_width"""
        from dialogs import RegistrationSettingsDialog
        dialog = RegistrationSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # the value has been written back to self.reg_downscale_width
            return True
        return False

    def show_thread_settings(self):
        """Show the thread-count settings dialog"""
        from dialogs import ThreadSettingsDialog
        dialog = ThreadSettingsDialog(self)
        dialog.exec()

    def show_stackmffv4_batch_settings(self):
        """Show the StackMFF V4 batch-size settings dialog"""
        dialog = StackMFFV4BatchSettingsDialog(self)
        dialog.exec()

    def save_all_settings(self):
        """Save all settings to openfocus.cfg.json."""
        self.settings_manager.save_all_settings()

    def rotate_stack(self, rotation_code):
        """Delegate stack rotation to the transform manager."""
        self.transform_manager.rotate_stack(rotation_code)

    def reload_image_stack(self):
        """Ensure legacy callers refresh through the transform manager."""
        self.transform_manager.reload_image_stack()

    def flip_stack(self, flip_code):
        """Mirror the stack via the transform manager."""
        self.transform_manager.flip_stack(flip_code)

    def resize_all_images(self):
        """Resize images through the transform manager."""
        self.transform_manager.resize_all_images()

    def _on_enter_source_preview(self):
        self._mouse_in_source_preview = True

    def _on_leave_source_preview(self):
        self._mouse_in_source_preview = False

    def _on_enter_result_preview(self):
        self._mouse_in_result_preview = True

    def _on_leave_result_preview(self):
        self._mouse_in_result_preview = False

    def _is_mouse_in_preview(self) -> bool:
        return self._mouse_in_source_preview or self._mouse_in_result_preview

    def closeEvent(self, event):
        """Clean up background threads before closing the window."""
        # Stop ROI alignment worker if running
        if self.roi_alignment_worker is not None:
            if self.roi_alignment_worker.isRunning():
                self.roi_alignment_worker.quit()
                self.roi_alignment_worker.wait(1000)  # Wait up to 1 second
            self.roi_alignment_worker = None

        # Stop render worker if running
        if hasattr(self, 'render_manager') and self.render_manager.worker is not None:
            if self.render_manager.worker.isRunning():
                self.render_manager.worker.quit()
                self.render_manager.worker.wait(1000)
            self.render_manager.worker = None

        # Stop batch worker if running
        if hasattr(self, 'batch_manager') and self.batch_manager._thread is not None:
            if self.batch_manager._worker:
                self.batch_manager._worker.cancel()
            self.batch_manager._teardown_worker()

        # Stop status update timer
        if hasattr(self, 'status_timer') and self.status_timer.isActive():
            self.status_timer.stop()

        # Give stdout/stderr back to the real terminal
        if hasattr(self, 'status_console'):
            self.status_console.restore_streams()

        super().closeEvent(event)

    # --- Status console ---

    def toggle_status_console(self, visible: bool) -> None:
        """Show or hide the bottom status output area."""
        if not hasattr(self, 'status_console'):
            return
        self.status_console.setVisible(visible)
        if visible and self.vertical_splitter.sizes()[1] == 0:
            total = self.vertical_splitter.height()
            console_height = max(120, int(total * 0.18))
            self.vertical_splitter.setSizes([total - console_height, console_height])

    # --- Language ---
    
    def set_language(self, lang_code: str) -> None:
        """Switch application language."""
        trans.set_language(lang_code)

    def update_ui_text(self) -> None:
        """Update all UI strings based on current language."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        # Update Menus
        if hasattr(self, 'ui_objs'):
             for key, obj in self.ui_objs.items():
                t_key = obj.property("trans_key")
                if not t_key:
                    t_key = key
                
                text = trans.t(t_key)
                if isinstance(obj, QMenu):
                    obj.setTitle(text)
                elif isinstance(obj, QAction):
                    obj.setText(text)
        
        # Update Right Panel
        c = self.right_panel_components
        c.method_group.setTitle(trans.t('group_fusion'))
        c.rb_a.setText(trans.t('radio_guided_filter'))
        c.rb_b.setText(trans.t('radio_dct'))
        c.rb_c.setText(trans.t('radio_dtcwt'))
        c.rb_gfg.setText(trans.t('radio_gfg'))
        c.rb_d.setText(trans.t('radio_stackmff'))
        
        c.registration_group.setTitle(trans.t('group_registration'))
        c.cb_align_ecc.setText(trans.t('check_align_ecc'))
        c.cb_align_homography.setText(trans.t('check_align_homography'))
        
        c.lbl_kernel.setText(trans.t('label_kernel'))
        c.btn_reset.setText(trans.t('btn_reset'))
        c.btn_render.setText(trans.t('btn_render'))
        
        # ROI Button
        if hasattr(self, 'btn_preview_roi'):
            self.btn_preview_roi.setText(trans.t('btn_roi'))

        # Lists labels
        count = self.file_list.count()
        c.source_images_label.setText(trans.t('label_source_images').format(count))
        
        out_count = self.output_list.count()
        c.output_label.setText(trans.t('label_output').format(out_count))
        
        # Status
        self.update_loaded_status()
        self._update_dynamic_status()

        # Status console
        if hasattr(self, 'status_console'):
            self.status_console.update_ui_text()
        
        # Drag hint
        if not self.stack_images:
            if hasattr(self, 'lbl_source_img'):
                self.lbl_source_img.setText(trans.t('drag_hint'))

        # Update language checked state
        if 'action_lang_en' in self.ui_objs:
            is_en = trans.current_lang == 'en'
            self.ui_objs['action_lang_en'].setChecked(is_en)
            self.ui_objs['action_lang_zh'].setChecked(not is_en)


if __name__ == "__main__":
    app = OpenFocusApplication(sys.argv)
    window = OpenFocus()
    app.set_main_window(window)
    
    # Handle files passed via command-line (drag-to-EXE, desktop file, etc.)
    if len(sys.argv) > 1:
        process_command_line_args(window, sys.argv[1:])
    
    window.show()
    sys.exit(app.exec())