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
from PyQt6.QtCore import Qt, QUrl, QEvent, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent
from core import ImageStackLoader
from core.app import OpenFocusApplication, process_command_line_args
from core.workers import ROIAlignmentWorker
import cv2
from ui.styles import GLOBAL_DARK_STYLE
from locales import trans
from dialogs import (
    EnvironmentInfoDialog,
    ContactInfoDialog,

    TileSettingsDialog,

    StackMFFV4BatchSettingsDialog,

)

from utils import (
    get_ui_font_family,
    get_monospace_font_family,
    get_default_font,
    show_message_box,
    show_warning_box,
    show_error_box,
    resource_path,
    bitdepth,
    cv2_to_pixmap,
)
from core import contrast
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
from fusion_methods.ifcnn import get_ifcnn_model_path, is_ifcnn_available
from constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    TILE_BLOCK_SIZE, TILE_OVERLAP, TILE_THRESHOLD,
    REG_DOWNSCALE_WIDTH, DEFAULT_THREAD_COUNT,
    STACKMFFV4_BATCH_SIZE, ECC_PARALLEL,
    REFERENCE_FRAME_MODE,
    KERNEL_SIZE_MAX, KERNEL_SIZE_MAX_DCT,
    KERNEL_SIZE_DEFAULT_DCT, KERNEL_SIZE_DEFAULT_GFF,
    KERNEL_SIZE_DEFAULT_GFG, KERNEL_SIZE_DEFAULT_DMAP,
    KERNEL_SIZE_DEFAULT_PYRAMID,
    DCT_PLATEAU_PRESETS,
    PYRAMID_BASE_PRESETS, PYRAMID_COHERENCE_PRESETS,
    PYRAMID_LEVELS, PYRAMID_SELECTIVITY_PRESETS,
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
        # Full paths of the source frames, index-aligned with image_filenames.
        # A saved result reads its EXIF back from the first of them, so the list
        # is kept in step with every append, delete and clear of the stack.
        self.image_paths = []
        self.raw_images = []  # Store the original numpy-array images, used for registration and fusion
        self.base_images = []  # Store the base-size images from the first load, used for restoring and resize calculations
        self.fusion_result = None  # Store the latest fusion result
        self.fusion_results = []  # Store the history of all fusion results
        # One RenderMetadata per entry of fusion_results, same order: what the
        # render was based on, when it ran and how long it took. Written into
        # the file when that result is saved.
        self.fusion_metadata = []
        self.fusion_result_metadata = None  # Metadata of the latest fusion result
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
        # Reference frame for registration: 'first' (frame 0, chained - default)
        # or 'middle'. User-configurable in Settings -> Registration.
        self.reference_frame_mode = REFERENCE_FRAME_MODE
        # Global thread-count setting, default 4 (can be changed in Settings)
        self.thread_count = DEFAULT_THREAD_COUNT
        # GPU image loading (nvJPEG / RAW develop), on by default and a no-op
        # without CUDA. Held on the window for save/restore; the loader reads
        # it from core.gpu_decode.
        self.gpu_loading_enabled = True
        # StackMFF V4 batch-size setting, default 2 (can be changed in Settings)
        self.stackmffv4_batch_size = STACKMFFV4_BATCH_SIZE
        # Processing bit depth: 'auto' follows the source files, '8'/'16' force one.
        # Held on the window only so it can be saved and restored; the pipeline
        # reads it from utils.bitdepth, which is set below.
        self.bit_depth_mode = bitdepth.get_mode()
        # Post-fusion contrast enhancement. 'off' by default so existing results
        # are byte-identical; applied at display and save time only, so the
        # stored fusion result stays pristine and the strength can be changed
        # without re-rendering.
        self.contrast_method = contrast.METHOD_OFF
        self.contrast_strength = 50  # 0-100, only applies when method != 'off'
        
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
        self.source_panel_widget = source_panel.widget
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
        # Double-clicking the output gives it the whole view area
        self.lbl_result_img.doubleClicked.connect(self.toggle_source_panel)

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
        self.btn_stop = right_panel_components.btn_stop
        self.btn_method_help = right_panel_components.btn_method_help
        self.btn_reg_help = right_panel_components.btn_reg_help
        self.rb_a = right_panel_components.rb_a
        self.rb_b = right_panel_components.rb_b
        self.rb_c = right_panel_components.rb_c
        self.rb_gfg = right_panel_components.rb_gfg
        self.rb_pyramid = right_panel_components.rb_pyramid
        self.rb_dmap_max = right_panel_components.rb_dmap_max
        self.rb_dmap_avg = right_panel_components.rb_dmap_avg
        self.rb_d = right_panel_components.rb_d
        self.cb_ifcnn = right_panel_components.cb_ifcnn
        self.cb_align_scale = right_panel_components.cb_align_scale
        self.cb_align_homography = right_panel_components.cb_align_homography
        self.cb_align_ecc = right_panel_components.cb_align_ecc
        self.slider_smooth = right_panel_components.slider_smooth
        self.lbl_smooth_value = right_panel_components.smooth_value_label
        self.smooth_widget = right_panel_components.smooth_widget
        self.slider_halo = right_panel_components.slider_halo
        self.lbl_halo_value = right_panel_components.halo_value_label
        self.halo_widget = right_panel_components.halo_widget
        self.combo_dct_block = right_panel_components.combo_dct_block
        self.combo_dct_plateau = right_panel_components.combo_dct_plateau
        self.cb_dct_blend = right_panel_components.cb_dct_blend
        self.dct_widget = right_panel_components.dct_widget
        self.combo_pyr_levels = right_panel_components.combo_pyr_levels
        self.combo_pyr_selectivity = right_panel_components.combo_pyr_selectivity
        self.combo_pyr_coherence = right_panel_components.combo_pyr_coherence
        self.combo_pyr_base = right_panel_components.combo_pyr_base
        self.cb_pyr_noise_gate = right_panel_components.cb_pyr_noise_gate
        self.cb_pyr_envelope = right_panel_components.cb_pyr_envelope
        self.pyramid_widget = right_panel_components.pyramid_widget
        self.combo_contrast = right_panel_components.combo_contrast
        self.slider_contrast = right_panel_components.slider_contrast
        self.lbl_contrast_value = right_panel_components.contrast_value_label
        self.source_images_label = right_panel_components.source_images_label
        self.file_list = right_panel_components.file_list
        self.btn_select_all = right_panel_components.btn_select_all
        self.btn_select_invert = right_panel_components.btn_select_invert
        self.btn_select_none = right_panel_components.btn_select_none
        self.btn_select_nth = right_panel_components.btn_select_nth
        self.spin_select_nth = right_panel_components.spin_select_nth
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
                    depth = bitdepth.depth_label(self.raw_images)
                    self.lbl_status_loaded.setText(
                        trans.t('status_loaded_depth_fmt').format(count, avg_size_mb, depth)
                    )

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
        self.rb_pyramid.setChecked(False)
        self.rb_dmap_max.setChecked(False)
        self.rb_dmap_avg.setChecked(False)
        self.rb_d.setChecked(False)

        # Reset the post-fusion refinement stage
        self.cb_ifcnn.setChecked(False)

        # Reset the registration options - ECC selected by default, Scale and Homography unselected
        self.cb_align_scale.setChecked(False)
        self.cb_align_homography.setChecked(False)
        self.cb_align_ecc.setChecked(True)
        
        # Reset the slider value to the default
        self.slider_smooth.setValue(31)

        # Halo suppression back to off
        self.slider_halo.setValue(0)

        # Reset contrast to off (strength back to the 50% default)
        self.combo_contrast.setCurrentIndex(0)
        self.slider_contrast.setValue(50)

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

        # The IFCNN stage needs its checkpoint on disk; it is not bundled
        if not is_ifcnn_available():
            self.cb_ifcnn.setChecked(False)
            self.cb_ifcnn.setEnabled(False)
            self.cb_ifcnn.setToolTip(trans.t("msg_ifcnn_unavailable_text").format(path=get_ifcnn_model_path()))
        else:
            self.cb_ifcnn.setEnabled(True)
            self.cb_ifcnn.setToolTip(trans.t("tip_ifcnn_refine"))


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
            for btn in [self.rb_a, self.rb_b, self.rb_c, self.rb_gfg, self.rb_pyramid,
                        self.rb_dmap_max, self.rb_dmap_avg, self.rb_d]:
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

    def handle_halo_slider_change(self, value):
        """Update the halo-radius display label; 0 reads as Off."""
        radius = int(value)
        if radius <= 0:
            self.lbl_halo_value.setText(trans.t('halo_off'))
        else:
            self.lbl_halo_value.setText(f"{radius} px")

    def _set_kernel_range(self, maximum):
        """Point the shared kernel slider at one method's usable range.

        The slider drives a different quantity per method, so its ceiling is not
        one number. The pixel-domain methods stop being useful well before 51.
        DCT's kernel median-filters its focal-plane map, one entry per block, so
        at the default block size it reaches eight times further per step - and
        the distance it has to cover scales with the image: 51 spans a fifth of
        the map on a 2048-wide frame and a fourteenth of it on a 6000-wide one.
        Measured on a 274-frame 2048x1364 stack, quality stops improving at
        about 51 and is flat to 301; the wider ceiling is there for the larger
        frames where the same reach needs a larger number.
        """
        if self.slider_smooth.maximum() == maximum:
            return
        current = self.slider_smooth.value()
        self.slider_smooth.blockSignals(True)
        self.slider_smooth.setMaximum(maximum)
        self.slider_smooth.setValue(min(current, maximum))
        self.slider_smooth.blockSignals(False)
        self.handle_kernel_slider_change(self.slider_smooth.value())

    def update_slider_availability(self):
        """Update slider availability and default value based on the selected fusion method"""
        # Guided Filter/DCT: shared kernel slider
        # DTCWT / StackMFF-V4: no slider used

        # Halo suppression only exists on the depth-map paths; the slider keeps
        # its value while disabled so switching methods does not forget it.
        self.halo_widget.setEnabled(
            self.rb_dmap_max.isChecked() or self.rb_dmap_avg.isChecked())

        # The DCT tuning block keeps its values while disabled, so switching
        # away and back does not forget them.
        self.dct_widget.setEnabled(self.rb_b.isChecked())
        # The pyramid block is hidden rather than greyed out - six controls is
        # too much dead panel to leave standing. Hidden widgets keep their
        # values, so this forgets nothing either.
        self.pyramid_widget.setVisible(self.rb_pyramid.isChecked())

        if self.rb_a.isChecked():
            self.smooth_widget.setEnabled(True)
            self._set_kernel_range(KERNEL_SIZE_MAX)
            if self.current_kernel_mode != "guided":
                self.slider_smooth.setValue(KERNEL_SIZE_DEFAULT_GFF)
            self.current_kernel_mode = "guided"
        elif self.rb_b.isChecked():
            self.smooth_widget.setEnabled(True)
            # DCT's kernel smooths its focal-plane map, which is one entry per
            # block rather than per pixel, so the same number reaches eight
            # times further than it does for the pixel-domain methods - and how
            # far it needs to reach grows with the image. See _set_kernel_range.
            self._set_kernel_range(KERNEL_SIZE_MAX_DCT)
            if self.current_kernel_mode != "dct":
                self.slider_smooth.setValue(KERNEL_SIZE_DEFAULT_DCT)
            self.current_kernel_mode = "dct"
        elif self.rb_gfg.isChecked():
            self._set_kernel_range(KERNEL_SIZE_MAX)
            # GFG-FGF uses the initial mean/blur kernel controlled by the same slider
            self.smooth_widget.setEnabled(True)
            if self.current_kernel_mode != "gfg":
                self.slider_smooth.setValue(KERNEL_SIZE_DEFAULT_GFG)
            self.current_kernel_mode = "gfg"
        elif self.rb_dmap_max.isChecked() or self.rb_dmap_avg.isChecked():
            # Both depth-map modes pool the focus measure over the same slider-
            # controlled window; 9 px is the method default.
            self.smooth_widget.setEnabled(True)
            self._set_kernel_range(KERNEL_SIZE_MAX)
            if self.current_kernel_mode != "dmap":
                self.slider_smooth.setValue(KERNEL_SIZE_DEFAULT_DMAP)
            self.current_kernel_mode = "dmap"
        elif self.rb_pyramid.isChecked():
            # The pyramid pools its band energy over the same slider-controlled
            # window, at every level of the decomposition; 5 px is small because
            # each level doubles what that window covers.
            self.smooth_widget.setEnabled(True)
            self._set_kernel_range(KERNEL_SIZE_MAX)
            if self.current_kernel_mode != "pyramid":
                self.slider_smooth.setValue(KERNEL_SIZE_DEFAULT_PYRAMID)
            self.current_kernel_mode = "pyramid"
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

    def refresh_recent_menus(self) -> None:
        """Rebuild the File > Recent Folders / Recent Videos submenus."""
        menus = (
            (getattr(self, 'menu_recent_folders', None),
             self.settings_manager.recent_folders,
             self.source_manager.load_image_stack,
             self.settings_manager.clear_recent_folders),
            (getattr(self, 'menu_recent_videos', None),
             self.settings_manager.recent_videos,
             self.source_manager.load_video_stack,
             self.settings_manager.clear_recent_videos),
        )

        for menu, paths, loader, clear in menus:
            if menu is None:
                continue
            menu.clear()

            if not paths:
                empty_action = QAction(trans.t('action_recent_empty'), self)
                empty_action.setEnabled(False)
                menu.addAction(empty_action)
                continue

            for path in paths:
                action = QAction(path, self)
                action.setToolTip(path)
                # Entries that have since been moved or deleted stay visible but inert.
                if os.path.exists(path):
                    action.triggered.connect(lambda _checked=False, p=path, fn=loader: fn(p))
                else:
                    action.setEnabled(False)
                menu.addAction(action)

            menu.addSeparator()
            clear_action = QAction(trans.t('action_recent_clear'), self)
            clear_action.triggered.connect(lambda _checked=False, fn=clear: fn())
            menu.addAction(clear_action)
    
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
            pixmap = cv2_to_pixmap(image)
            
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

    def set_gpu_loading_enabled(self, enabled: bool):
        """Toggle GPU image loading (nvJPEG decode and RAW develop).

        Syncs the window attribute, the gpu_decode module the loader actually
        reads, and the checkable menu action, so the three never disagree.
        """
        from core import gpu_decode
        enabled = bool(enabled)
        self.gpu_loading_enabled = enabled
        gpu_decode.set_enabled(enabled)
        action = getattr(self, 'ui_objs', {}).get('action_gpu_loading')
        if action is not None and action.isChecked() != enabled:
            action.setChecked(enabled)

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

    # --- Source stack panel ---

    def toggle_source_panel(self) -> None:
        """Flip the source stack between shown and hidden (output double-click)."""
        if not hasattr(self, 'action_show_source_stack'):
            return
        self.action_show_source_stack.setChecked(
            not self.action_show_source_stack.isChecked())

    def set_source_panel_visible(self, visible: bool) -> None:
        """Show or hide the left source stack panel.

        Hiding it hands the whole view area to the output. The splitter sizes
        are remembered so restoring does not snap back to a 50/50 split.
        """
        if not hasattr(self, 'source_panel_widget'):
            return
        if not visible:
            sizes = self.view_splitter.sizes()
            if sizes and sizes[0] > 0:
                self._view_splitter_sizes = sizes
            self.source_panel_widget.setVisible(False)
        else:
            self.source_panel_widget.setVisible(True)
            saved = getattr(self, '_view_splitter_sizes', None)
            if saved:
                self.view_splitter.setSizes(saved)

        # The output pixmap is scaled to the label, so re-render once the
        # layout has actually resized the labels.
        QTimer.singleShot(0, self._refresh_image_views)

    def _refresh_image_views(self) -> None:
        """Re-scale both previews to their current label size."""
        if hasattr(self, 'source_manager'):
            self.source_manager.refresh_current_source_view()
        if hasattr(self, 'output_manager'):
            self.output_manager.refresh_current_result_view()

    # --- Language ---
    
    def set_language(self, lang_code: str) -> None:
        """Switch application language."""
        trans.set_language(lang_code)

    def apply_output_contrast(self, image):
        """Apply the current contrast setting to a fused output image.

        A no-op (returns the input unchanged) when contrast is off, so callers
        can route every fused output through it unconditionally. Only fused
        results go through here - registered and input stacks are saved raw.
        """
        return contrast.apply_contrast(
            image, self.contrast_method, self.contrast_strength / 100.0)

    def refresh_result_display(self) -> None:
        """Re-show whatever fused result is on screen, picking up contrast.

        The contrast controls apply post-fusion, so moving the slider only has
        to re-run this cheap pixel step against the pristine stored result - no
        re-rendering.
        """
        displayed = getattr(self, "_displayed_fusion_result", None)
        if displayed is not None:
            self.output_manager.display_specific_fusion_result(displayed)

    def handle_contrast_change(self) -> None:
        """Read the contrast controls back onto the window and refresh preview."""
        method_data = self.combo_contrast.currentData()
        self.contrast_method = method_data or contrast.METHOD_OFF
        self.contrast_strength = self.slider_contrast.value()
        self.lbl_contrast_value.setText(f"{self.contrast_strength}%")
        # The strength slider is meaningless while the method is Off.
        self.slider_contrast.setEnabled(self.contrast_method != contrast.METHOD_OFF)
        self.refresh_result_display()

    def set_bit_depth_mode(self, mode: str) -> None:
        """Switch the processing bit depth.

        The mode governs how frames are *decoded*, so it takes effect on the
        next load. Reloading is left to the user rather than done automatically:
        a stack can be large and slow to read, and a mode change mid-session is
        often made before opening the next stack rather than to redo the current
        one.
        """
        if mode == bitdepth.get_mode():
            return
        bitdepth.set_mode(mode)
        self.bit_depth_mode = mode
        print(f"[Depth] Mode set to '{mode}'", flush=True)

        if self.raw_images:
            show_message_box(
                self,
                trans.t("msg_depth_mode_changed_title"),
                trans.t("msg_depth_mode_changed_text").format(
                    mode=trans.t(f"depth_mode_{mode}")),
                trans.t("msg_depth_mode_changed_info"),
                QMessageBox.Icon.Information,
            )

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
        
        # Recent submenus carry translated placeholder / clear entries
        self.refresh_recent_menus()

        # Update Right Panel
        c = self.right_panel_components
        c.method_group.setTitle(trans.t('group_fusion'))
        c.rb_a.setText(trans.t('radio_guided_filter'))
        c.rb_b.setText(trans.t('radio_dct'))
        c.rb_c.setText(trans.t('radio_dtcwt'))
        c.rb_gfg.setText(trans.t('radio_gfg'))
        c.rb_pyramid.setText(trans.t('radio_pyramid'))
        c.rb_dmap_max.setText(trans.t('radio_depthmap_max'))
        c.rb_dmap_avg.setText(trans.t('radio_depthmap_avg'))
        c.rb_d.setText(trans.t('radio_stackmff'))
        c.cb_ifcnn.setText(trans.t('check_ifcnn_refine'))
        # Availability tooltips carry translated text, so refresh them here too
        self._configure_fusion_method_availability()

        c.registration_group.setTitle(trans.t('group_registration'))
        c.cb_align_scale.setText(trans.t('check_align_scale'))
        c.cb_align_ecc.setText(trans.t('check_align_ecc'))
        c.cb_align_homography.setText(trans.t('check_align_homography'))
        
        c.lbl_kernel.setText(trans.t('label_kernel'))
        c.lbl_halo.setText(trans.t('label_halo'))
        c.lbl_dct_block.setText(trans.t('label_dct_block'))
        c.lbl_dct_plateau.setText(trans.t('label_dct_plateau'))
        c.cb_dct_blend.setText(trans.t('label_dct_blend'))
        for i, (key, _value) in enumerate(DCT_PLATEAU_PRESETS):
            c.combo_dct_plateau.setItemText(i, trans.t(f'dct_plateau_{key}'))

        c.lbl_pyr_levels.setText(trans.t('label_pyr_levels'))
        c.lbl_pyr_selectivity.setText(trans.t('label_pyr_selectivity'))
        c.lbl_pyr_coherence.setText(trans.t('label_pyr_coherence'))
        c.lbl_pyr_base.setText(trans.t('label_pyr_base'))
        c.cb_pyr_noise_gate.setText(trans.t('label_pyr_noise_gate'))
        c.cb_pyr_envelope.setText(trans.t('label_pyr_envelope'))
        for i, depth in enumerate(PYRAMID_LEVELS):
            if depth == 0:
                c.combo_pyr_levels.setItemText(i, trans.t('pyr_levels_auto'))
        for combo, presets, prefix in (
                (c.combo_pyr_selectivity, PYRAMID_SELECTIVITY_PRESETS, 'label_pyr_selectivity'),
                (c.combo_pyr_coherence, PYRAMID_COHERENCE_PRESETS, 'label_pyr_coherence'),
                (c.combo_pyr_base, PYRAMID_BASE_PRESETS, 'label_pyr_base')):
            for i, (key, _value) in enumerate(presets):
                combo.setItemText(i, trans.t(f'{prefix}_{key}'))

        self.handle_halo_slider_change(self.slider_halo.value())
        c.btn_reset.setText(trans.t('btn_reset'))
        c.btn_render.setText(trans.t('btn_render'))
        c.btn_stop.setText(trans.t('btn_stop'))
        
        # ROI Button
        if hasattr(self, 'btn_preview_roi'):
            self.btn_preview_roi.setText(trans.t('btn_roi'))

        # Lists labels
        self.source_manager.update_source_images_count()

        c.btn_select_all.setText(trans.t('btn_select_all'))
        c.btn_select_all.setToolTip(trans.t('tip_select_all'))
        c.btn_select_invert.setText(trans.t('btn_select_invert'))
        c.btn_select_invert.setToolTip(trans.t('tip_select_invert'))
        c.btn_select_none.setText(trans.t('btn_select_none'))
        c.btn_select_none.setToolTip(trans.t('tip_select_none'))
        c.btn_select_nth.setText(trans.t('btn_select_nth'))
        c.btn_select_nth.setToolTip(trans.t('tip_select_nth'))
        c.spin_select_nth.setToolTip(trans.t('tip_select_nth'))

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