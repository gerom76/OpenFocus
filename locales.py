
import datetime
from PyQt6.QtCore import QObject, pyqtSignal

class TranslationManager(QObject):
    languageChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_lang = 'en'
        
        # Auto-detect language based on timezone (UTC+8 -> China -> zh)
        try:
            offset = datetime.datetime.now().astimezone().utcoffset()
            if offset is not None and int(offset.total_seconds()) == 28800:
                self.current_lang = 'zh'
        except Exception:
            pass

        self.translations = {
            'en': {
                # Menu File
                'menu_file': 'File',
                'action_open_folder': 'Open Folder',
                'action_open_video': 'Open Video',
                'menu_recent_folders': 'Recent Folders',
                'menu_recent_videos': 'Recent Videos',
                'action_recent_empty': '(No recent items)',
                'action_recent_clear': 'Clear List',
                'action_save': 'Save',
                'menu_save_stack': 'Save Stack',
                'menu_registered_stack': 'Registered Stack',
                'menu_input_stack': 'Input Stack',
                'action_save_folder': 'Save as Folder',
                'action_save_gif': 'Save as GIF',
                'action_save_settings': 'Save All Settings',
                'action_clear_stack': 'Clear Stack',
                'action_exit': 'Exit',
                
                # Menu Edit
                'menu_edit': 'Edit',
                'menu_rotate': 'Rotate',
                'action_rotate_90_cw': '90° Clockwise',
                'action_rotate_90_ccw': '90° Counter-Clockwise',
                'action_rotate_180': '180°',
                'menu_flip': 'Flip',
                'action_flip_h': 'Horizontal Flip',
                'action_flip_v': 'Vertical Flip',
                'action_resize': 'Resize',
                'action_add_label': 'Add Label',
                'menu_del_label': 'Delete Label',
                'action_del_reg_label': 'Delete Registered Stack Labels',
                'action_del_input_label': 'Delete Input Stack Labels',
                
                # Menu Tools
                'menu_tools': 'Tools',
                'action_batch_process': 'Batch Processing',
                
                # Menu Settings
                'menu_settings': 'Settings',
                'menu_language': 'Language',
                'action_lang_en': 'English',
                'action_lang_zh': 'Chinese (Simplified)',
                'action_thread_settings': 'Thread Settings',
                'action_reg_settings': 'Registration Settings',
                'action_tile_settings': 'Tile Settings',
                'action_dng_settings': 'DNG Output...',
                'action_stackmffv4_batch_settings': 'StackMFF V4 Batch Size',
                'action_gpu_loading': 'GPU Image Loading',
                'action_force_cpu_fusion': 'Force CPU Fusion (no GPU)',
                'menu_bit_depth': 'Bit Depth',
                'action_depth_auto': 'Auto (follow source)',
                'action_depth_8': '8-bit',
                'action_depth_16': '16-bit',
                'depth_mode_auto': 'Auto (follow source)',
                'depth_mode_8': '8-bit',
                'depth_mode_16': '16-bit',
                
                # Menu View
                'menu_view': 'View',
                'action_show_console': 'Status Output',
                'action_show_source_stack': 'Source Stack',

                # Menu Help
                'menu_help': 'Help',
                'action_env_info': 'Environment Info',
                'action_contact': 'Contact Us',

                # Status console
                'console_title': 'Output',
                'console_clear': 'Clear',
                'console_collapse': 'Collapse',
                'console_expand': 'Expand',
                
                # Right Panel
                'group_fusion': 'Fusion',
                'radio_guided_filter': 'Guided Filter',
                'radio_dct': 'DCT',
                'radio_dtcwt': 'DTCWT',
                'radio_gfg': 'GFG-FGF',
                'radio_pyramid': 'Pyramid',
                'radio_depthmap_max': 'Depth Map (Max)',
                'radio_depthmap_avg': 'Depth Map (Average)',
                'radio_stackmff': 'StackMFF-V4',
                'check_ifcnn_refine': '+ IFCNN Refine',
                'tip_ifcnn_refine': 'Refine the fused image with IFCNN, recovering detail from the '
                                    'source stack where the fusion step left gaps.',
                'group_registration': 'Registration',
                'check_align_scale': 'Scale (focus breathing)',
                'check_align_ecc': 'ECC',
                'check_align_homography': 'Homography',
                'label_kernel': 'Kernel:',
                'label_halo': 'Halo suppression:',
                'halo_off': 'Off',
                'label_depth_smooth': 'Depth coherence:',
                'label_avg_selectivity': 'Selectivity:',
                'label_avg_coherence': 'Weight coherence:',
                # The same slider as above, named for the field it acts on in
                # the hard select: there it filters the depth map itself, guided
                # by the picture, so it follows edges rather than reaching over
                # them. 'Depth coherence' is already the fill dial's name.
                'label_max_coherence': 'Edge coherence:',
                'label_avg_slice': 'Slice coherence:',
                # Slice coherence pools this many frames either side of each
                # one, so the sign is part of the reading.
                'slice_radius_unit': '±{} frames',
                'label_dct_block': 'DCT block:',
                'label_dct_plateau': 'Focus tolerance:',
                'dct_plateau_crisp': 'Crisp',
                'dct_plateau_balanced': 'Balanced',
                'dct_plateau_smooth': 'Smooth',
                'label_dct_blend': 'Blend block seams',
                'label_pyr_levels': 'Pyramid levels:',
                'pyr_levels_auto': 'Auto',
                'label_pyr_selectivity': 'Selectivity:',
                'label_pyr_selectivity_average': 'Average',
                'label_pyr_selectivity_soft': 'Soft',
                'label_pyr_selectivity_balanced': 'Balanced',
                'label_pyr_selectivity_strict': 'Strict',
                'label_pyr_selectivity_winner': 'Winner takes all',
                'label_pyr_coherence': 'Scale coherence:',
                'label_pyr_coherence_off': 'Off',
                'label_pyr_coherence_light': 'Light',
                'label_pyr_coherence_medium': 'Medium',
                'label_pyr_coherence_strong': 'Strong',
                'label_pyr_base': 'Base band:',
                'label_pyr_base_mean': 'Mean',
                'label_pyr_base_gentle': 'Gentle',
                'label_pyr_base_balanced': 'Balanced',
                'label_pyr_base_strong': 'Strong',
                'label_pyr_noise_gate': 'Ignore grain when nothing is sharp',
                'label_pyr_envelope': 'Keep pixels within the source range',
                'label_contrast': 'Contrast:',
                'contrast_off': 'Off',
                'contrast_auto': 'Auto',
                'contrast_clahe': 'Local',
                'btn_reset': 'Reset Default',
                'btn_render': '▶ Start Render',
                'btn_stop': '■ Stop',
                'label_source_images': 'Source Images: {}',
                'label_source_images_sel': 'Source Images: {} ({} selected)',
                'btn_select_all': 'All',
                'btn_select_invert': 'Invert',
                'btn_select_none': 'None',
                'btn_select_nth': 'N-th',
                'tip_select_all': 'Check every source image',
                'tip_select_invert': 'Check the unchecked images and vice versa',
                'menu_mark': 'Mark',
                'menu_unmark': 'Unmark',
                'tip_select_none': 'Uncheck every source image',
                'tip_select_nth': 'Check every N-th image starting with the first',
                'label_output': 'Output: {}',
                'btn_roi': 'ROI',
                'btn_zoom_100': '100%',
                'tip_zoom_100': 'Zoom to 100% (actual size)',
                'btn_zoom_fit': 'Fit',
                'tip_zoom_fit': 'Zoom to fit the panel',
                'btn_render_processing': '⏳ Processing...',
                'btn_render_stopping': '⏳ Stopping...',
                
                # Status Panel
                'status_loaded': 'Loaded: {}',
                'status_gpu': 'GPU: {}',
                'status_res': 'Res: {}',
                'status_ram': 'RAM: {}',
                'status_loaded_fmt': 'Loaded: {} ({:.1f} MB/img)',
                'status_loaded_depth_fmt': 'Loaded: {} ({:.1f} MB/img, {})',
                'status_res_fmt': 'Res: {}x{}',
                'status_ram_fmt': 'RAM: {:.1f} GB / {:.0f} GB',
                
                # Main Window
                'drag_hint': 'Drag images here to add source files\nor use the image menu',
                'drag_roi_hint': 'Drag to select ROI',
                
                # Dialogs / Messages
                'msg_load_failed': 'Load Failed',
                'msg_load_error': 'Load Error',
                'msg_load_stack_failed_text': 'Failed to load image stack.',
                'msg_load_video_failed_text': 'Failed to load video.',
                'msg_load_stack_error_text': 'An error occurred while loading the image stack.',
                'msg_load_video_error_text': 'An error occurred while loading the video.',
                'msg_load_dropped_failed_text': 'Failed to load dropped images.',
                'msg_load_dropped_error_text': 'An error occurred while loading the dropped images.',
                'msg_drop_no_valid_files_text': 'No valid files were dropped',
                'msg_drop_no_supported_images_text': 'No supported image files were found in the dropped selection',
                'msg_update_error_title': 'Update Error',
                'msg_update_file_list_text': 'Failed to update the file list.',
                'msg_display_error_title': 'Display Error',
                'msg_display_source_failed_text': 'Failed to display the source image.',
                'msg_display_fusion_failed_text': 'Failed to display the fusion result.',
                'msg_display_registration_failed_text': 'Failed to display the registration result.',
                'msg_no_images_title': 'No Images',
                'msg_no_images_rotate': 'No images to rotate.',
                'msg_no_images_flip': 'No images to flip.',
                'msg_no_images_resize': 'No images to resize.',
                'msg_resize_success_text': 'Images resized to {percent}%. Existing outputs preserved.',
                'msg_resize_success_px_text': 'Images resized to {width}x{height} px. Existing outputs preserved.',
                'msg_resize_error_title': 'Resize Error',
                'msg_resize_error_text': 'An error occurred while resizing images.',
                'msg_reload_error_title': 'Reload Error',
                'msg_reload_error_text': 'An error occurred while reloading the image stack.',
                'msg_render_need_images_text': 'Please load at least 2 images before rendering.',
                'msg_render_need_selected_text': 'Please tick at least 2 source images before rendering.',
                'msg_stackmff_unavailable_title': 'StackMFF-V4 Unavailable',
                'msg_stackmff_unavailable_text': 'StackMFF-V4 requires torch + torchvision. Please install them or choose another fusion method.',
                'msg_ifcnn_unavailable_title': 'IFCNN Unavailable',
                'msg_ifcnn_unavailable_text': 'IFCNN refinement requires torch and the IFCNN weights file at:\n{path}',
                'msg_ifcnn_needs_fusion_title': 'IFCNN Needs a Fusion Method',
                'msg_ifcnn_needs_fusion_text': 'IFCNN refines a fused image. Please select a fusion method, or uncheck IFCNN Refine.',
                'msg_save_failed_title': 'Save Failed',
                'msg_save_failed_text': 'Failed to save the image.',
                'msg_save_failed_info_write': 'Unable to write image to the specified file path.',
                'msg_save_failed_info_opencv': 'OpenCV Error: {error}\n\nPlease check the file extension and ensure it is supported.',
                'msg_save_failed_info_unexpected': 'Unexpected error: {error}',
                'msg_image_saved_text': 'Image saved successfully!',
                'msg_image_saved_info': 'Image saved to:\n{path}',
                'msg_no_valid_image_text': 'No valid image to save.',
                'msg_no_valid_image_info': 'The selected item does not contain a valid image.',
                'msg_no_result_title': 'No Result',
                'msg_no_result_text': 'Please render the result first (registration or fusion).',
                'msg_no_stack_title': 'No Stack',
                'msg_no_stack_text': 'Please perform registration first to save the stack.',
                'msg_stack_saved_text': 'Stack saved successfully!',
                'msg_stack_saved_info': 'Successfully saved {saved}/{total} images to:\n{folder}',
                'msg_save_stack_failed_text': 'Failed to save stack:',
                'msg_save_stack_opencv_info': 'OpenCV Error: {error}\n\nPlease check the file path and permissions.',
                'msg_save_stack_unexpected_info': 'Unexpected error: {error}',
                'msg_no_registered_images_title': 'No Registered Images',
                'msg_no_registered_images_text': 'Please perform registration first.',
                'msg_no_input_images_title': 'No Input Images',
                'msg_no_input_images_text': 'Please load images first.',
                'msg_no_images_save_stack_text': 'Please load images first to save the stack.',
                'msg_size_mismatch_stack_text': 'Not all images have the same dimensions.',
                'msg_size_mismatch_open_info': 'Do you want to continue opening the stack?',
                'msg_load_images_failed_text': 'Failed to load images: {message}',
                'msg_no_folders_title': 'No Folders',
                'msg_no_folders_text': 'Please add at least one folder to process.',
                'msg_no_folder_title': 'No Folder',
                'msg_no_folder_text': 'Please select a folder first.',
                'msg_no_valid_files_title': 'No Valid Files',
                'msg_no_valid_files_text': 'No supported image or video files found.',
                'msg_gif_saving_text': 'Saving GIF animation...',
                'msg_gif_processing_title': 'Processing',
                'msg_gif_saved_text': 'GIF animation saved successfully!',
                'msg_gif_saved_info': '{message}\nFrame duration: {duration}ms',
                'msg_gif_save_failed_text': 'Failed to save as GIF animation:',
                'msg_gif_save_failed_info': 'Error: {message}',
                'msg_processed_stack_saved_text': 'Processed input stack saved successfully!',
                'msg_processed_stack_saved_info': 'Successfully saved {saved}/{total} images to:\n{folder}',
                'msg_size_mismatch_title': 'Image Size Mismatch',
                'msg_size_mismatch_text': 'New images have different dimensions from the existing stack.',
                'msg_size_mismatch_info': 'Do you want to append them anyway?',
                'btn_continue': 'Continue',
                'btn_cancel_generic': 'Cancel',

                # ROI Dialog
                'dialog_roi_title': 'ROI Processing Options',
                'dialog_roi_msg': 'You have selected a Region of Interest (ROI).\nHow would you like to process the fusion?',
                'dialog_roi_group_output': 'Output Mode',
                'dialog_roi_opt_crop': 'Fuse ROI Only (Output Cropped Image)',
                'dialog_roi_opt_crop_tooltip': 'Result will be a small image containing only the selected region.',
                'dialog_roi_opt_paste': 'Fuse and Overlay on Base Frame',
                'dialog_roi_opt_paste_tooltip': 'Result will be the full base image with the fused ROI pasted on top.',
                'dialog_roi_group_base': 'Base Frame Selection',
                'dialog_roi_lbl_base': 'Select Base Frame:',
                
                # ROI Alignment (New)
                'roi_need_images': 'Please load at least 2 images before using ROI mode.',
                'roi_aligning': 'Aligning...',
                'roi_ready_title': 'ROI Mode Ready',
                'roi_ready_msg': 'Images have been aligned using ECC method.',
                'roi_ready_detail': 'Alignment completed in {:.2f}s.\n\nYou can now draw a ROI on the right panel.\nPress Render to fuse only the selected region.',
                'roi_align_failed': 'ROI alignment failed.',
                'roi_aligned': 'Aligned',
                'roi_display_failed': 'Failed to display aligned image.',
                'result_hint': 'Result will appear here',
                
                # Completion Dialog
                'dialog_completed_title': 'Processing Completed',
                'dialog_completed_msg': 'Processing completed successfully!',
                
                'info_align_method': 'Alignment Method: {}',
                'info_align_time': 'Alignment Time: {:.2f}s',
                'info_fusion_method': 'Fusion Method: {}',
                'info_fusion_method_refined': '{} + IFCNN',
                'info_fusion_time': 'Fusion Time: {:.2f}s',
                'info_proc_unit': 'Processing Unit: {}',
                'info_total_time': 'Total Time: {:.2f}s',
                'info_align_none_cached': 'Alignment Method: None (using cached results)',
                'info_align_none': 'Alignment Method: None',
                'info_fusion_none': 'Fusion Method: None',
                'val_enabled': 'Enabled',
                'val_none': 'None',
                'val_using_cached': 'using cached results',
                'btn_ok': 'OK',
                'btn_cancel': 'Cancel',
                'msg_success': 'Success',
                'msg_error': 'Error',
                'msg_proc_error': 'An error occurred during processing:',
                'msg_warning': 'Warning',
                
                # Help Dialogs
                'dialog_env_title': 'Environment Information',
                'env_subtitle': 'OpenFocus Environment Dependencies',
                'env_python': 'Python Version',
                'env_installed': 'Installed: Version {}',
                'env_not_installed': 'Not installed',
                'env_cuda_avail': 'CUDA available: {}',
                'env_cuda_ver': 'CUDA version: {}',
                'env_mps_avail': 'MPS available (Apple Silicon)',
                'env_gpu_accel': 'StackMFF-V4: GPU acceleration available',
                'env_no_gpu': 'Warning: No GPU acceleration (CUDA/MPS)',
                'env_cpu_mode': 'StackMFF-V4: Available (CPU mode - slower)',
                'env_fusion_cpu_only': 'Force CPU Fusion is on in Settings: fusion will not use the GPU',
                'env_stackmff_unavailable': 'StackMFF-V4 fusion not available',
                'env_dtcwt_unavailable': 'Not installed (DTCWT fusion unavailable)',
                'env_summary': 'Summary',
                'env_core_dep': 'Core Dependencies:',
                'env_core_desc': '- OpenCV, NumPy, PyQt6: Required for basic functionality',
                'env_gpu_opt': 'GPU Acceleration (Optional):',
                'env_gpu_desc': '- PyTorch: Enables StackMFF-V4 (CPU fallback available but slower)',
                'env_fusion_alg': 'Fusion Algorithms:',
                'env_fusion_desc': '- DTCWT library: Required for DTCWT fusion',
                
                'dialog_contact_title': 'Contact Us',
                'contact_info_title': 'Contact Information',
                'contact_email': 'Email',
                'contact_institution': 'Institution',
                'contact_zju': 'Zhejiang University',
                'contact_github': 'GitHub',
                'contact_welcome': 'We warmly welcome contributors who would like to add new fusion methods and help OpenFocus grow.',
                
                'btn_close': 'Close',

                # Batch Processing Dialog
                'batch_title': 'Batch Processing',
                'batch_import_mode': 'Import Mode',
                'batch_mode_multi': 'Multiple Folders (one stack per folder)',
                'batch_mode_single': 'Single Folder (auto-split into multiple stacks)',
                'batch_stack_folders': 'Image Stack Folders',
                'batch_path_placeholder': 'Type a path and press Enter to refresh',
                'batch_btn_add': 'Add Folders',
                'batch_btn_remove': 'Remove Selected',
                'batch_single_split_settings': 'Single Folder Split Settings',
                'batch_folder_none': 'Folder: (none)',
                'batch_split_method': 'Split Method:',
                'batch_split_fixed': 'Fixed Count',
                'batch_split_time': 'Time Threshold',
                'batch_images_per_stack': 'Images per Stack:',
                'batch_time_threshold': 'Time Threshold:',
                'batch_unit_images': 'images',
                'batch_unit_seconds': 'seconds',
                'batch_preview_default': 'Preview: 0 images → 0 stacks',
                'batch_preview_fmt_count': 'Preview: {} images → {} stacks ({} images each)',
                'batch_preview_fmt_time': 'Preview: {} images → {} stacks (threshold: {}s)',
                'batch_preview_no_folder': 'Preview: No folder selected',
                'batch_btn_select_split': 'Select Folder and Split',
                'batch_output_format': 'Output Format',
                'batch_format_label': 'Format:',
                'batch_quality_label': 'Quality:',
                'batch_output_location': 'Output Location',
                'batch_out_subfolder': 'Create subfolder in source folder',
                'batch_subfolder_name': 'Subfolder Name:',
                'batch_out_same': 'Same as source folder',
                'batch_out_custom': 'Specify output folder',
                'batch_btn_browse': 'Browse...',
                'batch_save_aligned': 'Save Aligned Image Stack',
                'batch_proc_options': 'Processing Options',
                'batch_lbl_fusion': 'Fusion Method: {}',
                'batch_lbl_reg': 'Registration Methods: {}',
                'batch_lbl_kernel': 'Kernel Size: {}',
                'batch_btn_start': 'Start Batch Processing',
                'batch_btn_cancel': 'Cancel',
                
                # Help Dialogs
                'help_render_title': 'Fusion Help',
                'help_registration_title': 'Registration Help',
                'help_tile_title': 'Tile Settings Help',

                'dialog_tile_title': 'Tile Settings',
                'dialog_tile_group': 'Tile Options',
                'dialog_tile_enabled_label': 'Enable Tiling:',
                'dialog_tile_enabled': 'Enabled',
                'dialog_tile_disabled': 'Disabled',
                'dialog_tile_block_size': 'Tile Block Size:',
                'dialog_tile_overlap': 'Tile Overlap:',
                'dialog_tile_threshold': 'Tile Threshold:',
                'dialog_tile_help_title': 'Tile Settings Help',
                'dialog_tile_help_text': '''<h3>Tile Settings Help</h3>
        <p>tile_enabled: Enable or disable tiled processing. When enabled, large images
        will be processed in smaller blocks to reduce memory usage.</p>

        <p>tile_block_size: Size (in pixels) of each square tile block. Typical values
        are 512–2048 depending on memory and speed tradeoffs.</p>

        <p>tile_overlap: Overlap (in pixels) between adjacent tiles used to avoid seams
        when combining results. A positive overlap helps smooth boundaries.</p>

        <p>tile_threshold: If the image's longest side is larger than this threshold,
        tiled processing will be considered. Smaller images are processed as a whole.</p>''',
                'dialog_reg_title': 'Registration Settings',
                'dialog_reg_group': 'Registration Options',
                'dialog_reg_downscale': 'Downscale Width:',
                'dialog_reg_help_title': 'Registration Settings Help',
                'dialog_reg_help_text': '''<h3>Downscale Width</h3>
        <p>downscale_width controls the pre-processing (downsampling) width when extracting
        registration features. Smaller values can speed up feature detection and reduce
        memory usage, but may sacrifice some geometric accuracy.</p>

        <p>The value is used as set, at any image size. Up to version 1.30.5 it was
        silently replaced by 1024 whenever the longer side of a frame reached 2048px,
        which is every file from a modern camera, so the setting did nothing on real
        input.</p>

        <p><code>1024</code>, the default, is a good balance. Setting it to the full
        width of your frames measures the alignment at full resolution, which was worth
        1.3x to 1.8x better geometric accuracy on a 2560px test stack for roughly twice
        the registration time. Lowering it speeds registration up and reduces memory
        use, at the cost of accuracy. Values above the frame width behave the same as
        the frame width - the frames are never upsampled.</p>

        <h3>Parallel ECC Computation</h3>
        <p>Computes the ECC transform between each pair of adjacent frames concurrently
        across CPU cores. Results are identical to sequential computation, only faster
        (roughly 2x on typical stacks). Disable only for troubleshooting.</p>

        <h3>Reference Frame</h3>
        <p>The frame held fixed while every other frame is aligned onto it. Alignment
        chains each frame to its neighbour, so error accumulates with distance from
        the reference.</p>
        <p><b>First</b> (default) references frame 0, matching earlier behaviour.
        <b>Middle</b> references the centre frame, halving the longest chain and
        spreading any residual drift symmetrically across the stack - useful for
        long stacks where the far end drifts. <b>Last</b> anchors on the final
        frame, mirroring First from the other end.</p>''',
                'dialog_reg_parallel_ecc': 'Parallel ECC computation',
                'dialog_reg_reference': 'Reference Frame:',
                'dialog_reg_reference_first': 'First',
                'dialog_reg_reference_middle': 'Middle',
                'dialog_reg_reference_last': 'Last',
                'dialog_thread_title': 'Thread Settings',
                'dialog_thread_group': 'Thread Count',
                'dialog_thread_label': 'Thread Count:',
                'dialog_thread_help_title': 'Thread Settings Help',
                'dialog_thread_help_text': '''<h3>Thread Count Settings</h3>
        <p>This setting controls the number of processing threads used by algorithms
        that support multi-threading. Typically set to match CPU core count (2-16).</p>

        <h4>Current multi-threading support:</h4>
        <ul>
        <li>GFG-FGF: Supports user-controlled thread count (default max: 8)</li>
        <li>Guided Filter Fusion (GFF): Supports user-controlled thread count (default max: 4)</li>
        <li>Image Registration: Feature extraction/transformation uses specified thread count</li>
        <li>DCT, DTCWT, StackMFF-V4: Currently do not use this setting (no thread control)</li>
        </ul>

        <p>Algorithms that don't use this value will safely ignore it. For best performance,
        avoid setting thread count higher than physical core count.</p>

        <p>Note: Installing <code>opencv-contrib-python</code> can accelerate certain
        operations (e.g., guided filtering).</p>''',

                # DNG output settings dialog
                'dialog_dng_title': 'DNG Output Settings',
                'dialog_dng_group': 'DNG Output',
                'dialog_dng_compression_label': 'Compression:',
                'dialog_dng_quality_label': 'Lossy quality:',
                'dialog_dng_fast_load': 'Embed fast-load preview',
                'dialog_dng_camera_space': "Write in the source camera's colour space",
                'dng_camera_space_hint': "Stores the result as the camera's own raw "
                                         'data, so camera profiles apply correctly '
                                         'instead of shifting the colour. The file then '
                                         'renders like the raw in your converter rather '
                                         'than like the fused result. Needs the source '
                                         'RAW; other sources fall back to sRGB.',
                'dng_compression_none': 'Uncompressed',
                'dng_compression_lossless': 'Lossless (JPEG)',
                'dng_compression_lossy': 'Lossy (JPEG)',
                'dng_compression_hint_none': 'Largest files, fastest to save. '
                                             'The strips are the pixels.',
                'dng_compression_hint_lossless': 'Identical pixels, roughly half the '
                                                 'size at 16-bit and a third at 8-bit. '
                                                 'Slower to save.',
                'dng_compression_hint_lossy': 'Smallest files, but 8-bit only - a '
                                              '16-bit result is narrowed on the way '
                                              'out. For proxies, not masters.',
                'dialog_dng_help_title': 'DNG Output Settings Help',
                'dialog_dng_help_text': '''<h3>DNG Output Settings</h3>
        <p>These settings apply whenever a result is saved as <code>.dng</code>, whether
        from Save, a folder export or batch processing.</p>

        <h4>Compression</h4>
        <ul>
        <li><b>Uncompressed</b> - the default. No encode cost, and the file holds the
        pixels exactly as the pipeline produced them.</li>
        <li><b>Lossless (JPEG)</b> - lossless Huffman JPEG, the only lossless codec DNG
        permits for 16-bit linear data. The pixels come back identical; expect roughly
        50-55% of the uncompressed size at 16-bit and 35-40% at 8-bit. Saving takes a
        few seconds on a large frame.</li>
        <li><b>Lossy (JPEG)</b> - baseline DCT JPEG. DNG allows this only for 8-bit
        data, so a 16-bit result is narrowed to 8-bit on the way out and the loss is
        permanent. Useful for proxies and for sharing, not for a master.</li>
        </ul>

        <p><b>Note:</b> the lossless mode is only offered when the <code>imagecodecs</code>
        package is installed. OpenFocus can write it without that package, but could not
        read the file back, and a saved result has to be usable as the input to the next
        stack.</p>

        <h4>Lossy quality</h4>
        <p>The JPEG quality used by the lossy mode, from 1 to 100. The default of 92 is
        high enough that the compression is not what limits the image. The setting has no
        effect in the other two modes.</p>

        <h4>Embed fast-load preview</h4>
        <p>Stores a half-resolution JPEG rendering of the result inside the DNG. Without
        it, anything that wants a thumbnail - a file browser, Lightroom, a raw converter -
        has to decode the full-resolution linear raw first, which for a fused stack is the
        slowest thing in the file. The preview costs a few hundred kilobytes.</p>

        <p>This is the DNG specification's own preview mechanism. It is not Adobe's
        proprietary "Fast Load Data", which is a Camera Raw cache that only Adobe's
        converter can produce, but it serves the same purpose and works in more readers.</p>''',

                # StackMFF V4 Batch settings dialog
                'dialog_stackmffv4_batch_title': 'StackMFF V4 Batch Settings',
                'dialog_stackmffv4_batch_group': 'Batch Processing',
                'dialog_stackmffv4_batch_label': 'Batch Size:',
                'dialog_stackmffv4_batch_help_title': 'StackMFF V4 Batch Settings Help',
                'dialog_stackmffv4_batch_help_text': '''<h3>StackMFF V4 Batch Settings</h3>
        <p>This setting controls the batch size when processing tiles with the StackMFF V4
        neural network fusion algorithm. Batch processing allows multiple tiles to be
        processed in parallel on the GPU, improving efficiency.</p>

        <h4>Batch Size:</h4>
        <ul>
        <li><b>Default: 2</b> - Balanced between memory usage and speed</li>
        <li><b>Higher values (4-8)</b> - Faster processing but requires more GPU memory</li>
        <li><b>Lower values (1)</b> - Lower memory usage, useful for limited GPU memory</li>
        </ul>

        <p><b>Note:</b> If you encounter out-of-memory errors on GPU, try reducing the batch size.
        This setting only affects StackMFF V4 when processing large images with tiling enabled.</p>

        <p>CPU mode is not significantly affected by this setting but will use it for
        consistency.</p>''',

                'dialog_duration_title': 'GIF Duration Settings',
                'dialog_duration_group': 'Frame Duration',
                'dialog_duration_label': 'Duration (ms):',

                'dialog_export_format_title': 'Save Stack Format',
                'dialog_export_format_label': 'Format:',

                # Add Label Dialog
                'add_label_title': 'Add Label Configuration',
                'label_target_stack': 'Target Stack:',
                'label_target_input': 'Input Image Stack',
                'label_target_registered': 'Registered Image Stack',
                'label_format': 'Format String:',
                'label_start_val': 'Starting Value:',
                'label_interval': 'Interval:',
                'label_x': 'X Location:',
                'label_y': 'Y Location:',
                'label_font_size': 'Font Size:',
                'label_font_family': 'Font Family:',
                'label_custom_text': 'Custom Text:',
                'label_range': 'Range:',
                'label_transparent_bg': 'Transparent Background',
                'label_bg_color': 'Background Color:',
                'label_choose_bg': 'Choose Background Color',
                'label_font_color': 'Font Color:',
                'label_choose_font_color': 'Choose Font Color',
                
                'btn_ok': 'OK',
                'btn_cancel': 'Cancel',

                'msg_config_saved_title': 'Configuration Saved',
                'msg_config_saved_text': 'Label configuration saved successfully!',
                'msg_config_saved_info': 'The labels are now visible on the selected stack and will be included when saving.',

                'msg_settings_saved_title': 'Settings Saved',
                'msg_settings_saved_text': 'All settings saved successfully!',
                'msg_settings_saved_info': 'Settings saved to:\n{path}',
                'msg_depth_mode_changed_title': 'Bit Depth Changed',
                'msg_depth_mode_changed_text': 'Processing bit depth set to {mode}.',
                'msg_depth_mode_changed_info': 'The new depth applies when images are decoded, so reload the current stack for it to take effect.',
                'msg_settings_save_failed_text': 'Failed to save settings.',
                
                'msg_no_reg_labels': 'No Labels',
                'msg_reg_labels_disabled': 'Registered stack labels are not enabled.',
                'msg_reg_labels_removed': 'Labels removed from Registered Stack.',
                'msg_no_input_labels': 'No Labels',
                'msg_input_labels_disabled': 'Input stack labels are not enabled.',
                'msg_input_labels_removed': 'Labels removed from Input Stack.',
                
                 # Folder Import Dialog
                'import_folder_title': 'Import Folder',
                'import_folder_display': 'Folder: {}',
                'import_choice_label': 'How should this folder be imported?',
                'import_option_single': 'Single Image Stack (load as one stack)',
                'import_option_batch': 'Multiple Image Stacks (open batch processing)',
                
                # Downsample Dialog
                'ds_title': 'Downsample Settings',
                'ds_label': 'Set image loading scale (Downsampling):',
                'ds_mode_percent': 'By scale (%)',
                'ds_mode_long_edge': 'By longer edge (px)',
                'ds_long_edge_label': 'Longer edge:',
                'ds_hint': 'Use lower values for large images to save memory and speed up processing. '
                           'A longer-edge target is applied per image and never enlarges one.',

            },
            'zh': {
                # Menu File
                'menu_file': '文件',
                'action_open_folder': '打开文件夹',
                'action_open_video': '打开视频',
                'menu_recent_folders': '最近的文件夹',
                'menu_recent_videos': '最近的视频',
                'action_recent_empty': '（无最近项目）',
                'action_recent_clear': '清空列表',
                'action_save': '保存',
                'menu_save_stack': '保存堆栈',
                'menu_registered_stack': '已配准堆栈',
                'menu_input_stack': '输入堆栈',
                'action_save_folder': '保存为文件夹',
                'action_save_gif': '保存为 GIF',
                'action_save_settings': '保存所有设置',
                'action_clear_stack': '清空堆栈',
                'action_exit': '退出',
                
                # Menu Edit
                'menu_edit': '编辑',
                'menu_rotate': '旋转',
                'action_rotate_90_cw': '顺时针 90°',
                'action_rotate_90_ccw': '逆时针 90°',
                'action_rotate_180': '180°',
                'menu_flip': '翻转',
                'action_flip_h': '水平翻转',
                'action_flip_v': '垂直翻转',
                'action_resize': '调整大小',
                'action_add_label': '添加标签',
                'menu_del_label': '删除标签',
                'action_del_reg_label': '删除已配准堆栈标签',
                'action_del_input_label': '删除输入堆栈标签',
                
                # Menu Tools
                'menu_tools': '工具',
                'action_batch_process': '批处理',
                
                # Menu Settings
                'menu_settings': '设置',
                'menu_language': '语言',
                'action_lang_en': 'English',
                'action_lang_zh': '简体中文',
                'action_thread_settings': '线程设置',
                'action_reg_settings': '配准设置',
                'action_tile_settings': '分块设置',
                'action_dng_settings': 'DNG 输出...',
                'action_stackmffv4_batch_settings': 'StackMFF V4 批量大小',
                'action_gpu_loading': 'GPU 图像加载',
                'action_force_cpu_fusion': '强制 CPU 融合（不使用 GPU）',
                'menu_bit_depth': '位深度',
                'action_depth_auto': '自动（跟随源文件）',
                'action_depth_8': '8 位',
                'action_depth_16': '16 位',
                'depth_mode_auto': '自动（跟随源文件）',
                'depth_mode_8': '8 位',
                'depth_mode_16': '16 位',
                
                # Menu View
                'menu_view': '视图',
                'action_show_console': '状态输出',
                'action_show_source_stack': '源图像栈',

                # Menu Help
                'menu_help': '帮助',
                'action_env_info': '环境信息',
                'action_contact': '联系我们',

                # Status console
                'console_title': '输出',
                'console_clear': '清空',
                'console_collapse': '折叠',
                'console_expand': '展开',
                
                # Right Panel
                'group_fusion': '融合算法',
                'radio_guided_filter': '引导滤波',
                'radio_dct': '余弦离散变换',
                'radio_dtcwt': '双数复小波变换',
                'radio_gfg': '引导滤波2',
                'radio_pyramid': '拉普拉斯金字塔',
                'radio_depthmap_max': '深度图（最大值）',
                'radio_depthmap_avg': '深度图（加权平均）',
                'radio_stackmff': 'StackMFF-V4',
                'check_ifcnn_refine': '+ IFCNN 精修',
                'tip_ifcnn_refine': '使用 IFCNN 对融合结果进行精修，从原始图像栈中找回融合阶段遗漏的细节。',
                'group_registration': '图像配准',
                'check_align_scale': '缩放校正 (焦点呼吸)',
                'check_align_ecc': 'ECC',
                'check_align_homography': 'Homography',
                'label_kernel': '核大小:',
                'label_halo': '光晕抑制:',
                'halo_off': '关闭',
                'label_depth_smooth': '深度一致性:',
                'label_avg_selectivity': '选择性:',
                'label_avg_coherence': '权重一致性:',
                'label_max_coherence': '边缘一致性:',
                'label_avg_slice': '切片一致性:',
                'slice_radius_unit': '±{} 帧',
                'label_dct_block': 'DCT 分块:',
                'label_dct_plateau': '对焦容差:',
                'dct_plateau_crisp': '锐利',
                'dct_plateau_balanced': '均衡',
                'dct_plateau_smooth': '平滑',
                'label_dct_blend': '融合分块接缝',
                'label_pyr_levels': '金字塔层数:',
                'pyr_levels_auto': '自动',
                'label_pyr_selectivity': '选择强度:',
                'label_pyr_selectivity_average': '平均',
                'label_pyr_selectivity_soft': '柔和',
                'label_pyr_selectivity_balanced': '均衡',
                'label_pyr_selectivity_strict': '严格',
                'label_pyr_selectivity_winner': '胜者独占',
                'label_pyr_coherence': '跨尺度一致性:',
                'label_pyr_coherence_off': '关闭',
                'label_pyr_coherence_light': '轻度',
                'label_pyr_coherence_medium': '中度',
                'label_pyr_coherence_strong': '强',
                'label_pyr_base': '基频带:',
                'label_pyr_base_mean': '平均',
                'label_pyr_base_gentle': '轻微',
                'label_pyr_base_balanced': '均衡',
                'label_pyr_base_strong': '强',
                'label_pyr_noise_gate': '无清晰内容时忽略噪点',
                'label_pyr_envelope': '像素限制在源图范围内',
                'label_contrast': '对比度:',
                'contrast_off': '关闭',
                'contrast_auto': '自动',
                'contrast_clahe': '局部',
                'btn_reset': '重置默认',
                'btn_render': '▶ 开始渲染',
                'btn_stop': '■ 停止',
                'label_source_images': '源图像: {}',
                'label_source_images_sel': '源图像: {} (已选 {})',
                'btn_select_all': '全选',
                'btn_select_invert': '反选',
                'btn_select_none': '全不选',
                'btn_select_nth': '每 N 张',
                'tip_select_all': '勾选所有源图像',
                'tip_select_invert': '勾选未选的图像，取消勾选已选的图像',
                'menu_mark': '勾选',
                'menu_unmark': '取消勾选',
                'tip_select_none': '取消勾选所有源图像',
                'tip_select_nth': '从第一张开始每隔 N 张勾选一次',
                'label_output': '输出: {}',
                'btn_roi': '选择感兴趣区域',
                'btn_zoom_100': '100%',
                'tip_zoom_100': '缩放至 100%（实际大小）',
                'btn_zoom_fit': '适应',
                'tip_zoom_fit': '缩放以适应面板大小',
                'btn_render_processing': '⏳ 处理中...',
                'btn_render_stopping': '⏳ 正在停止...',
                
                # Status Panel
                'status_loaded': '已加载: {}',
                'status_gpu': 'GPU: {}',
                'status_res': '分辨率: {}',
                'status_ram': '内存: {}',
                'status_loaded_fmt': '已加载: {} ({:.1f} MB/张)',
                'status_loaded_depth_fmt': '已加载: {} ({:.1f} MB/张, {})',
                'status_res_fmt': '分辨率: {}x{}',
                'status_ram_fmt': '内存: {:.1f} GB / {:.0f} GB',
                
                # Main Window
                'drag_hint': '拖拽图像到此处添加源文件\n或使用图像菜单',
                'drag_roi_hint': '拖动鼠标选择感兴趣区域 (ROI)',

                # Dialogs / Messages
                'msg_load_failed': '加载失败',
                'msg_load_error': '加载错误',
                'msg_load_stack_failed_text': '加载图像栈失败。',
                'msg_load_video_failed_text': '加载视频失败。',
                'msg_load_stack_error_text': '加载图像栈时发生错误。',
                'msg_load_video_error_text': '加载视频时发生错误。',
                'msg_load_dropped_failed_text': '拖入图像加载失败。',
                'msg_load_dropped_error_text': '加载拖入图像时发生错误。',
                'msg_drop_no_valid_files_text': '未拖入有效文件。',
                'msg_drop_no_supported_images_text': '拖入的文件中没有受支持的图像格式。',
                'msg_update_error_title': '更新错误',
                'msg_update_file_list_text': '更新文件列表失败。',
                'msg_display_error_title': '显示错误',
                'msg_display_source_failed_text': '显示源图像失败。',
                'msg_display_fusion_failed_text': '显示融合结果失败。',
                'msg_display_registration_failed_text': '显示配准结果失败。',
                'msg_no_images_title': '没有图像',
                'msg_no_images_rotate': '没有可旋转的图像。',
                'msg_no_images_flip': '没有可翻转的图像。',
                'msg_no_images_resize': '没有可缩放的图像。',
                'msg_resize_success_text': '图像已缩放至 {percent}%。已保留现有输出。',
                'msg_resize_success_px_text': '图像已缩放至 {width}x{height} 像素。已保留现有输出。',
                'msg_resize_error_title': '缩放错误',
                'msg_resize_error_text': '缩放图像时发生错误。',
                'msg_reload_error_title': '重新加载错误',
                'msg_reload_error_text': '重新加载图像栈时发生错误。',
                'msg_render_need_images_text': '渲染前请至少加载 2 张图像。',
                'msg_render_need_selected_text': '渲染前请至少勾选 2 张源图像。',
                'msg_stackmff_unavailable_title': 'StackMFF-V4 不可用',
                'msg_stackmff_unavailable_text': 'StackMFF-V4 需要 torch + torchvision。请安装或选择其他融合方法。',
                'msg_ifcnn_unavailable_title': 'IFCNN 不可用',
                'msg_ifcnn_unavailable_text': 'IFCNN 精修需要 torch 以及权重文件：\n{path}',
                'msg_ifcnn_needs_fusion_title': 'IFCNN 需要融合方法',
                'msg_ifcnn_needs_fusion_text': 'IFCNN 用于精修融合结果。请选择一种融合方法，或取消勾选 IFCNN 精修。',
                'msg_save_failed_title': '保存失败',
                'msg_save_failed_text': '保存图像失败。',
                'msg_save_failed_info_write': '无法写入指定的文件路径。',
                'msg_save_failed_info_opencv': 'OpenCV 错误: {error}\n\n请检查文件扩展名并确保其受支持。',
                'msg_save_failed_info_unexpected': '未知错误: {error}',
                'msg_image_saved_text': '图像保存成功！',
                'msg_image_saved_info': '图像已保存到:\n{path}',
                'msg_no_valid_image_text': '没有可保存的有效图像。',
                'msg_no_valid_image_info': '所选项不包含有效图像。',
                'msg_no_result_title': '没有结果',
                'msg_no_result_text': '请先渲染结果（配准或融合）。',
                'msg_no_stack_title': '没有图像栈',
                'msg_no_stack_text': '请先完成配准以保存图像栈。',
                'msg_stack_saved_text': '图像栈保存成功！',
                'msg_stack_saved_info': '已成功保存 {saved}/{total} 张图像到:\n{folder}',
                'msg_save_stack_failed_text': '保存图像栈失败：',
                'msg_save_stack_opencv_info': 'OpenCV 错误: {error}\n\n请检查路径与权限。',
                'msg_save_stack_unexpected_info': '未知错误: {error}',
                'msg_no_registered_images_title': '没有已配准图像',
                'msg_no_registered_images_text': '请先进行配准。',
                'msg_no_input_images_title': '没有输入图像',
                'msg_no_input_images_text': '请先加载图像。',
                'msg_no_images_save_stack_text': '请先加载图像以保存图像栈。',
                'msg_size_mismatch_stack_text': '并非所有图像尺寸一致。',
                'msg_size_mismatch_open_info': '仍然要继续打开这组图像栈吗？',
                'msg_load_images_failed_text': '加载图像失败: {message}',
                'msg_no_folders_title': '没有文件夹',
                'msg_no_folders_text': '请至少添加一个文件夹进行处理。',
                'msg_no_folder_title': '未选择文件夹',
                'msg_no_folder_text': '请先选择一个文件夹。',
                'msg_no_valid_files_title': '没有有效文件',
                'msg_no_valid_files_text': '未找到受支持的图像或视频文件。',
                'msg_gif_saving_text': '正在保存 GIF 动画...',
                'msg_gif_processing_title': '处理中',
                'msg_gif_saved_text': 'GIF 动画保存成功！',
                'msg_gif_saved_info': '{message}\n帧时长: {duration}ms',
                'msg_gif_save_failed_text': '保存 GIF 动画失败：',
                'msg_gif_save_failed_info': '错误: {message}',
                'msg_processed_stack_saved_text': '处理后的输入图像栈保存成功！',
                'msg_processed_stack_saved_info': '已成功保存 {saved}/{total} 张图像到:\n{folder}',
                'dialog_export_format_title': '保存图像栈格式',
                'dialog_export_format_label': '格式:',
                'msg_size_mismatch_title': '图像尺寸不一致',
                'msg_size_mismatch_text': '新增图像尺寸与当前栈不一致。',
                'msg_size_mismatch_info': '仍然要追加吗？',
                'btn_continue': '继续',
                'btn_cancel_generic': '取消',

                # ROI Dialog
                'dialog_roi_title': 'ROI 处理选项',
                'dialog_roi_msg': '您已选择了感兴趣区域 (ROI)。\n您希望如何处理融合结果？',
                'dialog_roi_group_output': '输出模式',
                'dialog_roi_opt_crop': '仅融合 ROI (输出裁剪图像)',
                'dialog_roi_opt_crop_tooltip': '结果将是一个仅包含所选区域的小图像。',
                'dialog_roi_opt_paste': '融合并覆盖在底图上',
                'dialog_roi_opt_paste_tooltip': '结果将是完整的底图，其中 ROI 区域被融合结果覆盖。',
                'dialog_roi_group_base': '底图选择',
                'dialog_roi_lbl_base': '选择底图:',
                
                # ROI Alignment (New)
                'roi_need_images': '请先加载至少2张图像才能使用ROI模式。',
                'roi_aligning': '对齐中...',
                'roi_ready_title': 'ROI模式就绪',
                'roi_ready_msg': '图像已使用ECC方法完成对齐。',
                'roi_ready_detail': '对齐耗时 {:.2f}s。\n\n您现在可以在右侧面板上绘制ROI区域。\n点击渲染按钮仅融合所选区域。',
                'roi_align_failed': 'ROI对齐失败。',
                'roi_aligned': '已对齐',
                'roi_display_failed': '显示对齐图像失败。',
                'result_hint': '结果将显示在这里',
                
                # Completion Dialog
                'dialog_completed_title': '处理完成',
                'dialog_completed_msg': '处理成功完成！',
                
                'info_align_method': '配准方法: {}',
                'info_align_time': '配准耗时: {:.2f}s',
                'info_fusion_method': '融合方法: {}',
                'info_fusion_method_refined': '{} + IFCNN',
                'info_fusion_time': '融合耗时: {:.2f}s',
                'info_proc_unit': '处理单元: {}',
                'info_total_time': '总耗时: {:.2f}s',
                'info_align_none_cached': '配准方法: 无 (使用缓存结果)',
                'info_align_none': '配准方法: 无',
                'info_fusion_none': '融合方法: 无',
                'val_enabled': '已启用',
                'val_none': '无',
                'val_using_cached': '使用缓存结果',
                'msg_success': '成功',
                'msg_error': '错误',
                'msg_proc_error': '处理过程中发生错误:',
                'msg_warning': '警告',
                
                # Help Dialogs
                'dialog_env_title': '环境信息',
                'env_subtitle': 'OpenFocus 环境依赖',
                'env_python': 'Python 版本',
                'env_installed': '已安装: 版本 {}',
                'env_not_installed': '未安装',
                'env_cuda_avail': 'CUDA 可用: {}',
                'env_cuda_ver': 'CUDA 版本: {}',
                'env_mps_avail': 'MPS 可用 (Apple Silicon)',
                'env_gpu_accel': 'StackMFF-V4: GPU 加速可用',
                'env_no_gpu': '警告: 无 GPU 加速 (CUDA/MPS)',
                'env_cpu_mode': 'StackMFF-V4: 可用 (CPU 模式 - 较慢)',
                'env_fusion_cpu_only': '设置中已启用“强制 CPU 融合”: 融合不会使用 GPU',
                'env_stackmff_unavailable': 'StackMFF-V4 融合不可用',
                'env_dtcwt_unavailable': '未安装 (DTCWT 融合不可用)',
                'env_summary': '总结',
                'env_core_dep': '核心依赖:',
                'env_core_desc': '- OpenCV, NumPy, PyQt6: 基本功能所需',
                'env_gpu_opt': 'GPU 加速 (可选):',
                'env_gpu_desc': '- PyTorch: 启用 StackMFF-V4 (提供 CPU 回退模式，但较慢)',
                'env_fusion_alg': '融合算法:',
                'env_fusion_desc': '- DTCWT 库: DTCWT 融合所需',
                
                'dialog_contact_title': '联系我们',
                'contact_info_title': '联系信息',
                'contact_email': '邮箱',
                'contact_institution': '机构',
                'contact_zju': '浙江大学',
                'contact_github': 'GitHub',
                'contact_welcome': '我们热烈欢迎贡献者添加新的融合方法，帮助 OpenFocus 成长。',
                
                'btn_close': '关闭',

                # Batch Processing Dialog
                'batch_title': '批处理',
                'batch_import_mode': '导入模式',
                'batch_mode_multi': '多个文件夹 (每个文件夹一组)',
                'batch_mode_single': '单个文件夹 (自动分割为多组)',
                'batch_stack_folders': '图像栈文件夹',
                'batch_path_placeholder': '输入路径并按回车以刷新',
                'batch_btn_add': '添加文件夹',
                'batch_btn_remove': '移除选中项',
                'batch_single_split_settings': '单文件夹分割设置',
                'batch_folder_none': '文件夹: (无)',
                'batch_split_method': '分割方式:',
                'batch_split_fixed': '固定数量',
                'batch_split_time': '时间阈值',
                'batch_images_per_stack': '每组图像数:',
                'batch_time_threshold': '时间阈值:',
                'batch_unit_images': '张',
                'batch_unit_seconds': '秒',
                'batch_preview_default': '预览: 0 张图像 → 0 组',
                'batch_preview_fmt_count': '预览: {} 张图像 → {} 组 (每组 {} 张)',
                'batch_preview_fmt_time': '预览: {} 张图像 → {} 组 (阈值: {}秒)',
                'batch_preview_no_folder': '预览: 未选择文件夹',
                'batch_btn_select_split': '选择文件夹并分割',
                'batch_output_format': '输出格式',
                'batch_format_label': '格式:',
                'batch_quality_label': '质量:',
                'batch_output_location': '输出位置',
                'batch_out_subfolder': '在源文件夹中创建子文件夹',
                'batch_subfolder_name': '子文件夹名称:',
                'batch_out_same': '与源文件夹相同',
                'batch_out_custom': '指定输出文件夹',
                'batch_btn_browse': '浏览...',
                'batch_save_aligned': '保存已配准图像栈',
                'batch_proc_options': '处理选项',
                'batch_lbl_fusion': '融合方法: {}',
                'batch_lbl_reg': '配准方法: {}',
                'batch_lbl_kernel': '核大小: {}',
                'batch_btn_start': '开始批处理',
                'batch_btn_cancel': '取消',
                
                # Help Dialogs
                'help_render_title': '融合帮助',

                # Add Label Dialog
                'add_label_title': '添加标签配置',
                'label_target_stack': '目标堆栈:',
                'label_target_input': '输入图像栈',
                'label_target_registered': '已配准图像栈',
                'label_format': '格式字符串:',
                'label_start_val': '起始值:',
                'label_interval': '步长:',
                'label_x': 'X 坐标:',
                'label_y': 'Y 坐标:',
                'label_font_size': '字体大小:',
                'label_font_family': '字体:',
                'label_custom_text': '自定义文本:',
                'label_range': '范围:',
                'label_transparent_bg': '背景透明',
                'label_bg_color': '背景颜色:',
                'label_choose_bg': '选择背景颜色',
                'label_font_color': '字体颜色:',
                'label_choose_font_color': '选择字体颜色',
                
                'btn_ok': '确定',
                'btn_cancel': '取消',

                'msg_config_saved_title': '配置已保存',
                'msg_config_saved_text': '标签配置保存成功！',
                'msg_config_saved_info': '标签现在可见于选定的堆栈，并将在保存时包含在内。',

                'msg_settings_saved_title': '设置已保存',
                'msg_settings_saved_text': '所有设置保存成功！',
                'msg_settings_saved_info': '设置已保存到:\n{path}',
                'msg_depth_mode_changed_title': '位深度已更改',
                'msg_depth_mode_changed_text': '处理位深度已设置为 {mode}。',
                'msg_depth_mode_changed_info': '新的位深度在解码图像时生效，请重新加载当前图像栈。',
                'msg_settings_save_failed_text': '保存设置失败。',
                
                'msg_no_reg_labels': '无标签',
                'msg_reg_labels_disabled': '已配准堆栈标签未启用。',
                'msg_reg_labels_removed': '已从配准堆栈移除标签。',
                'msg_no_input_labels': '无标签',
                'msg_input_labels_disabled': '输入堆栈标签未启用。',
                'msg_input_labels_removed': '已从输入堆栈移除标签。',
                
                # Folder Import Dialog
                'import_folder_title': '导入文件夹',
                'import_folder_display': '文件夹: {}',
                'import_choice_label': '如何导入此文件夹？',
                'import_option_single': '单个图像栈 (作为一组加载)',
                'import_option_batch': '多个图像栈 (打开批处理)',
                
                # Downsample Dialog
                'ds_title': '下采样设置',
                'ds_label': '设置图像加载缩放比例 (下采样):',
                'ds_mode_percent': '按比例 (%)',
                'ds_mode_long_edge': '按长边 (像素)',
                'ds_long_edge_label': '长边:',
                'ds_hint': '对大图像使用较低的值以节省内存并加快处理速度。长边目标按每张图像应用，且不会放大图像。',

                # Thread settings dialog
                'dialog_thread_title': '线程数设置',
                'dialog_thread_group': '线程数',
                'dialog_thread_label': '线程数 (Thread Count):',
                'dialog_thread_help_title': '线程设置帮助',
                'dialog_thread_help_text': '''<h3>线程数设置</h3>
        <p>此设置控制支持多线程的算法所使用的处理线程数。通常设置为与CPU核心数匹配（一般为 2–16）。</p>

        <h4>当前的多线程支持情况：</h4>
        <ul>
        <li>GFG-FGF：支持用户控制线程数（默认上限：8）</li>
        <li>引导滤波融合 (GFF)：支持用户控制线程数（默认上限：4）</li>
        <li>图像配准：特征提取/变换操作使用指定线程数</li>
        <li>DCT, DTCWT, StackMFF-V4：目前不使用此设置（无线程控制）</li>
        </ul>

        <p>不使用此值的算法会安全地忽略它。为获得最佳性能，请避免将线程数设置得高于物理核心数。</p>

        <p>注意：安装 <code>opencv-contrib-python</code> 可以加速某些操作（例如引导滤波）。</p>
        ''',

                # DNG output settings dialog
                'dialog_dng_title': 'DNG 输出设置',
                'dialog_dng_group': 'DNG 输出',
                'dialog_dng_compression_label': '压缩方式:',
                'dialog_dng_quality_label': '有损质量:',
                'dialog_dng_fast_load': '嵌入快速加载预览',
                'dialog_dng_camera_space': '以源相机的色彩空间写入',
                'dng_camera_space_hint': '将结果存储为相机自身的 RAW 数据，'
                                         '因此相机配置文件能够正确应用，而不会造成偏色。'
                                         '此时文件在转换器中的呈现效果与 RAW 相同，'
                                         '而非与合成结果相同。需要源 RAW 文件；'
                                         '其他来源将回退为 sRGB。',
                'dng_compression_none': '不压缩',
                'dng_compression_lossless': '无损 (JPEG)',
                'dng_compression_lossy': '有损 (JPEG)',
                'dng_compression_hint_none': '文件最大，保存最快。条带中存放的就是像素本身。',
                'dng_compression_hint_lossless': '像素完全一致，16 位约为原来的一半，'
                                                 '8 位约为三分之一。保存较慢。',
                'dng_compression_hint_lossy': '文件最小，但仅支持 8 位——16 位结果在'
                                              '写出时会被降位。适合代理文件，不适合母版。',
                'dialog_dng_help_title': 'DNG 输出设置帮助',
                'dialog_dng_help_text': '''<h3>DNG 输出设置</h3>
        <p>无论是通过"保存"、文件夹导出还是批处理，只要结果保存为 <code>.dng</code>，
        这些设置都会生效。</p>

        <h4>压缩方式</h4>
        <ul>
        <li><b>不压缩</b> - 默认值。没有编码开销，文件中保存的正是处理流程输出的像素。</li>
        <li><b>无损 (JPEG)</b> - 无损霍夫曼 JPEG，这是 DNG 规范允许用于 16 位线性数据的
        唯一无损编码。像素可以完全还原；16 位约为不压缩体积的 50-55%，8 位约为 35-40%。
        大尺寸图像保存需要几秒钟。</li>
        <li><b>有损 (JPEG)</b> - 基线 DCT JPEG。DNG 仅允许它用于 8 位数据，因此 16 位结果
        在写出时会被降为 8 位，且不可恢复。适合代理文件和分享，不适合作为母版。</li>
        </ul>

        <p><b>注意：</b>只有安装了 <code>imagecodecs</code> 包时才会提供无损模式。
        即使没有该包，OpenFocus 也能写出无损文件，但无法再读回来；而保存的结果必须能够
        作为下一次堆栈的输入。</p>

        <h4>有损质量</h4>
        <p>有损模式使用的 JPEG 质量，范围 1 到 100。默认值 92 足够高，压缩不会成为画质瓶颈。
        该设置对其他两种模式无效。</p>

        <h4>嵌入快速加载预览</h4>
        <p>在 DNG 内部保存一份半分辨率的 JPEG 渲染结果。如果不嵌入，任何需要缩略图的程序
        （文件浏览器、Lightroom、RAW 转换器）都必须先解码完整分辨率的线性 RAW，而对于融合
        后的图像堆栈，这是文件中最慢的一步。预览只占几百 KB。</p>

        <p>这是 DNG 规范自带的预览机制，并非 Adobe 专有的"Fast Load Data"（那是只有 Adobe
        转换器才能生成的 Camera Raw 缓存），但用途相同，且兼容更多读取程序。</p>''',

                # StackMFF V4 Batch settings dialog
                'dialog_stackmffv4_batch_title': 'StackMFF V4 批量设置',
                'dialog_stackmffv4_batch_group': '批量处理',
                'dialog_stackmffv4_batch_label': '批量大小 (Batch Size):',
                'dialog_stackmffv4_batch_help_title': 'StackMFF V4 批量设置帮助',
                'dialog_stackmffv4_batch_help_text': '''<h3>StackMFF V4 批量设置</h3>
        <p>此设置控制使用 StackMFF V4 神经网络融合算法处理分块时的批量大小。
        批量处理允许在 GPU 上并行处理多个分块，提高效率。</p>

        <h4>批量大小：</h4>
        <ul>
        <li><b>默认值: 2</b> - 在内存使用和速度之间取得平衡</li>
        <li><b>较高值 (4-8)</b> - 处理更快，但需要更多 GPU 内存</li>
        <li><b>较低值 (1)</b> - 内存使用较低，适用于 GPU 内存有限的情况</li>
        </ul>

        <p><b>注意：</b>如果在 GPU 上遇到内存不足的错误，请尝试减小批量大小。
        此设置仅在处理大图像并启用分块时影响 StackMFF V4。</p>

        <p>CPU 模式不会显著受此设置影响，但会保持一致性。</p>''',

                # Tile settings dialog
                'dialog_tile_title': '分块设置',
                'dialog_tile_group': '分块选项',
                'dialog_tile_enabled_label': '启用分块:',
                'dialog_tile_enabled': '启用',
                'dialog_tile_disabled': '禁用',
                'dialog_tile_block_size': '分块大小:',
                'dialog_tile_overlap': '重叠大小:',
                'dialog_tile_threshold': '启用阈值:',
                'dialog_tile_help_title': '分块设置帮助',
                'dialog_tile_help_text': '''<h3>分块设置帮助</h3>
        <p>启用分块 (tile_enabled)：启用或禁用分块处理。启用时，大图像将被分割成较小的块进行处理，以减少内存使用。</p>

        <p>分块大小 (tile_block_size)：每个方形块的大小（像素）。根据内存和速度的平衡，典型值为 512–2048。</p>

        <p>重叠大小 (tile_overlap)：相邻块之间的重叠区域（像素），用于避免合并结果时的接缝。增加重叠有助于平滑边界。</p>

        <p>启用阈值 (tile_threshold)：如果图像的最长边大于此阈值，将考虑使用分块处理。较小的图像将作为整体处理。</p>''',

                # Registration settings dialog
                'dialog_reg_title': '配准设置',
                'dialog_reg_group': '配准选项',
                'dialog_reg_downscale': '下采样宽度:',
                'dialog_reg_help_title': '配准设置帮助',
                'dialog_reg_help_text': '''<h3>下采样宽度</h3>
        <p>downscale_width 控制在提取配准特征时的预处理（下采样）宽度。较小的值可以加快特征检测速度并减少内存使用，但可能会牺牲一些几何精度。</p>

        <p>该值在任何图像尺寸下都会按设置生效。在 1.30.5 及更早版本中，只要帧的长边达到 2048px（即所有现代相机文件），此值都会被悄悄替换为 1024，因此该设置对真实输入毫无作用。</p>

        <p>默认值 <code>1024</code> 是较好的折中。将其设为帧的完整宽度即在全分辨率下测量对齐，在 2560px 测试图像栈上可将几何精度提高 1.3 至 1.8 倍，代价约为两倍的配准时间。降低该值可加快配准并减少内存占用，但会损失精度。大于帧宽度的值与帧宽度等效——帧永远不会被放大。</p>

        <h3>并行 ECC 计算</h3>
        <p>在多个 CPU 核心上并发计算相邻帧之间的 ECC 变换。结果与串行计算完全一致，
        只是速度更快（典型图像栈约 2 倍）。仅在排查问题时才需要禁用。</p>

        <h3>参考帧</h3>
        <p>配准时保持固定、供其他帧对齐的基准帧。配准会将每一帧与相邻帧逐级串联，
        因此误差会随着与参考帧的距离而累积。</p>
        <p><b>首帧</b>（默认）以第 0 帧为参考，与旧版行为一致。
        <b>中间帧</b>以中心帧为参考，可将最长的串联链减半，并让残余漂移在图像栈中
        对称分布——适用于远端容易漂移的长图像栈。<b>末帧</b>以最后一帧为参考，
        相当于从另一端进行首帧对齐。</p>''',
                'dialog_reg_parallel_ecc': '并行 ECC 计算',
                'dialog_reg_reference': '参考帧:',
                'dialog_reg_reference_first': '首帧',
                'dialog_reg_reference_middle': '中间帧',
                'dialog_reg_reference_last': '末帧'
            }
        }

    def set_language(self, lang_code):
        if lang_code in self.translations:
            self.current_lang = lang_code
            self.languageChanged.emit()

    def get(self, key, default=None):
        val = self.translations.get(self.current_lang, {}).get(key)
        if val is None:
            # Fallback to english
            val = self.translations.get('en', {}).get(key)
        return val if val is not None else (default if default is not None else key)
    
    def t(self, key):
        return self.get(key)

# Global instance
trans = TranslationManager()
