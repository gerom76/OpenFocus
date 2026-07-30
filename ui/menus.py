from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QMainWindow
from locales import trans
from utils import bitdepth, dng


def setup_menus(window: QMainWindow) -> None:
    """Configure the main menu bar and attach actions to the window."""
    menubar = window.menuBar()

    # Initialize the ui_objs container if not present
    if not hasattr(window, 'ui_objs'):
        window.ui_objs = {}

    # --- File Menu ---
    file_menu = menubar.addMenu(trans.t('menu_file'))
    window.ui_objs['menu_file'] = file_menu

    open_action = QAction(trans.t('action_open_folder'), window)
    open_action.setShortcut("Ctrl+O")
    open_action.triggered.connect(window.open_folder_dialog)
    file_menu.addAction(open_action)
    window.ui_objs['action_open_folder'] = open_action

    open_video_action = QAction(trans.t('action_open_video'), window)
    open_video_action.setShortcut("Ctrl+Shift+O")
    open_video_action.triggered.connect(window.open_video_dialog)
    file_menu.addAction(open_video_action)
    window.ui_objs['action_open_video'] = open_video_action

    # Recently opened paths; contents are rebuilt by window.refresh_recent_menus()
    recent_folders_menu = QMenu(trans.t('menu_recent_folders'), window)
    file_menu.addMenu(recent_folders_menu)
    window.ui_objs['menu_recent_folders'] = recent_folders_menu
    window.menu_recent_folders = recent_folders_menu

    recent_videos_menu = QMenu(trans.t('menu_recent_videos'), window)
    file_menu.addMenu(recent_videos_menu)
    window.ui_objs['menu_recent_videos'] = recent_videos_menu
    window.menu_recent_videos = recent_videos_menu

    window.refresh_recent_menus()

    save_action = QAction(trans.t('action_save'), window)
    save_action.setShortcut("Ctrl+S")
    save_action.triggered.connect(window.export_manager.save_result)
    file_menu.addAction(save_action)
    window.ui_objs['action_save'] = save_action

    save_stack_menu = QMenu(trans.t('menu_save_stack'), window)
    file_menu.addMenu(save_stack_menu)
    window.ui_objs['menu_save_stack'] = save_stack_menu

    registered_stack_menu = QMenu(trans.t('menu_registered_stack'), window)
    save_stack_menu.addMenu(registered_stack_menu)
    window.ui_objs['menu_registered_stack'] = registered_stack_menu

    save_reg_folder_action = QAction(trans.t('action_save_folder'), window)
    save_reg_folder_action.setShortcut("Ctrl+Shift+S")
    save_reg_folder_action.triggered.connect(window.export_manager.save_result_stack)
    save_reg_folder_action.setProperty("trans_key", 'action_save_folder')
    registered_stack_menu.addAction(save_reg_folder_action)
    window.ui_objs['action_save_folder_reg'] = save_reg_folder_action

    save_reg_gif_action = QAction(trans.t('action_save_gif'), window)
    save_reg_gif_action.triggered.connect(lambda: window.export_manager.save_as_gif(target_type="registered"))
    save_reg_gif_action.setProperty("trans_key", 'action_save_gif')
    registered_stack_menu.addAction(save_reg_gif_action)
    window.ui_objs['action_save_gif_reg'] = save_reg_gif_action

    input_stack_menu = QMenu(trans.t('menu_input_stack'), window)
    save_stack_menu.addMenu(input_stack_menu)
    window.ui_objs['menu_input_stack'] = input_stack_menu

    save_input_folder_action = QAction(trans.t('action_save_folder'), window)
    save_input_folder_action.triggered.connect(window.export_manager.save_processed_input_stack)
    save_input_folder_action.setProperty("trans_key", 'action_save_folder')
    input_stack_menu.addAction(save_input_folder_action)
    window.ui_objs['action_save_folder_input'] = save_input_folder_action

    save_input_gif_action = QAction(trans.t('action_save_gif'), window)
    save_input_gif_action.triggered.connect(lambda: window.export_manager.save_as_gif(target_type="input"))
    save_input_gif_action.setProperty("trans_key", 'action_save_gif')
    input_stack_menu.addAction(save_input_gif_action)
    window.ui_objs['action_save_gif_input'] = save_input_gif_action

    file_menu.addSeparator()

    save_settings_action = QAction(trans.t('action_save_settings'), window)
    save_settings_action.triggered.connect(window.save_all_settings)
    file_menu.addAction(save_settings_action)
    window.ui_objs['action_save_settings'] = save_settings_action

    file_menu.addSeparator()

    clear_action = QAction(trans.t('action_clear_stack'), window)
    clear_action.setShortcut("Ctrl+W")
    clear_action.triggered.connect(window.source_manager.clear_image_stack)
    file_menu.addAction(clear_action)
    window.ui_objs['action_clear_stack'] = clear_action

    file_menu.addSeparator()

    exit_action = QAction(trans.t('action_exit'), window)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(window.close)
    file_menu.addAction(exit_action)
    window.ui_objs['action_exit'] = exit_action

    # --- Edit Menu ---
    edit_menu = menubar.addMenu(trans.t('menu_edit'))
    window.ui_objs['menu_edit'] = edit_menu

    rotate_menu = QMenu(trans.t('menu_rotate'), window)
    edit_menu.addMenu(rotate_menu)
    window.ui_objs['menu_rotate'] = rotate_menu

    rotate_90_cw_action = QAction(trans.t('action_rotate_90_cw'), window)
    rotate_90_cw_action.triggered.connect(lambda: window.rotate_stack(1))
    rotate_menu.addAction(rotate_90_cw_action)
    window.ui_objs['action_rotate_90_cw'] = rotate_90_cw_action

    rotate_90_ccw_action = QAction(trans.t('action_rotate_90_ccw'), window)
    rotate_90_ccw_action.triggered.connect(lambda: window.rotate_stack(2))
    rotate_menu.addAction(rotate_90_ccw_action)
    window.ui_objs['action_rotate_90_ccw'] = rotate_90_ccw_action

    rotate_180_action = QAction(trans.t('action_rotate_180'), window)
    rotate_180_action.triggered.connect(lambda: window.rotate_stack(0))
    rotate_menu.addAction(rotate_180_action)
    window.ui_objs['action_rotate_180'] = rotate_180_action

    flip_menu = QMenu(trans.t('menu_flip'), window)
    edit_menu.addMenu(flip_menu)
    window.ui_objs['menu_flip'] = flip_menu

    flip_horizontal_action = QAction(trans.t('action_flip_h'), window)
    flip_horizontal_action.triggered.connect(lambda: window.flip_stack(1))
    flip_menu.addAction(flip_horizontal_action)
    window.ui_objs['action_flip_h'] = flip_horizontal_action

    flip_vertical_action = QAction(trans.t('action_flip_v'), window)
    flip_vertical_action.triggered.connect(lambda: window.flip_stack(0))
    flip_menu.addAction(flip_vertical_action)
    window.ui_objs['action_flip_v'] = flip_vertical_action

    window.resize_action = QAction(trans.t('action_resize'), window)
    window.resize_action.triggered.connect(window.resize_all_images)
    window.resize_action.setEnabled(False)
    edit_menu.addAction(window.resize_action)
    window.ui_objs['action_resize'] = window.resize_action

    window.add_label_action = QAction(trans.t('action_add_label'), window)
    window.add_label_action.triggered.connect(window.label_manager.show_add_label_dialog)
    window.add_label_action.setEnabled(False)
    edit_menu.addAction(window.add_label_action)
    window.ui_objs['action_add_label'] = window.add_label_action

    delete_label_menu = QMenu(trans.t('menu_del_label'), window)
    edit_menu.addMenu(delete_label_menu)
    window.ui_objs['menu_del_label'] = delete_label_menu

    window.del_reg_label_action = QAction(trans.t('action_del_reg_label'), window)
    window.del_reg_label_action.triggered.connect(window.label_manager.delete_registered_labels)
    window.del_reg_label_action.setEnabled(False)
    delete_label_menu.addAction(window.del_reg_label_action)
    window.ui_objs['action_del_reg_label'] = window.del_reg_label_action

    window.del_input_label_action = QAction(trans.t('action_del_input_label'), window)
    window.del_input_label_action.triggered.connect(window.label_manager.delete_input_labels)
    window.del_input_label_action.setEnabled(False)
    delete_label_menu.addAction(window.del_input_label_action)
    window.ui_objs['action_del_input_label'] = window.del_input_label_action

    # --- Tools Menu ---
    batch_menu = menubar.addMenu(trans.t('menu_tools'))
    window.ui_objs['menu_tools'] = batch_menu

    batch_action = QAction(trans.t('action_batch_process'), window)
    batch_action.triggered.connect(window.show_batch_processing_dialog)
    batch_menu.addAction(batch_action)
    window.ui_objs['action_batch_process'] = batch_action

    # --- View Menu ---
    view_menu = menubar.addMenu(trans.t('menu_view'))
    window.ui_objs['menu_view'] = view_menu

    show_console_action = QAction(trans.t('action_show_console'), window)
    show_console_action.setCheckable(True)
    show_console_action.setChecked(True)
    show_console_action.setShortcut("Ctrl+`")
    show_console_action.toggled.connect(window.toggle_status_console)
    view_menu.addAction(show_console_action)
    window.ui_objs['action_show_console'] = show_console_action
    window.action_show_console = show_console_action

    show_source_action = QAction(trans.t('action_show_source_stack'), window)
    show_source_action.setCheckable(True)
    show_source_action.setChecked(True)
    show_source_action.setShortcut("Ctrl+1")
    show_source_action.toggled.connect(window.set_source_panel_visible)
    view_menu.addAction(show_source_action)
    window.ui_objs['action_show_source_stack'] = show_source_action
    window.action_show_source_stack = show_source_action

    # --- Settings Menu ---
    settings_menu = menubar.addMenu(trans.t('menu_settings'))
    window.ui_objs['menu_settings'] = settings_menu

    # Language Submenu
    lang_menu = QMenu(trans.t('menu_language'), window)
    settings_menu.addMenu(lang_menu)
    window.ui_objs['menu_language'] = lang_menu
    
    lang_group = QActionGroup(window)
    lang_group.setExclusive(True)
    
    lang_en = QAction(trans.t('action_lang_en'), window)
    lang_en.setCheckable(True)
    lang_en.setChecked(trans.current_lang == 'en')
    lang_en.triggered.connect(lambda: window.set_language('en'))
    lang_group.addAction(lang_en)
    lang_menu.addAction(lang_en)
    window.ui_objs['action_lang_en'] = lang_en
    
    lang_zh = QAction(trans.t('action_lang_zh'), window)
    lang_zh.setCheckable(True)
    lang_zh.setChecked(trans.current_lang == 'zh')
    lang_zh.triggered.connect(lambda: window.set_language('zh'))
    lang_group.addAction(lang_zh)
    lang_menu.addAction(lang_zh)
    window.ui_objs['action_lang_zh'] = lang_zh

    # Bit-depth Submenu. A menu rather than a dialog because switching depth is
    # a per-stack decision made often, not a setting configured once.
    depth_menu = QMenu(trans.t('menu_bit_depth'), window)
    settings_menu.addMenu(depth_menu)
    window.ui_objs['menu_bit_depth'] = depth_menu

    depth_group = QActionGroup(window)
    depth_group.setExclusive(True)

    current_depth_mode = getattr(window, 'bit_depth_mode', bitdepth.MODE_AUTO)
    for mode, key in (
        (bitdepth.MODE_AUTO, 'action_depth_auto'),
        (bitdepth.MODE_8, 'action_depth_8'),
        (bitdepth.MODE_16, 'action_depth_16'),
    ):
        action = QAction(trans.t(key), window)
        action.setCheckable(True)
        action.setChecked(current_depth_mode == mode)
        # Default-arg binding, or every lambda would close over the last mode
        action.triggered.connect(lambda _checked=False, m=mode: window.set_bit_depth_mode(m))
        depth_group.addAction(action)
        depth_menu.addAction(action)
        window.ui_objs[key] = action

    tile_action = QAction(trans.t('action_tile_settings'), window)
    # Open the tile settings dialog
    tile_action.triggered.connect(lambda: window.show_tile_settings())
    settings_menu.addAction(tile_action)
    window.ui_objs['action_tile_settings'] = tile_action

    registration_action = QAction(trans.t('action_reg_settings'), window)
    registration_action.triggered.connect(lambda: window.show_registration_settings())
    settings_menu.addAction(registration_action)
    window.ui_objs['action_reg_settings'] = registration_action

    thread_settings_action = QAction(trans.t('action_thread_settings'), window)
    thread_settings_action.triggered.connect(lambda: window.show_thread_settings())
    settings_menu.addAction(thread_settings_action)
    window.ui_objs['action_thread_settings'] = thread_settings_action

    # Only shown where DNG can be written at all, so the menu does not offer
    # settings for a format this build leaves out of the save dialogs.
    if dng.is_available():
        dng_action = QAction(trans.t('action_dng_settings'), window)
        dng_action.triggered.connect(lambda: window.show_dng_settings())
        settings_menu.addAction(dng_action)
        window.ui_objs['action_dng_settings'] = dng_action

    stackmffv4_batch_action = QAction(trans.t('action_stackmffv4_batch_settings'), window)
    stackmffv4_batch_action.triggered.connect(lambda: window.show_stackmffv4_batch_settings())
    settings_menu.addAction(stackmffv4_batch_action)
    window.ui_objs['action_stackmffv4_batch_settings'] = stackmffv4_batch_action

    gpu_loading_action = QAction(trans.t('action_gpu_loading'), window)
    gpu_loading_action.setCheckable(True)
    gpu_loading_action.setChecked(getattr(window, 'gpu_loading_enabled', True))
    gpu_loading_action.triggered.connect(lambda checked: window.set_gpu_loading_enabled(checked))
    settings_menu.addAction(gpu_loading_action)
    window.ui_objs['action_gpu_loading'] = gpu_loading_action

    # --- Help Menu ---
    help_menu = menubar.addMenu(trans.t('menu_help'))
    window.ui_objs['menu_help'] = help_menu

    env_action = QAction(trans.t('action_env_info'), window)
    env_action.triggered.connect(window.show_environment_info)
    help_menu.addAction(env_action)
    window.ui_objs['action_env_info'] = env_action

    help_menu.addSeparator()

    contact_action = QAction(trans.t('action_contact'), window)
    contact_action.triggered.connect(window.show_contact_info)
    help_menu.addAction(contact_action)
    window.ui_objs['action_contact'] = contact_action
