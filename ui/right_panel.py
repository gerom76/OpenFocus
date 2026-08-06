from dataclasses import dataclass

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from constants import (
    DCT_BLEND_DEFAULT,
    DCT_BLOCK_SIZES,
    DCT_BLOCK_SIZE_DEFAULT,
    DCT_PLATEAU_DEFAULT,
    DCT_PLATEAU_PRESETS,
    DEPTH_SMOOTHING_DEFAULT,
    KERNEL_SIZE_MAX_DCT,
    PYRAMID_BASE_DEFAULT,
    PYRAMID_BASE_PRESETS,
    PYRAMID_COHERENCE_DEFAULT,
    PYRAMID_COHERENCE_PRESETS,
    PYRAMID_ENVELOPE_DEFAULT,
    PYRAMID_LEVELS,
    PYRAMID_LEVELS_DEFAULT,
    PYRAMID_NOISE_GATE_DEFAULT,
    PYRAMID_SELECTIVITY_DEFAULT,
    PYRAMID_SELECTIVITY_PRESETS,
)
from dialogs import RegistrationHelpDialog, RenderMethodHelpDialog
from ui.styles import (
    HELP_BUTTON_STYLE,
    HOVER_HIGHLIGHT_BUTTON_STYLE,
    OUTPUT_LIST_STYLE,
    SOURCE_LIST_STYLE,
    SOURCE_TOOLBAR_STYLE,
    STOP_BUTTON_STYLE,
)
from locales import trans
from widgets.output_list import OutputListWidget


@dataclass
class RightPanelComponents:
    widget: QFrame
    splitter: QSplitter
    btn_reset: QPushButton
    btn_render: QPushButton
    btn_stop: QPushButton
    btn_method_help: QPushButton
    btn_reg_help: QPushButton
    rb_a: QRadioButton
    rb_b: QRadioButton
    rb_c: QRadioButton
    rb_gfg: QRadioButton
    rb_pyramid: QRadioButton
    rb_dmap_max: QRadioButton
    rb_dmap_avg: QRadioButton
    rb_d: QRadioButton
    cb_ifcnn: QCheckBox
    cb_align_scale: QCheckBox
    cb_align_homography: QCheckBox
    cb_align_ecc: QCheckBox
    slider_smooth: QSlider
    smooth_value_label: QLabel
    smooth_widget: QWidget
    slider_halo: QSlider
    halo_value_label: QLabel
    halo_widget: QWidget
    slider_depth_smooth: QSlider
    depth_smooth_value_label: QLabel
    coherent_widget: QWidget
    combo_dct_block: QComboBox
    combo_dct_plateau: QComboBox
    cb_dct_blend: QCheckBox
    dct_widget: QWidget
    lbl_dct_block: QLabel
    lbl_dct_plateau: QLabel
    combo_pyr_levels: QComboBox
    combo_pyr_selectivity: QComboBox
    combo_pyr_coherence: QComboBox
    combo_pyr_base: QComboBox
    cb_pyr_noise_gate: QCheckBox
    cb_pyr_envelope: QCheckBox
    pyramid_widget: QWidget
    lbl_pyr_levels: QLabel
    lbl_pyr_selectivity: QLabel
    lbl_pyr_coherence: QLabel
    lbl_pyr_base: QLabel
    combo_contrast: QComboBox
    slider_contrast: QSlider
    contrast_value_label: QLabel
    lbl_contrast: QLabel

    # Groups
    method_group: QGroupBox
    registration_group: QGroupBox
    lbl_kernel: QLabel
    lbl_halo: QLabel
    lbl_depth_smooth: QLabel

    source_images_label: QLabel
    file_list: QListWidget
    btn_select_all: QPushButton
    btn_select_invert: QPushButton
    btn_select_none: QPushButton
    btn_select_nth: QPushButton
    spin_select_nth: QSpinBox
    output_label: QLabel
    output_list: QListWidget
    
    # Status Panel Labels
    lbl_status_loaded: QLabel
    lbl_status_resolution: QLabel
    lbl_status_gpu: QLabel
    lbl_status_memory: QLabel


def create_right_panel() -> RightPanelComponents:
    right_panel = QFrame()
    right_panel.setMinimumWidth(280)
    right_panel.setStyleSheet("background-color: #2b2b2b; border-left: 1px solid #111;")
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)

    right_splitter = QSplitter(Qt.Orientation.Vertical)
    right_layout.addWidget(right_splitter)

    # Configuration group ----------------------------------
    config_widget = QWidget()
    config_layout = QVBoxLayout(config_widget)
    config_layout.setContentsMargins(10, 10, 10, 10)

    method_registration_layout = QHBoxLayout()

    method_group = QGroupBox(trans.t('group_fusion'))
    method_layout = QVBoxLayout(method_group)
    rb_a = QRadioButton(trans.t('radio_guided_filter'))
    rb_a.setChecked(True)
    rb_a.setAutoExclusive(False)
    rb_b = QRadioButton(trans.t('radio_dct'))
    rb_b.setAutoExclusive(False)
    rb_c = QRadioButton(trans.t('radio_dtcwt'))
    rb_c.setAutoExclusive(False)
    rb_gfg = QRadioButton(trans.t('radio_gfg'))
    rb_gfg.setAutoExclusive(False)
    rb_pyramid = QRadioButton(trans.t('radio_pyramid'))
    rb_pyramid.setAutoExclusive(False)
    rb_dmap_max = QRadioButton(trans.t('radio_depthmap_max'))
    rb_dmap_max.setAutoExclusive(False)
    rb_dmap_avg = QRadioButton(trans.t('radio_depthmap_avg'))
    rb_dmap_avg.setAutoExclusive(False)
    rb_d = QRadioButton(trans.t('radio_stackmff'))
    rb_d.setAutoExclusive(False)

    method_layout.addWidget(rb_a)
    method_layout.addWidget(rb_b)
    method_layout.addWidget(rb_c)
    # Post-fusion refinement stage, combinable with any fusion method above
    cb_ifcnn = QCheckBox(trans.t('check_ifcnn_refine'))
    cb_ifcnn.setChecked(False)

    method_layout.addWidget(rb_gfg)
    method_layout.addWidget(rb_pyramid)
    method_layout.addWidget(rb_dmap_max)
    method_layout.addWidget(rb_dmap_avg)
    method_layout.addWidget(rb_d)
    method_layout.addSpacing(6)
    method_layout.addWidget(cb_ifcnn)
    method_layout.addStretch()

    method_help_layout = QHBoxLayout()
    method_help_layout.addStretch()
    btn_method_help = QPushButton("?")
    btn_method_help.setFixedSize(22, 22)
    btn_method_help.setFont(QFont("Arial", 16))
    btn_method_help.setStyleSheet(HELP_BUTTON_STYLE)
    method_help_layout.addWidget(btn_method_help)
    method_layout.addLayout(method_help_layout)

    registration_group = QGroupBox(trans.t('group_registration'))
    registration_layout = QVBoxLayout(registration_group)
    cb_align_scale = QCheckBox(trans.t('check_align_scale'))
    cb_align_scale.setChecked(False)
    cb_align_ecc = QCheckBox(trans.t('check_align_ecc'))
    cb_align_ecc.setChecked(True)
    cb_align_homography = QCheckBox(trans.t('check_align_homography'))
    cb_align_homography.setChecked(False)

    registration_layout.addWidget(cb_align_scale)
    registration_layout.addWidget(cb_align_ecc)
    registration_layout.addWidget(cb_align_homography)
    registration_layout.addStretch()

    reg_help_layout = QHBoxLayout()
    reg_help_layout.addStretch()
    btn_reg_help = QPushButton("?")
    btn_reg_help.setFixedSize(22, 22)
    btn_reg_help.setFont(QFont("Arial", 16))
    btn_reg_help.setStyleSheet(HELP_BUTTON_STYLE)
    reg_help_layout.addWidget(btn_reg_help)
    registration_layout.addLayout(reg_help_layout)

    method_registration_layout.addWidget(registration_group)
    method_registration_layout.addWidget(method_group)
    config_layout.addLayout(method_registration_layout)

    smooth_widget = QWidget()
    smooth_layout = QVBoxLayout(smooth_widget)
    smooth_layout.setContentsMargins(0, 5, 0, 5)
    smooth_top = QHBoxLayout()
    lbl_kernel = QLabel(trans.t('label_kernel'))
    smooth_top.addWidget(lbl_kernel)
    smooth_top.addStretch()
    lbl_smooth_value = QLabel("31")
    smooth_top.addWidget(lbl_smooth_value)
    slider_smooth = QSlider(Qt.Orientation.Horizontal)
    slider_smooth.setRange(1, KERNEL_SIZE_MAX_DCT)
    slider_smooth.setSingleStep(2)
    slider_smooth.setPageStep(2)
    slider_smooth.setValue(31)
    smooth_layout.addLayout(smooth_top)
    smooth_layout.addWidget(slider_smooth)
    config_layout.addWidget(smooth_widget)

    # Halo suppression (depth-map methods only) --------------
    halo_widget = QWidget()
    halo_layout = QVBoxLayout(halo_widget)
    halo_layout.setContentsMargins(0, 5, 0, 5)
    halo_top = QHBoxLayout()
    lbl_halo = QLabel(trans.t('label_halo'))
    halo_top.addWidget(lbl_halo)
    halo_top.addStretch()
    lbl_halo_value = QLabel(trans.t('halo_off'))
    halo_top.addWidget(lbl_halo_value)
    slider_halo = QSlider(Qt.Orientation.Horizontal)
    slider_halo.setRange(0, 30)
    slider_halo.setSingleStep(1)
    slider_halo.setPageStep(2)
    slider_halo.setValue(0)
    halo_layout.addLayout(halo_top)
    halo_layout.addWidget(slider_halo)
    halo_widget.setEnabled(False)  # only the depth-map methods use it
    config_layout.addWidget(halo_widget)

    # Coherent depth (Depth Map (Max) only) ------------------
    # The depth map the hard select arrives at is not trustworthy everywhere,
    # and where it is not, saying so beats pretending otherwise: this sets how
    # readily a pixel's own measurement is given up for what its neighbourhood
    # implies.
    coherent_widget = QWidget()
    coherent_layout = QVBoxLayout(coherent_widget)
    coherent_layout.setContentsMargins(0, 5, 0, 5)

    depth_smooth_top = QHBoxLayout()
    lbl_depth_smooth = QLabel(trans.t('label_depth_smooth'))
    depth_smooth_top.addWidget(lbl_depth_smooth)
    depth_smooth_top.addStretch()
    lbl_depth_smooth_value = QLabel(f"{DEPTH_SMOOTHING_DEFAULT}%")
    depth_smooth_top.addWidget(lbl_depth_smooth_value)
    slider_depth_smooth = QSlider(Qt.Orientation.Horizontal)
    slider_depth_smooth.setRange(0, 100)
    slider_depth_smooth.setSingleStep(1)
    slider_depth_smooth.setPageStep(10)
    slider_depth_smooth.setValue(DEPTH_SMOOTHING_DEFAULT)
    coherent_layout.addLayout(depth_smooth_top)
    coherent_layout.addWidget(slider_depth_smooth)

    coherent_widget.setEnabled(False)  # only the hard per-pixel select uses it
    config_layout.addWidget(coherent_widget)

    # DCT tuning (DCT only) ---------------------------------
    dct_widget = QWidget()
    dct_layout = QVBoxLayout(dct_widget)
    dct_layout.setContentsMargins(0, 5, 0, 5)

    dct_block_row = QHBoxLayout()
    lbl_dct_block = QLabel(trans.t('label_dct_block'))
    dct_block_row.addWidget(lbl_dct_block)
    dct_block_row.addStretch()
    combo_dct_block = QComboBox()
    for size in DCT_BLOCK_SIZES:
        combo_dct_block.addItem(f"{size} px", size)
    combo_dct_block.setCurrentIndex(
        list(DCT_BLOCK_SIZES).index(DCT_BLOCK_SIZE_DEFAULT))
    dct_block_row.addWidget(combo_dct_block)
    dct_layout.addLayout(dct_block_row)

    dct_plateau_row = QHBoxLayout()
    lbl_dct_plateau = QLabel(trans.t('label_dct_plateau'))
    dct_plateau_row.addWidget(lbl_dct_plateau)
    dct_plateau_row.addStretch()
    combo_dct_plateau = QComboBox()
    # userData carries the stable preset key; the label is translated.
    for key, _value in DCT_PLATEAU_PRESETS:
        combo_dct_plateau.addItem(trans.t(f'dct_plateau_{key}'), key)
    combo_dct_plateau.setCurrentIndex(
        [k for k, _ in DCT_PLATEAU_PRESETS].index(DCT_PLATEAU_DEFAULT))
    dct_plateau_row.addWidget(combo_dct_plateau)
    dct_layout.addLayout(dct_plateau_row)

    cb_dct_blend = QCheckBox(trans.t('label_dct_blend'))
    cb_dct_blend.setChecked(DCT_BLEND_DEFAULT)
    dct_layout.addWidget(cb_dct_blend)

    dct_widget.setEnabled(False)  # only DCT uses these
    config_layout.addWidget(dct_widget)

    # Pyramid tuning (Pyramid only) -------------------------
    # Six controls is more than the panel can afford to keep greyed out the way
    # the DCT block above is, so this one is hidden instead of disabled. Hiding
    # a widget keeps its state, so switching methods and back does not forget
    # what was set.
    pyramid_widget = QWidget()
    pyramid_layout = QVBoxLayout(pyramid_widget)
    pyramid_layout.setContentsMargins(0, 5, 0, 5)

    def _preset_row(label_key, presets, default_key):
        """One 'label ........ [combo]' row; the combo carries the preset key."""
        row = QHBoxLayout()
        label = QLabel(trans.t(label_key))
        row.addWidget(label)
        row.addStretch()
        combo = QComboBox()
        for key, _value in presets:
            combo.addItem(trans.t(f'{label_key}_{key}'), key)
        combo.setCurrentIndex([k for k, _ in presets].index(default_key))
        row.addWidget(combo)
        pyramid_layout.addLayout(row)
        return label, combo

    pyr_levels_row = QHBoxLayout()
    lbl_pyr_levels = QLabel(trans.t('label_pyr_levels'))
    pyr_levels_row.addWidget(lbl_pyr_levels)
    pyr_levels_row.addStretch()
    combo_pyr_levels = QComboBox()
    for depth in PYRAMID_LEVELS:
        # 0 is "as deep as the image allows", which is what every render did
        # before the control existed.
        combo_pyr_levels.addItem(trans.t('pyr_levels_auto') if depth == 0
                                 else str(depth), depth)
    combo_pyr_levels.setCurrentIndex(list(PYRAMID_LEVELS).index(PYRAMID_LEVELS_DEFAULT))
    pyr_levels_row.addWidget(combo_pyr_levels)
    pyramid_layout.addLayout(pyr_levels_row)

    lbl_pyr_selectivity, combo_pyr_selectivity = _preset_row(
        'label_pyr_selectivity', PYRAMID_SELECTIVITY_PRESETS,
        PYRAMID_SELECTIVITY_DEFAULT)
    lbl_pyr_coherence, combo_pyr_coherence = _preset_row(
        'label_pyr_coherence', PYRAMID_COHERENCE_PRESETS,
        PYRAMID_COHERENCE_DEFAULT)
    lbl_pyr_base, combo_pyr_base = _preset_row(
        'label_pyr_base', PYRAMID_BASE_PRESETS, PYRAMID_BASE_DEFAULT)

    cb_pyr_noise_gate = QCheckBox(trans.t('label_pyr_noise_gate'))
    cb_pyr_noise_gate.setChecked(PYRAMID_NOISE_GATE_DEFAULT)
    pyramid_layout.addWidget(cb_pyr_noise_gate)

    cb_pyr_envelope = QCheckBox(trans.t('label_pyr_envelope'))
    cb_pyr_envelope.setChecked(PYRAMID_ENVELOPE_DEFAULT)
    pyramid_layout.addWidget(cb_pyr_envelope)

    pyramid_widget.setVisible(False)  # shown only while Pyramid is selected
    config_layout.addWidget(pyramid_widget)

    # Contrast (post-fusion output enhancement) --------------
    contrast_widget = QWidget()
    contrast_layout = QVBoxLayout(contrast_widget)
    contrast_layout.setContentsMargins(0, 5, 0, 5)
    contrast_top = QHBoxLayout()
    lbl_contrast = QLabel(trans.t('label_contrast'))
    contrast_top.addWidget(lbl_contrast)
    combo_contrast = QComboBox()
    # userData carries the stable method key; the label is translated.
    combo_contrast.addItem(trans.t('contrast_off'), 'off')
    combo_contrast.addItem(trans.t('contrast_auto'), 'auto')
    combo_contrast.addItem(trans.t('contrast_clahe'), 'clahe')
    contrast_top.addWidget(combo_contrast)
    contrast_top.addStretch()
    contrast_value_label = QLabel("50%")
    contrast_top.addWidget(contrast_value_label)
    slider_contrast = QSlider(Qt.Orientation.Horizontal)
    slider_contrast.setRange(0, 100)
    slider_contrast.setSingleStep(5)
    slider_contrast.setPageStep(10)
    slider_contrast.setValue(50)
    slider_contrast.setEnabled(False)  # off by default, so strength is inert
    contrast_layout.addLayout(contrast_top)
    contrast_layout.addWidget(slider_contrast)
    config_layout.addWidget(contrast_widget)

    button_bar = QHBoxLayout()
    btn_reset = QPushButton(trans.t('btn_reset'))
    btn_reset.setStyleSheet(HOVER_HIGHLIGHT_BUTTON_STYLE)
    btn_render = QPushButton(trans.t('btn_render'))
    btn_render.setFixedHeight(40)
    btn_render.setStyleSheet(HOVER_HIGHLIGHT_BUTTON_STYLE)
    # Stop sits next to Start Render and interrupts the running render; it is
    # inert until a render is in progress.
    btn_stop = QPushButton(trans.t('btn_stop'))
    btn_stop.setFixedHeight(40)
    btn_stop.setStyleSheet(STOP_BUTTON_STYLE)
    btn_stop.setEnabled(False)
    button_bar.addWidget(btn_reset)
    button_bar.addWidget(btn_render)
    button_bar.addWidget(btn_stop)
    config_layout.addLayout(button_bar)

    right_splitter.addWidget(config_widget)

    # Source list -----------------------------------------
    source_list_widget = QWidget()
    source_list_layout = QVBoxLayout(source_list_widget)
    source_list_layout.setContentsMargins(10, 10, 10, 10)
    source_images_label = QLabel(trans.t('label_source_images').format(0))

    # Header row: count label on the left, check-state toolbar (All / None /
    # every N-th) right-aligned so the buttons stay put as the label grows.
    source_toolbar = QHBoxLayout()
    source_toolbar.setContentsMargins(0, 0, 0, 2)
    source_toolbar.setSpacing(4)
    source_toolbar.addWidget(source_images_label)
    source_toolbar.addStretch()
    btn_select_all = QPushButton(trans.t('btn_select_all'))
    btn_select_all.setToolTip(trans.t('tip_select_all'))
    btn_select_invert = QPushButton(trans.t('btn_select_invert'))
    btn_select_invert.setToolTip(trans.t('tip_select_invert'))
    btn_select_none = QPushButton(trans.t('btn_select_none'))
    btn_select_none.setToolTip(trans.t('tip_select_none'))
    btn_select_nth = QPushButton(trans.t('btn_select_nth'))
    btn_select_nth.setToolTip(trans.t('tip_select_nth'))
    spin_select_nth = QSpinBox()
    spin_select_nth.setRange(1, 999)
    spin_select_nth.setValue(2)
    spin_select_nth.setFixedWidth(52)
    spin_select_nth.setToolTip(trans.t('tip_select_nth'))
    for button in (btn_select_all, btn_select_invert, btn_select_none, btn_select_nth):
        button.setFixedHeight(22)
    source_toolbar.addWidget(btn_select_all)
    source_toolbar.addWidget(btn_select_invert)
    source_toolbar.addWidget(btn_select_none)
    source_toolbar.addWidget(btn_select_nth)
    source_toolbar.addWidget(spin_select_nth)
    source_toolbar_widget = QWidget()
    source_toolbar_widget.setLayout(source_toolbar)
    source_toolbar_widget.setStyleSheet(SOURCE_TOOLBAR_STYLE)
    source_list_layout.addWidget(source_toolbar_widget)

    file_list = QListWidget()
    # Dense rows: uniform item sizes and thumbnails no taller than a text line
    # let many more frames fit on screen than the 40px icons of the output list.
    # SourceManager re-shapes the icon box to the loaded frames' aspect ratio.
    file_list.setIconSize(QSize(30, 20))
    file_list.setUniformItemSizes(True)
    file_list.setSpacing(0)
    file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    file_list.setStyleSheet(SOURCE_LIST_STYLE)
    source_list_layout.addWidget(file_list)

    right_splitter.addWidget(source_list_widget)

    # Output list -----------------------------------------
    output_list_widget = QWidget()
    output_list_layout = QVBoxLayout(output_list_widget)
    output_list_layout.setContentsMargins(10, 10, 10, 10)
    output_label = QLabel(trans.t('label_output').format(0))
    output_list_layout.addWidget(output_label)
    output_list = OutputListWidget()
    # OutputManager re-shapes the icon box to each result's aspect ratio so the
    # thumbnails fill their rows; this is only the size used while empty.
    output_list.setIconSize(QSize(30, 20))
    output_list.setUniformItemSizes(True)
    output_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    output_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    output_list.setDragEnabled(True)
    output_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
    output_list.setStyleSheet(OUTPUT_LIST_STYLE)
    output_list_layout.addWidget(output_list)

    right_splitter.addWidget(output_list_widget)

    right_splitter.setStretchFactor(0, 1)
    right_splitter.setStretchFactor(1, 3)
    right_splitter.setStretchFactor(2, 3)

    # Status Panel -----------------------------------------
    status_widget = QFrame()
    status_widget.setStyleSheet("""
        QFrame {
            background-color: #252526; 
            border-top: 1px solid #3e3e42;
        }
        QLabel {
            color: #cccccc; 
            font-size: 11px;
            font-family: Consolas, "Segoe UI", monospace;
        }
    """)
    status_layout = QGridLayout(status_widget)
    status_layout.setContentsMargins(10, 8, 10, 8)
    status_layout.setHorizontalSpacing(15)
    status_layout.setVerticalSpacing(4)

    lbl_status_loaded = QLabel(trans.t('status_loaded').format(0))
    lbl_status_gpu = QLabel(trans.t('status_gpu').format('-'))
    lbl_status_resolution = QLabel(trans.t('status_res').format('-'))
    lbl_status_memory = QLabel(trans.t('status_ram').format('-'))

    status_layout.addWidget(lbl_status_loaded, 0, 0)
    status_layout.addWidget(lbl_status_gpu, 0, 1)
    status_layout.addWidget(lbl_status_resolution, 1, 0)
    status_layout.addWidget(lbl_status_memory, 1, 1)
    
    status_layout.setColumnStretch(0, 1)
    status_layout.setColumnStretch(1, 1)

    right_layout.addWidget(status_widget)

    return RightPanelComponents(
        widget=right_panel,
        splitter=right_splitter,
        btn_reset=btn_reset,
        btn_render=btn_render,
        btn_stop=btn_stop,
        btn_method_help=btn_method_help,
        btn_reg_help=btn_reg_help,
        rb_a=rb_a,
        rb_b=rb_b,
        rb_c=rb_c,
        rb_gfg=rb_gfg,
        rb_pyramid=rb_pyramid,
        rb_dmap_max=rb_dmap_max,
        rb_dmap_avg=rb_dmap_avg,
        rb_d=rb_d,
        cb_ifcnn=cb_ifcnn,
        cb_align_scale=cb_align_scale,
        cb_align_homography=cb_align_homography,
        cb_align_ecc=cb_align_ecc,
        slider_smooth=slider_smooth,
        smooth_value_label=lbl_smooth_value,
        smooth_widget=smooth_widget,
        slider_halo=slider_halo,
        halo_value_label=lbl_halo_value,
        halo_widget=halo_widget,
        slider_depth_smooth=slider_depth_smooth,
        depth_smooth_value_label=lbl_depth_smooth_value,
        coherent_widget=coherent_widget,
        combo_dct_block=combo_dct_block,
        combo_dct_plateau=combo_dct_plateau,
        cb_dct_blend=cb_dct_blend,
        dct_widget=dct_widget,
        lbl_dct_block=lbl_dct_block,
        lbl_dct_plateau=lbl_dct_plateau,
        combo_pyr_levels=combo_pyr_levels,
        combo_pyr_selectivity=combo_pyr_selectivity,
        combo_pyr_coherence=combo_pyr_coherence,
        combo_pyr_base=combo_pyr_base,
        cb_pyr_noise_gate=cb_pyr_noise_gate,
        cb_pyr_envelope=cb_pyr_envelope,
        pyramid_widget=pyramid_widget,
        lbl_pyr_levels=lbl_pyr_levels,
        lbl_pyr_selectivity=lbl_pyr_selectivity,
        lbl_pyr_coherence=lbl_pyr_coherence,
        lbl_pyr_base=lbl_pyr_base,
        combo_contrast=combo_contrast,
        slider_contrast=slider_contrast,
        contrast_value_label=contrast_value_label,
        lbl_contrast=lbl_contrast,

        method_group=method_group,
        registration_group=registration_group,
        lbl_kernel=lbl_kernel,
        lbl_halo=lbl_halo,
        lbl_depth_smooth=lbl_depth_smooth,

        source_images_label=source_images_label,
        file_list=file_list,
        btn_select_all=btn_select_all,
        btn_select_invert=btn_select_invert,
        btn_select_none=btn_select_none,
        btn_select_nth=btn_select_nth,
        spin_select_nth=spin_select_nth,
        output_label=output_label,
        output_list=output_list,
        lbl_status_loaded=lbl_status_loaded,
        lbl_status_resolution=lbl_status_resolution,
        lbl_status_gpu=lbl_status_gpu,
        lbl_status_memory=lbl_status_memory,
    )


def bind_right_panel(window, components: RightPanelComponents) -> None:
    """Connect signals for the right-panel controls using the provided window."""

    if hasattr(components.output_list, "set_window"):
        components.output_list.set_window(window)

    components.rb_a.clicked.connect(lambda: window.handle_method_selection(components.rb_a))
    components.rb_b.clicked.connect(lambda: window.handle_method_selection(components.rb_b))
    components.rb_c.clicked.connect(lambda: window.handle_method_selection(components.rb_c))
    components.rb_gfg.clicked.connect(lambda: window.handle_method_selection(components.rb_gfg))
    components.rb_pyramid.clicked.connect(lambda: window.handle_method_selection(components.rb_pyramid))
    components.rb_dmap_max.clicked.connect(lambda: window.handle_method_selection(components.rb_dmap_max))
    components.rb_dmap_avg.clicked.connect(lambda: window.handle_method_selection(components.rb_dmap_avg))
    components.rb_d.clicked.connect(lambda: window.handle_method_selection(components.rb_d))

    components.btn_method_help.clicked.connect(lambda: RenderMethodHelpDialog(window).exec())
    components.btn_reg_help.clicked.connect(lambda: RegistrationHelpDialog(window).exec())

    components.btn_render.clicked.connect(window.render_manager.start_render)
    components.btn_stop.clicked.connect(window.render_manager.stop_render)
    components.btn_reset.clicked.connect(window.reset_to_default)

    components.slider_smooth.valueChanged.connect(window.handle_kernel_slider_change)
    components.slider_halo.valueChanged.connect(window.handle_halo_slider_change)
    components.slider_depth_smooth.valueChanged.connect(
        window.handle_depth_smooth_slider_change)

    # Contrast is a post-fusion output step, so both controls just re-apply it
    # to the already-rendered result via handle_contrast_change - no re-render.
    components.combo_contrast.currentIndexChanged.connect(window.handle_contrast_change)
    components.slider_contrast.valueChanged.connect(window.handle_contrast_change)

    components.rb_a.clicked.connect(window.update_slider_availability)
    components.rb_b.clicked.connect(window.update_slider_availability)
    components.rb_c.clicked.connect(window.update_slider_availability)
    components.rb_gfg.clicked.connect(window.update_slider_availability)
    components.rb_pyramid.clicked.connect(window.update_slider_availability)
    components.rb_dmap_max.clicked.connect(window.update_slider_availability)
    components.rb_dmap_avg.clicked.connect(window.update_slider_availability)
    components.rb_d.clicked.connect(window.update_slider_availability)

    components.file_list.customContextMenuRequested.connect(window.source_manager.show_source_context_menu)
    components.file_list.currentRowChanged.connect(window.source_manager.sync_slider_from_list)
    components.file_list.itemChanged.connect(window.source_manager.handle_source_item_changed)

    components.btn_select_all.clicked.connect(window.source_manager.check_all_sources)
    components.btn_select_invert.clicked.connect(window.source_manager.invert_source_checks)
    components.btn_select_none.clicked.connect(window.source_manager.uncheck_all_sources)
    components.btn_select_nth.clicked.connect(window.source_manager.check_every_nth_source)
    components.output_list.customContextMenuRequested.connect(window.output_manager.show_output_context_menu)
    components.output_list.currentRowChanged.connect(window.output_manager.sync_output_slider_from_list)
    components.output_list.itemClicked.connect(window.output_manager.display_output_image_in_result_view)

    delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.file_list)
    delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_shortcut.activated.connect(window.source_manager.delete_selected_source_images)
    components.file_list._delete_shortcut = delete_shortcut  # keep reference alive

    delete_output_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.output_list)
    delete_output_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_output_shortcut.activated.connect(window.output_manager.delete_selected_output_images)
    components.output_list._delete_shortcut = delete_output_shortcut

    window.update_slider_availability()

    components.cb_align_scale.installEventFilter(window)
    components.cb_align_ecc.installEventFilter(window)
    components.cb_align_homography.installEventFilter(window)
