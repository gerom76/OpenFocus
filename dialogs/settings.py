# dialogs/settings.py
"""Settings and configuration dialogs."""

import os
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QSpinBox,
    QSlider,
    QPushButton,
    QRadioButton,
)
from ui.styles import PRIMARY_BLUE
from locales import trans
from utils import resource_path


class DurationDialog(QDialog):
    """GIF Duration settings dialog (extracted from main.py)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_duration_title"))
        self.resize(350, 150)
        self.duration = 500  # Default 500 ms

        # Apply the dark theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 30px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 30px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: #555;
            }}
            QSpinBox::up-arrow, QSpinBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QSpinBox::up-arrow:disabled, QSpinBox::down-button:disabled {{
                image: none;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
            QGroupBox {{
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: normal;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)

        layout = QVBoxLayout(self)

        # Create the group box
        duration_group = QGroupBox(trans.t("dialog_duration_group"))
        duration_layout = QHBoxLayout()

        # Label
        label = QLabel(trans.t("dialog_duration_label"))
        label.setMinimumWidth(120)

        # Spin box
        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setRange(50, 10000)  # 50 ms to 10 s
        self.duration_spinbox.setValue(self.duration)
        self.duration_spinbox.setSingleStep(50)  # Increase by 50 ms per step
        self.duration_spinbox.setSuffix(" ms")
        self.duration_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.duration_spinbox.setMinimumHeight(30)
        self.duration_spinbox.setMinimumWidth(150)

        duration_layout.addWidget(label)
        duration_layout.addWidget(self.duration_spinbox)
        duration_group.setLayout(duration_layout)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton("OK")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addWidget(duration_group)
        layout.addLayout(button_layout)

    def get_duration(self):
        """Return the duration value set by the user (milliseconds)"""
        return self.duration_spinbox.value()


class DownsampleDialog(QDialog):
    """Downsampling settings dialog"""

    def __init__(self, parent=None, initial_scale=1.0):
        super().__init__(parent)
        self.setWindowTitle(trans.t('ds_title'))
        self.resize(400, 150)
        self.scale_percent = int(initial_scale * 100)

        # Apply the dark theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid #333;
                height: 6px;
                background: #202020;
                margin: 2px 0;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: #888;
                border: 1px solid #555;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #aaa;
            }}
            QSlider::sub-page:horizontal {{
                background: {PRIMARY_BLUE};
                border-radius: 3px;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 30px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        # Description text
        info_label = QLabel(trans.t('ds_label'))
        layout.addWidget(info_label)

        # Widget layout
        controls_layout = QHBoxLayout()

        # Decrease button
        self.decrease_btn = QPushButton("-")
        self.decrease_btn.setFixedSize(30, 30)
        self.decrease_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.decrease_btn.setAutoRepeat(True)  # Enable long-press repeat
        self.decrease_btn.setAutoRepeatDelay(300)  # Long-press delay
        self.decrease_btn.setAutoRepeatInterval(50)  # Repeat interval
        self.decrease_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        
        # Small button style
        btn_style = """
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 0px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """
        self.decrease_btn.setStyleSheet(btn_style)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(self.scale_percent)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(10)

        # Increase button
        self.increase_btn = QPushButton("+")
        self.increase_btn.setFixedSize(30, 30)
        self.increase_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.increase_btn.setAutoRepeat(True)  # Enable long-press repeat
        self.increase_btn.setAutoRepeatDelay(300)  # Long-press delay
        self.increase_btn.setAutoRepeatInterval(50)  # Repeat interval
        self.increase_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.increase_btn.setStyleSheet(btn_style)

        # Spin box
        self.spinbox = QSpinBox()
        self.spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)  # Hide the built-in buttons
        self.spinbox.setRange(1, 100)
        self.spinbox.setValue(self.scale_percent)
        self.spinbox.setSuffix("%")
        self.spinbox.setFixedWidth(60)

        # Connect signals
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)

        controls_layout.addWidget(self.decrease_btn)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.increase_btn)
        controls_layout.addWidget(self.spinbox)
        layout.addLayout(controls_layout)

        # Hint text
        hint_label = QLabel(trans.t('ds_hint'))
        hint_label.setStyleSheet("color: #aaa; font-size: 11px; font-style: italic;")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton(trans.t('btn_ok'))
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton(trans.t('btn_cancel'))
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def get_scale_factor(self):
        """Return the scale factor (0.0 - 1.0)"""
        return self.slider.value() / 100.0


class TileSettingsDialog(QDialog):
    """Dialog for user-customized Tile settings"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_tile_title"))
        self.resize(420, 260)

        # Apply the dark style consistent with the other dialogs
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 28px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        # Group for tile options
        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_tile_group"))
        g_layout = QVBoxLayout(group)

        # tile_enabled (radio buttons)
        enabled_layout = QHBoxLayout()
        enabled_label = QLabel(trans.t("dialog_tile_enabled_label"))
        enabled_layout.addWidget(enabled_label)
        self.rb_enabled = QRadioButton(trans.t("dialog_tile_enabled"))
        self.rb_disabled = QRadioButton(trans.t("dialog_tile_disabled"))
        enabled_layout.addWidget(self.rb_enabled)
        enabled_layout.addWidget(self.rb_disabled)
        enabled_layout.addStretch()
        g_layout.addLayout(enabled_layout)

        # tile_block_size
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel(trans.t("dialog_tile_block_size")))
        self.spin_block = QSpinBox()
        self.spin_block.setRange(64, 16384)
        self.spin_block.setSingleStep(1)
        self.spin_block.setValue(1024)
        # Remove the spin buttons on the right so the user can type directly or adjust with the keyboard/slider
        self.spin_block.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        block_layout.addWidget(self.spin_block)
        block_layout.addStretch()
        g_layout.addLayout(block_layout)

        # tile_overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel(trans.t("dialog_tile_overlap")))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 4096)
        self.spin_overlap.setSingleStep(1)
        self.spin_overlap.setValue(256)
        # Remove the spin buttons on the right
        self.spin_overlap.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        overlap_layout.addWidget(self.spin_overlap)
        overlap_layout.addStretch()
        g_layout.addLayout(overlap_layout)

        # tile_threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel(trans.t("dialog_tile_threshold")))
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(256, 131072)
        self.spin_threshold.setSingleStep(1)
        self.spin_threshold.setValue(2048)
        # Remove the spin buttons on the right
        self.spin_threshold.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        threshold_layout.addWidget(self.spin_threshold)
        threshold_layout.addStretch()
        g_layout.addLayout(threshold_layout)

        layout.addWidget(group)

        # Buttons: help and OK/Cancel
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for tile settings")
        help_btn.setFixedSize(26, 26)
        help_btn.setIcon(QIcon(resource_path('assets', 'help_white.svg')))
        help_btn.setIconSize(QSize(18, 18))
        help_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
        )
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()

        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        # load defaults from parent if available
        self.load_defaults()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "tile_enabled", True)
            if val:
                self.rb_enabled.setChecked(True)
            else:
                self.rb_disabled.setChecked(True)

            self.spin_block.setValue(getattr(self.parent_window, "tile_block_size", 1024))
            self.spin_overlap.setValue(getattr(self.parent_window, "tile_overlap", 256))
            self.spin_threshold.setValue(getattr(self.parent_window, "tile_threshold", 2048))
        else:
            self.rb_enabled.setChecked(True)

    def show_help(self):
        from dialogs.help import HelpDialog
        help_text = trans.t("dialog_tile_help_text")
        dlg = HelpDialog(trans.t("dialog_tile_help_title"), help_text, parent=self)
        dlg.exec()

    def on_accept(self):
        enabled = True if self.rb_enabled.isChecked() else False
        bsize = int(self.spin_block.value())
        overlap = int(self.spin_overlap.value())
        thr = int(self.spin_threshold.value())

        if self.parent_window:
            setattr(self.parent_window, "tile_enabled", enabled)
            setattr(self.parent_window, "tile_block_size", bsize)
            setattr(self.parent_window, "tile_overlap", overlap)
            setattr(self.parent_window, "tile_threshold", thr)

        self.accept()


class RegistrationSettingsDialog(QDialog):
    """Dialog to configure registration downscale_width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_reg_title"))
        self.resize(360, 180)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 28px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox, QCheckBox

        group = QGroupBox(trans.t("dialog_reg_group"))
        g_layout = QVBoxLayout(group)

        downscale_layout = QHBoxLayout()
        lbl = QLabel(trans.t("dialog_reg_downscale"))
        lbl.setMinimumWidth(120)
        downscale_layout.addWidget(lbl)

        self.spin_downscale = QSpinBox()
        self.spin_downscale.setRange(256, 8192)
        self.spin_downscale.setSingleStep(1)
        # Default value is set in load_defaults
        self.spin_downscale.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_downscale.setValue(1024)
        downscale_layout.addWidget(self.spin_downscale)
        downscale_layout.addStretch()
        g_layout.addLayout(downscale_layout)

        self.cb_parallel_ecc = QCheckBox(trans.t("dialog_reg_parallel_ecc"))
        self.cb_parallel_ecc.setChecked(True)
        g_layout.addWidget(self.cb_parallel_ecc)

        # Reference frame: which frame stays fixed while the others align onto it.
        ref_layout = QHBoxLayout()
        ref_label = QLabel(trans.t("dialog_reg_reference"))
        ref_label.setMinimumWidth(120)
        ref_layout.addWidget(ref_label)
        self.rb_ref_first = QRadioButton(trans.t("dialog_reg_reference_first"))
        self.rb_ref_middle = QRadioButton(trans.t("dialog_reg_reference_middle"))
        self.rb_ref_last = QRadioButton(trans.t("dialog_reg_reference_last"))
        self.rb_ref_first.setChecked(True)
        ref_layout.addWidget(self.rb_ref_first)
        ref_layout.addWidget(self.rb_ref_middle)
        ref_layout.addWidget(self.rb_ref_last)
        ref_layout.addStretch()
        g_layout.addLayout(ref_layout)

        layout.addWidget(group)

        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for registration settings")
        help_btn.setFixedSize(26, 26)
        help_btn.setIcon(QIcon(resource_path('assets', 'help_white.svg')))
        help_btn.setIconSize(QSize(18, 18))
        help_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
        )
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        self.load_defaults()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "reg_downscale_width", 1024)
            try:
                self.spin_downscale.setValue(int(val))
            except Exception:
                self.spin_downscale.setValue(1024)
            self.cb_parallel_ecc.setChecked(bool(getattr(self.parent_window, "ecc_parallel", True)))
            mode = getattr(self.parent_window, "reference_frame_mode", "first")
            if mode == "middle":
                self.rb_ref_middle.setChecked(True)
            elif mode == "last":
                self.rb_ref_last.setChecked(True)
            else:
                self.rb_ref_first.setChecked(True)

    def on_accept(self):
        val = int(self.spin_downscale.value())
        if self.parent_window:
            setattr(self.parent_window, "reg_downscale_width", val)
            setattr(self.parent_window, "ecc_parallel", self.cb_parallel_ecc.isChecked())
            if self.rb_ref_middle.isChecked():
                reference_mode = "middle"
            elif self.rb_ref_last.isChecked():
                reference_mode = "last"
            else:
                reference_mode = "first"
            setattr(self.parent_window, "reference_frame_mode", reference_mode)
        self.accept()

    def show_help(self):
        from dialogs.help import HelpDialog
        help_text = trans.t("dialog_reg_help_text")
        dlg = HelpDialog(trans.t("dialog_reg_help_title"), help_text, parent=self)
        dlg.exec()


class ThreadSettingsDialog(QDialog):
    """Thread count settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_thread_title"))
        self.resize(420, 160)

        # Apply the same dark dialog styling as TileSettingsDialog
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 28px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_thread_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_thread_label"))
        lbl.setMinimumWidth(140)
        g_layout.addWidget(lbl)

        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 256)
        self.spin_threads.setSingleStep(1)
        self.spin_threads.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        g_layout.addWidget(self.spin_threads)
        g_layout.addStretch()

        layout.addWidget(group)

        # Buttons: help and OK/Cancel (match Tile/Registration style)
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for application settings")
        help_btn.setFixedSize(26, 26)
        help_btn.setIcon(QIcon(resource_path('assets', 'help_white.svg')))
        help_btn.setIconSize(QSize(18, 18))
        help_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
        )
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()

        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        self.load_defaults()

    def show_help(self):
        from dialogs.help import HelpDialog
        help_text = trans.t("dialog_thread_help_text")
        dlg = HelpDialog(trans.t("dialog_thread_help_title"), help_text, parent=self)
        dlg.exec()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "thread_count", 4)
            try:
                self.spin_threads.setValue(int(val))
            except Exception:
                self.spin_threads.setValue(4)

    def on_accept(self):
        val = int(self.spin_threads.value())
        if self.parent_window:
            setattr(self.parent_window, "thread_count", val)
        self.accept()


class StackMFFV4BatchSettingsDialog(QDialog):
    """StackMFF V4 Batch Size settings dialog for tile batch processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_stackmffv4_batch_title"))
        self.resize(420, 160)

        # Apply the same dark dialog styling as other settings dialogs
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 28px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_stackmffv4_batch_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_stackmffv4_batch_label"))
        lbl.setMinimumWidth(140)
        g_layout.addWidget(lbl)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 16)
        self.spin_batch_size.setSingleStep(1)
        self.spin_batch_size.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        g_layout.addWidget(self.spin_batch_size)
        g_layout.addStretch()

        layout.addWidget(group)

        # Buttons: help and OK/Cancel
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for StackMFF V4 batch settings")
        help_btn.setFixedSize(26, 26)
        help_btn.setIcon(QIcon(resource_path('assets', 'help_white.svg')))
        help_btn.setIconSize(QSize(18, 18))
        help_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
        )
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()

        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        self.load_defaults()

    def show_help(self):
        from dialogs.help import HelpDialog
        help_text = trans.t("dialog_stackmffv4_batch_help_text")
        dlg = HelpDialog(trans.t("dialog_stackmffv4_batch_help_title"), help_text, parent=self)
        dlg.exec()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "stackmffv4_batch_size", 2)
            try:
                self.spin_batch_size.setValue(int(val))
            except Exception:
                self.spin_batch_size.setValue(2)
        else:
            self.spin_batch_size.setValue(2)

    def on_accept(self):
        val = int(self.spin_batch_size.value())
        if self.parent_window:
            setattr(self.parent_window, "stackmffv4_batch_size", val)
        self.accept()