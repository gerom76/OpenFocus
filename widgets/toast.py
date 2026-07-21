"""Non-modal toast notifications used in place of blocking message boxes.

A toast is a small frameless panel that slides into the bottom-right corner of
the application window, stays for a few seconds and then fades away on its own.
Multiple toasts stack upwards; clicking one dismisses it immediately.
"""

from typing import List, Optional

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


# Default lifetime of a toast, in milliseconds
TOAST_DURATION_MS = 5000

# Accent color and glyph per message severity
_ICON_LOOK = {
    QMessageBox.Icon.NoIcon: ("#5a5a5a", ""),
    QMessageBox.Icon.Information: ("#0a84ff", "i"),
    QMessageBox.Icon.Question: ("#0a84ff", "?"),
    QMessageBox.Icon.Warning: ("#ffab2e", "!"),
    QMessageBox.Icon.Critical: ("#ff4d4f", "x"),
}

# Margin from the anchor window edges and gap between stacked toasts
_EDGE_MARGIN = 24
_STACK_GAP = 10

# Currently visible toasts, bottom-most first
_active_toasts: List["Toast"] = []


def _anchor_window(parent: Optional[QWidget]) -> Optional[QWidget]:
    """Pick the window a toast should be positioned against."""
    if parent is not None:
        window = parent.window()
        if window is not None and window.isVisible():
            return window

    active = QApplication.activeWindow()
    if active is not None and active.isVisible():
        return active

    for widget in QApplication.topLevelWidgets():
        if widget.isWindow() and widget.isVisible():
            return widget
    return None


class Toast(QWidget):
    """A single auto-dismissing notification panel."""

    def __init__(
        self,
        parent: Optional[QWidget],
        title: str,
        text: str,
        informative_text: str = "",
        icon: QMessageBox.Icon = QMessageBox.Icon.Information,
        duration_ms: int = TOAST_DURATION_MS,
    ) -> None:
        super().__init__(None)
        self._anchor = _anchor_window(parent)
        self._duration_ms = duration_ms
        self._closing = False

        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        color, glyph = _ICON_LOOK.get(icon, _ICON_LOOK[QMessageBox.Icon.Information])
        self._build_ui(title, text, informative_text, color, glyph)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)

    def _build_ui(
        self,
        title: str,
        text: str,
        informative_text: str,
        color: str,
        glyph: str,
    ) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        frame = QWidget(self)
        frame.setObjectName("toastFrame")
        frame.setStyleSheet(
            f"""
            QWidget#toastFrame {{
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-left: 4px solid {color};
                border-radius: 6px;
            }}
            QLabel {{ background: transparent; }}
            QLabel#toastBadge {{
                color: {color};
                font-size: 18px;
                font-weight: bold;
            }}
            QLabel#toastTitle {{
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
            }}
            QLabel#toastText {{ color: #e0e0e0; font-size: 12px; }}
            QLabel#toastInfo {{ color: #a8a8a8; font-size: 11px; }}
            """
        )
        outer.addWidget(frame)

        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 12, 16, 12)
        row.setSpacing(12)

        if glyph:
            badge = QLabel(glyph, frame)
            badge.setObjectName("toastBadge")
            badge.setFixedWidth(16)
            badge.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            row.addWidget(badge)

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        row.addLayout(column)

        for name, content in (
            ("toastTitle", title),
            ("toastText", text),
            ("toastInfo", informative_text),
        ):
            if not content:
                continue
            label = QLabel(str(content), frame)
            label.setObjectName(name)
            label.setWordWrap(True)
            column.addWidget(label)

        self.setFixedWidth(380)
        self.adjustSize()

    # --- lifecycle -----------------------------------------------------

    def show_toast(self) -> None:
        _active_toasts.append(self)
        self.show()
        _reposition_toasts()

        self._fade.stop()
        self._fade.setDuration(180)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

        self._timer.start(self._duration_ms)

    def dismiss(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._timer.stop()

        self._fade.stop()
        self._fade.setDuration(220)
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.close)
        self._fade.start()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.dismiss()
        super().mousePressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self in _active_toasts:
            _active_toasts.remove(self)
            _reposition_toasts()
        self.deleteLater()
        super().closeEvent(event)


def _reposition_toasts() -> None:
    """Stack the visible toasts upwards from the bottom-right corner."""
    offset = 0
    for toast in _active_toasts:
        anchor = toast._anchor
        if anchor is not None and anchor.isVisible():
            rect = anchor.frameGeometry()
        else:
            screen = QApplication.primaryScreen()
            if screen is None:
                continue
            rect = screen.availableGeometry()

        x = rect.right() - toast.width() - _EDGE_MARGIN
        y = rect.bottom() - toast.height() - _EDGE_MARGIN - offset
        toast.move(QPoint(int(x), int(y)))
        offset += toast.height() + _STACK_GAP


def show_toast(
    parent: Optional[QWidget],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
    duration_ms: int = TOAST_DURATION_MS,
) -> Toast:
    """Create, show and return an auto-dismissing toast notification."""
    toast = Toast(parent, title, text, informative_text, icon, duration_ms)
    toast.show_toast()
    return toast
