"""
Status console: a bottom docked area that mirrors the terminal output
(stdout / stderr) produced by the loading, registration and fusion pipelines.
"""

import re
import sys

from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
)

from locales import trans
from utils import get_monospace_font_family

# Keep the console bounded so long batch runs cannot grow memory without limit.
MAX_CONSOLE_LINES = 2000

# Pipeline stages print counters such as "[12/90] Loaded ..." or
# "Tiled fusion progress: 5/20 tiles". The lookarounds keep decimals
# (e.g. "14.7/16.0 GB free") from being read as progress.
PROGRESS_PATTERN = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])")


class StreamRedirector(QObject):
    """File-like object that tees writes to the original stream and to a signal.

    Worker threads print during fusion/registration, so the text is delivered
    through a Qt signal and appended on the GUI thread via a queued connection.
    """

    textWritten = pyqtSignal(str, bool)  # (text, is_error)

    def __init__(self, original, is_error: bool = False, parent=None):
        super().__init__(parent)
        self._original = original
        self._is_error = is_error

    def write(self, text):
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass
        if text:
            self.textWritten.emit(str(text), self._is_error)
        return len(text) if text else 0

    def flush(self):
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    @property
    def original(self):
        return self._original


class StatusConsole(QWidget):
    """Collapsible output area shown at the bottom of the main window."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Header bar ---
        header = QWidget()
        header.setObjectName("consoleHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 2, 4, 2)
        header_layout.setSpacing(6)

        self.title_label = QLabel(trans.t('console_title'))
        self.title_label.setStyleSheet("color: #aaa;")
        header_layout.addWidget(self.title_label)

        # Progress bar sits right next to the title; hidden while idle.
        self.progress = QProgressBar()
        self.progress.setObjectName("consoleProgress")
        self.progress.setFixedWidth(180)
        self.progress.setFixedHeight(12)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0)  # indeterminate until a counter is seen
        self.progress.hide()
        header_layout.addWidget(self.progress)

        header_layout.addStretch()

        self.btn_clear = QPushButton(trans.t('console_clear'))
        self.btn_clear.setObjectName("consoleButton")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear)
        header_layout.addWidget(self.btn_clear)

        self.btn_toggle = QPushButton("▾")  # down-pointing triangle
        self.btn_toggle.setObjectName("consoleButton")
        self.btn_toggle.setFixedWidth(24)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.setToolTip(trans.t('console_collapse'))
        self.btn_toggle.clicked.connect(self.toggle_collapsed)
        header_layout.addWidget(self.btn_toggle)

        layout.addWidget(header)

        # --- Output view ---
        self.output = QPlainTextEdit()
        self.output.setObjectName("consoleOutput")
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(MAX_CONSOLE_LINES)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = self.output.font()
        font.setFamily(get_monospace_font_family())
        font.setPointSize(9)
        self.output.setFont(font)
        layout.addWidget(self.output)

        # Keep at least 2 text rows visible so the splitter can't squeeze
        # the output view down to a sliver while it's still expanded.
        metrics = self.output.fontMetrics()
        self._output_min_height = (
            metrics.lineSpacing() * 2
            + self.output.frameWidth() * 2
            + int(self.output.document().documentMargin() * 2)
        )
        self.output.setMinimumHeight(self._output_min_height)

        self._header_height = header.sizeHint().height()
        self.setMinimumHeight(self._header_height + self._output_min_height)

        self._collapsed = False
        self._pending = ""  # Incomplete line waiting for its newline
        self._progress_active = False

        # --- Redirect stdout / stderr ---
        self._stdout_redirector = StreamRedirector(sys.stdout, is_error=False)
        self._stderr_redirector = StreamRedirector(sys.stderr, is_error=True)
        # Queued so writes from worker threads reach the GUI thread safely.
        self._stdout_redirector.textWritten.connect(
            self.append_text, Qt.ConnectionType.QueuedConnection)
        self._stderr_redirector.textWritten.connect(
            self.append_text, Qt.ConnectionType.QueuedConnection)
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    def append_text(self, text: str, is_error: bool = False):
        """Append raw stream text, honouring carriage returns and colouring errors."""
        if self._progress_active and not is_error:
            self._track_progress(text)

        self._pending += text
        if "\n" not in self._pending:
            return

        chunk, _, self._pending = self._pending.rpartition("\n")
        if not chunk:
            return

        color = "#ff7b72" if is_error else "#d0d0d0"
        scrollbar = self.output.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4

        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for line in chunk.split("\n"):
            # Progress lines use \r to overwrite; keep only the last segment.
            line = line.rpartition("\r")[2].rstrip()
            cursor.insertHtml(f'<span style="color:{color};">{_escape(line)}</span>')
            cursor.insertBlock()

        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def log(self, message: str, is_error: bool = False):
        """Write an application message to the console (also goes to the terminal)."""
        stream = sys.stderr if is_error else sys.stdout
        print(message, file=stream, flush=True)

    def clear(self):
        self._pending = ""
        self.output.clear()

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------
    def start_progress(self):
        """Show a busy bar; it becomes determinate as soon as the pipeline
        prints an "i/N" counter."""
        self._progress_active = True
        self.progress.setRange(0, 0)
        self.progress.show()

    def stop_progress(self):
        self._progress_active = False
        self.progress.hide()
        self.progress.setRange(0, 0)
        self.progress.reset()

    def set_progress(self, current: int, total: int):
        """Drive the bar explicitly (used by callers that know their totals)."""
        if total <= 0:
            self.start_progress()
            return
        self._progress_active = True
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(max(0, min(current, total)))
        self.progress.show()

    def _track_progress(self, text: str):
        """Pick up "i/N" counters printed by the pipeline stages."""
        match = None
        for match in PROGRESS_PATTERN.finditer(text):
            pass  # keep the last match in the chunk
        if match is None:
            return

        current, total = int(match.group(1)), int(match.group(2))
        if total <= 0 or current > total:
            return

        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(current)

    # ------------------------------------------------------------------
    # Collapsing
    # ------------------------------------------------------------------
    def toggle_collapsed(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self.output.setVisible(not collapsed)
        self.btn_toggle.setText("▴" if collapsed else "▾")
        self.btn_toggle.setToolTip(
            trans.t('console_expand') if collapsed else trans.t('console_collapse'))
        if collapsed:
            self.setMaximumHeight(self.sizeHint().height())
            self.setMinimumHeight(self._header_height)
        else:
            self.setMaximumHeight(16777215)
            self.setMinimumHeight(self._header_height + self._output_min_height)

    def is_collapsed(self) -> bool:
        return self._collapsed

    # ------------------------------------------------------------------
    def update_ui_text(self):
        self.title_label.setText(trans.t('console_title'))
        self.btn_clear.setText(trans.t('console_clear'))
        self.btn_toggle.setToolTip(
            trans.t('console_expand') if self._collapsed else trans.t('console_collapse'))

    def restore_streams(self):
        """Put the original stdout/stderr back (called on window close)."""
        if sys.stdout is self._stdout_redirector:
            sys.stdout = self._stdout_redirector.original
        if sys.stderr is self._stderr_redirector:
            sys.stderr = self._stderr_redirector.original


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&nbsp;"))
