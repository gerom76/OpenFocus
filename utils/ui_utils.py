import os
import sys
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QMessageBox


def fit_list_rows_to_thumbnails(list_widget, min_row_height: int = 22, max_aspect: float = 5.0) -> None:
    """Pack a list view's rows so that consecutive thumbnails touch.

    Thumbnails keep their aspect ratio, so a square icon box letterboxes a
    landscape frame and the leftover box height reads as a gap between rows.
    The box is therefore shaped to the items' own aspect ratio at the height a
    row of text needs, and every row's size hint is pinned to that height - the
    item delegate would otherwise pad its hint with a focus-frame margin and
    leave a one-pixel line between images.
    """
    count = list_widget.count()
    if count == 0:
        return

    row_height = max(min_row_height, list_widget.fontMetrics().height())

    aspect = 1.0
    probe = QSize(4096, 4096)
    for row in range(count):
        item = list_widget.item(row)
        if item is None:
            continue
        icon_size = item.icon().actualSize(probe)
        if icon_size.height() > 0:
            aspect = min(icon_size.width() / icon_size.height(), max_aspect)
            break

    list_widget.setIconSize(QSize(max(1, round(row_height * aspect)), row_height))

    # The natural width is measured with the previous hints cleared, so a newly
    # added long filename can still widen the rows.
    for row in range(count):
        item = list_widget.item(row)
        if item is not None:
            item.setSizeHint(QSize())

    width = list_widget.sizeHintForColumn(0)
    for row in range(count):
        item = list_widget.item(row)
        if item is not None:
            item.setSizeHint(QSize(width, row_height))


def resource_path(*relative_parts: str) -> str:
    """Get the absolute path for resources, compatible with PyInstaller.
    
    Uses multiple project root markers to find the project root directory,
    making it robust against folder renaming. Prioritizes distinctive markers
    that are unlikely to exist in subdirectories.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Path when bundled (packaged)
        base_path = sys._MEIPASS
    else:
        # Path in the development environment
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # List of project-root identifiers (sorted by priority, most reliable first)
        # OpenFocus.spec is the most reliable, then main.py, and .git indicates the repository root
        project_root_markers = [
            'OpenFocus.spec',    # PyInstaller spec file, most reliable
            'main.py',           # Entry file
            '.git',              # Git repository directory (for git-cloned projects)
            'README.md',         # Project readme file
        ]
        
        # Keep the original location as a fallback
        original_base = base_path
        
        # Walk up to find the project root directory
        # Check the current directory first, then search upward level by level
        while True:
            # Check whether the current directory contains any project identifier
            found_marker = False
            for marker in project_root_markers:
                marker_path = os.path.join(base_path, marker)
                if os.path.exists(marker_path):
                    # Extra validation: make sure this is a reasonable project root
                    # Check for other project characteristics (such as common subdirectories)
                    if marker == 'OpenFocus.spec':
                        # OpenFocus.spec is the most reliable marker
                        found_marker = True
                        break
                    elif marker == 'main.py':
                        # Verify whether common project subdirectories exist
                        for subdir in ['utils', 'core', 'ui', 'dialogs']:
                            if os.path.isdir(os.path.join(base_path, subdir)):
                                found_marker = True
                                break
                        if found_marker:
                            break
                    elif marker == '.git':
                        # Verify whether other project files exist
                        for other_file in ['main.py', 'README.md', 'AGENTS.md']:
                            if os.path.exists(os.path.join(base_path, other_file)):
                                found_marker = True
                                break
                        if found_marker:
                            break
                    else:
                        # Other markers, use looser validation
                        found_marker = True
                        break
            
            if found_marker:
                # Found the project root, break out of the loop
                break
            elif os.path.dirname(base_path) != base_path:
                # Keep searching upward
                base_path = os.path.dirname(base_path)
            else:
                # Reached the filesystem root without finding it; use the original location as a fallback
                base_path = original_base
                break
    
    return os.path.normpath(os.path.join(base_path, *relative_parts))


# Import MESSAGE_BOX_STYLE from sibling module to avoid circular imports
import importlib.util
_styles_spec = importlib.util.spec_from_file_location("styles_module", resource_path("ui", "styles.py"))
_styles_module = importlib.util.module_from_spec(_styles_spec)
_styles_spec.loader.exec_module(_styles_module)
MESSAGE_BOX_STYLE = _styles_module.MESSAGE_BOX_STYLE


# Icon -> log level shown in the terminal / status console
_ICON_LEVELS = {
    QMessageBox.Icon.NoIcon: "INFO",
    QMessageBox.Icon.Information: "INFO",
    QMessageBox.Icon.Question: "QUESTION",
    QMessageBox.Icon.Warning: "WARNING",
    QMessageBox.Icon.Critical: "ERROR",
}


def log_message_box(
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
) -> None:
    """Mirror a dialog to the terminal (and therefore the status console).

    Every message box the user sees is timestamped to the millisecond so the
    console keeps a readable trace of when each one was raised.
    """
    level = _ICON_LEVELS.get(icon, "INFO")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    parts = [part for part in (title, text, informative_text) if part]
    message = " | ".join(str(part).replace("\n", " ") for part in parts)

    stream = sys.stderr if level == "ERROR" else sys.stdout
    print(f"[{stamp}] [{level}] {message}", file=stream, flush=True)


def exec_message_box(msg_box: QMessageBox):
    """Log a hand-built QMessageBox, then show it and return its result."""
    log_message_box(
        msg_box.windowTitle(),
        msg_box.text(),
        msg_box.informativeText(),
        msg_box.icon(),
    )
    return msg_box.exec()


def show_message_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
) -> None:
    log_message_box(title, text, informative_text, icon)
    # Notifications are non-blocking: a toast appears in the corner of the
    # window and disappears on its own after a few seconds.
    from widgets.toast import show_toast

    show_toast(parent, title, text, informative_text, icon)


def show_warning_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Warning)


def show_error_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Critical)


def show_success_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Information)


def show_custom_message_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
    style_sheet: str = MESSAGE_BOX_STYLE,
) -> None:
    # style_sheet is kept for call-site compatibility; toasts carry their own look.
    log_message_box(title, text, informative_text, icon)
    from widgets.toast import show_toast

    show_toast(parent, title, text, informative_text, icon)
