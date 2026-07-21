import os
import sys
from typing import Optional

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QPixmap, QImage


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


def show_message_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
) -> None:
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(MESSAGE_BOX_STYLE)
    msg_box.exec()


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
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(style_sheet)
    msg_box.exec()
