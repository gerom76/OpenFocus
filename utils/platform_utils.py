"""
Cross-platform utility module - handles Windows/macOS/Linux compatibility
"""

import sys
import platform


def get_os_type() -> str:
    """
    Get the current operating-system type
    
    Returns:
        'windows' | 'macos' | 'linux'
    """
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    else:
        return 'unknown'


def get_ui_font_family() -> str:
    """
    Get the UI font family suitable for the current OS
    
    Returns:
        Font-family name string
    """
    os_type = get_os_type()
    
    if os_type == 'macos':
        # macOS system font
        return '"SF Pro", "Helvetica Neue", Arial, sans-serif'
    elif os_type == 'windows':
        # Windows system font
        return '"Segoe UI", "Microsoft YaHei", Arial, sans-serif'
    else:
        # Linux system font
        return '"Ubuntu", "DejaVu Sans", Arial, sans-serif'


def get_monospace_font_family() -> str:
    """
    Get the monospace font family suitable for the current OS
    
    Returns:
        Font-family name string
    """
    os_type = get_os_type()
    
    if os_type == 'macos':
        # macOS monospace font
        return '"SF Mono", "Monaco", "Menlo", Consolas, monospace'
    elif os_type == 'windows':
        # Windows monospace font
        return 'Consolas, "Courier New", monospace'
    else:
        # Linux monospace font
        return '"Ubuntu Mono", "DejaVu Sans Mono", Consolas, monospace'


def get_default_font() -> str:
    """
    Get the default font suitable for the current OS
    
    Returns:
        Font-name string
    """
    os_type = get_os_type()
    
    if os_type == 'macos':
        return 'SF Pro'
    elif os_type == 'windows':
        return 'Microsoft YaHei'
    else:
        return 'Ubuntu'


def is_windows() -> bool:
    """Check whether the OS is Windows"""
    return get_os_type() == 'windows'


def is_macos() -> bool:
    """Check whether the OS is macOS"""
    return get_os_type() == 'macos'


def is_linux() -> bool:
    """Check whether the OS is Linux"""
    return get_os_type() == 'linux'


# Export commonly used functions
__all__ = [
    'get_os_type',
    'get_ui_font_family',
    'get_monospace_font_family',
    'get_default_font',
    'is_windows',
    'is_macos',
    'is_linux',
]
