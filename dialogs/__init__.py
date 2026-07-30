# dialogs/__init__.py
"""
OpenFocus Dialogs Package

Refactored dialog module, grouped by function:
- about: About and contact-info dialogs
- help: Help-info dialogs
- settings: Settings-related dialogs
- batch: Batch-processing-related dialogs
- roi: ROI-related dialogs
"""

from dialogs.about import (
    EnvironmentInfoDialog,
    ContactInfoDialog,
)

from dialogs.help import (
    HelpDialog,
    RenderMethodHelpDialog,
    RegistrationHelpDialog,
    TileHelpDialog,
)

from dialogs.settings import (
    DurationDialog,
    ExportFormatDialog,
    DownsampleDialog,
    TileSettingsDialog,
    RegistrationSettingsDialog,
    ThreadSettingsDialog,
    DngSettingsDialog,
    StackMFFV4BatchSettingsDialog,
)

from dialogs.batch import (
    BatchProcessingDialog,
    FolderImportDialog,
)

from dialogs.roi import (
    ROIRenderOptionsDialog,
)

__all__ = [
    # About dialogs
    'EnvironmentInfoDialog',
    'ContactInfoDialog',
    # Help dialogs
    'HelpDialog',
    'RenderMethodHelpDialog',
    'RegistrationHelpDialog',
    'TileHelpDialog',
    # Settings dialogs
    'DurationDialog',
    'ExportFormatDialog',
    'DownsampleDialog',
    'TileSettingsDialog',
    'RegistrationSettingsDialog',
    'ThreadSettingsDialog',
    'DngSettingsDialog',
    'StackMFFV4BatchSettingsDialog',
    # Batch dialogs
    'BatchProcessingDialog',
    'FolderImportDialog',
    # ROI dialogs
    'ROIRenderOptionsDialog',
]