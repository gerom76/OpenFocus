import json
import os
import sys
from typing import Any

from utils import show_error_box, show_success_box
from locales import trans

CONFIG_FILENAME = "openfocus.cfg.json"


def _config_path() -> str:
    """Return the absolute path to the settings file next to the app/executable."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        # controllers/ -> project root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, CONFIG_FILENAME)


class SettingsManager:
    """Persist and restore all user-configurable settings to a JSON file."""

    def __init__(self, window: Any) -> None:
        self.window = window

    # --- Serialization ---

    def _selected_fusion_method(self) -> str | None:
        window = self.window
        mapping = [
            (getattr(window, "rb_a", None), "guided"),
            (getattr(window, "rb_b", None), "dct"),
            (getattr(window, "rb_c", None), "dtcwt"),
            (getattr(window, "rb_gfg", None), "gfg"),
            (getattr(window, "rb_d", None), "stackmff"),
        ]
        for button, name in mapping:
            if button is not None and button.isChecked():
                return name
        return None

    def collect_settings(self) -> dict:
        """Gather the current settings from the window into a serializable dict."""
        window = self.window
        return {
            "language": trans.current_lang,
            "tile_enabled": window.tile_enabled,
            "tile_block_size": window.tile_block_size,
            "tile_overlap": window.tile_overlap,
            "tile_threshold": window.tile_threshold,
            "reg_downscale_width": window.reg_downscale_width,
            "ecc_parallel": window.ecc_parallel,
            "thread_count": window.thread_count,
            "stackmffv4_batch_size": window.stackmffv4_batch_size,
            "fusion_method": self._selected_fusion_method(),
            "align_homography": window.cb_align_homography.isChecked(),
            "align_ecc": window.cb_align_ecc.isChecked(),
            "smooth_kernel": window.slider_smooth.value(),
            "show_status_console": getattr(window, "status_console", None) is not None
                                   and window.status_console.isVisible(),
        }

    # --- Save ---

    def save_all_settings(self, silent: bool = False) -> bool:
        """Write all current settings to the config file. Returns True on success."""
        window = self.window
        path = _config_path()
        try:
            settings = self.collect_settings()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            show_error_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_settings_save_failed_text"),
                str(exc),
            )
            return False

        if not silent:
            show_success_box(
                window,
                trans.t("msg_settings_saved_title"),
                trans.t("msg_settings_saved_text"),
                trans.t("msg_settings_saved_info").format(path=path),
            )
        return True

    # --- Load ---

    def load_all_settings(self) -> bool:
        """Load settings from the config file and apply them. Returns True if applied."""
        path = _config_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return False

        if not isinstance(data, dict):
            return False

        self._apply_settings(data)
        return True

    def _reset_registration_and_fusion(self) -> None:
        """Clear registration and fusion selections to a clean state.

        Applied before restoring saved values so stale selections (e.g. a fusion
        method saved as None, or a radio checked in the current session but absent
        from the config) do not linger after loading.
        """
        window = self.window

        # Fusion method radios (mutually exclusive, cancelable) -> all unchecked
        for attr in ("rb_a", "rb_b", "rb_c", "rb_gfg", "rb_d"):
            button = getattr(window, attr, None)
            if button is not None:
                button.setChecked(False)

        # Registration option checkboxes -> unchecked
        for attr in ("cb_align_homography", "cb_align_ecc"):
            checkbox = getattr(window, attr, None)
            if checkbox is not None:
                checkbox.setChecked(False)

    def _apply_settings(self, data: dict) -> None:
        window = self.window

        # Start from a clean registration/fusion UI so only saved values take effect
        self._reset_registration_and_fusion()

        # Simple scalar settings written straight back onto the window
        for attr in (
            "tile_enabled",
            "tile_block_size",
            "tile_overlap",
            "tile_threshold",
            "reg_downscale_width",
            "ecc_parallel",
            "thread_count",
            "stackmffv4_batch_size",
        ):
            if attr in data:
                setattr(window, attr, data[attr])

        # Language
        lang = data.get("language")
        if lang:
            trans.set_language(lang)

        # Registration checkboxes
        if "align_homography" in data:
            window.cb_align_homography.setChecked(bool(data["align_homography"]))
        if "align_ecc" in data:
            window.cb_align_ecc.setChecked(bool(data["align_ecc"]))

        # Fusion method radio buttons
        method = data.get("fusion_method")
        method_buttons = {
            "guided": getattr(window, "rb_a", None),
            "dct": getattr(window, "rb_b", None),
            "dtcwt": getattr(window, "rb_c", None),
            "gfg": getattr(window, "rb_gfg", None),
            "stackmff": getattr(window, "rb_d", None),
        }
        if method in method_buttons and method_buttons[method] is not None:
            button = method_buttons[method]
            if button.isEnabled():
                button.setChecked(True)

        # Sync slider enablement/kernel mode to the selected method, then restore the
        # saved kernel value last so it is not overwritten by the per-method default.
        # Status console visibility (drives the View menu action, which hides the panel)
        if "show_status_console" in data and hasattr(window, "action_show_console"):
            window.action_show_console.setChecked(bool(data["show_status_console"]))

        window.update_slider_availability()
        if "smooth_kernel" in data:
            try:
                window.slider_smooth.setValue(int(data["smooth_kernel"]))
            except (TypeError, ValueError):
                pass
