import json
import os
import sys
from typing import Any

from utils import show_error_box, show_success_box, bitdepth
from locales import trans

CONFIG_FILENAME = "openfocus.cfg.json"

# How many entries the File > Recent submenus keep.
MAX_RECENT_PATHS = 10

# Saved fusion_method value -> the window attribute holding its radio button.
# Single source of truth for saving, restoring and clearing the selection, so a
# newly added method cannot end up persisted as null by being listed in only one
# of the three places.
_FUSION_METHOD_BUTTONS = {
    "guided": "rb_a",
    "dct": "rb_b",
    "dtcwt": "rb_c",
    "gfg": "rb_gfg",
    "pyramid": "rb_pyramid",
    "depthmap_max": "rb_dmap_max",
    "depthmap_avg": "rb_dmap_avg",
    "stackmff": "rb_d",
}


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
        self.recent_folders: list[str] = []
        self.recent_videos: list[str] = []
        self.output_dir: str = ""
        self.output_format: str = ""

    # --- Recently opened paths ---

    def add_recent_folder(self, path: str) -> None:
        """Remember an image-stack folder that was opened successfully."""
        self._add_recent(self.recent_folders, path)

    def add_recent_video(self, path: str) -> None:
        """Remember a video file that was opened successfully."""
        self._add_recent(self.recent_videos, path)

    def clear_recent_folders(self) -> None:
        self.recent_folders.clear()
        self._persist_recent()

    def clear_recent_videos(self) -> None:
        self.recent_videos.clear()
        self._persist_recent()

    # --- Output folder ---

    def set_output_dir(self, path: str) -> None:
        """Remember the folder the user last exported an image to."""
        if not path:
            return
        if not os.path.isdir(path):
            path = os.path.dirname(path)
        if not path or not os.path.isdir(path):
            return

        path = os.path.normpath(path)
        if os.path.normcase(path) == os.path.normcase(self.output_dir):
            return

        self.output_dir = path
        self._persist_recent()

    # --- Output image format ---

    def set_output_format(self, extension: str) -> None:
        """Remember the image format the user last exported in, e.g. '.jxl'.

        Stored as a bare extension; the export code validates it against the
        formats this build can actually write before offering it again.
        """
        extension = self._sanitize_format(extension)
        if not extension or extension == self.output_format:
            return

        self.output_format = extension
        self._persist_recent()

    @staticmethod
    def _sanitize_format(value: Any) -> str:
        """Accept only a plain lower-case file extension from config or callers."""
        if not isinstance(value, str) or not value.strip():
            return ""
        ext = value.strip().lower().lstrip(".")
        if not ext.isalnum() or len(ext) > 5:
            return ""
        return "." + ext

    def default_output_path(self, filename: str) -> str:
        """Prefix a suggested filename with the remembered output folder."""
        if self.output_dir and os.path.isdir(self.output_dir):
            return os.path.join(self.output_dir, filename)
        return filename

    def last_folder_dir(self) -> str:
        """Directory the Open Folder dialog should start in."""
        for path in self.recent_folders:
            if os.path.isdir(path):
                return path
        return ""

    def last_video_dir(self) -> str:
        """Directory the Open Video dialog should start in."""
        for path in self.recent_videos:
            parent = os.path.dirname(path)
            if os.path.isdir(parent):
                return parent
        return ""

    def _add_recent(self, entries: list[str], path: str) -> None:
        if not path:
            return
        path = os.path.normpath(path)

        # Case-insensitive de-duplication so Windows paths do not pile up.
        key = os.path.normcase(path)
        entries[:] = [p for p in entries if os.path.normcase(p) != key]
        entries.insert(0, path)
        del entries[MAX_RECENT_PATHS:]

        self._persist_recent()

    def _persist_recent(self) -> None:
        """Write only the remembered paths/format back, leaving other settings alone.

        Recent paths are recorded as soon as a stack loads, which must not
        silently overwrite settings the user has not chosen to save.
        """
        path = _config_path()
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}

        data["recent_folders"] = list(self.recent_folders)
        data["recent_videos"] = list(self.recent_videos)
        data["output_dir"] = self.output_dir
        data["output_format"] = self.output_format

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception:
            # Failing to record history must never interrupt loading a stack.
            pass

        refresh = getattr(self.window, "refresh_recent_menus", None)
        if callable(refresh):
            refresh()

    # --- Serialization ---

    def _selected_fusion_method(self) -> str | None:
        window = self.window
        for name, attr in _FUSION_METHOD_BUTTONS.items():
            button = getattr(window, attr, None)
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
            "reference_frame_mode": getattr(window, "reference_frame_mode", "first"),
            "thread_count": window.thread_count,
            "stackmffv4_batch_size": window.stackmffv4_batch_size,
            "gpu_loading": getattr(window, "gpu_loading_enabled", True),
            "bit_depth_mode": bitdepth.get_mode(),
            "contrast_method": getattr(window, "contrast_method", "off"),
            "contrast_strength": getattr(window, "contrast_strength", 50),
            "fusion_method": self._selected_fusion_method(),
            "ifcnn_refine": window.cb_ifcnn.isChecked(),
            "align_scale": window.cb_align_scale.isChecked(),
            "align_homography": window.cb_align_homography.isChecked(),
            "align_ecc": window.cb_align_ecc.isChecked(),
            "smooth_kernel": window.slider_smooth.value(),
            "halo_radius": window.slider_halo.value(),
            "dct_block_size": window.combo_dct_block.currentData(),
            "dct_plateau": window.combo_dct_plateau.currentData(),
            "dct_blend": window.cb_dct_blend.isChecked(),
            "pyramid_levels": window.combo_pyr_levels.currentData(),
            "pyramid_selectivity": window.combo_pyr_selectivity.currentData(),
            "pyramid_coherence": window.combo_pyr_coherence.currentData(),
            "pyramid_base": window.combo_pyr_base.currentData(),
            "pyramid_noise_gate": window.cb_pyr_noise_gate.isChecked(),
            "pyramid_envelope": window.cb_pyr_envelope.isChecked(),
            "show_status_console": getattr(window, "status_console", None) is not None
                                   and window.status_console.isVisible(),
            "show_source_stack": getattr(window, "action_show_source_stack", None) is None
                                 or window.action_show_source_stack.isChecked(),
            "recent_folders": list(self.recent_folders),
            "recent_videos": list(self.recent_videos),
            "output_dir": self.output_dir,
            "output_format": self.output_format,
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
        for attr in _FUSION_METHOD_BUTTONS.values():
            button = getattr(window, attr, None)
            if button is not None:
                button.setChecked(False)

        # Registration option and refinement stage checkboxes -> unchecked
        for attr in ("cb_align_scale", "cb_align_homography", "cb_align_ecc", "cb_ifcnn"):
            checkbox = getattr(window, attr, None)
            if checkbox is not None:
                checkbox.setChecked(False)

    @staticmethod
    def _sanitize_recent(value: Any) -> list[str]:
        """Accept only a list of non-empty strings from the config file."""
        if not isinstance(value, list):
            return []
        return [os.path.normpath(item) for item in value
                if isinstance(item, str) and item][:MAX_RECENT_PATHS]

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

        # GPU image loading toggle; the window setter also syncs the
        # gpu_decode module and the menu action's check mark.
        if isinstance(data.get("gpu_loading"), bool):
            window.set_gpu_loading_enabled(data["gpu_loading"])

        # Reference frame mode, validated so a hand-edited config can only ever
        # leave a value the registration pipeline understands.
        ref_mode = data.get("reference_frame_mode")
        window.reference_frame_mode = ref_mode if ref_mode in ("first", "middle", "last") else "first"

        # Recently opened paths
        self.recent_folders = self._sanitize_recent(data.get("recent_folders"))
        self.recent_videos = self._sanitize_recent(data.get("recent_videos"))
        output_dir = data.get("output_dir")
        self.output_dir = os.path.normpath(output_dir) if isinstance(output_dir, str) and output_dir else ""
        self.output_format = self._sanitize_format(data.get("output_format"))
        refresh = getattr(window, "refresh_recent_menus", None)
        if callable(refresh):
            refresh()

        # Processing bit depth. Applied to utils.bitdepth as well as the window,
        # since the pipeline reads the module and not the attribute. An
        # unrecognised value falls back to the default rather than raising.
        depth_mode = data.get("bit_depth_mode")
        if depth_mode in bitdepth.VALID_MODES:
            bitdepth.set_mode(depth_mode)
            window.bit_depth_mode = depth_mode
            action = getattr(window, "ui_objs", {}).get(f"action_depth_{depth_mode}")
            if action is not None:
                action.setChecked(True)

        # Post-fusion contrast. Driving the widgets is enough: their signals set
        # the window attributes and refresh the preview through the same path a
        # user interaction takes.
        combo = getattr(window, "combo_contrast", None)
        slider = getattr(window, "slider_contrast", None)
        if combo is not None and slider is not None:
            strength = data.get("contrast_strength")
            if isinstance(strength, (int, float)):
                slider.setValue(int(max(0, min(100, strength))))
            method = data.get("contrast_method", "off")
            index = combo.findData(method)
            combo.setCurrentIndex(index if index >= 0 else 0)

        # Language
        lang = data.get("language")
        if lang:
            trans.set_language(lang)

        # Registration checkboxes
        if "align_scale" in data:
            window.cb_align_scale.setChecked(bool(data["align_scale"]))
        if "align_homography" in data:
            window.cb_align_homography.setChecked(bool(data["align_homography"]))
        if "align_ecc" in data:
            window.cb_align_ecc.setChecked(bool(data["align_ecc"]))

        # Fusion method radio buttons
        method = data.get("fusion_method")
        attr = _FUSION_METHOD_BUTTONS.get(method)
        button = getattr(window, attr, None) if attr else None
        if button is not None and button.isEnabled():
            button.setChecked(True)

        # Post-fusion refinement stage (skipped when its weights are missing)
        if "ifcnn_refine" in data and window.cb_ifcnn.isEnabled():
            window.cb_ifcnn.setChecked(bool(data["ifcnn_refine"]))

        # Sync slider enablement/kernel mode to the selected method, then restore the
        # saved kernel value last so it is not overwritten by the per-method default.
        # Status console visibility (drives the View menu action, which hides the panel)
        if "show_status_console" in data and hasattr(window, "action_show_console"):
            window.action_show_console.setChecked(bool(data["show_status_console"]))

        # Source stack panel visibility (also toggled by double-clicking the output)
        if "show_source_stack" in data and hasattr(window, "action_show_source_stack"):
            window.action_show_source_stack.setChecked(bool(data["show_source_stack"]))

        window.update_slider_availability()
        if "smooth_kernel" in data:
            try:
                window.slider_smooth.setValue(int(data["smooth_kernel"]))
            except (TypeError, ValueError):
                pass
        if "halo_radius" in data:
            try:
                window.slider_halo.setValue(int(data["halo_radius"]))
            except (TypeError, ValueError):
                pass

        # DCT tuning. Each is matched by value/key, so a settings file written
        # by a build with different presets falls back to the current default
        # rather than selecting nothing.
        index = window.combo_dct_block.findData(data.get("dct_block_size"))
        if index >= 0:
            window.combo_dct_block.setCurrentIndex(index)
        index = window.combo_dct_plateau.findData(data.get("dct_plateau"))
        if index >= 0:
            window.combo_dct_plateau.setCurrentIndex(index)
        if "dct_blend" in data:
            window.cb_dct_blend.setChecked(bool(data["dct_blend"]))

        # Pyramid tuning, matched the same way. The combos carry preset keys
        # rather than numbers, so a build that retunes a preset moves an old
        # settings file with it instead of restoring a stale value.
        for key, combo in (("pyramid_levels", window.combo_pyr_levels),
                           ("pyramid_selectivity", window.combo_pyr_selectivity),
                           ("pyramid_coherence", window.combo_pyr_coherence),
                           ("pyramid_base", window.combo_pyr_base)):
            index = combo.findData(data.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        if "pyramid_noise_gate" in data:
            window.cb_pyr_noise_gate.setChecked(bool(data["pyramid_noise_gate"]))
        if "pyramid_envelope" in data:
            window.cb_pyr_envelope.setChecked(bool(data["pyramid_envelope"]))
