"""One readable description of the settings a render ran with.

The interactive render and the batch worker reach the same pipeline from
different directions - one from radio buttons and sliders, the other from a
settings dict - so the vocabulary lives here rather than in either of them.
Both produce the same property names and the same value wording, which is what
makes the XMP group of two files comparable at all.

The result is written into the OpenFocus group of a saved result's XMP packet
(see `utils/metadata.py`), so the wording is aimed at somebody reading the file
months later: units are spelled out, and a stage that did not run says so
instead of being absent.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from constants import (
    PYRAMID_BASE_DEFAULT, PYRAMID_BASE_PRESETS,
    PYRAMID_COHERENCE_DEFAULT, PYRAMID_COHERENCE_PRESETS,
    PYRAMID_ENVELOPE_DEFAULT, PYRAMID_NOISE_GATE_DEFAULT,
    PYRAMID_SELECTIVITY_DEFAULT, PYRAMID_SELECTIVITY_PRESETS,
)
from utils import auto_params, bitdepth

# Display names of the fusion algorithms, keyed by the internal algorithm id
# that both render paths use (see RenderWorker._get_fusion_algorithm and the
# batch dialog's fusion_method).
METHOD_NAMES = {
    "guided_filter": "Guided Filter",
    "dct": "DCT",
    "dtcwt": "DTCWT",
    "gfgfgf": "GFG-FGF",
    "pyramid": "Pyramid",
    "depthmap_max": "Depth Map (Max)",
    "depthmap_average": "Depth Map (Average)",
    "stackmffv4": "StackMFF V4",
}

# Methods driven by the kernel-size slider, and the subset that also takes a
# halo-suppression radius. The others ignore both, so quoting a value for them
# would suggest it changed something.
KERNEL_METHODS = {"guided_filter", "dct", "gfgfgf", "pyramid",
                  "depthmap_max", "depthmap_average"}
HALO_METHODS = {"depthmap_max", "depthmap_average"}

# The coherent-depth dials regularise an index map and blend across the slices
# it names, and only the hard per-pixel select builds one: the average is a
# blend of the whole stack already, so neither dial has anything to act on.
COHERENT_DEPTH_METHODS = {"depthmap_max"}

_CONTRAST_NAMES = {"auto": "Auto", "clahe": "Local (CLAHE)"}
_REFERENCE_NAMES = {"first": "First", "middle": "Middle", "last": "Last"}

_ON = "On"
_OFF = "Off"


def _on_off(flag: Any) -> str:
    return _ON if flag else _OFF


def _preset(value: Any,
            presets: Sequence[Tuple[str, float]],
            default_key: str) -> str:
    """Name the preset a pyramid exponent came from, e.g. 'Balanced (8)'.

    A render carries the number the combo stood for, not the key it was picked
    from, so the name is found back by value. Both are written: the name is what
    the UI called it, the number is what the method did, and a retuned preset
    would otherwise make two files claiming 'balanced' incomparable. A value no
    preset holds - a settings file edited by hand - is quoted on its own.

    `None` means the caller never chose, so the method ran on its own default;
    that is what gets recorded, since it is what produced the pixels.
    """
    values = dict(presets)
    if value is None:
        value = values[default_key]

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    # inf compares equal to itself, so the choose-max preset is found like any
    # other; it is spelled out because 'inf' names nothing on its own.
    text = "choose-max" if number == float("inf") else f"{number:g}"
    name = next((key for key, preset in presets if preset == number), None)
    return f"{name.capitalize()} ({text})" if name else text


def _resolved(requested: Optional[int],
              used: Optional[str],
              auto_label: str = "Auto") -> str:
    """A setting the method decided for itself, next to what was asked of it.

    'Auto' names the choice but not the outcome, and a number the method could
    not honour - a depth deeper than the frame allows - is not what produced
    the pixels either. Both are written the same way: what the render asked
    for, with what actually ran in brackets after it. A run whose value went
    unrecorded (an older file, or a stage that never got there) keeps the plain
    wording rather than claiming a value it does not have.
    """
    if not requested:
        return f"{auto_label} ({used})" if used else auto_label

    asked = f"{int(requested)}"
    return f"{asked} ({used} applied)" if used and used != asked else asked


def _pyramid_options(levels: Optional[int],
                     selectivity: Any,
                     coherence: Any,
                     base_selectivity: Any,
                     noise_gate: Optional[bool],
                     envelope: Optional[bool],
                     levels_used: Optional[str] = None) -> Dict[str, str]:
    """The pyramid's own tuning, as XMP properties.

    Every control the method reads is quoted, including the ones left alone: a
    property missing from the packet cannot be told apart from a build that had
    no such control, which is exactly the question somebody rereading an old
    render asks. The energy window is not repeated here - it is the kernel
    slider, and `KernelSize` above already carries it.
    """
    return {
        # 0 or None is the UI's "let the method decide", which resolves against
        # the image size at render time rather than to a fixed number; the depth
        # it settled on is quoted in brackets when the render reported it.
        "PyramidLevels": _resolved(levels, levels_used),
        "PyramidSelectivity": _preset(
            selectivity, PYRAMID_SELECTIVITY_PRESETS, PYRAMID_SELECTIVITY_DEFAULT),
        "PyramidCoherence": _preset(
            coherence, PYRAMID_COHERENCE_PRESETS, PYRAMID_COHERENCE_DEFAULT),
        "PyramidBaseWeighting": _preset(
            base_selectivity, PYRAMID_BASE_PRESETS, PYRAMID_BASE_DEFAULT),
        "PyramidNoiseGate": _on_off(
            PYRAMID_NOISE_GATE_DEFAULT if noise_gate is None else noise_gate),
        "PyramidEnvelopeClip": _on_off(
            PYRAMID_ENVELOPE_DEFAULT if envelope is None else envelope),
    }


def describe_contrast(method: str, strength: int) -> str:
    """The Contrast value of the group, e.g. 'Auto, strength 50%'.

    Kept separate because contrast is applied when a result is saved, from
    whatever the slider says then, not when it was rendered - so the save path
    refreshes this one property instead of trusting the render-time value.
    """
    name = _CONTRAST_NAMES.get((method or "off").lower())
    if not name or not strength or strength <= 0:
        return _OFF
    return f"{name}, strength {int(strength)}%"


def describe(
    algorithm: Optional[str] = None,
    kernel_size: Optional[int] = None,
    halo_radius: int = 0,
    depth_smoothing: Optional[int] = None,
    ifcnn_refine: bool = False,
    align_scale: bool = False,
    align_homography: bool = False,
    align_ecc: bool = False,
    reference_mode: str = "first",
    reg_downscale_width: Optional[int] = None,
    ecc_parallel: Optional[bool] = None,
    contrast_method: str = "off",
    contrast_strength: int = 0,
    source_count: Optional[int] = None,
    total_count: Optional[int] = None,
    roi_mode: Optional[str] = None,
    roi_base_index: int = 0,
    tile_enabled: Optional[bool] = None,
    tile_block_size: Optional[int] = None,
    tile_overlap: Optional[int] = None,
    tile_threshold: Optional[int] = None,
    stackmffv4_batch_size: Optional[int] = None,
    pyramid_levels: Optional[int] = None,
    pyramid_selectivity: Any = None,
    pyramid_coherence: Any = None,
    pyramid_base: Any = None,
    pyramid_noise_gate: Optional[bool] = None,
    pyramid_envelope: Optional[bool] = None,
    result_dtype: Any = None,
    device_name: Optional[str] = None,
    thread_count: Optional[int] = None,
    resolved_auto: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Dict[str, str]:
    """Describe a render as ordered XMP property name/value pairs.

    Every argument is optional so either caller can pass what it knows; a
    setting it cannot report is left out rather than guessed at. Values are
    already formatted for reading - this is the last step before they are
    written into the file.

    `resolved_auto` is what the run reported through `utils.auto_params` - the
    numbers behind the settings that read 'Auto' - as collected by
    `MultiFocusFusion.fuse`.
    """
    options: Dict[str, str] = {}

    if source_count is not None:
        if total_count is not None and total_count != source_count:
            # A subset render: only the ticked frames took part.
            options["SourceImages"] = f"{source_count} of {total_count} frames"
        else:
            options["SourceImages"] = f"{source_count} frames"

    stages = []
    if align_scale:
        stages.append("Scale")
    if align_homography:
        stages.append("Homography")
    if align_ecc:
        stages.append("ECC")
    options["Registration"] = " + ".join(stages) if stages else _OFF

    if stages:
        options["ReferenceFrame"] = _REFERENCE_NAMES.get(
            (reference_mode or "first").lower(), str(reference_mode)
        )
        options["RegistrationDownscale"] = (
            f"{int(reg_downscale_width)} px" if reg_downscale_width else "Full resolution"
        )
        if align_ecc and ecc_parallel is not None:
            options["EccParallel"] = _on_off(ecc_parallel)

    if algorithm:
        options["FusionMethod"] = METHOD_NAMES.get(algorithm, str(algorithm))
        if algorithm in KERNEL_METHODS and kernel_size:
            options["KernelSize"] = f"{int(kernel_size)} px"
        if algorithm in HALO_METHODS:
            options["HaloRadius"] = f"{int(halo_radius)} px" if halo_radius else _OFF
        if algorithm in COHERENT_DEPTH_METHODS:
            options["DepthSmoothing"] = (
                f"{int(depth_smoothing)}%" if depth_smoothing else _OFF)
        if algorithm == "stackmffv4" and stackmffv4_batch_size:
            # The batch halves itself when the card runs out of memory, so what
            # was asked for is not always what inferred the tiles.
            options["StackMffBatchSize"] = _resolved(
                stackmffv4_batch_size,
                auto_params.value_of(resolved_auto, auto_params.STACKMFF_BATCH_SIZE),
            )
        if algorithm == "pyramid":
            options.update(_pyramid_options(
                pyramid_levels, pyramid_selectivity, pyramid_coherence,
                pyramid_base, pyramid_noise_gate, pyramid_envelope,
                levels_used=auto_params.value_of(
                    resolved_auto, auto_params.PYRAMID_LEVELS),
            ))
        options["IfcnnRefinement"] = _on_off(ifcnn_refine)
    else:
        options["FusionMethod"] = "None (registration only)"

    options["Contrast"] = describe_contrast(contrast_method, contrast_strength)

    if roi_mode is None:
        options["Region"] = "Full frame"
    elif roi_mode == "crop":
        options["Region"] = "Selected region, saved cropped"
    else:
        options["Region"] = f"Selected region, blended onto frame {int(roi_base_index) + 1}"

    if tile_enabled is False:
        options["Tiling"] = _OFF
    elif tile_enabled and tile_block_size and tile_overlap:
        options["Tiling"] = (f"{int(tile_block_size)} px blocks, "
                             f"{int(tile_overlap)} px overlap")
        if tile_threshold:
            options["Tiling"] += f", above {int(tile_threshold)} px"

    # How many tiles ran at once, which is derived from free memory at render
    # time rather than from the thread count above - and forced to one for the
    # GPU methods, which are already parallel inside a tile. Recorded only by a
    # render that tiled, so its absence says the frame stayed in one piece.
    tile_workers = auto_params.value_of(resolved_auto, auto_params.TILE_WORKERS)
    if tile_workers:
        options["TilingWorkers"] = f"{tile_workers} tiles at once"

    mode = bitdepth.get_mode()
    depth = "Auto" if mode == bitdepth.MODE_AUTO else f"Forced {mode}-bit"
    if result_dtype is not None:
        depth += f" ({bitdepth.describe(result_dtype)} result)"
    options["BitDepth"] = depth

    if device_name:
        options["ProcessingUnit"] = str(device_name)
    if thread_count:
        options["ThreadCount"] = str(int(thread_count))

    return options
