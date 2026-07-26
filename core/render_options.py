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

from typing import Any, Dict, Optional

from utils import bitdepth

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
KERNEL_METHODS = {"guided_filter", "dct", "gfgfgf", "depthmap_max", "depthmap_average"}
HALO_METHODS = {"depthmap_max", "depthmap_average"}

_CONTRAST_NAMES = {"auto": "Auto", "clahe": "Local (CLAHE)"}
_REFERENCE_NAMES = {"first": "First", "middle": "Middle", "last": "Last"}

_ON = "On"
_OFF = "Off"


def _on_off(flag: Any) -> str:
    return _ON if flag else _OFF


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
    result_dtype: Any = None,
    device_name: Optional[str] = None,
    thread_count: Optional[int] = None,
) -> Dict[str, str]:
    """Describe a render as ordered XMP property name/value pairs.

    Every argument is optional so either caller can pass what it knows; a
    setting it cannot report is left out rather than guessed at. Values are
    already formatted for reading - this is the last step before they are
    written into the file.
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
        if algorithm == "stackmffv4" and stackmffv4_batch_size:
            options["StackMffBatchSize"] = str(int(stackmffv4_batch_size))
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
