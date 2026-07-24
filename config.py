from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from enum import Enum


class FusionMethod(Enum):
    GUIDED_FILTER = "guided_filter"
    DCT = "dct"
    DTCWT = "dtcwt"
    GFG_FGF = "gfgfgf"
    STACKMFFV4 = "stackmffv4"


class ROIMode(Enum):
    CROP = "crop"
    PASTE = "paste"


@dataclass
class RegistrationOptions:
    need_scale: bool = False
    need_homography: bool = False
    need_ecc: bool = False
    downscale_width: Optional[int] = None
    # Frame held fixed during alignment: 'first' (frame 0, chained) or 'middle'.
    reference_mode: str = "first"


@dataclass
class FusionOptions:
    method: FusionMethod = FusionMethod.GUIDED_FILTER
    kernel_size: int = 31
    tile_enabled: bool = True
    tile_block_size: int = 1024
    tile_overlap: int = 256
    tile_threshold: int = 2048
    stackmffv4_batch_size: int = 2


@dataclass
class ROIOptions:
    enabled: bool = False
    rect: Optional[Tuple[float, float, float, float]] = None
    mode: ROIMode = ROIMode.CROP
    base_index: int = 0


@dataclass
class RenderOptions:
    raw_images: List[Any]
    aligned_images: List[Any]
    is_images_aligned: bool
    registration_options: RegistrationOptions
    fusion_options: FusionOptions
    roi_options: ROIOptions
    thread_count: int = 4
    # (need_scale, need_homography, need_ecc, reference_mode) used for the last
    # registration, so cached aligned images can be reused only on an exact match.
    last_alignment_options: Optional[Tuple[bool, bool, bool, str]] = None
