"""
One calling convention for every fusion method in the project.

Each implementation has its own signature - some take a kernel size, some a
decomposition level, some a model path, and DCT refuses img_resize entirely.
This module wraps them all behind ``fuse(stack, **params)`` so the shared test
contract and the report tools can iterate over methods without special-casing
each one.

Availability is resolved lazily: a method whose optional dependency or weights
file is missing reports ``available() -> (False, reason)`` and its tests skip
with that reason rather than failing.
"""

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


def _have(module):
    return importlib.util.find_spec(module) is not None


def _have_torch_device():
    """True when torch is installed and exposes a CUDA or MPS device."""
    if not _have("torch"):
        return False
    import torch
    if torch.cuda.is_available():
        return True
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


@dataclass
class FusionMethod:
    """A fusion implementation reduced to a uniform interface."""

    key: str
    label: str
    fuse: Callable            # fuse(stack, **params) -> uint8 BGR image
    check: Callable           # () -> Optional[str]; a string means "unavailable because"
    params: dict = field(default_factory=dict)   # defaults for a normal run
    # Every tunable parameter, as (name, [values to sweep], what it controls).
    # A method can expose more than one - DCT has both a block size and a
    # median-filter kernel - and the report tools render one page per entry.
    sweeps: tuple = ()
    gpu: bool = False
    supports_resize: bool = True
    supports_folder: bool = True                 # accepts a directory path as input
    # Smallest edge the method can process. The neural models pool the input
    # several times over, so a small enough image collapses to a zero-sized
    # feature map deep in the network.
    min_dimension: Optional[int] = None
    deterministic: bool = True                   # bitwise-stable across repeat runs
    # Reconstruction floor on the shared photographic fixture, in dB. Set well
    # below the measured value so the test guards against regression, not noise.
    min_psnr: float = 30.0

    @property
    def sweep(self):
        """The primary sweep as (name, values), or None - kept for callers
        that only ever varied one parameter."""
        if not self.sweeps:
            return None
        name, values, _ = self.sweeps[0]
        return name, values

    def sweep_for(self, param):
        """Look up one declared parameter by name."""
        for name, values, blurb in self.sweeps:
            if name == param:
                return name, values, blurb
        known = ", ".join(n for n, _, _ in self.sweeps) or "none"
        raise KeyError(f"{self.label} has no tunable parameter {param!r}. "
                       f"Declared: {known}")

    def available(self):
        reason = self.check()
        return (reason is None), reason

    def run(self, stack, **overrides):
        params = dict(self.params)
        params.update(overrides)
        return self.fuse(stack, **params)


# ---------------------------------------------------------------------------
# Adapters - each normalises one implementation's signature
# ---------------------------------------------------------------------------

def _gff(stack, kernel_size=31, img_resize=None, thread_count=None):
    from fusion_methods.gff import gff_impl
    return gff_impl(stack, img_resize, kernel_size=kernel_size, thread_count=thread_count)


def _gff_torch(stack, kernel_size=31, img_resize=None, device=None):
    from fusion_methods.gff_torch import gff_torch_impl
    return gff_torch_impl(stack, img_resize, kernel_size=kernel_size, device=device)


def _gfgfgf(stack, kernel_size=7, img_resize=None, thread_count=None):
    from fusion_methods.gfg_fgf import gfgfgf_impl
    return gfgfgf_impl(stack, img_resize, kernel_size=kernel_size, thread_count=thread_count)


def _gfgfgf_torch(stack, kernel_size=7, img_resize=None, device=None):
    from fusion_methods.gfg_fgf_torch import gfgfgf_torch_impl
    return gfgfgf_torch_impl(stack, img_resize, kernel_size=kernel_size, device=device)


def _pyramid(stack, levels=None, img_resize=None, thread_count=None, **tuning):
    from fusion_methods.pyramid import pyramid_impl
    return pyramid_impl(stack, img_resize, levels=levels,
                        thread_count=thread_count, **tuning)


def _pyramid_torch(stack, levels=None, img_resize=None, device=None, **tuning):
    from fusion_methods.pyramid_torch import pyramid_torch_impl
    return pyramid_torch_impl(stack, img_resize, levels=levels, device=device,
                              **tuning)


def _depthmap_max(stack, kernel_size=9, halo_radius=0, img_resize=None,
                  thread_count=None):
    from fusion_methods.depthmap import depthmap_impl, MODE_MAX
    return depthmap_impl(stack, img_resize, mode=MODE_MAX,
                         kernel_size=kernel_size, halo_radius=halo_radius,
                         thread_count=thread_count)


def _depthmap_average(stack, kernel_size=9, halo_radius=0, selectivity=None,
                      coherence_radius=None, slice_radius=None,
                      img_resize=None, thread_count=None):
    from fusion_methods.depthmap import depthmap_impl, MODE_AVERAGE
    return depthmap_impl(stack, img_resize, mode=MODE_AVERAGE,
                         kernel_size=kernel_size, halo_radius=halo_radius,
                         selectivity=selectivity, thread_count=thread_count,
                         coherence_radius=coherence_radius,
                         slice_radius=slice_radius)


def _depthmap_max_torch(stack, kernel_size=9, halo_radius=0, img_resize=None,
                        device=None):
    from fusion_methods.depthmap_torch import depthmap_torch_impl, MODE_MAX
    return depthmap_torch_impl(stack, img_resize, mode=MODE_MAX,
                               kernel_size=kernel_size, halo_radius=halo_radius,
                               device=device)


def _depthmap_average_torch(stack, kernel_size=9, halo_radius=0,
                            selectivity=None, coherence_radius=None,
                            slice_radius=None, img_resize=None, device=None):
    from fusion_methods.depthmap_torch import depthmap_torch_impl, MODE_AVERAGE
    return depthmap_torch_impl(stack, img_resize, mode=MODE_AVERAGE,
                               kernel_size=kernel_size, halo_radius=halo_radius,
                               selectivity=selectivity, device=device,
                               coherence_radius=coherence_radius,
                               slice_radius=slice_radius)


def _dct(stack, block_size=8, kernel_size=7, img_resize=None):
    from fusion_methods.dct import dct_focus_stack_fusion
    if img_resize is not None:
        raise ValueError("DCT fusion does not support dynamic resizing")
    return dct_focus_stack_fusion(stack, output_path=None,
                                  block_size=block_size, kernel_size=kernel_size)


def _dct_torch(stack, block_size=8, kernel_size=7, img_resize=None):
    from fusion_methods.dct_torch import dct_torch_impl
    if img_resize is not None:
        raise ValueError("DCT fusion does not support dynamic resizing")
    return dct_torch_impl(stack, block_size=block_size, kernel_size=kernel_size)


def _dtcwt(stack, N=4, img_resize=None):
    from fusion_methods.dtcwt import _dtcwt_impl
    return _dtcwt_impl(stack, img_resize, N, False)


def _dtcwt_torch(stack, N=4, img_resize=None, device=None):
    from fusion_methods.dtcwt_torch import dtcwt_torch_impl
    return dtcwt_torch_impl(stack, img_resize, N=N, device=device)


def _stackmffv4(stack, img_resize=None, use_gpu=None):
    from fusion_methods.stackmffv4 import _stackmffv4_impl
    if use_gpu is None:
        use_gpu = _have_torch_device()
    return _stackmffv4_impl(stack, img_resize, os.path.join(WEIGHTS_DIR, "stackmffv4.pth"), use_gpu)


def _gff_then_ifcnn(stack, kernel_size=31, img_resize=None, use_gpu=None):
    """IFCNN is a refinement stage, so it needs a fusion result to refine."""
    from fusion_methods.ifcnn import _ifcnn_refine_impl
    if use_gpu is None:
        use_gpu = _have_torch_device()
    fused = _gff(stack, kernel_size=kernel_size, img_resize=img_resize)
    return _ifcnn_refine_impl(fused, stack, os.path.join(WEIGHTS_DIR, "ifcnn.pth"), use_gpu)


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _needs(*modules):
    def check():
        missing = [m for m in modules if not _have(m)]
        return f"requires {', '.join(missing)}" if missing else None
    return check


def _needs_gpu(*modules):
    def check():
        missing = [m for m in modules if not _have(m)]
        if missing:
            return f"requires {', '.join(missing)}"
        if not _have_torch_device():
            return "requires a CUDA or MPS device"
        return None
    return check


def _needs_weights(filename, *modules):
    def check():
        missing = [m for m in modules if not _have(m)]
        if missing:
            return f"requires {', '.join(missing)}"
        path = os.path.join(WEIGHTS_DIR, filename)
        return None if os.path.isfile(path) else f"requires weights/{filename}"
    return check


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

METHODS = [
    FusionMethod(
        key="guided_filter", label="Guided Filter", fuse=_gff,
        check=_needs("cv2"), params={"kernel_size": 31},
        sweeps=(
            ("kernel_size", [7, 15, 31, 63, 95],
             "Size of the averaging window that splits each frame into a smooth "
             "base layer and a detail layer, and (at a fixed 3:1 ratio) the "
             "radius that smooths the base-layer weights. Near-inert by design: "
             "base + detail reconstructs the frame exactly, so the split only "
             "reaches the output through the gap between the two weight maps. "
             "End to end the sweep moves PSNR by 0.03 dB and the outputs sit "
             "60+ dB apart, i.e. indistinguishable."),
        ),
        min_psnr=33.0,      # measured 39.5
    ),
    FusionMethod(
        key="gfgfgf", label="GFG-FGF", fuse=_gfgfgf,
        check=_needs("cv2"), params={"kernel_size": 7},
        sweeps=(
            ("kernel_size", [3, 7, 15, 31, 63],
             "Window used to measure local contrast before the guided filter "
             "refines the decision map. Small reacts to fine texture; large "
             "smooths the decision and can miss narrow in-focus regions."),
        ),
        deterministic=False,
        min_psnr=32.0,      # measured 38.2
    ),
    FusionMethod(
        key="pyramid", label="Pyramid", fuse=_pyramid,
        check=_needs("cv2"), params={"levels": None},
        sweeps=(
            ("levels", [2, 3, 4, 5, 6],
             "How many band-pass levels the frame is split into before the "
             "selection. Few levels decide focus on coarse blocks and "
             "can miss fine in-focus detail; many levels separate scales finely "
             "but cost more time. Self-limited so the coarsest band stays >= 2 px, "
             "so the very largest values collapse to the same depth on small "
             "images."),
            ("energy_window", [3, 5, 9, 15, 31],
             "Window each band's focus energy is pooled over before the frames "
             "are compared. Small follows fine detail and speckles on grain; "
             "large decides regionally and rounds off narrow in-focus "
             "structures. Pooled again at every level, so the window at level k "
             "already covers 2**k times as much picture."),
            ("selectivity", [2, 4, 8, 32, float("inf")],
             "How sharply each band's weights favour the sharpest frame. inf is "
             "the published choose-max rule, which has no answer where nothing "
             "is in focus and stitches those regions out of frames that "
             "disagree; low numbers average the stack and soften real focus "
             "decisions with it."),
            ("coherence", [0.0, 0.25, 0.5, 0.75],
             "How much of a band's decision comes from the coarser bands above "
             "it, so the bands of one frame decide together instead of a pixel "
             "taking fine detail from one frame and coarse from another. Costs "
             "sharpness at a depth boundary the coarse band cannot see."),
            ("base_selectivity", [0.0, 1.0, 3.0, 8.0],
             "How hard the coarse base band follows the frames that won the "
             "detail bands. 0 is the plain mean, which hazes the base whenever "
             "most of the stack is defocused."),
            ("noise_gate", [False, True],
             "Compare frames in units of their own grain rather than "
             "absolutely. Off lets the brightest, grainiest frame win every "
             "region that holds no detail, and stamp its tone there."),
            ("envelope", [False, True],
             "Clamp each pixel to the range its own frames span. Off is what a "
             "collapsed pyramid does unaided, which can reconstruct a value no "
             "frame had - a thin dark filament over a smooth background."),
        ),
        min_psnr=40.0,      # measured 50.9
    ),
    FusionMethod(
        key="depthmap_max", label="Depth Map (Max)", fuse=_depthmap_max,
        check=_needs("cv2"), params={"kernel_size": 9, "halo_radius": 0},
        sweeps=(
            ("kernel_size", [3, 7, 9, 15, 31],
             "Window the per-pixel Laplacian focus energy is pooled over before "
             "the hard per-pixel select. Small follows fine detail but speckles "
             "on noise; large is steadier but rounds off narrow in-focus "
             "regions."),
            ("halo_radius", [0, 2, 4, 8, 12],
             "Halo-suppression radius. Each frame's focus energy is grey-dilated "
             "by this many pixels before the select, so a sharply focused edge "
             "also claims the band its defocused glow contaminates in the other "
             "frames. 0 is off; set it to roughly the visible halo width. The "
             "cost is genuine detail from other frames within the band."),
        ),
        min_psnr=36.0,      # measured 44.6
    ),
    FusionMethod(
        key="depthmap_average", label="Depth Map (Average)", fuse=_depthmap_average,
        check=_needs("cv2"), params={"kernel_size": 9, "halo_radius": 0},
        sweeps=(
            ("kernel_size", [3, 7, 9, 15, 31],
             "Window the focus energy is pooled over before the contrast-weighted "
             "blend. Small follows fine detail but speckles on noise; large "
             "is steadier but rounds off narrow in-focus regions."),
            ("selectivity", [0, 25, 50, 75, 100],
             "How sharply the blend favours the frame holding the detail. 0 is "
             "the linear contrast weighting, which only selects while the stack "
             "is short: a defocused frame still measures a fraction of the peak, "
             "and a deep stack adds that fraction up hundreds of times until the "
             "blend is the plain mean of everything and the result is veiled. "
             "Higher trades the multi-frame noise reduction of regions that "
             "genuinely have nothing to choose between for detail in the ones "
             "that do."),
            ("coherence_radius", [0, 4, 8, 16, 24],
             "Radius of the edge-aware filter each frame's share of the blend "
             "is passed through before the pixels are gathered. At any useful "
             "selectivity the blend is a hard select in all but name, and over "
             "a region no frame resolves its choice is decided by noise: "
             "neighbouring pixels draw the same smooth surface from frames "
             "eight slices apart, which do not carry the same local brightness, "
             "and the region breaks into blotches. Filtering the shares averages "
             "those choices where the picture is featureless and leaves them "
             "alone where it is not. 0 is off, and costs nothing; anything else "
             "walks the stack twice."),
            ("slice_radius", [0, 2, 3, 4, 6],
             "The same coherence along the frame axis, in slices. The spatial "
             "filter above is edge-aware, so in the regions that hold detail it "
             "follows the guide and leaves the pixel rendered from about one "
             "frame - and no spatial setting reaches those. A stack that "
             "oversamples its depth of field resolves each pixel about equally "
             "well in the several frames around its focus peak, which differ "
             "mostly in their grain, so averaging across that band divides the "
             "grain at no cost in sharpness. Set it to about half the "
             "half-maximum width of the focus curve; 0 on a stack that steps a "
             "full depth of field per frame, where the neighbours are the worst "
             "frames that resolve the pixel at all. Unlike every other dial "
             "here it averages different frames into one pixel, so it needs a "
             "registered stack - unregistered, it averages a subject that "
             "moved."),
        ),
        min_psnr=30.0,      # measured 39.6 at selectivity 0, 43.9 at the
                            # default; blends rather than selects, so it still
                            # trails Max but recovers SNR in flat regions
    ),
    FusionMethod(
        key="dct", label="DCT", fuse=_dct,
        check=_needs("cv2"), params={"block_size": 8, "kernel_size": 7},
        sweeps=(
            ("block_size", [4, 8, 16, 32, 64],
             "Side of the square block the sharpness decision is made over. "
             "Small follows detail closely but is noise-sensitive; large is "
             "steadier but makes boundaries between near and far look stepped."),
            ("kernel_size", [3, 5, 7, 11, 15],
             "Median filter applied to the block decision map to remove isolated "
             "wrong picks. Larger cleans up more speckle but rounds off genuinely "
             "small in-focus regions. The decision map now arrives regionally "
             "pooled, so there is little speckle left for it to remove."),
        ),
        supports_resize=False,
        min_psnr=27.0,      # measured 35.6 (32.3 before item 17 reworked the
                            # focus measure); block decisions are still the
                            # coarsest of the classical methods
    ),
    FusionMethod(
        key="dtcwt", label="DTCWT", fuse=_dtcwt,
        check=_needs("dtcwt"), params={"N": 4},
        sweeps=(
            ("N", [1, 2, 3, 4, 5, 6],
             "How many times the frame is halved into coarser scales before "
             "fusing. More levels let the method reason about large soft "
             "structures, at the cost of time and of blurring fine decisions."),
        ),
        min_psnr=34.0,      # measured 40.8
    ),
    FusionMethod(
        key="stackmffv4", label="StackMFF-V4", fuse=_stackmffv4,
        check=_needs_weights("stackmffv4.pth", "torch"),
        min_psnr=36.0,      # measured 43.2
        # Measured: 112 px works, 96 px raises from torch.max_pool2d
        min_dimension=112,
    ),
    FusionMethod(
        key="gff_ifcnn", label="Guided Filter + IFCNN Refine", fuse=_gff_then_ifcnn,
        check=_needs_weights("ifcnn.pth", "torch"), params={"kernel_size": 31},
        sweeps=(
            ("kernel_size", [7, 31, 63],
             "Passed through to the guided filter that produces the image IFCNN "
             "then refines; the refinement stage itself has no dial."),
        ),
        deterministic=False,
        # IFCNN re-encodes an already near-perfect fusion, so on synthetic
        # stacks it still scores below the plain guided filter it refines (37.1
        # vs 39.7 averaged over the scenarios). It targets detail the fusion
        # stage missed on real stacks.
        supports_folder=False,
        min_psnr=27.0,      # measured 29.3 on the weakest scenario
    ),
    FusionMethod(
        key="guided_filter_gpu", label="Guided Filter (GPU)", fuse=_gff_torch,
        check=_needs_gpu("torch"), params={"kernel_size": 31}, gpu=True,
        sweeps=(("kernel_size", [7, 15, 31, 63, 95],
                 "Same dial as the CPU guided filter."),),
        min_psnr=33.0,      # measured 39.5
    ),
    FusionMethod(
        key="gfgfgf_gpu", label="GFG-FGF (GPU)", fuse=_gfgfgf_torch,
        check=_needs_gpu("torch"), params={"kernel_size": 7}, gpu=True,
        sweeps=(("kernel_size", [3, 7, 15, 31, 63],
                 "Same dial as the CPU GFG-FGF."),),
        min_psnr=32.0,      # measured 41.0
    ),
    FusionMethod(
        key="dct_gpu", label="DCT (GPU)", fuse=_dct_torch,
        check=_needs_gpu("torch"), params={"block_size": 8, "kernel_size": 7},
        sweeps=(("block_size", [4, 8, 16, 32, 64], "Same dial as the CPU DCT."),
                ("kernel_size", [3, 5, 7, 11, 15], "Same dial as the CPU DCT.")),
        gpu=True, supports_resize=False,
        min_psnr=27.0,      # measured 35.6, matching the CPU path exactly
    ),
    FusionMethod(
        key="dtcwt_gpu", label="DTCWT (GPU)", fuse=_dtcwt_torch,
        check=_needs_gpu("torch", "pytorch_wavelets", "pywt"), params={"N": 4},
        sweeps=(("N", [1, 2, 3, 4, 5, 6], "Same dial as the CPU DTCWT."),),
        gpu=True,
        min_psnr=34.0,
    ),
    FusionMethod(
        key="depthmap_max_gpu", label="Depth Map Max (GPU)", fuse=_depthmap_max_torch,
        check=_needs_gpu("torch"), params={"kernel_size": 9, "halo_radius": 0},
        gpu=True,
        sweeps=(("kernel_size", [3, 7, 9, 15, 31],
                 "Same dial as the CPU depth map."),
                ("halo_radius", [0, 2, 4, 8, 12],
                 "Same dial as the CPU depth map; the elliptical element is "
                 "reproduced span by span, so a radius means the same thing.")),
        min_psnr=36.0,      # matches the CPU path away from the frame border
    ),
    FusionMethod(
        key="depthmap_average_gpu", label="Depth Map Average (GPU)",
        fuse=_depthmap_average_torch,
        check=_needs_gpu("torch"), params={"kernel_size": 9, "halo_radius": 0},
        gpu=True,
        sweeps=(("kernel_size", [3, 7, 9, 15, 31],
                 "Same dial as the CPU depth map."),
                ("selectivity", [0, 25, 50, 75, 100],
                 "Same dial as the CPU depth map.")),
        min_psnr=30.0,
    ),
    FusionMethod(
        key="pyramid_gpu", label="Pyramid (GPU)", fuse=_pyramid_torch,
        check=_needs_gpu("torch"), params={"levels": None}, gpu=True,
        sweeps=(("levels", [2, 3, 4, 5, 6], "Same dial as the CPU pyramid."),
                ("selectivity", [2, 4, 8, 32, float("inf")],
                 "Same dial as the CPU pyramid."),
                ("coherence", [0.0, 0.25, 0.5, 0.75],
                 "Same dial as the CPU pyramid.")),
    ),
]

BY_KEY = {m.key: m for m in METHODS}


def get(key):
    if key not in BY_KEY:
        raise KeyError(f"Unknown fusion method {key!r}. "
                       f"Known: {', '.join(BY_KEY)}")
    return BY_KEY[key]


def available_methods(include_gpu=True):
    """Every method whose dependencies are satisfied on this machine."""
    return [m for m in METHODS
            if (include_gpu or not m.gpu) and m.available()[0]]


# CPU/GPU pairs that implement the same algorithm and must agree
PARITY_PAIRS = [
    ("guided_filter", "guided_filter_gpu"),
    ("gfgfgf", "gfgfgf_gpu"),
    ("dct", "dct_gpu"),
    ("dtcwt", "dtcwt_gpu"),
    ("pyramid", "pyramid_gpu"),
    ("depthmap_max", "depthmap_max_gpu"),
    ("depthmap_average", "depthmap_average_gpu"),
]
