"""Render one capture through a matrix of settings, and keep what each produced.

The suites next to this file answer "is it broken" and "is it worse than last
time" on fixtures small enough to run in CI. This answers the question that only
a real capture can: *which settings should this stack actually be rendered at*.
It loads a folder of frames, registers them once, then renders that one aligned
stack through every combination of fusion settings it was given, writing each
result, a set of matched 1:1 crops, and a table of what the numbers say.

    python -m tests.render_matrix --list
    python -m tests.render_matrix --suite depthmap_average
    python -m tests.render_matrix --suite depthmap_average --step 8   # rig check
    python -m tests.render_matrix \
        --source F:/stack --dest F:/work \
        --registration homography+ecc --fusion depthmap_average \
        --sweep kernel_size=5,9,15 --sweep average_selectivity=0,50,100

Named suites live in tests/render_matrix_suites.py; the --sweep form takes a
cross product of any parameters the method accepts, so a new dial needs no code
here.

## What it is built around

**Registration runs once.** Aligning a few hundred RAW frames through homography
and ECC costs far more than fusing them does, and every candidate in a sweep
wants the same aligned pixels. The stack is loaded once, registered once per
distinct registration setting, and held in memory for the whole run - which is
what makes a 25-candidate sweep affordable at all, and what makes the candidates
comparable: they differ by their settings and by nothing else.

**Each run gets its own folder**, named `<timestamp>_<registration>_<fusion>`
under the destination, holding the renders, the crops, `results.xlsx` (one row
per rendered image, every setting in a column of its own), `manifest.json` with
every number, `settings.json` with everything needed to repeat it, `report.md`
opening with the top five, and the console log. Nothing is overwritten and
nothing is cleaned up: two runs a minute apart are two folders.

**The stack is 16-bit and the renders are JPEG XL**, by default and regardless
of what the source folder holds. Both are properties of the test set rather than
of the input: a sweep that ran at whatever depth its files happened to carry
cannot be read against one that ran at another, and JPEG XL is what the app
saves - full depth at a fifth of a PNG, which is what makes thirty renders a
folder rather than a disk. `--bit-depth` and `--format` move either.

**The crops are the point.** A 2 MP render shown at screen size hides exactly
the differences a sweep is looking for. Three regions are picked automatically
from where the stack has the most detail to offer, the same regions for every
candidate, and written both individually and as a labelled montage per region.
Given a `--reference-render`, that render is fitted onto the run's geometry,
saved beside the results and cropped to the same regions - so it opens every
montage as the yardstick, and each candidate also gets a `vs_reference_` sheet
of its own crops beside the reference's, which is the pair to read at 1:1.

## Reading the numbers

`focus_retention` and `flat_noise` (see tests/fusion_metrics.py) are the pair
that matters and they pull against each other: retention is how much of the
available local contrast survived - haze reads low - and flat_noise is how much
grain is left in the background - a hard select reads high. `qabf` is the
established no-reference fusion score; `defocus_seam_visible` and
`block_speckle` catch a result that tore the background into patches. The
`score` column is a within-run heuristic over those, useful for sorting the
table and for nothing else. The crops decide.

**None of those can tell recovered texture from kept grain**, and on a real
capture that is the distinction a sweep of this method is usually looking for.
`focus_retention` measures local contrast against the most any frame of the
stack offered, so the render that kept the most sensor grain scores highest -
which is not a subtle effect: on the ant capture it ranked the sharpest, most
speckled candidate of 31 first and the one that actually looked closest to a
shipped competitor's render eighth.

`--reference-render` is the answer to that. Point it at another program's render
of the same capture and every candidate is also scored against it, as `RefGap`
(how far our fine-detail energy sits from that render's, measured separately in
each fifth of the frame by how much detail *it* found there) and `RefAgree`
(whether we found detail in the same places, over tiles). It is a reference and
not a ground truth: the two programs aligned the stack differently, so they are
fitted by a similarity and then compared only in pools - see
fusion_metrics.fit_reference for what that fit does and does not remove.
"""

import argparse
import datetime
import itertools
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import contrast as contrast_module                 # noqa: E402
from core.image_loader import ImageStackLoader               # noqa: E402
from core.multi_focus_fusion import MultiFocusFusion         # noqa: E402
from core.registration import (                              # noqa: E402
    align_stack, resolve_reference_index, resolve_stages,
)
from tests import fusion_metrics as fm                       # noqa: E402
from tests import xlsx                                       # noqa: E402
from utils import bitdepth                                   # noqa: E402
from utils.image_utils import read_image_any_depth, write_image   # noqa: E402

# Filename vocabulary, kept the same as the app's own export names
# (controllers/export_manager.py) so a render from here and a render from the
# UI can be filed next to each other and still be told apart by their names.
FUSION_SLUGS = {
    "guided_filter": "GuidedFilter",
    "dct": "DCT",
    "dtcwt": "DTCWT",
    "gfgfgf": "GFGFGF",
    "pyramid": "Pyramid",
    "depthmap_max": "DepthMapMax",
    "depthmap_average": "DepthMapAvg",
    "stackmffv4": "StackMFFV4",
}

STAGE_SLUGS = {"scale": "Scale", "homography": "Homography", "ecc": "ECC"}

# Short tags for the settings that go in a filename. The app abbreviates the
# handful of dials it exposes; anything not listed falls back to its own name,
# which is long but never ambiguous.
PARAM_TAGS = {
    "kernel_size": "k",
    "halo_radius": "h",
    "depth_smoothing": "ds",
    "average_selectivity": "sel",
    "block_size": "b",
    "levels": "L",
    "N": "N",
    "energy_window": "win",
    "selectivity": "psel",
    "coherence": "coh",
    "base_selectivity": "base",
    "noise_gate": "gate",
    "envelope": "env",
}

# Metric key -> (column header, format, higher is better). The order is the
# order of the report table.
COLUMNS: Sequence[Tuple[str, str, str, Optional[bool]]] = (
    ("score", "Score", "{:.3f}", True),
    ("focus_retention", "Retention", "{:.4f}", True),
    ("flat_noise", "FlatNoise", "{:.3f}", False),
    ("reference_gap", "RefGap", "{:.3f}", False),
    ("reference_agreement", "RefAgree", "{:.4f}", True),
    ("qabf", "Q_ABF", "{:.4f}", True),
    ("spatial_frequency", "SpatFreq", "{:.3f}", True),
    ("entropy", "Entropy", "{:.3f}", True),
    ("std_dev", "StdDev", "{:.2f}", True),
    ("defocus_seam_visible", "BgSeam%", "{:.2f}", False),
    ("block_speckle", "Speckle%", "{:.3f}", False),
    ("seconds", "Time(s)", "{:.1f}", None),
)

# What the within-run `score` is built from. Sharpness is the factor and
# cleanliness only modulates it, because the obvious additive form does not
# work: averaging a deep stack into mush scores perfectly on every cleanliness
# term at once - no grain, no seams, no speckle - and a mean of three hundred
# defocused frames would come out top of a sweep looking for a sharp result.
# Multiplying instead means a candidate has to have kept the detail before its
# quiet background counts for anything.
CLEANLINESS_TERMS = ("flat_noise", "defocus_seam_visible", "block_speckle")

# How far cleanliness may move a score: a candidate that loses every
# cleanliness term keeps this share of its retention, one that wins them all
# keeps it whole. 0.4 makes the dirtiest and the cleanest render of equal
# sharpness 2.5x apart, which is enough to break a tie and not enough to
# outrank a visibly sharper result.
CLEANLINESS_FLOOR = 0.4

# The reference render, saved into the run folder on the render's own grid. A
# copy rather than a path in the report, because the fitted version is not the
# file on disk - it has been warped onto this stack's geometry - and because a
# run folder that still means something in a year cannot depend on a path
# outside it.
REFERENCE_FILE = "reference.png"


# ---------------------------------------------------------------------------
# What a run is made of
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackSpec:
    """Which frames to render, and at what size.

    `step` and `limit` exist because the same suite has to be runnable twice:
    once at `--step 8` in a couple of minutes to check the rig and see roughly
    where the settings land, and once over every frame for the answer. A
    subsampled stack is a shallower stack, so its best settings are not
    automatically the full stack's - it narrows the search, it does not finish
    it.
    """

    source: str
    start: int = 0
    step: int = 1
    limit: Optional[int] = None
    scale: float = 1.0
    long_edge: Optional[int] = None
    # 16-bit rather than 'auto', so the depth is a property of the test set and
    # not of whatever the source folder happens to hold: an 8-bit stack and a
    # RAW one otherwise run through different arithmetic, and two sweeps that
    # differ in their headroom cannot be read against each other. It also keeps
    # the blending headroom the average mode needs, which is the mode most of
    # these suites are about.
    bit_depth_mode: str = bitdepth.MODE_16

    def select(self, paths: Sequence[str]) -> List[str]:
        chosen = list(paths)[self.start::max(1, int(self.step))]
        return chosen[:self.limit] if self.limit else chosen

    def describe(self) -> str:
        parts = [os.path.basename(os.path.normpath(self.source))]
        if self.step > 1:
            parts.append(f"every {self.step}th frame")
        if self.limit:
            parts.append(f"first {self.limit}")
        if self.scale != 1.0:
            parts.append(f"scale {self.scale:g}")
        if self.long_edge:
            parts.append(f"long edge {self.long_edge} px")
        return ", ".join(parts)


@dataclass(frozen=True)
class RegistrationSpec:
    """One registration setting, and the identity a cached alignment is keyed on."""

    method: str = "homography+ecc"
    downscale_width: int = 1024
    reference_mode: str = "middle"
    ecc_parallel: bool = True

    @property
    def stages(self) -> Tuple[str, ...]:
        if not self.method or self.method.lower() in ("none", "off", "noalign"):
            return ()
        return resolve_stages(self.method)

    @property
    def slug(self) -> str:
        stages = self.stages
        return "+".join(STAGE_SLUGS[s] for s in stages) if stages else "NoAlign"

    @property
    def cache_key(self) -> Tuple:
        # Only the settings that change the pixels; the slug is cosmetic and
        # 'both' and 'homography+ecc' must share one alignment.
        if not self.stages:
            return ()
        return (self.stages, self.downscale_width, self.reference_mode,
                self.ecc_parallel)


@dataclass
class Variant:
    """One candidate: the settings handed to the fusion method, and its name."""

    params: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    note: str = ""


@dataclass
class Suite:
    """One registration and one fusion method, over a list of candidates.

    One suite is one output folder, which is why it holds exactly one of each:
    the folder is named after them.
    """

    fusion: str
    variants: Sequence[Variant]
    registration: RegistrationSpec = field(default_factory=RegistrationSpec)
    name: Optional[str] = None
    # Contrast is a save-time step in the app, applied after fusion. Off by
    # default here: it is a monotone curve over the result, so it moves every
    # candidate the same way while making them all harder to tell apart.
    contrast_method: str = "off"
    contrast_strength: int = 0
    use_gpu: bool = False
    tile_enabled: bool = True
    tile_block_size: int = 1024
    tile_overlap: int = 256
    tile_threshold: int = 2048
    note: str = ""

    @property
    def fusion_slug(self) -> str:
        return FUSION_SLUGS.get(self.fusion, self.fusion)


@dataclass
class Plan:
    """Everything one invocation does: one stack, one destination, N suites."""

    stack: StackSpec
    destination: str
    suites: Sequence[Suite]
    thread_count: int = 0                 # 0 = every core
    # JPEG XL, which is what the app saves and what these renders are compared
    # against: it keeps all 16 bits, and a 2 MP result costs a fifth of the PNG
    # - which matters when one sweep is thirty of them.
    output_format: str = ".jxl"
    # An 8-bit copy beside each render, because JPEG XL still opens in very
    # little. Set to "" to skip it.
    preview_format: str = ".jpg"
    crop_count: int = 3
    crop_size: int = 448
    # Sources Q_ABF is measured against. It is the one metric that walks the
    # whole stack per candidate - four Sobel passes and an exponential per
    # frame - so a deep stack is subsampled for it. A few dozen frames spread
    # through the stack rank a sweep the way every frame does, at a thirtieth
    # of the cost. The focus ceiling behind `focus_retention` is built from
    # every frame regardless: it costs one pass for the whole run, and
    # subsampling it would raise the retention of every candidate at once.
    metric_source_limit: int = 24
    save_aligned: bool = False
    # Another program's render of the same capture, to score every candidate
    # against. Not a ground truth and not scored as one - the two are not
    # registered to each other and no warp relates them, so the comparison is
    # band energy and tile agreement only (see fusion_metrics.band_ratios). What
    # it answers is the question the no-reference metrics cannot: `retention`
    # rewards keeping the most local contrast, and on a real capture the most
    # local contrast belongs to the render that kept the most grain. Left None,
    # the reference columns are simply absent.
    reference: Optional[str] = None

    @property
    def workers(self) -> int:
        return self.thread_count if self.thread_count > 0 else (os.cpu_count() or 4)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def _tag_value(value: Any) -> str:
    """One setting's value, as it appears in a filename."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None:
        return "auto"
    if isinstance(value, float):
        if value == float("inf"):
            return "max"
        if value.is_integer():
            return str(int(value))
        # A dot would read as a file extension halfway through the name.
        return f"{value:g}".replace(".", "p")
    return str(value)


def variant_label(suite: Suite, variant: Variant) -> str:
    """The filename stem for one candidate, e.g. 'DepthMapAvg_k9_sel75_h4'.

    Every setting the candidate was given is named, including the ones left at
    their default. The app leaves defaults out to keep its filenames short; a
    sweep cannot, because the whole point of the folder is that two names differ
    exactly where the settings did.
    """
    if variant.label:
        return variant.label

    name = suite.fusion_slug
    for key, value in variant.params.items():
        if key == "thread_count":
            continue
        name += f"_{PARAM_TAGS.get(key, key)}{_tag_value(value)}"

    tag = contrast_module.describe(suite.contrast_method,
                                   suite.contrast_strength / 100.0)
    if tag:
        name += f"+{tag}"
    return name


def run_folder_name(suite: Suite, when: datetime.datetime) -> str:
    """`<timestamp>_<registration>_<fusion>`, as the run folder is named."""
    return (f"{when.strftime('%Y%m%d_%H%M%S')}_"
            f"{suite.registration.slug}_{suite.fusion_slug}")


# ---------------------------------------------------------------------------
# Loading and registering, once per run
# ---------------------------------------------------------------------------

def stack_paths(spec: StackSpec) -> List[str]:
    """The source files this spec selects, in filename order."""
    if not os.path.isdir(spec.source):
        raise SystemExit(f"Source folder does not exist: {spec.source}")

    names = sorted(
        name for name in os.listdir(spec.source)
        if os.path.splitext(name)[1].lower() in ImageStackLoader.SUPPORTED_FORMATS
    )
    if not names:
        raise SystemExit(f"No supported image files in {spec.source}")

    chosen = spec.select([os.path.join(spec.source, name) for name in names])
    if len(chosen) < 2:
        raise SystemExit(
            f"{len(chosen)} frame(s) selected from {len(names)} - a stack needs at least 2")
    return chosen


def load_stack(spec: StackSpec, workers: int) -> Tuple[List[np.ndarray], List[str]]:
    """Decode the selected frames, through the same reader the app uses.

    Which matters more than it sounds: a camera DNG is developed by LibRaw with
    the app's own white balance and tone handling, and reading it any other way
    would tune settings against pixels the app never sees.
    """
    bitdepth.set_mode(spec.bit_depth_mode)
    paths = stack_paths(spec)

    print(f"[Stack] Loading {len(paths)} frame(s) from {spec.source}", flush=True)
    started = time.perf_counter()
    loader = ImageStackLoader()
    ok, message, images, filenames = loader.load_from_filepaths(
        paths, scale_factor=spec.scale, target_long_edge=spec.long_edge)
    if not ok:
        raise SystemExit(f"Could not load the stack: {message}")

    height, width = images[0].shape[:2]
    print(f"[Stack] {len(images)} frame(s) at {width}x{height}, "
          f"{bitdepth.describe(images[0].dtype)}, "
          f"in {time.perf_counter() - started:.1f} s", flush=True)
    return images, filenames


def register_stack(images: Sequence[np.ndarray], spec: RegistrationSpec,
                   workers: int) -> List[np.ndarray]:
    """Align the stack, or hand it back untouched when no stage was asked for."""
    stages = spec.stages
    if not stages:
        print("[Register] No registration stage requested; using the frames as loaded.",
              flush=True)
        return list(images)

    reference = resolve_reference_index(spec.reference_mode, len(images))
    print(f"[Register] {' -> '.join(stages)} on {len(images)} frame(s), "
          f"reference frame {reference + 1}, detection width {spec.downscale_width} px",
          flush=True)

    started = time.perf_counter()
    aligned = align_stack(list(images), stages,
                          downscale_width=spec.downscale_width,
                          thread_count=workers,
                          parallel_ecc=spec.ecc_parallel,
                          reference_index=reference)
    print(f"[Register] Done in {time.perf_counter() - started:.1f} s", flush=True)
    return aligned


def load_reference(path: Optional[str]) -> Optional[np.ndarray]:
    """Another program's render of this capture, as 8-bit BGR, or None.

    A path that is set but unreadable is a warning rather than a failure - the
    sweep is still worth running without the reference columns, and a missing
    file should not cost a twenty-minute render.
    """
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"[Metrics] Reference render not found, skipping those columns: {path}",
              flush=True)
        return None

    image = read_image_any_depth(path)
    if image is None:
        print(f"[Metrics] Reference render could not be decoded, skipping those "
              f"columns: {path}", flush=True)
        return None

    image = bitdepth.to_display8(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def fit_reference(reference: Optional[np.ndarray],
                  fused: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
    """Put the reference on the render's grid, and take its zoning off it.

    Done once, against the first candidate that rendered, and reused for the
    rest: every candidate in a run comes out of the same aligned stack and so
    shares one geometry, and re-fitting per candidate would let the yardstick
    move between two rows of the same table. Fitting at all is what makes the
    zones cover the same content - see fusion_metrics.fit_reference.
    """
    if reference is None:
        return None, None
    fitted = fm.fit_reference(reference, bitdepth.to_display8(fused))
    if fitted is None:
        print("[Metrics] Could not fit the reference render to this stack's "
              "geometry; scoring against it uncropped instead, which is only "
              "meaningful if the two already share a scale.", flush=True)
        fitted = reference
    else:
        print(f"[Metrics] Reference fitted onto the render grid "
              f"({fitted.shape[1]}x{fitted.shape[0]})", flush=True)
    return fitted, fm.reference_bands(fitted)


def metric_sources(aligned: Sequence[np.ndarray], limit: int) -> List[np.ndarray]:
    """An evenly spread subsample of the stack, as 8-bit, for the metrics.

    8-bit because that is the scale the metrics in fusion_metrics.py are defined
    on, and evenly spread because the ends of a focus stack are the frames that
    hold the foreground and the background - dropping either would measure
    against a stack that never existed.
    """
    if limit and len(aligned) > limit:
        indices = np.linspace(0, len(aligned) - 1, limit).round().astype(int)
        chosen = [aligned[int(i)] for i in sorted(set(indices.tolist()))]
    else:
        chosen = list(aligned)
    return [bitdepth.to_display8(img) for img in chosen]


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------

def pick_crop_regions(ceiling: np.ndarray, count: int, size: int,
                      margin: int = 8) -> List[Tuple[int, int, int, int]]:
    """Pick the regions worth looking at, from where the stack has most detail.

    Reuses the focus ceiling the metrics already built: the tiles the stack can
    resolve the most detail in are the tiles where two settings will differ
    visibly. Picks are suppressed within a tile's width of each other, so three
    crops are three parts of the picture rather than three views of one edge.

    Returns (x, y, w, h) rectangles, the same ones for every candidate in a run.
    """
    height, width = ceiling.shape[:2]
    size = int(min(size, height - 2 * margin, width - 2 * margin))
    if size < 32:
        return []

    score = cv2.boxFilter(ceiling, cv2.CV_32F, (size, size))
    half = size // 2
    inside = np.zeros(score.shape, bool)
    inside[half + margin:height - half - margin,
           half + margin:width - half - margin] = True
    if not inside.any():
        return []
    score = np.where(inside, score, -1.0).astype(np.float32)

    regions = []
    for _ in range(max(0, count)):
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        if score[y, x] < 0:
            break
        x0 = int(np.clip(x - half, 0, width - size))
        y0 = int(np.clip(y - half, 0, height - size))
        regions.append((x0, y0, size, size))
        cv2.rectangle(score, (int(x) - size, int(y) - size),
                      (int(x) + size, int(y) + size), -1.0, thickness=-1)
    return regions


def crop(img: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = region
    return img[y:y + h, x:x + w]


def build_montage(tiles: Sequence[np.ndarray], labels: Sequence[str],
                  columns: Optional[int] = None, pad: int = 8,
                  label_height: int = 26) -> Optional[np.ndarray]:
    """All candidates' crops of one region, side by side and labelled.

    The one artefact that answers the question directly: two numbers a
    thousandth apart may or may not be a visible difference, and this is where
    that gets settled.
    """
    tiles = [bitdepth.to_display8(t) for t in tiles if t is not None and t.size]
    if not tiles:
        return None

    cell_h = max(t.shape[0] for t in tiles)
    cell_w = max(t.shape[1] for t in tiles)
    if columns is None:
        columns = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    rows = int(np.ceil(len(tiles) / columns))

    sheet = np.full((rows * (cell_h + label_height + pad) + pad,
                     columns * (cell_w + pad) + pad, 3), 32, np.uint8)

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row, column = divmod(index, columns)
        x = pad + column * (cell_w + pad)
        y = pad + row * (cell_h + label_height + pad)
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        cv2.putText(sheet, label, (x + 2, y + tile.shape[0] + label_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return sheet


def build_comparison(tiles: Sequence[np.ndarray],
                     reference_tiles: Sequence[np.ndarray],
                     label: str, reference_name: str) -> Optional[np.ndarray]:
    """One candidate against the reference render, region by region.

    Two columns, one row per crop region: ours on the left, the other program's
    on the right. The region montage puts thirty candidates side by side and is
    the right sheet for ranking them; this one is for the question that comes
    after - what is this render doing that the reference is not - and at a
    thirtieth of the width it can be read at 1:1.

    The reference has already been fitted onto the render's grid, so the two
    halves of a row are the same part of the scene. What the fit cannot remove
    is the last few pixels of local drift - focus breathing is a different
    magnification per frame, so no single warp relates two programs' output -
    which is why these are read as pictures and not differenced.
    """
    pairs: List[np.ndarray] = []
    labels: List[str] = []
    for index, (ours, theirs) in enumerate(zip(tiles, reference_tiles), start=1):
        pairs += [ours, theirs]
        labels += [f"r{index}  {label}", f"r{index}  {reference_name}"]
    return build_montage(pairs, labels, columns=2) if pairs else None


# ---------------------------------------------------------------------------
# Rendering one candidate
# ---------------------------------------------------------------------------

def render_variant(aligned: Sequence[np.ndarray], suite: Suite, variant: Variant,
                   workers: int) -> Tuple[np.ndarray, float]:
    """Fuse the aligned stack at one candidate's settings, then grade it.

    Everything goes through MultiFocusFusion rather than the method module, so
    tiling, the GPU fallback and the auto-resolved settings behave exactly as
    they do in a render from the app.
    """
    engine = MultiFocusFusion(
        algorithm=suite.fusion,
        use_gpu=suite.use_gpu,
        tile_enabled=suite.tile_enabled,
        tile_block_size=suite.tile_block_size,
        tile_overlap=suite.tile_overlap,
        tile_threshold=suite.tile_threshold,
    )

    params = dict(variant.params)
    params.setdefault("thread_count", workers)

    started = time.perf_counter()
    fused = engine.fuse(list(aligned), None, **params)
    elapsed = time.perf_counter() - started

    fused = contrast_module.apply_contrast(
        fused, suite.contrast_method, suite.contrast_strength / 100.0)
    return fused, elapsed


def measure(fused: np.ndarray, sources: Sequence[np.ndarray],
            ceiling: np.ndarray,
            reference: Optional[np.ndarray] = None,
            bands: Optional[Tuple] = None) -> Dict[str, float]:
    """Every metric this harness reports, for one render."""
    fused8 = bitdepth.to_display8(fused)
    fused8, matched, _ = fm.align_to_common_size(fused8, list(sources))

    scores = fm.evaluate(fused8, matched)
    scores["focus_retention"] = fm.focus_retention(fused8, ceiling)
    scores["flat_noise"] = fm.flat_noise(fused8)

    if reference is not None and bands is not None:
        ours, theirs, _ = fm.align_to_common_size(fused8, [reference])
        ratios = fm.band_ratios(ours, theirs[0], bands)
        scores["reference_gap"] = fm.reference_gap(ratios)
        agreement, share = fm.detail_agreement(ours, theirs[0], block=32)
        scores["reference_agreement"] = agreement
        scores["reference_share"] = share
        for index, ratio in enumerate(ratios, start=1):
            scores[f"reference_band{index}"] = ratio
    return scores


def add_scores(rows: Sequence[Dict[str, Any]]) -> None:
    """Fill in the within-run `score` column, in place.

    `focus_retention` carries the score and the cleanliness terms scale it (see
    CLEANLINESS_TERMS). Each cleanliness term is min-max normalised across the
    candidates that ran, so the modulator says where a candidate sits in *this*
    sweep and means nothing outside it; retention is already an absolute
    fraction and is used as it stands. A term every candidate ties on
    contributes nothing rather than dividing by zero, and a sweep where they all
    tie leaves the score equal to retention.
    """
    usable = [row for row in rows if row.get("metrics")]
    if not usable:
        return

    cleanliness = [0.0] * len(usable)
    contributing = 0
    for key in CLEANLINESS_TERMS:
        values = [row["metrics"].get(key) for row in usable]
        if any(value is None or not np.isfinite(value) for value in values):
            continue
        low, high = min(values), max(values)
        if high - low <= 0:
            continue
        contributing += 1
        for index, value in enumerate(values):
            cleanliness[index] += 1.0 - (value - low) / (high - low)

    for index, row in enumerate(usable):
        clean = cleanliness[index] / contributing if contributing else 1.0
        retention = row["metrics"].get("focus_retention") or 0.0
        row["metrics"]["score"] = retention * (
            CLEANLINESS_FLOOR + (1.0 - CLEANLINESS_FLOOR) * clean)


# ---------------------------------------------------------------------------
# Writing the run out
# ---------------------------------------------------------------------------

class _Tee:
    """Echo everything printed into the run folder's log as well as the console."""

    def __init__(self, stream, path):
        self._stream = stream
        self._log = open(path, "a", encoding="utf-8", errors="replace")

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def close(self):
        self._log.close()


def _jsonable(value: Any) -> Any:
    """Settings as JSON, with the values JSON has no spelling for."""
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _jsonable(float(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_settings(path: str, plan: Plan, suite: Suite, frames: int,
                   filenames: Sequence[str]) -> None:
    """Everything needed to run this again, next to what it produced."""
    payload = {
        "source": plan.stack.source,
        "frames_rendered": frames,
        "frame_selection": {
            "start": plan.stack.start, "step": plan.stack.step,
            "limit": plan.stack.limit, "scale": plan.stack.scale,
            "long_edge": plan.stack.long_edge,
        },
        "first_frame": filenames[0] if filenames else None,
        "last_frame": filenames[-1] if filenames else None,
        "bit_depth_mode": plan.stack.bit_depth_mode,
        "registration": {
            "method": suite.registration.method,
            "stages": list(suite.registration.stages),
            "downscale_width": suite.registration.downscale_width,
            "reference_mode": suite.registration.reference_mode,
            "ecc_parallel": suite.registration.ecc_parallel,
        },
        "fusion": suite.fusion,
        "contrast": {"method": suite.contrast_method,
                     "strength": suite.contrast_strength},
        "gpu": suite.use_gpu,
        "tiling": {"enabled": suite.tile_enabled,
                   "block_size": suite.tile_block_size,
                   "overlap": suite.tile_overlap,
                   "threshold": suite.tile_threshold},
        "thread_count": plan.workers,
        "output_format": plan.output_format,
        "reference": plan.reference,
        "variants": [_jsonable(v.params) for v in suite.variants],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _parameter_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Every settings key any candidate carried, in the order they first appear."""
    names: List[str] = []
    for row in rows:
        for key in row.get("params", {}):
            if key != "thread_count" and key not in names:
                names.append(key)
    return names


def write_workbook(path: str, plan: Plan, suite: Suite,
                   rows: Sequence[Dict[str, Any]], frames: int,
                   elapsed: float) -> None:
    """The run as a spreadsheet: one row per rendered image, sortable.

    The same numbers as report.md and manifest.json, in the form that is
    actually used to pick a setting - sorted by one column, filtered on another,
    two runs pasted under each other. Each candidate's settings get a column of
    their own rather than being folded into its name, so the sweep can be
    re-sorted by the dial being studied.
    """
    parameters = _parameter_columns(rows)
    header = (["Rank", "Variant", "File"] + parameters
              + [head for _, head, _, _ in COLUMNS] + ["Status"])

    ranked = sorted(rows, key=lambda r: ((r.get("metrics") or {}).get("score", float("-inf"))),
                    reverse=True)
    results: List[List[Any]] = [header]
    for rank, row in enumerate(ranked, start=1):
        metrics = row.get("metrics")
        line: List[Any] = [rank if metrics else None, row["label"],
                           row.get("file", "")]
        line += [row.get("params", {}).get(name) for name in parameters]
        for key, _, _, _ in COLUMNS:
            value = (metrics or {}).get(key)
            line.append(round(float(value), 6)
                        if value is not None and np.isfinite(value) else None)
        line.append(row.get("error") or row.get("note") or "ok")
        results.append(line)

    stages = " -> ".join(suite.registration.stages) or "none"
    settings: List[List[Any]] = [
        ["Setting", "Value"],
        ["Suite", suite.name or suite.fusion_slug],
        ["Source", plan.stack.source],
        ["Frames", frames],
        ["Frame selection", plan.stack.describe()],
        ["Bit depth", f"{plan.stack.bit_depth_mode}-bit"
            if plan.stack.bit_depth_mode != bitdepth.MODE_AUTO else "auto"],
        ["Registration", stages],
        ["Reference frame", suite.registration.reference_mode],
        ["Detection width", suite.registration.downscale_width],
        ["ECC parallel", suite.registration.ecc_parallel],
        ["Fusion", suite.fusion],
        ["Device", "GPU" if suite.use_gpu else "CPU"],
        ["Contrast", contrast_module.describe(
            suite.contrast_method, suite.contrast_strength / 100.0) or "off"],
        ["Tiling", f"{suite.tile_block_size} px blocks, {suite.tile_overlap} px overlap, "
                   f"above {suite.tile_threshold} px" if suite.tile_enabled else "off"],
        ["Threads", plan.workers],
        ["Output format", plan.output_format],
        ["Preview format", plan.preview_format or "none"],
        ["Q_ABF source frames", min(plan.metric_source_limit or frames, frames)],
        ["Candidates", len(rows)],
        ["Rendered", sum(1 for row in rows if row.get("metrics"))],
        ["Total minutes", round(elapsed / 60.0, 2)],
    ]

    xlsx.write(path, [("Results", results), ("Settings", settings)])


def _summary(ranked: Sequence[Dict[str, Any]], top: int = 5) -> List[str]:
    """The top few candidates, and the ends of the trade-off they sit on.

    First thing in the report because it is the thing being asked for. The two
    extremes are named alongside the ranking because the score is one weighting
    of a trade-off, and the sharpest and the cleanest render are both answers to
    somebody's question even when neither tops the table.
    """
    scored = [row for row in ranked if row.get("metrics")]
    if not scored:
        return ["## Summary", "", "Nothing rendered.", ""]

    lines = ["## Summary", "",
             f"Top {min(top, len(scored))} of {len(ranked)} by within-run score:", ""]
    lines += ["| # | Variant | Settings | Score | Retention | FlatNoise | Q_ABF |",
              "|---|---|---|---:|---:|---:|---:|"]
    for rank, row in enumerate(scored[:top], start=1):
        metrics = row["metrics"]
        settings = ", ".join(f"{key} {_tag_value(value)}"
                             for key, value in row.get("params", {}).items()
                             if key != "thread_count") or "defaults"
        lines.append(
            f"| {rank} | `{row['label']}` | {settings} | {metrics['score']:.3f} | "
            f"{metrics['focus_retention']:.4f} | {metrics['flat_noise']:.3f} | "
            f"{metrics['qabf']:.4f} |")

    sharpest = max(scored, key=lambda r: r["metrics"]["focus_retention"])
    cleanest = min(scored, key=lambda r: r["metrics"]["flat_noise"])
    lines += [
        "",
        f"- Sharpest: `{sharpest['label']}` "
        f"(retention {sharpest['metrics']['focus_retention']:.4f}, "
        f"flat noise {sharpest['metrics']['flat_noise']:.3f})",
        f"- Cleanest: `{cleanest['label']}` "
        f"(flat noise {cleanest['metrics']['flat_noise']:.3f}, "
        f"retention {cleanest['metrics']['focus_retention']:.4f})",
        "",
        "The score weights those two one way; the crops are what settle which "
        "weighting was right for this capture.",
        "",
    ]

    # The same candidates, ranked the other way. Worth its own table whenever a
    # reference was given, because the two orderings can disagree completely:
    # `score` is carried by focus_retention, which measures local contrast
    # against the best any frame offered and so cannot tell recovered texture
    # from kept grain. Where they disagree it is the more interesting half of
    # the run, and reading only the table above would miss it entirely.
    referenced = [row for row in scored if "reference_agreement" in row["metrics"]]
    if referenced:
        by_reference = sorted(referenced,
                              key=lambda r: -r["metrics"]["reference_agreement"])
        lines += [f"Top {min(top, len(by_reference))} by agreement with the "
                  f"reference render:", ""]
        lines += ["| # | Variant | Settings | RefAgree | RefGap | Retention | Score |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for rank, row in enumerate(by_reference[:top], start=1):
            metrics = row["metrics"]
            settings = ", ".join(f"{key} {_tag_value(value)}"
                                 for key, value in row.get("params", {}).items()
                                 if key != "thread_count") or "defaults"
            lines.append(
                f"| {rank} | `{row['label']}` | {settings} | "
                f"{metrics['reference_agreement']:.4f} | "
                f"{metrics['reference_gap']:.3f} | "
                f"{metrics['focus_retention']:.4f} | {metrics['score']:.3f} |")

        top_scored = {row["label"] for row in scored[:top]}
        overlap = sum(1 for row in by_reference[:top] if row["label"] in top_scored)
        lines += [
            "",
            f"{overlap} of {min(top, len(by_reference))} candidates appear in both "
            f"tables. " + (
                "The two rankings disagree, which is the run's finding rather "
                "than a fault in either: retention rewards whichever render kept "
                "the most local contrast, and on a real capture the grainiest "
                "render keeps the most. The crops settle it."
                if overlap <= top // 2 else
                "The two rankings broadly agree, so the score is being carried "
                "by something the reference also recognises."),
            "",
        ]
    return lines


def _table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The report's ranked table, as markdown."""
    headers = ["Variant"] + [head for _, head, _, _ in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] + ["---:"] * len(COLUMNS)) + "|"]

    best = {}
    for key, _, _, higher in COLUMNS:
        if higher is None:
            continue
        values = [row["metrics"].get(key) for row in rows if row.get("metrics")]
        values = [v for v in values if v is not None and np.isfinite(v)]
        if values:
            best[key] = max(values) if higher else min(values)

    for row in rows:
        cells = [row["label"]]
        metrics = row.get("metrics")
        for key, _, fmt, _ in COLUMNS:
            value = (metrics or {}).get(key)
            if value is None or not np.isfinite(value):
                cells.append("-" if metrics else "failed")
                continue
            text = fmt.format(value)
            # The best value in a column is marked, so the table can be read
            # without comparing every row against every other row.
            cells.append(f"**{text}**" if key in best and value == best[key] else text)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(path: str, plan: Plan, suite: Suite, rows: Sequence[Dict[str, Any]],
                 regions: Sequence[Tuple[int, int, int, int]],
                 frames: int, elapsed: float) -> None:
    ranked = sorted(
        rows,
        key=lambda r: (r.get("metrics") or {}).get("score", float("-inf")),
        reverse=True,
    )

    lines = [
        f"# {suite.name or suite.fusion_slug} - {suite.registration.slug}",
        "",
        f"- Source: `{plan.stack.source}`",
        f"- Frames: {frames} ({plan.stack.describe()})",
        f"- Registration: {' -> '.join(suite.registration.stages) or 'none'}"
        f" (reference {suite.registration.reference_mode},"
        f" detection width {suite.registration.downscale_width} px)",
        f"- Fusion: `{suite.fusion}`"
        + (f", GPU" if suite.use_gpu else ", CPU"),
        f"- Contrast: {contrast_module.describe(suite.contrast_method, suite.contrast_strength / 100.0) or 'off'}",
        f"- Bit depth: {plan.stack.bit_depth_mode}"
        + ("" if plan.stack.bit_depth_mode == bitdepth.MODE_AUTO else "-bit")
        + f", saved as {plan.output_format}",
        f"- Candidates: {len(rows)}, total {elapsed / 60.0:.1f} min",
        "",
    ]
    if suite.note:
        lines += [suite.note, ""]

    lines += _summary(ranked)
    lines += [
        "## Results",
        "",
        "Sorted by the within-run score; **bold** is the best value in its column.",
        "",
        "- **Retention** - share of the local contrast the stack had to offer that",
        "  survived into the render. This is the haze reading: a blend that averages",
        "  a defocused majority into the answer scores low however sharp its sources",
        "  were. Absolute, so it compares across runs.",
        "- **FlatNoise** - grain left where the picture holds no detail. Lower is",
        "  cleaner, and a blend beats a hard select here because averaging frames is",
        "  what removes grain.",
    ]
    if plan.reference:
        lines += [
            f"- **RefGap** / **RefAgree** - against `{os.path.basename(plan.reference)}`,",
            "  which is another program's render of this capture and not a ground",
            "  truth: the two are not registered to each other and no warp relates",
            "  them, so nothing here is read pixel by pixel. RefGap is how far our",
            "  fine-detail energy sits from that render's, measured separately in each",
            "  fifth of the frame by how much detail *it* found there and combined as",
            "  an RMS log ratio - 0 is the reference exactly. RefAgree is whether we",
            "  found detail in the same places, over 32 px tiles. They are reported",
            "  together because either alone is easy to satisfy: a uniformly blurred",
            "  render keeps some agreement while recovering nothing, and a render that",
            "  is grainy in the flat fifth and soft in the busy one can average out to",
            "  a small gap. Read them against Retention, which cannot tell recovered",
            "  texture from kept grain and rewards both.",
        ]
    lines += [
        "- **Score** - retention, scaled by how the cleanliness columns rank *within",
        f"  this run* (between {CLEANLINESS_FLOOR:g}x and 1x of it). A sorting aid, not a",
        "  verdict: it cannot see a halo, a stitching seam across a highlight, or a",
        "  colour shift. The crops decide.",
        "",
    ]
    lines += _table(ranked)
    lines += ["", "## Crops", ""]
    if regions:
        for index, (x, y, w, h) in enumerate(regions, start=1):
            lines.append(f"- Region {index}: {w}x{h} at ({x}, {y}) - "
                         f"`crops/region{index}_montage.png`")
    else:
        lines.append("- The frame was too small for a crop region.")

    if plan.reference:
        name = os.path.basename(plan.reference)
        lines += [
            "",
            "## Against the reference render",
            "",
            f"![{name}]({REFERENCE_FILE})",
            "",
            f"`{REFERENCE_FILE}` is `{name}` warped onto this run's geometry by a",
            "similarity fit, which is what makes a crop of it the same part of the",
            "scene as the same crop of a render. It is a copy rather than a link,",
            "because the fitted version is not the file it came from.",
            "",
            "- `crops/reference_rN.png` - the reference's own crop of each region",
            "- `crops/vs_reference_<variant>.png` - one candidate beside the",
            "  reference, region by region, at 1:1. This is the sheet for *what is",
            "  this render doing that the reference is not*; the region montages",
            "  above, which now open with the reference tile, are the sheet for",
            "  ranking candidates against each other.",
            "",
            "Neither is differenced, and the numbers above are not read pixel by",
            "pixel either. The two programs aligned the stack against different",
            "frames and focus breathing means the magnification each removed varies",
            "frame to frame, so the fit leaves several pixels of local drift that",
            "nothing rigid removes - see `fusion_metrics.fit_reference`.",
        ]

    failed = [row for row in rows if not row.get("metrics")]
    if failed:
        lines += ["", "## Failures", ""]
        lines += [f"- `{row['label']}`: {row.get('error')}" for row in failed]

    lines += ["", "## Files", "",
              "- `results.xlsx` - the table above as a spreadsheet, one row per "
              "rendered image, with each setting in its own column",
              "- `manifest.json` - every number, per candidate",
              "- `settings.json` - what to repeat this run with",
              "- `run.log` - the console output"]
    if plan.reference:
        lines.append(f"- `{REFERENCE_FILE}` - the reference render, on this "
                     f"run's geometry")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def run_suite(plan: Plan, suite: Suite, aligned: Sequence[np.ndarray],
              filenames: Sequence[str]) -> str:
    """Render every candidate of one suite into a folder of its own."""
    started_at = datetime.datetime.now()
    run_dir = os.path.join(plan.destination, run_folder_name(suite, started_at))
    crops_dir = os.path.join(run_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    tee = _Tee(sys.stdout, os.path.join(run_dir, "run.log"))
    real_stdout = sys.stdout
    sys.stdout = tee
    started = time.perf_counter()

    try:
        print(f"\n=== {suite.name or suite.fusion_slug} -> {run_dir} ===", flush=True)
        write_settings(os.path.join(run_dir, "settings.json"), plan, suite,
                       len(aligned), filenames)

        sources = metric_sources(aligned, plan.metric_source_limit)
        print(f"[Metrics] Q_ABF against {len(sources)} of {len(aligned)} frame(s); "
              f"focus ceiling from all {len(aligned)}", flush=True)

        # Loaded now so a bad path is reported before the first render, but
        # fitted against the first result - it is the render's geometry the
        # reference has to be put on, and no frame of the stack carries it.
        raw_reference = load_reference(plan.reference)
        reference, bands = None, None
        reference_tiles: List[np.ndarray] = []
        reference_name = os.path.basename(plan.reference) if plan.reference else ""
        if raw_reference is not None:
            print(f"[Metrics] Reference render {reference_name} "
                  f"at {raw_reference.shape[1]}x{raw_reference.shape[0]}", flush=True)
        # The frames go in at their own depth: focus_energy_map brings 16-bit
        # onto the 8-bit scale itself, so the ceiling matches the 8-bit renders
        # it is compared against without a copy of the stack being made.
        ceiling = fm.stack_focus_ceiling(aligned)
        regions = pick_crop_regions(ceiling, plan.crop_count, plan.crop_size)
        print(f"[Crops] {len(regions)} region(s): "
              + ", ".join(f"{w}x{h} at ({x},{y})" for x, y, w, h in regions), flush=True)

        rows: List[Dict[str, Any]] = []
        # Crops are held by candidate rather than appended, so the montage can be
        # laid out best-first once the scores exist - which is the order somebody
        # comparing thirty tiles wants to read them in.
        tiles: Dict[str, List[np.ndarray]] = {}

        for index, variant in enumerate(suite.variants, start=1):
            label = variant_label(suite, variant)
            print(f"\n[{index}/{len(suite.variants)}] {label}", flush=True)
            row: Dict[str, Any] = {"label": label,
                                   "params": _jsonable(dict(variant.params)),
                                   "note": variant.note}

            try:
                fused, seconds = render_variant(aligned, suite, variant, plan.workers)
            except Exception as exc:                      # one bad setting, not a bad run
                print(f"    FAILED: {exc}", flush=True)
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                continue

            if raw_reference is not None and reference is None:
                reference, bands = fit_reference(raw_reference, fused)
                # Written out on the render's own grid, and cropped to the same
                # regions, so every comparison below is the same pixels of the
                # same scene rather than two programs' idea of where it is.
                cv2.imwrite(os.path.join(run_dir, REFERENCE_FILE), reference)
                cropped = [crop(reference, region) for region in regions]
                # Only when the fit succeeded does the reference share the
                # render's geometry; the uncropped fallback can be smaller than
                # a region, and a short tile would slide every label in the
                # montage along by one.
                if all(tile.shape[:2] == (h, w)
                       for tile, (_, _, w, h) in zip(cropped, regions)):
                    reference_tiles = cropped
                    for region_index, tile in enumerate(reference_tiles, start=1):
                        cv2.imwrite(os.path.join(crops_dir,
                                                 f"reference_r{region_index}.png"),
                                    tile)
                else:
                    print("[Crops] Reference render does not cover the crop "
                          "regions; leaving it out of the montages.", flush=True)

            image_path = os.path.join(run_dir, label + plan.output_format)
            write_image(image_path, fused)
            if plan.preview_format and plan.preview_format != plan.output_format:
                write_image(os.path.join(run_dir, label + plan.preview_format),
                            bitdepth.to_display8(fused))

            tiles[label] = []
            for region_index, region in enumerate(regions):
                tile = crop(fused, region)
                tiles[label].append(tile)
                cv2.imwrite(
                    os.path.join(crops_dir, f"{label}_r{region_index + 1}.png"),
                    bitdepth.prepare_for_write(tile, ".png"))

            if reference_tiles:
                sheet = build_comparison(tiles[label], reference_tiles, label,
                                         reference_name)
                if sheet is not None:
                    cv2.imwrite(os.path.join(crops_dir, f"vs_reference_{label}.png"),
                                sheet)

            row["metrics"] = {k: float(v) for k, v in
                              measure(fused, sources, ceiling,
                                      reference, bands).items()}
            row["metrics"]["seconds"] = seconds
            row["file"] = os.path.basename(image_path)
            rows.append(row)
            line = (f"    {seconds:.1f} s, retention {row['metrics']['focus_retention']:.4f}, "
                    f"flat noise {row['metrics']['flat_noise']:.3f}, "
                    f"Q_ABF {row['metrics']['qabf']:.4f}")
            if "reference_gap" in row["metrics"]:
                line += (f", ref gap {row['metrics']['reference_gap']:.3f}, "
                         f"ref agree {row['metrics']['reference_agreement']:.4f}")
            print(line, flush=True)
            del fused

        add_scores(rows)

        montage_order = sorted(
            (row for row in rows if row["label"] in tiles),
            key=lambda r: (r.get("metrics") or {}).get("score", float("-inf")),
            reverse=True)
        for region_index in range(len(regions)):
            region_tiles = [tiles[row["label"]][region_index]
                            for row in montage_order]
            region_labels = [
                f"{index}. {row['label']}"
                f"  ret {(row.get('metrics') or {}).get('focus_retention', 0):.3f}"
                for index, row in enumerate(montage_order, start=1)]
            # The reference leads the sheet rather than trailing it: it is the
            # thing every other tile is being read against, and a yardstick
            # thirty tiles down is one nobody looks at.
            if reference_tiles:
                region_tiles.insert(0, reference_tiles[region_index])
                region_labels.insert(0, f"REFERENCE  {reference_name}")
            sheet = build_montage(region_tiles, region_labels)
            if sheet is not None:
                cv2.imwrite(os.path.join(crops_dir,
                                         f"region{region_index + 1}_montage.png"), sheet)

        elapsed = time.perf_counter() - started
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "suite": suite.name or suite.fusion_slug,
                "started": started_at.isoformat(timespec="seconds"),
                "seconds": elapsed,
                "frames": len(aligned),
                "registration": suite.registration.slug,
                "fusion": suite.fusion,
                "crop_regions": [list(r) for r in regions],
                "results": _jsonable(rows),
            }, handle, indent=2)

        write_report(os.path.join(run_dir, "report.md"), plan, suite, rows,
                     regions, len(aligned), elapsed)
        write_workbook(os.path.join(run_dir, "results.xlsx"), plan, suite, rows,
                       len(aligned), elapsed)

        ranked = sorted((r for r in rows if r.get("metrics")),
                        key=lambda r: r["metrics"]["score"], reverse=True)
        print(f"\n=== {len(ranked)}/{len(rows)} rendered in {elapsed / 60.0:.1f} min ===",
              flush=True)
        for row in ranked[:5]:
            print(f"    {row['metrics']['score']:.3f}  {row['label']}", flush=True)
    finally:
        sys.stdout = real_stdout
        tee.close()

    return run_dir


def run_plan(plan: Plan) -> List[str]:
    """Load once, register once per distinct setting, render every suite."""
    os.makedirs(plan.destination, exist_ok=True)
    images, filenames = load_stack(plan.stack, plan.workers)

    aligned_cache: Dict[Tuple, List[np.ndarray]] = {}
    run_dirs = []
    for suite in plan.suites:
        key = suite.registration.cache_key
        if key not in aligned_cache:
            aligned_cache[key] = register_stack(images, suite.registration, plan.workers)
            if plan.save_aligned and key:
                folder = os.path.join(plan.destination,
                                      f"_aligned_{suite.registration.slug}")
                os.makedirs(folder, exist_ok=True)
                for name, frame in zip(filenames, aligned_cache[key]):
                    cv2.imwrite(os.path.join(folder, os.path.splitext(name)[0] + ".png"),
                                bitdepth.prepare_for_write(frame, ".png"))
                print(f"[Register] Aligned frames written to {folder}", flush=True)
        run_dirs.append(run_suite(plan, suite, aligned_cache[key], filenames))
    return run_dirs


# ---------------------------------------------------------------------------
# Building suites from the command line
# ---------------------------------------------------------------------------

def parse_value(text: str) -> Any:
    """One sweep value, as the type its name implies."""
    lowered = text.strip().lower()
    if lowered in ("true", "on", "yes"):
        return True
    if lowered in ("false", "off", "no"):
        return False
    if lowered in ("none", "auto", "default"):
        return None
    if lowered in ("inf", "max"):
        return float("inf")
    if re.fullmatch(r"[+-]?\d+", lowered):
        return int(lowered)
    try:
        return float(lowered)
    except ValueError:
        return text.strip()


def sweep_variants(sweeps: Sequence[str], fixed: Sequence[str]) -> List[Variant]:
    """Cross product of `--sweep name=a,b,c`, with `--set name=v` held constant.

    The parameters keep the order they were given, which is the order they
    appear in a filename - so the dial being studied can be put first and the
    folder listing sorts by it.
    """
    base: Dict[str, Any] = {}
    for item in fixed:
        name, _, value = item.partition("=")
        base[name.strip()] = parse_value(value)

    axes: List[Tuple[str, List[Any]]] = []
    for item in sweeps:
        name, _, values = item.partition("=")
        if not values:
            raise SystemExit(f"--sweep needs name=v1,v2,...; got {item!r}")
        axes.append((name.strip(), [parse_value(v) for v in values.split(",")]))

    if not axes:
        return [Variant(params=dict(base))]

    variants = []
    for combination in itertools.product(*(values for _, values in axes)):
        params = dict(base)
        params.update({name: value for (name, _), value in zip(axes, combination)})
        variants.append(Variant(params=params))
    return variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a stack through a matrix of registration and fusion settings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("## What it is built around")[0])

    parser.add_argument("--list", action="store_true",
                        help="list the named suites and exit")
    parser.add_argument("--suite", action="append", default=[],
                        help="a named suite from tests/render_matrix_suites.py "
                             "(repeatable)")

    parser.add_argument("--source", help="folder of source frames")
    parser.add_argument("--dest", help="destination folder; a run folder is created inside it")
    parser.add_argument("--registration", default=None,
                        help="'homography+ecc', 'ecc', 'scale+homography', 'none', ...")
    parser.add_argument("--fusion", help="fusion algorithm id, e.g. depthmap_average")
    parser.add_argument("--sweep", action="append", default=[], metavar="NAME=V1,V2",
                        help="a parameter to sweep; repeat for a cross product")
    parser.add_argument("--set", action="append", default=[], dest="fixed",
                        metavar="NAME=VALUE", help="a parameter held constant")

    parser.add_argument("--start", type=int, default=None, help="first frame index")
    parser.add_argument("--step", type=int, default=None,
                        help="use every Nth frame - a quick pass over a deep stack")
    parser.add_argument("--limit", type=int, default=None, help="cap on frames used")
    parser.add_argument("--scale", type=float, default=None, help="decode scale factor")
    parser.add_argument("--long-edge", type=int, default=None,
                        help="decode with the long edge at this many pixels")
    parser.add_argument("--bit-depth", default=None,
                        choices=list(bitdepth.VALID_MODES),
                        help="depth the stack is loaded and processed at (default 16)")

    parser.add_argument("--reference", default=None,
                        choices=["first", "middle", "last"], help="registration reference frame")
    parser.add_argument("--detection-width", type=int, default=None,
                        help="width frames are downsampled to before a transform is measured")
    parser.add_argument("--contrast", default=None, choices=["off", "auto", "clahe"])
    parser.add_argument("--contrast-strength", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="fuse on the GPU when there is one")
    parser.add_argument("--no-tiling", action="store_true")

    parser.add_argument("--threads", type=int, default=0, help="0 = every core")
    parser.add_argument("--format", default=None,
                        help="output format (default .jxl), e.g. .jxl .png .tif .dng")
    parser.add_argument("--preview-format", default=None,
                        help="8-bit copy alongside each render; '' to skip")
    parser.add_argument("--crops", type=int, default=None, help="how many crop regions")
    parser.add_argument("--crop-size", type=int, default=None, help="crop side in pixels")
    parser.add_argument("--metric-frames", type=int, default=None,
                        help="frames the no-reference metrics are measured against")
    parser.add_argument("--save-aligned", action="store_true",
                        help="also write the registered frames")
    parser.add_argument("--reference-render", default=None,
                        help="another program's render of this capture, to score "
                             "every candidate against; '' to drop a suite's own")
    return parser


def apply_overrides(plan: Plan, args: argparse.Namespace) -> Plan:
    """Let the command line override a named suite's own settings."""
    stack = plan.stack
    overrides = {
        name: getattr(args, name)
        for name in ("start", "step", "limit", "scale")
        if getattr(args, name) is not None
    }
    if args.long_edge is not None:
        overrides["long_edge"] = args.long_edge
    if args.bit_depth is not None:
        overrides["bit_depth_mode"] = args.bit_depth
    if args.source:
        overrides["source"] = args.source
    if overrides:
        stack = StackSpec(**{**stack.__dict__, **overrides})

    for suite in plan.suites:
        registration = suite.registration.__dict__.copy()
        if args.registration is not None:
            registration["method"] = args.registration
        if args.reference is not None:
            registration["reference_mode"] = args.reference
        if args.detection_width is not None:
            registration["downscale_width"] = args.detection_width
        suite.registration = RegistrationSpec(**registration)

        if args.contrast is not None:
            suite.contrast_method = args.contrast
        if args.contrast_strength is not None:
            suite.contrast_strength = args.contrast_strength
        if args.gpu:
            suite.use_gpu = True
        if args.no_tiling:
            suite.tile_enabled = False

    return Plan(
        stack=stack,
        destination=args.dest or plan.destination,
        suites=plan.suites,
        thread_count=args.threads or plan.thread_count,
        output_format=args.format or plan.output_format,
        preview_format=(plan.preview_format if args.preview_format is None
                        else args.preview_format),
        crop_count=plan.crop_count if args.crops is None else args.crops,
        crop_size=plan.crop_size if args.crop_size is None else args.crop_size,
        metric_source_limit=(plan.metric_source_limit if args.metric_frames is None
                             else args.metric_frames),
        save_aligned=args.save_aligned or plan.save_aligned,
        reference=(plan.reference if args.reference_render is None
                   else (args.reference_render or None)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tests import render_matrix_suites as suites

    args = build_parser().parse_args(argv)

    if args.list:
        print("Named suites (tests/render_matrix_suites.py):\n")
        for name, builder in sorted(suites.SUITES.items()):
            print(f"  {name:<28} {(builder.__doc__ or '').strip().splitlines()[0]}")
        return 0

    if args.suite:
        plan = suites.build(args.suite)
    elif args.fusion:
        if not args.source or not args.dest:
            raise SystemExit("--source and --dest are required without --suite")
        plan = Plan(
            stack=StackSpec(source=args.source),
            destination=args.dest,
            suites=[Suite(
                fusion=args.fusion,
                variants=sweep_variants(args.sweep, args.fixed),
                registration=RegistrationSpec(method=args.registration or "homography+ecc"),
            )],
        )
    else:
        raise SystemExit("Nothing to do: pass --suite, or --fusion with --source/--dest. "
                         "--list shows the named suites.")

    plan = apply_overrides(plan, args)
    total = sum(len(suite.variants) for suite in plan.suites)
    print(f"{len(plan.suites)} suite(s), {total} render(s) -> {plan.destination}",
          flush=True)

    for run_dir in run_plan(plan):
        print(f"\nWrote {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
