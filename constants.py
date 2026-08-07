WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 950

TILE_BLOCK_SIZE = 1024
TILE_OVERLAP = 256
TILE_THRESHOLD = 2048
TILE_BLOCK_MIN = 64
TILE_BLOCK_MAX = 16384
TILE_OVERLAP_MAX = 4096
TILE_THRESHOLD_MIN = 256
TILE_THRESHOLD_MAX = 131072

REG_DOWNSCALE_WIDTH = 1024
ECC_PARALLEL = True
HOMOGRAPHY_DOWNSCALE_WIDTH = 1600
ECC_DOWNSCALE_WIDTH = 1000

# Which frame the registration chain is referenced to. 'first' keeps the
# historical behaviour (frame 0 held fixed); 'middle' halves the maximum chain
# length, spreading accumulated drift symmetrically across the stack; 'last'
# anchors on the final frame.
REFERENCE_FRAME_MODE = "first"
REFERENCE_FRAME_MODES = ("first", "middle", "last")

STATUS_UPDATE_INTERVAL_MS = 2000

BYTES_TO_GB = 1024 ** 3

FEATURE_MAX_KEYPOINTS = 200
FEATURE_MATCH_THRESHOLD = 0.01
FEATURE_RANSAC_REJECTS = 30
FEATURE_RANSAC_MAX_ITER = 3

MAGNIFIER_SIZE = 320
ANIMATION_DURATION_MS = 400

DEFAULT_THREAD_COUNT = 4
MIN_THREAD_COUNT = 1
MAX_THREAD_COUNT = 16

STACKMFFV4_BATCH_SIZE = 2
STACKMFFV4_BATCH_SIZE_MIN = 1
STACKMFFV4_BATCH_SIZE_MAX = 16

KERNEL_SIZE_MIN = 3
KERNEL_SIZE_MAX = 51

# DCT's kernel median-filters its focal-plane map, which holds one entry per
# block rather than per pixel, so at the default block size each step reaches
# eight times further than it does for the pixel-domain methods - and the
# distance it must cover grows with the frame: 51 spans a fifth of the map on a
# 2048-wide image and a fourteenth of it on a 6000-wide one. Measured on a
# 274-frame 2048x1364 stack, quality stops improving around 51 and is flat out
# to 301; the higher ceiling exists for the larger frames, where the same reach
# needs a larger number.
KERNEL_SIZE_MAX_DCT = 151

# Raised from 7 at 1.17.2. The rewrite in item 18 of docs/ALGORITHM_IMPROVEMENTS
# made this slider DCT's main regulariser rather than the near-inert control it
# had become, and 7 leaves isolated blocks showing: on the stack above, the
# share of flat-area blocks standing out from their neighbours runs 0.385% with
# no filtering, 0.333% at 7, 0.304% here and 0.281% at 51, against 0.103% for
# the pyramid. Past 31 the gain is small and fine in-focus detail starts being
# smoothed away with the speckle.
KERNEL_SIZE_DEFAULT_DCT = 31
KERNEL_SIZE_DEFAULT_GFF = 31
KERNEL_SIZE_DEFAULT_GFG = 7
KERNEL_SIZE_DEFAULT_DMAP = 9

# The pyramid pools its band energy over this window before frames are
# compared - the same job the slider does for the depth-map methods, one level
# down. 5 px is the method default; it is deliberately small because the
# pyramid pools again at every level, so the window at level k already covers
# 2**k times as much of the picture.
KERNEL_SIZE_DEFAULT_PYRAMID = 5

# --- DCT tuning exposed in the UI ---------------------------------------
# The grid every DCT decision is made over. Smaller follows fine detail and is
# noisier; larger is steadier but steps harder where near meets far.
DCT_BLOCK_SIZES = (4, 8, 16, 32)
DCT_BLOCK_SIZE_DEFAULT = 8

# How far below the peak energy a frame still counts as in focus on a block.
# Presets rather than a slider: the usable range ends at 0.85, and above it
# detail-free regions go back to being decided by grain (item 17 in
# docs/ALGORITHM_IMPROVEMENTS.md), which is a defect, not a preference.
# Higher keeps a never-in-focus background closer to its sharpest frame; lower
# averages more frames there and comes out smoother and flatter.
DCT_PLATEAU_PRESETS = (("crisp", 0.85), ("balanced", 0.80), ("smooth", 0.70))
DCT_PLATEAU_DEFAULT = "balanced"

# Blend neighbouring frames across the block lattice (default), or copy each
# block from a single frame. Off restores verbatim source pixels at the cost of
# the lattice showing again.
DCT_BLEND_DEFAULT = True

# --- Pyramid tuning exposed in the UI ------------------------------------
# See item 19 in docs/ALGORITHM_IMPROVEMENTS.md for what each control is for
# and what it was measured to do; fusion_methods/pyramid.py holds the defaults
# themselves, and these are the values the UI offers.

# Decomposition depth. 0 means "let the method decide", which is 5 levels
# clamped to whatever the image can carry.
PYRAMID_LEVELS = (0, 2, 3, 4, 5, 6, 7, 8)
PYRAMID_LEVELS_DEFAULT = 0

# How sharply each band's weights favour the sharpest frame. The last preset is
# the published choose-max rule, kept because it is the published rule - it is
# also what stitches a defocused background out of frames that disagree.
PYRAMID_SELECTIVITY_PRESETS = (
    ("average", 2.0), ("soft", 4.0), ("balanced", 8.0),
    ("strict", 32.0), ("winner", float("inf")),
)
PYRAMID_SELECTIVITY_DEFAULT = "balanced"

# How much of a band's decision comes from the coarser bands above it. Off by
# default: it costs sharpness at a depth boundary, and on the test fixtures the
# envelope clamp already removes what it was there to prevent. Worth reaching
# for when the bands visibly disagree - fine detail sitting on a base that came
# from somewhere else.
PYRAMID_COHERENCE_PRESETS = (
    ("off", 0.0), ("light", 0.25), ("medium", 0.5), ("strong", 0.75),
)
PYRAMID_COHERENCE_DEFAULT = "off"

# How hard the coarse base band follows the frames that won the detail bands.
# "Mean" is the plain average of every frame, which hazes the base whenever
# most of the stack is defocused.
PYRAMID_BASE_PRESETS = (
    ("mean", 0.0), ("gentle", 1.0), ("balanced", 3.0), ("strong", 8.0),
)
PYRAMID_BASE_DEFAULT = "balanced"

# Compare frames in units of their own grain rather than absolutely, so a
# bright noisy frame cannot win the regions where nothing is in focus.
PYRAMID_NOISE_GATE_DEFAULT = True

# Hold every pixel inside the range its own frames span, so collapsing the
# pyramid cannot reconstruct a value no frame had.
PYRAMID_ENVELOPE_DEFAULT = True

# --- Depth Map (Max) coherent-depth tuning exposed in the UI --------------
# What stops a region no frame ever resolves from coming out as a mosaic of
# hard-edged patches taken from frames at opposite ends of the stack; see the
# block above DEFAULT_DEPTH_SMOOTHING in fusion_methods/depthmap.py for the
# mechanism and what each one was measured to fix. Both at 0 is the plain hard
# per-pixel select these defaults replaced.

# How far a confident pixel's depth is allowed to propagate into its
# unresolvable neighbours. Higher fills larger dead regions coherently; lower
# leaves each pixel closer to the frame its own measurement named.
DEPTH_SMOOTHING_DEFAULT = 50

# --- Depth Map (Avg) selectivity exposed in the UI ------------------------
# How sharply the blend favours the frame that holds the detail. Linear
# weighting only selects while the stack is short - a defocused frame still
# measures a fraction of the peak, and a deep stack adds that fraction up
# hundreds of times until the blend is the plain mean of everything and the
# result is veiled. See the block above DEFAULT_SELECTIVITY in
# fusion_methods/depthmap.py. 0 is that linear weighting; higher trades the
# multi-frame noise reduction of the regions that genuinely have nothing to
# choose between for detail in the regions that do.
AVERAGE_SELECTIVITY_DEFAULT = 50

# --- Depth Map (Avg) weight coherence exposed in the UI --------------------
# Selectivity above decides how sharply the blend leans on the frame that holds
# the detail; these two decide how far that decision is allowed to be pooled
# before it is applied - one across the frame, one along the stack. See the
# blocks above DEFAULT_COHERENCE_RADIUS and DEFAULT_SLICE_RADIUS in
# fusion_methods/depthmap.py for what each was measured to fix.
#
# Both default to off, which is what every render did while they were reachable
# only from the test harness, so exposing them changes no existing result.

# Spatial reach, in pixels. The filter is edge-aware, so it costs nothing over a
# region the blend already agreed with itself, and a great deal over one whose
# correct frame changes every few pixels: measured best at 16 on a 333-frame
# capture of a real surface, and best at 0-4 on the banded synthetic stacks in
# tests/fusion_scenarios.py, which step depth every 27 px. The ceiling sits past
# the measured optimum rather than past the point of harm, because where the
# harm starts is a property of the scene and not of the number.
COHERENCE_RADIUS_MAX = 32
COHERENCE_RADIUS_DEFAULT = 0

# Reach along the stack, in slices either side. The right value follows the
# capture's sampling rather than its content - about half the half-maximum width
# of the focus curve, so 4-5 on a sweep that oversamples its depth of field
# sevenfold, and 0 on one that steps a full depth of field per frame, where the
# neighbouring slices are the ones that resolve the pixel worst. Overshooting is
# not subtle: at 8 slices on the stack it was measured on, every fifth of the
# frame got worse at once. The ceiling is generous against that measurement
# because a denser sweep wants proportionally more, not because 16 is safe.
#
# It is also the one dial here that depends on registration, since it averages
# different frames into one pixel - see DEFAULT_SLICE_RADIUS.
#
# depthmap_impl clamps this to (frames - 1) // 2 on top of the ceiling, which is
# what keeps it honest on a stack too short to pool over. That clamp is the
# engine's rather than the slider's, so a subset render narrows it further
# without the panel having to track the tick boxes.
SLICE_RADIUS_MAX = 16
SLICE_RADIUS_DEFAULT = 0

# --- Depth Map halo-suppression ceiling ------------------------------------
# The halo dial fills the band around a sharply focused region with that frame's
# defocused pixels. Where there really was a glow that is the trade it is sold
# on; where there was not, it is just a band of defocus - and the focus measure
# cannot tell the two apart, because the energy being spread is real either way.
# So the dial does it around *every* contour in the frame, and the result reads
# as soft blobs roughly 2r across hugging every edge, every dust speck and every
# silhouette.
#
# That cost scales with the radius and nothing else, which is what makes a
# ceiling the place to draw the line. Measured on the reference ant stack at
# kernel 5, share of the frame moved by more than 8 levels against the same
# render with the dial off: r=2 8.5%, r=5 18.3%, r=9 24.8%, r=15 29.9% - clean,
# faint, obvious, bad. Nothing about the scene changed; only the radius did.
#
# Tied to the kernel because the radius is a claim on ground the measurement has
# to be able to speak for, and a pooling window of side k is what decides how far
# that is. This is the module's own long-standing advice - "keep the radius well
# under the kernel size", in the block above DEFAULT_HALO_RADIUS - turned into a
# range the UI enforces instead of a sentence in a comment.
#
# The engine is deliberately left permissive: this bounds the slider, not
# depthmap_impl, so a scripted caller with a genuinely wide glow to fight can
# still ask for more (tests/test_depthmap_halo.py needs r=8 at k=9 on a fixture
# blurred by 31 px, which is exactly that case).
HALO_RADIUS_MAX = 30


def halo_radius_ceiling(kernel_size: int) -> int:
    """Largest halo radius the UI offers at this pooling window."""
    return max(0, min(HALO_RADIUS_MAX, int(kernel_size)))

