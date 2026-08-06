"""
Render the worked examples used by quality-metrics-vocabulary.html.

Every figure on that sheet is real output: a synthetic stack, fused by the
actual methods in tests/fusion_registry.py and aligned by the actual pipeline in
core/registration.py, scored by the actual metrics in tests/fusion_metrics.py.
Nothing is drawn by hand, and no number in a caption is typed in - the measured
values are written to figures/values.json and injected into the document next to
the images, so a caption cannot drift away from the picture above it.

Two things had to be got right for the artefact figures to mean anything, and
both are worth knowing before reading them:

* ``make_stack`` in tests/synthetic_stack.py gives every frame the *same*
  blurred image outside its own focus band. A block method that picks the wrong
  frame there therefore copies identical pixels, so no seam can appear however
  badly it chooses. ``graded_stack`` below blurs each band by its distance from
  the focused one, the way a real lens does, which is what makes a wrong choice
  cost something visible.

* Every artefact metric has a *content floor*: the pristine reference image
  scores well above zero on all of them, because real detail sometimes happens
  to lie on the block lattice. The floor is measured and reported beside each
  figure, since only the excess over it is the artefact.

Run it from the repository root:

    python docs/algorithms/make_vocabulary_figures.py

It writes docs/algorithms/figures/*.png and values.json, then fills in the
<img data-fig="..."> and <span data-val="..."> placeholders in the HTML with the
figures inlined as base64, so the document stays a single self-contained file.
"""

import base64
import json
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from core.registration import ImageRegistration
from tests import fusion_metrics as fm
from tests import fusion_registry
from tests.synthetic_stack import make_reference

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
DOCUMENT = os.path.join(HERE, "quality-metrics-vocabulary.html")

SIZE = 768          # fixture edge, px
TILE = 384          # px per figure tile - 32 mm at 300 dpi, so it prints sharp
BLOCK = 8           # the lattice the block figures are measured on
values = {}


# ------------------------------------------------------------------ helpers

def save(name, img):
    """Write one tile, forcing it to TILE x TILE so the page grid stays even."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[0] != TILE or img.shape[1] != TILE:
        interp = cv2.INTER_NEAREST if img.shape[0] < TILE else cv2.INTER_AREA
        img = cv2.resize(img, (TILE, TILE), interpolation=interp)
    cv2.imwrite(os.path.join(FIGURES, f"{name}.png"), img)


def record(key, value, fmt="{:.2f}"):
    """Remember a measured value for the caption that quotes it."""
    values[key] = fmt.format(value) if isinstance(value, float) else str(value)
    return value


def crop(img, y, x, size=TILE):
    return img[y:y + size, x:x + size]


def heat(gray):
    """Render a single-channel map as an ink-on-paper ramp, not a rainbow."""
    norm = gray.astype(np.float32)
    norm = (norm - norm.min()) / max(norm.max() - norm.min(), 1e-6)
    ramp = np.stack([1.0 - norm * 0.95, 1.0 - norm * 0.88, 1.0 - norm * 0.55], axis=2)
    return (ramp * 255).astype(np.uint8)


def graded_stack(reference, slices=5, step=3.2):
    """
    A focus stack whose defocus grows with distance from the focused band.

    Frame k renders band b blurred by ``|b - k| * step``, so every frame differs
    everywhere from every other - which is what makes a wrong per-block choice
    produce a visible step instead of copying identical pixels.
    """
    height = reference.shape[0]
    edges = np.linspace(0, height, slices + 1).round().astype(int)
    cache = {0.0: reference.astype(np.float32)}
    frames = []
    for k in range(slices):
        out = np.zeros(reference.shape, np.float32)
        for b in range(slices):
            sigma = abs(b - k) * step
            if sigma not in cache:
                cache[sigma] = cv2.GaussianBlur(reference, (0, 0), sigma).astype(np.float32)
            out[edges[b]:edges[b + 1]] = cache[sigma][edges[b]:edges[b + 1]]
        frames.append(np.clip(out, 0, 255).astype(np.uint8))
    return frames, edges


def naive_block_selection(stack, block=BLOCK):
    """
    Textbook per-block winner-takes-all selection: no median filter over the
    decisions, no overlap, no blending across boundaries.

    This is the algorithm the seam and speckle metrics were written to catch. It
    is *not* what OpenFocus ships - the DCT method defends against exactly this
    with its kernel_size median filter - and it is built here so the defect can
    be shown next to a render that does not have it.
    """
    height, width = stack[0].shape[:2]
    height, width = (height // block) * block, (width // block) * block
    energies = []
    for frame in stack:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[:height, :width]
        energy = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
        energies.append(cv2.resize(energy, (width // block, height // block),
                                   interpolation=cv2.INTER_AREA))
    choice = np.argmax(np.stack(energies), axis=0)
    out = np.zeros((height, width, 3), np.uint8)
    for k, frame in enumerate(stack):
        mask = cv2.resize((choice == k).astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        out[mask] = frame[:height, :width][mask]
    return out


def degrade_to_psnr(img, target_db, kind):
    """Add noise, or blur, until the result sits at `target_db` against `img`."""
    lo, hi = (0.1, 80.0) if kind == "noise" else (0.1, 12.0)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 1.0, img.shape).astype(np.float32)
    out = img
    for _ in range(40):
        mid = (lo + hi) / 2
        if kind == "noise":
            out = np.clip(img.astype(np.float32) + noise * mid, 0, 255).astype(np.uint8)
        else:
            out = cv2.GaussianBlur(img, (0, 0), mid)
        if fm.psnr(out, img) > target_db:
            lo = mid
        else:
            hi = mid
    return out


def phase_residual(images):
    """Leftover misalignment of each frame against the first, in pixels."""
    first = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = []
    for image in images[1:]:
        other = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(first, other)
        out.append(float(np.hypot(dx, dy)))
    return out


# ------------------------------------------------------------------ fixtures

print("building the fixtures ...")
REFERENCE = make_reference(height=SIZE, width=SIZE, seed=3, style="photographic")
STACK, BANDS = graded_stack(REFERENCE)

# Broadband noise: the artefact metrics have a near-zero floor on it, which is
# what a speckle figure needs. On the photographic reference the same metric
# floors at 13 %, because the thin diagonals read as speckle at an 8 px block.
TEXTURE = make_reference(height=SIZE, width=SIZE, seed=3, style="texture")
TEXTURE_STACK, _ = graded_stack(TEXTURE)

PYRAMID = fusion_registry.get("pyramid")
DCT = fusion_registry.get("dct")
FUSED_PYRAMID = PYRAMID.fuse(STACK, **PYRAMID.params)
FUSED_DCT, _, _ = fm.align_to_common_size(DCT.fuse(STACK, **DCT.params), STACK, REFERENCE)
NAIVE = naive_block_selection(STACK)

def zoom(img, y, x, source=TILE // 2):
    """
    A `source`-wide crop blown up to a full tile, nearest-neighbour.

    Block seams and speckle are pixel-scale defects. Printed at 1:1 in a 32 mm
    tile they are below what the page can resolve, so the artefact figures are
    shown at 2x - magnified, not enhanced: nearest-neighbour invents nothing.
    """
    patch = img[y:y + source, x:x + source]
    return cv2.resize(patch, (TILE, TILE), interpolation=cv2.INTER_NEAREST)


def save_centre_zoom(name, img, factor=2):
    """
    Save the middle of `img` magnified, for comparisons whose difference is
    pixel-scale. The metric beside it is still computed on the whole crop - this
    magnifies the evidence, it does not change what was measured.
    """
    source = TILE // factor
    offset = (img.shape[0] - source) // 2
    save(name, zoom(img, offset, offset, source=source))


def busiest_window(img, size=TILE):
    """Top-left of the `size` window with the most edge energy - shapes and lines,
    where a misalignment shows as a double edge."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pooled = cv2.boxFilter(np.abs(cv2.Laplacian(grey, cv2.CV_32F)), -1, (size, size))
    half = size // 2
    inner = pooled[half:img.shape[0] - half, half:img.shape[1] - half]
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return y, x


def most_different_window(a, b, size=TILE // 2):
    """
    Top-left of the `size` window where two renders disagree most.

    A defect metric can be right about an image while the window someone happened
    to crop shows none of it. Letting the difference choose the window is what
    makes the figure evidence for the number beside it rather than an assertion.
    """
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    delta = np.abs(a[:height, :width].astype(np.float32) -
                   b[:height, :width].astype(np.float32)).max(axis=2)
    pooled = cv2.boxFilter(delta, -1, (size, size))
    half = size // 2
    inner = pooled[half:height - half, half:width - half]
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return y, x


def quietest_window(img, size=TILE):
    """Top-left of the `size` window with the least detail - the flat background."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
    pooled = cv2.boxFilter(detail, -1, (size, size), normalize=True)
    half = size // 2
    inner = pooled[half:img.shape[0] - half, half:img.shape[1] - half]
    y, x = np.unravel_index(int(np.argmin(inner)), inner.shape)
    return y, x


# A window straddling a depth boundary - where the interesting failures live -
# and one on the smoothest part of the frame, where a seam is least forgivable.
EDGE_Y, EDGE_X = BANDS[2] - TILE // 2, 200
FLAT_Y, FLAT_X = quietest_window(REFERENCE)


# ------------------------------------------------------- page 3: full reference

def figures_full_reference():
    print("full-reference ...")
    truth = crop(REFERENCE, EDGE_Y, EDGE_X)
    save("psnr_truth", truth)
    for db in (34.0, 26.0):
        shown = degrade_to_psnr(truth, db, "noise")
        record(f"psnr_{int(db)}", fm.psnr(shown, truth), "{:.1f}")
        save(f"psnr_{int(db)}", shown)

    # The demonstration that matters: two degradations matched on PSNR, so the
    # only thing left that can tell them apart is SSIM.
    blurred = degrade_to_psnr(truth, 30.0, "blur")
    noised = degrade_to_psnr(truth, 30.0, "noise")
    save("ssim_blur", blurred)
    save("ssim_noise", noised)
    record("ssim_blur_psnr", fm.psnr(blurred, truth), "{:.1f}")
    record("ssim_noise_psnr", fm.psnr(noised, truth), "{:.1f}")
    record("ssim_blur_ssim", fm.ssim(blurred, truth), "{:.3f}")
    record("ssim_noise_ssim", fm.ssim(noised, truth), "{:.3f}")

    # Boundary band: the strip the region metric scores, and a halo inside it
    masks = []
    for k in range(len(BANDS) - 1):
        mask = np.zeros((SIZE, SIZE), np.float32)
        mask[BANDS[k]:BANDS[k + 1]] = 1.0
        masks.append(mask)
    band = fm.boundary_band(masks, width=6)
    overlay = REFERENCE.copy()
    overlay[band] = (0.3 * overlay[band] + 0.7 * np.array([70, 70, 240])).astype(np.uint8)
    save("region_band", crop(overlay, EDGE_Y, EDGE_X))

    glow = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 6)[:, :, None]
    haloed = np.clip(FUSED_PYRAMID.astype(np.float32) + glow * 60, 0, 255).astype(np.uint8)
    save("region_halo", crop(haloed, EDGE_Y, EDGE_X))
    record("region_whole_psnr", fm.psnr(haloed, REFERENCE), "{:.1f}")
    record("region_band_psnr", fm.region_psnr(haloed, REFERENCE, band), "{:.1f}")

    # Colour error: a cast that leaves structure untouched
    cast = np.clip(REFERENCE.astype(np.float32) * np.array([1.06, 1.0, 0.94]),
                   0, 255).astype(np.uint8)
    save("colour_ref", crop(REFERENCE, EDGE_Y, EDGE_X))
    save("colour_cast", crop(cast, EDGE_Y, EDGE_X))
    record("colour_error", fm.colour_error(cast, REFERENCE), "{:.1f}")
    record("colour_psnr", fm.psnr(cast, REFERENCE), "{:.1f}")
    record("colour_ssim", fm.ssim(cast, REFERENCE), "{:.3f}")


# --------------------------------------------------------- page 5: no reference

def figures_no_reference():
    print("no-reference ...")
    sharp = crop(REFERENCE, EDGE_Y, EDGE_X)
    softened = cv2.GaussianBlur(sharp, (0, 0), 1.6)
    rng = np.random.default_rng(11)
    noisy = np.clip(sharp.astype(np.float32) + rng.normal(0, 11, sharp.shape),
                    0, 255).astype(np.uint8)
    for key, img in (("sf_sharp", sharp), ("sf_soft", softened), ("sf_noise", noisy)):
        save_centre_zoom(key, img)
        record(f"{key}_sf", fm.spatial_frequency(img), "{:.1f}")
        record(f"{key}_entropy", fm.entropy(img), "{:.2f}")
        record(f"{key}_std", fm.std_dev(img), "{:.1f}")

    # QABF: one source frame, the same stack fused well, and fused into mush
    mush = np.mean(np.stack(STACK).astype(np.float32), axis=0).astype(np.uint8)
    save_centre_zoom("qabf_source", crop(STACK[0], EDGE_Y, EDGE_X))
    save_centre_zoom("qabf_good", crop(FUSED_PYRAMID, EDGE_Y, EDGE_X))
    save_centre_zoom("qabf_mush", crop(mush, EDGE_Y, EDGE_X))
    record("qabf_good", fm.qabf(FUSED_PYRAMID, STACK), "{:.3f}")
    record("qabf_mush", fm.qabf(mush, STACK), "{:.3f}")
    record("qabf_source", fm.qabf(STACK[0], STACK), "{:.3f}")
    record("qabf_noise", fm.qabf(np.clip(FUSED_PYRAMID.astype(np.float32) +
                                         rng.normal(0, 11, FUSED_PYRAMID.shape),
                                         0, 255).astype(np.uint8), STACK), "{:.3f}")

    # Entropy is blind to arrangement: shuffling every pixel cannot move it
    shuffled = sharp.reshape(-1, 3).copy()
    np.random.default_rng(5).shuffle(shuffled)
    shuffled = shuffled.reshape(sharp.shape)
    save_centre_zoom("entropy_shuffled", shuffled)
    record("entropy_shuffled", fm.entropy(shuffled), "{:.2f}")
    record("sf_shuffled", fm.spatial_frequency(shuffled), "{:.1f}")

    # Std dev: what hedging between frames costs in contrast
    save_centre_zoom("std_fused", crop(FUSED_PYRAMID, EDGE_Y, EDGE_X))
    save_centre_zoom("std_washed", crop(mush, EDGE_Y, EDGE_X))
    record("std_fused", fm.std_dev(FUSED_PYRAMID), "{:.1f}")
    record("std_washed", fm.std_dev(mush), "{:.1f}")


# ------------------------------------------------------------ page 7: artefacts

def figures_artefacts():
    print("artefacts ...")
    # The content floor: what the pristine reference already scores
    floor_excess, floor_visible = fm.block_seams(REFERENCE, block=BLOCK)
    record("floor_seam_excess", floor_excess)
    record("floor_seam_visible", floor_visible, "{:.1f}")
    record("floor_defocus_excess", fm.defocus_seams(REFERENCE, block=BLOCK)[0])
    record("floor_defocus_visible", fm.defocus_seams(REFERENCE, block=BLOCK)[1], "{:.1f}")

    # Let the disagreement pick the window, so the picture is evidence for the
    # number beside it rather than a smooth patch that shows neither.
    seam_y, seam_x = most_different_window(NAIVE, FUSED_PYRAMID)
    record("seam_window", f"{TILE // 2} px shown at 2x", "{}")

    for key, image in (("seam_naive", NAIVE), ("seam_dct_b8_k7", FUSED_DCT),
                       ("seam_pyramid", FUSED_PYRAMID)):
        save(key, zoom(image, seam_y, seam_x))
        excess, visible = fm.block_seams(image, block=BLOCK)
        record(f"{key}_excess", excess)
        record(f"{key}_visible", visible, "{:.1f}")
        record(f"{key}_psnr", fm.psnr(image, REFERENCE[:image.shape[0], :image.shape[1]]), "{:.1f}")

    # Where the metric is looking: the lattice itself, amplified six times
    grey = cv2.cvtColor(NAIVE, cv2.COLOR_BGR2GRAY).astype(np.float32)
    step = np.abs(np.diff(grey, axis=1))
    lattice = np.zeros_like(step)
    lattice[:, BLOCK - 1::BLOCK] = step[:, BLOCK - 1::BLOCK]
    save("seam_lattice", heat(zoom(np.clip(lattice * 14, 0, 255).astype(np.uint8),
                                   seam_y, seam_x)))

    # Defocus seams: the same defect where there is nothing to hide behind. The
    # window has to be genuinely flat, so detail rules windows out first and the
    # disagreement only chooses among what is left.
    height, width = NAIVE.shape[:2]
    span = TILE // 2
    detail = cv2.boxFilter(np.abs(cv2.Laplacian(
        cv2.cvtColor(REFERENCE, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)),
        -1, (span, span))[:height, :width]
    delta = np.abs(NAIVE.astype(np.float32) -
                   FUSED_PYRAMID[:height, :width].astype(np.float32)).max(axis=2)
    disagreement = cv2.boxFilter(delta, -1, (span, span))
    disagreement[detail > np.percentile(detail, 25)] = -1.0
    half = span // 2
    inner = disagreement[half:height - half, half:width - half]
    flat_y, flat_x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    save("defocus_naive", zoom(NAIVE, flat_y, flat_x))
    save("defocus_pyramid", zoom(FUSED_PYRAMID, flat_y, flat_x))

    # The distinctive part of this metric is not the defect but the region: the
    # quiet mask it restricts itself to, computed the way the metric computes it.
    grey_naive = cv2.cvtColor(NAIVE, cv2.COLOR_BGR2GRAY).astype(np.float32)
    texture = cv2.blur(np.abs(grey_naive - cv2.blur(grey_naive, (9, 9))), (33, 33))
    quiet = texture <= np.percentile(texture, 40.0)
    shown = NAIVE.copy()
    shown[~quiet] = (shown[~quiet] * 0.25 + 190 * 0.75).astype(np.uint8)
    save("defocus_region", shown)
    record("defocus_quiet_share", 100.0 * quiet.mean(), "{:.0f}")
    excess, visible = fm.defocus_seams(NAIVE, block=BLOCK)
    record("defocus_naive_excess", excess)
    record("defocus_naive_visible", visible, "{:.1f}")
    record("defocus_pyramid_visible", fm.defocus_seams(FUSED_PYRAMID, block=BLOCK)[1], "{:.1f}")

    # Speckle, on the broadband fixture where the metric floors near zero. Dust
    # that is sharp in exactly one frame is what makes a lone block flip.
    clean = naive_block_selection(TEXTURE_STACK)
    dusty = [f.copy() for f in TEXTURE_STACK]
    rng = np.random.default_rng(4)
    for _ in range(80):
        cv2.circle(dusty[-1], (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))),
                   2, (252, 252, 252), -1)
    speckled = naive_block_selection(dusty)
    # A tighter zoom than the seam figures: a flipped block is 8 px across, and
    # the point is that it is one square, not a smudge.
    spot_y, spot_x = most_different_window(speckled, clean, size=TILE // 4)
    save("speckle_clean", zoom(clean, spot_y, spot_x, source=TILE // 4))
    save("speckle_dust", zoom(speckled, spot_y, spot_x, source=TILE // 4))
    record("speckle_floor", fm.block_speckle(TEXTURE, block=BLOCK))
    record("speckle_clean", fm.block_speckle(clean, block=BLOCK))
    record("speckle_dust", fm.block_speckle(speckled, block=BLOCK))

    # Envelope: values no source frame ever held, marked where they occur
    lo = np.min(np.stack(STACK), axis=0).astype(np.int32)
    hi = np.max(np.stack(STACK), axis=0).astype(np.int32)
    # A gentle unsharp mask - the amount a photographer would not think twice
    # about - is already enough to leave the envelope, which is what makes this
    # a veto rather than a score.
    ringing = cv2.addWeighted(FUSED_PYRAMID, 1.08,
                              cv2.GaussianBlur(FUSED_PYRAMID, (0, 0), 2.5), -0.08, 0)
    result = ringing.astype(np.int32)
    outside = np.any((result < lo) | (result > hi), axis=2)
    excursion = np.maximum(np.maximum(result - hi, lo - result), 0).max(axis=2)
    save("envelope_clean", crop(FUSED_PYRAMID, EDGE_Y, EDGE_X))
    save("envelope_ringing", crop(ringing, EDGE_Y, EDGE_X))
    save("envelope_marked", heat(crop(excursion.astype(np.float32), EDGE_Y, EDGE_X)))
    record("envelope_over", int(np.maximum(result - hi, 0).max()), "{}")
    record("envelope_under", int(np.maximum(lo - result, 0).max()), "{}")
    record("envelope_share", 100.0 * outside.mean(), "{:.1f}")
    good = FUSED_PYRAMID.astype(np.int32)
    record("envelope_good", int(max(np.maximum(good - hi, 0).max(),
                                    np.maximum(lo - good, 0).max())), "{}")


# ---------------------------------------------------- page 9: foreign reference

def figures_foreign():
    print("foreign-reference ...")
    ours = FUSED_PYRAMID

    # Stand in for another program's render of the same capture: its own grade,
    # its own sharpening, and its own alignment. A real foreign render differs
    # by all three, and only the third is invisible to a histogram.
    graded = np.clip(ours.astype(np.float32) * 1.18 - 16, 0, 255).astype(np.uint8)
    graded = cv2.addWeighted(graded, 1.25, cv2.GaussianBlur(graded, (0, 0), 2), -0.25, 0)
    theirs = cv2.warpAffine(graded, np.float32([[1, 0, 5], [0, 1, -3]]),
                            (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)

    save("tone_ours", crop(ours, EDGE_Y, EDGE_X))
    save("tone_theirs", crop(theirs, EDGE_Y, EDGE_X))
    save("tone_matched", crop(fm.match_tone(theirs, ours), EDGE_Y, EDGE_X))
    # Measured without the shift, so the number reports what regrading alone
    # recovers; the shift is the separate problem the tiles exist to solve.
    record("tone_before_psnr", fm.psnr(graded, ours), "{:.1f}")
    record("tone_after_psnr", fm.psnr(fm.match_tone(graded, ours), ours), "{:.1f}")
    record("tone_shifted_psnr", fm.psnr(fm.match_tone(theirs, ours), ours), "{:.1f}")

    save("detail_source", crop(ours, EDGE_Y, EDGE_X))
    ours_map = fm.detail_map(ours, block=64)
    save("detail_map_ours", heat(ours_map))
    save("detail_map_theirs", heat(fm.detail_map(fm.match_tone(theirs, ours), block=64)))
    record("detail_tiles", f"{ours_map.shape[1]} x {ours_map.shape[0]}", "{}")

    agreement, share = fm.detail_agreement(ours, theirs, block=64)
    record("detail_agreement", agreement, "{:.3f}")
    record("detail_share", share, "{:.2f}")

    # A soft render: agreement largely survives, share collapses. Which is the
    # whole reason the two are never quoted apart.
    soft = cv2.GaussianBlur(ours, (0, 0), 2.2)
    soft_agreement, soft_share = fm.detail_agreement(soft, theirs, block=64)
    save("detail_soft", crop(soft, EDGE_Y, EDGE_X))
    save("detail_map_soft", heat(fm.detail_map(soft, block=64)))
    record("detail_soft_agreement", soft_agreement, "{:.3f}")
    record("detail_soft_share", soft_share, "{:.2f}")


# --------------------------------------------------------- page 11: registration

def figures_registration():
    print("registration ...")
    drifts = [(0.0, 0.0), (9.0, -6.0), (-7.0, 11.0), (14.0, 4.0)]
    frames = [cv2.warpAffine(REFERENCE, np.float32([[1, 0, dx], [0, 1, dy]]),
                             (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
              for dx, dy in drifts]

    def overlay(a, b):
        """Two frames in one picture, so a misalignment reads as a double edge."""
        return cv2.addWeighted(a, 0.5, b, 0.5, 0)

    before = phase_residual(frames)
    registered = ImageRegistration(method=["homography"]).process(
        [f.copy() for f in frames], output_path=None, thread_count=4)
    after = phase_residual(registered)

    # Where the two frames disagree most, which is where the misalignment reads
    # as a double edge; a smooth gradient hides a fourteen-pixel error entirely.
    ghost = overlay(frames[0], frames[3])
    busy_y, busy_x = most_different_window(frames[0], frames[3])
    save("reg_unregistered", zoom(ghost, busy_y, busy_x, source=TILE // 4))
    lost_y = (SIZE - registered[0].shape[0]) // 2
    lost_x = (SIZE - registered[0].shape[1]) // 2
    save("reg_registered", zoom(overlay(registered[0], registered[3]),
                                max(busy_y - lost_y, 0), max(busy_x - lost_x, 0),
                                source=TILE // 4))
    record("reg_before_mean", float(np.mean(before)), "{:.2f}")
    record("reg_before_worst", float(np.max(before)), "{:.2f}")
    record("reg_after_mean", float(np.mean(after)), "{:.3f}")
    record("reg_after_worst", float(np.max(after)), "{:.3f}")
    record("reg_gain", float(np.mean(before) / max(np.mean(after), 1e-9)), "{:.0f}")

    # Kept: the area the pipeline's own crop left, reported by its own output
    kept_h, kept_w = registered[0].shape[:2]
    record("reg_kept", 100.0 * kept_h * kept_w / (SIZE * SIZE), "{:.1f}")
    record("reg_kept_size", f"{kept_w} x {kept_h}", "{}")
    shown = REFERENCE.copy()
    lost = np.ones((SIZE, SIZE), bool)
    top, left = (SIZE - kept_h) // 2, (SIZE - kept_w) // 2
    lost[top:top + kept_h, left:left + kept_w] = False
    shown[lost] = (shown[lost] * 0.28).astype(np.uint8)
    cv2.rectangle(shown, (left, top), (left + kept_w, top + kept_h), (60, 200, 240), 3)
    save("reg_kept", shown)

    # Selection: which frame the focus measure sources each pixel from, once on
    # the aligned stack and once with a drift left in
    def focus_argmax(images):
        energies = [cv2.GaussianBlur(np.abs(cv2.Laplacian(
            cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)),
            (0, 0), 4) for i in images]
        return np.argmax(np.stack(energies), axis=0)

    palette = np.array([[74, 120, 214], [88, 174, 96], [196, 132, 62],
                        [150, 96, 190], [90, 190, 210]], np.uint8)
    truth = np.zeros((SIZE, SIZE), np.int64)
    for k in range(len(BANDS) - 1):
        truth[BANDS[k]:BANDS[k + 1]] = k

    aligned_choice = focus_argmax(STACK)
    drifted = [cv2.warpAffine(f, np.float32([[1, 0, 22 * (k - 2)], [0, 1, 16 * (k - 2)]]),
                              (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
               for k, f in enumerate(STACK)]
    drifted_choice = focus_argmax(drifted)

    save("select_truth", palette[truth])
    save("select_aligned", palette[aligned_choice])
    save("select_drifted", palette[drifted_choice])
    record("select_aligned", 100.0 * float((aligned_choice == truth).mean()), "{:.1f}")
    record("select_drifted", 100.0 * float((drifted_choice == truth).mean()), "{:.1f}")

    # Detection width: what the estimator is given when the frames are downscaled
    save("width_full", crop(REFERENCE, EDGE_Y, EDGE_X))
    small = cv2.resize(REFERENCE, (192, 192), interpolation=cv2.INTER_AREA)
    save("width_small", crop(cv2.resize(small, (SIZE, SIZE),
                                        interpolation=cv2.INTER_NEAREST), EDGE_Y, EDGE_X))


# ------------------------------------------------------------------- injection

def inject():
    """Inline every figure and measured value into the document."""
    html = open(DOCUMENT, encoding="utf-8").read()
    missing = []

    def data_uri(name):
        with open(os.path.join(FIGURES, f"{name}.png"), "rb") as handle:
            return "data:image/png;base64," + base64.b64encode(handle.read()).decode()

    def replace_img(match):
        name = match.group(1)
        if not os.path.exists(os.path.join(FIGURES, f"{name}.png")):
            missing.append(f"figure {name}")
            return match.group(0)
        return f'<img data-fig="{name}" src="{data_uri(name)}" alt="{name}">'

    def replace_val(match):
        name = match.group(1)
        if name not in values:
            missing.append(f"value {name}")
            return match.group(0)
        return f'<span data-val="{name}">{values[name]}</span>'

    html = re.sub(r'<img data-fig="([a-z0-9_]+)"[^>]*>', replace_img, html)
    html = re.sub(r'<span data-val="([a-z0-9_]+)"[^>]*>.*?</span>', replace_val, html)
    open(DOCUMENT, "w", encoding="utf-8").write(html)

    if missing:
        print("  MISSING:", ", ".join(sorted(set(missing))))
    return missing


def main():
    os.makedirs(FIGURES, exist_ok=True)
    figures_full_reference()
    figures_no_reference()
    figures_artefacts()
    figures_foreign()
    figures_registration()

    with open(os.path.join(FIGURES, "values.json"), "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=1, sort_keys=True)

    tiles = len([n for n in os.listdir(FIGURES) if n.endswith(".png")])
    print(f"\n{tiles} tiles, {len(values)} measured values -> {FIGURES}")
    for key in sorted(values):
        print(f"  {key:26} {values[key]}")

    if os.path.exists(DOCUMENT):
        print("\ninjecting into the document ...")
        inject()


if __name__ == "__main__":
    main()
