"""
Fusion quality metrics implemented on numpy/OpenCV only (no scikit-image).

Two families are provided:

* Full-reference (need the sharp ground truth): psnr, ssim.
* No-reference (only need the fused image and the source stack):
  entropy, spatial_frequency, std_dev, qabf.
* Artefact-specific (need only the fused image): block_seams, defocus_seams -
  how much of the block lattice a block-selection method left showing.

The no-reference set is what applies to real focus stacks, where no all-in-focus
ground truth exists; the full-reference set is used against the synthetic stacks
in tests/synthetic_stack.py.
"""

import cv2
import numpy as np

_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2


def _gray32(img):
    """Convert a BGR or single-channel uint8 image to float32 grayscale."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def psnr(fused, reference):
    """Peak signal-to-noise ratio in dB; higher is better, inf on exact match."""
    a = fused.astype(np.float64)
    b = reference.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(fused, reference, sigma=1.5):
    """Mean structural similarity in [-1, 1] on the Gaussian-weighted window."""
    a = _gray32(fused)
    b = _gray32(reference)

    mu_a = cv2.GaussianBlur(a, (0, 0), sigma)
    mu_b = cv2.GaussianBlur(b, (0, 0), sigma)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    var_a = cv2.GaussianBlur(a * a, (0, 0), sigma) - mu_aa
    var_b = cv2.GaussianBlur(b * b, (0, 0), sigma) - mu_bb
    cov_ab = cv2.GaussianBlur(a * b, (0, 0), sigma) - mu_ab

    num = (2 * mu_ab + _SSIM_C1) * (2 * cov_ab + _SSIM_C2)
    den = (mu_aa + mu_bb + _SSIM_C1) * (var_a + var_b + _SSIM_C2)
    return float(np.mean(num / den))


def entropy(img):
    """Shannon entropy of the grayscale histogram, in bits; higher = more information."""
    gray = _gray32(img).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def spatial_frequency(img):
    """
    Spatial frequency (Eskicioglu & Fisher): RMS of the row and column
    gradients. Higher means more detail retained; blurry results score low.
    """
    gray = _gray32(img)
    rf = np.mean(np.diff(gray, axis=1) ** 2)
    cf = np.mean(np.diff(gray, axis=0) ** 2)
    return float(np.sqrt(rf + cf))


def std_dev(img):
    """Grayscale standard deviation: a coarse proxy for contrast."""
    return float(np.std(_gray32(img)))


def _sobel_gradient(gray):
    """Return (magnitude, orientation) of the Sobel gradient."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx + np.finfo(np.float32).eps)
    return mag, ang


def _edge_preservation(src_gray, fused_gray):
    """Xydeas-Petrovic per-pixel edge preservation Q^{AB/F} for one source."""
    # Constants from the original paper
    gamma_g, kappa_g, sigma_g = 0.9994, -15.0, 0.5
    gamma_a, kappa_a, sigma_a = 0.9879, -22.0, 0.8

    g_src, a_src = _sobel_gradient(src_gray)
    g_fus, a_fus = _sobel_gradient(fused_gray)

    eps = np.finfo(np.float32).eps
    # Relative strength: ratio folded so it always lands in (0, 1]
    ratio = np.where(g_src > g_fus, g_fus / (g_src + eps), g_src / (g_fus + eps))
    ratio = np.clip(np.nan_to_num(ratio), 0.0, 1.0)

    # Orientation agreement, 1 when the edges point the same way
    da = 1.0 - np.abs(a_src - a_fus) / (np.pi / 2)

    q_g = gamma_g / (1.0 + np.exp(kappa_g * (ratio - sigma_g)))
    q_a = gamma_a / (1.0 + np.exp(kappa_a * (da - sigma_a)))
    return q_g * q_a, g_src


def qabf(fused, sources):
    """
    Q^{AB/F} gradient-based fusion quality (Xydeas & Petrovic, 2000), in [0, 1].

    Measures how much of the edge information present in the source stack
    survives in the fused image, weighted by source edge strength. No ground
    truth needed, so this is the headline metric for real stacks.
    """
    fused_gray = _gray32(fused)
    num = 0.0
    den = 0.0
    for src in sources:
        q, weight = _edge_preservation(_gray32(src), fused_gray)
        num += float(np.sum(q * weight))
        den += float(np.sum(weight))
    if den == 0:
        return 0.0
    return num / den


def boundary_band(masks, width=6):
    """
    Pixels within `width` of a focus-region border.

    Where two depths meet is where fusion has to make its hardest choice, so
    scoring that strip separately from the interior isolates halos and bleeding
    from overall reconstruction quality.
    """
    edge = np.zeros(masks[0].shape[:2], dtype=bool)
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8)
        grown = cv2.dilate(binary, kernel)
        shrunk = cv2.erode(binary, kernel)
        edge |= (grown != shrunk)
    return edge


def region_psnr(fused, reference, region):
    """PSNR restricted to a boolean mask; inf when the region matches exactly."""
    if not np.any(region):
        return float("inf")
    a = fused[region].astype(np.float64)
    b = reference[region].astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def colour_error(fused, reference):
    """
    Mean absolute per-channel deviation in levels (0-255).

    Luminance metrics like PSNR can look healthy while hues drift, so this reads
    the channels directly - it is what "the colours came out wrong" measures as.
    """
    a = fused.astype(np.float32)
    b = reference.astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def align_to_common_size(fused, sources, reference=None):
    """
    Crop everything to the largest geometry they share.

    Not every method returns the input size: DCT crops to a multiple of its
    block size and DTCWT pads an odd edge up to even. Comparing arrays of
    different shapes would raise, so callers measuring arbitrary methods should
    pass their images through here first. Returns (fused, sources, reference).
    """
    shapes = [fused.shape[:2]] + [s.shape[:2] for s in sources]
    if reference is not None:
        shapes.append(reference.shape[:2])

    if len(set(shapes)) == 1:
        return fused, sources, reference

    height = min(s[0] for s in shapes)
    width = min(s[1] for s in shapes)
    return (fused[:height, :width],
            [s[:height, :width] for s in sources],
            None if reference is None else reference[:height, :width])


def block_seams(fused, block=8, factor=3.0):
    """
    How much of the image's gradient sits on the block lattice, and how badly.

    A block-selection method puts its mistakes in a very particular place: a
    step exactly on a block boundary, where the two sides came from frames that
    do not match. Real detail does not know where the lattice is, so comparing
    the step across each boundary with the typical step just inside the
    neighbouring blocks isolates the artefact from the content.

    Returns (excess, visible):
      excess  - mean step on the lattice beyond the local typical step, in
                8-bit levels. 0 means the lattice is invisible.
      visible - percentage of boundary pixels stepping more than `factor` times
                the local typical step, i.e. how much of the lattice shows.

    Both are reported because they answer different questions: a hundred
    one-level steps and one hundred-level tear are equally bad by a plain mean
    gradient ratio, and only the second is a defect anyone would notice.
    """
    grey = _gray32(fused)
    if fused.dtype == np.uint16:
        grey = grey / 257.0   # score 16-bit stacks on the same 8-bit scale
    dx = np.abs(np.diff(grey, axis=1))
    dy = np.abs(np.diff(grey, axis=0))

    # Typical local step, over a neighbourhood wide enough to span a few blocks
    span = max(block * 3, 3)
    ref_x = cv2.blur(dx, (span, span))
    ref_y = cv2.blur(dy, (span, span))

    cols = np.zeros(dx.shape[1], bool)
    cols[block - 1::block] = True
    rows = np.zeros(dy.shape[0], bool)
    rows[block - 1::block] = True
    if not cols.any() or not rows.any():
        return 0.0, 0.0

    step = np.concatenate([dx[:, cols].ravel(), dy[rows, :].ravel()])
    local = np.concatenate([ref_x[:, cols].ravel(), ref_y[rows, :].ravel()])
    excess = float(np.maximum(step - local, 0.0).mean())
    visible = float(np.mean(step > factor * np.maximum(local, 0.25)) * 100.0)
    return excess, visible


def defocus_seams(fused, block=8, quiet_percentile=40.0):
    """
    `block_seams`, restricted to the parts of the picture with no detail.

    Seams are most objectionable exactly where there is nothing to hide behind -
    a defocused background - and a method can score well overall while tearing
    the background to pieces. The quiet region is measured on the fused image
    itself, so this needs no reference and works on real stacks.
    """
    grey = _gray32(fused)
    if fused.dtype == np.uint16:
        grey = grey / 257.0   # score 16-bit stacks on the same 8-bit scale
    detail = cv2.blur(np.abs(grey - cv2.blur(grey, (9, 9))), (33, 33))
    quiet = detail <= np.percentile(detail, quiet_percentile)

    dx = np.abs(np.diff(grey, axis=1))
    span = max(block * 3, 3)
    ref_x = cv2.blur(dx, (span, span))
    cols = np.zeros(dx.shape[1], bool)
    cols[block - 1::block] = True
    if not cols.any():
        return 0.0, 0.0

    mask = quiet[:, :-1][:, cols]
    if not mask.any():
        return 0.0, 0.0
    step, local = dx[:, cols][mask], ref_x[:, cols][mask]
    excess = float(np.maximum(step - local, 0.0).mean())
    visible = float(np.mean(step > 3.0 * np.maximum(local, 0.25)) * 100.0)
    return excess, visible


def evaluate(fused, sources, reference=None, block=8):
    """Collect every applicable metric into one dict."""
    seam_excess, seam_visible = block_seams(fused, block)
    bg_excess, bg_visible = defocus_seams(fused, block)
    scores = {
        "qabf": qabf(fused, sources),
        "entropy": entropy(fused),
        "spatial_frequency": spatial_frequency(fused),
        "std_dev": std_dev(fused),
        "seam_excess": seam_excess,
        "seam_visible": seam_visible,
        "defocus_seam_excess": bg_excess,
        "defocus_seam_visible": bg_visible,
    }
    if reference is not None:
        scores["psnr"] = psnr(fused, reference)
        scores["ssim"] = ssim(fused, reference)
    return scores
