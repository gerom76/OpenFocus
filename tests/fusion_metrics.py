"""
Fusion quality metrics implemented on numpy/OpenCV only (no scikit-image).

Two families are provided:

* Full-reference (need the sharp ground truth): psnr, ssim.
* No-reference (only need the fused image and the source stack):
  entropy, spatial_frequency, std_dev, qabf.
* Artefact-specific (need only the fused image): block_seams, defocus_seams -
  how much of the block lattice a block-selection method left showing.
* Foreign-reference (need a render of the same capture by other software, which
  is neither registered to the stack nor graded like it): match_tone,
  detail_map, detail_agreement.

The no-reference set is what applies to real focus stacks, where no all-in-focus
ground truth exists; the full-reference set is used against the synthetic stacks
in tests/synthetic_stack.py. The foreign-reference set is for the captures in
samples/captures.py, where a reference exists but is not a ground truth.
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


def match_tone(source, target):
    """
    Regrade `source` onto `target`'s tone, per channel, by histogram matching.

    For comparing a result against a reference that came out of other software.
    Another program's render of the same capture carries its own exposure,
    contrast curve and colour grade, and none of that is a fusion property - but
    every intensity metric reads it as error, and a large one. Matching the
    histograms removes exactly the class of difference that a global monotone
    curve can express, and nothing else: where the detail is cannot survive a
    per-channel lookup table.

    Applied per candidate rather than once, so a single frame is regraded onto
    its own tone the same way the fused result is and neither is charged for a
    grade it never chose. uint8 only, which is the scale the metrics here are
    defined on.
    """
    if source.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("match_tone works on 8-bit images")
    if source.ndim != target.ndim:
        raise ValueError("match_tone needs both images in the same layout")

    source = source if source.ndim == 3 else source[:, :, None]
    target = target if target.ndim == 3 else target[:, :, None]

    out = np.empty_like(source)
    for channel in range(source.shape[2]):
        src = np.bincount(source[:, :, channel].ravel(), minlength=256).cumsum()
        dst = np.bincount(target[:, :, channel].ravel(), minlength=256).cumsum()
        lut = np.searchsorted(dst / dst[-1], src / src[-1])
        out[:, :, channel] = lut.clip(0, 255).astype(np.uint8)[source[:, :, channel]]
    return out if out.shape[2] > 1 else out[:, :, 0]


def detail_map(img, block=64):
    """
    Mean local contrast per `block`x`block` tile: where the picture has detail.

    Pooling to tiles is what makes the map comparable across images that are not
    registered to each other. Two renders of the same capture by different
    programs sit a few pixels apart - each aligned its own way, and focus
    breathing means no single warp relates them - so a per-pixel comparison
    measures the offset. A tile several times wider than that offset does not.
    """
    grey = _gray32(img)
    if img.dtype == np.uint16:
        grey = grey / 257.0
    energy = np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3))
    height = (energy.shape[0] // block) * block
    width = (energy.shape[1] // block) * block
    if height < block or width < block:
        raise ValueError(f"image is smaller than one {block}px block")
    return cv2.resize(energy[:height, :width], (width // block, height // block),
                      interpolation=cv2.INTER_AREA)


def detail_agreement(fused, reference, block=64):
    """
    How far `fused` found detail where `reference` did. Returns (agreement, share).

    agreement - correlation between the two `detail_map`s, in [-1, 1]. 1 means
                the two agree everywhere about which parts of the frame resolve.
    share     - mean tile contrast of `fused` over that of `reference`, so 1.0
                is as much local contrast recovered, and less is a softer render.

    Both are relative to `reference` after `match_tone`, and both survive the
    misalignment between two programs' renders that stops PSNR working at all.
    Reported together because either alone is easy to satisfy: an image of pure
    noise agrees with nothing but scores a huge share, and a heavily blurred
    copy of the reference keeps some agreement while recovering nothing.
    """
    graded = match_tone(reference, fused)
    ours = detail_map(fused, block)
    theirs = detail_map(graded, block)
    agreement = float(np.corrcoef(ours.ravel(), theirs.ravel())[0, 1])
    return agreement, float(ours.mean() / max(theirs.mean(), 1e-6))


def fit_reference(reference, fused, min_inliers=40):
    """Warp another program's render onto ours by a similarity fit, or None.

    Needed because the two are not the same size and cropping does not relate
    them: each program aligned the stack against its own choice of reference
    frame, and focus breathing means the magnification each removed differs. On
    the ant capture the two renders sit 9% apart in scale, so a crop leaves the
    corners over a hundred pixels out and a zone taken off one lands nowhere
    near the same content in the other.

    A similarity - scale, rotation, translation, no perspective - is deliberately
    the most that is fitted. The residual is real: focus breathing is not one
    global magnification but a different one per frame, so no single warp can
    remove it, and after this fit the fine detail still correlates at only about
    0.35 with several pixels of local drift. That is why the metrics built on
    the result stay pooled - `band_ratios` over zones and `detail_agreement` over
    tiles - rather than becoming a per-pixel comparison. What the fit buys is
    that those pools now cover the same content: tile agreement on the ant
    capture rises from about 0.49 unfitted to about 0.95 fitted, which is the
    difference between a number that ranks candidates and one that ranks how
    badly each is misaligned.

    Returns the warped reference on the render's grid, or None when too few
    features match to trust the fit.
    """
    detector = getattr(cv2, "SIFT_create", None)
    if detector is None:                       # very old OpenCV build
        return None
    sift = detector(8000)

    grey_ref = _grey8(reference).astype(np.uint8)
    grey_ours = _grey8(fused).astype(np.uint8)
    key_ref, desc_ref = sift.detectAndCompute(grey_ref, None)
    key_ours, desc_ours = sift.detectAndCompute(grey_ours, None)
    if desc_ref is None or desc_ours is None or len(key_ref) < 2 or len(key_ours) < 2:
        return None

    matches = cv2.BFMatcher().knnMatch(desc_ref, desc_ours, k=2)
    # Lowe's ratio test: a descriptor that matches two places about equally well
    # has matched neither, and a focus stack is full of repeated texture.
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None

    src = np.float32([key_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([key_ours[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                  ransacReprojThreshold=3.0)
    if matrix is None or inliers is None or int(inliers.sum()) < min_inliers:
        return None
    return cv2.warpAffine(reference, matrix, (fused.shape[1], fused.shape[0]),
                          flags=cv2.INTER_LANCZOS4)


def reference_bands(reference, sigma=1.5, zones=5):
    """Split another program's render into activity zones, once for a whole sweep.

    The companion to `band_ratios` below, kept separate because it walks the
    reference rather than the candidate: a sweep computes it once and hands the
    same zoning to every render, which is also what makes the candidates
    comparable - they are scored over the same pixels rather than each over its
    own idea of where the detail is.

    Returns (band, masks): the reference's own high-pass, and the pixel sets its
    local activity splits into, quietest first.
    """
    grey = _grey8(reference)
    band = grey - cv2.GaussianBlur(grey, (0, 0), sigma)
    activity = cv2.blur(np.abs(band), (33, 33))
    edges = np.percentile(activity, np.linspace(0, 100, zones + 1))
    masks = [(activity >= edges[i]) & (activity <= edges[i + 1])
             for i in range(zones)]
    return band, masks


def band_ratios(fused, reference, bands, sigma=1.5):
    """Fine-detail energy against another program's, by how busy that region is.

    The reading that separates "sharper" from "grainier" as far as anything can
    when the two images are not registered to each other. Energy in a band is
    shift-tolerant where a correlation is not - see `detail_map` - so this is
    measured as a ratio of standard deviations inside each zone rather than as
    any kind of per-pixel agreement.

    Splitting by the reference's activity is what makes it say something. One
    number over the whole frame cannot tell a render that kept more texture from
    one that kept more grain, because both read high; a render that is grainier
    stands above the reference *hardest where the reference found nothing*,
    which is the first zone, while one that is genuinely sharper stands above it
    in the last. 1.0 in every zone is the reference exactly.

    `fused` is tone-matched onto the reference first, so a different exposure or
    contrast curve - which is not a fusion property - cannot move the result.
    """
    band_ref, masks = bands
    graded = match_tone(to_display8(fused), to_display8(reference))
    grey = _grey8(graded)
    band = grey - cv2.GaussianBlur(grey, (0, 0), sigma)

    height = min(band.shape[0], band_ref.shape[0])
    width = min(band.shape[1], band_ref.shape[1])
    band, band_ref = band[:height, :width], band_ref[:height, :width]
    return [float(band[m[:height, :width]].std()
                  / max(band_ref[m[:height, :width]].std(), 1e-6))
            for m in masks]


def reference_gap(ratios):
    """One number for a whole `band_ratios` curve: distance from the reference.

    Root-mean-square of the log ratios, so being twice as grainy and half as
    sharp cost the same - which they should, since neither is the render the
    reference made. 0 is the reference exactly.

    A gap and nothing else would be satisfied by any render that is uniformly
    wrong in both directions at once, so it is reported next to
    `detail_agreement` rather than instead of it.
    """
    if not ratios:
        return 0.0
    logs = np.log(np.maximum(np.asarray(ratios, dtype=np.float64), 1e-6))
    return float(np.sqrt(np.mean(logs ** 2)))


def to_display8(img):
    """8-bit view of a render, whatever depth it carries.

    A local copy of utils.bitdepth.to_display8's behaviour for the two dtypes
    these metrics see, so tests/fusion_metrics.py keeps depending on numpy and
    OpenCV alone.
    """
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 257.0).round().clip(0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


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


def block_speckle(fused, block=8, factor=4.0, quiet_percentile=40.0):
    """
    Percentage of flat-area blocks standing far outside their neighbours.

    The seam metrics above average over the lattice, which is the wrong shape
    for the other artefact a block method produces: a *lone* block taken from
    the wrong frame - a speck of dust that is sharp in exactly one frame winning
    its block, say. One square contributes almost nothing to a mean and is
    immediately obvious to a viewer. Measured only where there is no detail, so
    genuine fine texture is not counted as speckle.

    This was added after a mean-based seam score ranked an obviously speckled
    render as the best of a sweep.
    """
    grey = _gray32(fused)
    if fused.dtype == np.uint16:
        grey = grey / 257.0   # score 16-bit stacks on the same 8-bit scale
    h = (grey.shape[0] // block) * block
    w = (grey.shape[1] // block) * block
    if h < block * 5 or w < block * 5:
        return 0.0

    small = cv2.resize(grey[:h, :w], (w // block, h // block),
                       interpolation=cv2.INTER_AREA)
    detail = cv2.blur(np.abs(small - cv2.blur(small, (5, 5))), (17, 17))
    quiet = detail <= np.percentile(detail, quiet_percentile)
    if not quiet.any():
        return 0.0

    deviation = np.abs(small - cv2.medianBlur(small, 5))
    spread = cv2.blur(deviation, (9, 9)) + 0.5     # half a level, so flat
                                                   # areas do not divide by ~0
    return float((deviation > factor * spread)[quiet].sum()
                 / quiet.sum() * 100.0)


def _grey8(img):
    """Grayscale float32 on the 0-255 scale, whatever depth the image carries.

    The metrics below are defined in levels rather than in ratios, so a 16-bit
    render has to be brought onto the same scale as an 8-bit one before it is
    measured - otherwise the same picture scores 257 times worse for being
    stored more precisely.
    """
    grey = _gray32(img)
    return grey / 257.0 if img.dtype == np.uint16 else grey


def focus_energy_map(img, window=9):
    """Local Laplacian energy - the focus measure a depth-map select runs on.

    Reproduced here rather than imported from fusion_methods so that a metric
    keeps meaning the same thing when a method is retuned: a score that moves
    because the thing measuring it moved cannot rank two releases.
    """
    grey = _grey8(img)
    lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    return cv2.boxFilter(lap * lap, cv2.CV_32F, (window, window))


def stack_focus_ceiling(sources, window=9):
    """Per-pixel best focus energy the stack has to offer.

    The ceiling every fused image is judged against by `focus_retention`, and
    the expensive half of it: it walks the whole stack, so a sweep computes it
    once and hands the same array to every candidate. Frames are consumed one
    at a time, so this costs one energy map of memory, not one per frame.
    """
    ceiling = None
    for src in sources:
        energy = focus_energy_map(src, window)
        ceiling = energy if ceiling is None else np.maximum(ceiling, energy, out=ceiling)
    return ceiling


def focus_retention(fused, ceiling, window=9, percentile=85.0):
    """Share of the stack's available local contrast the fusion kept, ~[0, 1].

    This is the metric for haze. Q_ABF asks whether the fused image has edges
    where the sources had edges, and a veiled result still does - washed out,
    but there. This asks the sharper question: in the places where some frame
    resolved detail, how much of that frame's contrast survived? A hard select
    approaches 1; a blend that averages a defocused majority into the answer
    reads well below it, which is exactly what "the output looks hazy" is.

    Scored only over the `percentile` of pixels where the stack has the most to
    offer, since the ratio means nothing where no frame found anything. Energy
    is squared contrast, so the square root is taken and the number reads as a
    contrast ratio: 0.8 is "kept four fifths of the contrast that was there".

    `ceiling` comes from `stack_focus_ceiling` over the same source stack. A
    subsampled stack lowers the ceiling and so flatters every candidate equally
    - fine for ranking a sweep, not comparable across different subsamples.
    """
    energy = focus_energy_map(fused, window)
    height = min(energy.shape[0], ceiling.shape[0])
    width = min(energy.shape[1], ceiling.shape[1])
    energy, ceiling = energy[:height, :width], ceiling[:height, :width]

    strong = ceiling >= np.percentile(ceiling, percentile)
    if not strong.any():
        return 0.0
    eps = np.finfo(np.float32).eps
    ratio = energy[strong] / np.maximum(ceiling[strong], eps)
    # The median rather than the mean: a handful of pixels where the fusion
    # sharpened past any source - a seam, a haloed edge - would otherwise pull
    # the average up and report a cleaner result than the one on screen.
    return float(np.median(np.sqrt(ratio)))


def flat_noise(fused, sigma=2.0, quiet_percentile=30.0):
    """Grain left in the parts of the picture that hold no detail, in levels.

    The other half of "clean". A stack is many exposures of one scene, so a
    method that blends where there is nothing to choose between recovers the
    noise reduction that averaging gives and scores low here; a hard per-pixel
    select takes one frame's grain whole and scores high. Measured on the
    high-pass residual so a smooth tonal gradient is not counted as noise, and
    restricted to the quietest `quiet_percentile` of the frame so real texture
    is not either.

    Lower is cleaner. It trades against `focus_retention` - the settings that
    keep the most contrast are usually the ones that keep the most grain - and
    the pair is the trade-off a sweep is looking for.
    """
    grey = _grey8(fused)
    residual = grey - cv2.GaussianBlur(grey, (0, 0), sigma)
    detail = cv2.blur(np.abs(residual), (33, 33))
    quiet = detail <= np.percentile(detail, quiet_percentile)
    if not quiet.any():
        return 0.0
    return float(np.std(residual[quiet]))


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
        "block_speckle": block_speckle(fused, block),
    }
    if reference is not None:
        scores["psnr"] = psnr(fused, reference)
        scores["ssim"] = ssim(fused, reference)
    return scores
