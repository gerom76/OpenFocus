"""
Generate the sample focus stacks that live next to this script.

The stacks in tests/ exist to answer "does fusion still work"; they are built by
blurring everything outside a mask, which is cheap and repeatable but is not how
a camera defocuses a scene. These samples are built the other way round: from a
depth map. Every scene is a set of layers at known distances, each frame picks a
focus distance, and each layer is convolved with the disc its circle of
confusion would have at that distance before being composited front to back.

That difference is the point of the folder. Because layers are blurred
separately and then composited, a defocused foreground spreads *over* what is
behind it and takes some of the background's colour with it - the partial
occlusion that produces halos in a real stack, and that mask-based fixtures
cannot produce at all. Tuning against it is therefore worth something.

Every scene ships its ground truth: the all-in-focus render, the depth map, and
the index of the frame that is sharpest at each pixel. Those give both a target
to score a fusion against and a label to train a selection map on.

Regenerate everything with:

    python samples/generate_samples.py

Output is deterministic - same seeds, same bytes - so regenerating never shows
up as a diff unless the generator itself changed.
"""

import json
import os
import sys

import cv2
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Scene distances are in arbitrary "metres"; only ratios matter. Blur radius in
# pixels is COC_SCALE * |1/z - 1/focus|, which is the thin-lens circle of
# confusion up to a constant, so this one number sets how aggressive the
# defocus is across every scene.
COC_SCALE = 26.0
MAX_RADIUS = 22.0       # clamp, so a far background stays cheap to convolve
SHARP_RADIUS = 0.35     # below this a layer is copied rather than convolved


# ---------------------------------------------------------------------------
# Defocus rendering
# ---------------------------------------------------------------------------

_KERNEL_CACHE = {}


def _disc_kernel(radius):
    """
    An anti-aliased disc, which is what an iris actually convolves with.

    A Gaussian is the usual shortcut and it is the wrong shape: a disc leaves
    the doubled edges and the flat-topped bokeh that make out-of-focus detail
    hard to tell from focused detail, and those are exactly the cases a focus
    measure gets wrong.
    """
    key = round(radius * 4) / 4.0
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    size = int(np.ceil(key)) * 2 + 1
    coords = np.arange(size, dtype=np.float32) - size // 2
    dist = np.hypot(coords[:, None], coords[None, :])
    # One pixel of soft edge, otherwise the disc itself aliases into rings
    kernel = np.clip(key + 0.5 - dist, 0.0, 1.0).astype(np.float32)
    kernel /= kernel.sum()
    _KERNEL_CACHE[key] = kernel
    return kernel


def _defocus(img, radius, max_radius=MAX_RADIUS):
    if radius < SHARP_RADIUS:
        return img
    return cv2.filter2D(img, -1, _disc_kernel(min(radius, max_radius)),
                        borderType=cv2.BORDER_REFLECT)


def coc_radius(depth, focus, coc_scale=COC_SCALE):
    """
    Blur radius in pixels for a layer at `depth` when focused at `focus`.

    `coc_scale` is in pixels, so it has to grow with the render: the same scene
    at twice the width has to blur twice as far to look like the same aperture.
    """
    return coc_scale * abs(1.0 / depth - 1.0 / focus)


class Layer:
    """One depth plane: premultiplied colour, coverage, and a distance."""

    def __init__(self, depth, rgb, alpha):
        self.depth = float(depth)
        self.rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        self.alpha = np.ascontiguousarray(alpha, dtype=np.float32)


def composite(layers, focus=None, coc_scale=COC_SCALE, max_radius=MAX_RADIUS):
    """
    Composite far to near, defocusing each layer on the way.

    `focus=None` renders everything sharp, which is the all-in-focus ground
    truth. Colour is premultiplied by coverage before blurring so that a
    defocused edge fades out instead of dragging black in behind it.
    """
    order = sorted(layers, key=lambda l: -l.depth)
    out = np.zeros(order[0].rgb.shape, dtype=np.float32)
    for layer in order:
        premul = layer.rgb * layer.alpha[:, :, None]
        alpha = layer.alpha
        if focus is not None:
            radius = coc_radius(layer.depth, focus, coc_scale)
            premul = _defocus(premul, radius, max_radius)
            alpha = _defocus(alpha, radius, max_radius)
        out = premul + out * (1.0 - alpha[:, :, None])
    return out


def depth_of_scene(layers):
    """
    Distance to the front-most surface at each pixel.

    Composited the same way as colour, so a half-covered pixel gets a blend of
    the two depths rather than an arbitrary pick.
    """
    order = sorted(layers, key=lambda l: -l.depth)
    depth = np.full(order[0].alpha.shape, order[0].depth, dtype=np.float32)
    for layer in order:
        depth = layer.depth * layer.alpha + depth * (1.0 - layer.alpha)
    return depth


def focus_indices(depth, focus_distances):
    """Which frame is sharpest at each pixel - the label a depth map predicts."""
    disparity = 1.0 / depth
    scores = np.stack([np.abs(disparity - 1.0 / f) for f in focus_distances])
    return np.argmin(scores, axis=0).astype(np.uint8)


def focus_schedule(near, far, count):
    """
    Focus distances spaced evenly in 1/distance.

    Even spacing in distance would crowd the far end of the stack and leave a
    gap at the near end, because depth of field grows with distance. Stepping in
    reciprocal distance gives every frame a comparable share of the scene, which
    is what a focus rail set up properly does.
    """
    return (1.0 / np.linspace(1.0 / near, 1.0 / far, count)).astype(np.float32)


# ---------------------------------------------------------------------------
# Procedural texture
# ---------------------------------------------------------------------------

def fbm(shape, seed, octaves=5, persistence=0.55, base=4):
    """Fractal value noise - the backbone of every surface texture here."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float32)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        res = base * (2 ** octave)
        grid = rng.random((res, res)).astype(np.float32)
        layer = cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        out += amplitude * layer
        total += amplitude
        amplitude *= persistence
    return out / total


def tint(mono, low, high):
    """Map a 0..1 field onto a colour ramp; returns HxWx3 in 0..255."""
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    return low + (high - low) * mono[:, :, None]


def bands_from_depthmap(rgb, alpha, depthmap, count=26):
    """
    Split a continuously curved surface into discrete depth planes.

    The renderer composites planes, but real subjects are curved, so a surface
    has to be quantised into `count` of them. Assigning each pixel to its
    nearest plane does not work: neighbouring planes get visibly different blur
    discs, and the boundary between them prints as a hard contour line across a
    surface that is perfectly smooth in the scene.

    So a pixel is shared between the two planes that straddle it, by a linear
    weight, and its blur ends up interpolated between theirs. The weights are
    then rewritten as `over` alphas - each plane's alpha is its share of what
    the planes in front of it have left - so that the shares composite to
    exactly the coverage the surface had.
    """
    valid = alpha > 1e-3
    if not valid.any():
        return []

    lo, hi = float(depthmap[valid].min()), float(depthmap[valid].max())
    if hi - lo < 1e-4:
        return [Layer(lo, rgb, alpha)]

    # Spaced evenly in 1/distance, and weighted there too. Blur radius goes as
    # |1/z - 1/focus|, so planes spaced evenly in distance are *not* evenly
    # spaced in blur: the near end of a wide depth range gets radius steps
    # several times larger than the far end. Sharing a pixel between two planes
    # blends two blur discs where one intermediate disc belongs, and that
    # approximation degrades with the gap between them - which is why a smooth
    # receding background used to print faint contours at the plane pitch even
    # though the depth ramp behind it was perfectly smooth.
    disparities = np.linspace(1.0 / lo, 1.0 / hi, count)
    step = abs(disparities[1] - disparities[0])
    depth_disparity = 1.0 / np.maximum(depthmap, 1e-6)
    shares = [(np.clip(1.0 - np.abs(depth_disparity - disparity) / step, 0.0, 1.0) * alpha,
               1.0 / disparity)
              for disparity in disparities]

    layers = []
    remaining = np.ones_like(alpha)     # light not yet blocked by a nearer plane
    for share, plane in sorted(shares, key=lambda item: item[1]):
        plane_alpha = np.clip(share / np.maximum(remaining, 1e-4), 0.0, 1.0)
        remaining = np.clip(remaining - share, 0.0, 1.0)
        if plane_alpha.max() < 1e-3:
            continue
        layers.append(Layer(plane, rgb, plane_alpha))
    return layers


def draw_text(canvas, text, org, scale, colour, thickness=1):
    """
    Blend text into a float canvas.

    cv2.putText refuses anything but 8-bit input, and every surface here is
    float, so the glyphs are rasterised into a mask and composited by hand.
    """
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.putText(mask, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255,
                thickness, cv2.LINE_AA)
    coverage = (mask.astype(np.float32) / 255.0)[:, :, None]
    canvas *= 1.0 - coverage
    canvas += coverage * np.asarray(colour, dtype=np.float32)


def _ellipse_alpha(shape, centre, axes, softness=1.5):
    """A filled ellipse with a soft edge, as a float coverage mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    alpha = mask.astype(np.float32) / 255.0
    if softness:
        alpha = cv2.GaussianBlur(alpha, (0, 0), softness)
    return alpha


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def scene_beetle(shape=(480, 640)):
    """
    The everyday macro case: a domed subject on a receding substrate.

    Depth is continuous over the subject and over the ground, so no single
    frame is sharp everywhere and the correct answer varies smoothly across the
    picture. This is the scene to tune against before any of the harder ones.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # Substrate: recedes from the bottom of the frame towards the top
    ground_depth = 1.05 + 1.55 * (1.0 - ys / h) ** 1.4
    grain = fbm(shape, seed=11, octaves=6, base=6)
    ground = tint(grain, (48, 62, 86), (126, 152, 186))
    ground += (fbm(shape, seed=12, octaves=3, base=48)[:, :, None] - 0.5) * 40.0

    # Subject: an ellipsoid, so the shell is nearest at the centre
    cx, cy = int(w * 0.46), int(h * 0.56)
    ax, ay = int(w * 0.27), int(h * 0.33)
    r2 = ((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2
    dome = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    body_alpha = _ellipse_alpha(shape, (cx, cy), (ax, ay), softness=1.2)
    body_depth = 0.92 - 0.30 * dome

    shell = fbm(shape, seed=21, octaves=4, base=8)
    body = tint(shell, (18, 24, 30), (58, 96, 122))

    # Punctate rows and setae: the fine detail that only survives in the frames
    # focused on that part of the shell
    rows = np.sin((xs - cx) * 0.55) * np.sin((ys - cy) * 0.16)
    body += (rows > 0.72)[:, :, None] * np.float32([22, 34, 44])
    for k in range(-9, 10):
        x0 = cx + int(k * ax / 9.5)
        cv2.line(body, (x0, cy - ay), (x0 + 6, cy + ay), (86, 128, 150), 1, cv2.LINE_AA)

    # Elytral seam and a specular streak, both strong sharpness cues
    cv2.line(body, (cx, cy - ay), (cx, cy + ay), (8, 10, 14), 2, cv2.LINE_AA)
    highlight = np.exp(-(((xs - cx + ax * 0.35) / (ax * 0.30)) ** 2
                         + ((ys - cy + ay * 0.45) / (ay * 0.22)) ** 2))
    body += highlight[:, :, None] * np.float32([120, 130, 138])
    body *= (0.45 + 0.55 * dome)[:, :, None]

    layers = bands_from_depthmap(ground, np.ones(shape, np.float32), ground_depth, 22)
    layers += bands_from_depthmap(body, body_alpha, body_depth, 26)
    return layers, focus_schedule(0.62, 2.45, 16)


def scene_tilted_print(shape=(480, 640)):
    """
    A flat card raked steeply away from the lens - a coin or a printed page.

    Depth is a plain linear ramp, so the sharpest frame changes monotonically
    down the picture and any error in the selection shows up as a band. Detail
    is fine engraving and small type, which defocus removes before it removes
    anything else.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    paper = fbm(shape, seed=31, octaves=5, base=10)
    card = tint(paper, (168, 176, 186), (226, 232, 240))

    centre = (w // 2, h // 2)
    for radius in range(40, min(h, w) // 2, 9):
        cv2.circle(card, centre, radius, (96, 104, 118), 1, cv2.LINE_AA)
    for degrees in range(0, 360, 3):
        angle = np.radians(degrees)
        inner = min(h, w) * 0.30
        outer = min(h, w) * 0.46
        cv2.line(card,
                 (int(centre[0] + inner * np.cos(angle)), int(centre[1] + inner * np.sin(angle))),
                 (int(centre[0] + outer * np.cos(angle)), int(centre[1] + outer * np.sin(angle))),
                 (58, 64, 78), 1, cv2.LINE_AA)

    # Type at several sizes: the smallest is the real test
    for i, (text, scale, thick) in enumerate([("OPENFOCUS", 1.1, 2),
                                              ("focus stacking sample", 0.5, 1),
                                              ("1234567890 abcdefghij", 0.34, 1)]):
        draw_text(card, text, (int(w * 0.14), int(h * (0.34 + 0.13 * i))),
                  scale, (24, 28, 38), thick)

    # 1 px hatching, right at the resolution limit
    card[::4, :] *= 0.86
    card[:, ::7] *= 0.92

    depth = 0.68 + 1.05 * (ys / h)
    return bands_from_depthmap(card, np.ones(shape, np.float32), depth, 30), \
        focus_schedule(0.68, 1.78, 14)


def scene_circuit(shape=(480, 640)):
    """
    Stepped hardware: four flat planes with hard silhouettes between them.

    Nothing is curved, so the depth map is piecewise constant and every error
    lands on an occlusion boundary. This is where halos, dark rims and colour
    bleed are visible, and it is the scene to look at when an edge in a fused
    result looks wrong.
    """
    h, w = shape
    rng = np.random.default_rng(41)
    opaque = np.ones(shape, np.float32)

    # Board
    board = tint(fbm(shape, seed=42, octaves=4, base=8), (34, 78, 40), (52, 116, 58))
    for _ in range(70):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        for _ in range(int(rng.integers(2, 5))):
            x1 = x0 + int(rng.choice([-1, 1])) * int(rng.integers(20, 90))
            cv2.line(board, (x0, y0), (x1, y0), (152, 186, 168), 2, cv2.LINE_AA)
            y1 = y0 + int(rng.choice([-1, 1])) * int(rng.integers(20, 90))
            cv2.line(board, (x1, y0), (x1, y1), (152, 186, 168), 2, cv2.LINE_AA)
            x0, y0 = x1, y1
    for _ in range(120):
        cv2.circle(board, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   3, (168, 200, 214), -1, cv2.LINE_AA)

    layers = [Layer(1.55, board, opaque)]

    # Chips at two heights, plus a connector nearest the lens
    plates = [(1.24, [(70, 60, 210, 150), (330, 90, 470, 190)], (26, 28, 32), (196, 204, 212)),
              (1.02, [(140, 250, 330, 360), (380, 260, 540, 340)], (18, 20, 24), (206, 214, 224))]
    for depth, boxes, body_colour, pin_colour in plates:
        rgb = np.zeros((h, w, 3), np.float32)
        alpha = np.zeros(shape, np.float32)
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(rgb, (x0, y0), (x1, y1), body_colour, -1)
            cv2.rectangle(alpha, (x0, y0), (x1, y1), 1.0, -1)
            draw_text(rgb, "IC-%d" % int(depth * 100), (x0 + 12, y0 + 34),
                      0.55, (188, 194, 202))
            for x in range(x0 + 8, x1 - 6, 10):     # leads
                cv2.line(rgb, (x, y1 - 5), (x, y1 + 4), pin_colour, 2)
                cv2.line(alpha, (x, y1 - 5), (x, y1 + 4), 1.0, 2)
        layers.append(Layer(depth, rgb, alpha))

    connector = np.zeros((h, w, 3), np.float32)
    conn_alpha = np.zeros(shape, np.float32)
    cv2.rectangle(connector, (30, 380), (600, 450), (30, 34, 44), -1)
    cv2.rectangle(conn_alpha, (30, 380), (600, 450), 1.0, -1)
    for x in range(44, 596, 14):
        cv2.rectangle(connector, (x, 392), (x + 7, 438), (198, 176, 96), -1)
    layers.append(Layer(0.82, connector, conn_alpha))

    return layers, focus_schedule(0.78, 1.70, 12)


def scene_thicket(shape=(480, 640)):
    """
    Overlapping thin strands spread through depth - moss, hair, fibres.

    The hard case. A strand is narrower than the blur disc that hits it, so when
    it is out of focus it goes translucent and whatever sits behind it shows
    through. A method that decides one frame owns a pixel cannot represent
    that, and this scene is where that shows up as ghosting and as strands that
    appear in the fused result at two depths at once.
    """
    h, w = shape
    rng = np.random.default_rng(51)

    backdrop = tint(fbm(shape, seed=52, octaves=5, base=5), (26, 30, 24), (86, 104, 72))
    layers = [Layer(2.6, backdrop, np.ones(shape, np.float32))]

    bands = 20
    depths = np.linspace(0.62, 2.2, bands)
    for band, depth in enumerate(depths):
        rgb = np.zeros((h, w, 3), np.float32)
        alpha = np.zeros(shape, np.float32)
        shade = 0.4 + 0.6 * (1.0 - band / bands)        # nearer strands catch more light
        for _ in range(7):
            x = int(rng.integers(-40, w + 40))
            y = int(rng.integers(-40, h + 40))
            colour = tuple(float(v) * shade for v in rng.integers(90, 235, 3))
            points = [(x, y)]
            for step in range(int(rng.integers(6, 14))):
                x += int(rng.integers(-26, 27))
                y += int(rng.integers(6, 34))
                points.append((x, y))
            curve = np.array(points, np.int32).reshape(-1, 1, 2)
            width = int(rng.integers(1, 4))
            cv2.polylines(rgb, [curve], False, colour, width, cv2.LINE_AA)
            cv2.polylines(alpha, [curve], False, 1.0, width, cv2.LINE_AA)
        if alpha.max() > 1e-3:
            layers.append(Layer(float(depth), rgb, alpha))

    return layers, focus_schedule(0.62, 2.40, 18)


def scene_pale_specimen(shape=(480, 640)):
    """
    A pale, softly lit subject shot at high ISO, with vignetting.

    Contrast is low enough that grain is a large fraction of the local
    variation, so any focus measure reading variance can prefer a noisy
    defocused frame to a clean focused one. Tune denoising and focus-measure
    windows here.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    base = tint(fbm(shape, seed=61, octaves=4, base=6), (196, 198, 200), (214, 217, 221))
    ground_depth = np.full(shape, 1.9, np.float32)
    layers = bands_from_depthmap(base, np.ones(shape, np.float32), ground_depth, 1)

    cx, cy = int(w * 0.52), int(h * 0.50)
    ax, ay = int(w * 0.31), int(h * 0.34)
    r2 = ((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2
    dome = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    alpha = _ellipse_alpha(shape, (cx, cy), (ax, ay), softness=2.0)

    specimen = tint(fbm(shape, seed=62, octaves=5, base=9), (198, 202, 206), (228, 231, 234))
    # Broad markings, only a few levels away from the surface they sit on
    rng = np.random.default_rng(63)
    for i in range(18):
        c = (int(rng.integers(cx - ax, cx + ax)), int(rng.integers(cy - ay, cy + ay)))
        delta = 9.0 if i % 2 else -9.0
        cv2.circle(specimen, c, int(rng.integers(10, 30)),
                   (float(210 + delta), float(213 + delta), float(217 + delta)), -1, cv2.LINE_AA)

    # Fine striations and pitting. Without these the subject has no detail at
    # any scale, and then no frame is sharper than any other - the scene would
    # be merely flat rather than hard. They sit 6 to 12 levels off the
    # background, which is the point: real detail, but barely above the grain.
    for k in range(-14, 15):
        x0 = cx + int(k * ax / 15.0)
        cv2.line(specimen, (x0, cy - ay), (x0 + 10, cy + ay),
                 (204.0, 207.0, 211.0), 1, cv2.LINE_AA)
    for _ in range(260):
        c = (int(rng.integers(cx - ax, cx + ax)), int(rng.integers(cy - ay, cy + ay)))
        shade = float(rng.integers(200, 224))
        cv2.circle(specimen, c, int(rng.integers(1, 3)),
                   (shade, shade + 2, shade + 5), -1, cv2.LINE_AA)

    specimen *= (0.90 + 0.10 * dome)[:, :, None]

    layers += bands_from_depthmap(specimen, alpha, 1.28 - 0.34 * dome, 24)
    return layers, focus_schedule(0.90, 2.05, 12)


# ---------------------------------------------------------------------------
# Capture simulation
# ---------------------------------------------------------------------------

def breathing_transform(shape, index, count, strength):
    """
    The frame-to-frame movement a real stack has and a synthetic one does not.

    Racking focus changes magnification (focus breathing), and on a handheld or
    lightly built rig the frame also drifts and rolls a little. Registration is
    the first stage of the pipeline and it needs something to correct.
    """
    h, w = shape
    t = index / max(count - 1, 1)
    scale = 1.0 + strength * 0.030 * t
    angle = strength * 0.45 * (t - 0.5)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += strength * (2.6 * t + 0.8 * np.sin(index * 1.7))
    matrix[1, 2] += strength * (1.9 * t + 0.8 * np.cos(index * 2.1))
    return matrix


def apply_capture(frame, rng, noise, vignette=0.0, exposure=1.0, transform=None):
    """Everything between the ideal render and the file a camera writes."""
    out = frame * exposure

    if vignette:
        h, w = out.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2))
        out *= (1.0 - vignette * np.clip(r, 0, 1.4) ** 2)[:, :, None]

    if transform is not None:
        out = cv2.warpAffine(out, transform, (out.shape[1], out.shape[0]),
                             flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    if noise:
        # Shot noise dominates in the highlights, read noise in the shadows
        shot = np.sqrt(np.clip(out, 0, 255) / 255.0) * noise
        out = out + rng.normal(0.0, 1.0, out.shape).astype(np.float32) * (shot + noise * 0.35)

    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# Writing a scene out
# ---------------------------------------------------------------------------

def _write(path, image, bits=8):
    if bits == 16:
        data = np.clip(image / 255.0 * 65535.0, 0, 65535).astype(np.uint16)
    else:
        data = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(path, data, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def _contact_sheet(frames, columns=6, thumb=160):
    """A single strip of the whole stack, for eyeballing a scene at a glance."""
    scaled = [cv2.resize(f, (thumb, int(thumb * f.shape[0] / f.shape[1])))
              for f in frames]
    rows = int(np.ceil(len(scaled) / columns))
    th, tw = scaled[0].shape[:2]
    sheet = np.full((rows * th, columns * tw, 3), 24, np.uint8)
    for i, img in enumerate(scaled):
        r, c = divmod(i, columns)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = np.clip(img, 0, 255).astype(np.uint8)
    return sheet


def label_agreement(frames, index_map):
    """
    How often the sharpest-looking frame is the one the label names.

    The focus index is derived from the depth map, so it is exact by
    construction - but only where "the depth at this pixel" is a meaningful
    thing to say. On a translucent strand, or on a flat patch with nothing to
    resolve, one integer per pixel cannot describe what the stack contains, and
    a label that cannot be recovered from the pixels is a label nothing should
    be trained against. Measuring the disagreement with a plain focus measure
    is a cheap way to publish that caveat as a number instead of a warning.
    """
    energy = []
    for frame in frames:
        grey = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
        energy.append(cv2.GaussianBlur(laplacian * laplacian, (0, 0), 6.0))
    measured = np.argmax(np.stack(energy), axis=0).astype(np.int16)
    error = np.abs(measured - index_map.astype(np.int16))
    return round(float((error <= 1).mean()), 4)


def build_scene(name, builder, blurb, seed, noise=1.6, vignette=0.0,
                bits=8, drift=0.0, exposure_drift=0.0,
                coc_scale=COC_SCALE, max_radius=MAX_RADIUS, extra_meta=None):
    print("  %-18s" % name, end="", flush=True)
    layers, focus_distances = builder()
    h, w = layers[0].alpha.shape

    root = os.path.join(OUT_DIR, name)
    frames_dir = os.path.join(root, "frames")
    truth_dir = os.path.join(root, "ground_truth")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    depth = depth_of_scene(layers)
    sharp = composite(layers, focus=None)

    frames, transforms = [], []
    for index, focus in enumerate(focus_distances):
        rendered = composite(layers, focus=float(focus),
                             coc_scale=coc_scale, max_radius=max_radius)
        matrix = (breathing_transform((h, w), index, len(focus_distances), drift)
                  if drift else None)
        exposure = 1.0 + exposure_drift * np.sin(index * 0.9)
        frame = apply_capture(rendered, rng, noise, vignette, exposure, matrix)
        _write(os.path.join(frames_dir, "frame_%02d.png" % index), frame, bits)
        frames.append(frame)
        transforms.append(None if matrix is None else matrix.tolist())
        print(".", end="", flush=True)

    # Ground truth. The all-in-focus render carries the same noise floor and
    # vignetting as the frames but none of the movement, so a fused result can
    # be compared against it directly once it has been registered.
    _write(os.path.join(truth_dir, "all_in_focus.png"),
           apply_capture(sharp, np.random.default_rng(seed + 1), noise, vignette), bits)

    # 16-bit, because 8 bits cannot hold a useful depth range
    normalised = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    cv2.imwrite(os.path.join(truth_dir, "depth_map.png"),
                (normalised * 65535).astype(np.uint16))

    index_map = focus_indices(depth, focus_distances)
    cv2.imwrite(os.path.join(truth_dir, "focus_index.png"), index_map)
    cv2.imwrite(os.path.join(truth_dir, "focus_index_preview.png"),
                cv2.applyColorMap(
                    (index_map * (255 // max(len(focus_distances) - 1, 1))).astype(np.uint8),
                    cv2.COLORMAP_TURBO))

    cv2.imwrite(os.path.join(root, "contact_sheet.jpg"), _contact_sheet(frames),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    meta = {
        "name": name,
        "description": blurb,
        "resolution": [w, h],
        "bit_depth": bits,
        "frame_count": len(focus_distances),
        "focus_distances": [round(float(f), 5) for f in focus_distances],
        "depth_range": [round(float(depth.min()), 5), round(float(depth.max()), 5)],
        "coc_scale_px": coc_scale,
        "max_blur_radius_px": max_radius,
        "max_blur_radius_used_px": round(float(max(
            min(coc_radius(l.depth, float(f), coc_scale), max_radius)
            for l in layers for f in focus_distances)), 3),
        # Fraction of pixels where a Laplacian focus measure picks the labelled
        # frame or a neighbour. Low means the label is ambiguous there, not
        # wrong - see the note in README.md before training on it.
        "focus_index_agreement": label_agreement(frames, index_map),
        "noise_sigma": noise,
        "vignette": vignette,
        "exposure_drift": exposure_drift,
        "alignment_drift": drift,
        "per_frame_affine": transforms if drift else None,
        "seed": seed,
    }
    meta.update(extra_meta or {})
    with open(os.path.join(root, "scene.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(" ok")
    return meta


SCENES = [
    ("macro_dome", scene_beetle,
     "Domed subject on a receding substrate. Continuous depth everywhere, so no "
     "single frame is sharp anywhere but a ring. Start here.",
     dict(seed=101, noise=1.8)),

    ("tilted_print", scene_tilted_print,
     "A printed card raked away from the lens. Linear depth ramp and fine type, "
     "recorded at 16 bits; selection errors show as horizontal bands.",
     dict(seed=202, noise=1.1, bits=16)),

    ("circuit_steps", scene_circuit,
     "Four flat planes with hard silhouettes. Every error lands on an occlusion "
     "edge, which is where halos and colour bleed become visible.",
     dict(seed=303, noise=1.4)),

    ("fibre_thicket", scene_thicket,
     "Thin strands spread through depth. Defocused strands go translucent and "
     "let the background through - partial occlusion, the hard case.",
     dict(seed=404, noise=2.0)),

    ("pale_specimen", scene_pale_specimen,
     "Flat, pale subject at high ISO with vignetting. Grain rivals the real "
     "detail, so variance-based focus measures pick badly.",
     dict(seed=505, noise=6.0, vignette=0.13)),

    ("handheld_drift", scene_beetle,
     "The macro_dome scene recaptured with focus breathing, frame drift, roll "
     "and exposure flicker. Fails without registration; scene.json carries the "
     "affine that was applied to each frame.",
     dict(seed=606, noise=2.2, drift=1.0, exposure_drift=0.045)),
]


def main():
    print("Generating focus-stack samples in %s" % OUT_DIR)
    manifest = [build_scene(name, builder, blurb, **options)
                for name, builder, blurb, options in SCENES]

    # The photo-derived scenes are written by the same builder and land in the
    # same manifest, so there is one command to rebuild the folder and one file
    # describing it. Imported here rather than at the top because that module
    # imports this one back.
    from samples.photo_stacks import build_all
    manifest += build_all()

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"scenes": manifest}, handle, indent=2)
    total = sum(m["frame_count"] for m in manifest)
    print("Done: %d scenes, %d frames" % (len(manifest), total))


if __name__ == "__main__":
    # Run as a script, samples/ is on the path but the repository root is not,
    # and the photo scenes are imported as samples.photo_stacks
    sys.path.insert(0, os.path.dirname(OUT_DIR))
    main()
