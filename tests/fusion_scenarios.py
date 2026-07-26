"""
Scenarios that isolate one fusion property each.

The shared fixture in synthetic_stack.py answers "does fusion work at all". These
answer "where does each method struggle", by building stacks that stress a single
thing: fine texture, sensor noise, a hard depth boundary, a long stack, flat
low-contrast subjects, saturated colour.

Every scenario returns (stack, reference, masks). The reference is what a perfect
fusion would reproduce, so any method can be scored against it, and the masks say
which slice owns which region - needed to measure how far focus bleeds across a
depth boundary.
"""

import cv2
import numpy as np

BLUR = 15  # defocus strength shared by every scenario, so results stay comparable


def _stack_from_masks(reference, masks, blur_ksize=BLUR):
    """Blur everything outside each mask to synthesise one slice per focus region."""
    blurred = cv2.GaussianBlur(reference, (blur_ksize, blur_ksize), 0)
    stack = []
    for mask in masks:
        m = mask[:, :, None]
        img = reference.astype(np.float32) * m + blurred.astype(np.float32) * (1.0 - m)
        stack.append(np.clip(img, 0, 255).astype(np.uint8))
    return stack


def _band_masks(height, width, count):
    """Horizontal bands - the simple case, boundaries are straight and axis-aligned."""
    edges = np.linspace(0, height, count + 1).round().astype(int)
    masks = []
    for k in range(count):
        m = np.zeros((height, width), dtype=np.float32)
        m[edges[k]:edges[k + 1]] = 1.0
        masks.append(m)
    return masks


def _base_scene(height, width, seed):
    """Smooth gradient ground with solid shapes - the common backdrop."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    img = np.stack([xs.repeat(height, 0) * 130 + 70,
                    ys.repeat(width, 1) * 110 + 80,
                    (1.0 - xs).repeat(height, 0) * 120 + 60], axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)

    for i in range(10):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        colour = tuple(int(v) for v in rng.integers(40, 215, 3))
        if i % 2:
            cv2.circle(img, centre, int(rng.integers(12, 30)), colour, -1)
        else:
            cv2.rectangle(img, centre, (centre[0] + int(rng.integers(16, 40)),
                                        centre[1] + int(rng.integers(16, 40))), colour, -1)
    return img


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def fine_texture(size=320, slices=3, seed=5):
    """Hair-thin lines and small type-like marks: can the method keep fine detail?"""
    img = _base_scene(size, size, seed)

    # Dense 1 px hatching in both directions - the first thing defocus destroys
    for x in range(0, size, 6):
        cv2.line(img, (x, 0), (x, size), (250, 250, 250), 1)
    for y in range(0, size, 10):
        cv2.line(img, (0, y), (size, y), (25, 25, 25), 1)

    # Small filled squares standing in for text
    rng = np.random.default_rng(seed + 1)
    for _ in range(120):
        x, y = int(rng.integers(4, size - 6)), int(rng.integers(4, size - 6))
        cv2.rectangle(img, (x, y), (x + 2, y + 3), (15, 15, 15), -1)

    masks = _band_masks(size, size, slices)
    return _stack_from_masks(img, masks), img, masks


def sensor_noise(size=320, slices=3, seed=5, sigma=22.0):
    """A high-ISO frame: heavy grain that a focus measure can mistake for detail."""
    rng = np.random.default_rng(seed)
    img = _base_scene(size, size, seed).astype(np.float32)
    img += rng.normal(0.0, sigma, size=(size, size, 3)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    masks = _band_masks(size, size, slices)
    return _stack_from_masks(img, masks), img, masks


def depth_edge(size=320, seed=5):
    """
    A subject at one depth against a background at another, split by a circle.

    Halos and bleeding show up along that boundary, and a curved edge is harder
    than the straight band splits used elsewhere.
    """
    img = _base_scene(size, size, seed)
    cv2.circle(img, (size // 2, size // 2), size // 4, (240, 240, 245), -1)
    for r in range(10, size // 4, 8):
        cv2.circle(img, (size // 2, size // 2), r, (40, 60, 160), 1)

    near = np.zeros((size, size), dtype=np.float32)
    cv2.circle(near, (size // 2, size // 2), size // 4, 1.0, -1)
    masks = [near, 1.0 - near]
    return _stack_from_masks(img, masks), img, masks


def long_stack(size=320, slices=12, seed=5):
    """Twelve slices instead of three: does quality or speed fall apart?"""
    img = _base_scene(size, size, seed)
    for x in range(0, size, 8):
        cv2.line(img, (x, 0), (x + 30, size), (235, 235, 235), 1)
    masks = _band_masks(size, size, slices)
    return _stack_from_masks(img, masks), img, masks


def deep_stack(size=320, slices=64, seed=5):
    """
    A macro stack's worth of frames over a subject that does not fill the frame.

    The other scenarios are 3 to 12 frames, which is not the regime the methods
    are used in and not the regime they fail in. With 64 frames no two
    neighbours differ by much, so any rule of the form "take the frame that
    clearly wins this block" stops discriminating and whatever breaks the tie
    draws the picture. The background here is never in focus in any frame and
    its appearance drifts steadily across the stack, so a tie broken badly shows
    up as steps between frames that look nothing alike - which is what a
    block-selection method tears the defocused background into.
    """
    rng = np.random.default_rng(seed)
    subject = _base_scene(size, size, seed)
    for _ in range(90):                     # fine detail worth resolving
        x, y = int(rng.integers(4, size - 8)), int(rng.integers(4, size - 8))
        cv2.rectangle(subject, (x, y), (x + 3, y + 2), (245, 245, 245), -1)

    # Subject occupies the middle band; everything else is background.
    near = np.zeros((size, size), np.float32)
    near[size // 3:2 * size // 3] = 1.0
    near = cv2.GaussianBlur(near, (0, 0), 3.0)

    # Background: broad soft blobs, bright enough that switching between two
    # renderings of it is plainly visible.
    background = np.full((size, size, 3), 40, np.float32)
    for _ in range(7):
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.circle(background, c, int(rng.integers(30, 70)),
                   tuple(float(v) for v in rng.integers(120, 235, 3)), -1)

    focus = np.linspace(0.0, 1.0, slices)
    subject_plane = 0.35
    stack = []
    for k, f in enumerate(focus):
        s_sigma = abs(f - subject_plane) * slices * 0.55
        b_sigma = 4.0 + f * 26.0            # never sharp, always drifting
        s = subject if s_sigma < 0.3 else cv2.GaussianBlur(subject, (0, 0), s_sigma)
        # Focus breathing: the background also drifts sideways through the
        # stack, so two frames far apart do not merely differ in blur - they
        # disagree about where things are. Splicing them shows.
        shift = np.float32([[1, 0, k * 0.35], [0, 1, k * 0.12]])
        b = cv2.warpAffine(cv2.GaussianBlur(background, (0, 0), b_sigma), shift,
                           (size, size), borderMode=cv2.BORDER_REFLECT)
        a = near[:, :, None]
        img = s.astype(np.float32) * a + b * (1.0 - a)
        img += rng.normal(0.0, 1.5, img.shape)
        stack.append(np.clip(img, 0, 255).astype(np.uint8))

    a = near[:, :, None]
    reference = np.clip(subject.astype(np.float32) * a
                        + cv2.GaussianBlur(background, (0, 0), 4.0) * (1.0 - a),
                        0, 255).astype(np.uint8)
    return stack, reference, [near, 1.0 - near]


def low_contrast(size=320, slices=3, seed=5):
    """
    A flat, softly lit subject. With little local variation, focus measures have
    almost nothing to compare, so wrong-slice picks become likely.
    """
    rng = np.random.default_rng(seed)
    ys = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, size, dtype=np.float32)[None, :]
    base = 128 + 14 * (xs.repeat(size, 0) + ys.repeat(size, 1))
    img = np.stack([base, base + 4, base - 4], axis=2)
    img += rng.normal(0.0, 1.5, size=(size, size, 3)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Faint markings, only a few levels above the background
    for i in range(14):
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        shade = int(base[c[1], c[0]]) + (10 if i % 2 else -10)
        cv2.circle(img, c, int(rng.integers(14, 34)), (shade, shade, shade), -1)

    masks = _band_masks(size, size, slices)
    return _stack_from_masks(img, masks), img, masks


def saturated_colour(size=320, slices=3, seed=5):
    """Strong primaries: does the fused result keep the colours it was given?"""
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    palette = [(0, 0, 235), (0, 200, 235), (0, 190, 40),
               (215, 60, 0), (200, 0, 190), (30, 220, 220)]
    cell = size // 3
    for i in range(3):
        for j in range(3):
            colour = palette[(i * 3 + j) % len(palette)]
            cv2.rectangle(img, (j * cell, i * cell),
                          ((j + 1) * cell - 1, (i + 1) * cell - 1), colour, -1)
    # Boundaries between saturated fields are where bleeding is visible
    for k in range(1, 3):
        cv2.line(img, (k * cell, 0), (k * cell, size), (255, 255, 255), 1)
        cv2.line(img, (0, k * cell), (size, k * cell), (255, 255, 255), 1)

    masks = _band_masks(size, size, slices)
    return _stack_from_masks(img, masks), img, masks


# key, builder, headline question, what a poor score looks like
SCENARIOS = [
    ("fine_texture", fine_texture,
     "Keeping fine detail",
     "Hair-thin lines and small marks. A weak result looks softened, as if the "
     "photo were slightly out of focus everywhere."),
    ("sensor_noise", sensor_noise,
     "Coping with a noisy, high-ISO shot",
     "Heavy grain. Methods that judge sharpness by how much a region varies can "
     "mistake noise for detail and pick the wrong frame."),
    ("depth_edge", depth_edge,
     "Clean edges where near meets far",
     "A subject against a distant background. A weak result shows a halo or a "
     "blurred fringe tracing the outline."),
    ("long_stack", long_stack,
     "Handling a long stack",
     "Twelve frames instead of three, as in macro work. Watch both quality and "
     "how much slower it gets."),
    ("low_contrast", low_contrast,
     "Flat, low-contrast subjects",
     "A softly lit scene with little texture. With almost nothing to compare, a "
     "method can pick the wrong frame and lose what detail there was."),
    ("saturated_colour", saturated_colour,
     "Holding colour true",
     "Strong, saturated colours meeting at hard borders. A weak result shifts "
     "hues or smears colour across the border."),
]

# Regimes the quality ratchet tracks but the characteristics report does not.
# The report's claims are written about the six above and were validated against
# them; a stack this deep changes where several methods place, so putting it in
# that list would silently rewrite statements nobody has re-checked. It is worth
# a look on its own terms - the guided filter fails to beat a single frame here,
# and StackMFF-V4 loses its colour lead - but that is a separate investigation.
EXTRA_SCENARIOS = [
    ("deep_stack", deep_stack,
     "A real stack's depth, with a background that is never sharp",
     "Sixty-four frames, and a defocused background that drifts sideways as "
     "well as in and out of focus. No two neighbouring frames differ by much, "
     "so a method that needs a clear per-region winner has to fall back on "
     "something - and if that fallback can pick frames far apart in the stack, "
     "the background tears into visibly mismatched patches."),
]

BY_KEY = {key: (builder, title, blurb)
          for key, builder, title, blurb in SCENARIOS + EXTRA_SCENARIOS}


def build(key, **kwargs):
    """Build one scenario by key; returns (stack, reference, masks)."""
    if key not in BY_KEY:
        raise KeyError(f"Unknown scenario {key!r}. Known: {', '.join(BY_KEY)}")
    return BY_KEY[key][0](**kwargs)
