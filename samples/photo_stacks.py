"""
Focus stacks rendered from a real photograph.

The scenes in generate_samples.py are procedural all the way down, which makes
them exactly repeatable but leaves them looking like what they are. A method
tuned only against fractal noise and drawn shapes has never met the things that
actually break focus measures on a real subject: pollen grains a few pixels
across, translucent petal edges, specular sheen that moves with the light,
sensor grain that is already in the image before anything is added to it.

So these take the opposite trade. The texture is a photograph - a fuchsia shot
against slate, itself the output of a focus stack, so it is sharp throughout and
can serve as the all-in-focus truth. Everything else works exactly as in the
synthetic scenes: the frame is split into depth planes, each is convolved with
the disc its circle of confusion would have, and the planes are composited front
to back so a defocused filament spreads over the corolla behind it.

What is *not* real here is the depth. A single photograph does not carry one,
and nothing in this repository can recover it, so the depth map is assigned:

* the flower is segmented from the background by saturation and brightness
* within the flower, depth ramps along the axis from the style tip at the lower
  left to the far sepal at the upper right - which is how the subject is
  actually arranged in this frame, the near-to-far axis a photographer would
  have racked along
* the stamens and style are picked out as the structures a morphological
  opening removes, and pushed in front of the bell they radiate from, because
  in this composition they genuinely are in front of it
* the slate is a plane receding towards the top of the frame

That makes the depth plausible and the *rendering* physically consistent with
it, which is what the training signal needs: the labels are exact for the images
that were produced, because the images were produced from the labels. It does
not make the depth true to the flower that was photographed. Anything measuring
depth-estimation accuracy against these will be measuring agreement with an
assumption - see README.md.

Generated as part of `python samples/generate_samples.py`; run this module
directly to rebuild only these scenes.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samples.generate_samples import (COC_SCALE, MAX_RADIUS,  # noqa: E402
                                      bands_from_depthmap, build_scene,
                                      focus_schedule)

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(SAMPLES_DIR, "flower01", "Sharing.png")

# Render width. The source is 2560 px wide; halving it keeps the pollen grains
# and the filament edges - the detail worth stacking for - while keeping a
# frame small enough that a stack of them is a reasonable thing to store.
WIDTH = 1280

# The source's own width, for the high-resolution variant. Nothing is
# interpolated up to reach it, and the aperture scales with the frame, so it is
# the same scene rendered onto four times the pixels rather than a different
# one: a method can be compared against its own result at 1280 and the only
# thing that changed is resolution. Frames are ~4 MB each and take ~15 s to
# render, which is why one sequence carries the flag rather than all three.
WIDTH_HIRES = 2560

# Scene distances, in the same arbitrary units as the synthetic scenes.
SUBJECT_NEAR = 0.95     # the bell's nearest rim
SUBJECT_FAR = 1.45      # the sepal tip pointing away at the upper right
FILAMENT_LIFT = 0.14    # how far in front of the bell the stamens stand
GROUND_NEAR = 1.62      # the slate at the bottom edge of the frame
GROUND_SPAN = 1.05      # and how much further it has gone by the top

# Ends of the near-far axis in source pixels: the style tip at the lower left,
# and the tip of the sepal pointing away at the upper right.
AXIS_NEAR = (400.0, 1150.0)
AXIS_FAR = (1800.0, 60.0)


# ---------------------------------------------------------------------------
# Reading the photograph
# ---------------------------------------------------------------------------

def subject_mask(image):
    """
    Separate the flower from the slate.

    The background is near-black and unsaturated and the subject is neither, so
    a threshold on saturation and brightness does most of the work. The rest is
    cleanup: the pale style and the pollen at the filament tips are bright but
    barely saturated, which is why brightness alone has to be able to claim a
    pixel, and the corolla has dark folds inside it that have to be filled
    rather than punched out of the middle of the subject.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    mask = (((saturation > 80) & (value > 55)) | (value > 110)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # Drop specks: dust on the slate, and the odd bright fleck of pollen that
    # has fallen off the flower and is not part of it
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 300:
            kept[labels == index] = 1

    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    height, width = kept.shape
    outside = kept.copy()
    cv2.floodFill(outside, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 1)
    return (kept | (1 - outside)).astype(np.uint8)


def filament_mask(mask):
    """
    The stamens and the style: everything a morphological opening removes.

    An opening deletes whatever is narrower than its kernel, so the difference
    between the mask and its opening is exactly the thin structure. That also
    catches the pointed tips of the sepals, which are thin but are not in front
    of anything, so only long components survive the filter.
    """
    thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((45, 45), np.uint8))
    thin = (mask.astype(bool) & ~thick.astype(bool)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    strands = np.zeros(mask.shape, dtype=bool)
    for index in range(1, count):
        _, _, box_w, box_h, area = stats[index]
        if area > 2500 and max(box_w, box_h) > 250:
            strands |= (labels == index)
    return strands


def depth_map(mask, strands):
    """Assign every pixel a distance. See the module docstring for the model."""
    height, width = mask.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    axis = np.float32(AXIS_FAR) - np.float32(AXIS_NEAR)
    axis /= np.linalg.norm(axis)
    along = (xs - AXIS_NEAR[0]) * axis[0] + (ys - AXIS_NEAR[1]) * axis[1]

    subject_pixels = mask.astype(bool)
    along = np.clip(along / float(along[subject_pixels].max()), 0.0, 1.0)

    subject = SUBJECT_NEAR + (SUBJECT_FAR - SUBJECT_NEAR) * along
    subject[strands] -= FILAMENT_LIFT

    ground = GROUND_NEAR + GROUND_SPAN * (1.0 - ys / height) ** 1.3
    return subject.astype(np.float32), ground.astype(np.float32)


def inpaint_background(image, mask):
    """
    Invent the slate hidden behind the flower.

    Every layer is blurred independently and then composited, so when the
    foreground defocuses it thins out and whatever is behind it shows through.
    Behind the flower there is nothing - it was never photographed - and
    compositing against that hole would ring the subject with a dark halo in
    every defocused frame, teaching a method that occlusion edges go dark.

    The slate is smooth and almost featureless, so filling it is easy and the
    fill is done at half size, which is faster and no worse.
    """
    height, width = mask.shape
    grown = cv2.dilate(mask, np.ones((9, 9), np.uint8))   # keep petal colour out

    small = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(grown, (width // 2, height // 2),
                            interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_mask, 12, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.where(grown[:, :, None].astype(bool), filled, image)


# ---------------------------------------------------------------------------
# Assembling the scene
# ---------------------------------------------------------------------------

_SCENE_CACHE = {}


def build_layers(width=WIDTH):
    """
    The photograph as depth planes, ready to render. Cached: the segmentation
    and the inpainting are the same for every sequence built from this frame.
    """
    if width in _SCENE_CACHE:
        return _SCENE_CACHE[width]

    image = cv2.imread(SOURCE, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(SOURCE)
    image = image[:, :, :3]     # the source carries an opaque alpha channel

    mask = subject_mask(image)
    strands = filament_mask(mask)
    subject_depth, ground_depth = depth_map(mask, strands)
    background = inpaint_background(image, mask)

    # Downscale last, so the segmentation ran on every pixel that was captured.
    # INTER_AREA on the mask is what feathers the silhouette: a pixel half
    # covered by a petal comes back as half opaque, which is what it was.
    height = int(round(image.shape[0] * width / image.shape[1]))
    size = (width, height)
    scaled = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    scaled_bg = cv2.resize(background, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    subject_depth = cv2.resize(subject_depth, size, interpolation=cv2.INTER_LINEAR)
    ground_depth = cv2.resize(ground_depth, size, interpolation=cv2.INTER_LINEAR)

    # The slate is a full plane behind everything, not a cut-out.
    #
    # It gets more planes than the flower does despite being far simpler, and
    # that is not waste: it is smooth, dark and always defocused, so a step
    # between two planes has nothing to hide behind. At 16 planes the strongest
    # vertical periodicity in the rendered slate sat exactly on the plane pitch
    # and stood above the photograph's own texture - visible banding, and a
    # structure a fusion method would have learnt to see. At 32 it moves off
    # the pitch and what is left is smoother than the real slate.
    layers = bands_from_depthmap(scaled_bg, np.ones(alpha.shape, np.float32),
                                 ground_depth, 32)
    layers += bands_from_depthmap(scaled, alpha, subject_depth, 30)

    scale = width / 640.0       # same aperture as the synthetic scenes
    detail = {
        "layers": layers,
        "coc_scale": COC_SCALE * scale,
        "max_radius": MAX_RADIUS * scale,
        "subject_fraction": round(float(alpha.mean()), 4),
    }
    _SCENE_CACHE[width] = detail
    return detail


def _builder(near, far, frames, width=WIDTH):
    """A build_scene-compatible builder: () -> (layers, focus distances)."""
    def build():
        return build_layers(width)["layers"], focus_schedule(near, far, frames)
    return build


# Where the focus is racked to, and how far. The subject spans roughly 0.81 to
# 1.45 and the slate runs from 1.62 back to 2.67.
SEQUENCES = [
    ("flower01_subject",
     "A macro stack of the flower alone: focus covers the subject and stops, so "
     "the slate behind it is defocused in every frame and never resolves. The "
     "everyday case, and the one where a method has to leave the background "
     "alone instead of hunting for detail in it.",
     dict(near=0.78, far=1.52, frames=14),
     dict(seed=701, noise=1.2)),

    ("flower01_full",
     "The same frame with focus racked all the way through the slate as well, "
     "so every part of the picture is sharp in some frame. Twenty frames over a "
     "much longer throw, which is where neighbouring frames stop differing by "
     "much and tie-breaking starts to decide the picture.",
     dict(near=0.78, far=2.75, frames=20),
     dict(seed=702, noise=1.6)),

    ("flower01_handheld",
     "The subject stack again, with focus breathing, drift, roll and exposure "
     "flicker on top - what the same shoot looks like off a rail that is not "
     "quite rigid. scene.json carries the affine applied to each frame.",
     dict(near=0.78, far=1.52, frames=14),
     dict(seed=703, noise=2.2, drift=1.0, exposure_drift=0.04)),

    ("flower01_subject_hires",
     "flower01_subject at the photograph's own 2560x1430, with the aperture "
     "scaled to match so it is the same scene on four times the pixels. For "
     "the questions that only appear at capture resolution: whether a fixed "
     "block or kernel size still spans the detail it was tuned for, what a "
     "method costs in time and memory on a real frame, and whether tiling a "
     "large image leaves seams a small one never showed.",
     dict(near=0.78, far=1.52, frames=14, width=WIDTH_HIRES),
     dict(seed=704, noise=1.2)),
]


def build_all(width=None, only=None):
    """
    Render every sequence. Returns one metadata dict per scene.

    `width` overrides what each sequence asks for, so the whole set can be
    rebuilt at one size; `only` restricts the run to scenes whose name contains
    it, which is what makes rebuilding a single expensive scene bearable.
    """
    if not os.path.exists(SOURCE):
        print("  (no %s - skipping the photo scenes)"
              % os.path.relpath(SOURCE, SAMPLES_DIR))
        return []

    metas = []
    for name, blurb, schedule, options in SEQUENCES:
        if only and only not in name:
            continue

        schedule = dict(schedule)
        scene_width = width or schedule.pop("width", WIDTH)
        schedule.pop("width", None)
        detail = build_layers(scene_width)

        metas.append(build_scene(
            name, _builder(width=scene_width, **schedule), blurb,
            coc_scale=detail["coc_scale"], max_radius=detail["max_radius"],
            extra_meta={
                "source_image": os.path.relpath(SOURCE, SAMPLES_DIR).replace("\\", "/"),
                "depth_source": ("assigned, not measured - a photograph carries no "
                                 "depth map; see samples/photo_stacks.py"),
                "subject_fraction": detail["subject_fraction"],
            },
            **options))
    return metas


def merge_into_manifest(metas):
    """
    Fold freshly built scenes into manifest.json, leaving the others alone.

    generate_samples.py writes the manifest wholesale because it builds
    everything. This module can be asked to rebuild one scene, and a rebuild
    that left the manifest describing the previous render would be worse than
    no rebuild at all: load_stack() reads the manifest to find out what exists,
    so a scene missing from it is a scene nothing can open.
    """
    import json

    path = os.path.join(SAMPLES_DIR, "manifest.json")
    scenes = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            scenes = json.load(handle)["scenes"]

    by_name = {scene["name"]: index for index, scene in enumerate(scenes)}
    for meta in metas:
        if meta["name"] in by_name:
            scenes[by_name[meta["name"]]] = meta
        else:
            scenes.append(meta)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"scenes": scenes}, handle, indent=2)
    return scenes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the photo-derived stacks.")
    parser.add_argument("--width", type=int, default=None,
                        help="render every sequence at this width instead of its own")
    parser.add_argument("--only", default=None,
                        help="only build scenes whose name contains this")
    args = parser.parse_args()

    print("Rendering the photo-derived stacks")
    built = build_all(args.width, args.only)
    if built:
        total = merge_into_manifest(built)
        print("Manifest now lists %d scenes" % len(total))
