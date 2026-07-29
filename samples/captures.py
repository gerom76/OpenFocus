"""
Focus stacks shot on a real rig, and what can honestly be measured against them.

Everything else in samples/ is *rendered*: layers at known distances, each
convolved with its circle of confusion before being composited. That is what
makes those scenes testable - every pixel carries the index of the frame focused
on it, and a metric can be checked against the answer - and it is also what
limits them. The defocus is the one the generator applied, the grain is the one
it drew, and the frames register perfectly because nothing ever moved.

A capture takes the opposite trade. Nothing about it is known and everything
about it is real: the lens's own defocus and its doubled out-of-focus edges, the
magnification creeping between frames as the rail advances, sensor grain that
was there before anything was added, and a subject with genuine occlusion - legs
and antennae standing clear of the board they lie on.

    from samples.captures import load_capture, load_render, fit_render

    frames, meta = load_capture("electronics_ant", step=12)
    fused = my_fusion(frames)
    ours, theirs = fit_render(fused, load_render("electronics_ant", "HF-C-6"))

## What is under rendered/

Each capture ships fused results on the same stack, one folder per program. They
are not ground truth and the folder is not called that.

* **Affinity Photo** - one render, at the capture's full 6064x4040, 16-bit.
* **Helicon Focus** - ten renders across all three of its methods, at the
  stack's own resolution. Method A is its weighted average, B its depth map and
  C its pyramid, which is the direct counterpart of fusion_methods/pyramid.py.
  A and B are named ``<method>-<radius>-<smoothing>`` and C, which has no
  radius, ``C-<smoothing>``.
* **OpenFocus** - this project's own output. Categorically different from the
  other two and flagged ``ours``, because agreeing with ourselves is evidence of
  nothing: it is a *baseline*, not a reference. Callers comparing against an
  independent result want ``third_party_renders``, which leaves it out.

Two programs, not one, because a single one would be indistinguishable from a
house style. Having Helicon's C at four smoothing settings is what makes the set
worth more than a set of single references: it brackets. A result that recovers
less local contrast than Helicon's pyramid at its lightest smoothing is soft,
one that recovers more than at its heaviest is crunchy, and anything between is
inside the range a mature implementation of the same algorithm considers
reasonable.

## The OpenFocus render is reproducible, and that is its whole value

Every render this project writes carries its settings in an XMP box, and
``read_render_options`` reads them back. The shipped one records 333 frames,
Scale then Homography registration off the first frame, the pyramid at levels 5
/ selectivity 8 / coherence 0.75 / base weighting 0, and auto contrast at 50%
- and running that recipe again reproduces the file to within 0.4 of an 8-bit
level.

So it is the one comparison in this set where pixels *are* comparable, because
both sides come out of the same pipeline rather than two programs that each
aligned the stack their own way. That makes it the anchor for "did a change move
what we ship", which is what the third-party renders cannot answer and what
improving the method most needs.

## None of them is a ground truth

Not one of these renders can be compared pixel by pixel, and the reason is the
same for both programs: each aligned the stack before fusing it, and focus
breathing means the magnification each removed differs frame to frame. Their
output therefore matches the geometry of no raw frame, and no single warp
relates them either - fitting one by ECC reaches a correlation of 0.45 to 0.50
and leaves a residual shift of five to ten pixels. Affinity adds two more
obstacles of its own: it renders at the capture's full size while the frames are
a downscale, so it resolves detail no frame in the stack contains, and it
carries its own tone curve at about half the mean level of the frames.

Read pixel by pixel every one of them ranks a *single defocused frame* about
five decibels above a correct fusion. What they do answer is the coarser
question - did a method find detail in the same places these programs found it,
and how much of it - which ``fusion_metrics.detail_agreement`` measures over
tiles wide enough to absorb the misalignment.

## The captures are not generated

Unlike the rest of samples/, nothing here can be rebuilt by running a script -
these are photographs. Callers that cannot find one should skip rather than
fail; ``is_available`` is there for that.
"""

import glob
import os
import re

import cv2
import numpy as np

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

# Helicon Focus names its methods by letter. Spelt out here because "HF-C-6" on
# a failure line says nothing about what was being compared against.
HELICON_METHODS = {
    "A": "weighted average",
    "B": "depth map",
    "C": "pyramid",
}


def _helicon(filename):
    """Describe one Helicon render from its filename: HF-<method>-<params>."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("-HF-")[-1].split("-")
    method, params = parts[0], [int(p) for p in parts[1:]]
    # C exposes smoothing only; A and B take a radius first.
    smoothing = params[-1] if params else None
    radius = params[0] if method in ("A", "B") and len(params) > 1 else None
    label = f"Helicon Focus method {method} ({HELICON_METHODS.get(method, '?')})"
    if radius is not None:
        label += f", radius {radius}"
    if smoothing is not None:
        label += f", smoothing {smoothing}"
    return {
        "key": f"HF-{'-'.join([method] + [str(p) for p in params])}",
        "program": "Helicon Focus",
        "method": method,
        "method_name": HELICON_METHODS.get(method),
        "radius": radius,
        "smoothing": smoothing,
        "label": label,
        "path": os.path.join("rendered", "HeliconFocus", os.path.basename(filename)),
    }


CAPTURES = {
    "electronics_ant": {
        "description": (
            "An ant on a circuit board, over a USB connector, shot on a focus "
            "rail. 333 frames covering a subject with hard occlusion - legs and "
            "antennae standing clear of the board - against dark, glossy, "
            "largely detail-free plastic, which is the pairing that produces "
            "halos at the edges and grain-driven speckle everywhere else."),
        "frames": "frames/*.webp",
        "frame_count": 333,
        "frame_size": (1819, 1212),      # (width, height)
        "render_globs": ("rendered/HeliconFocus/*.png",),
        "ours_glob": "rendered/OpenFocus/*.jxl",
        "extra_renders": {
            "Affinity": {
                "key": "Affinity",
                "program": "Affinity Photo",
                "method": None,
                "method_name": None,
                "radius": None,
                "smoothing": None,
                "label": "Affinity Photo, at the capture's full 6064x4040",
                "path": os.path.join("rendered", "Affinity",
                                     "electonics_ant_Affinity_20260729.jxl"),
            },
        },
        # Helicon's own pyramid at a middling smoothing: the like-for-like
        # counterpart of fusion_methods/pyramid.py, and the render to reach for
        # when one is wanted rather than all of them.
        "counterpart": "HF-C-6",
    },
}


def list_captures():
    """Every capture name, whether or not its files are present."""
    return sorted(CAPTURES)


def capture_meta(name):
    """The static description of one capture."""
    if name not in CAPTURES:
        raise KeyError(f"Unknown capture {name!r}. Known: {', '.join(list_captures())}")
    return dict(CAPTURES[name], name=name, root=os.path.join(SAMPLES_DIR, name))


def frame_paths(name, step=1, limit=None):
    """
    The capture's frames, in shooting order, taking every `step`-th one.

    Subsampling is by frame count rather than by resolution on purpose: dropping
    frames widens the focus step, which a fusion method has to cope with anyway,
    while resizing would change the one thing a capture is here to provide - what
    the lens actually did at the pixel level.
    """
    meta = capture_meta(name)
    paths = sorted(glob.glob(os.path.join(meta["root"], meta["frames"])))
    paths = paths[::max(1, int(step))]
    return paths[:limit] if limit else paths


def renders(name):
    """
    Every other program's result on this capture, keyed by render name.

    Discovered from disk rather than listed, so dropping another Helicon setting
    into the folder makes it available without editing this file - which is how
    the set grew in the first place.
    """
    meta = capture_meta(name)
    found = {}
    for pattern in meta.get("render_globs", ()):
        for path in sorted(glob.glob(os.path.join(meta["root"], pattern))):
            entry = _helicon(path)
            entry["ours"] = False
            found[entry["key"]] = entry
    for key, entry in meta.get("extra_renders", {}).items():
        if os.path.isfile(os.path.join(meta["root"], entry["path"])):
            found[key] = dict(entry, ours=False)

    # This project's own output, keyed by filename so more than one can sit
    # there - a render per version is exactly how a baseline gets compared.
    for path in sorted(glob.glob(os.path.join(meta["root"], meta.get("ours_glob", "")))
                       if meta.get("ours_glob") else []):
        stem = os.path.splitext(os.path.basename(path))[0]
        found[stem] = {
            "key": stem,
            "program": "OpenFocus",
            "method": None,
            "method_name": None,
            "radius": None,
            "smoothing": None,
            "ours": True,
            "label": f"OpenFocus, {stem}",
            "path": os.path.relpath(path, meta["root"]),
        }
    return found


def third_party_renders(name):
    """
    Every render *not* produced by this project, keyed by render name.

    What the quality comparisons run over. Our own output belongs to the
    baseline question, not to the "did we find what an independent
    implementation found" one, and folding it in would quietly let the method
    grade its own paper.
    """
    return {key: entry for key, entry in renders(name).items() if not entry["ours"]}


def our_renders(name):
    """Only this project's own output: the baselines to measure against."""
    return {key: entry for key, entry in renders(name).items() if entry["ours"]}


def render_meta(name, key):
    """One render's description, including what produced it and at what settings."""
    found = renders(name)
    if key not in found:
        raise KeyError(f"Unknown render {key!r} for {name!r}. "
                       f"Known: {', '.join(sorted(found))}")
    return found[key]


def is_available(name):
    """True when the capture's frames and at least one render are both on disk."""
    if name not in CAPTURES:
        return False
    return bool(frame_paths(name)) and bool(renders(name))


def load_capture(name, step=1, limit=None):
    """
    Return (frames, meta) for one capture.

    frames - list of BGR uint8 images in shooting order
    meta   - the CAPTURES entry, plus the paths actually loaded
    """
    paths = frame_paths(name, step, limit)
    if not paths:
        raise FileNotFoundError(
            f"No frames for capture {name!r} under {capture_meta(name)['root']}")

    frames = []
    for path in paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not decode {path}")
        frames.append(frame)
    return frames, dict(capture_meta(name), paths=paths, loaded=len(frames))


def load_render(name, key=None):
    """
    One other program's result, as 8-bit BGR at whatever size it was made.

    `key` defaults to the capture's counterpart render - Helicon's own pyramid,
    for a capture being used to test ours. Use `fit_render` to put the result on
    the same grid as your own; the sizes differ between programs and none of
    them matches the frames exactly.

    Returns None when the file is present but no decoder is: JPEG XL needs
    imagecodecs, which is an optional dependency.
    """
    meta = capture_meta(name)
    entry = render_meta(name, key or meta["counterpart"])
    path = os.path.join(meta["root"], entry["path"])

    if os.path.splitext(path)[1].lower() == ".jxl":
        from utils import jxl
        if not jxl.is_available():
            return None
        image = jxl.read(path)
    else:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    image = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.dtype == np.uint16:
        image = (image.astype(np.float32) / 257.0).round().clip(0, 255).astype(np.uint8)
    return image


def read_render_options(name, key):
    """
    The settings a render of ours was made with, read back out of its own file.

    Every result this project writes carries them in an XMP box - the same
    strings the render dialog showed - so a file on disk says what produced it
    without a sidecar to lose. Returns an ordered dict of the OpenFocus group,
    or None when the file carries no such box (any render by other software).

    This is what makes the shipped render a baseline rather than a picture: the
    recipe is in it, so it can be run again and the two compared.
    """
    meta = capture_meta(name)
    path = os.path.join(meta["root"], render_meta(name, key)["path"])
    with open(path, "rb") as handle:
        data = handle.read()

    # Walk the JPEG XL container to its XMP box. Small enough to do here; the
    # alternative is importing the app's metadata writer to read its own output.
    payload, position = None, 0
    while position + 8 <= len(data):
        size = int.from_bytes(data[position:position + 4], "big")
        box = data[position + 4:position + 8]
        if size == 0:
            payload = data[position + 8:] if box == b"xml " else None
            break
        if size < 8:
            break
        if box == b"xml ":
            payload = data[position + 8:position + size]
            break
        position += size
    if payload is None:
        return None

    xmp = payload.decode("utf-8", "replace")
    found = {tag: value.strip() for tag, value in
             re.findall(r"<OpenFocus:(\w+)>(.*?)</OpenFocus:\1>", xmp, re.S)}
    return found or None


def fit_render(image, render):
    """
    Put `image` and `render` on one grid, and return both.

    Two corrections, in this order. A render made at the capture's full size is
    downscaled to the image's - INTER_AREA, since it is always a reduction, and
    the aspect ratios agree to well under a percent. Then both are cropped to
    the geometry they share, because a program that aligned the stack trimmed to
    the region every frame covered and so returns a few pixels less than it was
    given.

    What is deliberately *not* corrected is the residual shift between them.
    Nothing rigid removes it - see the module docstring - so the metrics used on
    the pair have to tolerate it rather than pretend it is gone.
    """
    if render.shape[1] > image.shape[1] * 1.5:
        render = cv2.resize(render, (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_AREA)
    height = min(image.shape[0], render.shape[0])
    width = min(image.shape[1], render.shape[1])
    return image[:height, :width], render[:height, :width]
