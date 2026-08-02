"""
Registration accuracy benchmark - terminal tables for every alignment pipeline.

Unlike the fusion benchmark, this one has exact geometric ground truth. The
handheld sample scenes record in ``scene.json`` the 2x3 affine that was applied
to every frame, so the error of an estimated alignment can be computed in
pixels rather than inferred from the picture.

A point ``p`` of the scene lands at ``A_i(p)`` in frame i. Registration
estimates ``H_i`` mapping frame i onto the reference canvas, so the point ends
up at ``H_i(A_i(p))``. Perfect registration makes ``H_i . A_i`` the same map for
every frame, hence

    residual_i = RMS over p of | H_i(A_i(p)) - H_ref(A_ref(p)) |

which is the misregistration in output pixels and is independent of cropping.
``H_i`` is read back by wrapping the crop helper every stage calls, so the
numbers describe the code as it runs rather than a reimplementation of it.

Four modes:

* --accuracy (default) - geometric error of each pipeline, against ground truth
* --quality             - registration -> fusion -> PSNR/SSIM against the
                          scene's own all-in-focus image
* --selection           - which frame the focus measure picks after each
                          pipeline, against the scene's focus_index map
* --detection-width     - geometric error against the width the transform is
                          measured at, on a full-size stack (item 5)

There used to be a fifth, --devices, which ran everything twice: once with the
CuPy warp path and once without. It is gone with the path (item 4), and with it
the device column every table used to carry - registration now produces the same
pixels on every machine, so there are no longer two answers to report.

Examples:
    python tests/benchmark_registration.py
    python tests/benchmark_registration.py --scene flower01_handheld
    python tests/benchmark_registration.py --quality
    python tests/benchmark_registration.py --selection --reference middle
    python tests/benchmark_registration.py --detection-width
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.registration as registration_module
from core.registration import ImageRegistration, resolve_reference_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scenes that carry a per-frame affine, i.e. the ones with geometric truth.
HANDHELD_SCENES = ("handheld_drift", "flower01_handheld")

PIPELINES = [
    ([], "none"),
    (["scale"], "scale"),
    (["homography"], "homography"),
    (["ecc"], "ecc"),
    (["both"], "both (hom+ecc)"),
    (["scale", "homography"], "scale+hom"),
    (["scale", "ecc"], "scale+ecc"),
    (["scale", "both"], "scale+hom+ecc"),
]

class capture_transforms:
    """Record the (transforms, crop) each registration stage finally applies.

    Every implementation calls _compute_valid_region_from_transforms exactly
    once, with the re-referenced transform list and just before folding in the
    crop translation - so wrapping it yields both halves of the map that is
    actually applied, for whichever stages the caller ran.
    """

    def __enter__(self):
        self.stages = []
        self._original = registration_module._compute_valid_region_from_transforms

        def spy(H_matrices, img_shape, margin=2):
            top, bottom, left, right = self._original(H_matrices, img_shape, margin)
            T_crop = np.eye(3)
            if not (top >= bottom or left >= right):
                T_crop[0, 2] = -left
                T_crop[1, 2] = -top
            self.stages.append(
                ([np.array(H, dtype=np.float64) for H in H_matrices], T_crop))
            return top, bottom, left, right

        registration_module._compute_valid_region_from_transforms = spy
        return self

    def __exit__(self, *exc):
        registration_module._compute_valid_region_from_transforms = self._original

    def total(self):
        """Compose every stage into one map per frame, crops included."""
        if not self.stages:
            raise RuntimeError("no registration stage ran")
        count = len(self.stages[0][0])
        composed = []
        for i in range(count):
            M = np.eye(3)
            for H_matrices, T_crop in self.stages:
                M = T_crop @ H_matrices[i] @ M
            composed.append(M)
        return composed


# --------------------------------------------------------------- scene data

def load_scene(name):
    """Return (frames, per-frame affines, ground truth, focus index, metadata)."""
    root = os.path.join(ROOT, "samples", name)
    scene_file = os.path.join(root, "scene.json")
    if not os.path.exists(scene_file):
        raise SystemExit(f"scene {name!r} not found - run python samples/generate_samples.py")

    with open(scene_file, encoding="utf-8") as fh:
        meta = json.load(fh)

    frame_dir = os.path.join(root, "frames")
    names = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))
    frames = [cv2.imread(os.path.join(frame_dir, f), cv2.IMREAD_UNCHANGED) for f in names]

    affines = None
    if "per_frame_affine" in meta:
        affines = [np.vstack([np.array(m, dtype=np.float64), [0.0, 0.0, 1.0]])
                   for m in meta["per_frame_affine"]]

    gt_dir = os.path.join(root, "ground_truth")
    all_in_focus = cv2.imread(os.path.join(gt_dir, "all_in_focus.png"), cv2.IMREAD_UNCHANGED)
    focus_index = cv2.imread(os.path.join(gt_dir, "focus_index.png"), cv2.IMREAD_UNCHANGED)

    return frames, affines, all_in_focus, focus_index, meta


# The detection width only ever did anything below 2048 px, because above it the
# stages replaced the caller's value with 1024 (item 5). Neither scene carrying
# a per-frame affine is that large - 640 and 1280 px - so measuring what the
# setting is worth on a real camera file needs a full-size stack with exact
# geometric truth, which is built here rather than shipped: the frames of
# flower01_subject_hires (2560x1430) put through the same breathing/drift affine
# samples/generate_samples.py applies to handheld_drift.
HIRES_SOURCE = "flower01_subject_hires"


def build_hires_drift(seed=909, noise=1.2, drift=1.0):
    """Return (frames, affines) for a 2560 px stack with known ground truth."""
    from samples.generate_samples import breathing_transform

    frame_dir = os.path.join(ROOT, "samples", HIRES_SOURCE, "frames")
    if not os.path.isdir(frame_dir):
        raise SystemExit(f"{HIRES_SOURCE} not found - run python samples/generate_samples.py")

    names = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))
    rng = np.random.default_rng(seed)
    frames, affines = [], []
    for index, name in enumerate(names):
        img = cv2.imread(os.path.join(frame_dir, name), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        matrix = breathing_transform((h, w), index, len(names), drift)
        warped = cv2.warpAffine(img.astype(np.float32), matrix, (w, h),
                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        warped += rng.normal(0.0, noise, warped.shape).astype(np.float32)
        frames.append(np.clip(warped, 0, 255).astype(np.uint8))
        affines.append(np.vstack([matrix, [0.0, 0.0, 1.0]]))
    return frames, affines


# ------------------------------------------------------------------ metrics

def _project(M, pts):
    q = M @ np.vstack([pts.T, np.ones(len(pts))])
    w = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    return (q[:2] / w).T


def residual_px(H_estimated, affines, shape, reference_index, step=8):
    """Per-frame misregistration in output pixels, on a grid over the frame."""
    h, w = shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    ref = _project(H_estimated[reference_index] @ affines[reference_index], pts)
    return [float(np.sqrt(np.mean(np.sum(
        (_project(H_estimated[i] @ affines[i], pts) - ref) ** 2, axis=1))))
        for i in range(len(H_estimated))]


def _gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img.astype(np.float32)


def locate_and_score(image, ground_truth):
    """PSNR/SSIM of a cropped result, located inside the ground truth first."""
    from tests.fusion_metrics import psnr, ssim
    if image.dtype != ground_truth.dtype:
        image = image.astype(ground_truth.dtype)
    h, w = image.shape[:2]
    match = cv2.matchTemplate(_gray(ground_truth), _gray(image), cv2.TM_SQDIFF)
    _, _, (x, y), _ = cv2.minMaxLoc(match)
    reference = ground_truth[y:y + h, x:x + w]
    return psnr(image, reference), ssim(image, reference)


def focus_argmax(stack, kernel=11):
    """The decision every selection-based fusion method makes: pooled |Laplacian|."""
    maps = []
    for img in stack:
        response = np.abs(cv2.Laplacian(_gray(img), cv2.CV_32F, ksize=3))
        maps.append(cv2.boxFilter(response, -1, (kernel, kernel)))
    return np.argmax(np.stack(maps, axis=0), axis=0)


def sharpness(img):
    return float(cv2.Laplacian(_gray(img), cv2.CV_32F).var())


# -------------------------------------------------------------------- runner

def run_pipeline(frames, stages, downscale_width, reference_index):
    """Apply the stages in order and return (images, composed maps, seconds)."""
    with capture_transforms() as captured:
        images = [f.copy() for f in frames]
        started = time.time()
        for mode in stages:
            images = ImageRegistration(
                method=mode, downscale_width=downscale_width,
                reference_index=reference_index).process(
                    images, output_path=None, thread_count=4)
        elapsed = time.time() - started
    composed = captured.total() if stages else [np.eye(3)] * len(frames)
    return images, composed, elapsed


# --------------------------------------------------------------------- modes

def report_accuracy(scene, args):
    frames, affines, _, _, meta = load_scene(scene)
    if affines is None:
        print(f"  {scene}: no per_frame_affine in scene.json - skipped")
        return
    shape = frames[0].shape[:2]
    reference_index = resolve_reference_index(args.reference, len(frames))
    baseline = residual_px([np.eye(3)] * len(frames), affines, shape, reference_index)

    print(f"\n### {scene}  {meta['resolution'][0]}x{meta['resolution'][1]}, "
          f"{meta['frame_count']} frames, reference={args.reference}")
    print(f"    unregistered: mean {np.mean(baseline):5.2f} px, worst {np.max(baseline):5.2f} px")
    print(f"    {'pipeline':<18}{'mean px':>9}{'worst px':>10}"
          f"{'gain':>7}{'time':>8}{'kept':>8}")
    for stages, label in PIPELINES:
        if not stages:
            continue
        images, composed, elapsed = run_pipeline(
            frames, stages, args.downscale_width, reference_index)
        residual = residual_px(composed, affines, shape, reference_index)
        kept = 100.0 * images[0].shape[0] * images[0].shape[1] / (shape[0] * shape[1])
        gain = np.mean(baseline) / np.mean(residual)
        flag = "" if gain >= 1.0 else "  <- worse than no registration"
        print(f"    {label:<18}{np.mean(residual):9.2f}"
              f"{np.max(residual):10.2f}{gain:6.2f}x{elapsed:7.2f}s{kept:7.1f}%{flag}")


def report_quality(scene, args):
    from tests import fusion_registry
    frames, _, all_in_focus, _, meta = load_scene(scene)
    if all_in_focus is None:
        print(f"  {scene}: no all_in_focus ground truth - skipped")
        return
    method = fusion_registry.get(args.method)
    reference_index = resolve_reference_index(args.reference, len(frames))

    print(f"\n### {scene}: registration -> {method.label} -> vs ground truth")
    print(f"    {'pipeline':<18}{'PSNR':>8}{'SSIM':>9}{'sharpness':>12}")
    for stages, label in PIPELINES:
        images, _, _ = run_pipeline(
            frames, stages, args.downscale_width, reference_index)
        fused = method.fuse(images, **method.params)
        score, structure = locate_and_score(fused, all_in_focus)
        print(f"    {label:<18}{score:8.2f}{structure:9.4f}{sharpness(fused):12.1f}")


def report_selection(scene, args):
    frames, _, _, focus_index, meta = load_scene(scene)
    if focus_index is None:
        print(f"  {scene}: no focus_index ground truth - skipped")
        return
    reference_index = resolve_reference_index(args.reference, len(frames))
    truth = 100.0 * float((focus_index == reference_index).mean())

    print(f"\n### {scene}: share of the picture the focus measure sources from "
          f"the reference frame (index {reference_index})")
    print(f"    ground truth: {truth:.1f}% of pixels are genuinely sharpest there")
    print(f"    {'pipeline':<18}{'share':>8}{'vs truth':>10}")
    for stages, label in PIPELINES:
        images, _, _ = run_pipeline(
            frames, stages, args.downscale_width, reference_index)
        share = 100.0 * float((focus_argmax(images) == reference_index).mean())
        # A truth share of zero - the reference frame is nowhere the
        # sharpest - makes the ratio meaningless, so report the share alone.
        ratio = f"{share / truth:8.1f}x" if truth > 0 else "     n/a"
        print(f"    {label:<18}{share:7.1f}%{ratio:>10}")


def report_detection_width(args):
    """What the detection width buys, on a frame large enough to have been overridden."""
    frames, affines = build_hires_drift(seed=args.seed)
    shape = frames[0].shape[:2]
    reference_index = resolve_reference_index(args.reference, len(frames))
    baseline = residual_px([np.eye(3)] * len(frames), affines, shape, reference_index)

    widths = [int(w) for w in args.widths.split(",")]
    print(f"\n### {HIRES_SOURCE} + per-frame affine  {shape[1]}x{shape[0]}, "
          f"{len(frames)} frames, reference={args.reference}")
    print(f"    unregistered: mean {np.mean(baseline):5.2f} px, "
          f"worst {np.max(baseline):5.2f} px")
    print("    detection above the frame width is full-resolution detection - "
          "the frames are never upsampled")
    print(f"    {'pipeline':<14}{'width':>7}{'mean px':>10}{'worst px':>10}"
          f"{'gain':>7}{'time':>8}{'kept':>8}")
    for stages, label in PIPELINES:
        if not stages:
            continue
        for width in widths:
            images, composed, elapsed = run_pipeline(
                frames, stages, width, reference_index)
            residual = residual_px(composed, affines, shape, reference_index)
            kept = 100.0 * images[0].shape[0] * images[0].shape[1] / (shape[0] * shape[1])
            print(f"    {label:<14}{width:>7}{np.mean(residual):10.2f}"
                  f"{np.max(residual):10.2f}"
                  f"{np.mean(baseline) / np.mean(residual):6.2f}x"
                  f"{elapsed:7.2f}s{kept:7.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Registration accuracy benchmark against sample ground truth")
    parser.add_argument("--scene", default=None,
                        help=f"one scene name (default: all of {', '.join(HANDHELD_SCENES)})")
    parser.add_argument("--accuracy", action="store_true",
                        help="geometric error against per_frame_affine (default)")
    parser.add_argument("--quality", action="store_true",
                        help="fused PSNR/SSIM against all_in_focus.png")
    parser.add_argument("--selection", action="store_true",
                        help="which frame the focus measure picks, vs focus_index.png")
    parser.add_argument("--detection-width", action="store_true", dest="detection_width",
                        help="sweep the detection width on a full-size stack (item 5)")
    parser.add_argument("--reference", default="first",
                        choices=["first", "middle", "last"], help="reference frame mode")
    parser.add_argument("--downscale-width", type=int, default=1024, dest="downscale_width",
                        help="detection width, honoured at every frame size")
    parser.add_argument("--widths", default="512,1024,2048,4096",
                        help="widths swept by --detection-width")
    parser.add_argument("--seed", type=int, default=909,
                        help="noise seed for the stack --detection-width builds")
    parser.add_argument("--method", default="pyramid", help="fusion method for --quality")
    args = parser.parse_args()

    if not (args.accuracy or args.quality or args.selection or args.detection_width):
        args.accuracy = True

    if args.detection_width:
        report_detection_width(args)
        if not (args.accuracy or args.quality or args.selection):
            return 0

    scenes = [args.scene] if args.scene else list(HANDHELD_SCENES)
    for scene in scenes:
        if args.accuracy:
            report_accuracy(scene, args)
        if args.quality:
            report_quality(scene, args)
        if args.selection:
            report_selection(scene, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
