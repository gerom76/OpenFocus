"""
Load the sample stacks and their ground truth.

Small on purpose: reading a folder of frames is three lines, but getting the
details wrong is easy and quiet. The 16-bit scene has to come back at the same
scale as the 8-bit ones or every metric shifts; the depth map is stored
normalised and has to be mapped back to scene distances before it means
anything. Both are handled here so no caller has to remember.

    from samples.load import load_stack, list_scenes

    stack, truth, meta = load_stack("macro_dome")
    fused = my_fusion(stack)
    print(psnr(fused, truth["all_in_focus"]))
"""

import glob
import json
import os

import cv2
import numpy as np

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


def list_scenes():
    """Every scene name, in the order the generator writes them."""
    with open(os.path.join(SAMPLES_DIR, "manifest.json"), encoding="utf-8") as handle:
        return [scene["name"] for scene in json.load(handle)["scenes"]]


def scene_meta(name):
    """The scene.json for one sample: focus distances, noise, drift, seeds."""
    with open(os.path.join(SAMPLES_DIR, name, "scene.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _read(path, as_uint8=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if as_uint8 and img.dtype == np.uint16:
        # 16-bit scenes are stored at full range; the fusion methods and the
        # metrics both work in 8-bit, so scale rather than truncate.
        img = (img.astype(np.float32) / 257.0).round().clip(0, 255).astype(np.uint8)
    return img


def load_stack(name, as_uint8=True):
    """
    Return (frames, ground_truth, meta) for one scene.

    frames       - list of BGR images, ordered near focus to far focus
    ground_truth - dict with:
                     all_in_focus  BGR, what a perfect fusion reproduces
                     depth         float32, scene distances, same units as
                                   meta["focus_distances"]
                     focus_index   uint8, index of the sharpest frame per pixel
    meta         - the scene.json contents

    Pass as_uint8=False to keep the 16-bit scene at 16 bits.
    """
    root = os.path.join(SAMPLES_DIR, name)
    if not os.path.isdir(root):
        raise KeyError(f"Unknown sample {name!r}. Known: {', '.join(list_scenes())}")

    meta = scene_meta(name)
    frames = [_read(path, as_uint8)
              for path in sorted(glob.glob(os.path.join(root, "frames", "*.png")))]

    truth_dir = os.path.join(root, "ground_truth")
    normalised = _read(os.path.join(truth_dir, "depth_map.png"), as_uint8=False)
    near, far = meta["depth_range"]
    depth = near + (far - near) * (normalised.astype(np.float32) / 65535.0)

    ground_truth = {
        "all_in_focus": _read(os.path.join(truth_dir, "all_in_focus.png"), as_uint8),
        "depth": depth,
        "focus_index": _read(os.path.join(truth_dir, "focus_index.png"), as_uint8=False),
    }
    return frames, ground_truth, meta


def load_all(as_uint8=True):
    """Every scene, for a sweep that runs across the whole set."""
    return {name: load_stack(name, as_uint8) for name in list_scenes()}
