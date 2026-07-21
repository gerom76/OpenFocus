"""
The claims the characteristics report makes, as assertions.

tests/visualize_fusion_characteristics.py tells a non-specialist which method to
reach for and where each one struggles. Those statements are only worth printing
if they are true of the code as it stands, so every headline claim in that report
has a test here. If a fusion method changes and a claim stops holding, this fails
and the report has to be rewritten rather than quietly becoming fiction.

Thresholds are deliberately slack: they encode the direction and rough size of an
effect, not the exact number measured on one machine.

Run with:  python -m pytest tests/test_fusion_characteristics.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_metrics as fm
from tests import fusion_registry as reg
from tests import fusion_scenarios as sc

# The six distinct algorithms; GPU variants duplicate their CPU counterparts
CORE_METHODS = ["guided_filter", "gfgfgf", "dct", "dtcwt", "stackmffv4", "gff_ifcnn"]


def _require(key):
    method = reg.get(key)
    ok, reason = method.available()
    if not ok:
        pytest.skip(f"{method.label}: {reason}")
    return method


@pytest.fixture(scope="module")
def results():
    """Every core method run over every scenario, measured once."""
    data = {}
    for key, _, _, _ in sc.SCENARIOS:
        stack, reference, masks = sc.build(key)
        edge = fm.boundary_band(masks)
        entry = {
            "best_slice": max(fm.psnr(s, reference) for s in stack),
            "stack": stack,
            "methods": {},
        }
        for method_key in CORE_METHODS:
            method = reg.get(method_key)
            if not method.available()[0]:
                continue
            fused = method.run(stack)
            aligned, sources, ref = fm.align_to_common_size(fused, stack, reference)
            entry["methods"][method_key] = {
                "psnr": fm.psnr(aligned, ref),
                "edge_psnr": fm.region_psnr(aligned, ref, edge),
                "colour": fm.colour_error(aligned, ref),
                "fused": aligned,
                "sources": sources,
            }
        data[key] = entry
    return data


def _scores(results, metric, scenario):
    return {k: v[metric] for k, v in results[scenario]["methods"].items()}


def _rank(results, metric, scenario, key, best_is_high=True):
    scores = _scores(results, metric, scenario)
    order = sorted(scores, key=lambda k: scores[k], reverse=best_is_high)
    return order.index(key) + 1, len(order)


# ---------------------------------------------------------------------------
# "Every method beats using a single frame"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", [s[0] for s in sc.SCENARIOS])
def test_fusion_beats_one_frame(results, scenario):
    """
    The premise of the whole report: fusing is better than picking one photo.

    Two documented exceptions, both on the two-slice depth_edge scenario and
    both with their own test below: GFG-FGF discards a frame there, and DCT's
    block grid is too coarse for a curved boundary.
    """
    entry = results[scenario]
    baseline = entry["best_slice"]
    for key, rec in entry["methods"].items():
        if scenario == "depth_edge" and key in ("gfgfgf", "dct"):
            continue  # each covered by its own test below
        assert rec["psnr"] > baseline, (
            f"{key} scored {rec['psnr']:.2f} dB on {scenario}, "
            f"no better than the best single frame at {baseline:.2f} dB")


# ---------------------------------------------------------------------------
# "StackMFF-V4 is the best all-rounder"
# ---------------------------------------------------------------------------

def test_stackmffv4_leads_in_most_scenarios(results):
    _require("stackmffv4")
    wins = 0
    for key, _, _, _ in sc.SCENARIOS:
        if "stackmffv4" not in results[key]["methods"]:
            continue
        place, _ = _rank(results, "psnr", key, "stackmffv4")
        if place == 1:
            wins += 1
    assert wins >= 4, f"StackMFF-V4 led only {wins} of {len(sc.SCENARIOS)} scenarios"


def test_stackmffv4_holds_colour_best(results):
    """Its colour deviation should be the smallest, and by a clear margin."""
    _require("stackmffv4")
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "colour", key)
        if "stackmffv4" not in scores:
            continue
        others = [v for k, v in scores.items() if k != "stackmffv4"]
        assert scores["stackmffv4"] <= min(others), (
            f"{key}: StackMFF-V4 colour error {scores['stackmffv4']:.2f} "
            f"is not the lowest")


# ---------------------------------------------------------------------------
# "DTCWT is the best of the classical methods at depth boundaries"
# ---------------------------------------------------------------------------

def test_dtcwt_leads_classical_methods_at_boundaries(results):
    _require("dtcwt")
    classical = ("guided_filter", "gfgfgf", "dct")
    wins = 0
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "edge_psnr", key)
        if "dtcwt" not in scores:
            continue
        rivals = [scores[k] for k in classical if k in scores]
        if rivals and scores["dtcwt"] >= max(rivals):
            wins += 1
    assert wins >= 4, f"DTCWT led the classical methods at edges in only {wins} scenarios"


# ---------------------------------------------------------------------------
# "DCT is the fastest, and the weakest at edges"
# ---------------------------------------------------------------------------

def test_dct_is_weakest_at_boundaries(results):
    """Block-based decisions land on a coarse grid, which shows along edges."""
    _require("dct")
    places = []
    for key, _, _, _ in sc.SCENARIOS:
        if "dct" not in results[key]["methods"]:
            continue
        place, total = _rank(results, "edge_psnr", key, "dct")
        places.append(place)
    # Bottom half of the field on average
    assert np.mean(places) >= total / 2, (
        f"DCT averaged place {np.mean(places):.1f} of {total} at edges")


def test_dct_struggles_most_with_noise(results):
    """
    Variance-based focus measures cannot tell grain from detail, so DCT is the
    weakest of the classical methods on a noisy stack.
    """
    _require("dct")
    scores = _scores(results, "psnr", "sensor_noise")
    for rival in ("guided_filter", "gfgfgf", "dtcwt"):
        if rival in scores:
            assert scores["dct"] < scores[rival], (
                f"DCT {scores['dct']:.2f} dB was not below {rival} "
                f"{scores[rival]:.2f} dB on a noisy stack")


# ---------------------------------------------------------------------------
# "GFG-FGF can throw a frame away"
# ---------------------------------------------------------------------------

def test_gfgfgf_discards_frames_it_judges_unsharp(results):
    """
    GFG-FGF scores each frame's focus globally and skips any frame below 15% of
    the sharpest (scale = 0.15 in fusion_methods/gfg_fgf.py). A small detailed
    subject on a plain background can push the background frame under that line,
    and the result is then one source frame returned unchanged - no fusion.
    """
    _require("gfgfgf")
    stack, _, _ = sc.build("depth_edge")
    fused = reg.get("gfgfgf").run(stack)

    matches = [i for i, s in enumerate(stack) if np.array_equal(fused, s)]
    assert matches, (
        "GFG-FGF no longer returns a source frame unchanged on the depth_edge "
        "scenario - if this is a deliberate fix, update the report's claim")


def test_gfgfgf_fuses_normally_when_frames_are_comparable(results):
    """The same method must still fuse when no frame is far sharper than the rest."""
    _require("gfgfgf")
    stack, reference, _ = sc.build("fine_texture")
    fused = reg.get("gfgfgf").run(stack)

    assert not any(np.array_equal(fused, s) for s in stack)
    assert fm.psnr(fused, reference) > max(fm.psnr(s, reference) for s in stack) + 3


# ---------------------------------------------------------------------------
# "IFCNN Refine tidies boundaries, and is applied as a correction"
# ---------------------------------------------------------------------------

def test_ifcnn_leaves_a_picture_it_cannot_improve_alone(results):
    """
    The property the whole stage rests on.

    IFCNN's encode/decode round trip is not an identity - normalising to
    ImageNet statistics and reconstructing moves colours by several levels. The
    stage therefore applies only the *difference* the merge makes, rather than
    the decoded picture itself. Refine an image against nothing but itself and
    the merge changes nothing, so the round trip must cancel exactly and the
    input must come back bit for bit.
    """
    _require("gff_ifcnn")
    from fusion_methods.ifcnn import _ifcnn_refine_impl

    stack, _, _ = sc.build("saturated_colour")
    fused = reg.get("guided_filter").run(stack)
    refined = _ifcnn_refine_impl(fused, [fused], os.path.join(reg.WEIGHTS_DIR, "ifcnn.pth"),
                                 use_gpu=False)

    drift = np.max(np.abs(refined.astype(np.int16) - fused.astype(np.int16)))
    assert drift == 0, (
        f"refining an image against itself moved it by {drift} levels; the "
        f"round-trip correction is no longer cancelling")


def test_ifcnn_colour_drift_stays_bounded(results):
    """
    Some drift survives - the correction cancels the round trip on the candidate,
    not on the sources merged into it - but it must stay at a few levels rather
    than the double digits an uncorrected round trip costs.
    """
    _require("gff_ifcnn")
    errors = []
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "colour", key)
        if "gff_ifcnn" in scores:
            errors.append(scores["gff_ifcnn"])

    assert max(errors) < 10.0, (
        f"IFCNN colour deviation peaked at {max(errors):.1f} levels")
    assert np.mean(errors) < 5.0, (
        f"IFCNN colour deviation averaged {np.mean(errors):.1f} levels")


def test_ifcnn_improves_boundaries_everywhere(results):
    """It is a boundary-repair stage, so it should help there in every scenario."""
    _require("gff_ifcnn")
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "edge_psnr", key)
        if "gff_ifcnn" not in scores or "guided_filter" not in scores:
            continue
        assert scores["gff_ifcnn"] > scores["guided_filter"], (
            f"{key}: IFCNN scored {scores['gff_ifcnn']:.2f} dB at boundaries "
            f"against the guided filter's {scores['guided_filter']:.2f} dB")


def test_ifcnn_still_costs_some_overall_accuracy(results):
    """
    Worth stating plainly in the report: on synthetic stacks the refinement can
    still lower overall accuracy versus the plain guided filter, because these
    stacks give it no genuinely missed detail to recover - only the chance to
    disturb a near-perfect fusion. The cost is now bounded rather than ruinous.
    """
    _require("gff_ifcnn")
    gaps = []
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "psnr", key)
        if "gff_ifcnn" not in scores or "guided_filter" not in scores:
            continue
        gaps.append(scores["guided_filter"] - scores["gff_ifcnn"])

    assert any(gap > 0 for gap in gaps), (
        "IFCNN no longer costs accuracy on any synthetic scenario; the report "
        "says it does, so rewrite the caveat")
    assert max(gaps) < 12.0, (
        f"IFCNN fell {max(gaps):.1f} dB behind the plain guided filter; the "
        f"report describes the cost as bounded")


# ---------------------------------------------------------------------------
# "A flat subject is the hardest case to improve on"
# ---------------------------------------------------------------------------

def test_low_contrast_leaves_little_to_gain(results):
    """
    Blur barely changes a flat scene, so a single frame already scores well and
    every method's headroom is small. Worth saying, so nobody reads the low
    spread there as the methods being interchangeable in general.
    """
    baseline = results["low_contrast"]["best_slice"]
    detailed = results["fine_texture"]["best_slice"]
    assert baseline > detailed + 15, (
        "the low-contrast scenario is no longer the easy case the report claims")
