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
    both with their own test below: GFG-FGF discards a frame there, and IFCNN's
    colour shift costs more than the detail it recovers.
    """
    entry = results[scenario]
    baseline = entry["best_slice"]
    for key, rec in entry["methods"].items():
        if scenario == "depth_edge" and key in ("gfgfgf", "dct", "gff_ifcnn"):
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
# "IFCNN Refine tidies boundaries but shifts colour"
# ---------------------------------------------------------------------------

def test_ifcnn_shifts_colour(results):
    """
    The refinement stage normalises to ImageNet statistics and reconstructs, and
    the round trip costs colour fidelity - several times the deviation of the
    guided filter result it started from.
    """
    _require("gff_ifcnn")
    ratios = []
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "colour", key)
        if "gff_ifcnn" not in scores or "guided_filter" not in scores:
            continue
        ratios.append(scores["gff_ifcnn"] / max(scores["guided_filter"], 1e-6))

    assert np.mean(ratios) > 3.0, (
        f"IFCNN colour deviation averaged only {np.mean(ratios):.1f}x the "
        f"guided filter's")


def test_ifcnn_can_score_below_a_single_frame(results):
    """
    The colour shift is large enough to outweigh the detail recovered: on the
    two-slice depth_edge scenario the refined result scores below simply keeping
    the sharpest source frame. Worth stating outright in the report, because it
    is the one case where running a stage makes the picture measurably worse
    than doing nothing.
    """
    _require("gff_ifcnn")
    entry = results["depth_edge"]
    refined = entry["methods"]["gff_ifcnn"]["psnr"]
    assert refined < entry["best_slice"], (
        f"IFCNN now scores {refined:.2f} dB against a {entry['best_slice']:.2f} dB "
        f"single frame; if this improved, update the report's warning")


def test_ifcnn_improves_boundaries_more_often_than_not(results):
    """It is a boundary-repair stage, so it should help there even as colour drifts."""
    _require("gff_ifcnn")
    better = 0
    total = 0
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "edge_psnr", key)
        if "gff_ifcnn" not in scores or "guided_filter" not in scores:
            continue
        total += 1
        if scores["gff_ifcnn"] > scores["guided_filter"]:
            better += 1
    assert better > total / 2, (
        f"IFCNN improved boundaries in only {better} of {total} scenarios")


def test_ifcnn_costs_overall_accuracy(results):
    """
    Worth stating plainly in the report: on synthetic stacks the refinement
    lowers overall accuracy versus the plain guided filter, because the stacks
    give it no genuinely missed detail to recover.
    """
    _require("gff_ifcnn")
    worse = 0
    total = 0
    for key, _, _, _ in sc.SCENARIOS:
        scores = _scores(results, "psnr", key)
        if "gff_ifcnn" not in scores or "guided_filter" not in scores:
            continue
        total += 1
        if scores["gff_ifcnn"] < scores["guided_filter"]:
            worse += 1
    assert worse >= total - 1, (
        f"IFCNN was worse than the plain guided filter in {worse} of {total} "
        f"scenarios; the report says it costs accuracy on synthetic stacks")


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
