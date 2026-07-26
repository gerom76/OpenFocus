"""
A quality ratchet for the fusion methods.

The existing suites answer "is this method broken" with fixed thresholds that
sit far below where the methods actually score, so a change can halve a
method's quality and still pass. This one answers the other question - "is this
method worse than it was last time" - by scoring every method on every scenario
and comparing against a recorded baseline.

Two directions, both enforced:

* A metric that gets worse by more than its tolerance fails. That is the point.
* A metric that gets *better* by a wide margin also fails, asking for the
  baseline to be re-recorded. Otherwise improvements silently raise the real
  quality above the recorded one, and the next regression is measured against a
  number nobody has looked at in months.

Re-record after a deliberate change, having looked at the diff:

    python -m tests.test_fusion_regression --update

Run with:  python -m pytest tests/test_fusion_regression.py -v
"""

import argparse
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_metrics as fm
from tests import fusion_registry as reg
from tests import fusion_scenarios as sc

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fusion_quality_baseline.json")

# The methods worth ratcheting: the six distinct algorithms. GPU twins duplicate
# their CPU counterparts and are covered by the parity test instead.
METHODS = ["guided_filter", "gfgfgf", "dct", "dtcwt", "pyramid", "depthmap_max"]

# Metric -> (direction, tolerance). Direction is +1 when higher is better.
# Tolerances are per-metric because they are not on the same scale, and are
# deliberately loose: they encode "this method got materially worse", not run to
# run jitter.
METRICS = {
    "psnr": (+1, 0.75),                    # dB
    "ssim": (+1, 0.02),
    "qabf": (+1, 0.03),
    "spatial_frequency": (+1, 1.0),
    "seam_excess": (-1, 0.25),             # 8-bit levels on the block lattice
    "defocus_seam_excess": (-1, 0.25),
    "defocus_seam_visible": (-1, 2.0),     # percent of the lattice showing
    "block_speckle": (-1, 0.15),           # percent of flat blocks standing out
}

# A metric beating its baseline by more than this many tolerances means the
# baseline is stale rather than the run being lucky.
STALE_FACTOR = 3.0


def _scenarios():
    """Everything the ratchet tracks: the report's six plus the deep-stack regime."""
    return sc.SCENARIOS + sc.EXTRA_SCENARIOS


def _measure_one(method_key, scenario_key):
    stack, reference, _ = sc.build(scenario_key)
    fused = reg.get(method_key).run(stack)
    aligned, sources, ref = fm.align_to_common_size(fused, stack, reference)
    scores = fm.evaluate(aligned, sources, ref)
    return {k: float(v) for k, v in scores.items() if k in METRICS}


def measure_all(verbose=False):
    """Every available method on every scenario. Missing methods are skipped."""
    results = {}
    for method_key in METHODS:
        method = reg.get(method_key)
        ok, why = method.available()
        if not ok:
            if verbose:
                print(f"  skipping {method_key}: {why}")
            continue
        results[method_key] = {}
        for scenario_key, _, _, _ in _scenarios():
            results[method_key][scenario_key] = _measure_one(method_key, scenario_key)
            if verbose:
                print(f"  {method_key:<16}{scenario_key}")
    return results


def load_baseline():
    if not os.path.exists(BASELINE):
        return None
    with open(BASELINE, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline():
    data = load_baseline()
    if data is None:
        pytest.skip(f"no baseline recorded; run "
                    f"`python -m tests.test_fusion_regression --update`")
    return data["scores"]


@pytest.mark.parametrize("method_key", METHODS)
def test_no_method_got_worse(method_key, baseline):
    """Every recorded metric must hold within its tolerance."""
    method = reg.get(method_key)
    ok, why = method.available()
    if not ok:
        pytest.skip(f"{method.label}: {why}")
    if method_key not in baseline:
        pytest.skip(f"{method_key} is not in the baseline; re-record it")

    failures = []
    for scenario_key, _, _, _ in _scenarios():
        want = baseline[method_key].get(scenario_key)
        if not want:
            continue
        got = _measure_one(method_key, scenario_key)
        for metric, (direction, tolerance) in METRICS.items():
            if metric not in want or metric not in got:
                continue
            drop = (want[metric] - got[metric]) * direction
            if drop > tolerance:
                failures.append(
                    f"{scenario_key}/{metric}: {want[metric]:.3f} -> "
                    f"{got[metric]:.3f} ({drop:+.3f}, tolerance {tolerance})")

    assert not failures, (
        f"{method.label} regressed against the recorded baseline:\n  "
        + "\n  ".join(failures)
        + "\nIf the change was deliberate, re-record with "
          "`python -m tests.test_fusion_regression --update`")


@pytest.mark.parametrize("method_key", METHODS)
def test_baseline_is_not_stale(method_key, baseline):
    """
    A method scoring far above its baseline means the baseline is out of date.

    Left alone, the ratchet would keep protecting a quality level the code left
    behind long ago, and the next real regression would slip under it.
    """
    method = reg.get(method_key)
    if not method.available()[0] or method_key not in baseline:
        pytest.skip("not measurable here")

    stale = []
    for scenario_key, _, _, _ in _scenarios():
        want = baseline[method_key].get(scenario_key)
        if not want:
            continue
        got = _measure_one(method_key, scenario_key)
        for metric, (direction, tolerance) in METRICS.items():
            if metric not in want or metric not in got:
                continue
            gain = (got[metric] - want[metric]) * direction
            if gain > tolerance * STALE_FACTOR:
                stale.append(f"{scenario_key}/{metric}: {want[metric]:.3f} -> "
                             f"{got[metric]:.3f} ({gain:+.3f})")

    assert not stale, (
        f"{method.label} now scores well above its baseline - re-record it so "
        f"the ratchet protects the quality the code actually has:\n  "
        + "\n  ".join(stale)
        + "\n  python -m tests.test_fusion_regression --update")


# ---------------------------------------------------------------------------
# Claims that must hold regardless of the baseline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_key", METHODS)
def test_no_method_leaves_its_lattice_showing(method_key):
    """
    Whatever a method does internally, the result must not advertise it.

    A block or tile lattice visible in the output is a defect no user asked
    for, and it is the one artefact a selection-based method produces that a
    PSNR against a synthetic reference barely notices.
    """
    method = reg.get(method_key)
    ok, why = method.available()
    if not ok:
        pytest.skip(f"{method.label}: {why}")

    stack, _, _ = sc.build("low_contrast")   # least detail to hide behind
    fused = method.run(stack)
    excess, visible = fm.defocus_seams(fused)
    assert excess < 1.5, (
        f"{method.label} leaves {excess:.2f} levels of step on the block "
        f"lattice in flat areas ({visible:.1f}% of it visible)")


# ---------------------------------------------------------------------------
# Re-recording
# ---------------------------------------------------------------------------

def _update(verbose=True):
    print("Measuring every method on every scenario...")
    scores = measure_all(verbose=verbose)
    previous = load_baseline()

    if previous:
        print("\nChanges against the recorded baseline:")
        for method_key, scenarios in scores.items():
            for scenario_key, metrics in scenarios.items():
                was = previous["scores"].get(method_key, {}).get(scenario_key, {})
                for metric, value in metrics.items():
                    if metric not in was:
                        continue
                    delta = value - was[metric]
                    if abs(delta) > METRICS[metric][1] / 2:
                        print(f"  {method_key:<16}{scenario_key:<18}"
                              f"{metric:<22}{was[metric]:9.3f} -> {value:9.3f}"
                              f" ({delta:+.3f})")

    payload = {
        "note": ("Recorded fusion quality. Regenerate with "
                 "`python -m tests.test_fusion_regression --update` after a "
                 "deliberate change, and review the printed diff before "
                 "committing."),
        "metrics": {k: {"higher_is_better": v[0] > 0, "tolerance": v[1]}
                    for k, v in METRICS.items()},
        "scores": scores,
    }
    with open(BASELINE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nWrote {BASELINE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="re-record the baseline from the current code")
    args = parser.parse_args()
    if args.update:
        _update()
    else:
        for method_key, scenarios in measure_all(verbose=False).items():
            print(f"\n{method_key}")
            for scenario_key, metrics in scenarios.items():
                line = "  ".join(f"{k} {v:.3f}" for k, v in sorted(metrics.items()))
                print(f"  {scenario_key:<18}{line}")
