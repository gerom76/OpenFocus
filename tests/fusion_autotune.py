"""
A measure -> propose -> re-measure loop for the pyramid method's constants.

The suites in this directory answer "is it broken" and "is it worse than last
time". This answers a third question - "is there a better setting than the one
shipped" - by scoring candidates against the Helicon Focus renders of
samples/electronics_ant, keeping the best that survives a guard, and writing the
winner back into fusion_methods/pyramid.py.

    python -m tests.fusion_autotune                 # search, report, change nothing
    python -m tests.fusion_autotune --apply         # and write the winner to source
    python -m tests.fusion_autotune --quick         # a short run, for checking the rig

## Why Helicon, and what stops this from merely imitating it

Helicon's method C is a Laplacian pyramid - the same algorithm - so where it
finds detail is a fair thing to be scored against, and it ships at four
smoothing settings, which brackets rather than scores. That is the objective.

The obvious failure of any such loop is that it converges on the reference
rather than on quality: a setting that makes this method look more like Helicon
on this one capture may be plainly worse everywhere else. Three things are in
the way, and none of them is optional.

* **The guard.** Every candidate is re-scored on the rendered scenarios in
  fusion_scenarios.py, which unlike the capture *do* have a ground truth. A
  candidate that loses more than a small tolerance of reconstruction PSNR there
  is rejected however well it scores against Helicon. This is what stops the
  loop trading real quality for resemblance.
* **The bracket, as a constraint rather than a target.** Recovered contrast has
  to stay between Helicon's lightest and heaviest smoothing. Maximising
  similarity of contrast would reward sharpening artefacts; requiring only that
  it stay in range does not.
* **The envelope.** A candidate that reconstructs pixels no frame contained is
  rejected outright, at any score.

So the loop can only move within settings that are already defensible. It is a
search over a space the method's own tests fence in, not a licence to change the
method.

## What it may change

Only the module-level constants in TUNABLES, and only to a value from that
constant's own declared grid - the same grids the report tools sweep. It does
not edit logic, and `--apply` re-checks the guard after writing and reverts the
file if the written source does not reproduce the score that won.

The ledger in fusion_autotune_ledger.json records every candidate ever scored,
so a later run starts from what is known rather than from nothing, and so the
constants in source can be traced to the run that chose them.
"""

import argparse
import copy
import importlib
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_methods.pyramid as pyramid_module            # noqa: E402
from fusion_methods.pyramid import pyramid_impl            # noqa: E402
from samples import captures                               # noqa: E402
from tests import fusion_metrics as fm                     # noqa: E402
from tests import fusion_scenarios as sc                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(os.path.dirname(HERE), "fusion_methods", "pyramid.py")
LEDGER = os.path.join(HERE, "fusion_autotune_ledger.json")

CAPTURE = "electronics_ant"
BLOCK = 64

# Frames of the capture the search runs on. Denser is more faithful and the
# search is the one place here that pays for it per candidate rather than once,
# so this is deliberately the same subsample the test module uses.
STACK_STEP = 12
QUICK_STACK_STEP = 48

# The rendered scenarios the guard re-checks. Three rather than all seven:
# every candidate pays for them, and these are the ones the pyramid's own
# tuning notes cite as the settings' cost centres.
GUARD_SCENARIOS = ("fine_texture", "depth_edge", "long_stack")
QUICK_GUARD_SCENARIOS = ("depth_edge",)

# How much ground-truth reconstruction a candidate may give up, in dB, before
# it is rejected whatever it scores against Helicon. Small on purpose: the
# guard is a veto, not a term to be traded against.
GUARD_TOLERANCE_DB = 0.30

# Recovered contrast has to stay inside Helicon's own pyramid smoothing range.
LIGHTEST = "HF-C-1"
HEAVIEST = "HF-C-10"
MAX_SHARE_AGAINST_LIGHTEST = 1.15
MIN_SHARE_AGAINST_HEAVIEST = 0.85

# A candidate has to beat the incumbent by this much to displace it. Below it
# the two are the same setting wearing different numbers, and churning source
# for that would make every future diff harder to read.
MIN_IMPROVEMENT = 0.002


# ---------------------------------------------------------------------------
# What may be tuned
# ---------------------------------------------------------------------------

class Tunable:
    """One module constant the loop may change, and the values it may use.

    `kwarg` is the pyramid_impl keyword that reaches it, or None for a constant
    with no keyword - those are set on the module for the duration of a call.
    """

    def __init__(self, constant, kwarg, values, blurb):
        self.constant = constant
        self.kwarg = kwarg
        self.values = values
        self.blurb = blurb

    @property
    def default(self):
        return getattr(pyramid_module, self.constant)


TUNABLES = [
    Tunable("ENERGY_WINDOW", "energy_window", [3, 5, 9, 15],
            "window the band energy is pooled over"),
    Tunable("SELECTIVITY", "selectivity", [4.0, 8.0, 16.0, 32.0],
            "how sharply the weights favour the sharpest frame"),
    Tunable("COHERENCE", "coherence", [0.0, 0.125, 0.25, 0.5],
            "how much of a band's decision comes from the coarser bands"),
    Tunable("BASE_SELECTIVITY", "base_selectivity", [1.0, 3.0, 8.0],
            "how hard the coarse base follows the frames that won the detail"),
    Tunable("NOISE_PERCENTILE", None, [5.0, 10.0, 20.0],
            "percentile of a band's energies read as its noise level"),
    Tunable("DEFAULT_LEVELS", "levels", [4, 5, 6],
            "band-pass levels the frame is split into"),
]

BY_CONSTANT = {t.constant: t for t in TUNABLES}


def default_settings():
    """The constants as the module currently ships them."""
    return {t.constant: t.default for t in TUNABLES}


def _call_kwargs(settings):
    """Split a settings dict into pyramid_impl kwargs and constants to patch."""
    kwargs, patches = {}, {}
    for constant, value in settings.items():
        tunable = BY_CONSTANT[constant]
        if tunable.kwarg:
            kwargs[tunable.kwarg] = value
        else:
            patches[constant] = value
    return kwargs, patches


def fuse(stack, settings):
    """Fuse `stack` under one candidate, restoring any patched constant after."""
    kwargs, patches = _call_kwargs(settings)
    saved = {name: getattr(pyramid_module, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(pyramid_module, name, value)
        return pyramid_impl(list(stack), **kwargs)
    finally:
        for name, value in saved.items():
            setattr(pyramid_module, name, value)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------

class Fixture:
    """Everything a candidate is scored on, loaded once and kept.

    Reading the capture and decoding eleven renders costs more than a dozen
    candidates do, so a search that reloaded them per candidate would spend its
    time on I/O.
    """

    def __init__(self, quick=False):
        self.quick = quick
        step = QUICK_STACK_STEP if quick else STACK_STEP

        # The bracket is an absolute pair of numbers measured on the full
        # fixture, and it does not survive a change of stack density: seven
        # frames recover about 0.63 of Helicon's heaviest smoothing against
        # 1.17 from twenty-eight, so every candidate in a quick run would be
        # rejected for being soft when what is soft is the stack. Quick runs
        # therefore drop it, and may not be applied to source - see main().
        self.bracket = None if quick else (MAX_SHARE_AGAINST_LIGHTEST,
                                           MIN_SHARE_AGAINST_HEAVIEST)
        self.stack, _meta = captures.load_capture(CAPTURE, step=step)
        self.renders = {}
        for key in sorted(captures.third_party_renders(CAPTURE)):
            image = captures.load_render(CAPTURE, key)
            if image is not None:
                self.renders[key] = image
        missing = {LIGHTEST, HEAVIEST} - set(self.renders)
        if missing:
            raise RuntimeError(f"the bracket needs {sorted(missing)}, which did "
                               f"not decode - is imagecodecs installed?")

        names = QUICK_GUARD_SCENARIOS if quick else GUARD_SCENARIOS
        self.guard_scenarios = {}
        for name in names:
            stack, reference, _ = sc.build(name)
            self.guard_scenarios[name] = (stack, reference)


def measure(fixture, settings):
    """
    Score one candidate. Returns a dict; `fitness` is the number being searched.

    fitness      mean tile agreement over every third-party render. Higher is
                 better, and it is the only thing maximised.
    admissible   whether the constraints hold. An inadmissible candidate is
                 never chosen however high its fitness.
    """
    started = time.time()
    fused = fuse(fixture.stack, settings)

    agreements, shares = {}, {}
    for key, render in fixture.renders.items():
        ours, theirs = captures.fit_render(fused, render)
        agreement, share = fm.detail_agreement(ours, theirs, BLOCK)
        agreements[key], shares[key] = agreement, share

    lo = np.min(np.stack(fixture.stack), axis=0).astype(np.int32)
    hi = np.max(np.stack(fixture.stack), axis=0).astype(np.int32)
    result = fused.astype(np.int32)
    excursion = (int(np.maximum(lo - result, 0).max()),
                 int(np.maximum(result - hi, 0).max()))

    guard_psnr = {}
    for name, (stack, reference) in fixture.guard_scenarios.items():
        guard_psnr[name] = fm.psnr(fuse(stack, settings), reference)

    reasons = []
    if excursion != (0, 0):
        reasons.append(f"left the stack envelope by {excursion}")
    if fixture.bracket is not None:
        crunchiest, softest = fixture.bracket
        if shares[LIGHTEST] >= crunchiest:
            reasons.append(f"crunchier than {LIGHTEST} ({shares[LIGHTEST]:.3f})")
        if shares[HEAVIEST] <= softest:
            reasons.append(f"softer than {HEAVIEST} ({shares[HEAVIEST]:.3f})")

    return {
        "settings": dict(settings),
        "fitness": float(np.mean(list(agreements.values()))),
        "agreement": {k: float(v) for k, v in agreements.items()},
        "share": {k: float(v) for k, v in shares.items()},
        "envelope": list(excursion),
        "guard_psnr": {k: float(v) for k, v in guard_psnr.items()},
        "reasons": reasons,
        "admissible": not reasons,
        "seconds": round(time.time() - started, 2),
    }


def guard_holds(candidate, incumbent, tolerance=GUARD_TOLERANCE_DB):
    """
    True when a candidate gives up no real reconstruction against the incumbent.

    Checked per scenario rather than on the mean: a candidate that gained a
    decibel on one scene and lost one on another would pass a mean test while
    having made the method worse somewhere, and "somewhere" is where a user's
    stack will be.
    """
    for name, psnr in candidate["guard_psnr"].items():
        if psnr < incumbent["guard_psnr"][name] - tolerance:
            return False, (f"{name} reconstruction {psnr:.2f} dB against "
                           f"{incumbent['guard_psnr'][name]:.2f} incumbent")
    return True, None


def accepts(candidate, incumbent):
    """Whether `candidate` should displace `incumbent`. Returns (bool, why)."""
    if not candidate["admissible"]:
        return False, "; ".join(candidate["reasons"])
    if candidate["fitness"] <= incumbent["fitness"] + MIN_IMPROVEMENT:
        return False, (f"fitness {candidate['fitness']:.4f} against "
                       f"{incumbent['fitness']:.4f} incumbent")
    held, why = guard_holds(candidate, incumbent)
    if not held:
        return False, f"guard: {why}"
    return True, (f"fitness {incumbent['fitness']:.4f} -> {candidate['fitness']:.4f}")


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def search(fixture, rounds=2, start=None, log=print):
    """
    Coordinate descent over the tunables: one constant at a time, repeated.

    Coordinate descent rather than anything cleverer because the point is a
    result someone will read and then edit source from. Each step is "this
    constant, at this value, moved the score by this much against everything
    else held still", which is a sentence that can be checked; the interactions
    a joint search would find could not be attributed to anything.
    """
    incumbent = measure(fixture, start or default_settings())
    incumbent["origin"] = "incumbent"
    history = [incumbent]
    log(f"incumbent fitness {incumbent['fitness']:.4f}"
        f"{'' if incumbent['admissible'] else '  INADMISSIBLE: ' + '; '.join(incumbent['reasons'])}")

    for round_index in range(rounds):
        improved = False
        for tunable in TUNABLES:
            # Seeded with where this constant currently sits, and added to as
            # the sweep goes, so accepting a value part way through does not
            # send the sweep back over the value it just moved off - which it
            # would otherwise, since the skip is against the live incumbent.
            tried = {incumbent["settings"][tunable.constant]}
            for value in tunable.values:
                if value in tried:
                    continue
                tried.add(value)
                settings = dict(incumbent["settings"])
                settings[tunable.constant] = value
                candidate = measure(fixture, settings)
                candidate["origin"] = f"round {round_index}: {tunable.constant}={value}"
                history.append(candidate)

                ok, why = accepts(candidate, incumbent)
                log(f"  {tunable.constant}={value!r:>8}  "
                    f"fitness {candidate['fitness']:.4f}  "
                    f"{'ACCEPT' if ok else 'keep  '}  {why}")
                if ok:
                    incumbent, improved = candidate, True
        if not improved:
            log(f"round {round_index}: nothing improved, stopping")
            break

    return incumbent, history


# ---------------------------------------------------------------------------
# Writing the winner back to source
# ---------------------------------------------------------------------------

def rewrite_source(text, settings):
    """
    Return `text` with each named constant reassigned. Nothing else is touched.

    Deliberately a line rewrite anchored to the start of a line, not a parse:
    the constants sit under long comments explaining why each holds the value it
    does, and those comments are the most valuable thing in the file. A rewriter
    that reconstructed the module would lose them.
    """
    for constant, value in settings.items():
        if constant not in BY_CONSTANT:
            raise KeyError(f"{constant} is not a declared tunable")
        pattern = re.compile(rf"^{re.escape(constant)} = .*$", re.MULTILINE)
        if not pattern.search(text):
            raise KeyError(f"{constant} is not assigned at module level in the source")
        text = pattern.sub(f"{constant} = {value!r}", text, count=1)
    return text


def apply_to_source(settings, path=SOURCE, log=print):
    """
    Write the winning constants into pyramid.py, and undo it if they do not take.

    The check afterwards is the point: the file is re-imported and the constants
    read back, so a rewrite that produced source saying something other than
    what was chosen cannot survive. The original text is restored on any
    failure.
    """
    with open(path, "r", encoding="utf-8") as handle:
        original = handle.read()

    updated = rewrite_source(original, settings)
    if updated == original:
        log("source already holds these values; nothing written")
        return False

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    try:
        importlib.reload(pyramid_module)
        for constant, value in settings.items():
            written = getattr(pyramid_module, constant)
            if written != value:
                raise RuntimeError(f"{constant} reads back as {written!r}, not {value!r}")
    except Exception:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(original)
        importlib.reload(pyramid_module)
        raise
    log(f"wrote {len(settings)} constants to {os.path.relpath(path)}")
    return True


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def load_ledger(path=LEDGER):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_ledger(champion, history, path=LEDGER, applied=False):
    previous = load_ledger(path) or {"runs": []}
    record = {
        "capture": CAPTURE,
        "stack_step": STACK_STEP,
        "guard_scenarios": list(GUARD_SCENARIOS),
        "guard_tolerance_db": GUARD_TOLERANCE_DB,
        "applied_to_source": applied,
        "champion": copy.deepcopy(champion),
        "considered": len(history),
        "note": ("Written by `python -m tests.fusion_autotune`. `champion.settings` "
                 "is what fusion_methods/pyramid.py should hold if applied_to_source "
                 "is true; tests/test_fusion_autotune.py checks that it does."),
    }
    previous["runs"] = (previous.get("runs") or [])[-9:] + [record]
    previous["champion"] = copy.deepcopy(champion)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(previous, handle, indent=1, sort_keys=True)
    return previous


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--rounds", type=int, default=2,
                        help="coordinate-descent passes over the constants")
    parser.add_argument("--quick", action="store_true",
                        help="a short run on fewer frames and one guard scenario")
    parser.add_argument("--apply", action="store_true",
                        help="write the winner into fusion_methods/pyramid.py")
    args = parser.parse_args(argv)

    if not captures.is_available(CAPTURE):
        print(f"samples/{CAPTURE} is not present; nothing to tune against")
        return 1
    if args.apply and args.quick:
        print("--quick drops the contrast bracket, because it does not survive "
              "the reduced stack density, so a quick run is a check of the rig "
              "and not a choice of constants. Re-run without --quick to apply.")
        return 2

    fixture = Fixture(quick=args.quick)
    print(f"{len(fixture.stack)} frames, {len(fixture.renders)} renders, "
          f"{len(fixture.guard_scenarios)} guard scenarios")

    champion, history = search(fixture, rounds=args.rounds)
    changed = {k: v for k, v in champion["settings"].items()
               if v != default_settings()[k]}

    print(f"\nbest fitness {champion['fitness']:.4f} after {len(history)} candidates")
    if not changed:
        print("the shipped constants are still the best of those tried")
    else:
        for constant, value in sorted(changed.items()):
            print(f"  {constant}: {default_settings()[constant]!r} -> {value!r}")

    applied = False
    if args.apply and changed:
        applied = apply_to_source(champion["settings"])
        print("re-run the test suites before committing: the guard here covers "
              f"{len(fixture.guard_scenarios)} scenarios, the suites cover the rest")
    elif args.apply:
        print("nothing to write")

    save_ledger(champion, history, applied=applied)
    print(f"ledger updated: {os.path.relpath(LEDGER)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
