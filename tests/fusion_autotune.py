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

Every run writes a folder under logs/ holding its log, a summary of every
candidate it scored, and what each of those candidates rendered - see Session.
A quick run stays inside that folder: it drops the contrast bracket and checks
one guard scenario rather than three, so it accepts settings a full run refuses,
and neither the ledger nor the source may be written from one.

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
  it stay in range does not. Because that measure rises with frame count and
  the search is subsampled, the bracket is checked again on the whole capture
  before anything is written - see CONFIRM_AT_FULL_DENSITY.
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
ROOT = os.path.dirname(HERE)
SOURCE = os.path.join(ROOT, "fusion_methods", "pyramid.py")
LEDGER = os.path.join(HERE, "fusion_autotune_ledger.json")
LOGS = os.path.join(ROOT, "logs")

# What a session writes for each candidate. Lossy WebP rather than PNG because
# a session is 16 to 30 candidates at 2.2 megapixels and these are for looking
# at, not for measuring - the numbers that decide anything are in summary.json.
# At quality 88 a candidate is a few hundred KB against three megabytes.
IMAGE_QUALITY = 88

# Difference images are amplified, because what separates two candidates is
# usually a couple of levels over a small part of the frame and is invisible
# at 1:1. The factor is chosen per image so that the largest difference present
# reaches full scale - a fixed one renders most pairs almost black, and the
# pairs it does suit are the ones that needed no help. It is capped so that two
# identical candidates do not come back as an amplified picture of nothing, and
# it goes in the filename so nobody reads an amplified difference as the real
# one.
DIFF_MAX_AMPLIFY = 64

CAPTURE = "electronics_ant"
BLOCK = 64

# Frames of the capture the search runs on. Denser is more faithful and the
# search is the one place here that pays for it per candidate rather than once,
# so this is deliberately the same subsample the test module uses.
STACK_STEP = 12
QUICK_STACK_STEP = 48

# The rendered scenarios the guard re-checks. Five rather than all seven:
# every candidate pays for them, and these are the ones the pyramid's own
# tuning notes cite as the settings' cost centres.
#
# deep_stack and veil joined the list after a sweep found settings that gained
# on all three of the others while quietly costing a decibel on veil. They are
# the two fixtures item 19 of docs/ALGORITHM_IMPROVEMENTS.md was written
# against - the reconstructed-pixel artefact and the bright-grain wash - so a
# guard that did not watch them was not watching the defects this method's
# departures from choose-max exist to prevent. veil is borrowed from the DCT
# suite rather than declared in fusion_scenarios.py, exactly as
# tests/visualize_pyramid_quality.py borrows it.
GUARD_SCENARIOS = ("fine_texture", "depth_edge", "long_stack", "deep_stack",
                   "veil")
QUICK_GUARD_SCENARIOS = ("depth_edge",)


def build_guard(name):
    """(stack, reference) for one guard scenario, wherever it is declared."""
    if name == "veil":
        from tests.test_dct_defocus_wash import _veil_stack
        stack, reference, _interior = _veil_stack()
        return stack, reference
    stack, reference, _ = sc.build(name)
    return stack, reference

# How much ground-truth reconstruction a candidate may give up, in dB, before
# it is rejected whatever it scores against Helicon. Small on purpose: the
# guard is a veto, not a term to be traded against.
GUARD_TOLERANCE_DB = 0.30

# Recovered contrast has to stay inside Helicon's own pyramid smoothing range.
LIGHTEST = "HF-C-1"
HEAVIEST = "HF-C-10"
MAX_SHARE_AGAINST_LIGHTEST = 1.15
MIN_SHARE_AGAINST_HEAVIEST = 0.85

# The bracket above is the one constraint that does not mean the same thing at
# the density the search runs at, and it is the reason for the confirmation
# pass below.
#
# Recovered contrast rises with the number of frames: on this capture the
# shipped settings measure 0.916 of HF-C-1 over 28 frames and 1.112 over all
# 333. So at STACK_STEP every candidate lands around 0.9-1.0 and the "crunchier
# than 1.15" half of the bracket cannot fire at all - it is not a loose
# constraint there, it is an absent one. A candidate can therefore sweep the
# search, satisfy every guard, and still be outside the bracket on the stack a
# user actually renders. That is not hypothetical: selectivity 16 did exactly
# that, winning on all five ground-truth scenarios and reaching 1.233 at full
# density (see SELECTIVITY in fusion_methods/pyramid.py).
#
# So the champion is re-measured on the whole capture before it may be written
# to source or recorded in the ledger. One fuse of 333 frames is about half a
# minute and 2.2 GB, which is affordable once per run and not once per
# candidate - which is also why the search itself stays subsampled.
CONFIRM_AT_FULL_DENSITY = True

# A candidate has to beat the incumbent by this much to displace it. Below it
# the two are the same setting wearing different numbers, and churning source
# for that would make every future diff harder to read.
MIN_IMPROVEMENT = 0.002


# ---------------------------------------------------------------------------
# The session: one folder per run, holding its log and what it rendered
# ---------------------------------------------------------------------------

class Session:
    """
    One tuning run's own directory under logs/.

        logs/autotune-20260729-201530/
          session.log      every line the run printed, timestamped
          summary.json     every candidate's settings and scores
          images/          what each candidate rendered, and how it differs

    A run is a search for a setting somebody will have to agree with, and the
    numbers alone do not settle that: two candidates a thousandth apart in
    fitness can look quite different, and which one is right is a judgement
    about the picture. So the pictures are kept, next to the log that says what
    was decided about them, in a directory that is not overwritten by the next
    run.

    Images are optional (`images="none"`) and off the hot path either way - a
    candidate is fused whether or not its result is written out.
    """

    def __init__(self, root=None, tag="autotune", images="all", stamp=None):
        if images not in ("all", "accepted", "none"):
            raise ValueError(f"images must be all, accepted or none, not {images!r}")
        self.images = images
        self.started = time.time()
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(root or LOGS, f"{tag}-{stamp}")
        # A second run inside the same second would otherwise share a folder.
        suffix = 1
        while os.path.exists(self.path):
            suffix += 1
            self.path = os.path.join(root or LOGS, f"{tag}-{stamp}-{suffix}")
        self.image_dir = os.path.join(self.path, "images")
        os.makedirs(self.image_dir if images != "none" else self.path)

        self.log_path = os.path.join(self.path, "session.log")
        self.summary_path = os.path.join(self.path, "summary.json")
        self._handle = open(self.log_path, "w", encoding="utf-8")
        self._saved = []
        self.log(f"session {os.path.basename(self.path)} started "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}")

    # -- logging ----------------------------------------------------------
    def log(self, *parts):
        """Print, and keep. Everything the run says lands in both places."""
        line = " ".join(str(part) for part in parts)
        print(line)
        self._handle.write(f"[{time.time() - self.started:7.1f}s] {line}\n")
        self._handle.flush()

    # -- images -----------------------------------------------------------
    @staticmethod
    def _safe(name):
        return re.sub(r"[^A-Za-z0-9_.=+-]", "_", name)

    def save_image(self, name, image, against=None, accepted=True):
        """
        Write one candidate's render, and its difference from the incumbent.

        `against` is the image to difference with - the incumbent, normally, so
        the pair answers "what did this setting actually change" rather than
        leaving it to be found by flipping between two near-identical frames.
        """
        if image is None or self.images == "none" or \
                (self.images == "accepted" and not accepted):
            return None
        import cv2

        stem = f"{len(self._saved):02d}-{self._safe(name)}"
        params = [cv2.IMWRITE_WEBP_QUALITY, IMAGE_QUALITY]
        path = os.path.join(self.image_dir, f"{stem}.webp")
        cv2.imwrite(path, image, params)
        written = [os.path.basename(path)]

        if against is not None and against.shape == image.shape:
            diff = cv2.absdiff(image, against)
            peak = int(diff.max())
            factor = 1 if peak == 0 else min(DIFF_MAX_AMPLIFY,
                                             max(1, round(255.0 / peak)))
            scaled = (diff.astype(np.int32) * factor).clip(0, 255).astype(np.uint8)
            diff_path = os.path.join(self.image_dir, f"{stem}.diff-x{factor}.webp")
            cv2.imwrite(diff_path, scaled, params)
            written.append(os.path.basename(diff_path))

        self._saved.append({"name": name, "files": written})
        return path

    # -- closing ----------------------------------------------------------
    def finish(self, champion=None, history=None, applied=False, extra=None):
        """Write summary.json and close the log. Safe to call more than once."""
        summary = {
            "session": os.path.basename(self.path),
            "seconds": round(time.time() - self.started, 1),
            "capture": CAPTURE,
            "applied_to_source": applied,
            "shipped_settings": default_settings(),
            "champion": champion,
            "history": history or [],
            "images": self._saved,
        }
        summary.update(extra or {})
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1, sort_keys=True, default=str)
        if not self._handle.closed:
            self.log(f"wrote {os.path.relpath(self.summary_path, ROOT)} "
                     f"and {len(self._saved)} image sets")
            self._handle.close()
        return self.path


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
    Tunable("ENERGY_WINDOW", "energy_window", [3, 5, 7, 9, 15],
            "window the band energy is pooled over"),
    Tunable("SELECTIVITY", "selectivity", [4.0, 8.0, 16.0, 32.0, 64.0],
            "how sharply the weights favour the sharpest frame"),
    Tunable("COHERENCE", "coherence", [0.0, 0.125, 0.25, 0.5],
            "how much of a band's decision comes from the coarser bands"),
    Tunable("BASE_SELECTIVITY", "base_selectivity", [1.0, 3.0, 5.0, 8.0],
            "how hard the coarse base follows the frames that won the detail"),
    Tunable("NOISE_PERCENTILE", "noise_percentile", [2.0, 5.0, 10.0, 20.0],
            "percentile of a band's energies read as its noise level"),
    Tunable("DEFAULT_LEVELS", "levels", [3, 4, 5, 6, 7],
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
            self.guard_scenarios[name] = build_guard(name)


def measure(fixture, settings, keep_image=False):
    """
    Score one candidate. Returns a dict; `fitness` is the number being searched.

    fitness      mean tile agreement over every third-party render. Higher is
                 better, and it is the only thing maximised.
    admissible   whether the constraints hold. An inadmissible candidate is
                 never chosen however high its fitness.

    `keep_image` attaches the fused result under "image" so a session can write
    it out. Left off by default because holding one 2.2 megapixel frame per
    candidate for the length of a search is megabytes for nothing when nobody
    is going to look at them.
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

    scored = {
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
    if keep_image:
        scored["image"] = fused
    return scored


def _without_image(scored):
    """A candidate record fit for JSON: everything but the frame itself."""
    return {key: value for key, value in scored.items() if key != "image"}


def confirm_at_full_density(settings, log=print):
    """
    Re-measure one candidate's bracket on every frame of the capture.

    Returns (ok, report). `ok` is False when the settings leave the bracket or
    the stack envelope at full density, whatever they scored during the search -
    see the note above CONFIRM_AT_FULL_DENSITY for why that can differ.

    Loads its own copy of the stack and drops it again: this runs once per
    session, and holding 2.2 GB for the length of a search to use it at the end
    would be the expensive way round.
    """
    frames, _meta = captures.load_capture(CAPTURE)
    try:
        fused = fuse(frames, settings)

        shares = {}
        for key in (LIGHTEST, HEAVIEST):
            render = captures.load_render(CAPTURE, key)
            if render is None:
                return False, f"{key} did not decode; cannot confirm the bracket"
            ours, theirs = captures.fit_render(fused, render)
            shares[key] = fm.detail_agreement(ours, theirs, BLOCK)[1]

        lo = np.min(np.stack(frames), axis=0).astype(np.int32)
        hi = np.max(np.stack(frames), axis=0).astype(np.int32)
        result = fused.astype(np.int32)
        excursion = (int(np.maximum(lo - result, 0).max()),
                     int(np.maximum(result - hi, 0).max()))
    finally:
        frames = None

    reasons = []
    if shares[LIGHTEST] >= MAX_SHARE_AGAINST_LIGHTEST:
        reasons.append(f"crunchier than {LIGHTEST} on the full stack "
                       f"({shares[LIGHTEST]:.3f} >= {MAX_SHARE_AGAINST_LIGHTEST})")
    if shares[HEAVIEST] <= MIN_SHARE_AGAINST_HEAVIEST:
        reasons.append(f"softer than {HEAVIEST} on the full stack "
                       f"({shares[HEAVIEST]:.3f} <= {MIN_SHARE_AGAINST_HEAVIEST})")
    if excursion != (0, 0):
        reasons.append(f"left the stack envelope by {excursion}")

    log(f"full density: share {LIGHTEST} {shares[LIGHTEST]:.3f}, "
        f"{HEAVIEST} {shares[HEAVIEST]:.3f}, envelope {excursion}")
    return not reasons, "; ".join(reasons)


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

def search(fixture, rounds=2, start=None, log=print, session=None):
    """
    Coordinate descent over the tunables: one constant at a time, repeated.

    Coordinate descent rather than anything cleverer because the point is a
    result someone will read and then edit source from. Each step is "this
    constant, at this value, moved the score by this much against everything
    else held still", which is a sentence that can be checked; the interactions
    a joint search would find could not be attributed to anything.

    A `session` collects the log and every candidate's render as it goes; pass
    one when the run is meant to be looked at afterwards.
    """
    if session is not None and log is print:
        log = session.log
    keep = session is not None and session.images != "none"

    incumbent = measure(fixture, start or default_settings(), keep_image=keep)
    incumbent["origin"] = "incumbent"
    history = [incumbent]
    log(f"incumbent fitness {incumbent['fitness']:.4f}"
        f"{'' if incumbent['admissible'] else '  INADMISSIBLE: ' + '; '.join(incumbent['reasons'])}")
    if session is not None:
        session.save_image("incumbent", incumbent.get("image"))

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
                candidate = measure(fixture, settings, keep_image=keep)
                candidate["origin"] = f"round {round_index}: {tunable.constant}={value}"
                history.append(candidate)

                ok, why = accepts(candidate, incumbent)
                log(f"  {tunable.constant}={value!r:>8}  "
                    f"fitness {candidate['fitness']:.4f}  "
                    f"{'ACCEPT' if ok else 'keep  '}  {why}")
                if session is not None:
                    # Differenced against the incumbent it was competing with,
                    # which is what "what did this change" means at that point.
                    session.save_image(f"{tunable.constant}={value}",
                                       candidate.get("image"),
                                       against=incumbent.get("image"),
                                       accepted=ok)
                if ok:
                    incumbent, improved = candidate, True
        if not improved:
            log(f"round {round_index}: nothing improved, stopping")
            break

    return incumbent, [_without_image(c) for c in history]


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
    parser.add_argument("--images", choices=("all", "accepted", "none"), default="all",
                        help="which candidates' renders to keep in the session folder")
    parser.add_argument("--logs", default=None,
                        help=f"where session folders go (default {os.path.relpath(LOGS, ROOT)}/)")
    args = parser.parse_args(argv)

    if not captures.is_available(CAPTURE):
        print(f"samples/{CAPTURE} is not present; nothing to tune against")
        return 1
    if args.apply and args.quick:
        print("--quick drops the contrast bracket, because it does not survive "
              "the reduced stack density, so a quick run is a check of the rig "
              "and not a choice of constants. Re-run without --quick to apply.")
        return 2

    session = Session(root=args.logs, images=args.images,
                      tag="autotune-quick" if args.quick else "autotune")
    try:
        fixture = Fixture(quick=args.quick)
        session.log(f"{len(fixture.stack)} frames, {len(fixture.renders)} renders, "
                    f"{len(fixture.guard_scenarios)} guard scenarios, "
                    f"images={args.images}")

        champion, history = search(fixture, rounds=args.rounds, session=session)
        champion = _without_image(champion)
        changed = {k: v for k, v in champion["settings"].items()
                   if v != default_settings()[k]}

        session.log(f"best fitness {champion['fitness']:.4f} after "
                    f"{len(history)} candidates")
        if not changed:
            session.log("the shipped constants are still the best of those tried")
        else:
            for constant, value in sorted(changed.items()):
                session.log(f"  {constant}: {default_settings()[constant]!r} "
                            f"-> {value!r}")

        # The search ran on every STACK_STEP-th frame, where the bracket cannot
        # bite. Nothing that changed a constant may reach source or the ledger
        # until it has been re-measured on the whole capture.
        confirmed, why = True, None
        if changed and not args.quick and CONFIRM_AT_FULL_DENSITY:
            session.log(f"confirming the champion on all "
                        f"{captures.capture_meta(CAPTURE)['frame_count']} frames...")
            confirmed, why = confirm_at_full_density(champion["settings"],
                                                     log=session.log)
            champion["full_density_confirmed"] = confirmed
            if not confirmed:
                champion["full_density_reasons"] = why
                session.log(f"REJECTED at full density: {why}")
                session.log("the search's own stack is subsampled, and recovered "
                            "contrast rises with frame count - see the note above "
                            "CONFIRM_AT_FULL_DENSITY. The shipped constants stand.")
                champion = measure(fixture, default_settings())
                champion = _without_image(champion)
                champion["origin"] = "incumbent (champion rejected at full density)"
                changed = {}

        applied = False
        if args.apply and changed:
            applied = apply_to_source(champion["settings"], log=session.log)
            session.log("re-run the test suites before committing: the guard here "
                        f"covers {len(fixture.guard_scenarios)} scenarios, the "
                        f"suites cover the rest")
        elif args.apply:
            session.log("nothing to write")

        if args.quick:
            # A quick run drops the bracket and checks one guard scenario
            # instead of three, so it accepts things a full run refuses - this
            # one took COHERENCE=0.25, which the full run rejects for costing
            # 2.5 dB on fine_texture, a scenario quick mode never looks at.
            # Letting that reach the shared ledger would put a champion on
            # record that the real objective does not agree with, so a quick
            # run stays inside its own session folder.
            session.log("quick run: ledger left alone, results are in this "
                        "session only")
        else:
            save_ledger(champion, history, applied=applied)
            session.log(f"ledger updated: {os.path.relpath(LEDGER, ROOT)}")
        session.finish(champion, history, applied=applied,
                       extra={"rounds": args.rounds, "quick": args.quick})
    except BaseException as exc:
        # A run that died half way is exactly the one whose log is worth having.
        session.log(f"session failed: {type(exc).__name__}: {exc}")
        session.finish()
        raise
    print(f"session: {os.path.relpath(session.path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
