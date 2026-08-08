"""
Tests for the self-tuning loop in fusion_autotune.py.

The loop measures the pyramid against the Helicon Focus renders of
samples/electronics_ant, proposes a change to the method's constants, and writes
it into fusion_methods/pyramid.py. That is a mechanism which edits the code it
is judging, so what needs testing is less "does it find good settings" - which
is a question about the search space, not about the code - than "can it do harm,
and does it stop itself".

Four things are checked, roughly in order of what it would cost to get wrong:

* The guard rejects candidates that damage reconstruction, so the loop cannot
  trade real quality for resemblance to the reference it is scored against.
  This is the one that matters: without it the loop optimises towards Helicon
  and away from being right.
* The source rewriter changes the constants it was given and nothing else, and
  what it writes reads back as what was chosen.
* The declared tunables are real: each names a constant the module has and a
  keyword the method accepts.
* The ledger round-trips, and the constants in source match the champion it
  records - so the settings pyramid.py ships can be traced to the run that
  chose them.

The slow parts run on the quick fixture, which drops the contrast bracket
because it does not survive the reduced stack density. Nothing here applies
anything to the real source file; the rewriter is exercised on a copy.

Run with:  python -m pytest tests/test_fusion_autotune.py -v
"""

import json
import os
import re
import shutil
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_methods.pyramid as pyramid_module
from samples import captures
from tests import fusion_autotune as auto

if not captures.is_available(auto.CAPTURE):
    pytest.skip(f"samples/{auto.CAPTURE} is not present, so there is nothing "
                f"for the loop to tune against", allow_module_level=True)


@pytest.fixture(scope="module")
def fixture():
    """The quick fixture: seven frames, one guard scenario."""
    return auto.Fixture(quick=True)


@pytest.fixture(scope="module")
def incumbent(fixture):
    """The shipped constants, scored."""
    return auto.measure(fixture, auto.default_settings())


# ---------------------------------------------------------------------------
# The tunables are real
# ---------------------------------------------------------------------------

def test_every_tunable_names_a_constant_the_module_has():
    """
    A tunable naming a constant that no longer exists would be silently
    unreachable - the search would sweep it, nothing would change, and the
    result would read as "this constant does not matter".
    """
    for tunable in auto.TUNABLES:
        assert hasattr(pyramid_module, tunable.constant), \
            f"pyramid.py has no {tunable.constant}"
        assert tunable.values, f"{tunable.constant} has no values to try"
        assert tunable.blurb, f"{tunable.constant} is undocumented"
        # The grid has to include where the method currently sits, or the search
        # starts by comparing the incumbent against a set of values none of
        # which is the incumbent, and the first round always looks like a change.
        assert tunable.default in tunable.values, (
            f"{tunable.constant} ships at {tunable.default!r}, which is not in "
            f"its own grid {tunable.values}")


def test_every_tunable_actually_reaches_the_method(fixture):
    """
    Each declared constant has to change the picture when it is changed.

    Two ways it could fail to: a keyword that the method no longer accepts, and
    a constant read once at import rather than per call. The second is the
    quiet one - the search would report the value as inert and move on.
    """
    stack = fixture.stack[:4]
    base = auto.fuse(stack, auto.default_settings())

    for tunable in auto.TUNABLES:
        other = next((v for v in tunable.values if v != tunable.default), None)
        assert other is not None, f"{tunable.constant} has only its own default"
        settings = dict(auto.default_settings())
        settings[tunable.constant] = other
        moved = np.abs(auto.fuse(stack, settings).astype(np.int32)
                       - base.astype(np.int32)).mean()
        assert moved > 0.001, (
            f"{tunable.constant}={other} changed nothing ({moved:.5f} levels) - "
            f"it is not reaching the method")


def test_patched_constants_are_put_back(fixture):
    """
    The constants with no keyword are set on the module for the duration of one
    call. If a failure left one set, every later measurement in the run would
    silently use it - including the incumbent's, which is what everything else
    is compared against.
    """
    before = {t.constant: getattr(pyramid_module, t.constant) for t in auto.TUNABLES}
    settings = dict(auto.default_settings())
    settings["NOISE_PERCENTILE"] = 20.0
    auto.fuse(fixture.stack[:3], settings)
    after = {t.constant: getattr(pyramid_module, t.constant) for t in auto.TUNABLES}
    assert after == before

    # And after a failure inside the call, which is the case that matters.
    with pytest.raises(Exception):
        auto.fuse([], settings)
    assert {t.constant: getattr(pyramid_module, t.constant)
            for t in auto.TUNABLES} == before


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def test_the_incumbent_scores_finite_and_within_the_envelope(incumbent):
    assert np.isfinite(incumbent["fitness"])
    assert 0.0 < incumbent["fitness"] < 1.0
    assert incumbent["envelope"] == [0, 0]
    assert incumbent["guard_psnr"], "the guard measured nothing"
    assert all(np.isfinite(v) for v in incumbent["guard_psnr"].values())


def test_a_setting_that_stops_selecting_scores_worse(fixture, incumbent):
    """
    Sanity on the objective itself: it has to rank a method that has stopped
    fusing below one that has not. An objective that did not is not measuring
    quality, and every acceptance built on it would be noise.
    """
    settings = dict(auto.default_settings())
    settings["SELECTIVITY"] = 0.0
    settings["BASE_SELECTIVITY"] = 0.0
    flattened = auto.measure(fixture, settings)

    assert flattened["fitness"] < incumbent["fitness"], (
        f"averaging the stack scored {flattened['fitness']:.4f} against "
        f"{incumbent['fitness']:.4f} for the shipped settings")


# ---------------------------------------------------------------------------
# The guard - the part that stops the loop doing harm
# ---------------------------------------------------------------------------

def test_the_guard_rejects_a_candidate_that_damages_reconstruction(incumbent):
    """
    The loop is scored against Helicon, so left alone it would drift towards
    Helicon whatever that cost elsewhere. The guard is the whole defence: a
    candidate that loses ground-truth PSNR on a rendered scenario is refused
    however well it scored.

    Synthesised rather than searched for, because the point is that the rule
    fires, not that some particular setting trips it.
    """
    damaged = dict(incumbent)
    damaged["guard_psnr"] = {name: value - auto.GUARD_TOLERANCE_DB - 1.0
                             for name, value in incumbent["guard_psnr"].items()}
    damaged["fitness"] = incumbent["fitness"] + 10.0     # far better on the objective

    held, why = auto.guard_holds(damaged, incumbent)
    assert not held and why

    accepted, reason = auto.accepts(damaged, incumbent)
    assert not accepted, "a candidate that damaged reconstruction was accepted"
    assert "guard" in reason


def test_the_guard_checks_every_scenario_not_their_mean(incumbent):
    """
    A candidate that gained on one scene and lost on another would pass a test
    on the mean while having made the method worse somewhere - and somewhere is
    where a user's stack will be. Only meaningful with more than one scenario,
    so it is built rather than measured.
    """
    names = ["a", "b"]
    base = dict(incumbent, guard_psnr={"a": 40.0, "b": 40.0})
    mixed = dict(incumbent, guard_psnr={"a": 44.0, "b": 38.0},
                 fitness=incumbent["fitness"] + 1.0)
    assert sum(mixed["guard_psnr"][n] for n in names) > \
        sum(base["guard_psnr"][n] for n in names), "the mean has to look better"

    held, _ = auto.guard_holds(mixed, base)
    assert not held, "a loss on one scenario was hidden by a gain on another"


def test_an_inadmissible_candidate_is_never_accepted(incumbent):
    """Leaving the stack envelope disqualifies a candidate at any fitness."""
    outside = dict(incumbent, admissible=False, reasons=["left the stack envelope"],
                   fitness=incumbent["fitness"] + 10.0)
    accepted, reason = auto.accepts(outside, incumbent)
    assert not accepted and "envelope" in reason


def test_a_marginal_gain_does_not_churn_the_source(incumbent):
    """
    Below the improvement floor the two settings are the same one wearing
    different numbers, and rewriting source for that makes every later diff
    harder to read for nothing.
    """
    marginal = dict(incumbent, fitness=incumbent["fitness"] + auto.MIN_IMPROVEMENT / 2)
    accepted, _ = auto.accepts(marginal, incumbent)
    assert not accepted

    real = dict(incumbent, fitness=incumbent["fitness"] + auto.MIN_IMPROVEMENT * 2)
    assert auto.accepts(real, incumbent)[0]


# ---------------------------------------------------------------------------
# Writing to source
# ---------------------------------------------------------------------------

def test_the_rewriter_changes_the_named_constants_and_nothing_else():
    # Both values have to differ from what the module currently ships, or the
    # rewrite is a no-op and the line count below proves nothing. 4.0 is a
    # declared rung of the selectivity grid that the default is not.
    with open(auto.SOURCE, "r", encoding="utf-8") as handle:
        original = handle.read()

    updated = auto.rewrite_source(original, {"SELECTIVITY": 4.0, "COHERENCE": 0.25})

    assert "\nSELECTIVITY = 4.0\n" in updated
    assert "\nCOHERENCE = 0.25\n" in updated

    changed = [(a, b) for a, b in
               zip(original.splitlines(), updated.splitlines()) if a != b]
    assert len(changed) == 2, f"the rewrite touched {len(changed)} lines"
    assert len(original.splitlines()) == len(updated.splitlines())


def test_the_rewriter_keeps_the_comments_that_explain_the_constants():
    """
    Every constant in pyramid.py sits under a long note saying why it holds the
    value it does, and those notes are worth more than the numbers. A rewriter
    that regenerated the module would lose them, which is why this one edits
    lines.
    """
    with open(auto.SOURCE, "r", encoding="utf-8") as handle:
        original = handle.read()
    updated = auto.rewrite_source(original, {"SELECTIVITY": 4.0})

    assert updated.count("#") == original.count("#")
    assert "The published rule is choose-max" in updated


def test_the_rewriter_refuses_a_constant_it_does_not_declare():
    with open(auto.SOURCE, "r", encoding="utf-8") as handle:
        original = handle.read()
    with pytest.raises(KeyError):
        auto.rewrite_source(original, {"NOISE_FLOOR": 1e-9})


def test_applying_to_a_copy_reads_back_what_was_written(tmp_path, monkeypatch):
    """
    End to end on a copy of the module: write, re-import, read the constants
    back. The real file is never touched - `apply_to_source` reloads whatever
    module object it is pointed at, so the test points it at a throwaway.
    """
    copied = tmp_path / "pyramid_copy.py"
    shutil.copy(auto.SOURCE, copied)

    settings = {"SELECTIVITY": 4.0, "COHERENCE": 0.25}
    reloaded = {}

    def fake_reload(module):
        namespace = {}
        with open(copied, "r", encoding="utf-8") as handle:
            for line in handle:
                for name in settings:
                    if line.startswith(f"{name} = "):
                        namespace[name] = eval(line.split("=", 1)[1].strip())
        reloaded.update(namespace)
        for name, value in namespace.items():
            monkeypatch.setattr(pyramid_module, name, value, raising=False)
        return module

    monkeypatch.setattr(auto.importlib, "reload", fake_reload)
    assert auto.apply_to_source(settings, path=str(copied), log=lambda *a: None)
    assert reloaded == settings

    # Writing the same values again is a no-op rather than a second edit.
    assert not auto.apply_to_source(settings, path=str(copied), log=lambda *a: None)


def test_a_rewrite_that_does_not_take_is_reverted(tmp_path, monkeypatch):
    """
    The check after writing is the safety rail: if the file does not read back
    as what was chosen, the original has to come back. Simulated by making the
    reload report the wrong value, which is what a botched rewrite would look
    like from the outside.
    """
    copied = tmp_path / "pyramid_copy.py"
    shutil.copy(auto.SOURCE, copied)
    with open(copied, "r", encoding="utf-8") as handle:
        before = handle.read()

    monkeypatch.setattr(auto.importlib, "reload", lambda module: module)
    monkeypatch.setattr(pyramid_module, "SELECTIVITY", 999.0, raising=False)

    with pytest.raises(RuntimeError, match="reads back"):
        auto.apply_to_source({"SELECTIVITY": 4.0}, path=str(copied),
                             log=lambda *a: None)

    with open(copied, "r", encoding="utf-8") as handle:
        assert handle.read() == before, "the failed rewrite was left in place"


# ---------------------------------------------------------------------------
# The session folder
# ---------------------------------------------------------------------------

def test_a_session_makes_its_own_folder(tmp_path):
    """Each run gets a directory of its own, so no run overwrites another."""
    first = auto.Session(root=str(tmp_path), stamp="fixed")
    second = auto.Session(root=str(tmp_path), stamp="fixed")

    assert first.path != second.path, "two runs shared a folder"
    assert os.path.isdir(first.path) and os.path.isdir(second.path)
    for session in (first, second):
        assert os.path.isfile(session.log_path)
        session.finish()


def test_the_log_reaches_the_file_as_well_as_the_screen(tmp_path, capsys):
    session = auto.Session(root=str(tmp_path), stamp="log")
    session.log("candidate SELECTIVITY=16.0 rejected")
    session.finish()

    assert "candidate SELECTIVITY=16.0 rejected" in capsys.readouterr().out
    with open(session.log_path, encoding="utf-8") as handle:
        written = handle.read()
    assert "candidate SELECTIVITY=16.0 rejected" in written
    assert "session " in written, "the log does not say which session it is"


def test_a_session_keeps_the_render_and_a_difference(tmp_path):
    """
    The images are the point of the folder: two candidates a thousandth apart
    in fitness can look quite different, and which is right is a judgement about
    the picture rather than about the number.
    """
    session = auto.Session(root=str(tmp_path), stamp="img")
    rng = np.random.default_rng(3)
    base = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
    other = base.copy()
    other[10:20, 10:20] = 0

    session.save_image("incumbent", base)
    session.save_image("SELECTIVITY=16.0", other, against=base)
    session.finish()

    written = sorted(os.listdir(session.image_dir))
    assert written[0] == "00-incumbent.webp"
    assert written[2] == "01-SELECTIVITY=16.0.webp"
    # The amplification is chosen per image and named, so the file says how far
    # what it shows is from the difference that was actually there.
    assert re.fullmatch(r"01-SELECTIVITY=16\.0\.diff-x\d+\.webp", written[1]), written
    diff = cv2.imread(os.path.join(session.image_dir, written[1]))
    assert diff is not None and diff.max() > 0


def test_a_faint_difference_is_amplified_and_an_identical_one_is_not(tmp_path):
    """
    The factor exists so a two-level change is visible; a fixed one would render
    most candidate pairs almost black. Two identical candidates must not come
    back as an amplified picture of rounding.
    """
    session = auto.Session(root=str(tmp_path), stamp="amp")
    base = np.full((32, 32, 3), 120, np.uint8)

    faint = base.copy()
    faint[4:8, 4:8] = 122                       # two levels
    session.save_image("faint", faint, against=base)
    session.save_image("identical", base.copy(), against=base)
    session.finish()

    names = sorted(n for n in os.listdir(session.image_dir) if ".diff-" in n)
    factors = {n.split(".diff-x")[1].split(".webp")[0] for n in names}
    assert "1" in factors, "an identical pair was amplified"
    amplified = [int(f) for f in factors if f != "1"]
    assert amplified and max(amplified) > 8, \
        f"a two-level difference was not brought up ({factors})"
    assert max(amplified) <= auto.DIFF_MAX_AMPLIFY


def test_a_session_can_be_told_to_keep_no_images(tmp_path):
    session = auto.Session(root=str(tmp_path), stamp="noimg", images="none")
    assert session.save_image("anything", np.zeros((8, 8, 3), np.uint8)) is None
    session.finish()
    assert not os.path.isdir(session.image_dir)
    assert os.path.isfile(session.summary_path)


def test_a_session_keeps_only_the_accepted_candidates_when_asked(tmp_path):
    session = auto.Session(root=str(tmp_path), stamp="acc", images="accepted")
    frame = np.zeros((8, 8, 3), np.uint8)
    assert session.save_image("kept", frame, accepted=True)
    assert session.save_image("dropped", frame, accepted=False) is None
    session.finish()
    assert [n for n in os.listdir(session.image_dir)] == ["00-kept.webp"]


def test_a_name_that_is_not_a_filename_is_made_into_one(tmp_path):
    session = auto.Session(root=str(tmp_path), stamp="safe")
    session.save_image("a/b:c*?", np.zeros((8, 8, 3), np.uint8))
    session.finish()
    written = os.listdir(session.image_dir)
    assert written == ["00-a_b_c__.webp"], written


def test_the_summary_is_json_and_carries_no_pixels(tmp_path, incumbent):
    """
    summary.json has to stay readable, which means the fused frames attached to
    a candidate for the image writer must not end up in it.
    """
    session = auto.Session(root=str(tmp_path), stamp="sum", images="none")
    withimage = dict(incumbent, image=np.zeros((4, 4, 3), np.uint8))
    session.finish(champion=auto._without_image(withimage),
                   history=[auto._without_image(withimage)])

    with open(session.summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert "image" not in summary["champion"]
    assert "image" not in summary["history"][0]
    assert summary["champion"]["settings"] == incumbent["settings"]
    assert summary["shipped_settings"] == auto.default_settings()


def test_a_failed_run_still_leaves_its_log(tmp_path):
    """The run that died half way is the one whose log is worth having."""
    session = auto.Session(root=str(tmp_path), stamp="boom")
    session.log("about to fail")
    session.finish()

    assert os.path.isfile(session.summary_path)
    with open(session.log_path, encoding="utf-8") as handle:
        assert "about to fail" in handle.read()


# ---------------------------------------------------------------------------
# The loop, and the ledger it leaves behind
# ---------------------------------------------------------------------------

def test_a_search_writes_its_candidates_into_the_session(fixture, tmp_path,
                                                         monkeypatch):
    """The search and the session have to actually be wired together."""
    only = [t for t in auto.TUNABLES if t.constant == "SELECTIVITY"]
    monkeypatch.setattr(auto, "TUNABLES", only)
    session = auto.Session(root=str(tmp_path), stamp="wired")

    champion, history = auto.search(fixture, rounds=1, session=session)
    session.finish(auto._without_image(champion), history)

    # One render per candidate, plus a difference for all but the incumbent.
    names = os.listdir(session.image_dir)
    assert len([n for n in names if ".diff-" not in n]) == len(history)
    assert any(re.search(r"\.diff-x\d+\.", n) for n in names)
    with open(session.log_path, encoding="utf-8") as handle:
        assert "incumbent fitness" in handle.read()

def test_a_search_round_returns_an_admissible_champion(fixture, monkeypatch):
    """
    One pass over one constant, which is enough to exercise the loop: measure,
    propose, compare, keep. Restricted to a single tunable so it stays a test of
    the mechanism rather than a tuning run - the full grid is a few minutes.
    """
    only = [t for t in auto.TUNABLES if t.constant == "SELECTIVITY"]
    assert only, "SELECTIVITY is no longer a declared tunable"
    monkeypatch.setattr(auto, "TUNABLES", only)

    champion, history = auto.search(fixture, rounds=1, log=lambda *a: None)

    assert history and champion in history
    assert len(history) == len(only[0].values), \
        "the search did not try every value in the grid exactly once"
    assert champion["fitness"] >= history[0]["fitness"], \
        "the search returned something worse than the incumbent it started from"
    # The champion carries a full settings dict even when one constant was swept,
    # because that is what gets written to source.
    assert set(champion["settings"]) == {t.constant for t in auto.TUNABLES}


def test_the_ledger_round_trips(tmp_path, incumbent):
    path = str(tmp_path / "ledger.json")
    auto.save_ledger(incumbent, [incumbent], path=path, applied=False)
    loaded = auto.load_ledger(path)

    assert loaded["champion"]["settings"] == incumbent["settings"]
    assert loaded["runs"][-1]["applied_to_source"] is False
    assert loaded["runs"][-1]["capture"] == auto.CAPTURE
    # A second run appends rather than replacing, so the trail survives.
    auto.save_ledger(incumbent, [incumbent], path=path, applied=True)
    assert len(auto.load_ledger(path)["runs"]) == 2

    with open(path, "r", encoding="utf-8") as handle:
        json.load(handle)


def test_the_shipped_constants_match_the_recorded_champion():
    """
    The ratchet: pyramid.py has to hold what the last applied run chose.

    Without this the ledger is a diary rather than a record - source could drift
    from it and nothing would say which of the two was current. Only runs that
    were applied are binding; a search that changed nothing leaves source alone
    by design.
    """
    ledger = auto.load_ledger()
    if ledger is None:
        pytest.skip("no tuning run has been recorded yet")

    applied = [run for run in ledger.get("runs", []) if run.get("applied_to_source")]
    if not applied:
        # Nothing has been written to source, so the shipped constants are the
        # ones a human chose. That is a valid state and the check is what the
        # last run measured them at, not what it would have preferred.
        assert ledger["champion"]["settings"], "the ledger records no champion"
        return

    for constant, value in applied[-1]["champion"]["settings"].items():
        assert getattr(pyramid_module, constant) == value, (
            f"pyramid.py has {constant}={getattr(pyramid_module, constant)!r} but "
            f"the last applied tuning run chose {value!r} - source and ledger "
            f"disagree about which is current")
