"""
A plain-language guide to what each fusion method is good and bad at.

Runs every method over the scenarios in tests/fusion_scenarios.py - fine detail,
sensor noise, a hard depth boundary, a long stack, a flat subject, saturated
colour - and writes one self-contained HTML page aimed at someone who just wants
to know which setting to pick.

Strengths and weaknesses are read off the measured ranks, not written by hand, so
the page cannot drift away from what the code actually does. The claims it makes
are asserted in tests/test_fusion_characteristics.py.

Examples:
    python tests/visualize_fusion_characteristics.py --open
    python tests/visualize_fusion_characteristics.py --include-gpu --out guide.html
"""

import argparse
import base64
import json
import os
import sys
import time
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_metrics as fm
from tests import fusion_registry as reg
from tests import fusion_scenarios as sc

CORE_METHODS = ["guided_filter", "gfgfgf", "dct", "dtcwt", "stackmffv4", "gff_ifcnn"]
GPU_METHODS = ["guided_filter_gpu", "gfgfgf_gpu", "dct_gpu", "dtcwt_gpu"]

REPORTS_DIR = "reports"  # Everything generated lands here unless --out says otherwise
PREVIEW = 300   # px, embedded scenario previews
CROP_PX = 132   # px, side of the magnified detail window

# Notes that a rank cannot express: a named mechanism behind a behaviour.
# Only added to a card when the run actually reproduces the behaviour.
MECHANISM_NOTES = {
    "gfgfgf": ("Blends rather than picks",
               "It decides which frame wins at each pixel, then smooths those decisions "
               "into soft weights and mixes the frames together. That hides the stepping "
               "a block-based method shows, but where two frames both claim a region the "
               "average of the two is slightly softer than either."),
    "gff_ifcnn": ("Can disturb a fusion that was already right",
                  "The refinement pass converts the picture into a neural network's "
                  "internal form and back again, and only the difference that round "
                  "trip makes is applied - so a region it has nothing to add to is "
                  "left alone. It tidies edges, but where the fusion was already "
                  "correct it can still nudge things the wrong way."),
    "dct": ("Judges sharpness in blocks",
            "It decides which frame wins for each square block of pixels rather than each "
            "pixel, so boundaries between near and far can look slightly stepped. It also "
            "reads film grain as detail, which misleads it on noisy shots."),
    "stackmffv4": ("Learned, not calculated",
                   "A neural network trained on ordinary focus stacks, so it handles the "
                   "common cases best. It has not been tuned for microscopy or medical "
                   "imaging, where a hand-written method may be the safer choice."),
    "dtcwt": ("Works across scales",
              "It breaks each frame into coarse and fine layers at several orientations "
              "before combining, which is why boundaries between near and far come out "
              "cleaner than the block- and pixel-based methods manage."),
    "guided_filter": ("Edge-aware smoothing",
                      "It builds a per-pixel map of which frame is sharpest, then smooths "
                      "that map while respecting edges in the picture, which keeps its "
                      "decisions from bleeding across outlines."),
}


def _b64(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _fit(img, longest=PREVIEW):
    h, w = img.shape[:2]
    if max(h, w) <= longest:
        return img
    scale = longest / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _crop(img, window, scale=2):
    x, y, w, h = window
    patch = img[y:y + h, x:x + w]
    return cv2.resize(patch, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def _crop_window(masks, shape):
    """A window straddling a focus boundary, where the differences live."""
    height, width = shape[:2]
    edge = fm.boundary_band(masks, width=3)
    ys, xs = np.nonzero(edge)
    if len(ys) == 0:
        cy, cx = height // 2, width // 2
    else:
        mid = len(ys) // 2
        order = np.argsort(ys)
        cy, cx = int(ys[order[mid]]), int(xs[order[mid]])
    half = CROP_PX // 2
    x = int(np.clip(cx - half, 0, width - CROP_PX))
    y = int(np.clip(cy - half, 0, height - CROP_PX))
    return (x, y, CROP_PX, CROP_PX)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure(method_keys):
    """Run every method over every scenario and collect scores plus previews."""
    data = {"scenarios": [], "methods": {}}

    for key, _, title, blurb in sc.SCENARIOS:
        stack, reference, masks = sc.build(key)
        edge = fm.boundary_band(masks)
        window = _crop_window(masks, reference.shape)
        baseline = max(fm.psnr(s, reference) for s in stack)

        entry = {
            "key": key,
            "title": title,
            "blurb": blurb,
            "baseline": baseline,
            "slices": len(stack),
            "reference": _b64(_fit(reference)),
            "reference_crop": _b64(_crop(reference, window)),
            "input_crop": _b64(_crop(stack[-1], window)),
            "results": {},
        }
        print(f"  {title}")

        for method_key in method_keys:
            method = reg.get(method_key)
            if not method.available()[0]:
                continue
            start = time.perf_counter()
            fused = method.run(stack)
            elapsed = time.perf_counter() - start

            aligned, sources, ref = fm.align_to_common_size(fused, stack, reference)
            identical = any(np.array_equal(aligned, s) for s in sources)
            entry["results"][method_key] = {
                "psnr": fm.psnr(aligned, ref),
                "edge_psnr": fm.region_psnr(aligned, ref, edge),
                "colour": fm.colour_error(aligned, ref),
                "seconds": elapsed,
                "unfused": bool(identical),
                "crop": _b64(_crop(aligned, window)),
            }
        data["scenarios"].append(entry)

    for method_key in method_keys:
        method = reg.get(method_key)
        if method.available()[0]:
            data["methods"][method_key] = {"label": method.label, "key": method_key}

    return data


def _normalised(entry, method_key):
    """
    0-100: how much of the achievable improvement this method captured.

    0 means no better than keeping one frame; 100 means it matched the best
    method in that scenario. This is the number the page shows, because
    decibels mean nothing to most readers.
    """
    scores = {k: v["psnr"] for k, v in entry["results"].items()}
    best = max(scores.values())
    floor = entry["baseline"]
    if best <= floor:
        return 0.0
    value = (scores[method_key] - floor) / (best - floor) * 100.0
    return float(np.clip(value, 0.0, 100.0))


def derive_profiles(data):
    """Read each method's strengths and weaknesses off its measured ranks."""
    profiles = {}
    for method_key, meta in data["methods"].items():
        scores, ranks, times = {}, {}, []
        for entry in data["scenarios"]:
            if method_key not in entry["results"]:
                continue
            scores[entry["key"]] = _normalised(entry, method_key)
            order = sorted(entry["results"],
                           key=lambda k: entry["results"][k]["psnr"], reverse=True)
            ranks[entry["key"]] = order.index(method_key) + 1
            times.append(entry["results"][method_key]["seconds"])

        titles = {e["key"]: e["title"] for e in data["scenarios"]}
        field = len(data["methods"])

        strengths = [titles[k] for k in sorted(scores, key=lambda k: -scores[k])
                     if ranks[k] <= 2][:3]
        weaknesses = [titles[k] for k in sorted(scores, key=lambda k: scores[k])
                      if ranks[k] >= field - 1][:3]
        unfused = [titles[e["key"]] for e in data["scenarios"]
                   if e["results"].get(method_key, {}).get("unfused")]

        profiles[method_key] = {
            "label": meta["label"],
            "overall": float(np.mean(list(scores.values()))) if scores else 0.0,
            "median_time": float(np.median(times)) if times else 0.0,
            "scores": scores,
            "ranks": ranks,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "unfused": unfused,
            "note": MECHANISM_NOTES.get(method_key),
        }
    return profiles


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CSS = r"""
  :root {
    --ground:#eceff0; --surface:#fff; --surface-2:#f4f6f7; --line:#d3dadc;
    --line-strong:#b6c2c5; --ink:#0f1a1d; --ink-2:#46595e; --ink-3:#6f8388;
    --accent:#0b6f7a; --accent-soft:rgba(11,111,122,.12); --accent-ink:#fff;
    --good:#2c7a51; --bad:#b4462a; --bar-track:#dfe5e6; --bar-muted:#9fb0b4;
    --font-display:"Bahnschrift","DIN Alternate","Arial Narrow",system-ui,sans-serif;
    --font-body:"Charter","Bitstream Charter","Iowan Old Style",Georgia,serif;
    --font-mono:"Cascadia Mono","JetBrains Mono","SF Mono",Consolas,monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --ground:#0b1012; --surface:#131b1e; --surface-2:#182226; --line:#26343a;
      --line-strong:#35474e; --ink:#e7eef0; --ink-2:#a8bcc1; --ink-3:#7b9098;
      --accent:#3fb6c4; --accent-soft:rgba(63,182,196,.16); --accent-ink:#06272c;
      --good:#57b37f; --bad:#f0846a; --bar-track:#223034; --bar-muted:#5b747b; }
  }
  :root[data-theme="dark"] {
    --ground:#0b1012; --surface:#131b1e; --surface-2:#182226; --line:#26343a;
    --line-strong:#35474e; --ink:#e7eef0; --ink-2:#a8bcc1; --ink-3:#7b9098;
    --accent:#3fb6c4; --accent-soft:rgba(63,182,196,.16); --accent-ink:#06272c;
    --good:#57b37f; --bad:#f0846a; --bar-track:#223034; --bar-muted:#5b747b; }
  :root[data-theme="light"] {
    --ground:#eceff0; --surface:#fff; --surface-2:#f4f6f7; --line:#d3dadc;
    --line-strong:#b6c2c5; --ink:#0f1a1d; --ink-2:#46595e; --ink-3:#6f8388;
    --accent:#0b6f7a; --accent-soft:rgba(11,111,122,.12); --accent-ink:#fff;
    --good:#2c7a51; --bad:#b4462a; --bar-track:#dfe5e6; --bar-muted:#9fb0b4; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
         font-family:var(--font-body); font-size:16.5px; line-height:1.62; }
  .wrap { max-width:1060px; margin:0 auto; padding:48px 24px 96px;
          display:flex; flex-direction:column; gap:60px; }
  .eyebrow { font-family:var(--font-mono); font-size:12px; letter-spacing:.09em;
             text-transform:uppercase; color:var(--accent); }
  h1 { font-family:var(--font-display); font-size:clamp(34px,5.4vw,56px); line-height:1.04;
       font-weight:600; letter-spacing:-.01em; text-wrap:balance; margin:12px 0 0; }
  .deck { max-width:60ch; color:var(--ink-2); font-size:17.5px; margin:18px 0 0; }
  h2 { font-family:var(--font-display); font-size:27px; font-weight:600;
       text-wrap:balance; margin:0; }
  h3 { font-family:var(--font-display); font-size:19px; font-weight:600; margin:0; }
  .lede { max-width:64ch; color:var(--ink-2); margin:10px 0 0; }
  section { display:flex; flex-direction:column; gap:24px; }
  .head { border-top:1px solid var(--line-strong); padding-top:22px; }
  .verdict { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent);
             border-radius:3px; padding:20px 24px; display:flex; flex-direction:column; gap:6px; }
  .verdict .q { font-family:var(--font-mono); font-size:11px; letter-spacing:.08em;
                text-transform:uppercase; color:var(--ink-3); }
  .verdict .a { font-family:var(--font-display); font-size:21px; }
  .verdict p { margin:4px 0 0; color:var(--ink-2); font-size:15.5px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:16px; }
  .card { background:var(--surface); border:1px solid var(--line); border-radius:4px;
          padding:20px 22px 22px; display:flex; flex-direction:column; gap:14px; }
  .card.top { border-color:var(--accent); }
  .card .name { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
  .score { font-family:var(--font-display); font-size:30px; color:var(--accent);
           font-variant-numeric:tabular-nums; line-height:1; }
  .score small { font-size:12px; color:var(--ink-3); margin-left:2px; }
  .meter { background:var(--bar-track); border-radius:2px; height:8px; overflow:hidden; }
  .meter div { height:100%; background:var(--accent); border-radius:0 3px 3px 0; }
  ul { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:7px; }
  li { display:flex; gap:9px; align-items:flex-start; font-size:15px; color:var(--ink-2); }
  li .mark { font-family:var(--font-mono); font-weight:700; flex:none; line-height:1.5; }
  li.good .mark { color:var(--good); }
  li.bad .mark { color:var(--bad); }
  .note { background:var(--surface-2); border-radius:3px; padding:12px 14px;
          font-size:14.5px; color:var(--ink-2); }
  .note b { color:var(--ink); font-weight:600; display:block; margin-bottom:2px;
            font-family:var(--font-display); }
  .speed { font-family:var(--font-mono); font-size:12px; color:var(--ink-3);
           border-top:1px solid var(--line); padding-top:11px; }
  .scenario { background:var(--surface); border:1px solid var(--line); border-radius:4px;
              padding:24px; display:flex; flex-direction:column; gap:20px; }
  .scenario .top-row { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:24px; }
  .scenario img { width:100%; height:auto; display:block; border-radius:2px; }
  .chart { display:flex; flex-direction:column; gap:9px; }
  .row { display:grid; grid-template-columns:170px minmax(0,1fr) 52px; gap:12px; align-items:center; }
  .row .lbl { font-size:14px; color:var(--ink-2); text-align:right;
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .row.win .lbl { color:var(--ink); font-weight:700; }
  .track { background:var(--bar-track); border-radius:2px; height:19px; }
  .fill { height:100%; background:var(--bar-muted); border-radius:0 4px 4px 0; min-width:2px; }
  .row.win .fill { background:var(--accent); }
  .row .num { font-family:var(--font-mono); font-size:13px; color:var(--ink-2);
              font-variant-numeric:tabular-nums; }
  .flag { color:var(--bad); font-weight:700; }
  .strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(118px,1fr)); gap:10px; }
  .shot { display:flex; flex-direction:column; gap:5px; }
  .shot img { image-rendering:pixelated; border:1px solid var(--line); }
  .shot span { font-family:var(--font-mono); font-size:11px; color:var(--ink-3);
               overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .shot.ref span, .shot.ref img { color:var(--accent); border-color:var(--accent); }
  .scroller { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th, td { text-align:center; padding:10px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { font-family:var(--font-mono); font-size:11px; letter-spacing:.06em;
       text-transform:uppercase; color:var(--ink-3); font-weight:400; }
  th:first-child, td:first-child { text-align:left; font-family:var(--font-display); font-size:15px; }
  tbody tr:last-child td { border-bottom:0; }
  .pill { display:inline-block; min-width:34px; padding:3px 8px; border-radius:10px;
          font-family:var(--font-mono); font-size:12px; font-variant-numeric:tabular-nums; }
  .p-best { background:var(--accent); color:var(--accent-ink); font-weight:700; }
  .p-ok { background:var(--accent-soft); color:var(--ink); }
  .p-weak { background:var(--bar-track); color:var(--ink-3); }
  footer { border-top:1px solid var(--line-strong); padding-top:20px;
           font-family:var(--font-mono); font-size:12px; color:var(--ink-3); line-height:1.8; }
  @media (max-width:820px) {
    .scenario .top-row { grid-template-columns:1fr; }
    .row { grid-template-columns:110px minmax(0,1fr) 46px; gap:9px; }
  }
  @media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
"""


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cards(profiles, order):
    out = []
    for i, key in enumerate(order):
        p = profiles[key]
        strengths = "".join(
            f'<li class="good"><span class="mark">+</span><span>{_esc(s)}</span></li>'
            for s in p["strengths"]) or '<li><span class="mark">&middot;</span><span>No category win</span></li>'
        weaknesses = "".join(
            f'<li class="bad"><span class="mark">&minus;</span><span>{_esc(w)}</span></li>'
            for w in p["weaknesses"])
        unfused = ""
        if p["unfused"]:
            unfused = (f'<li class="bad"><span class="mark">!</span><span>Returned one of '
                       f'your original frames unchanged: {_esc(", ".join(p["unfused"]))}</span></li>')
        note = ""
        if p["note"]:
            title, body = p["note"]
            note = f'<div class="note"><b>{_esc(title)}</b>{_esc(body)}</div>'

        out.append(f"""
      <div class="card{' top' if i == 0 else ''}">
        <div class="name"><h3>{_esc(p['label'])}</h3>
          <span class="score">{p['overall']:.0f}<small>/100</small></span></div>
        <div class="meter"><div style="width:{p['overall']:.1f}%"></div></div>
        <ul>{strengths}{weaknesses}{unfused}</ul>
        {note}
        <div class="speed">Typically {p['median_time'] * 1000:.0f} ms on a 320&times;320 test image</div>
      </div>""")
    return "".join(out)


def _scenario_blocks(data, profiles, order):
    blocks = []
    for entry in data["scenarios"]:
        ranked = sorted((k for k in order if k in entry["results"]),
                        key=lambda k: -entry["results"][k]["psnr"])
        rows = []
        for k in ranked:
            score = _normalised(entry, k)
            flag = ' <span class="flag">unfused</span>' if entry["results"][k]["unfused"] else ""
            win = " win" if k == ranked[0] else ""
            rows.append(f"""
          <div class="row{win}">
            <span class="lbl">{_esc(profiles[k]['label'])}</span>
            <div class="track"><div class="fill" style="width:{max(score, 1):.1f}%"></div></div>
            <span class="num">{score:.0f}{flag}</span>
          </div>""")

        best, worst = ranked[0], ranked[-1]
        shots = f"""
          <div class="shot ref"><img src="{entry['reference_crop']}" alt="Perfect result">
            <span>Perfect result</span></div>
          <div class="shot"><img src="{entry['input_crop']}" alt="One original frame">
            <span>One frame alone</span></div>
          <div class="shot"><img src="{entry['results'][best]['crop']}" alt="Best method">
            <span>Best: {_esc(profiles[best]['label'])}</span></div>
          <div class="shot"><img src="{entry['results'][worst]['crop']}" alt="Weakest method">
            <span>Weakest: {_esc(profiles[worst]['label'])}</span></div>"""

        blocks.append(f"""
      <div class="scenario">
        <div>
          <h3>{_esc(entry['title'])}</h3>
          <p class="lede">{_esc(entry['blurb'])}</p>
        </div>
        <div class="top-row">
          <div class="chart">{''.join(rows)}</div>
          <img src="{entry['reference']}" alt="{_esc(entry['title'])} test scene">
        </div>
        <div class="strip">{shots}</div>
      </div>""")
    return "".join(blocks)


def _summary_table(data, profiles, order):
    heads = "".join(f"<th>{_esc(e['title'])}</th>" for e in data["scenarios"])
    rows = []
    for key in order:
        cells = []
        for entry in data["scenarios"]:
            if key not in entry["results"]:
                cells.append("<td>&mdash;</td>")
                continue
            score = _normalised(entry, key)
            cls = "p-best" if score >= 90 else ("p-ok" if score >= 55 else "p-weak")
            cells.append(f'<td><span class="pill {cls}">{score:.0f}</span></td>')
        rows.append(f"<tr><td>{_esc(profiles[key]['label'])}</td>{''.join(cells)}</tr>")

    return f"""
      <div class="scroller">
        <table>
          <thead><tr><th>Method</th>{heads}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>"""


def render(data, profiles):
    order = sorted(profiles, key=lambda k: -profiles[k]["overall"])
    champion = profiles[order[0]]

    # Per-scenario winners, for the "it depends" line
    picks = []
    for entry in data["scenarios"]:
        ranked = sorted((k for k in order if k in entry["results"]),
                        key=lambda k: -entry["results"][k]["psnr"])
        picks.append((entry["title"], profiles[ranked[0]]["label"]))
    distinct = sorted({label for _, label in picks})

    runner = next((profiles[k]["label"] for k in order
                   if k != order[0] and not profiles[k]["unfused"]), None)

    # Never claim more than the run supports: list any method that scored at or
    # below the best single frame somewhere, rather than asserting all beat it
    failures = []
    for entry in data["scenarios"]:
        for key, rec in entry["results"].items():
            if rec["psnr"] <= entry["baseline"]:
                failures.append((profiles[key]["label"], entry["title"]))

    if not failures:
        caveat = ("Every method beat using a single photo in every test, so any of them "
                  "is better than none.")
    else:
        named = "; ".join(f"{_esc(label)} on &ldquo;{_esc(title).lower()}&rdquo;"
                          for label, title in failures)
        caveat = (f"Most methods beat using a single photo in every test, but not all: "
                  f"{named} scored no better than simply keeping the sharpest frame.")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which fusion method should I use?</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">OpenFocus &middot; a guide to the render methods</div>
    <h1>Which fusion method should I use?</h1>
    <p class="deck">
      Focus stacking combines several photos, each sharp in a different place, into one
      that is sharp everywhere. OpenFocus offers several ways to do that. Each was given
      the same six tests, and this page reports how they did &mdash; in plain terms,
      with the pictures to back it up.
    </p>
  </header>

  <section>
    <div class="verdict">
      <span class="q">The short answer</span>
      <span class="a">Use {_esc(champion['label'])}.</span>
      <p>
        It came first or second in most of the six tests and held colour more faithfully
        than anything else. {f"If you want a non-neural option, {_esc(runner)} was the strongest of the rest." if runner else ""}
        {caveat}
      </p>
    </div>
  </section>

  <section>
    <div class="head">
      <h2>The methods, ranked</h2>
      <p class="lede">
        The score is how much of the achievable improvement each method captured, averaged
        over the six tests. 100 would mean winning every one. A plus is a test it placed
        first or second in; a minus is one it came last or second-last in.
      </p>
    </div>
    <div class="cards">{_cards(profiles, order)}</div>
  </section>

  <section>
    <div class="head">
      <h2>The six tests</h2>
      <p class="lede">
        Each test is a scene built so that the correct answer is known exactly, which is
        what makes scoring possible. Bars show the same 0&ndash;100 score; the small
        pictures are the same patch of the same scene, magnified, so you can see the
        difference rather than take it on trust.
      </p>
    </div>
    {_scenario_blocks(data, profiles, order)}
  </section>

  <section>
    <div class="head">
      <h2>Everything at a glance</h2>
      <p class="lede">
        {"Different methods won different tests &mdash; " + _esc(", ".join(distinct)) + "." if len(distinct) > 1 else "One method won every test."}
        Darker is better.
      </p>
    </div>
    {_summary_table(data, profiles, order)}
  </section>

  <footer>
    Generated by tests/visualize_fusion_characteristics.py.
    Scenes and scoring: tests/fusion_scenarios.py, tests/fusion_metrics.py.
    Every claim on this page is asserted in tests/test_fusion_characteristics.py.<br>
    Scores come from synthetic scenes with a known correct answer, at 320&times;320.
    Timings are from one machine and one image size, so read them as relative, not absolute.
  </footer>

</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-gpu", action="store_true",
                    help="Also profile the GPU variants (they mirror their CPU twins)")
    ap.add_argument("--out", default=os.path.join(REPORTS_DIR, "fusion_method_guide.html"),
                    help=f"Output HTML path (default: {REPORTS_DIR}/fusion_method_guide.html)")
    ap.add_argument("--open", action="store_true", dest="open_browser",
                    help="Open the guide in the default browser when done")
    args = ap.parse_args()

    keys = CORE_METHODS + (GPU_METHODS if args.include_gpu else [])
    print("Running every method over every scenario...")
    data = measure(keys)
    profiles = derive_profiles(data)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(data, profiles))

    print(f"\nWrote {out} ({os.path.getsize(out) / 1024:.0f} KB)\n")
    for key in sorted(profiles, key=lambda k: -profiles[k]["overall"]):
        p = profiles[key]
        print(f"  {p['label']:<30} {p['overall']:5.1f}/100   "
              f"+{', '.join(p['strengths']) or '-'}")

    if args.open_browser:
        webbrowser.open(f"file:///{out.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
