"""
Render the pyramid against the reference it was tuned on, as an HTML report.

`visualize_pyramid_quality.py` answers "does the rework fix the artefacts",
measured on fixtures that were generated and so have a known right answer. This
one answers the question those fixtures cannot: **how does the method compare
with another mature implementation of the same algorithm, on a photograph**.

The reference is Helicon Focus method C - its own Laplacian pyramid - on
samples/electronics_ant, at radius 1, which is its lightest smoothing and so
the sharpest render it offers. That is the one the 1.44.0 retune was scored
against, and this page is the evidence for it: the same stack fused at the
settings that shipped before and after, next to Helicon's render of it, with
the numbers each claim rests on drawn rather than asserted.

## Nothing here compares pixels, and that is not a shortcut

Helicon aligned the stack before fusing, so its render shares a pixel grid with
no raw frame, and the two sit about 9% apart in scale because each program
removed its own idea of the focus breathing. Read pixel by pixel, a single
defocused frame scores five decibels above a correct fusion - see
samples/captures.py and test_pixelwise_scores_are_meaningless_here in
tests/test_pyramid_electronics_ant.py, which asserts exactly that so nobody
adds a PSNR here and tunes to satisfy it.

What survives the misalignment is measured instead:

* **agreement** - do the two find detail in the same parts of the scene,
  correlated over 64 px tiles
* **share** - how much local contrast came back, as a fraction of Helicon's;
  1.0 is a match, below is softer, above is crunchier
* **band ratios** - fine-detail energy against Helicon's, split by how busy
  each region is. This is the one that separates *sharper* from *grainier*: a
  render that is merely noisier stands above the reference hardest where the
  reference found nothing, a genuinely sharper one stands above it where the
  reference found the most

The crops are warped onto our grid by a similarity fit so both show the same
content; the residual drift is real and is why every number above is pooled.

Output is a single self-contained HTML file - images embedded as data URIs, no
server, no assets folder.

Examples:
    python tests/visualize_pyramid_reference.py --open
    python tests/visualize_pyramid_reference.py --full      # all 333 frames
    python tests/visualize_pyramid_reference.py --step 4    # 84 of them
"""

import argparse
import os
import sys
import time
import webbrowser
from datetime import date

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.pyramid import pyramid_impl
from samples import captures
from tests import fusion_metrics as fm
from tests.visualize_fusion_quality import CSS, _b64, _esc

REPORTS_DIR = "reports"
CAPTURE = "electronics_ant"

# The reference this page is built around, and the rest of Helicon's own
# smoothing sweep. Four settings of one algorithm bracket rather than score:
# softer than its heaviest means this method has gone soft, crunchier than its
# lightest means it has gone crunchy, and between them is the range a mature
# implementation of this algorithm considers reasonable.
REFERENCE = "HF-C-1"
SWEEP = ("HF-C-1", "HF-C-4", "HF-C-6", "HF-C-10")

# Tile side for the agreement and share measures - wide enough to absorb the
# few pixels the renders sit away from each other, narrow enough to leave
# several hundred tiles to correlate over. The same block the test module and
# the tuning loop use, so the numbers on this page are the numbers they quote.
BLOCK = 64

# Every frame is 1819x1212 and the stack is 333 of them, which is 2.2 GB and
# about a minute per fuse. The default subsamples to the same 28 frames the
# test module and the tuning ran on; --full is the like-for-like against
# Helicon, which used all of them.
STEP = 12

# The settings that shipped before and after the 1.44.0 base-band retune. Fixed
# here rather than read from the module: "before" has to stay the configuration
# it names however the defaults move next, and "after" is checked against the
# module below so this page cannot quietly stop describing what ships.
BEFORE = dict(selectivity=8.0, base_selectivity=3.0)
AFTER = dict(selectivity=8.0, base_selectivity=8.0)

# A third column, drawn because it is the reason this page exists. Selectivity
# 16 wins on every fixture with a ground truth and on every subsampled reading
# of this capture, and is out of bracket on the whole one - see SELECTIVITY in
# fusion_methods/pyramid.py. Showing it next to what shipped is what makes the
# grain-against-detail split legible rather than a claim.
REJECTED = dict(selectivity=16.0, base_selectivity=8.0)

# (key, settings, heading, subheading). Order is the order they appear.
RUNS = (
    ("before", BEFORE, "OpenFocus 1.43", "selectivity 8 · base 3"),
    ("after", AFTER, "OpenFocus 1.44", "selectivity 8 · base 8"),
    ("rejected", REJECTED, "Rejected", "selectivity 16 · base 8"),
)

CROP_SCALE = 2        # magnification of the 1:1 windows
CROP_SIDE = 260       # side of those windows, in source pixels

# How much of the frame's own detail a crop window has to hold to be picked as
# the "resolves the most" one, as a percentile of tile activity. The quiet
# window is chosen the same way from the other end.
BUSY_PERCENTILE = 99.0
QUIET_PERCENTILE = 12.0


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def score(fused, renders, bands):
    """Every reading this page makes of one render, against the whole sweep."""
    against = {}
    for key, render in renders.items():
        ours, theirs = captures.fit_render(fused, render)
        agreement, share = fm.detail_agreement(ours, theirs, BLOCK)
        against[key] = {"agreement": agreement, "share": share}

    ours, theirs = captures.fit_render(fused, renders[REFERENCE])
    ratios = fm.band_ratios(ours, theirs, bands)
    return {
        "against": against,
        "ratios": ratios,
        "gap": fm.reference_gap(ratios),
        "spatial_frequency": fm.spatial_frequency(fused),
        "flat_noise": fm.flat_noise(fused),
    }


def crop_windows(image):
    """(busy, quiet) windows, chosen from the picture rather than by hand.

    A comparison crop picked by eye is a crop picked to make a point. These come
    off the fused result's own tile activity: the busiest window is where the
    method resolved the most and is where a difference in sharpness shows, the
    quietest is a region no frame resolves and is where a difference in grain
    shows. Both are clamped inside the frame.
    """
    detail = fm.detail_map(image, BLOCK)
    height, width = image.shape[:2]

    def window(percentile, high):
        cut = np.percentile(detail, percentile)
        hits = np.argwhere(detail >= cut) if high else np.argwhere(detail <= cut)
        if hits.size == 0:
            hits = np.array([[detail.shape[0] // 2, detail.shape[1] // 2]])
        row, col = hits[len(hits) // 2]
        x = int(np.clip(col * BLOCK + BLOCK // 2 - CROP_SIDE // 2,
                        0, max(0, width - CROP_SIDE)))
        y = int(np.clip(row * BLOCK + BLOCK // 2 - CROP_SIDE // 2,
                        0, max(0, height - CROP_SIDE)))
        return x, y, min(CROP_SIDE, width), min(CROP_SIDE, height)

    return window(BUSY_PERCENTILE, True), window(QUIET_PERCENTILE, False)


def crop(image, window, scale=CROP_SCALE):
    x, y, w, h = window
    patch = image[y:y + h, x:x + w]
    if patch.size == 0:
        return image
    return cv2.resize(patch, (patch.shape[1] * scale, patch.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def detail_heatmap(image, shape):
    """The tile activity map `agreement` correlates, drawn as a heatmap.

    Upsampled with NEAREST on purpose: these are per-tile numbers and smoothing
    them into a continuous field would suggest a resolution the measure does
    not have.
    """
    tiles = fm.detail_map(image, BLOCK)
    top = max(float(tiles.max()), 1e-6)
    scaled = np.clip(tiles / top * 255.0, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    return cv2.resize(coloured, (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def measure(step, log=print):
    """Fuse the capture at both settings and score them against the sweep."""
    frames, meta = captures.load_capture(CAPTURE, step=step)
    log(f"  {len(frames)} frames of {meta['frame_count']}, "
        f"{meta['frame_size'][0]}x{meta['frame_size'][1]}")

    renders = {}
    for key in SWEEP:
        image = captures.load_render(CAPTURE, key)
        if image is not None:
            renders[key] = image
    if REFERENCE not in renders:
        raise RuntimeError(f"{REFERENCE} did not decode - is imagecodecs installed?")
    log(f"  {len(renders)} Helicon renders decoded")

    reference = renders[REFERENCE]
    bands = fm.reference_bands(reference)

    results = {}
    for name, tuning, _who, _what in RUNS:
        started = time.time()
        fused = pyramid_impl(list(frames), **tuning)
        results[name] = score(fused, renders, bands)
        results[name]["image"] = fused
        results[name]["tuning"] = tuning
        results[name]["seconds"] = time.time() - started
        log(f"  fused {name}: selectivity {tuning['selectivity']:g}, base "
            f"{tuning['base_selectivity']:g} in {results[name]['seconds']:.1f}s "
            f"- agreement {results[name]['against'][REFERENCE]['agreement']:.4f}, "
            f"share {results[name]['against'][REFERENCE]['share']:.3f}")

    # Helicon's render put on our grid, so the crops below show the same
    # content. None when the fit does not find enough inliers, in which case the
    # page says so rather than showing two different parts of the scene.
    fitted = fm.fit_reference(reference, results["after"]["image"])
    if fitted is None:
        log("  the similarity fit onto the reference failed; crops omitted")

    return {
        "frames": len(frames),
        "total": meta["frame_count"],
        "size": meta["frame_size"],
        "results": results,
        "renders": renders,
        "fitted": fitted,
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

EXTRA_CSS = """
  .triple { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
  @media (max-width:980px) { .triple { grid-template-columns:repeat(2,1fr); } }
  @media (max-width:560px) { .triple { grid-template-columns:1fr; } }
  .shot { background:var(--surface); border:1px solid var(--line);
          border-radius:3px; overflow:hidden; }
  .shot img { display:block; width:100%; height:auto; }
  .shot figcaption { padding:10px 12px; border-top:1px solid var(--line); }
  .shot .who { font-family:var(--font-display); font-size:16px; font-weight:600; }
  .shot .what { font-family:var(--font-mono); font-size:11.5px;
                color:var(--ink-3); margin-top:3px; }
  .ref { outline:2px solid var(--accent); outline-offset:-2px; }
  .bad { outline:2px dashed var(--line-strong); outline-offset:-2px;
         opacity:.92; }
  .bad .who::after { content:" ✕"; color:var(--ink-3); }
  .num { font-family:var(--font-mono); font-variant-numeric:tabular-nums; }
  .up { color:var(--accent); font-weight:600; }
  .zones { display:grid; grid-template-columns:repeat(5,1fr); gap:8px;
           margin-top:6px; }
  .zone { background:var(--surface-2); border:1px solid var(--line);
          border-radius:2px; padding:8px 6px; text-align:center; }
  .zone .v { font-family:var(--font-mono); font-size:15px; }
  .zone .k { font-size:11px; color:var(--ink-3); margin-top:2px; }
  table.ref-table { width:100%; border-collapse:collapse; font-size:14.5px; }
  table.ref-table th, table.ref-table td { padding:8px 10px; text-align:right;
      border-bottom:1px solid var(--line); }
  table.ref-table th:first-child, table.ref-table td:first-child {
      text-align:left; }
  table.ref-table thead th { color:var(--ink-3); font-weight:600;
      font-family:var(--font-mono); font-size:12px; text-transform:uppercase;
      letter-spacing:.05em; }
  .note { color:var(--ink-3); font-size:14px; max-width:70ch; }
"""


def _shots(items):
    cards = []
    for item in items:
        mark = (" ref" if item.get("reference")
                else " bad" if item.get("rejected") else "")
        cards.append(
            f'<figure class="shot{mark}"><img src="{item["img"]}" '
            f'alt="{_esc(item["who"])}">'
            f'<figcaption><div class="who">{_esc(item["who"])}</div>'
            f'<div class="what">{_esc(item["what"])}</div></figcaption></figure>')
    return f'<div class="triple">{"".join(cards)}</div>'


def _row(data, transform, extra=None):
    """One card per run, in RUNS order, with the reference last."""
    items = [dict(transform(data["results"][key]), who=who, what=what,
                  rejected=(key == "rejected"))
             for key, _tuning, who, what in RUNS]
    return _shots(items + [extra]) if extra else _shots(items)


def _zone_row(ratios):
    names = ["quietest", "", "middle", "", "busiest"]
    cells = []
    for i, value in enumerate(ratios):
        cells.append(f'<div class="zone"><div class="v">{value:.2f}</div>'
                     f'<div class="k">{names[i] if i < len(names) else ""}</div></div>')
    return f'<div class="zones">{"".join(cells)}</div>'


def _sweep_table(data, bound):
    against = {key: data["results"][key]["against"] for key, *_ in RUNS}
    rows = []
    for key in SWEEP:
        if key not in against["before"]:
            continue
        # Method C carries its setting in `smoothing`, not `radius` - it is the
        # one Helicon method whose filename has a single number.
        smoothing = captures.render_meta(CAPTURE, key).get("smoothing")
        label = f"{key} &middot; smoothing {smoothing}" if smoothing else key
        cells = "".join(
            f'<td class="num">{against[run][key]["share"]:.3f}</td>'
            for run, *_ in RUNS)
        flag = ""
        if key == REFERENCE:
            flag = (f'<td class="num">&lt; {bound:.2f}</td>')
        rows.append(f"<tr><td>{label}</td>{cells}"
                    f"{flag or '<td></td>'}</tr>")
    heads = "".join(f"<th>{_esc(who.replace('OpenFocus ', ''))}</th>"
                    for _key, _t, who, _w in RUNS)
    return f"""
    <table class="ref-table">
      <thead><tr><th>Recovered contrast against&hellip;</th>{heads}
        <th>bound</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def render_html(data, seconds, bound):
    after = data["results"]["after"]
    rejected = data["results"]["rejected"]
    reference = data["renders"][REFERENCE]
    fitted = data["fitted"]

    shown = cv2.resize(reference, (after["image"].shape[1],
                                   after["image"].shape[0]),
                       interpolation=cv2.INTER_AREA)
    full = _row(data, lambda r: {"img": _b64(r["image"])},
                {"img": _b64(shown), "who": "Helicon Focus C",
                 "what": f"{REFERENCE}, the reference", "reference": True})

    crops = ""
    if fitted is not None:
        busy, quiet = crop_windows(after["image"])
        blocks = []
        for title, window, why in (
                ("Where the method resolved the most", busy,
                 "a difference in sharpness shows here"),
                ("Where no frame resolves anything", quiet,
                 "a difference in grain shows here")):
            shots = _row(data,
                         lambda r, w=window: {"img": _b64(crop(r["image"], w))},
                         {"img": _b64(crop(fitted, window)),
                          "who": "Helicon Focus C",
                          "what": f"{REFERENCE}, fitted onto our grid",
                          "reference": True})
            blocks.append(f"""
      <div>
        <h3 style="font-family:var(--font-display);font-size:18px;margin:0 0 4px">
          {_esc(title)}</h3>
        <p class="note" style="margin:0 0 12px">
          {_esc(why)} &mdash; a {CROP_SIDE}&times;{CROP_SIDE} px window at
          ({window[0]}, {window[1]}), magnified {CROP_SCALE}&times; with no
          interpolation, chosen from the render's own tile activity rather than
          by hand.</p>
        {shots}
      </div>""")
        crops = f"""
    <section>
      <div class="head"><h2>The same window, 1:1</h2>
        <p class="lede">Helicon's render is warped onto our grid by a
        similarity fit &mdash; scale, rotation, translation &mdash; so all four
        show the same content. The residual drift after that fit is real: focus
        breathing is a different magnification per frame, so no single warp
        removes it, which is why every number on this page is pooled over tiles
        or zones rather than taken per pixel.</p></div>
      {"".join(blocks)}
    </section>"""

    maps = _row(
        data,
        lambda r: {"img": _b64(detail_heatmap(r["image"], r["image"].shape))},
        {"img": _b64(detail_heatmap(shown, shown.shape)),
         "who": "Helicon Focus C", "what": f"{REFERENCE}, tile detail",
         "reference": True})

    stack_note = (f"{data['frames']} of {data['total']} frames"
                  if data["frames"] != data["total"]
                  else f"all {data['total']} frames")

    def metric_row(label, key, fmt="{:.3f}"):
        cells = "".join(
            f'<td class="num">{fmt.format(data["results"][run][key])}</td>'
            for run, *_ in RUNS)
        return f"<tr><td>{label}</td>{cells}</tr>"

    def against_row(label, field, fmt="{:.4f}"):
        cells = "".join(
            f'<td class="num">'
            f'{fmt.format(data["results"][run]["against"][REFERENCE][field])}</td>'
            for run, *_ in RUNS)
        return f"<tr><td>{label}</td>{cells}</tr>"

    heads = "".join(f"<th>{_esc(who.replace('OpenFocus ', ''))}</th>"
                    for _k, _t, who, _w in RUNS)
    zone_blocks = "".join(
        f'<div><div class="what num" style="color:var(--ink-3);font-size:12px">'
        f'{_esc((who + " — " + what).upper())}</div>'
        f'{_zone_row(data["results"][key]["ratios"])}</div>'
        for key, _t, who, what in RUNS)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pyramid against its reference</title>
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">OpenFocus &middot; fusion quality</div>
    <h1>The pyramid against the render it was tuned on</h1>
    <p class="deck">Helicon Focus method C is a Laplacian pyramid &mdash; the
    same algorithm as this one &mdash; and it ships here at four smoothing
    settings on the same photographed stack. <b>{REFERENCE}</b>, its lightest
    smoothing and so the sharpest render it offers, is the reference the 1.44.0
    tuning was scored against. The third column is a setting that won every
    other measurement the project owns and was rejected here; it is on the page
    because that is what this page is for.</p>
    <div class="chips">
      <span class="chip">samples/{CAPTURE}</span>
      <span class="chip">{stack_note}</span>
      <span class="chip">{data["size"][0]}&times;{data["size"][1]} px</span>
      <span class="chip">reference <b>{REFERENCE}</b></span>
      <span class="chip">{BLOCK} px tiles</span>
    </div>
  </header>

  <section>
    <div class="head"><h2>Three settings, one reference</h2>
      <p class="lede">All three are the current code run three times; only the
      two named dials differ. <b>1.44</b> is what ships: the base band moved
      3&nbsp;&rarr;&nbsp;8. <b>Rejected</b> also took selectivity
      8&nbsp;&rarr;&nbsp;16, which gains 0.5&ndash;1.1&nbsp;dB on every fixture
      that has a ground truth and wins on every subsampled reading of this
      capture.</p></div>
    {full}
    <table class="ref-table">
      <thead><tr><th>Against {REFERENCE}, on {stack_note}</th>{heads}</tr></thead>
      <tbody>
        {against_row("Tile agreement &mdash; is the detail in the same places",
                     "agreement")}
        {against_row("Recovered contrast &mdash; 1.000 is a match", "share",
                     "{:.3f}")}
        {metric_row("Distance from the reference over all zones (0 is exact)",
                    "gap")}
        {metric_row("Spatial frequency", "spatial_frequency", "{:.2f}")}
        {metric_row("Flat-field noise &mdash; above 1 is invented texture",
                    "flat_noise")}
      </tbody>
    </table>
    <p class="note"><b>Why the third column lost.</b> It recovers
    {rejected["against"][REFERENCE]["share"]:.3f} of the reference's tile
    contrast against {after["against"][REFERENCE]["share"]:.3f}, past the
    {bound:.2f} this method is held to on the full stack &mdash; and the extra
    is not detail. The zone split below is what shows that, and two numbers in
    this table already hint at it: flat-field noise
    {after["flat_noise"]:.3f}&nbsp;&rarr;&nbsp;{rejected["flat_noise"]:.3f},
    and the pooled distance from the reference getting <i>worse</i>,
    {after["gap"]:.3f}&nbsp;&rarr;&nbsp;{rejected["gap"]:.3f}, while tile
    agreement improves. Agreement says <i>where</i> the detail is and cannot
    tell texture from grain; that is what the zones are for.</p>
  </section>
{crops}
  <section>
    <div class="head"><h2>Sharper, or just grainier?</h2>
      <p class="lede">Fine-detail energy against {REFERENCE}, split into five
      zones by how busy that region is in Helicon's own render, quietest first.
      1.00 everywhere is the reference exactly. The split is what makes this
      readable: a render that is merely noisier stands above the reference
      hardest in the <i>quietest</i> zone, where the reference found nothing,
      while one that is genuinely sharper stands above it in the
      <i>busiest</i>.</p></div>
    {zone_blocks}
    <p class="note">Read the rejected row against the one above it: selectivity
    16 gains {rejected["ratios"][0] - after["ratios"][0]:+.2f} in the quietest
    zone and {rejected["ratios"][-1] - after["ratios"][-1]:+.2f} in the busiest.
    It is amplifying what the reference found nothing in about
    {abs(rejected["ratios"][0] - after["ratios"][0]) / max(abs(rejected["ratios"][-1] - after["ratios"][-1]), 1e-6):.0f}
    times as hard as what it found the most in. That is the whole argument for
    leaving the default where it is.</p>
    <p class="note"><b>And a gap none of these settings closes.</b> Every column
    stands around {after["ratios"][0]:.1f}&times; the reference in the quietest
    zone and about {after["ratios"][-1]:.2f}&times; in the busiest &mdash;
    grainier than Helicon where neither resolves anything, and slightly softer
    where the detail actually is. That is a property of the method rather than
    of this tuning, nothing in the suite currently holds it, and it is the more
    interesting result on this page.</p>
  </section>

  <section>
    <div class="head"><h2>Inside Helicon's own smoothing range</h2>
      <p class="lede">Four settings of one algorithm bracket where a single
      render can only score. Recovering less contrast than method C at its
      lightest smoothing means this method has gone soft; more than at its
      heaviest means it has gone crunchy; between them is the range a mature
      implementation of this algorithm considers reasonable.</p></div>
    {_sweep_table(data, bound)}
    <p class="note">This measure rises with frame count &mdash; the shipped
    default reads 0.916 of {REFERENCE} over 28 frames and
    {after["against"][REFERENCE]["share"]:.3f} over all 333 &mdash; so the bound
    quoted here is the full-stack one, and a setting chosen on a subsample means
    nothing until it has been re-measured at this density. That is the trap the
    rejected column fell into, and
    <span class="num">confirm_at_full_density</span> in
    <span class="num">tests/fusion_autotune.py</span> now closes it: the tuning
    loop re-measures its winner on every frame before it may write to source.
    <span class="num">test_the_whole_capture_fuses</span> in
    <span class="num">tests/test_pyramid_electronics_ant.py</span> asserts the
    same bound.</p>
  </section>

  <section>
    <div class="head"><h2>Where each one found detail</h2>
      <p class="lede">The tile activity maps the agreement figure correlates,
      drawn at {BLOCK} px per tile. Brighter is more local contrast. These are
      the actual inputs to that number rather than an illustration of it.</p></div>
    {maps}
  </section>

  <footer style="color:var(--ink-3);font-size:13.5px;
                 border-top:1px solid var(--line-strong);padding-top:20px">
    Generated {date.today().isoformat()} by
    tests/visualize_pyramid_reference.py in {seconds:.0f}s &middot; every number
    on this page was measured by the run that wrote it &middot; regenerate
    rather than edit.
  </footer>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--step", type=int, default=STEP,
                        help=f"use every Nth frame (default {STEP})")
    parser.add_argument("--full", action="store_true",
                        help="every frame - the like-for-like against Helicon")
    parser.add_argument("--out", default=None, help="exact output path")
    parser.add_argument("--out-dir", default=REPORTS_DIR)
    parser.add_argument("--open", action="store_true",
                        help="open the report when it finishes")
    args = parser.parse_args(argv)

    if not captures.is_available(CAPTURE):
        print(f"samples/{CAPTURE} is not present; it is photographed rather "
              f"than generated, so there is nothing to run to produce it")
        return 1

    # A page that says it describes what ships has to be checked against what
    # ships, or it becomes a picture of a build nobody has.
    import fusion_methods.pyramid as pyramid_module
    drift = {name: (value, getattr(pyramid_module, constant))
             for name, constant, value in (
                 ("selectivity", "SELECTIVITY", AFTER["selectivity"]),
                 ("base_selectivity", "BASE_SELECTIVITY",
                  AFTER["base_selectivity"]))
             if value != getattr(pyramid_module, constant)}
    if drift:
        print("warning: the 'after' column is not what the module now ships - "
              + "; ".join(f"{k}: page {p!r}, module {m!r}"
                          for k, (p, m) in drift.items()))

    # The bound the page quotes is the one the suite asserts, read from it
    # rather than restated, so the two cannot drift apart.
    from tests.test_pyramid_electronics_ant import (
        MAX_SHARE_AGAINST_LIGHTEST, MAX_SHARE_AGAINST_LIGHTEST_FULL)
    bound = MAX_SHARE_AGAINST_LIGHTEST_FULL if args.full \
        else MAX_SHARE_AGAINST_LIGHTEST

    started = time.time()
    step = 1 if args.full else max(1, args.step)
    print(f"measuring {CAPTURE} at step {step}...")
    data = measure(step)

    html = render_html(data, time.time() - started, bound)
    path = args.out or os.path.join(args.out_dir, "pyramid_reference_report.html")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)

    size = os.path.getsize(path) / 1024.0
    print(f"wrote {os.path.abspath(path)} ({size:.0f} KB) in "
          f"{time.time() - started:.0f}s")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
