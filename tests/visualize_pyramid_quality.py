"""
Render the pyramid method's quality testing as a standalone HTML report.

The sweep reports written by `visualize_fusion_quality.py` answer "what does
this dial change". This one answers the question item 19 of
docs/ALGORITHM_IMPROVEMENTS.md was opened by: **what was wrong with the
published choose-max rule, and does the rework actually fix it** - measured on
the same fixtures the test suite guards, with the evidence drawn rather than
asserted.

Five sections:

* the two artefacts, each with the picture before, after, and a map of exactly
  which pixels are at fault
* every fixture in the project, before against after
* what each of the six exposed controls does to both quality and artefact
* CPU against GPU
* the tests that hold each claim, so a reader can go and run them

Output is a single self-contained HTML file - images embedded as data URIs, no
server, no assets folder. Requires only numpy and opencv-python; the GPU
section is skipped with a printed reason when no torch device is present.

Examples:
    python tests/visualize_pyramid_quality.py --open
    python tests/visualize_pyramid_quality.py --quick        # skip the sweeps
    python tests/visualize_pyramid_quality.py --out /tmp/p.html
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
from tests import fusion_metrics as fm
from tests import fusion_registry as reg
from tests import fusion_scenarios as scenarios
from tests.test_dct_defocus_wash import _veil_stack
from tests.visualize_fusion_quality import CSS, _b64, _esc

REPORTS_DIR = "reports"
CROP_SCALE = 3        # magnification of the "worst window" insets
CROP_SIDE = 96        # side of that window, in source pixels

# The method as it shipped before item 19: Burt-Adelson choose-max, no noise
# gate, activity-proportional base, no clamp. Every "before" number on the page
# is this configuration, run by the current code - so the comparison is of
# rules, not of two builds that might differ elsewhere.
PUBLISHED = dict(selectivity=float("inf"), coherence=0.0, noise_gate=False,
                 base_selectivity=1.0, envelope=False)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def envelope_breach(fused, stack):
    """Where, and how badly, the result falls outside what its sources support.

    Returns (worst, share, mask). A pixel below the darkest source or above the
    brightest cannot have come from the stack: the true all-in-focus value is
    one of the sources by construction, and any blend of them lies between. It
    is the reconstruction's own invention, and over a smooth background a
    connected run of them is the filament this method was reported for.
    """
    frames = np.stack(stack).astype(np.int32)
    lowest, highest = frames.min(axis=0), frames.max(axis=0)
    out = fused.astype(np.int32)
    breach = np.maximum(lowest - out, 0) + np.maximum(out - highest, 0)
    worst = int(breach.max())
    share = float(np.mean(breach.max(axis=2) > 8) * 100.0)
    return worst, share, breach.max(axis=2)


def _patch_count(mask):
    """How many separate regions the mask is made of."""
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8))
    return count - 1          # label 0 is the background


def flat_field_noise(fused, sources):
    """Detail the fusion invented where every source is smooth.

    The fused image's high-pass energy over the typical frame's, measured only
    where no source has anything. 1.0 means the background came through as
    smooth as it went in; above that the fusion added structure of its own -
    grain it amplified, or filaments stitched out of mismatched frames. Below
    it, the averaging bought back signal-to-noise.
    """
    def highpass(img):
        grey = fm._gray32(img)
        return cv2.blur(np.abs(grey - cv2.blur(grey, (7, 7))), (21, 21))

    source_hp = np.stack([highpass(s) for s in sources])
    loudest = source_hp.max(axis=0)
    quiet = loudest <= np.percentile(loudest, 35.0)
    typical = float(np.median(np.median(source_hp, axis=0)[quiet])) + 1e-6
    return float(np.median(highpass(fused)[quiet])) / typical


def veil_share(fused, reference, interior):
    """Percentage of the smooth dark body rendered far brighter than it should be."""
    grey = fm._gray32(fused)
    ref_grey = fm._gray32(reference)
    return float(np.mean((grey - ref_grey > 45)[interior]) * 100.0)


def veil_mask(fused, reference, interior):
    grey = fm._gray32(fused)
    ref_grey = fm._gray32(reference)
    return ((grey - ref_grey > 45) & interior)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _overlay(image, mask, colour=(60, 60, 240), strength=0.62):
    """The picture, dimmed, with the offending pixels washed in red.

    Painted in place rather than shown as a separate heatmap: the point of the
    map is *where on the subject* the fault lands, and a mask floating on black
    loses that. Washed rather than replaced for the same reason - one of these
    faults covers a whole region, and a solid fill would hide the very thing it
    is pointing at.
    """
    base = image.astype(np.float32) * 0.5 + 34.0
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    wash = base[grown] * (1.0 - strength) + np.float32(colour) * strength
    base[grown] = wash
    return np.clip(base, 0, 255).astype(np.uint8)


def _worst_window(mask, side=CROP_SIDE):
    """The `side`x`side` window showing the most of `mask`'s edge.

    Its edge rather than its area: one of these faults covers a whole region,
    and the densest patch of it is a window entirely inside the fault, which
    shows a reader nothing to compare against. Sitting on the boundary puts the
    damage and the intact picture side by side. For a fault made of scattered
    specks - the other one - every speck is edge, so this picks the same window
    an area search would.
    """
    height, width = mask.shape[:2]
    side = min(side, height, width)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8))
    density = cv2.boxFilter(edge.astype(np.float32), -1, (side, side),
                            normalize=False, borderType=cv2.BORDER_CONSTANT)
    half = side // 2
    inner = density[half:height - side + half + 1, half:width - side + half + 1]
    if inner.size == 0:
        return (0, 0, side, side)
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return (int(x), int(y), side, side)


def _crop(img, window):
    """A window of the picture, magnified to a fixed width by nearest neighbour.

    Fixed width rather than fixed magnification, because the two artefacts want
    windows of very different sizes - filaments a few pixels across against a
    region covering a third of the frame - and the cards they land in are the
    same size either way.
    """
    x, y, w, h = window
    patch = img[y:y + h, x:x + w]
    scale = max(1, round(CROP_SIDE * CROP_SCALE / max(w, h)))
    return cv2.resize(patch, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _fuse(stack, **tuning):
    start = time.perf_counter()
    fused = pyramid_impl(list(stack), **tuning)
    return fused, time.perf_counter() - start


def artefact_case(key, title, question, fault, stack, reference, mask_fn, extra_metric,
                  crop_side=CROP_SIDE):
    """One artefact: the same stack fused by both rules, with the evidence."""
    print(f"[{key}] fusing before/after...", flush=True)
    before, before_s = _fuse(stack, **PUBLISHED)
    after, after_s = _fuse(stack)

    rows = []
    for label, fused, seconds in (("Choose-max (published rule)", before, before_s),
                                  ("OpenFocus 1.19", after, after_s)):
        worst, share, breach = envelope_breach(fused, stack)
        rows.append({
            "label": label,
            "img": _b64(fused),
            "mask": _b64(_overlay(fused, mask_fn(fused))),
            "psnr": fm.psnr(fused, reference),
            "flat": flat_field_noise(fused, stack),
            "breach_worst": worst,
            "breach_share": share,
            "breach": breach,
            "extra": extra_metric(fused),
            "seconds": seconds,
            "raw": fused,
        })

    # The crop follows the failure: whichever window of the *before* render
    # holds the most of it is where a reader should be looking in both.
    window = _worst_window(mask_fn(before), crop_side)
    for row in rows:
        row["crop"] = _b64(_crop(row["raw"], window))
        row["crop_mask"] = _b64(_crop(_overlay(row["raw"], mask_fn(row["raw"])), window))
        del row["raw"], row["breach"]
    if reference is not None:
        reference_crop = _b64(_crop(reference, window))
    else:
        reference_crop = None

    return {
        "key": key, "title": title, "question": question, "fault": fault,
        "rows": rows, "window": list(window),
        "reference": _b64(reference) if reference is not None else None,
        "reference_crop": reference_crop,
        "frames": len(stack),
        "size": f"{stack[0].shape[1]}x{stack[0].shape[0]}",
    }


def fixture_table():
    """Every scenario the project owns, both rules, four measurements each."""
    cases = [(key, builder()) for key, builder in (
        ("fine_texture", scenarios.fine_texture),
        ("sensor_noise", scenarios.sensor_noise),
        ("depth_edge", scenarios.depth_edge),
        ("long_stack", scenarios.long_stack),
        ("low_contrast", scenarios.low_contrast),
        ("saturated_colour", scenarios.saturated_colour),
        ("deep_stack", scenarios.deep_stack),
    )]
    veil_stack, veil_reference, _interior = _veil_stack()
    cases.append(("veil", (veil_stack, veil_reference, None)))

    rows = []
    for key, (stack, reference, _masks) in cases:
        print(f"[{key}] {len(stack)} frames...", flush=True)
        entry = {"key": key, "frames": len(stack)}
        for side, tuning in (("before", PUBLISHED), ("after", {})):
            fused, seconds = _fuse(stack, **tuning)
            worst, share, _mask = envelope_breach(fused, stack)
            entry[side] = {
                "psnr": fm.psnr(fused, reference),
                "qabf": fm.qabf(fused, stack[::max(1, len(stack) // 8)]),
                "flat": flat_field_noise(fused, stack),
                "breach_worst": worst,
                "breach_share": share,
                "seconds": seconds,
            }
        rows.append(entry)
    return rows


def control_sweeps():
    """What each exposed control does, on a fixture that can feel it.

    Every sweep is read from `tests/fusion_registry.py`, the same declaration
    the test suite iterates, so this page cannot drift from what is tested.
    Two scenarios, because the controls trade one against the other: deep_stack
    holds the artefact, fine_texture holds the detail a fix can cost.
    """
    from tests.visualize_fusion_quality import _value_label

    method = reg.get("pyramid")
    stacks = {
        "deep_stack": scenarios.deep_stack(),
        "fine_texture": scenarios.fine_texture(),
    }
    defaults = {"levels": None, "energy_window": 5, "selectivity": 8.0,
                "coherence": 0.0, "base_selectivity": 3.0, "noise_gate": True,
                "envelope": True}

    sweeps = []
    for name, values, blurb in method.sweeps:
        print(f"[sweep {name}]", flush=True)
        points = []
        for value in values:
            point = {"label": _value_label(value),
                     "default": value == defaults.get(name)}
            for scene, (stack, reference, _m) in stacks.items():
                fused, _s = _fuse(stack, **{name: value})
                worst, _share, _mask = envelope_breach(fused, stack)
                point[scene] = {"psnr": fm.psnr(fused, reference),
                                "flat": flat_field_noise(fused, stack),
                                "breach": worst}
            points.append(point)
        sweeps.append({"name": name, "blurb": blurb, "points": points})
    return sweeps


def gpu_agreement():
    """The device twin, on the stack that shows the artefact."""
    try:
        import torch
        has_device = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if not has_device:
            return {"skipped": "no CUDA or MPS device on this machine"}
        from fusion_methods.pyramid_torch import pyramid_torch_impl
    except Exception as exc:
        return {"skipped": f"{type(exc).__name__}: {exc}"}

    stack, reference, _m = scenarios.deep_stack()
    rows = []
    for label, tuning in (("Defaults", {}), ("Scale coherence 0.5", {"coherence": 0.5}),
                          ("Noise gate off", {"noise_gate": False}),
                          ("Winner takes all", {"selectivity": float("inf")})):
        print(f"[gpu] {label}", flush=True)
        cpu, cpu_s = _fuse(stack, **tuning)
        start = time.perf_counter()
        gpu = pyramid_torch_impl(list(stack), **tuning)
        gpu_s = time.perf_counter() - start
        rows.append({
            "label": label,
            "agreement": fm.psnr(gpu, cpu),
            "worst": int(np.abs(cpu.astype(np.int32) - gpu.astype(np.int32)).max()),
            "cpu_psnr": fm.psnr(cpu, reference),
            "gpu_psnr": fm.psnr(gpu, reference),
            "cpu_seconds": cpu_s,
            "gpu_seconds": gpu_s,
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

EXTRA_CSS = """
  .split { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
  .split .stage { line-height:0; }
  .verdict { display:flex; flex-wrap:wrap; gap:10px; margin-top:2px; }
  .tag { font-family:var(--font-mono); font-size:12px; padding:5px 10px;
         border-radius:2px; border:1px solid var(--line); background:var(--surface-2);
         color:var(--ink-2); }
  .tag.bad { border-color:#b4462a; color:#b4462a; }
  .tag.good { border-color:var(--accent); color:var(--accent); }
  @media (prefers-color-scheme: dark) { .tag.bad { border-color:#f0846a; color:#f0846a; } }
  :root[data-theme="dark"] .tag.bad { border-color:#f0846a; color:#f0846a; }
  :root[data-theme="light"] .tag.bad { border-color:#b4462a; color:#b4462a; }
  .pair { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }
  .delta { font-family:var(--font-mono); font-size:12px; }
  .delta.up { color:var(--accent); }
  .delta.down { color:#b4462a; }
  @media (prefers-color-scheme: dark) { .delta.down { color:#f0846a; } }
  :root[data-theme="dark"] .delta.down { color:#f0846a; }
  :root[data-theme="light"] .delta.down { color:#b4462a; }
  .sweep { border-top:1px solid var(--line); padding-top:18px; margin-top:18px; }
  .sweep:first-child { border-top:0; padding-top:0; margin-top:0; }
  .sweep h3 { font-family:var(--font-display); font-size:17px; margin:0 0 2px;
              font-weight:600; }
  .sweep .blurb { font-size:13px; color:var(--ink-3); max-width:70ch; margin:0 0 14px; }
  .legend { display:flex; gap:18px; font-family:var(--font-mono); font-size:11px;
            letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3);
            margin-bottom:10px; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px;
              margin-right:6px; vertical-align:middle; }
  .row.dim .fill { background:var(--bar-track); border:1px solid var(--bar-muted); }
  .row .num.small { font-size:11px; }
  ul.guards { margin:0; padding-left:20px; color:var(--ink-2); }
  ul.guards li { margin-bottom:9px; }
  ul.guards code { font-family:var(--font-mono); font-size:12.5px; color:var(--ink); }
"""


def _stage(image_id, layers, first_stamp):
    """A picture with a segmented control switching what is drawn on it.

    Each layer carries its own stamp, so the caption over the image always says
    which of them is showing.
    """
    imgs = "".join(
        f'<img data-layer="{key}" data-hidden="{"false" if i == 0 else "true"}" '
        f'src="{src}" alt="{_esc(alt)}">'
        for i, (key, src, alt, _button, _stamp) in enumerate(layers))
    segs = "".join(
        f'<button type="button" data-target="{image_id}" data-layer="{key}" '
        f'data-stamp="{_esc(stamp)}" aria-pressed="{"true" if i == 0 else "false"}">'
        f'{_esc(button)}</button>'
        for i, (key, _src, _alt, button, stamp) in enumerate(layers))
    return f"""
      <div class="stage" id="{image_id}">
        <div class="layers">{imgs}</div>
        <span class="stamp">{_esc(first_stamp)}</span>
      </div>
      <div class="seg switch" style="margin-top:10px">{segs}</div>"""


def _artefact_section(case):
    before, after = case["rows"]

    def tags(row, is_after):
        # Coloured by what the number says, not by which column it is in: the
        # envelope tags read themselves, and only the headline metric of this
        # particular artefact is marked by side.
        breach = "good" if row["breach_worst"] == 0 else (
            "bad" if row["breach_worst"] > 8 else "")
        return (f'<span class="tag {breach}">{row["breach_worst"]} levels outside the '
                f'sources</span>'
                f'<span class="tag {breach}">{row["breach_share"]:.2f}% of the frame</span>'
                f'<span class="tag {"good" if is_after else "bad"}">{row["extra"]}</span>'
                f'<span class="tag">flat-field noise {row["flat"]:.2f}</span>'
                f'<span class="tag">{row["psnr"]:.2f} dB</span>')

    panels = []
    for i, (row, is_after) in enumerate(((before, False), (after, True))):
        stage = _stage(
            f'{case["key"]}-{i}',
            [("fused", row["img"], f'{row["label"]}, fused',
              "Result", row["label"]),
             ("mask", row["mask"], f'{row["label"]}, faulty pixels marked',
              "What is wrong", case["fault"])],
            row["label"])
        panels.append(f"""
      <div>
        {stage}
        <div class="verdict" style="margin-top:12px">{tags(row, is_after)}</div>
      </div>""")

    # Old render, the same crop with its faults marked, new render, truth - so the
    # wide view above and the detail here are unmistakably the same pixels.
    cells = [(before["crop"], before["label"], ""),
             (before["crop_mask"], case["fault"], ""),
             (after["crop"], after["label"], ""),
             (case["reference_crop"], "Ground truth", " truth")]
    crops = "".join(f"""
      <figure class="card{cls}">
        <div class="frame strip"><img src="{src}" alt="Detail, {_esc(label)}"></div>
        <figcaption class="cap"><span class="name">{_esc(label)}</span></figcaption>
      </figure>""" for src, label, cls in cells if src)

    x, y, w, h = case["window"]
    zoom = max(1, round(CROP_SIDE * CROP_SCALE / max(w, h)))
    return f"""
    <section>
      <div class="head">
        <h2>{_esc(case['title'])}</h2>
        <p class="lede">{case['question']}</p>
        <div class="chips">
          <span class="chip"><b>{case['frames']}</b> frames</span>
          <span class="chip"><b>{case['size']}</b></span>
          <span class="chip">before / after on <b>one</b> build</span>
        </div>
      </div>
      <div class="split">{''.join(panels)}</div>
      <div class="head" style="border-top-width:0;padding-top:0">
        <p class="lede">The worst {w}&times;{h} window of the old render
        {f'at {zoom}&times;, ' if zoom > 1 else ''}the same window with its faulty
        pixels marked, and the same window again in the new render.</p>
      </div>
      <div class="grid pair">{crops}</div>
      <p class="note">Window at ({x},&nbsp;{y}), chosen automatically as the densest
      patch of faulty pixels in the old render - not picked by hand.</p>
    </section>"""


def _fixture_section(rows):
    metrics = [
        ("psnr", "PSNR", "{:.2f}", True),
        ("qabf", "Q<sup>AB/F</sup>", "{:.3f}", True),
        ("flat", "Flat-field noise", "{:.2f}", False),
        ("breach_worst", "Outside sources", "{:.0f}", False),
    ]

    body = []
    for row in rows:
        cells = []
        for key, _label, fmt, higher in metrics:
            old, new = row["before"][key], row["after"][key]
            change = new - old
            better = change > 0 if higher else change < 0
            arrow = "" if abs(change) < (0.005 if key != "breach_worst" else 0.5) else (
                f'<span class="delta {"up" if better else "down"}">'
                f'{"+" if change > 0 else "&minus;"}{fmt.format(abs(change))}</span>')
            cells.append(f"<td>{fmt.format(old)}</td>"
                         f'<td class="{"best" if better else ""}">{fmt.format(new)} {arrow}</td>')
        body.append(f'<tr><td>{_esc(row["key"])}</td><td>{row["frames"]}</td>'
                    f'{"".join(cells)}</tr>')

    headers = "".join(f'<th colspan="2">{label}</th>' for _k, label, _f, _h in metrics)
    subheads = "".join("<th>old</th><th>new</th>" for _ in metrics)

    top = max(max(r["before"]["psnr"], r["after"]["psnr"]) for r in rows)
    chart = []
    for row in rows:
        for side, cls in (("before", " dim"), ("after", " hero")):
            value = row[side]["psnr"]
            label = row["key"] if side == "before" else ""
            chart.append(f"""
        <div class="row{cls}">
          <span class="lbl">{_esc(label) or "&nbsp;"}</span>
          <div class="track"><div class="fill" style="width:{value / top * 100:.1f}%"></div></div>
          <span class="num{'' if side == 'after' else ' small'}">{value:.1f}</span>
        </div>""")

    return f"""
    <section>
      <div class="head">
        <h2>Every fixture the project owns</h2>
        <p class="lede">The six characterisation scenarios, the deep stack the artefact
        was reported on, and the veil fixture borrowed from the DCT suite. Both rules,
        same code, same machine. Nothing here is a spot check: this is the whole set.</p>
      </div>
      <div class="panel">
        <div class="legend">
          <span><i style="background:var(--bar-track);border:1px solid var(--bar-muted)"></i>choose-max</span>
          <span><i style="background:var(--accent)"></i>OpenFocus 1.19</span>
        </div>
        <div class="chart">{''.join(chart)}</div>
        <div class="axis" style="margin-top:12px">
          <span></span><span class="ticks"><span>0</span><span>{top / 4:.0f}</span>
          <span>{top / 2:.0f}</span><span>{top * 3 / 4:.0f}</span>
          <span>{top:.0f} dB</span></span><span></span>
        </div>
      </div>
      <div class="panel">
        <div class="scroller">
          <table>
            <caption class="note">Flat-field noise is the fused image's high-pass energy
            over the typical frame's, where every source is smooth: above 1 the fusion
            invented texture, below it the averaging recovered signal-to-noise.
            "Outside sources" is the worst pixel, in 8-bit levels, falling beyond the
            range its own frames span - a value no frame supports.</caption>
            <thead>
              <tr><th rowspan="2">Fixture</th><th rowspan="2">Frames</th>{headers}</tr>
              <tr>{subheads}</tr>
            </thead>
            <tbody>{''.join(body)}</tbody>
          </table>
        </div>
      </div>
    </section>"""


def _sweep_section(sweeps):
    from tests.visualize_fusion_quality import PARAM_LABELS

    blocks = []
    for sweep in sweeps:
        points = sweep["points"]
        rows = []
        for scene, metric, unit in (("deep_stack", "psnr", "dB"),
                                    ("fine_texture", "psnr", "dB")):
            top = max(p[scene][metric] for p in points) * 1.02
            bars = "".join(f"""
        <div class="row{' hero' if p['default'] else ''}">
          <span class="lbl">{_esc(p['label'])}</span>
          <div class="track"><div class="fill" style="width:{p[scene][metric] / top * 100:.1f}%"></div></div>
          <span class="num">{p[scene][metric]:.1f}</span>
        </div>""" for p in points)
            rows.append(f"""
      <div>
        <div class="legend"><span>{_esc(scene)} &middot; {unit}</span></div>
        <div class="chart">{bars}</div>
      </div>""")

        breaches = ", ".join(
            f"{p['label']} {p['deep_stack']['breach']}" for p in points)
        blocks.append(f"""
    <div class="sweep">
      <h3>{_esc(PARAM_LABELS.get(sweep['name'], sweep['name']))}
        <code style="font-family:var(--font-mono);font-size:12px;color:var(--ink-3)">
        {_esc(sweep['name'])}</code></h3>
      <p class="blurb">{_esc(sweep['blurb'])}</p>
      <div class="split">{''.join(rows)}</div>
      <p class="note">Worst pixel outside the source range on deep_stack, per value:
      {_esc(breaches)}. Highlighted bar is the shipped default.</p>
    </div>""")

    return f"""
    <section>
      <div class="head">
        <h2>What each control actually does</h2>
        <p class="lede">Every dial the panel exposes, swept over the values declared in
        <code>tests/fusion_registry.py</code> - the same declaration the test suite
        iterates, so this page cannot drift from what is tested. Two scenarios each,
        because these controls trade one against the other: <b>deep_stack</b> holds the
        artefact, <b>fine_texture</b> holds the detail a fix can cost.</p>
      </div>
      <div class="panel">{''.join(blocks)}</div>
    </section>"""


def _gpu_section(gpu):
    if "skipped" in gpu:
        return f"""
    <section>
      <div class="head">
        <h2>CPU against GPU</h2>
        <p class="lede">Not measured on the machine that built this page:
        {_esc(gpu['skipped'])}. <code>tests/test_pyramid_flat_field.py</code> runs the
        comparison wherever a device exists.</p>
      </div>
    </section>"""

    rows = "".join(f"""
      <tr><td>{_esc(r['label'])}</td><td>{r['agreement']:.1f} dB</td>
      <td>{r['worst']}</td><td>{r['cpu_psnr']:.2f}</td><td>{r['gpu_psnr']:.2f}</td>
      <td>{r['cpu_seconds']:.2f}s</td><td>{r['gpu_seconds']:.2f}s</td></tr>"""
                   for r in gpu["rows"])
    return f"""
    <section>
      <div class="head">
        <h2>CPU against GPU</h2>
        <p class="lede">The device path reimplements pyrDown, pyrUp, the pooling window
        and the noise percentile, so it has to be shown landing in the same place rather
        than assumed to. Agreement is the two results measured against each other; the
        suite's bar is 40&nbsp;dB.</p>
      </div>
      <div class="panel">
        <div class="scroller">
          <table>
            <caption class="note">Timings on a 320&times;320, 64-frame fixture, where the
            transfer cost is never amortised - they say nothing about a real render.</caption>
            <thead><tr><th>Setting</th><th>Agreement</th><th>Worst level</th>
            <th>CPU PSNR</th><th>GPU PSNR</th><th>CPU</th><th>GPU</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
    </section>"""


GUARDS = [
    ("tests/test_pyramid_flat_field.py",
     "The two artefacts, as assertions - and a guard on each guard that fails if "
     "the old rule stops producing the fault, so a test cannot quietly start "
     "passing for the wrong reason. Also checks every control changes the "
     "result, that selectivity 0 really is the plain average, and that a "
     "clipped-black backdrop does not upset the noise estimate."),
    ("tests/test_fusion_regression.py",
     "The quality ratchet. Every method's score on every scenario is recorded in "
     "fusion_quality_baseline.json and re-checked; it fails both on a drop and on "
     "a score far above the baseline, because a stale baseline protects a quality "
     "level the code has left behind."),
    ("tests/test_fusion_quality.py",
     "The shared contract every method satisfies: beats each single slice, "
     "reproduces the geometry it was given, is bit-depth honest, deterministic, "
     "and agrees with its GPU twin."),
    ("tests/test_bit_depth.py",
     "An 8-bit stack in, 8-bit out; a 16-bit stack in, 16-bit out, at real 16-bit "
     "precision rather than 8-bit values in a wide type."),
    ("tests/test_fusion_characteristics.py",
     "The claims the characteristics report makes about this method, as assertions."),
]


def render_html(cases, fixtures, sweeps, gpu, seconds):
    deep = next(r for r in fixtures if r["key"] == "deep_stack")
    veil = next(r for r in fixtures if r["key"] == "veil")
    worst_before = max(r["before"]["breach_worst"] for r in fixtures)
    worst_after = max(r["after"]["breach_worst"] for r in fixtures)
    gained = sum(1 for r in fixtures if r["after"]["psnr"] > r["before"]["psnr"])

    sweep_html = _sweep_section(sweeps) if sweeps else ""
    guards = "".join(f"<li><code>{_esc(path)}</code><br>{_esc(text)}</li>"
                     for path, text in GUARDS)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pyramid fusion - quality testing</title>
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">OpenFocus &middot; fusion quality</div>
    <h1>Does the reworked pyramid actually fix what was wrong with it?</h1>
    <p class="deck">The textbook Laplacian-pyramid rule copies the sharpest coefficient
    of every band and discards the rest. That is right where one frame is plainly
    sharper and wrong everywhere else - and everywhere else is most of a macro frame.
    This page measures both rules on the same fixtures, with the same code, and draws
    the evidence rather than asserting it. Item 19 of
    <code>docs/ALGORITHM_IMPROVEMENTS.md</code> is the written version.</p>
    <div class="chips">
      <span class="chip"><b>{len(fixtures)}</b> fixtures</span>
      <span class="chip"><b>{gained}</b> improved</span>
      <span class="chip">deep stack <b>+{deep['after']['psnr'] - deep['before']['psnr']:.1f} dB</b></span>
      <span class="chip">veil <b>+{veil['after']['psnr'] - veil['before']['psnr']:.1f} dB</b></span>
      <span class="chip">invented pixels <b>{worst_before} &rarr; {worst_after}</b> levels</span>
    </div>
  </header>

  {''.join(_artefact_section(c) for c in cases)}

  {_fixture_section(fixtures)}

  {sweep_html}

  {_gpu_section(gpu)}

  <section>
    <div class="head">
      <h2>What holds this in place</h2>
      <p class="lede">A measurement taken once is an anecdote. These run on every
      commit, and each one fails if the rework is undone.</p>
    </div>
    <div class="panel"><ul class="guards">{guards}</ul></div>
  </section>

  <footer>
    Generated {date.today().isoformat()} by <code>tests/visualize_pyramid_quality.py</code>
    in {seconds:.0f}s &middot; every number on this page was measured by the run that
    wrote it &middot; regenerate rather than edit.
  </footer>

</div>
<script>
document.querySelectorAll(".seg.switch").forEach(seg => {{
  seg.addEventListener("click", e => {{
    const btn = e.target.closest("button");
    if (!btn) return;
    const stage = document.getElementById(btn.dataset.target);
    seg.querySelectorAll("button").forEach(b =>
      b.setAttribute("aria-pressed", String(b === btn)));
    stage.querySelectorAll("img[data-layer]").forEach(img =>
      img.dataset.hidden = String(img.dataset.layer !== btn.dataset.layer));
    stage.querySelector(".stamp").textContent = btn.dataset.stamp;
  }});
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="Output HTML path")
    parser.add_argument("--out-dir", default=REPORTS_DIR, dest="out_dir",
                        help=f"Directory for the report (default: {REPORTS_DIR}/)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip the parameter sweeps, which are most of the runtime")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="Open the report in the default browser when done")
    args = parser.parse_args()

    started = time.perf_counter()

    deep_stack, deep_reference, _m = scenarios.deep_stack()
    veil_stack, veil_reference, veil_interior = _veil_stack()

    cases = [
        artefact_case(
            "filament", "Structure no frame had",
            "A background that is never sharp in any frame, drifting sideways as it "
            "defocuses. With nothing to choose between, choose-max lets grain pick the "
            "winner, and neighbouring pixels take their coefficients from frames that "
            "disagree - so collapsing the pyramid reconstructs values no frame ever "
            "held. Marked in red: every pixel outside the range its own sources span.",
            "Pixels outside the source range",
            deep_stack, deep_reference,
            lambda fused: envelope_breach(fused, deep_stack)[2] > 8,
            # Counted rather than measured: the fault is not one bad region but
            # hundreds of separate runs scattered over the background, and that
            # is what makes it read as filaments rather than as a stain.
            lambda fused: f"{_patch_count(envelope_breach(fused, deep_stack)[2] > 8)} "
                          "separate patches"),
        artefact_case(
            "veil", "Grain winning where nothing is sharp",
            "A stack whose far frames are focused in front of everything: a bright, "
            "grainy veil holding no detail at all. Photon noise grows with brightness, "
            "so in any region where nothing is in focus the veil carries more band "
            "energy than the dark frames do - purely as grain - and wins it. Marked in "
            "red: the smooth dark body rendered far brighter than the truth.",
            "Body taken from a veil frame",
            veil_stack, veil_reference,
            lambda fused: veil_mask(fused, veil_reference, veil_interior),
            lambda fused: f"{veil_share(fused, veil_reference, veil_interior):.1f}% of the body veiled",
            # The fault here is a whole region rather than a filament, so the
            # window has to be wide enough to hold the subject and its edge.
            crop_side=240),
    ]

    fixtures = fixture_table()
    sweeps = [] if args.quick else control_sweeps()
    gpu = gpu_agreement()

    elapsed = time.perf_counter() - started
    html = render_html(cases, fixtures, sweeps, gpu, elapsed)

    out = args.out
    if not out:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, "pyramid_quality_report.html")
    out = os.path.abspath(out)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB) in {elapsed:.0f}s")

    if args.open_browser:
        webbrowser.open(f"file:///{out.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
