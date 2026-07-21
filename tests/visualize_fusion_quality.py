"""
Render fusion quality results as a standalone HTML report.

Covers every fusion method in the project through the uniform interface in
tests/fusion_registry.py. Two modes:

* sweep (default) - one method, swept across its tuning parameter (kernel size
  for the guided filter and GFG-FGF, block size for DCT, levels for DTCWT)
* --compare       - every available method side by side at its default setting

The output is a single self-contained HTML file: images are embedded as data
URIs, so it needs no server, no network and no assets folder.

Requires only what OpenFocus already needs: numpy and opencv-python. Methods
with unmet dependencies are skipped with a printed reason.

Examples:
    python tests/visualize_fusion_quality.py --synthetic --open
    python tests/visualize_fusion_quality.py --synthetic --method dtcwt --open
    python tests/visualize_fusion_quality.py --synthetic --compare --open
    python tests/visualize_fusion_quality.py path/to/stack --method dct --open
    python tests/visualize_fusion_quality.py --list
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
import time
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_metrics as fm
from tests import fusion_registry as reg
from tests.synthetic_stack import make_stack

DISPLAY_MAX = 512      # Longest edge of the embedded preview images
CROP_SCALE = 2         # Magnification of the detail insets
ERROR_FULL_SCALE = 48  # Abs error mapped to the top of the heatmap ramp

# How each sweepable parameter is presented in the report
PARAM_LABELS = {
    "kernel_size": "Kernel size",
    "block_size": "Block size",
    "N": "Levels",
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _fit(img, longest=DISPLAY_MAX):
    """Downscale for display only; fusion always runs at full resolution."""
    h, w = img.shape[:2]
    if max(h, w) <= longest:
        return img
    scale = longest / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _b64(img, quality=88):
    ok, buf = cv2.imencode(".jpg", _fit(img), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _crop(img, window):
    x, y, w, h = window
    patch = img[y:y + h, x:x + w]
    return cv2.resize(patch, (w * CROP_SCALE, h * CROP_SCALE),
                      interpolation=cv2.INTER_NEAREST)


def _error_map(fused, reference):
    """Absolute error against the ground truth as an inferno heatmap."""
    err = np.abs(fused.astype(np.float32) - reference.astype(np.float32)).mean(axis=2)
    err = np.clip(err / ERROR_FULL_SCALE * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(err, cv2.COLORMAP_INFERNO)


def load_stack(path):
    """Load every readable image in a directory, sorted numerically by filename."""
    paths = [p for p in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isfile(p)]
    try:
        paths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]))
    except IndexError:
        paths.sort()

    stack = [img for img in (cv2.imread(p) for p in paths) if img is not None]
    if not stack:
        raise SystemExit(f"No readable images found in {path}")
    return stack


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _plan(method_key, compare, values_override):
    """
    Work out what the report varies along its axis.

    Returns (runs, param_label, subtitle), each run being
    (id, label, fuse_callable).
    """
    if compare:
        methods = reg.available_methods()
        if not methods:
            raise SystemExit("No fusion method is available in this environment")
        runs = [(m.key, m.label, m.run) for m in methods]
        return runs, "Method", "every available method at its default setting"

    method = reg.get(method_key)
    ok, why = method.available()
    if not ok:
        raise SystemExit(f"{method.label} is unavailable: {why}")

    if method.sweep is None:
        return ([(method.key, method.label, method.run)], "Setting",
                f"{method.label}, which exposes no tuning parameter")

    param, values = method.sweep
    if values_override:
        values = values_override
    label = PARAM_LABELS.get(param, param)

    def make(value):
        return lambda stack, **kw: method.run(stack, **{param: value}, **kw)

    runs = [(str(v), str(v), make(v)) for v in values]
    return runs, label, f"{method.label}, swept across {label.lower()}"


def build_payload(stack, reference, method_key, compare, values_override, crop_window=None):
    """Fuse every planned run and collect the images and metrics the report shows."""
    height, width = stack[0].shape[:2]
    runs_plan, param_label, subtitle = _plan(method_key, compare, values_override)

    if crop_window is None:
        side = max(64, min(height, width) // 3)
        if reference is not None:
            # Straddle the first focus-band boundary, where fusion is hardest
            boundary = round(height / len(stack))
            y = max(0, boundary - side // 2)
        else:
            y = (height - side) // 2
        crop_window = ((width - side) // 2, y, side, side)

    payload = {
        "width": width,
        "height": height,
        "crop": list(crop_window),
        "has_reference": reference is not None,
        "param_label": param_label,
        "subtitle": subtitle,
        "compare": bool(compare),
        "slices": [],
        "runs": [],
        "reference": None,
    }

    if reference is not None:
        payload["reference"] = {
            "img": _b64(reference),
            "crop": _b64(_crop(reference, crop_window)),
            "spatial_frequency": fm.spatial_frequency(reference),
        }

    edges = np.linspace(0, height, len(stack) + 1).round().astype(int)
    for i, src in enumerate(stack):
        entry = {
            "index": i,
            "img": _b64(src),
            "crop": _b64(_crop(src, crop_window)),
            "spatial_frequency": fm.spatial_frequency(src),
        }
        if reference is not None:
            # Only synthetic stacks have a known in-focus band to draw
            entry["band_pct"] = [float(edges[i] / height * 100),
                                 float(edges[i + 1] / height * 100)]
            entry["psnr"] = fm.psnr(src, reference)
        payload["slices"].append(entry)

    for run_id, run_label, fuse in runs_plan:
        start = time.perf_counter()
        try:
            fused = fuse(stack)
        except Exception as exc:  # one broken method must not sink the whole report
            print(f"  {run_label:<30} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - start
        print(f"  {run_label:<30} {elapsed:6.2f}s", flush=True)

        entry = {
            "id": run_id,
            "label": run_label,
            "img": _b64(fused),
            "crop": _b64(_crop(fused, crop_window)),
            "seconds": elapsed,
            # Recorded per run: a method that alters the geometry it was given
            # misaligns every caller downstream, so the table shows it outright
            "size": f"{fused.shape[1]}x{fused.shape[0]}",
        }
        # DCT and DTCWT can return a slightly different geometry than they were
        # given, so measure on the region every image actually shares
        m_fused, m_stack, m_ref = fm.align_to_common_size(fused, stack, reference)
        entry.update(fm.evaluate(m_fused, m_stack, m_ref))
        if m_ref is not None:
            entry["diff"] = _b64(_error_map(m_fused, m_ref))
        payload["runs"].append(entry)

    if not payload["runs"]:
        raise SystemExit("Every fusion run failed; nothing to report")

    return payload


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

CSS = r"""
  :root {
    --ground:#eceff0; --surface:#fff; --surface-2:#f4f6f7; --line:#d3dadc;
    --line-strong:#b6c2c5; --ink:#0f1a1d; --ink-2:#46595e; --ink-3:#6f8388;
    --accent:#0b6f7a; --accent-soft:rgba(11,111,122,.12); --accent-ink:#fff;
    --bar-track:#dfe5e6; --bar-muted:#93a6ab;
    --font-display:"Bahnschrift","DIN Alternate","Arial Narrow",system-ui,sans-serif;
    --font-body:"Charter","Bitstream Charter","Iowan Old Style",Georgia,serif;
    --font-mono:"Cascadia Mono","JetBrains Mono","SF Mono",Consolas,monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground:#0b1012; --surface:#131b1e; --surface-2:#182226; --line:#26343a;
      --line-strong:#35474e; --ink:#e7eef0; --ink-2:#a8bcc1; --ink-3:#7b9098;
      --accent:#3fb6c4; --accent-soft:rgba(63,182,196,.16); --accent-ink:#06272c;
      --bar-track:#223034; --bar-muted:#566f76;
    }
  }
  :root[data-theme="dark"] {
    --ground:#0b1012; --surface:#131b1e; --surface-2:#182226; --line:#26343a;
    --line-strong:#35474e; --ink:#e7eef0; --ink-2:#a8bcc1; --ink-3:#7b9098;
    --accent:#3fb6c4; --accent-soft:rgba(63,182,196,.16); --accent-ink:#06272c;
    --bar-track:#223034; --bar-muted:#566f76;
  }
  :root[data-theme="light"] {
    --ground:#eceff0; --surface:#fff; --surface-2:#f4f6f7; --line:#d3dadc;
    --line-strong:#b6c2c5; --ink:#0f1a1d; --ink-2:#46595e; --ink-3:#6f8388;
    --accent:#0b6f7a; --accent-soft:rgba(11,111,122,.12); --accent-ink:#fff;
    --bar-track:#dfe5e6; --bar-muted:#93a6ab;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
         font-family:var(--font-body); font-size:16px; line-height:1.6;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1080px; margin:0 auto; padding:48px 24px 96px;
          display:flex; flex-direction:column; gap:64px; }
  .eyebrow { font-family:var(--font-mono); font-size:12px; letter-spacing:.09em;
             text-transform:uppercase; color:var(--accent); }
  h1 { font-family:var(--font-display); font-size:clamp(34px,5.2vw,54px);
       line-height:1.05; font-weight:600; letter-spacing:-.01em;
       text-wrap:balance; margin:12px 0 0; }
  .deck { max-width:62ch; color:var(--ink-2); font-size:17px; margin:16px 0 0; }
  h2 { font-family:var(--font-display); font-size:25px; font-weight:600;
       letter-spacing:.01em; text-wrap:balance; margin:0; }
  .lede { max-width:64ch; color:var(--ink-2); margin:10px 0 0; }
  section { display:flex; flex-direction:column; gap:22px; }
  .head { border-top:1px solid var(--line-strong); padding-top:20px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:24px; }
  .chip { font-family:var(--font-mono); font-size:12px; color:var(--ink-2);
          background:var(--surface); border:1px solid var(--line);
          border-radius:2px; padding:5px 10px; }
  .chip b { color:var(--ink); font-weight:600; }
  .grid { display:grid; gap:14px; }
  .card { background:var(--surface); border:1px solid var(--line);
          border-radius:3px; overflow:hidden; display:flex;
          flex-direction:column; margin:0; }
  .card.truth { border-color:var(--accent); }
  .frame { position:relative; line-height:0; }
  .frame img { width:100%; height:auto; display:block; }
  .strip img { image-rendering:pixelated; }
  .band { position:absolute; left:0; right:0; border-top:1px solid var(--accent);
          border-bottom:1px solid var(--accent); background:var(--accent-soft); }
  .band span { position:absolute; top:50%; left:6px; transform:translateY(-50%);
               font-family:var(--font-mono); font-size:10px; letter-spacing:.06em;
               text-transform:uppercase; color:var(--accent-ink);
               background:var(--accent); padding:2px 5px; border-radius:2px;
               line-height:1.4; }
  .cap { padding:10px 12px 12px; border-top:1px solid var(--line);
         display:flex; flex-direction:column; gap:3px; }
  .cap .name { font-family:var(--font-display); font-size:15px; letter-spacing:.01em; }
  .cap .val { font-family:var(--font-mono); font-size:12px; color:var(--ink-3);
              font-variant-numeric:tabular-nums; }
  .controls { display:flex; flex-wrap:wrap; gap:20px; align-items:flex-end; }
  .ctl { display:flex; flex-direction:column; gap:7px; }
  .ctl > span { font-family:var(--font-mono); font-size:11px; letter-spacing:.08em;
                text-transform:uppercase; color:var(--ink-3); }
  .seg { display:flex; flex-wrap:wrap; border:1px solid var(--line-strong);
         border-radius:3px; overflow:hidden; }
  .seg button { font-family:var(--font-mono); font-size:13px; color:var(--ink-2);
                background:var(--surface); border:0; border-right:1px solid var(--line);
                padding:8px 15px; cursor:pointer; font-variant-numeric:tabular-nums; }
  .seg button:last-child { border-right:0; }
  .seg button:hover { background:var(--surface-2); color:var(--ink); }
  .seg button[aria-pressed="true"] { background:var(--accent); color:var(--accent-ink); }
  .seg button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  .inspect { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(230px,1fr); gap:20px; }
  .stage { position:relative; background:var(--surface); border:1px solid var(--line);
           border-radius:3px; overflow:hidden; line-height:0; }
  .layers { display:grid; }
  .layers img { width:100%; height:auto; display:block; grid-area:1/1;
                transition:opacity .18s ease; }
  .layers img[data-hidden="true"] { opacity:0; }
  .stamp { position:absolute; top:10px; left:10px; font-family:var(--font-mono);
           font-size:11px; letter-spacing:.07em; text-transform:uppercase;
           background:var(--accent); color:var(--accent-ink); padding:4px 8px;
           border-radius:2px; line-height:1.4; }
  .readout { background:var(--surface); border:1px solid var(--line);
             border-radius:3px; padding:4px 16px 14px; display:flex;
             flex-direction:column; }
  .metric { display:flex; justify-content:space-between; align-items:baseline;
            gap:12px; padding:11px 0; border-bottom:1px solid var(--line); }
  .metric:last-child { border-bottom:0; }
  .metric .k { font-family:var(--font-mono); font-size:11px; letter-spacing:.07em;
               text-transform:uppercase; color:var(--ink-3); }
  .metric .v { font-family:var(--font-display); font-size:21px;
               font-variant-numeric:tabular-nums; color:var(--ink); }
  .metric .v small { font-size:12px; color:var(--ink-3); margin-left:3px; }
  .note { font-size:13px; color:var(--ink-3); max-width:64ch; }
  .panel { background:var(--surface); border:1px solid var(--line);
           border-radius:3px; padding:22px 24px 24px; }
  .chart { display:flex; flex-direction:column; gap:12px; margin-top:4px; }
  .row, .axis { display:grid; grid-template-columns:190px minmax(0,1fr) 74px;
                gap:14px; align-items:center; }
  .row .lbl { font-family:var(--font-mono); font-size:12px; color:var(--ink-2);
              text-align:right; overflow:hidden; text-overflow:ellipsis;
              white-space:nowrap; }
  .track { background:var(--bar-track); border-radius:2px; height:20px; overflow:hidden; }
  .fill { height:100%; background:var(--bar-muted); border-radius:0 4px 4px 0;
          transition:width .5s cubic-bezier(.22,.7,.3,1); }
  .row.hero .fill { background:var(--accent); }
  .row.hero .lbl, .row.hero .num { color:var(--ink); font-weight:700; }
  .row .num { font-family:var(--font-mono); font-size:13px;
              font-variant-numeric:tabular-nums; color:var(--ink-2); }
  .axis { font-family:var(--font-mono); font-size:11px; color:var(--ink-3); }
  .axis .ticks { display:flex; justify-content:space-between; }
  .scroller { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-family:var(--font-mono); font-size:13px; }
  th, td { text-align:right; padding:9px 12px; border-bottom:1px solid var(--line);
           font-variant-numeric:tabular-nums; white-space:nowrap; }
  th { font-size:11px; letter-spacing:.07em; text-transform:uppercase;
       color:var(--ink-3); font-weight:400; }
  th:first-child, td:first-child { text-align:left; }
  tbody tr:last-child td { border-bottom:0; }
  td.best { color:var(--accent); font-weight:700; }
  td.warn { color:#b4462a; font-weight:700; }
  @media (prefers-color-scheme: dark) { td.warn { color:#f0846a; } }
  :root[data-theme="dark"] td.warn { color:#f0846a; }
  :root[data-theme="light"] td.warn { color:#b4462a; }
  caption { caption-side:bottom; text-align:left; padding-top:12px; }
  footer { border-top:1px solid var(--line-strong); padding-top:20px;
           font-family:var(--font-mono); font-size:12px; color:var(--ink-3); }
  @media (max-width:860px) {
    .inspect { grid-template-columns:1fr; }
    .row, .axis { grid-template-columns:110px minmax(0,1fr) 66px; gap:10px; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _slice_cards(payload):
    cards = []
    for s in payload["slices"]:
        band = ""
        if "band_pct" in s:
            top, bottom = s["band_pct"]
            band = (f'<div class="band" style="top:{top:.2f}%;height:{bottom - top:.2f}%">'
                    f'<span>in focus</span></div>')
        detail = (f"{s['psnr']:.2f} dB &middot; SF {s['spatial_frequency']:.1f}"
                  if "psnr" in s else f"SF {s['spatial_frequency']:.1f}")
        cards.append(f"""
      <figure class="card">
        <div class="frame"><img src="{s['img']}" alt="Focus stack slice {s['index'] + 1}">{band}</div>
        <figcaption class="cap"><span class="name">Slice {s['index'] + 1}</span>
        <span class="val">{detail}</span></figcaption>
      </figure>""")

    if payload["reference"]:
        ref = payload["reference"]
        cards.append(f"""
      <figure class="card truth">
        <div class="frame"><img src="{ref['img']}" alt="All-in-focus ground truth"></div>
        <figcaption class="cap"><span class="name">Ground truth</span>
        <span class="val">&infin; dB &middot; SF {ref['spatial_frequency']:.1f}</span></figcaption>
      </figure>""")
    return "".join(cards)


def _crop_cells(payload, default_run):
    cells = [f"""
      <figure class="card">
        <div class="frame"><img src="{s['crop']}" alt="Detail crop, slice {s['index'] + 1}"></div>
        <figcaption class="cap"><span class="name">Slice {s['index'] + 1}</span></figcaption>
      </figure>""" for s in payload["slices"]]

    cells.append(f"""
      <figure class="card truth">
        <div class="frame"><img src="{default_run['crop']}" alt="Detail crop, fused"></div>
        <figcaption class="cap"><span class="name">Fused</span>
        <span class="val">{_esc(default_run['label'])}</span></figcaption>
      </figure>""")

    if payload["reference"]:
        cells.append(f"""
      <figure class="card">
        <div class="frame"><img src="{payload['reference']['crop']}" alt="Detail crop, ground truth"></div>
        <figcaption class="cap"><span class="name">Ground truth</span></figcaption>
      </figure>""")
    return "".join(cells)


def _axis_max(payload):
    if not payload["has_reference"]:
        return 1.0
    return max(42.0, max(r["psnr"] for r in payload["runs"]) * 1.05)


def _chart(payload, default_run):
    """
    With a ground truth, chart reconstruction accuracy per slice against the
    fused result. Without one, chart Q_ABF across the runs instead.
    """
    if payload["has_reference"]:
        top = _axis_max(payload)
        rows = [f"""
        <div class="row">
          <span class="lbl">Slice {s['index'] + 1}</span>
          <div class="track"><div class="fill" style="width:{s['psnr'] / top * 100:.1f}%"></div></div>
          <span class="num">{s['psnr']:.2f}</span>
        </div>""" for s in payload["slices"]]
        rows.append(f"""
        <div class="row hero">
          <span class="lbl">{_esc(default_run['label'])}</span>
          <div class="track"><div class="fill" style="width:{default_run['psnr'] / top * 100:.1f}%"></div></div>
          <span class="num">{default_run['psnr']:.2f}</span>
        </div>""")
        ticks = "".join(f"<span>{v}</span>" for v in
                        [0, round(top / 4), round(top / 2), round(top * 3 / 4),
                         f"{round(top)} dB"])
        heading = "Fused against the individual slices"
        title = ("Reconstruction accuracy versus the ground truth, in decibels. The test "
                 "requires only that fusion beat every slice by 3&nbsp;dB.")
    else:
        best = max(r["qabf"] for r in payload["runs"])
        rows = []
        for r in payload["runs"]:
            hero = " hero" if r["qabf"] == best else ""
            rows.append(f"""
        <div class="row{hero}">
          <span class="lbl">{_esc(r['label'])}</span>
          <div class="track"><div class="fill" style="width:{r['qabf'] * 100:.1f}%"></div></div>
          <span class="num">{r['qabf']:.4f}</span>
        </div>""")
        ticks = "".join(f"<span>{v}</span>" for v in ["0", "0.25", "0.50", "0.75", "1.0"])
        heading = "Which run retains the most detail"
        title = ("Edge information retained from the source stack. No ground truth exists "
                 "for a real stack, so Q<sup>AB/F</sup> is the deciding metric.")

    return f"""
    <div class="head">
      <h2>{heading}</h2>
      <p class="lede">{title}</p>
    </div>
    <div class="panel">
      <div class="chart" id="chart">{''.join(rows)}</div>
      <div class="axis" style="margin-top:12px">
        <span></span><span class="ticks">{ticks}</span><span></span>
      </div>
    </div>"""


def _table(payload):
    keys = [("qabf", "Q<sup>AB/F</sup>", "{:.4f}")]
    if payload["has_reference"]:
        keys += [("psnr", "PSNR", "{:.2f}"), ("ssim", "SSIM", "{:.4f}")]
    keys += [("spatial_frequency", "Spatial freq.", "{:.2f}"),
             ("entropy", "Entropy", "{:.3f}")]

    bests = {key: max(r[key] for r in payload["runs"]) for key, _, _ in keys}

    expected = f"{payload['width']}x{payload['height']}"
    rows = []
    for r in payload["runs"]:
        cells = "".join(
            "<td{}>{}</td>".format(' class="best"' if r[key] == bests[key] else "",
                                   fmt.format(r[key]))
            for key, _, fmt in keys)
        size = r.get("size", expected)
        size_cell = (f"<td>{size}</td>" if size == expected
                     else f'<td class="warn">{size}</td>')
        rows.append(f"<tr><td>{_esc(r['label'])}</td>{size_cell}{cells}"
                    f"<td>{r['seconds']:.2f}s</td></tr>")

    headers = "".join(f"<th>{label}</th>" for _, label, _ in keys)

    if payload["compare"]:
        caveat = ("Times are not comparable across rows: the neural methods include model "
                  "load on first use, and the GPU variants pay a transfer cost a small "
                  "test image never amortises.")
    elif payload["has_reference"]:
        caveat = ("On a synthetic stack the focus regions are large horizontal bands, "
                  "which flatters wide kernels. Point this script at a real stack to tune "
                  "the setting for your own images.")
    else:
        caveat = "The best-scoring run here is the setting to use in OpenFocus."

    return f"""
    <div class="panel">
      <div class="scroller">
        <table>
          <caption class="note">{caveat}</caption>
          <thead><tr><th>{_esc(payload['param_label'])}</th><th>Output</th>{headers}<th>Time</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>"""


def render_html(payload, source_label, title):
    runs = payload["runs"]
    # Mid-sweep is the representative setting; in compare mode lead with the first method
    default_run = runs[0] if payload["compare"] else runs[len(runs) // 2]
    has_ref = payload["has_reference"]

    view_buttons = '<button type="button" data-view="fused" aria-pressed="true">Fused</button>'
    if has_ref:
        view_buttons += (
            '<button type="button" data-view="truth" aria-pressed="false">Ground truth</button>'
            '<button type="button" data-view="diff" aria-pressed="false">Error map</button>')

    truth_layer = ""
    if has_ref:
        truth_layer = (
            f'<img id="layer-truth" data-hidden="true" alt="Ground truth" '
            f'src="{payload["reference"]["img"]}">'
            f'<img id="layer-diff" data-hidden="true" alt="Error against ground truth" '
            f'src="{default_run.get("diff", "")}">')

    ref_metrics = ""
    if has_ref:
        ref_metrics = """
          <div class="metric"><span class="k">PSNR</span><span class="v" id="m-psnr">&mdash;</span></div>
          <div class="metric"><span class="k">SSIM</span><span class="v" id="m-ssim">&mdash;</span></div>"""

    run_buttons = "".join(
        f'<button type="button" data-run="{_esc(r["id"])}" '
        f'aria-pressed="{"true" if r is default_run else "false"}">{_esc(r["label"])}</button>'
        for r in runs)

    # The script only drives the inspector; other images are inline in the markup
    script_data = json.dumps({
        "runs": [{key: r[key] for key in
                  ("id", "label", "img", "diff", "seconds", "qabf", "psnr",
                   "ssim", "spatial_frequency") if key in r}
                 for r in runs],
        "defaultId": default_run["id"],
        "hasReference": has_ref,
    })

    slice_cols = min(4, len(payload["slices"]) + (1 if has_ref else 0))
    crop_cols = min(5, len(payload["slices"]) + (2 if has_ref else 1))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">OpenFocus &middot; fusion quality</div>
    <h1>{_esc(title)}</h1>
    <p class="deck">
      Every image below is a real fusion result, not an illustration. This report covers
      {_esc(payload['subtitle'])}, measured against {'its known all-in-focus ground truth'
      if has_ref else 'the source slices themselves'}.
    </p>
    <div class="chips">
      <span class="chip">{_esc(source_label)}</span>
      <span class="chip"><b>{len(payload['slices'])}</b> slices</span>
      <span class="chip"><b>{payload['width']}&times;{payload['height']}</b> px</span>
      <span class="chip">{_esc(payload['param_label'])}
        <b>{_esc(', '.join(r['label'] for r in runs))}</b></span>
    </div>
  </header>

  <section>
    <div class="head">
      <h2>The stack going in</h2>
      <p class="lede">{'Each slice is sharp on one horizontal band and blurred elsewhere, so no single input is usable alone. The ground truth is the image every slice was derived from.' if has_ref else 'The source slices, each focused at a different depth.'}</p>
    </div>
    <div class="grid" style="grid-template-columns:repeat({slice_cols},1fr)">{_slice_cards(payload)}</div>
  </section>

  <section>
    <div class="head">
      <h2>The fused result</h2>
      <p class="lede">
        Switch between runs to compare them at full size.{' Use the view toggle to check against the ground truth, or read the error map - brighter means further from truth.' if has_ref else ''}
      </p>
    </div>

    <div class="controls">
      <div class="ctl">
        <span id="run-label">{_esc(payload['param_label'])}</span>
        <div class="seg" role="group" aria-labelledby="run-label" id="run-seg">{run_buttons}</div>
      </div>
      <div class="ctl">
        <span id="view-label">View</span>
        <div class="seg" role="group" aria-labelledby="view-label" id="view-seg">{view_buttons}</div>
      </div>
    </div>

    <div class="inspect">
      <div class="stage">
        <div class="layers">
          <img id="layer-fused" alt="Fused output" src="{default_run['img']}">{truth_layer}
        </div>
        <div class="stamp" id="stamp">Fused</div>
      </div>
      <div class="readout">
          <div class="metric"><span class="k">Q<sup>AB/F</sup></span><span class="v" id="m-qabf">&mdash;</span></div>{ref_metrics}
          <div class="metric"><span class="k">Spatial freq.</span><span class="v" id="m-sf">&mdash;</span></div>
          <div class="metric"><span class="k">Fuse time</span><span class="v" id="m-time">&mdash;</span></div>
      </div>
    </div>

    <p class="note">
      Q<sup>AB/F</sup> measures how much source edge information survived and needs no ground
      truth, which makes it the metric that carries over to real stacks.
    </p>
  </section>

  <section>
    <div class="head">
      <h2>Same crop, every image</h2>
      <p class="lede">
        A {payload['crop'][2]}&times;{payload['crop'][3]} px window magnified {CROP_SCALE}&times; with no interpolation,
        taken at the same coordinates in every frame.
      </p>
    </div>
    <div class="grid strip" style="grid-template-columns:repeat({crop_cols},1fr)">{_crop_cells(payload, default_run)}</div>
  </section>

  <section>{_chart(payload, default_run)}{_table(payload)}</section>

  <footer>
    Generated by tests/visualize_fusion_quality.py &middot; OpenCV {cv2.__version__} &middot; Python {sys.version.split()[0]}
  </footer>

</div>

<script>
const DATA = {script_data};
const AXIS_MAX = {_axis_max(payload):.2f};
const byId = Object.fromEntries(DATA.runs.map(r => [r.id, r]));
let current = DATA.defaultId;
let view = "fused";

const layers = {{}};
for (const name of ["fused", "truth", "diff"]) {{
  const el = document.getElementById("layer-" + name);
  if (el) layers[name] = el;
}}
const stamp = document.getElementById("stamp");
const viewNames = {{ fused: "Fused", truth: "Ground truth", diff: "Error map" }};

function set(id, text) {{
  const el = document.getElementById(id);
  if (el) el.innerHTML = text;
}}

function render() {{
  const r = byId[current];
  layers.fused.src = r.img;
  if (layers.diff && r.diff) layers.diff.src = r.diff;

  for (const [name, el] of Object.entries(layers)) {{
    el.dataset.hidden = name === view ? "false" : "true";
  }}
  stamp.textContent = view === "truth"
    ? "Ground truth"
    : viewNames[view] + " \\u00b7 " + r.label;

  set("m-qabf", r.qabf.toFixed(4));
  set("m-sf", r.spatial_frequency.toFixed(2));
  set("m-time", r.seconds.toFixed(2) + "<small>s</small>");
  if (DATA.hasReference) {{
    set("m-psnr", r.psnr.toFixed(2) + "<small>dB</small>");
    set("m-ssim", r.ssim.toFixed(4));
  }}

  document.querySelectorAll("#run-seg button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.run === current)));
  document.querySelectorAll("#view-seg button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.view === view)));

  const hero = document.querySelector(".row.hero");
  if (hero && DATA.hasReference) {{
    hero.querySelector(".fill").style.width = (r.psnr / AXIS_MAX * 100).toFixed(1) + "%";
    hero.querySelector(".num").textContent = r.psnr.toFixed(2);
    hero.querySelector(".lbl").textContent = r.label;
  }}
}}

document.getElementById("run-seg").addEventListener("click", e => {{
  const btn = e.target.closest("button");
  if (btn) {{ current = btn.dataset.run; render(); }}
}});
document.getElementById("view-seg").addEventListener("click", e => {{
  const btn = e.target.closest("button");
  if (btn) {{ view = btn.dataset.view; render(); }}
}});
render();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_methods():
    print(f"{'key':<20}{'status':<9}{'sweep':<28}reason")
    for m in reg.METHODS:
        ok, why = m.available()
        sweep = f"{m.sweep[0]}={m.sweep[1]}" if m.sweep else "-"
        print(f"{m.key:<20}{'ready' if ok else 'skip':<9}{sweep:<28}{'' if ok else why}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", nargs="?",
                    help="Directory of focus-stack images (omit with --synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate a stack with a known all-in-focus reference")
    ap.add_argument("--method", default="guided_filter",
                    help="Fusion method key; see --list")
    ap.add_argument("--compare", action="store_true",
                    help="Compare every available method instead of sweeping one")
    ap.add_argument("--list", action="store_true", dest="list_methods",
                    help="List registered methods and their availability, then exit")
    ap.add_argument("--slices", type=int, default=3, help="Slices for --synthetic")
    ap.add_argument("--size", type=int, default=384, help="Edge length for --synthetic")
    ap.add_argument("--style", default="photographic", choices=("photographic", "texture"),
                    help="Synthetic reference style (default: photographic)")
    ap.add_argument("--values", default=None,
                    help="Comma-separated override for the swept parameter")
    ap.add_argument("--out", default=None, help="Output HTML path")
    ap.add_argument("--open", action="store_true", dest="open_browser",
                    help="Open the report in the default browser when done")
    args = ap.parse_args()

    if args.list_methods:
        _list_methods()
        return

    if not args.synthetic and not args.stack:
        ap.error("provide a stack directory or pass --synthetic")

    values = [int(v) for v in args.values.split(",") if v.strip()] if args.values else None

    if args.synthetic:
        stack, reference, _ = make_stack(num_slices=args.slices, height=args.size,
                                         width=args.size, seed=7, style=args.style)
        source_label = f"synthetic stack ({args.style})"
    else:
        stack = load_stack(args.stack)
        reference = None
        source_label = os.path.basename(os.path.abspath(args.stack))

    title = ("Fusion methods compared" if args.compare
             else f"{reg.get(args.method).label} fusion quality")

    print(f"Fusing {len(stack)} slices at {stack[0].shape[1]}x{stack[0].shape[0]}...")
    payload = build_payload(stack, reference, args.method, args.compare, values)

    out = args.out or ("fusion_comparison_report.html" if args.compare
                       else f"{args.method}_quality_report.html")
    out = os.path.abspath(out)

    with open(out, "w", encoding="utf-8") as f:
        f.write(render_html(payload, source_label, title))

    print(f"\nWrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    if args.open_browser:
        webbrowser.open(f"file:///{out.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
