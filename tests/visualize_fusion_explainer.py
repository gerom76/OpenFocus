"""
Build the illustrated explainer: how each fusion method actually works.

Writes docs/how_fusion_works.html - a self-contained page for someone with no
background in image processing. Where a figure shows a step inside a method, it
is computed here with the same OpenCV operations that method uses, so the
pictures track the real algorithms rather than being drawn by hand. Two figures
are lifted straight out of the running code: StackMFF-V4's focus map comes from
the network itself, and every "result" image is the method's genuine output.

Diagrams are inline SVG, so the page needs no scripts, no fonts and no network.

Examples:
    python tests/visualize_fusion_explainer.py --open
    python tests/visualize_fusion_explainer.py --out docs/how_fusion_works.html
"""

import argparse
import base64
import os
import sys
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_registry as reg
from tests import fusion_scenarios as sc
from tests.synthetic_stack import make_stack

OUT_DEFAULT = os.path.join("docs", "how_fusion_works.html")
SIZE = 300      # demo stack edge, px
SLICES = 3

# One hue per source frame, kept in this order everywhere a decision map appears.
# Teal / amber / violet stay distinguishable for the common colour-vision types.
FRAME_COLOURS = [(122, 127, 27), (58, 121, 209), (138, 75, 91)]  # BGR
FRAME_NAMES = ["Frame 1", "Frame 2", "Frame 3"]


def b64(img, quality=86):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def heat(gray, invert=False):
    """A single-channel map rendered as a warm ramp, normalised to its own range."""
    g = gray.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    norm = np.zeros_like(g) if hi - lo < 1e-6 else (g - lo) / (hi - lo)
    if invert:
        norm = 1.0 - norm
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def decision_image(indices, count):
    """Paint a per-pixel winner map with one flat colour per source frame."""
    out = np.zeros((*indices.shape, 3), dtype=np.uint8)
    for k in range(count):
        out[indices == k] = FRAME_COLOURS[k % len(FRAME_COLOURS)]
    return out


def grid_overlay(img, block):
    """Draw the block lattice a block-based method reasons over."""
    out = img.copy()
    for x in range(0, out.shape[1], block):
        cv2.line(out, (x, 0), (x, out.shape[0]), (255, 255, 255), 1)
    for y in range(0, out.shape[0], block):
        cv2.line(out, (0, y), (out.shape[1], y), (255, 255, 255), 1)
    return out


def amplify(a, b, gain=6):
    """Difference between two images, scaled up so a subtle change is visible."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) * gain
    return np.clip(diff, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-method figures, each mirroring that method's own steps
# ---------------------------------------------------------------------------

def figures_guided_filter(stack):
    """Base/detail split, the raw winner map, and the same map after refining."""
    grays = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) for s in stack]

    base = cv2.blur(stack[0], (31, 31))
    detail = np.clip(np.abs(stack[0].astype(np.float32) - base.astype(np.float32)) * 5,
                     0, 255).astype(np.uint8)

    # Sharpness proxy: blurred magnitude of the Laplacian, as in gff.py
    saliency = [cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1)),
                                 (0, 0), 5) for g in grays]
    raw = np.argmax(np.stack(saliency), axis=0)

    # Refinement: guided filtering pulls the decision onto real edges
    mask = (raw == 0).astype(np.float32)
    guide = grays[0] / 255.0
    r, eps, ksize = 45, 0.3, (91, 91)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(mask, cv2.CV_32F, ksize)
    cov = cv2.boxFilter(guide * mask, cv2.CV_32F, ksize) - mean_i * mean_p
    var = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize) - mean_i * mean_i
    a = cov / (var + eps)
    refined = cv2.boxFilter(a, cv2.CV_32F, ksize) * guide + \
        cv2.boxFilter(mean_p - a * mean_i, cv2.CV_32F, ksize)

    return [
        (b64(base), "Base layer", "The broad shapes and colour, with all fine detail removed."),
        (b64(detail), "Detail layer", "What was taken away: only the fine texture, brightened here to be visible."),
        (b64(decision_image(raw, len(stack))), "First guess",
         "For every pixel, which frame looked sharpest. Speckled, because a single pixel is weak evidence."),
        (b64(heat(refined)), "After tidying",
         "The same guess, smoothed while respecting edges in the picture, so decisions stop leaking across outlines."),
    ]


def figures_gfgfgf(stack):
    """A per-frame sharpness score, then the local contrast map it works from."""
    grays = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) for s in stack]
    scores = [float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean()) for g in grays]

    contrast = [np.abs(g - cv2.blur(g, (7, 7))) for g in grays]
    raw = np.argmax(np.stack(contrast), axis=0)

    return [
        (b64(heat(contrast[0])), "Local contrast",
         "How much each small neighbourhood of frame 1 differs from its own average. Bright means detail."),
        (b64(decision_image(raw, len(stack))), "Winner per pixel",
         "The frame with the most local contrast at each point."),
    ], scores


def focus_scores(stack):
    """The single per-frame sharpness number GFG-FGF shortlists on."""
    return [float(np.abs(cv2.Laplacian(
        cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)).mean())
        for s in stack]


def figures_dct(stack, block=8):
    """The block lattice, per-block sharpness, and the median-filter clean-up."""
    grays = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) for s in stack]
    h, w = grays[0].shape
    mh, mw = h // block, w // block

    variances = []
    for g in grays:
        mean = cv2.resize(g, (mw, mh), interpolation=cv2.INTER_AREA)
        mean_sq = cv2.resize(g ** 2, (mw, mh), interpolation=cv2.INTER_AREA)
        variances.append(mean_sq - mean ** 2)

    small = np.argmax(np.stack(variances), axis=0).astype(np.uint8)
    cleaned = cv2.medianBlur(small, 7)

    up = lambda m: cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return [
        (b64(grid_overlay(stack[0], block)), f"The {block}-pixel grid",
         "Sharpness is judged one square at a time, not one pixel at a time."),
        (b64(heat(cv2.resize(variances[0], (w, h), interpolation=cv2.INTER_NEAREST))),
         "Variation per block",
         "How much the pixels inside each square differ from each other. More variation is read as sharper."),
        (b64(decision_image(up(small), len(stack))), "Winning frame per block",
         "Blocky by nature - this is why edges between near and far can look stepped."),
        (b64(decision_image(up(cleaned), len(stack))), "After clean-up",
         "A median filter removes isolated wrong squares, at the cost of rounding off genuinely small details."),
    ]


def figures_dtcwt(stack, levels=3):
    """The coarse-to-fine pyramid: what the method looks at on each pass."""
    gray = cv2.cvtColor(stack[0], cv2.COLOR_BGR2GRAY)
    figures = []
    current = gray.astype(np.float32)
    for level in range(levels):
        blurred = cv2.GaussianBlur(current, (0, 0), 1.6)
        fine = np.abs(current - blurred)
        figures.append((
            b64(heat(cv2.resize(fine, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))),
            f"Level {level + 1}",
            ["The finest texture - individual pixels and thin lines.",
             "Medium structures - small shapes and thicker edges.",
             "Coarse structures - large forms and soft gradients."][level],
        ))
        current = cv2.resize(blurred, (current.shape[1] // 2, current.shape[0] // 2),
                             interpolation=cv2.INTER_AREA)
    return figures


def figures_stackmffv4(stack):
    """The network's own output: which frame it chose for every pixel."""
    try:
        import torch
        from fusion_methods.stackmffv4 import _get_model_and_device, _resize_to_multiple_of_32
    except Exception:
        return None

    path = os.path.join(reg.WEIGHTS_DIR, "stackmffv4.pth")
    if not os.path.isfile(path):
        return None

    try:
        model, device = _get_model_and_device(path, reg._have_torch_device())
        tensors = [torch.from_numpy(
            cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0) for s in stack]
        with torch.no_grad():
            batch = torch.stack(tensors).unsqueeze(0).to(device)
            resized, _ = _resize_to_multiple_of_32(batch)
            _, indices = model(resized)
            indices = indices.squeeze().cpu().numpy()
    except Exception:
        return None

    indices = cv2.resize(indices.astype(np.float32), (stack[0].shape[1], stack[0].shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return [(b64(decision_image(np.round(indices).astype(int), len(stack))),
             "The network's choice",
             "Produced by the trained model itself, not recreated here. Smooth regions and "
             "clean boundaries, without the speckle the hand-written methods have to filter out.")]


def figures_ifcnn(stack):
    """What the refinement stage changes, before and after."""
    gff = reg.get("guided_filter")
    ifcnn = reg.get("gff_ifcnn")
    if not (gff.available()[0] and ifcnn.available()[0]):
        return None
    before = gff.run(stack)
    after = ifcnn.run(stack)
    return [
        (b64(before), "Before refining", "The guided filter's result, on its own."),
        (b64(after), "After refining", "The same picture passed through IFCNN."),
        (b64(amplify(before, after, gain=6)), "What changed",
         "The difference between the two, multiplied six times so it can be seen. "
         "Edges are touched up - but the change covers everything, which is the colour shift."),
    ]


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

CSS = r"""
  :root {
    --ground:#eceff0; --surface:#fff; --surface-2:#f5f7f7; --line:#d3dadc;
    --line-strong:#b6c2c5; --ink:#0f1a1d; --ink-2:#46595e; --ink-3:#6f8388;
    --accent:#0b6f7a; --accent-soft:rgba(11,111,122,.10); --accent-ink:#fff;
    --f1:#1b7f79; --f2:#d1793a; --f3:#5b4b8a;
    --font-display:"Bahnschrift","DIN Alternate","Arial Narrow",system-ui,sans-serif;
    --font-body:"Charter","Bitstream Charter","Iowan Old Style",Georgia,serif;
    --font-mono:"Cascadia Mono","JetBrains Mono","SF Mono",Consolas,monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --ground:#0b1012; --surface:#131b1e; --surface-2:#182226; --line:#26343a;
      --line-strong:#35474e; --ink:#e7eef0; --ink-2:#a8bcc1; --ink-3:#7b9098;
      --accent:#3fb6c4; --accent-soft:rgba(63,182,196,.14); --accent-ink:#06272c;
      --f1:#3aa79f; --f2:#e0955a; --f3:#8b7ab8; }
  }
  :root[data-theme="dark"] {
    --ground:#0b1012; --surface:#131b1e; --surface-2:#182226; --line:#26343a;
    --line-strong:#35474e; --ink:#e7eef0; --ink-2:#a8bcc1; --ink-3:#7b9098;
    --accent:#3fb6c4; --accent-soft:rgba(63,182,196,.14); --accent-ink:#06272c;
    --f1:#3aa79f; --f2:#e0955a; --f3:#8b7ab8; }
  :root[data-theme="light"] {
    --ground:#eceff0; --surface:#fff; --surface-2:#f5f7f7; --line:#d3dadc;
    --line-strong:#b6c2c5; --ink:#0f1a1d; --ink-2:#46595e; --ink-3:#6f8388;
    --accent:#0b6f7a; --accent-soft:rgba(11,111,122,.10); --accent-ink:#fff;
    --f1:#1b7f79; --f2:#d1793a; --f3:#5b4b8a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
         font-family:var(--font-body); font-size:17px; line-height:1.68; }
  .wrap { max-width:1020px; margin:0 auto; padding:52px 24px 96px;
          display:flex; flex-direction:column; gap:64px; }
  .eyebrow { font-family:var(--font-mono); font-size:12px; letter-spacing:.09em;
             text-transform:uppercase; color:var(--accent); }
  h1 { font-family:var(--font-display); font-size:clamp(36px,6vw,60px); line-height:1.02;
       font-weight:600; letter-spacing:-.015em; text-wrap:balance; margin:14px 0 0; }
  .deck { max-width:58ch; color:var(--ink-2); font-size:19px; margin:20px 0 0; }
  h2 { font-family:var(--font-display); font-size:30px; font-weight:600;
       text-wrap:balance; margin:0; }
  h3 { font-family:var(--font-display); font-size:21px; font-weight:600; margin:0; }
  p { max-width:66ch; margin:14px 0 0; }
  p.tight { margin-top:8px; }
  section { display:flex; flex-direction:column; gap:26px; }
  .head { border-top:1px solid var(--line-strong); padding-top:24px; }
  .lede { color:var(--ink-2); }
  .method { background:var(--surface); border:1px solid var(--line); border-radius:5px;
            padding:28px 30px 30px; display:flex; flex-direction:column; gap:22px; }
  .method > .title { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  .tag { font-family:var(--font-mono); font-size:11px; letter-spacing:.07em;
         text-transform:uppercase; color:var(--accent); background:var(--accent-soft);
         padding:4px 9px; border-radius:2px; }
  .strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }
  figure { margin:0; display:flex; flex-direction:column; gap:7px; }
  figure img { width:100%; height:auto; display:block; border:1px solid var(--line);
               border-radius:3px; }
  figcaption { font-size:14px; color:var(--ink-3); line-height:1.5; }
  figcaption b { display:block; color:var(--ink); font-family:var(--font-display);
                 font-size:15px; font-weight:600; margin-bottom:2px; }
  .arrows { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  svg { display:block; max-width:100%; height:auto; }
  .diagram { background:var(--surface-2); border-radius:4px; padding:22px;
             overflow-x:auto; }
  .legend { display:flex; gap:18px; flex-wrap:wrap; font-size:14px; color:var(--ink-2); }
  .legend span { display:flex; align-items:center; gap:7px; }
  .swatch { width:13px; height:13px; border-radius:2px; flex:none; }
  .callout { background:var(--surface-2); border-left:3px solid var(--accent);
             border-radius:0 3px 3px 0; padding:16px 20px; }
  .callout p { margin:0; font-size:15.5px; color:var(--ink-2); }
  .callout b { color:var(--ink); }
  .scores { display:flex; flex-direction:column; gap:8px; max-width:520px; }
  .score-row { display:grid; grid-template-columns:92px minmax(0,1fr) 54px; gap:12px;
               align-items:center; font-size:14px; }
  .score-row .bar { background:var(--surface-2); border-radius:2px; height:18px; }
  .score-row .bar div { height:100%; background:var(--accent); border-radius:0 3px 3px 0; }
  .score-row .val { font-family:var(--font-mono); font-size:13px; color:var(--ink-2);
                    font-variant-numeric:tabular-nums; }
  .cutoff { display:grid; grid-template-columns:92px minmax(0,1fr) 54px; gap:12px; }
  .cutoff span { grid-column:2; position:relative; font-family:var(--font-mono);
                 font-size:11px; color:var(--f2); padding-left:15%; }
  .cutoff span::before { content:""; position:absolute; left:15%; top:-24px; bottom:14px;
                         border-left:2px dashed var(--f2); }
  .cut { border-color:var(--f2) !important; }
  footer { border-top:1px solid var(--line-strong); padding-top:22px;
           font-family:var(--font-mono); font-size:12px; color:var(--ink-3); line-height:1.9; }
  @media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
"""


def svg_pipeline():
    """The recipe every method follows, as one flow."""
    steps = [("Frames in", "several photos,\neach sharp somewhere"),
             ("Measure", "how sharp is\neach part of each?"),
             ("Decide", "which frame wins\nwhere?"),
             ("Blend", "stitch the winners\ntogether"),
             ("One photo", "sharp everywhere")]
    box_w, box_h, gap = 168, 88, 34
    width = len(steps) * box_w + (len(steps) - 1) * gap
    parts = []
    for i, (title, sub) in enumerate(steps):
        x = i * (box_w + gap)
        fill = "var(--accent)" if i in (0, len(steps) - 1) else "var(--surface)"
        ink = "var(--accent-ink)" if i in (0, len(steps) - 1) else "var(--ink)"
        sub_ink = "var(--accent-ink)" if i in (0, len(steps) - 1) else "var(--ink-3)"
        lines = "".join(
            f'<tspan x="{x + box_w / 2}" dy="{15 if j else 0}">{line}</tspan>'
            for j, line in enumerate(sub.split("\n")))
        parts.append(f"""
    <rect x="{x}" y="0" width="{box_w}" height="{box_h}" rx="4"
          style="fill:{fill};stroke:var(--line-strong)"/>
    <text x="{x + box_w / 2}" y="34" text-anchor="middle" font-size="17" font-weight="600"
          style="fill:{ink};font-family:var(--font-display)">{title}</text>
    <text x="{x + box_w / 2}" y="54" text-anchor="middle" style="fill:{sub_ink}"
          font-size="12.5">{lines}</text>""")
        if i < len(steps) - 1:
            ax = x + box_w + 6
            parts.append(f"""
    <path d="M{ax} {box_h / 2} L{ax + gap - 12} {box_h / 2}" style="stroke:var(--line-strong)"
          stroke-width="1.6"/>
    <path d="M{ax + gap - 16} {box_h / 2 - 4.5} L{ax + gap - 11} {box_h / 2} L{ax + gap - 16} {box_h / 2 + 4.5}"
          fill="none" style="stroke:var(--line-strong)" stroke-width="1.6"/>""")

    return (f'<svg viewBox="0 0 {width} {box_h}" width="{width}" role="img" '
            f'aria-label="Frames in, measure, decide, blend, one photo out">'
            f'{"".join(parts)}</svg>')


def svg_depth():
    """Why several photos are needed at all: one plane of focus at a time."""
    return """
    <svg viewBox="0 0 660 200" width="660" role="img"
         aria-label="A camera focuses at one distance at a time, so near and far cannot both be sharp">
      <rect x="16" y="78" width="54" height="44" rx="4" style="fill:var(--accent)"/>
      <circle cx="76" cy="100" r="13" style="fill:var(--accent)"/>
      <text x="43" y="146" text-anchor="middle" style="fill:var(--ink-2)" font-size="13">Camera</text>
      <g style="stroke:var(--line-strong)" stroke-width="1.4" fill="none">
        <path d="M90 100 L620 46"/><path d="M90 100 L620 154"/>
      </g>
      <g font-size="13" text-anchor="middle">
        <line x1="230" y1="34" x2="230" y2="166" style="stroke:var(--f1)" stroke-width="2.5"/>
        <text x="230" y="186" style="fill:var(--f1)">Near: sharp in frame 1</text>
        <line x1="400" y1="34" x2="400" y2="166" style="stroke:var(--f2)" stroke-width="2.5"
              stroke-dasharray="5 4"/>
        <text x="400" y="186" style="fill:var(--f2)">Middle: frame 2</text>
        <line x1="570" y1="34" x2="570" y2="166" style="stroke:var(--f3)" stroke-width="2.5"
              stroke-dasharray="2 4"/>
        <text x="570" y="186" style="fill:var(--f3)">Far: frame 3</text>
      </g>
      <text x="330" y="22" text-anchor="middle" style="fill:var(--ink-3)" font-size="13">
        Only one distance is truly in focus at a time
      </text>
    </svg>"""


def svg_pixel_vs_block():
    """The single clearest difference between the classical methods."""
    cells, out = 10, []
    for row in range(cells):
        for col in range(cells):
            near = (col + row) < 9
            out.append(f'<rect x="{col * 17}" y="{row * 17}" width="16" height="16" rx="1.5" '
                       f'style="fill:{"var(--f1)" if near else "var(--f2)"}" opacity=".85"/>')
    fine = "".join(out)

    coarse = []
    for row in range(0, cells, 2):
        for col in range(0, cells, 2):
            near = (col + row) < 8
            coarse.append(f'<rect x="{240 + col * 17}" y="{row * 17}" width="33" height="33" '
                          f'rx="2" style="fill:{"var(--f1)" if near else "var(--f2)"}" opacity=".85"/>')
    return f"""
    <svg viewBox="0 0 420 210" width="420" role="img"
         aria-label="Per-pixel decisions follow a diagonal edge smoothly; per-block decisions make it stepped">
      <g>{fine}</g>
      <text x="85" y="196" text-anchor="middle" style="fill:var(--ink-2)" font-size="13.5">
        Decided per pixel</text>
      <g>{"".join(coarse)}</g>
      <text x="325" y="196" text-anchor="middle" style="fill:var(--ink-2)" font-size="13.5">
        Decided per block</text>
    </svg>"""


def svg_scales():
    """Coarse-to-fine, as nested frames."""
    boxes = []
    for i, (size, label) in enumerate([(150, "fine"), (104, "medium"), (62, "coarse")]):
        off = (150 - size) / 2
        boxes.append(f'<rect x="{20 + off}" y="{20 + off}" width="{size}" height="{size}" '
                     f'rx="3" style="fill:none;stroke:var(--f{i + 1})" stroke-width="2.2"/>')
    return f"""
    <svg viewBox="0 0 460 200" width="460" role="img"
         aria-label="The picture is examined at fine, medium and coarse scales">
      {"".join(boxes)}
      <g font-size="13.5" style="fill:var(--ink-2)">
        <line x1="180" y1="45" x2="228" y2="45" style="stroke:var(--f1)" stroke-width="2"/>
        <text x="238" y="50">Fine: thin lines, grain</text>
        <line x1="180" y1="95" x2="228" y2="95" style="stroke:var(--f2)" stroke-width="2"/>
        <text x="238" y="100">Medium: small shapes</text>
        <line x1="180" y1="145" x2="228" y2="145" style="stroke:var(--f3)" stroke-width="2"/>
        <text x="238" y="150">Coarse: large forms</text>
      </g>
    </svg>"""


def strip(figures):
    return "".join(f"""
        <figure><img src="{src}" alt="{title}">
          <figcaption><b>{title}</b>{caption}</figcaption></figure>"""
                   for src, title, caption in figures)


def legend(count):
    names = FRAME_NAMES[:count]
    return ('<div class="legend">' + "".join(
        f'<span><i class="swatch" style="background:var(--f{i + 1})"></i>{n}</span>'
        for i, n in enumerate(names)) +
        '<span style="color:var(--ink-3)">&mdash; colours used in every decision map below</span></div>')


def build(out_path):
    stack, reference, _ = make_stack(num_slices=SLICES, height=SIZE, width=SIZE,
                                     seed=7, style="photographic")

    inputs = [(b64(s), FRAME_NAMES[i],
               ["Sharp across the top band, soft elsewhere.",
                "Sharp across the middle.",
                "Sharp across the bottom."][i]) for i, s in enumerate(stack)]

    gf = reg.get("guided_filter")
    result = gf.run(stack) if gf.available()[0] else reference
    outcome = [(b64(reference), "What we want", "Every part sharp at once - impossible in a single exposure."),
               (b64(result), "What fusion produces", "Built only from the frames on the left.")]

    print("  guided filter figures")
    gff_figs = figures_guided_filter(stack)
    print("  gfg-fgf figures")
    gfg_figs, gfg_scores = figures_gfgfgf(stack)
    print("  dct figures")
    dct_figs = figures_dct(stack)
    print("  dtcwt figures")
    dtcwt_figs = figures_dtcwt(stack)
    print("  stackmff figures")
    smff_figs = figures_stackmffv4(stack)
    print("  ifcnn figures")
    ifcnn_figs = figures_ifcnn(stack)

    def score_bars(scores, names):
        top = max(scores) if scores else 1.0
        rows = []
        for i, value in enumerate(scores):
            pct = value / top * 100
            dropped = pct < 15
            rows.append(f"""
          <div class="score-row"><span{' style="color:var(--f2)"' if dropped else ''}>{names[i]}</span>
            <div class="bar"><div style="width:{max(pct, 1):.0f}%{';background:var(--f2)' if dropped else ''}"></div></div>
            <span class="val"{' style="color:var(--f2)"' if dropped else ''}>{pct:.0f}%{' &times;' if dropped else ''}</span></div>""")
        return "".join(rows)

    score_rows = score_bars(gfg_scores, FRAME_NAMES)

    # The scenario where the shortlist rule actually fires: a small detailed
    # subject on a plain background pushes the background frame under the line
    edge_stack, _, _ = sc.build("depth_edge")
    edge_rows = score_bars(focus_scores(edge_stack), ["Subject frame", "Background frame"])

    smff_section = ""
    if smff_figs:
        smff_section = f"""
      <div class="method">
        <div class="title"><h3>StackMFF-V4</h3><span class="tag">learned</span></div>
        <p class="lede">
          Instead of a rule written by a person, this one was <b>shown thousands of focus
          stacks</b> during training along with the right answer for each, and adjusted itself
          until it could produce those answers. Nobody told it what sharpness looks like; it
          worked that out from examples.
        </p>
        <p>
          That is why it usually wins: recognising which frame is in focus is exactly the kind
          of judgement that is awkward to write as a formula but straightforward to learn from
          examples. The catch is the flip side - it is only reliable on the sort of pictures it
          was trained on. Ordinary photography is well covered; microscopy, medical imaging and
          other specialist material are not.
        </p>
        <div class="strip">{strip(smff_figs)}</div>
      </div>"""

    ifcnn_section = ""
    if ifcnn_figs:
        ifcnn_section = f"""
      <div class="method">
        <div class="title"><h3>IFCNN Refine</h3><span class="tag">optional extra pass</span></div>
        <p class="lede">
          This is not a fusion method - it is a <b>second pass that runs after one</b>. It takes
          the fused picture plus your original frames, and looks for detail the first pass lost,
          copying it back from whichever frame actually had it.
        </p>
        <p>
          The middle picture below is the guided filter's result after refining; on its own it
          looks much the same. The third picture is the difference between the two, multiplied
          six times. Notice the change is not confined to the edges it was meant to repair -
          it covers the whole frame. That spread is a colour shift, and it is the reason this
          stage scores below the plain guided filter in our measurements.
        </p>
        <div class="strip">{strip(ifcnn_figs)}</div>
      </div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How focus stacking works</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">OpenFocus &middot; a guide with no jargon</div>
    <h1>How focus stacking works</h1>
    <p class="deck">
      OpenFocus takes several photos of the same scene, each focused at a different distance,
      and builds one picture that is sharp all the way through. This page explains how &mdash;
      and how the five methods on the menu differ. No background required.
    </p>
  </header>

  <section>
    <div class="head"><h2>The problem</h2></div>
    <p class="lede">
      A camera lens can only focus at one distance at a time. Point it at something close and
      the background goes soft; focus on the background and the near object goes soft. For a
      landscape that hardly matters. For a close-up of an insect, a coin or a circuit board,
      it means no single photo can show the whole subject sharply.
    </p>
    <div class="diagram">{svg_depth()}</div>
    <p>
      The way round it is to take several photos, moving the focus a little each time, and then
      combine them &mdash; keeping, from each one, only the parts that came out sharp.
    </p>
    <div class="strip">{strip(inputs)}</div>
    <div class="strip">{strip(outcome)}</div>
  </section>

  <section>
    <div class="head">
      <h2>The recipe they all follow</h2>
      <p class="lede">
        Every method here does the same four things. They differ only in <em>how</em> they do
        the middle two &mdash; and that is where all the trade-offs come from.
      </p>
    </div>
    <div class="diagram">{svg_pipeline()}</div>
    <div class="callout">
      <p>
        <b>The hard part is "decide".</b> Sharpness has to be judged from the picture itself,
        with no knowledge of the scene. The usual clue is that a sharp region changes a lot from
        one pixel to the next, while a blurred one changes smoothly. That clue is good but not
        perfect &mdash; film grain also changes a lot from pixel to pixel, which is why noisy
        photos confuse some methods.
      </p>
    </div>
    {legend(SLICES)}
  </section>

  <section>
    <div class="head">
      <h2>Pixel by pixel, or block by block?</h2>
      <p class="lede">
        The single biggest difference between the classical methods is how finely they decide.
        Judging each pixel separately follows an outline exactly but is easily fooled by noise;
        judging a whole square at a time is steadier but leaves visible steps along a diagonal.
      </p>
    </div>
    <div class="diagram">{svg_pixel_vs_block()}</div>
  </section>

  <section>
    <div class="head">
      <h2>The methods, one at a time</h2>
      <p class="lede">
        The pictures in each section are real: they are produced by running that method's own
        steps on the three frames above, not drawn by hand.
      </p>
    </div>

    <div class="method">
      <div class="title"><h3>Guided Filter</h3><span class="tag">pixel by pixel</span></div>
      <p class="lede">
        It starts by splitting each frame in two: a <b>base layer</b> holding the broad shapes
        and colour, and a <b>detail layer</b> holding the fine texture. Sharpness lives entirely
        in the detail layer, so that is where the comparison happens.
      </p>
      <p>
        Comparing pixel by pixel gives a speckled, unreliable first guess &mdash; a single pixel
        is thin evidence. So the guess is smoothed. The clever part is that the smoothing is
        <b>guided by the photo itself</b>: it spreads decisions freely across a flat area, but
        refuses to spread them across a visible edge. Decisions therefore stop exactly where
        objects stop, instead of bleeding into the background.
      </p>
      <div class="strip">{strip(gff_figs)}</div>
    </div>

    <div class="method">
      <div class="title"><h3>GFG-FGF</h3><span class="tag">pixel by pixel, with a shortlist</span></div>
      <p class="lede">
        Closely related to the guided filter, and faster. It measures how much each small
        neighbourhood stands out from its surroundings, then refines the result the same
        edge-aware way.
      </p>
      <p>
        It adds one step the others do not have: before looking at any detail, it gives each
        frame <b>a single overall sharpness score</b>, and drops any frame scoring under 15% of
        the best one. The idea is to save time by ignoring hopelessly blurred frames.
      </p>
      <p class="tight"><b>A normal stack</b> &mdash; every frame comfortably above the line,
        so all three are used:</p>
      <div class="scores">{score_rows}<div class="cutoff"><span>15% cut-off</span></div></div>
      <div class="callout">
        <p>
          <b>This is worth knowing about.</b> If your subject is small and detailed against a
          plain background, the frame focused on the background can score below that 15% line
          and be thrown away &mdash; even though it held the only sharp version of the
          background. When that happens you get one of your original photos back, unchanged.
          We reproduce it in our tests, so it is not hypothetical.
        </p>
      </div>
      <p class="tight"><b>A small subject on a plain background</b> &mdash; the background frame
        falls under the line and is discarded, and the result is the other frame, unchanged:</p>
      <div class="scores">{edge_rows}<div class="cutoff"><span>15% cut-off</span></div></div>
      <div class="strip">{strip(gfg_figs)}</div>
    </div>

    <div class="method">
      <div class="title"><h3>DCT</h3><span class="tag">block by block</span></div>
      <p class="lede">
        This one chops each frame into small squares &mdash; eight pixels across by default
        &mdash; and asks a simple question of each: <b>how much do the pixels inside it differ
        from one another?</b> A sharp square contains strong contrast; a blurred square is
        comparatively uniform. The square with the most variation wins.
      </p>
      <p>
        Working on squares makes it by far the fastest method, and it needs no special
        libraries. The cost is that every decision lands on a grid, so a diagonal boundary
        between near and far comes out slightly stepped. A median filter cleans up isolated
        mistakes afterwards, which helps &mdash; though it also rounds off details genuinely
        smaller than a block.
      </p>
      <p>
        Its weak spot is grain: noise looks exactly like "pixels differing from one another",
        so on a high-ISO shot it can rate a noisy blurred square above a clean sharp one.
      </p>
      <div class="strip">{strip(dct_figs)}</div>
    </div>

    <div class="method">
      <div class="title"><h3>DTCWT</h3><span class="tag">coarse to fine</span></div>
      <p class="lede">
        Rather than choosing one scale to work at, this method looks at the picture at
        <b>several scales at once</b> &mdash; first the finest texture, then progressively
        larger structures &mdash; and in several directions at each scale.
      </p>
      <p>
        Because it can see both a thin line and the large soft shape it sits on, its decisions
        near a boundary between near and far are better informed than a method judging one
        small neighbourhood at a time. In our measurements it produces the cleanest edges of
        any non-neural method. The price is time: every extra level is more work.
      </p>
      <div class="diagram">{svg_scales()}</div>
      <div class="strip">{strip(dtcwt_figs)}</div>
    </div>

    {smff_section}
    {ifcnn_section}
  </section>

  <section>
    <div class="head">
      <h2>So which should you pick?</h2>
    </div>
    <p class="lede">
      For ordinary photography, <b>StackMFF-V4</b> needs no tuning and usually wins. If you want
      something predictable and hand-written, <b>DTCWT</b> was the strongest of those in our
      tests, with <b>Guided Filter</b> close behind and quicker. <b>DCT</b> is the one to reach
      for when speed matters more than perfect edges &mdash; but not on a noisy photo.
    </p>
    <p>
      For measurements rather than explanations, see <code>reports/fusion_method_guide.html</code>,
      which scores every method across six tests, and the per-parameter reports beside it.
    </p>
  </section>

  <footer>
    Generated by tests/visualize_fusion_explainer.py.
    Figures are produced by running each method's own steps on the three frames shown at the
    top; StackMFF-V4's map comes from the trained network itself.<br>
    The scenes are synthetic, with a known correct answer, which is what makes the comparisons
    on this page possible.
  </footer>

</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT, help=f"Output path (default: {OUT_DEFAULT})")
    ap.add_argument("--open", action="store_true", dest="open_browser",
                    help="Open the document in the default browser when done")
    args = ap.parse_args()

    print("Building figures...")
    html = build(args.out)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nWrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    if args.open_browser:
        webbrowser.open(f"file:///{out.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
