"""Regenerate the measured figures inside how-depthmap-works.html.

Nothing in that document's three data figures is drawn by eye:

  * ``leak``      - the real two-scale pooling of a 1-D energy signal taken
                    across an occlusion boundary, at the kernel the artefact
                    was reported at (51), using the shipped ``_near_window``.
  * ``depthmap``  - a real per-pixel argmax over a synthetic 24-frame stack
                    whose background no frame resolves, put through the
                    shipped ``_regularise_index``.
  * ``baseline``  - the closed form of MODE_AVERAGE's baseline blend.

Each figure is written into the HTML between its own marker pair::

    <!-- FIG-BEGIN leak -->  ...generated markup...  <!-- FIG-END leak -->

so re-running this script updates the document in place and touches nothing
else. Run it from anywhere:

    python docs/algorithms/make_depthmap_figures.py

Companion to make_vocabulary_figures.py, which does the same job for
quality-metrics-vocabulary.html.
"""

import base64
import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
TARGET = os.path.join(HERE, "how-depthmap-works.html")

sys.path.insert(0, REPO)
from fusion_methods.depthmap import _near_window, _regularise_index  # noqa: E402

# The document's palette, so the figures cannot drift from the page they sit
# on. s1/s2 are its categorical pair (validated for CVD separation); RAMP is a
# single-hue sequential scale, light to dark, for the depth index - a depth map
# is a magnitude, so it never gets a categorical or rainbow scale.
INK, INK2, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
S1_INK, S2_INK = "#1c5cab", "#b8420f"
RAMP = ["#eef4fc", "#d2e3f7", "#b0cdef", "#8bb5e5", "#669ada",
        "#4480cd", "#2f68b6", "#255496", "#1c4076", "#132d56"]

FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


# --------------------------------------------------------------- helpers
def box1d(sig, window):
    """Box mean with reflected borders - the 1-D twin of cv2.boxFilter."""
    pad = window // 2
    padded = np.pad(sig, pad, mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def poly(xs, ys, box, xlim, ylim):
    """Map data to an SVG polyline point string inside (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    xmin, xmax = xlim
    ymin, ymax = ylim
    out = []
    for x, y in zip(xs, ys):
        px = x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
        py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
        out.append(f"{px:.1f},{py:.1f}")
    return " ".join(out)


# ------------------------------------------------- figure 1: the leak
def figure_leak():
    """Why a single pooling window rings every subject, and what fixes it."""
    N, BOUND = 256, 128
    x = np.arange(N, dtype=np.float64)

    # Raw squared-Laplacian energy along a slice crossing an occlusion.
    # The frame holding the foreground: a sharply focused, high-contrast
    # contour right at the boundary, and its own background defocused.
    a = np.full(N, 0.40)
    a[:BOUND - 4] = 3.0
    a[BOUND - 4:BOUND] = 120.0
    # The frame holding the background: ordinary texture behind the boundary,
    # the foreground defocused.
    b = np.full(N, 0.60)
    b[BOUND:] = 8.0

    wide = 51                       # the kernel the artefact was reported at
    near = _near_window(wide)       # what the shipped rule picks for it

    aw, bw = box1d(a, wide), box1d(b, wide)
    an, bn = box1d(a, near), box1d(b, near)
    ag = np.sqrt(np.maximum(aw * an, 0))
    bg = np.sqrt(np.maximum(bw * bn, 0))

    band_w = np.where((x > BOUND) & (aw > bw))[0]
    band_g = np.where((x > BOUND) & (ag > bg))[0]

    PANEL_W, PANEL_H = 186, 76
    X0, Y0 = 30, 16
    box = (X0, X0 + PANEL_W, Y0, Y0 + PANEL_H)
    lo, hi = 84, 208
    sel = slice(lo, hi)

    def px(v):
        return X0 + (v - lo) / (hi - lo) * PANEL_W

    def panel(dx, ca, cb, band, title, sub, colour):
        scale = max(ca[sel].max(), cb[sel].max())
        pa = poly(x[sel], ca[sel] / scale, box, (lo, hi), (0, 1.08))
        pb = poly(x[sel], cb[sel] / scale, box, (lo, hi), (0, 1.08))
        bx0, bx1 = px(BOUND), px(band.max() + 1) if band.size else px(BOUND)
        width = int(band.size)
        s = [f'<g transform="translate({dx},0)">']
        # Title and qualifier on one left-anchored line: right-anchoring the
        # qualifier collides with the title as soon as either grows.
        s.append(f'<text x="{X0}" y="9" font-size="8.4" font-weight="700" '
                 f'fill="{INK}" {FONT}>{title}'
                 f'<tspan font-size="7" font-weight="400" fill="{MUTED}">'
                 f'  — {sub}</tspan></text>')
        # the band the wrong frame wins
        if width:
            s.append(f'<rect x="{bx0:.1f}" y="{Y0}" width="{bx1 - bx0:.1f}" '
                     f'height="{PANEL_H}" fill="{colour}" fill-opacity="0.13"/>')
            s.append(f'<line x1="{bx1:.1f}" y1="{Y0}" x2="{bx1:.1f}" '
                     f'y2="{Y0 + PANEL_H}" stroke="{colour}" stroke-width="1" '
                     f'stroke-dasharray="2 2"/>')
        s.append(f'<line x1="{X0}" y1="{Y0 + PANEL_H}" x2="{X0 + PANEL_W}" '
                 f'y2="{Y0 + PANEL_H}" stroke="{RULE}" stroke-width="1"/>')
        # the occlusion boundary itself
        s.append(f'<line x1="{px(BOUND):.1f}" y1="{Y0}" x2="{px(BOUND):.1f}" '
                 f'y2="{Y0 + PANEL_H + 3}" stroke="{INK}" stroke-width="1"/>')
        # ...named below the axis, where nothing else is competing for the room
        s.append(f'<text x="{px(BOUND) - 3:.1f}" y="{Y0 + PANEL_H + 10}" '
                 f'font-size="6.6" fill="{MUTED}" text-anchor="end" {FONT}>'
                 f'foreground ◀</text>')
        s.append(f'<text x="{px(BOUND) + 3:.1f}" y="{Y0 + PANEL_H + 10}" '
                 f'font-size="6.6" fill="{MUTED}" {FONT}>▶ background</text>')
        s.append(f'<polyline points="{pb}" fill="none" stroke="{S2}" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
        s.append(f'<polyline points="{pa}" fill="none" stroke="{S1}" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
        label = (f"{width} px of background taken from the foreground’s frame"
                 if width else "no band")
        s.append(f'<text x="{X0}" y="{Y0 + PANEL_H + 22}" font-size="6.6" '
                 f'fill="{colour}" font-weight="600" {FONT}>'
                 f'▶ {label}</text>')
        s.append('</g>')
        return "".join(s), width

    left, w_wide = panel(0, aw, bw, band_w,
                         "One pooling window", "as it was", S2_INK)
    right, w_geo = panel(228, ag, bg, band_g,
                         "Two windows, geometric mean", "as shipped", S1_INK)

    svg = [f'<svg viewBox="0 0 444 128" width="100%" role="img" '
           f'aria-label="Pooled focus energy across an occlusion boundary, '
           f'one window versus two">']
    svg.append(left)
    svg.append(right)
    svg.append('</svg>')

    legend = (
        f'<div class="legend">'
        f'<span><i style="background:{S1}"></i>Frame holding the <b>foreground</b> contour</span>'
        f'<span><i style="background:{S2}"></i>Frame holding the <b>background</b> texture</span>'
        f'<span class="muted">Pooled focus energy, each panel scaled to its own peak · kernel 51, narrow window {near}</span>'
        f'</div>'
    )
    caption = (
        f'<p class="fnote">A 1-D slice across an occlusion. A box window is '
        f'edge-blind, so one pooling smears the contour’s enormous energy '
        f'<code>51 // 2</code> px in <i>every</i> direction — including out over '
        f'background belonging to another slice, which is then taken from a frame '
        f'where it is defocused. The narrow window barely leaks, so requiring a '
        f'frame to win on both collapses the band from <b>{w_wide} px to '
        f'{w_geo}</b>.</p>'
    )
    return legend + svg[0] + "".join(svg[1:]) + caption


# --------------------------------------- figure 2: the depth map itself
def figure_depthmap():
    """The argmax over a region nothing resolves, before and after the median."""
    rng = np.random.default_rng(7)
    H, W, FRAMES = 76, 128, 24
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # A subject: a slanted surface inside a blob, whose sharp slice runs 5..18.
    blob = (((xx - 46) / 30.0) ** 2 + ((yy - 38) / 27.0) ** 2) < 1.0
    true_idx = np.clip(5 + (xx + 0.45 * yy) * 0.16, 0, FRAMES - 1)

    energy = np.empty((FRAMES, H, W))
    for k in range(FRAMES):
        # Everywhere the same defocused floor, so off the subject nothing but
        # grain separates the frames - the regime the argmax has no answer for.
        e = 1.0 + rng.normal(0, 0.075, size=(H, W))
        e += np.where(blob, 9.0 * np.exp(-((k - true_idx) ** 2) / (2 * 1.6 ** 2)), 0.0)
        energy[k] = e

    raw = np.argmax(energy, axis=0).astype(np.uint8)
    fixed = _regularise_index(raw.copy())

    def incoherence(idx, mask, radius=4, slices=3):
        med = cv2.medianBlur(idx, 2 * radius + 1)
        off = np.abs(idx.astype(np.int16) - med.astype(np.int16)) > slices
        return float(off[mask].mean() * 100.0)

    bg = ~blob
    before, after = incoherence(raw, bg), incoherence(fixed, bg)
    subj_shift = float(np.abs(fixed[blob].astype(float)
                              - raw[blob].astype(float)).mean())

    def ramp_rgb(t):
        t = float(np.clip(t, 0, 1)) * (len(RAMP) - 1)
        i = int(min(len(RAMP) - 2, math.floor(t)))
        f = t - i
        c0 = tuple(int(RAMP[i][j:j + 2], 16) for j in (1, 3, 5))
        c1 = tuple(int(RAMP[i + 1][j:j + 2], 16) for j in (1, 3, 5))
        return tuple(int(round(c0[c] + (c1[c] - c0[c]) * f)) for c in range(3))

    lut = np.array([ramp_rgb(v / (FRAMES - 1)) for v in range(FRAMES)],
                   dtype=np.uint8)

    def data_uri(idx, scale=2):
        rgb = lut[idx]
        big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        ok, buf = cv2.imencode(".png", big[:, :, ::-1])
        assert ok, "PNG encode failed"
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    stops = ", ".join(RAMP)
    return (
        f'<div class="dmaps">'
        f'  <figure><img src="{data_uri(raw)}" alt="Depth map straight from the '
        f'argmax: the subject is coherent, the background is confetti">'
        f'    <figcaption><b>Straight from the argmax.</b> '
        f'<span class="bad">{before:.0f}% of background pixels</span> chose a '
        f'frame more than 3 slices from their neighbourhood’s.</figcaption>'
        f'  </figure>'
        f'  <figure><img src="{data_uri(fixed)}" alt="The same depth map after '
        f'three passes of a 5 by 5 median: the background is coherent, the '
        f'subject unchanged">'
        f'    <figcaption><b>After 3 × 5×5 median.</b> '
        f'<span class="good">{after:.1f}%</span> — and the subject’s '
        f'own depth moved by {subj_shift:.2f} of a slice.</figcaption>'
        f'  </figure>'
        f'</div>'
        f'<div class="rampkey">'
        f'  <span>frame 0<br><span class="t">far</span></span>'
        f'  <i style="background:linear-gradient(90deg, {stops})"></i>'
        f'  <span class="r">frame 23<br><span class="t">near</span></span>'
        f'</div>'
    )


# --------------------------------- figure 3: the baseline weight, exactly
def figure_baseline():
    """What the baseline weight does to a region no frame resolves."""
    LO, HI = 0.01, 100.0
    xs = np.logspace(math.log10(LO), math.log10(HI), 200)
    # Twelve frames; grain alone leaves one of them at 3x the others' energy.
    # x is the region's energy in units of the baseline b.
    with_b = (3 * xs + 1) / (14 * xs + 12)
    without_b = np.full_like(xs, 3.0 / 14.0)

    PW, PH = 150, 76
    X0, Y0 = 34, 12
    box = (X0, X0 + PW, Y0, Y0 + PH)
    xlim = (math.log10(LO), math.log10(HI))
    ylim = (0.0, 0.26)
    lx = np.log10(xs)

    def ypx(v):
        return Y0 + PH - (v - ylim[0]) / (ylim[1] - ylim[0]) * PH

    def xpx(v):
        return X0 + (math.log10(v) - xlim[0]) / (xlim[1] - xlim[0]) * PW

    s = [f'<svg viewBox="0 0 200 122" width="100%" role="img" '
         f'aria-label="Share of the output taken by the strongest of twelve '
         f'frames, with and without the baseline weight">']
    # gridlines
    for v in (0.083, 0.15, 0.214):
        s.append(f'<line x1="{X0}" y1="{ypx(v):.1f}" x2="{X0 + PW}" '
                 f'y2="{ypx(v):.1f}" stroke="{GRID}" stroke-width="0.6"/>')
    s.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y0 + PH}" '
             f'stroke="{RULE}" stroke-width="1"/>')
    s.append(f'<line x1="{X0}" y1="{Y0 + PH}" x2="{X0 + PW}" y2="{Y0 + PH}" '
             f'stroke="{RULE}" stroke-width="1"/>')
    # the plain-mean floor, direct-labelled rather than left to the legend
    s.append(f'<line x1="{X0}" y1="{ypx(1 / 12):.1f}" x2="{X0 + PW}" '
             f'y2="{ypx(1 / 12):.1f}" stroke="{INK2}" stroke-width="1" '
             f'stroke-dasharray="3 2"/>')
    # Labelled below its own line: above it the two curves converge and any
    # label there would sit on top of them.
    s.append(f'<text x="{X0 + 3}" y="{ypx(1 / 12) + 8:.1f}" font-size="6.4" '
             f'fill="{INK2}" {FONT}>1/12 — the plain mean, and with it the '
             f'√N noise gain</text>')
    s.append(f'<polyline points="{poly(lx, without_b, box, xlim, ylim)}" '
             f'fill="none" stroke="{S2}" stroke-width="2"/>')
    s.append(f'<polyline points="{poly(lx, with_b, box, xlim, ylim)}" '
             f'fill="none" stroke="{S1}" stroke-width="2"/>')
    # axes
    for v, lab in ((0.01, "0.01"), (0.1, "0.1"), (1, "b"), (10, "10"), (100, "100")):
        s.append(f'<line x1="{xpx(v):.1f}" y1="{Y0 + PH}" x2="{xpx(v):.1f}" '
                 f'y2="{Y0 + PH + 2.5}" stroke="{RULE}" stroke-width="0.8"/>')
        s.append(f'<text x="{xpx(v):.1f}" y="{Y0 + PH + 9}" font-size="6.4" '
                 f'fill="{MUTED}" text-anchor="middle" {FONT}>{lab}</text>')
    for v in (0.0, 0.083, 0.15, 0.214):
        s.append(f'<text x="{X0 - 3}" y="{ypx(v) + 2:.1f}" font-size="6.4" '
                 f'fill="{MUTED}" text-anchor="end" {FONT}>{v * 100:.0f}%</text>')
    s.append(f'<text x="{X0 + PW / 2:.0f}" y="{Y0 + PH + 19}" font-size="6.8" '
             f'fill="{INK2}" text-anchor="middle" {FONT}>region’s focus '
             f'energy, in units of the baseline b →</text>')
    s.append(f'<text transform="translate(9,{Y0 + PH / 2:.0f}) rotate(-90)" '
             f'font-size="6.8" fill="{INK2}" text-anchor="middle" {FONT}>'
             f'strongest frame’s share</text>')
    s.append('</svg>')

    legend = (
        f'<div class="legend">'
        f'<span><i style="background:{S1}"></i>With the baseline — as shipped</span>'
        f'<span><i style="background:{S2}"></i>Pure contrast weighting</span>'
        f'</div>'
    )
    caption = (
        f'<p class="fnote">Twelve frames over a region none of them resolves, '
        f'where grain alone leaves one frame’s energy at 3× the '
        f'others’. Pure contrast weighting hands that frame '
        f'<b>{100 * 3 / 14:.1f}%</b> of the output no matter how meaningless '
        f'the energy is — it is ranking noise. The baseline is a fixed '
        f'fraction of the <i>stack’s</i> mean energy, so below it the '
        f'weights flatten to <b>{100 / 12:.1f}%</b> each and the region '
        f'becomes the plain mean; above it the sharp frame still takes over.</p>'
    )
    return legend + "".join(s) + caption


# ------------------------------------------------------------- injection
FIGURES = {
    "leak": figure_leak,
    "depthmap": figure_depthmap,
    "baseline": figure_baseline,
}


def main():
    with open(TARGET, encoding="utf-8") as fh:
        html = fh.read()

    for name, build in FIGURES.items():
        pattern = re.compile(
            r"(<!-- FIG-BEGIN " + name + r" -->).*?(<!-- FIG-END " + name + r" -->)",
            re.S)
        if not pattern.search(html):
            raise SystemExit(f"marker pair for {name!r} not found in {TARGET}")
        markup = build()
        html = pattern.sub(lambda m: m.group(1) + "\n" + markup + "\n" + m.group(2),
                           html, count=1)
        print(f"  {name}: {len(markup):,} chars")

    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
