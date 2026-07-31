"""
Build the illustrated explainer: how a RAW becomes a DNG, and what the options change.

Writes docs/how_dng_export_works.html - a self-contained page for a photographer
who has opened an OpenFocus DNG in Lightroom or Luminar and found the colour did
not match. Every figure is computed here from utils.dng's own constants and its
own writer, so the pictures show what the module actually does rather than an
illustration of what it is supposed to do:

  * the wrong renderings are produced by reproducing the mistake - a converter
    that ignores the transfer function, a camera profile applied to sRGB - not
    by tinting a correct image until it looks wrong;
  * the transfer curves are plotted from `dng._bt709_knee`;
  * the file sizes and round-trip errors come from writing real DNGs to a
    temporary directory and reading them back.

The demo frame is synthetic, so the page builds on any machine with no photo to
hand. `--raw` additionally renders the same comparisons from a real camera file,
which is what the numbers in the text were originally measured on.

Diagrams are inline SVG, so the page needs no scripts, no fonts and no network.

Examples:
    python tests/visualize_dng_pipeline.py --open
    python tests/visualize_dng_pipeline.py --raw "F:/photos/Img473.nef"
"""

import argparse
import base64
import os
import shutil
import sys
import tempfile
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import dng
# The design system is shared with the fusion explainer rather than copied, so
# the two pages cannot drift apart visually.
from tests.visualize_fusion_explainer import CSS
from tests.synthetic_stack import make_stack

OUT_DEFAULT = os.path.join("docs", "how_dng_export_works.html")
SIZE = 320

# A Nikon Z 6_2, as LibRaw holds it and as Adobe's DNG Converter writes it into
# ColorMatrix2 - the two agree to every published digit, which is what lets the
# camera colour space take its matrix from whatever raw the stack came from.
# Used here only so the page has a concrete camera to demonstrate with.
NIKON_Z62 = np.array([
    [0.9943, -0.3269, -0.0839],
    [-0.5323, 1.3269, 0.2259],
    [-0.1198, 0.2083, 0.7557],
])

SRGB_TO_XYZ = np.array(dng._SRGB_TO_XYZ_D65)
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
D65 = np.array(dng._D65_WHITE)

KNEE, OFFSET = dng._bt709_knee()


# ----------------------------------------------------------------------
# The transfer function, as the pipeline applies it
# ----------------------------------------------------------------------
def to_linear(display):
    """Display-referred 0..1 -> scene-linear 0..1, the curve the develop applied."""
    display = np.clip(display, 0.0, 1.0)
    return np.where(display < KNEE,
                    display / dng._BT709_SLOPE,
                    ((display + OFFSET) / (1.0 + OFFSET)) ** (1.0 / dng._BT709_POWER))


def to_display(linear):
    """Scene-linear 0..1 -> display-referred 0..1."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear < KNEE / dng._BT709_SLOPE,
                    linear * dng._BT709_SLOPE,
                    (1.0 + OFFSET) * linear ** dng._BT709_POWER - OFFSET)


def b64(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def as_float(img8):
    """8-bit BGR -> float BGR in 0..1."""
    return img8.astype(np.float64) / 255.0


def as_bytes(imgf):
    return np.rint(np.clip(imgf, 0.0, 1.0) * 255.0).astype(np.uint8)


def apply_matrix(bgr, matrix):
    """Apply a 3x3 RGB matrix to a BGR image, returning BGR."""
    return (bgr[:, :, ::-1] @ matrix.T)[:, :, ::-1]


# ----------------------------------------------------------------------
# The three renderings the page is about
# ----------------------------------------------------------------------
def render_correct(frame):
    """What the file should look like: the frame itself."""
    return frame


def render_ignored_transfer(frame):
    """A converter that never read the transfer function.

    It takes display-referred samples for scene-linear light and puts its own
    output curve on them, so the curve is applied twice and the picture comes
    back flat and washed out. Reproducing the mistake rather than drawing it is
    what makes this figure evidence.
    """
    return as_bytes(to_display(as_float(frame)))


def render_camera_profile_on_srgb(frame):
    """A camera profile applied to sRGB samples - the red cast.

    The profile does exactly its job: divide by the camera neutral, invert the
    camera matrix into XYZ, convert to sRGB for display. It is only wrong about
    what it was handed, because these samples left camera space during the
    develop and are already sRGB.
    """
    # The converter divides by the file's AsShotNeutral first, which for an sRGB
    # file is (1, 1, 1) and so does nothing, and then inverts the camera matrix
    # into XYZ. That inversion is the whole of the damage.
    profile = XYZ_TO_SRGB @ np.linalg.inv(NIKON_Z62)
    linear = to_linear(as_float(frame))
    return as_bytes(to_display(apply_matrix(linear, profile)))


def render_camera_space(frame):
    """The same profile over camera-space samples, which is what it was built for.

    The pixels are taken sRGB -> XYZ -> camera first, so the profile's inverse
    lands back on the frame. This is the whole of what `COLOR_CAMERA` buys.
    """
    # The white balance a converter applies and the matrix it follows it with are
    # a matched pair - it normalises by the neutral and its matrix carries the
    # neutral back - so the two cancel and what is left is the camera matrix's
    # own inverse. Modelling only the division, and not what compensates for it,
    # is a red cast of one's own making.
    gain = (NIKON_Z62 @ D65).max()
    forward = NIKON_Z62 @ SRGB_TO_XYZ / gain
    profile = XYZ_TO_SRGB @ np.linalg.inv(NIKON_Z62) * gain

    linear = to_linear(as_float(frame))
    camera = np.clip(apply_matrix(linear, forward), 0.0, 1.0)
    return as_bytes(to_display(apply_matrix(camera, profile)))


def camera_space_samples(frame):
    """The stored samples of a camera-space file, shown as a picture.

    Not a rendering of anything - raw data never is - but worth seeing: the green
    channel carries most of the signal and the red is far down, which is the
    shape of every Bayer sensor's response and is what AsShotNeutral undoes.
    """
    neutral = NIKON_Z62 @ D65
    forward = NIKON_Z62 @ SRGB_TO_XYZ / neutral.max()
    camera = np.clip(apply_matrix(to_linear(as_float(frame)), forward), 0.0, 1.0)
    return as_bytes(camera)


def linear_samples(frame):
    """The stored samples of an sRGB file: the frame with its curve taken off."""
    return as_bytes(to_linear(as_float(frame)))


# ----------------------------------------------------------------------
# Measurements - written and read back, not asserted
# ----------------------------------------------------------------------
def measure_modes(frame16):
    """Write the frame in every compression mode and report size and fidelity."""
    work = tempfile.mkdtemp(prefix="dngdoc_")
    rows = []
    try:
        for mode in dng.VALID_COMPRESSIONS:
            path = os.path.join(work, f"{mode}.dng")
            if not dng.write(path, frame16, compression=mode):
                continue
            # The lossy mode hands back 8-bit, so both sides are lifted to the
            # 16-bit scale before they are compared - scaling the *expected*
            # frame instead would be multiplying something that is already
            # 16-bit, and reports an overflow as a codec error.
            back = dng.read(path)
            error = None
            if back is not None:
                if back.dtype == np.uint8:
                    back = back.astype(np.uint16) * 257
                difference = np.abs(back.astype(np.int32) - frame16.astype(np.int32))
                error = (float(difference.mean()), int(difference.max()))
            tags = dng._read_ifd0(path) or {}
            rows.append({
                "mode": mode,
                "size": os.path.getsize(path),
                "bits": int(tags.get(dng._BITS_PER_SAMPLE, [0])[0]),
                "table": dng._LINEARIZATION_TABLE in tags,
                "error": error,
            })
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return rows


def measure_color_spaces(frame16):
    """Write both colour spaces and report what each file claims to be."""
    work = tempfile.mkdtemp(prefix="dngdoc_")
    rows = []

    def fake_camera(_path):
        neutral = NIKON_Z62 @ D65
        return dng._Colorimetry(
            matrix=tuple(tuple(r) for r in NIKON_Z62), forward=None,
            neutral=tuple(neutral / neutral.max()), camera="NIKON Z 6_2",
            make="NIKON CORPORATION", model="NIKON Z 6_2",
            profile=None, signature=None)

    real = dng._camera_colorimetry
    try:
        dng._camera_colorimetry = fake_camera
        for space in dng.VALID_COLOR_SPACES:
            path = os.path.join(work, f"{space}.dng")
            if not dng.write(path, frame16, color_space=space, source_path=__file__):
                continue
            tags = dng._read_ifd0(path) or {}
            pairs = np.frombuffer(tags[dng._AS_SHOT_NEUTRAL], "<u4").reshape(-1, 2)
            rows.append({
                "space": space,
                "make": tags.get(dng._MAKE),
                "camera": tags.get(dng._UNIQUE_CAMERA_MODEL),
                "neutral": (pairs[:, 0] / pairs[:, 1]),
                "profile": tags.get(dng._PROFILE_NAME, "-"),
                "signed": dng._CAMERA_CALIBRATION_SIGNATURE in tags,
                "own": dng.read_linear(path) is not None,
            })
    finally:
        dng._camera_colorimetry = real
        shutil.rmtree(work, ignore_errors=True)
    return rows


# ----------------------------------------------------------------------
# SVG diagrams
# ----------------------------------------------------------------------
def svg_pipeline():
    """Where the pixels come from, and where camera space is left behind."""
    return """
    <svg viewBox="0 0 900 268" width="900" role="img"
         aria-label="A NEF is demosaiced and developed into sRGB, fused, then written as DNG in one of two colour spaces">
      <defs>
        <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
                orient="auto"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--line-strong)"/></marker>
      </defs>
      <g font-size="13" text-anchor="middle">
        <rect x="8" y="60" width="132" height="70" rx="5"
              style="fill:var(--accent-soft);stroke:var(--accent)" stroke-width="1.4"/>
        <text x="74" y="86" style="fill:var(--ink)" font-size="14" font-weight="600">NEF</text>
        <text x="74" y="106" style="fill:var(--ink-2)">sensor mosaic</text>
        <text x="74" y="122" style="fill:var(--ink-3)" font-size="12">camera space</text>

        <rect x="186" y="60" width="146" height="70" rx="5"
              style="fill:var(--surface);stroke:var(--line-strong)" stroke-width="1.2"/>
        <text x="259" y="82" style="fill:var(--ink)" font-size="14" font-weight="600">Develop</text>
        <text x="259" y="100" style="fill:var(--ink-2)" font-size="12">demosaic, balance,</text>
        <text x="259" y="115" style="fill:var(--ink-2)" font-size="12">matrix, tone curve</text>

        <rect x="378" y="60" width="132" height="70" rx="5"
              style="fill:var(--surface);stroke:var(--line-strong)" stroke-width="1.2"/>
        <text x="444" y="86" style="fill:var(--ink)" font-size="14" font-weight="600">Fuse</text>
        <text x="444" y="106" style="fill:var(--ink-2)">many frames, one</text>
        <text x="444" y="121" style="fill:var(--ink-3)" font-size="12">sRGB, display-referred</text>

        <rect x="586" y="8" width="300" height="104" rx="5"
              style="fill:var(--surface);stroke:var(--f1)" stroke-width="1.6"/>
        <text x="736" y="32" style="fill:var(--f1)" font-size="14" font-weight="600">DNG &middot; sRGB (default)</text>
        <text x="736" y="52" style="fill:var(--ink-2)" font-size="12">samples stay sRGB, linearised</text>
        <text x="736" y="69" style="fill:var(--ink-2)" font-size="12">carries its own profile</text>
        <text x="736" y="90" style="fill:var(--ink)" font-size="12">renders as the fused result</text>
        <text x="736" y="105" style="fill:var(--ink-3)" font-size="11.5">camera profiles shift it red</text>

        <rect x="586" y="150" width="300" height="110" rx="5"
              style="fill:var(--surface);stroke:var(--f3)" stroke-width="1.6"/>
        <text x="736" y="174" style="fill:var(--f3)" font-size="14" font-weight="600">DNG &middot; camera space</text>
        <text x="736" y="194" style="fill:var(--ink-2)" font-size="12">samples put back into camera RGB</text>
        <text x="736" y="211" style="fill:var(--ink-2)" font-size="12">labelled with the camera</text>
        <text x="736" y="232" style="fill:var(--ink)" font-size="12">camera profiles apply correctly</text>
        <text x="736" y="247" style="fill:var(--ink-3)" font-size="11.5">renders as the raw, not as the fusion</text>
      </g>
      <g style="stroke:var(--line-strong)" stroke-width="1.6" fill="none" marker-end="url(#a)">
        <path d="M142 95 L182 95"/>
        <path d="M334 95 L374 95"/>
        <path d="M512 95 L556 95 L556 60 L582 60"/>
        <path d="M512 95 L556 95 L556 205 L582 205"/>
      </g>
      <text x="444" y="168" text-anchor="middle" style="fill:var(--ink-3)" font-size="12">
        the mosaic is gone by here - fusion needs finished pixels
      </text>
      <text x="444" y="186" text-anchor="middle" style="fill:var(--ink-3)" font-size="12">
        so the DNG can never hold what the NEF held
      </text>
    </svg>"""


def svg_transfer_curve():
    """The curve, and what applying it twice does."""
    x = np.linspace(0.0, 1.0, 160)
    lin = to_linear(x)
    twice = to_display(x)

    def path(values, colour_ignored=None):
        pts = ["%.1f,%.1f" % (40 + v0 * 300, 220 - v1 * 180)
               for v0, v1 in zip(x, values)]
        return "M" + " L".join(pts)

    return f"""
    <svg viewBox="0 0 400 260" width="400" role="img"
         aria-label="The BT.709 transfer curve, its inverse, and the effect of applying it twice">
      <g style="stroke:var(--line)" stroke-width="1">
        <line x1="40" y1="220" x2="340" y2="220"/>
        <line x1="40" y1="220" x2="40" y2="40"/>
      </g>
      <g font-size="11" style="fill:var(--ink-3)">
        <text x="40" y="238" text-anchor="middle">0</text>
        <text x="340" y="238" text-anchor="middle">1</text>
        <text x="190" y="252" text-anchor="middle">stored sample</text>
        <text x="32" y="44" text-anchor="end">1</text>
        <text x="32" y="224" text-anchor="end">0</text>
      </g>
      <path d="M40 220 L340 40" style="stroke:var(--line-strong)" stroke-width="1.2"
            stroke-dasharray="4 4" fill="none"/>
      <path d="{path(lin)}" style="stroke:var(--f1)" stroke-width="2.4" fill="none"/>
      <path d="{path(twice)}" style="stroke:var(--f2)" stroke-width="2.4" fill="none"/>
      <g font-size="12">
        <text x="250" y="196" style="fill:var(--f1)">what the samples mean</text>
        <text x="120" y="66" style="fill:var(--f2)">curve applied twice</text>
        <text x="255" y="120" style="fill:var(--ink-3)" font-size="11">no change</text>
      </g>
    </svg>"""


def svg_gamut():
    """Why the camera transform never clips: sRGB sits inside the camera's gamut."""
    return """
    <svg viewBox="0 0 360 240" width="360" role="img"
         aria-label="The sRGB gamut sits inside the camera gamut, so the transform does not clip">
      <polygon points="60,196 300,196 180,44"
               style="fill:none;stroke:var(--f3)" stroke-width="2" stroke-dasharray="6 4"/>
      <polygon points="96,176 264,176 180,80"
               style="fill:var(--accent-soft);stroke:var(--f1)" stroke-width="2"/>
      <g font-size="12.5" text-anchor="middle">
        <text x="180" y="130" style="fill:var(--f1)">sRGB</text>
        <text x="180" y="34" style="fill:var(--f3)">camera gamut</text>
        <text x="180" y="222" style="fill:var(--ink-3)" font-size="12">
          every sRGB colour has a camera coordinate - nothing to clip
        </text>
      </g>
    </svg>"""


# ----------------------------------------------------------------------
# Page assembly
# ----------------------------------------------------------------------
def figure(src, title, caption):
    return (f'<figure><img src="{src}" alt="{title}">'
            f'<figcaption><b>{title}</b>{caption}</figcaption></figure>')


def frame_from_raw(path, edge=560):
    """Develop a real camera file into a demo frame, and take its matrix with it.

    Returns None if it cannot be read, so `--raw` degrades to the synthetic frame
    rather than failing the build - the page is the deliverable, not the photo.
    """
    try:
        import rawpy
        with open(path, "rb") as handle:
            with rawpy.imread(handle) as raw:
                matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
                rgb = raw.postprocess(output_bps=8, use_camera_wb=True,
                                      no_auto_bright=True)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  ! could not read {os.path.basename(path)}: {exc}")
        return None
    if not np.isfinite(matrix).all() or not matrix.any():
        print(f"  ! no colour matrix for {os.path.basename(path)}")
        return None

    scale = edge / max(rgb.shape[:2])
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb[:, :, ::-1]), matrix


def build(raw_path=None):
    global NIKON_Z62

    frame = None
    if raw_path:
        print(f"  developing {os.path.basename(raw_path)}")
        loaded = frame_from_raw(raw_path)
        if loaded is not None:
            # The demo camera becomes the one the photo came from, so every
            # figure below is that body's own colour rather than a stand-in.
            frame, NIKON_Z62 = loaded

    if frame is None:
        print("  demo frame")
        _, frame, _ = make_stack(num_slices=3, height=SIZE, width=SIZE,
                                 seed=7, style="photographic")
    frame16 = (frame.astype(np.uint16) * 257)

    matrix_note = (f"{os.path.basename(raw_path)} as held by LibRaw" if raw_path and frame is not None
                   else "Nikon Z 6_2 as held by LibRaw")
    print("  renderings")
    correct = b64(render_correct(frame))
    ignored = b64(render_ignored_transfer(frame))
    on_srgb = b64(render_camera_profile_on_srgb(frame))
    in_camera = b64(render_camera_space(frame))
    stored_srgb = b64(linear_samples(frame))
    stored_cam = b64(camera_space_samples(frame))

    print("  writing files to measure")
    modes = measure_modes(frame16)
    spaces = measure_color_spaces(frame16)

    plain = next((r for r in modes if r["mode"] == dng.COMPRESSION_NONE), None)
    mode_rows = "".join(
        "<tr><td><code>%s</code></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            r["mode"],
            "%.0f KB" % (r["size"] / 1024),
            "%d%%" % round(100 * r["size"] / plain["size"]) if plain else "-",
            "%d-bit%s" % (r["bits"], ", + table" if r["table"] else ""),
            ("-" if r["error"] is None else
             "&plusmn;%d of 65535" % r["error"][1] if r["error"][1] <= 8 else
             "&plusmn;%d typical, %d worst" % (round(r["error"][0]), r["error"][1])),
        ) for r in modes)

    space_rows = "".join(
        "<tr><td><code>%s</code></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            r["space"], r["make"], r["camera"],
            ", ".join("%.3f" % v for v in r["neutral"]),
            r["profile"] if r["profile"] != "-" else "&mdash;",
        ) for r in spaces)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How a RAW becomes a DNG</title>
<style>{CSS}
  table {{ border-collapse:collapse; width:100%; font-size:14.5px; }}
  th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); }}
  th {{ font-family:var(--font-display); font-size:13px; letter-spacing:.02em;
        color:var(--ink-2); font-weight:600; border-bottom:1px solid var(--line-strong); }}
  td code {{ font-family:var(--font-mono); font-size:13px; color:var(--accent); }}
  .tablewrap {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line);
                border-radius:5px; padding:6px 14px 4px; }}
  .two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:26px; }}
  .verdict {{ font-family:var(--font-display); font-size:15px; font-weight:600; }}
  .bad {{ color:var(--f2); }} .good {{ color:var(--f1); }}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">OpenFocus &middot; DNG export</div>
    <h1>How a RAW becomes a DNG</h1>
    <p class="deck">
      A fused stack saved as DNG will not always look like the RAW it came from, and the
      settings that decide this are not obvious from their names. This page follows the pixels
      from the sensor to the file, and shows exactly what each option changes &mdash; including
      the two ways it can go wrong.
    </p>
  </header>

  <section>
    <div class="head"><h2>Two different kinds of file</h2></div>
    <p class="lede">
      A NEF and an OpenFocus DNG are both called &ldquo;raw&rdquo;, but they hold different
      things. The NEF holds the sensor's own mosaic &mdash; one colour measurement per pixel,
      untouched. Adobe's DNG Converter repackages exactly that, which is why a converted NEF
      behaves identically to the original and accepts any camera profile.
    </p>
    <p>
      A fused stack cannot be that file. Focus stacking compares frames pixel by pixel, so it
      needs finished pixels: the mosaic has to be demosaiced, white balanced and developed
      before any of it can happen. By the time there is a result to save, the sensor data is
      several irreversible steps behind.
    </p>
    <div class="diagram">{svg_pipeline()}</div>
    <div class="callout"><p>
      <b>This is the root of every difference on this page.</b> The DNG holds a picture that
      has already been rendered once. Everything the writer does is about stopping a converter
      rendering it a second time &mdash; or, if you ask for it, about putting the pixels back
      where a camera profile can find them.
    </p></div>
  </section>

  <section>
    <div class="head"><h2>Discrepancy 1: the transfer function</h2></div>
    <p class="lede">
      A raw converter assumes the numbers it reads are <em>scene-linear</em> &mdash; proportional
      to light. The pipeline's frames are <em>display-referred</em>: a curve has already been
      applied so they look right on a screen. Something has to reconcile the two.
    </p>
    <p>
      DNG has a tag for saying so, <code>LinearizationTable</code>, and OpenFocus used it: keep
      the samples as they are, and describe the curve. It is the tidier answer and it costs
      nothing in the shadows. It is also, in practice, not always read. Luminar Neo ignores the
      tag, takes the encoded samples for light and applies its own curve on top.
    </p>
    <div class="strip">
      {figure(correct, "As it should look", "The fused result, rendered by a converter that understood the file.")}
      {figure(ignored, "The curve applied twice", "A converter that skipped the tag. Flat, washed out, shadows lifted &mdash; the file was correct and unreadable at the same time.")}
    </div>
    <div class="two">
      <div class="diagram">{svg_transfer_curve()}</div>
      <div>
        <h3>The fix: apply it, don't declare it</h3>
        <p class="tight">
          The samples are now put through the curve on the way out, so the file really is
          scene-linear and there is no tag left to ignore. A converter cannot skip an encoding
          that is not there.
        </p>
        <p>
          The cost is precision in the deep shadows, where linear light has few codes to spare,
          so samples are always written at 16 bits &mdash; an 8-bit frame is promoted, and the
          file is larger for it. Reading one back undoes the curve, which is exact everywhere
          the curve is steep enough to separate its inputs and within a couple of parts in
          65535 at the very bottom.
        </p>
      </div>
    </div>
    <div class="callout"><p>
      The one exception is the <b>lossy</b> mode, which keeps the table. DNG restricts lossy to
      8-bit samples, and 8 bits of <em>linear</em> light collapses the whole shadow half of the
      range into two or three values. It costs nothing there: any reader that can open a lossy
      DNG supports the tag, because Adobe's own lossy files are built the same way.
    </p></div>
  </section>

  <section>
    <div class="head"><h2>Discrepancy 2: the colour space</h2></div>
    <p class="lede">
      The second difference is the one people notice in the profile menu. A camera profile
      &mdash; &ldquo;Nikon Z 6 2 Adobe Standard&rdquo; and its relatives &mdash; exists to turn
      that sensor's RGB into a picture. Handed sRGB instead, it does its job perfectly and
      produces nonsense.
    </p>
    <div class="strip">
      {figure(correct, "sRGB file, its own profile", "The default. The converter uses the profile written into the file and reproduces the fused result.")}
      {figure(on_srgb, "sRGB file, camera profile", "The same file with a camera profile selected. Not a bug in the profile: it inverted a camera matrix that was never applied.")}
      {figure(in_camera, "Camera-space file, camera profile", "The same profile over samples that really are camera RGB. The inversion now lands where it should.")}
    </div>
    <p>
      Camera space takes the pixels sRGB &rarr; XYZ &rarr; the camera's own coordinates, labels
      the file with that camera and its measured matrix, and sets <code>AsShotNeutral</code> to
      where daylight lands in those coordinates. The file becomes a synthetic RAW of the body
      the stack was shot on.
    </p>
    <div class="two">
      <div>
        <div class="strip">
          {figure(stored_srgb, "sRGB samples", "What an sRGB file stores: the frame with its curve removed.")}
          {figure(stored_cam, "Camera samples", "What a camera-space file stores. Green dominates and red sits far down &mdash; the shape of a real sensor's response, which the white balance undoes.")}
        </div>
      </div>
      <div>
        <h3>Nothing is lost in the transform</h3>
        <p class="tight">
          A camera's gamut contains sRGB's with room to spare, so every colour in the frame has
          a camera coordinate and nothing clips in either direction. Measured across saturated
          primaries, secondaries, neutrals and mixed tones, the round trip is exact to within a
          handful of parts in 65535.
        </p>
        <div class="diagram">{svg_gamut()}</div>
      </div>
    </div>
    <div class="callout"><p>
      <b>What it costs is the default rendering.</b> The file is a camera RAW now, so a converter
      develops it with that camera's profile and tone curve. It will look like the RAW developed
      in that converter &mdash; not like the fused result sitting beside it. The two modes answer
      different questions and only one can be answered at a time, which is why sRGB stays the
      default.
    </p></div>
  </section>

  <section>
    <div class="head"><h2>What each file claims to be</h2></div>
    <p class="lede">
      Written by the real writer and read back from the bytes. The difference is entirely in
      what the tags say about the samples.
    </p>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Colour space</th><th>Make</th><th>UniqueCameraModel</th>
          <th>AsShotNeutral</th><th>Embedded profile</th></tr></thead>
        <tbody>{space_rows}</tbody>
      </table>
    </div>
    <p>
      The neutral is the tell. <code>1, 1, 1</code> says the samples are already balanced and
      need nothing done to them. The camera-space triple says where daylight falls on that
      sensor, and a converter divides by it &mdash; which is exactly the step that makes a
      camera profile work.
    </p>
    <p>
      A camera-space file also stops being <em>our</em> file. It names the camera, so OpenFocus
      no longer recognises it as its own output and reloads it by developing it, the same as
      any other RAW. An sRGB file reloads as the frame that was saved.
    </p>
  </section>

  <section>
    <div class="head"><h2>Discrepancy 3: compression</h2></div>
    <p class="lede">
      Independent of colour, and a smaller decision than it looks. Sizes below are the same
      frame written three ways and read back.
    </p>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Mode</th><th>Size</th><th>Of uncompressed</th><th>Samples</th>
          <th>Round trip</th></tr></thead>
        <tbody>{mode_rows}</tbody>
      </table>
    </div>
    <p>
      <b>Uncompressed</b> is the default: the strips are the pixels, nothing to go wrong, fastest
      to write. <b>Lossless</b> stores prediction errors rather than approximations, so it gives
      back the same frame for a smaller file, at the cost of a slower save and a dependency
      &mdash; how much smaller depends entirely on the picture, and the demo frame above is
      noisier than a real photograph, so a real stack usually does better than the figure shown.
      <b>Lossy</b> is a proxy format: 8-bit only, so a 16-bit result is narrowed on the way out,
      and it is the one mode that still declares its transfer function rather than applying it.
    </p>
  </section>

  <section>
    <div class="head"><h2>Which to choose</h2></div>
    <div class="two">
      <div class="method">
        <div class="title"><h3>sRGB</h3><span class="tag">default</span></div>
        <p class="tight">
          Choose it when the DNG should look like what OpenFocus produced &mdash; when you are
          handing the fused result to someone, archiving it, or matching it against a JPEG XL or
          TIFF written from the same pixels.
        </p>
        <p class="verdict good">Renders as the fused result.</p>
        <p class="tight" style="font-size:14.5px;color:var(--ink-3)">
          Do not pick a camera profile for it. The menu will offer them; they are wrong for this
          file and no tag can stop you.
        </p>
      </div>
      <div class="method">
        <div class="title"><h3>Camera colour space</h3><span class="tag">opt-in</span></div>
        <p class="tight">
          Choose it when you want to finish the picture in Lightroom or Luminar with that
          camera's profiles and rendering, treating the fused stack as though it were a single
          exposure off the body.
        </p>
        <p class="verdict good">Camera profiles behave.</p>
        <p class="tight" style="font-size:14.5px;color:var(--ink-3)">
          Needs the source RAW &mdash; the colour matrix lives in LibRaw, not in any EXIF block.
          A stack fused from JPEGs, a monochrome result or the lossy mode falls back to sRGB and
          says so in the log.
        </p>
      </div>
    </div>
  </section>

  <footer>
    Figures computed from utils.dng &middot; transfer curve BT.709, knee {KNEE:.6f}, slope
    {dng._BT709_SLOPE} &middot; camera matrix {matrix_note} &middot;
    sizes and round-trip errors measured by writing and reading real files<br>
    Rebuild with: python tests/visualize_dng_pipeline.py
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
    ap.add_argument("--raw", default=None,
                    help="Draw the figures from this camera RAW instead of the "
                         "synthetic frame, using its own colour matrix")
    args = ap.parse_args()

    print("Building figures...")
    html = build(args.raw)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nWrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    if args.open_browser:
        webbrowser.open(f"file:///{out.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
