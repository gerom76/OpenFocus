"""Adobe DNG reading and writing.

DNG is a TIFF-based raw container, and the two directions are asymmetric enough
that they are worth describing separately.

**Reading** covers two quite different kinds of file that share the extension. A
camera DNG (or one produced by Adobe DNG Converter) holds undemosaiced sensor
data, and is developed by LibRaw through `rawpy` exactly as `.nef` is - the
loader routes it through the same RAW path, so it also picks up the GPU
postprocess and the 8/16-bit depth mode. A DNG written by OpenFocus holds
already-demosaiced pixels, and is read back verbatim from its strips by this
module: no development, no white balance, no tone curve, so a saved result
reloads bit-for-bit as the frame that was saved.

**Writing** produces a *linear* DNG - `PhotometricInterpretation` = LinearRaw,
one sample per channel, no CFA - because a fused result is already demosaiced
and there is no sensor mosaic left to describe. OpenCV cannot write DNG and
LibRaw cannot write at all, so the container is assembled here: a TIFF is a
length-prefixed tag table followed by pixel strips, and the DNG-specific part is
the handful of extra tags that tell a raw converter how to interpret those
pixels.

Those tags are what make the file mean something to Lightroom or RawTherapee
rather than merely parse:

* `ColorMatrix1` with `CalibrationIlluminant1` = D65 declares the data to be in
  sRGB primaries, so no colour twist is applied on top of it.
* `AsShotNeutral` = (1, 1, 1) declares it already neutral, so no white balance
  is applied either.
* The samples themselves are **scene-linear**, which is what a raw converter
  assumes the values it reads are. The pipeline's frames are display-referred,
  so they are put through the BT.709 EOTF on the way out - that being the curve
  the develop actually applies, since both `rawpy.postprocess` and
  core.gpu_decode encode with dcraw's default gamma of (2.222, 4.5).

  Declaring the encoding instead of undoing it is the other way to do this, and
  is what this module used to do: `LinearizationTable` is the tag meant for
  exactly that, it keeps the stored samples untouched, and it costs no precision
  in the shadows. It is also, in practice, not read. Luminar Neo ignores the
  tag, takes the encoded samples for linear light and applies its own gamma on
  top, and the doubly-encoded result is flat and washed out - which is how a
  fused stack came to open there looking nothing like the JPEG XL written beside
  it from the same pixels. LibRaw honours the table, so every converter built on
  it rendered the file correctly and the fault stayed invisible from here.

  A converter cannot ignore an encoding that is not there, so the curve is
  applied rather than described. What that costs is precision in the deep
  shadows, where linear light has few codes to spare: samples are therefore
  always written at 16 bits, an 8-bit frame being promoted on the way out, and
  the file is larger for it. `read` puts the samples back through the inverse
  curve, so a saved result still reloads as the frame that was saved - to within
  a handful of parts in 65535 at the bottom of the range, where the forward
  curve is many-to-one and no inverse can be exact.

  The lossy mode is the exception and keeps the table, because DNG restricts it
  to 8-bit samples and 8-bit *linear* data bands catastrophically - the whole
  shadow half of the range would collapse into two or three codes. That is also
  why the table is no obstacle there: a reader that can open a lossy DNG at all
  supports it, since Adobe's own lossy files are built the same way.

  Either way the curve is BT.709's. Declaring sRGB instead - which this module
  did until the table was corrected, and which the same knee now serves - is a
  curve the pixels were never put through. The two differ by up to 3109 parts in
  65535 of linear light, worst in the shadows, which is what made a DNG saved
  from a NEF render some 13 levels in 255 darker than the NEF it came from.
  Primaries and white point are shared between the two standards, so only the
  transfer function is at stake and `ColorMatrix1` is unaffected.

Rendering
---------
The tags above describe what the pixels *are*. They do not, on their own, stop a
converter deciding what to *do* with them, and that is a separate problem with a
separate answer.

A raw converter exists to render scene-referred sensor data into a picture, so
by default it applies a baseline tone curve, a black-point rendering and a
baseline exposure on top of whatever it reads. Against a camera raw that is the
whole point. Against a linear DNG it is a second rendering of a frame that has
already been rendered once, and it is what made a saved result open in Lightroom
brighter and flatter than the same frame in OpenFocus even after the
LinearizationTable was corrected - the table settles the transfer function, not
the converter's intentions.

DNG's answer is the embedded camera profile, so one is written:

* `ProfileToneCurve` is the identity, two points from (0,0) to (1,1). A
  converter with no profile to consult uses its own baseline curve; given one
  that says "no curve", it applies none.
* `DefaultBlackRender` = None says the black point is already where it belongs,
  so no automatic shadow rendering is applied on top.
* `BaselineExposure` and `BaselineExposureOffset` are zero, for the same reason
  in the other direction.
* `ForwardMatrix1` maps the camera neutral straight onto D50, which is what a
  profile-aware converter uses in preference to inverting `ColorMatrix1`, and
  removes the white-balance guesswork that inversion leaves it.
* `ProfileName` gives the result a profile a converter can name in its menu
  instead of showing a camera's, which is what a file that left camera space
  during the develop should say for itself, and `AsShotProfileName` names it
  again as the one to select by default rather than merely offer.
* `CameraCalibrationSignature` and `ProfileCalibrationSignature` are written
  with the same string, which is how DNG says which profiles a file may be
  developed with: a converter must not apply one whose signature differs. Adobe
  signs its camera profiles, so this is what stops one of *those* being
  substituted for the embedded profile. That substitution is not a subtle
  error - a Nikon profile's matrix expects that sensor's RGB, and applied to
  samples that are already sRGB it throws the frame hard towards magenta.
* `CameraCalibration1` and `AnalogBalance` are the identity, closing the last
  two places a converter that thinks it recognises the camera could insert a
  correction of its own.

The other half of that is what the file does *not* say. IFD 0's `Make` and
`Model` are the writer's own and not the source camera's, because they are what
a converter reads to decide which camera it is developing; see
`_CAMERA_IFD0_TAGS`. The exception is the camera colour space below, where the
samples really are that camera's and naming it is the point.

What this cannot do is make the file render the way the *source raw* renders in
the same converter. The DNG holds pixels a develop has already finished with;
the NEF beside it holds a mosaic that Lightroom develops with Adobe's own engine
and Adobe's own profile for that camera, which is not the engine that produced
these pixels. Those two are different renderings of the same exposure and no tag
reconciles them. What the profile buys is the achievable half: the DNG renders
as the frame OpenFocus produced, rather than as that frame with a converter's
rendering stacked on it.

Camera colour space
-------------------
Signatures stop a converter *substituting* a camera profile. They do not stop a
user picking one from the menu, and picking one wrecks the frame - a Nikon
profile's matrix expects that sensor's RGB and throws sRGB samples hard towards
magenta. The file cannot prevent that while its samples are sRGB, because the
profile is not wrong about what it does, only about what it has been given.

`COLOR_CAMERA` answers that by giving it the right thing. The pixels are taken
sRGB -> XYZ -> the source camera's own space, the file is labelled with that
camera and its measured matrix, and `AsShotNeutral` is set to where D65 lands in
those coordinates. The result is a synthetic raw of the body the stack was shot
on: every camera profile applies correctly, because the data is finally what
they were built for.

The transform is exact and loses nothing. A camera's gamut contains sRGB's with
room to spare - across saturated primaries, secondaries and neutrals nothing
clips in either direction - and a converter that reads the neutral, undoes the
balance and applies the matrix arrives back at the frame that was written, to
within a handful of parts in 65535.

What it costs is the default rendering. The file is now a camera raw, so a
converter develops it with that camera's profile and tone curve, and it looks
like the raw developed there rather than like the fused result beside it. That
is the trade the mode exists to offer, and it is why sRGB remains the default:
the two modes answer different questions, and only one of them can be answered
at a time.

It also behaves like a raw in ways that are not obvious. LibRaw lowers the white
level to the brightest data it finds unless `adjust_maximum_thr` is turned off,
which rescales any file whose pixels stop short of saturation - a real camera
file included - so a camera-space DNG picks up the same treatment.

The mode needs the source raw, whose matrix lives in LibRaw rather than in any
EXIF block, which is why `write` takes a path as well as a block. Without one -
a stack fused from JPEGs, a monochrome result, the lossy mode, a build without
rawpy - the write falls back to sRGB and says so. And because such a file names
the camera rather than this writer, `read_linear` no longer recognises it as our
own output and `read` develops it through LibRaw, which is correct: it is a
camera raw now, and its samples are not the frame that was saved.

Compression
-----------
DNG is strict about which codec may carry which kind of data, and the rules
decide the three modes offered here (`COMPRESSION_*`):

- ``none``     `Compression` = 1. What this module wrote before the modes
               existed, and still the default: no dependency, no encode cost,
               and the strips are the pixels.
- ``lossless`` `Compression` = 7, lossless Huffman JPEG. The *only* lossless
               codec DNG permits for 16-bit LinearRaw data - Deflate (8) is
               restricted by the spec to floating point, 32-bit integer,
               transparency mask and depth map data, so it is not an option for
               integer image data however well it would compress. Costs nothing
               in quality and takes a fused master to roughly 50-55% of its
               uncompressed size at 16 bits, 35-40% at 8.
- ``lossy``    `Compression` = 34892, baseline DCT JPEG. The spec allows this
               only for **8-bit** LinearRaw, so choosing it narrows a 16-bit
               result on the way out; `encode` says so rather than doing it
               quietly. Meant for proxies, not for masters.

A compressed image is written as one strip - not for tidiness, but because
LibRaw walks the strips of a compressed *stripped* DNG by continuing from
wherever the codec left the file pointer rather than by consulting
`StripOffsets`. A multi-strip lossless file therefore decodes its first strip
and renders the rest black. `StripByteCounts` also holds compressed lengths, so
the strip has to be encoded before the tag table around it can be laid out.

The lossless codestream is produced by utils.ljpeg, written for this, rather
than by a library: see that module for why the single-component encoding the
spec would have allowed - and which `imagecodecs` could have written - is not
usable in practice.

Fast-load preview
-----------------
`fast_load` embeds a half-resolution, JPEG-compressed rendering of the result in
a SubIFD marked `NewSubFileType` = 1, alongside the `Preview*` tags that say
what it is and which colour space it is in. Without it a viewer has nothing to
show but the LinearRaw itself, so every thumbnail costs a full decode of the
main image - and for a fused stack that is the slowest thing in the file. With
it, browsers, Explorer and raw converters draw from a few hundred kilobytes of
baseline JPEG.

It is reached through the `SubIFDs` tag rather than by moving the raw out of
IFD 0: DNG only *recommends* a thumbnail in the first IFD, and keeping the
full-resolution image where it has always been means files written by earlier
versions still read back through the same path.

This is not Adobe's "Embed Fast Load Data", which is a partially-processed
Camera Raw cache in a private SubIFD and is not part of the DNG specification;
its contents cannot be reproduced from outside Adobe's converter. What is
embedded here is the spec's own preview mechanism, which is what gives
non-Adobe readers - and Adobe's own browsers - a fast path to pixels.

Source EXIF
-----------
A DNG written here can carry the EXIF block of the file the frame came from, so
a processed stack keeps the camera, lens and exposure it was shot with. Unlike
JPEG, PNG and JPEG XL - where utils.metadata splices a block into the encoded
file afterwards - a TIFF cannot be added to after the fact: every offset behind
an insertion would have to move. The block is therefore taken apart and re-laid
out as the file is written, which is also why `write` is the one that takes it.

What crosses is the record of the shot - the exposure, the lens, the date, the
MakerNote - and not the camera's `Make` and `Model`, which stay behind in IFD 0
for the reason the "Rendering" section gives.

That is a translation and not a copy. An EXIF block is itself a TIFF stream, so
its tags are already in the right shape, but its offsets are measured from its
own start; the tags whose values *are* offsets - the IFD pointers and the
embedded thumbnail - are left behind rather than written pointing at nothing.
Everything else moves across, byte order included, since camera blocks are as
often big-endian as not.

MakerNote is decided one note at a time, by `_relocatable_maker_note`. A note
whose offsets run from the file it was written into cannot be moved and is
dropped; a note that brings its own TIFF header, as Nikon's does, is measured
from itself and travels intact - which is what puts the lens, the shutter count
and the rest of a Nikon's private record in the DNG rather than only in the NEF.

One thing such a note may not bring is a preview. A Nikon's holds a small JPEG
of the frame it came from, and a browser looking for a raw file's preview will
take it whichever IFD it is in - so a DNG written without a fast-load preview
would show one source frame at 640x424 in place of the fused result.
`_without_maker_note_preview` takes it out; see there for why the bytes go too.

Memory
------
A save holds the frame, and beyond that only what the codec forces it to. The
tag table is laid out before any pixel is touched - an uncompressed strip's size
follows from its geometry - so the strips are converted and written one at a
time straight to the file rather than accumulated into a second copy of the
image and then a third of the whole file. The compressed modes still hold their
one strip whole, because that strip is the entire image.

Reading is the mirror: the strips of an uncompressed file are read directly into
the array that will be returned, and the channel swap that follows is done in
place.

Dependencies
------------
Every mode *writes* with numpy and OpenCV alone. Reading back a lossless file is
what needs `imagecodecs` - the same optional package JPEG XL uses - because
Huffman decoding is inherently serial and a numpy implementation of it would be
far slower than a stack load can afford. The mode is therefore offered only when
that package is present: a format this app can write but not reopen would break
the one thing a saved result has to do, which is serve as the input to the next
stack. `available_compressions` reports what this build can actually round trip,
and the setting is validated against it.

Reading a camera DNG needs `rawpy`, which is a core requirement, so
`is_available()` tracks that the same way the loader's RAW support does.
"""

import datetime
import io
import os
import struct
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import ljpeg

# Extensions routed to this module instead of cv2.imwrite / cv2.imdecode.
EXTENSIONS = (".dng",)

# Written into UniqueCameraModel, and the marker `read` uses to recognise its own
# output and take the verbatim path instead of developing the file.
CAMERA_MODEL = "OpenFocus"

# Strips are sized to about this many bytes of *uncompressed* pixels. A single
# strip spanning a 24 MP 16-bit image would be a 144 MB run, which some readers
# handle poorly; a few megabytes per strip is what camera DNGs use. It also
# bounds the working set of a compressed strip's encode and decode.
_STRIP_TARGET_BYTES = 8 << 20

try:
    import rawpy
    _RAWPY_AVAILABLE = True
except ImportError:
    rawpy = None
    _RAWPY_AVAILABLE = False

try:
    import imagecodecs
    _LJPEG_AVAILABLE = bool(
        getattr(imagecodecs, "LJPEG", None) and imagecodecs.LJPEG.available
    )
except Exception:  # pylint: disable=broad-except
    imagecodecs = None
    _LJPEG_AVAILABLE = False


# ----------------------------------------------------------------------
# Compression modes
# ----------------------------------------------------------------------
COMPRESSION_NONE = "none"
COMPRESSION_LOSSLESS = "lossless"
COMPRESSION_LOSSY = "lossy"

# Every mode this module knows, in increasing order of what it costs the pixels.
VALID_COMPRESSIONS = (COMPRESSION_NONE, COMPRESSION_LOSSLESS, COMPRESSION_LOSSY)

# Uncompressed, so a build that gains or loses `imagecodecs` writes the same
# file, and so the default never trades quality or a dependency for size.
DEFAULT_COMPRESSION = COMPRESSION_NONE

# Which colour space the samples are written in.
#
# `COLOR_SRGB` stores the result as it is - sRGB primaries, its own profile, and
# a converter's default rendering reproduces the fused frame. What it cannot do
# is accept a *camera* profile: "Nikon Z 6 2 Adobe Standard" and the like are
# built to develop sensor RGB, and applied to sRGB they double-transform it into
# a heavy red cast. The profile menu still offers them, and picking one is a
# mistake the file cannot prevent.
#
# `COLOR_CAMERA` transforms the pixels back into the source camera's own space
# and labels the file with that camera, making it a synthetic raw of the body
# the stack was shot on. Camera profiles then apply correctly, because the data
# is finally what they were built for. The trade is that the converter's default
# rendering is now the camera's - Adobe Standard's tone curve and hue twists on
# top of an already-finished frame - so the file looks like the raw developed in
# that converter rather than like the fused result beside it.
#
# Needs the source raw, whose matrix is read through LibRaw; a stack fused from
# JPEGs, or one whose source cannot be opened, falls back to `COLOR_SRGB`.
COLOR_SRGB = "srgb"
COLOR_CAMERA = "camera"

VALID_COLOR_SPACES = (COLOR_SRGB, COLOR_CAMERA)

# sRGB, because it is the mode that needs nothing of the source and reproduces
# the frame the pipeline actually produced. Camera space answers a narrower
# question - "let me develop this like the raw" - and is asked for explicitly.
DEFAULT_COLOR_SPACE = COLOR_SRGB

# Quality for the lossy mode, on OpenCV's 1-100 JPEG scale. 92 is high enough
# that the DCT is not the thing limiting a proxy, and well short of the point
# where the file grows for no visible return.
DEFAULT_LOSSY_QUALITY = 92
MIN_LOSSY_QUALITY = 1
MAX_LOSSY_QUALITY = 100

# Whether a fast-load preview is embedded by default. On: the preview costs a
# few hundred kilobytes and one downscale, and is the difference between a
# thumbnail appearing at once and a viewer decoding the whole LinearRaw first.
DEFAULT_FAST_LOAD = True

# Preview quality, and the divisor applied to each side. Half resolution is what
# Adobe's own fast-load data uses: enough to fill a develop-module window
# without approaching the size of the raw it stands in for.
_PREVIEW_QUALITY = 85
_PREVIEW_DIVISOR = 2

# A preview smaller than this on its long side is not worth a SubIFD - the main
# image is already thumbnail-sized, so decoding it costs nothing to avoid.
_PREVIEW_MIN_EDGE = 160


# ----------------------------------------------------------------------
# TIFF field types and tags
# ----------------------------------------------------------------------
_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL, _SRATIONAL = 1, 2, 3, 4, 5, 10

# FLOAT is written for one tag only - ProfileToneCurve, which the spec defines
# as pairs of floats rather than as the rationals the rest of the colorimetry
# uses.
_FLOAT = 11

_NEW_SUBFILE_TYPE = 254
_IMAGE_WIDTH = 256
_IMAGE_LENGTH = 257
_BITS_PER_SAMPLE = 258
_COMPRESSION = 259
_PHOTOMETRIC = 262
_MAKE = 271
_MODEL = 272
_STRIP_OFFSETS = 273
_ORIENTATION = 274
_SAMPLES_PER_PIXEL = 277
_ROWS_PER_STRIP = 278
_STRIP_BYTE_COUNTS = 279
_X_RESOLUTION = 282
_Y_RESOLUTION = 283
_PLANAR_CONFIG = 284
_RESOLUTION_UNIT = 296
_SUB_IFDS = 330
_SOFTWARE = 305
_DATE_TIME = 306
_DATE_TIME_ORIGINAL = 36867
_YCBCR_SUB_SAMPLING = 530
_SAMPLE_FORMAT = 339
_IMAGE_DESCRIPTION = 270
_ARTIST = 315
_COPYRIGHT = 33432
_EXIF_IFD = 34665
_GPS_IFD = 34853
_INTEROP_IFD = 40965
_MAKER_NOTE = 37500
_JPEG_INTERCHANGE_FORMAT = 513
_JPEG_INTERCHANGE_LENGTH = 514
_PREVIEW_APPLICATION_NAME = 50966
_PREVIEW_APPLICATION_VERSION = 50967
_PREVIEW_COLOR_SPACE = 50970
_PREVIEW_DATE_TIME = 50971
_DNG_VERSION = 50706
_DNG_BACKWARD_VERSION = 50707
_UNIQUE_CAMERA_MODEL = 50708
_LINEARIZATION_TABLE = 50712
_WHITE_LEVEL = 50717
_COLOR_MATRIX_1 = 50721
_CAMERA_CALIBRATION_1 = 50723
_ANALOG_BALANCE = 50727
_AS_SHOT_NEUTRAL = 50728
_BASELINE_EXPOSURE = 50730
_CALIBRATION_ILLUMINANT_1 = 50778
_CAMERA_CALIBRATION_SIGNATURE = 50931
_PROFILE_CALIBRATION_SIGNATURE = 50932
_AS_SHOT_PROFILE_NAME = 50934
_PROFILE_NAME = 50936
_PROFILE_TONE_CURVE = 50940
_PROFILE_EMBED_POLICY = 50941
_FORWARD_MATRIX_1 = 50964
_BASELINE_EXPOSURE_OFFSET = 51109
_DEFAULT_BLACK_RENDER = 51110

# PhotometricInterpretation for demosaiced raw data - the whole point of a
# linear DNG, as opposed to 32803 (CFA) for a sensor mosaic.
_LINEAR_RAW = 34892

# PhotometricInterpretation for a JPEG-compressed preview. The spec names this
# as the value to use for one, and pins the JPEG to baseline DCT when it is
# paired with 8/8/8 BitsPerSample - which is what OpenCV writes.
_YCBCR = 6

# Compression tag values, and the mode each one stands for. Lossy JPEG shares
# its code with LinearRaw's PhotometricInterpretation, which is a coincidence of
# the spec's numbering and not a relationship.
_COMPRESSION_UNCOMPRESSED = 1
_COMPRESSION_JPEG = 7
_COMPRESSION_LOSSY_JPEG = 34892

_COMPRESSION_CODES = {
    COMPRESSION_NONE: _COMPRESSION_UNCOMPRESSED,
    COMPRESSION_LOSSLESS: _COMPRESSION_JPEG,
    COMPRESSION_LOSSY: _COMPRESSION_LOSSY_JPEG,
}
_COMPRESSION_MODES = {code: mode for mode, code in _COMPRESSION_CODES.items()}

# PreviewColorSpace: the preview is rendered from sRGB-encoded pipeline pixels.
_PREVIEW_COLOR_SPACE_SRGB = 2

# CalibrationIlluminant code for D65, the white point sRGB is defined against.
_ILLUMINANT_D65 = 21

# Everything the LinearizationTable maps into, and therefore WhiteLevel. 8-bit
# samples are given the same 16-bit linear range as 16-bit ones: undoing a
# display curve compresses the shadows hard, and 256 output levels there would
# band.
_LINEAR_MAX = 65535

# dcraw's default gamma, which is BT.709's: an exponent of 0.45 on the way out
# with a linear segment of slope 4.5 near black. LibRaw exposes it as
# (2.222, 4.5) - the reciprocal - and applies it unless a caller overrides it,
# so it is the curve on every frame the pipeline develops from a raw file, by
# way of `rawpy.postprocess` or of core.gpu_decode's reimplementation.
_BT709_POWER = 0.45
_BT709_SLOPE = 4.5

# XYZ (D65) -> linear sRGB. ColorMatrix1 is defined in that direction: it takes
# XYZ under the calibration illuminant into the "camera" space, which for this
# writer is sRGB itself. sRGB and BT.709 share primaries and white point - they
# part company only over the transfer function - so this matrix describes the
# pixels whichever of the two curves the LinearizationTable declares.
_XYZ_D65_TO_SRGB = (
    (3.2406, -1.5372, -0.4986),
    (-0.9689, 1.8758, 0.0415),
    (0.0557, -0.2040, 1.0570),
)

# Linear sRGB -> XYZ (D50), Bradford-adapted: the matrix an ICC sRGB profile
# carries. ForwardMatrix1 is defined into D50 whatever the calibration
# illuminant is, because D50 is the profile connection space, so this is not
# the inverse of ColorMatrix1 and the two coexist without contradiction. It has
# to map the camera neutral onto the D50 white point, and with AsShotNeutral
# (1, 1, 1) the camera neutral is (1, 1, 1); the rows sum to (0.9642, 1.0000,
# 0.8252), which is exactly D50, so the requirement holds by construction.
_SRGB_TO_XYZ_D50 = (
    (0.4360747, 0.3850649, 0.1430804),
    (0.2225045, 0.7168786, 0.0606169),
    (0.0139322, 0.0971045, 0.7141733),
)

# The profile's name, which is what a converter shows in its profile menu. A
# linear DNG has no camera to name - the pixels left camera space during the
# develop - so it names what it actually is.
_PROFILE_NAME_TEXT = "OpenFocus Linear"

# Linear sRGB -> XYZ (D65). The way back out of sRGB, needed only by the camera
# colour space below; the D50 matrix above cannot stand in for it, being adapted
# to a different white point.
_SRGB_TO_XYZ_D65 = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

# D65, the white point the pixels are already balanced to. `AsShotNeutral` in a
# camera-space file is this point taken into camera coordinates, which is what
# tells a converter the result needs no further white balancing.
_D65_WHITE = (0.95047, 1.0, 1.08883)

# ProfileEmbedPolicy 0, "allow copying": the profile describes nothing
# proprietary, so there is no reason to stop a converter carrying it elsewhere.
_PROFILE_EMBED_ALLOW_COPYING = 0

# Written into both CameraCalibrationSignature and ProfileCalibrationSignature,
# which is the spec's own mechanism for saying which profiles a file may be
# developed with: a converter must not apply a profile whose calibration
# signature differs from the file's. Adobe signs its camera profiles "com.adobe",
# so signing this file with something of its own is what stops one of them being
# used - and an Adobe Nikon profile applied to these pixels is not a small error.
# Its matrix expects the sensor RGB of a Nikon, while these samples are already
# sRGB, so it twists a finished frame hard towards magenta.
#
# The name is reverse-DNS because that is the convention the spec asks for; the
# domain need not resolve, it only has to be unlikely to collide.
_CALIBRATION_SIGNATURE = "com.openfocus"

# CameraCalibration1 and AnalogBalance as the identity. Both sit between the
# camera's raw values and the colour matrix, and both default to the identity
# when absent - but a converter that has decided which camera it is looking at
# may reach for its own values instead of the default. Writing them says there
# is nothing to correct for, because the develop is already behind these pixels.
_IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# ProfileToneCurve as the two points of the identity, which is what stops a
# converter applying its own baseline curve - see the "Rendering" section.
_IDENTITY_TONE_CURVE = (0.0, 0.0, 1.0, 1.0)

# DefaultBlackRender 1, "none": the result already has its black point where it
# belongs, so a converter must not go looking for another one.
_DEFAULT_BLACK_RENDER_NONE = 1

# Denominator for the SRATIONAL colour matrix entries; six digits is well past
# the precision the matrix itself is quoted to.
_MATRIX_DENOMINATOR = 1000000

# DNG 1.4 is claimed so monochrome results are legal. The backward version is
# 1.1 for the uncompressed and lossless modes - the layout they use, a
# full-resolution image in IFD 0 with an optional preview SubIFD, is the one DNG
# has accepted since 1.0 - and 1.4 for the lossy mode, whose compression code
# the spec says to declare that way.
_DNG_VERSION_BYTES = bytes((1, 4, 0, 0))
_DNG_BACKWARD_VERSION_BYTES = bytes((1, 1, 0, 0))
_DNG_BACKWARD_VERSION_LOSSY_BYTES = bytes((1, 4, 0, 0))


def is_available() -> bool:
    """Whether this build can read DNG.

    Writing needs nothing beyond numpy and OpenCV, but a build that cannot
    develop a camera DNG cannot honestly offer the format, so this tracks
    `rawpy` - the same dependency the loader's RAW formats have.
    """
    return _RAWPY_AVAILABLE


def is_dng(extension: str) -> bool:
    """Whether a file extension names the DNG container."""
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext in EXTENSIONS


def unavailable_reason() -> str:
    """One line explaining why DNG is off, for logs and message boxes."""
    if _RAWPY_AVAILABLE:
        return ""
    return "DNG support needs the 'rawpy' package: pip install rawpy"


def extensions() -> tuple:
    """The extensions this build can handle - empty when rawpy is absent.

    Mirrors utils.jxl.extensions, so the loader and the folder-scanning fusion
    methods can fold DNG into their supported sets without each repeating the
    availability check.
    """
    return EXTENSIONS if _RAWPY_AVAILABLE else ()


# ----------------------------------------------------------------------
# Compression settings
# ----------------------------------------------------------------------
# Module-level rather than arguments threaded through every caller, the same way
# utils.bitdepth holds the depth mode and utils.jxl its decode thread budget:
# the writer is reached through image_utils.write_image, which every save path in
# the app shares and none of which has an opinion about DNG's codec. Only the
# settings UI sets these.
_compression = DEFAULT_COMPRESSION
_lossy_quality = DEFAULT_LOSSY_QUALITY
_fast_load = DEFAULT_FAST_LOAD
_color_space = DEFAULT_COLOR_SPACE


def lossless_available() -> bool:
    """Whether the lossless mode can be both written and read by this build."""
    return _LJPEG_AVAILABLE


def lossless_unavailable_reason() -> str:
    """One line explaining why the lossless mode is off, for logs and dialogs."""
    if _LJPEG_AVAILABLE:
        return ""
    return ("Reading back lossless DNG needs the 'imagecodecs' package: "
            "pip install imagecodecs")


def available_compressions() -> tuple:
    """The compression modes this build can round trip, in the documented order.

    The lossless mode is dropped when `imagecodecs` is absent - writing it would
    work, but the result could not be reopened - the same kind of gate JPEG XL
    applies to the format as a whole.
    """
    return tuple(mode for mode in VALID_COMPRESSIONS
                 if mode != COMPRESSION_LOSSLESS or _LJPEG_AVAILABLE)


def set_compression(mode: str) -> str:
    """Set the compression mode for subsequent writes. Returns the mode applied.

    An unknown mode, or the lossless mode on a build without `imagecodecs`,
    raises rather than being quietly downgraded: a caller asking for lossless
    and getting uncompressed would only find out from the file size.
    """
    global _compression
    if mode not in VALID_COMPRESSIONS:
        raise ValueError(
            f"Unknown DNG compression mode: {mode!r}. Expected one of {VALID_COMPRESSIONS}."
        )
    if mode == COMPRESSION_LOSSLESS and not _LJPEG_AVAILABLE:
        raise RuntimeError(lossless_unavailable_reason())
    _compression = mode
    return _compression


def get_compression() -> str:
    """The compression mode subsequent writes will use."""
    return _compression


def set_lossy_quality(quality: int) -> int:
    """Set the lossy mode's JPEG quality, clamped to 1-100. Returns what stuck."""
    global _lossy_quality
    _lossy_quality = int(max(MIN_LOSSY_QUALITY, min(MAX_LOSSY_QUALITY, int(quality))))
    return _lossy_quality


def get_lossy_quality() -> int:
    """The JPEG quality the lossy mode will use."""
    return _lossy_quality


def set_fast_load(enabled: bool) -> bool:
    """Set whether writes embed a fast-load preview. Returns what stuck."""
    global _fast_load
    _fast_load = bool(enabled)
    return _fast_load


def get_fast_load() -> bool:
    """Whether writes embed a fast-load preview."""
    return _fast_load


def camera_space_available() -> bool:
    """Whether a camera-space write can be attempted at all by this build."""
    return _RAWPY_AVAILABLE


def camera_space_unavailable_reason() -> str:
    """One line explaining why camera space is off, for logs and dialogs."""
    if _RAWPY_AVAILABLE:
        return ""
    return ("Writing camera-space DNG needs the 'rawpy' package to read the "
            "source raw's colour matrix: pip install rawpy")


def set_color_space(mode: str) -> str:
    """Set the colour space for subsequent writes. Returns the mode applied.

    Camera space is refused outright on a build without rawpy, rather than
    quietly writing sRGB: a caller that asked for a file its camera profiles
    would accept, and got one they wreck, would only find out by opening it.
    Falling back per-write is different and does happen - see `_camera_color` -
    because there the setting is right and only that one source cannot serve it.
    """
    global _color_space
    if mode not in VALID_COLOR_SPACES:
        raise ValueError(
            f"Unknown DNG colour space: {mode!r}. Expected one of {VALID_COLOR_SPACES}."
        )
    if mode == COLOR_CAMERA and not _RAWPY_AVAILABLE:
        raise RuntimeError(camera_space_unavailable_reason())
    _color_space = mode
    return _color_space


def get_color_space() -> str:
    """The colour space subsequent writes will use."""
    return _color_space


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def _bt709_knee() -> Tuple[float, float]:
    """Where BT.709's linear segment meets its power segment, and the offset.

    Returns the breakpoint in *encoded* terms and the offset the power segment
    is shifted by - dcraw's ``g[2]`` and ``g[4]``. The standard quotes these as
    0.081 and 0.099, but dcraw does not use the quoted figures: it solves for
    the pair that makes the two segments meet with a continuous slope, by the
    same 48-step bisection reproduced here. The difference is small - seven
    parts in 65535 at worst - but this is the tag that has to invert what LibRaw
    applies, and solving costs a few dozen floating-point operations once.
    """
    low, high = 0.0, 1.0
    for _ in range(48):
        knee = (low + high) / 2.0
        # Slope of the power segment at `knee` against the linear segment's.
        if ((knee / _BT709_SLOPE) ** -_BT709_POWER - 1) / _BT709_POWER - 1 / knee > -1:
            high = knee
        else:
            low = knee
    knee = (low + high) / 2.0
    return knee, knee * (1.0 / _BT709_POWER - 1.0)


def _linearization_table(bits: int) -> np.ndarray:
    """The pipeline's EOTF, tabulated for every value a sample of `bits` can hold.

    Entry *i* is the scene-linear value that a display-referred sample *i* stands
    for, scaled to 0..`_LINEAR_MAX`. It is used two ways: applied, to linearise
    the frame a save writes, and written out as `LinearizationTable` by the lossy
    mode, which cannot afford to store linear samples in the 8 bits DNG allows
    it. Both directions want the same curve, so there is one table.

    The curve is BT.709's, which is what the develop puts the pixels through -
    see the module docstring for why declaring sRGB here rendered a saved frame
    darker than the raw it was made from.
    """
    knee, offset = _bt709_knee()
    encoded = np.linspace(0.0, 1.0, 1 << bits, dtype=np.float64)
    linear = np.where(
        encoded < knee,
        encoded / _BT709_SLOPE,
        ((encoded + offset) / (1.0 + offset)) ** (1.0 / _BT709_POWER),
    )
    return np.rint(linear * _LINEAR_MAX).astype(np.uint16)


def _display_table() -> np.ndarray:
    """`_linearization_table`'s inverse: scene-linear back to display-referred.

    Entry *i* is the 16-bit display-referred sample that scene-linear value *i*
    encodes to. This is what `read` puts a stored frame through, so that a result
    saved as DNG and loaded again is the frame that was saved rather than the
    linear light it was stored as.

    The inverse is taken analytically rather than by searching the forward table,
    because the forward curve is many-to-one near black - it spreads one linear
    code over some 4.5 encoded ones - so a search there has no single answer to
    return. That same flattening is the round trip's error bound: a few parts in
    65535 in the deepest shadows, and exact everywhere the curve is steep enough
    to separate its inputs.
    """
    knee, offset = _bt709_knee()
    linear = np.linspace(0.0, 1.0, _LINEAR_MAX + 1, dtype=np.float64)
    encoded = np.where(
        linear < knee / _BT709_SLOPE,
        linear * _BT709_SLOPE,
        (1.0 + offset) * linear ** _BT709_POWER - offset,
    )
    return np.rint(encoded * _LINEAR_MAX).clip(0, _LINEAR_MAX).astype(np.uint16)


class _Colorimetry(NamedTuple):
    """Everything the tag table says about what the samples mean as colour.

    Gathered into one value because the two colour spaces disagree about all of
    it at once - matrix, neutral, which camera the file claims to be, which
    profiles may be applied to it - and a writer that took them as six loose
    arguments could be asked for half of one and half of the other.
    """

    matrix: Sequence[Sequence[float]]     # XYZ under the illuminant -> this space
    forward: Optional[Sequence[Sequence[float]]]   # this space -> XYZ (D50)
    neutral: Sequence[float]              # where the white point lands here
    camera: str                           # UniqueCameraModel
    make: Optional[str]                   # EXIF Make/Model, when claiming a body
    model: Optional[str]
    profile: Optional[str]                # ProfileName / AsShotProfileName
    signature: Optional[str]              # who may develop it; None for anyone


# What the pipeline's own output is: sRGB, balanced already, developable only by
# the profile embedded beside it, and claiming no camera - see the module
# docstring for what naming the source body here used to cost.
_SRGB_COLORIMETRY = _Colorimetry(
    matrix=_XYZ_D65_TO_SRGB,
    forward=_SRGB_TO_XYZ_D50,
    neutral=(1.0, 1.0, 1.0),
    camera=CAMERA_MODEL,
    make=None,
    model=None,
    profile=_PROFILE_NAME_TEXT,
    signature=_CALIBRATION_SIGNATURE,
)

# Rows converted to camera space at a time. The matrix multiply wants floats,
# and a 24 MP frame in float64 is 1.7 GB against the 144 MB it occupies as
# samples, so it is done in bands and the frame itself is never promoted.
_CAMERA_SPACE_BAND_ROWS = 256


def _camera_colorimetry(source_path: Optional[str]) -> Optional[_Colorimetry]:
    """What `source_path`'s camera looks like as colour, or None if it cannot say.

    LibRaw holds a measured XYZ->camera matrix for every body it supports, and it
    is the same matrix Adobe's converter writes as ColorMatrix2 - checked against
    a DNG Converter file for the Z 6_2, where the two agree to every published
    digit. Taking it from the source raw rather than from a table of our own is
    what keeps this working for cameras this module has never heard of.

    None whenever that cannot be had - no rawpy, not a raw file, a body LibRaw
    has no matrix for - and the caller writes sRGB instead. The calibration
    signature is dropped rather than set to "com.adobe": the file really is
    developable by that camera's profiles, and signing it as Adobe's own
    calibration would be a claim about who measured it, made to tidy a menu.
    """
    if not _RAWPY_AVAILABLE or not source_path or not os.path.isfile(source_path):
        return None
    try:
        with open(source_path, "rb") as handle:
            with rawpy.imread(handle) as raw:
                matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    except Exception:  # pylint: disable=broad-except
        return None

    # LibRaw hands back zeros for a body it holds no calibration for, and an
    # all-zero matrix would take every pixel to black.
    if not np.isfinite(matrix).all() or not matrix.any():
        return None
    neutral = matrix @ np.asarray(_D65_WHITE, dtype=np.float64)
    if not np.isfinite(neutral).all() or neutral.min() <= 0.0:
        return None

    # rawpy exposes no camera name, but every raw format this can open is a TIFF
    # underneath, so the names come from the source's own IFD 0 - and they are
    # carried verbatim, because a converter matches its profiles against the
    # strings the camera wrote, not against a tidied version of them.
    tags = _read_ifd0(source_path) or {}

    def name(tag: int) -> Optional[str]:
        value = tags.get(tag)
        return value.strip() or None if isinstance(value, str) else None

    make, model = name(_MAKE), name(_MODEL)

    return _Colorimetry(
        matrix=tuple(tuple(row) for row in matrix),
        # No ForwardMatrix: it is camera->XYZ(D50), and would have to come from
        # the same calibration as the matrix above. LibRaw carries none, and
        # inverting the matrix would state as measured what is merely arithmetic.
        forward=None,
        neutral=tuple(neutral / neutral.max()),
        camera=model or CAMERA_MODEL,
        make=make,
        model=model,
        # No embedded profile either. The point of this mode is that the
        # converter's own profile for that camera is the right one to use.
        profile=None,
        signature=None,
    )


def _to_camera_space(linear: np.ndarray, color: _Colorimetry) -> np.ndarray:
    """Linear sRGB samples as the camera's own, filling `_LINEAR_MAX`.

    The pixels go sRGB -> XYZ -> camera and are then divided by the largest
    component of the unnormalised neutral, which is green on every Bayer sensor.
    That puts a white pixel exactly on the neutral the tags declare - green at
    full scale, the other two left where the sensor's weaker response puts them -
    which is the shape of a real raw, and is what a converter's white balance
    then undoes.

    Nothing is clipped in practice: a camera's gamut contains sRGB's with room to
    spare, so the transform of an sRGB frame stays inside 0..1. The clip is there
    for the corner where it does not, rather than to be relied on.
    """
    to_camera = np.asarray(color.matrix, dtype=np.float64)
    matrix = to_camera @ np.asarray(_SRGB_TO_XYZ_D65, dtype=np.float64)
    matrix /= float((to_camera @ np.asarray(_D65_WHITE, dtype=np.float64)).max())

    out = np.empty_like(linear)
    for start in range(0, linear.shape[0], _CAMERA_SPACE_BAND_ROWS):
        stop = start + _CAMERA_SPACE_BAND_ROWS
        # BGR in and BGR out, so the channels are reversed for the multiply and
        # reversed back after - cheaper than transposing the frame twice.
        band = linear[start:stop, :, ::-1].astype(np.float64) @ matrix.T
        np.clip(band, 0.0, _LINEAR_MAX, out=band)
        out[start:stop] = np.rint(band)[:, :, ::-1]
    return out


def _stores_scene_linear(mode: str) -> bool:
    """Whether a mode writes linear samples, as opposed to encoded ones plus a table.

    Every mode but the lossy one, which DNG restricts to 8 bits - too few for
    linear light to survive. See the module docstring's transfer-function bullet.
    """
    return mode != COMPRESSION_LOSSY


def _scene_linear(source: np.ndarray, bits: int) -> np.ndarray:
    """A display-referred frame as the 16-bit scene-linear samples DNG stores.

    A table lookup rather than the arithmetic, because the arithmetic would run
    over every pixel in float while the table has one entry per *value* a sample
    can take - 65536 of them at most, against tens of millions of pixels. It is
    also the same table the lossy mode writes, so the two paths cannot drift.
    """
    return np.take(_linearization_table(bits), source)


def _display_referred(pixels: np.ndarray) -> np.ndarray:
    """Scene-linear samples back as the display-referred frame they were saved from."""
    return np.take(_display_table(), pixels)


class _Raw(NamedTuple):
    """A tag's field data already serialised, with the TIFF `count` it stands for.

    Tags copied out of a source EXIF block arrive as bytes of a type this module
    never constructs itself - FLOAT, UNDEFINED, a vendor's own - so they are
    carried through the layout as they are rather than decoded into values and
    re-encoded. The count cannot be derived from the length alone, since it is
    values and not bytes, so it travels alongside.
    """

    payload: bytes
    count: int


def _pack(field_type: int, values) -> bytes:
    """Serialise a tag's values as little-endian TIFF field data."""
    if isinstance(values, _Raw):
        return values.payload
    if field_type == _BYTE:
        return bytes(values)
    if field_type == _ASCII:
        return values.encode("ascii", "replace") + b"\x00"
    if field_type == _SHORT:
        return np.asarray(values, dtype="<u2").tobytes()
    if field_type == _LONG:
        return np.asarray(values, dtype="<u4").tobytes()
    if field_type == _RATIONAL:
        return np.asarray(values, dtype="<u4").tobytes()
    if field_type == _SRATIONAL:
        return np.asarray(values, dtype="<i4").tobytes()
    if field_type == _FLOAT:
        return np.asarray(values, dtype="<f4").tobytes()
    raise ValueError(f"Unsupported TIFF field type: {field_type}")


def _count(field_type: int, values) -> int:
    """The TIFF `count` for a tag - the number of values, not of bytes.

    RATIONALs are stored as numerator/denominator pairs, so the pair is one
    value; ASCII counts the terminating NUL.
    """
    if isinstance(values, _Raw):
        return values.count
    if field_type == _ASCII:
        return len(values.encode("ascii", "replace")) + 1
    if field_type == _BYTE:
        return len(values)
    size = len(np.asarray(values).ravel())
    if field_type in (_RATIONAL, _SRATIONAL):
        return size // 2
    return size


def _rational_matrix(matrix) -> list:
    """Flatten a 3x3 float matrix into SRATIONAL numerator/denominator pairs."""
    flat = []
    for row in matrix:
        for value in row:
            flat.append(int(round(value * _MATRIX_DENOMINATOR)))
            flat.append(_MATRIX_DENOMINATOR)
    return flat


def _rational_vector(values) -> list:
    """Flatten a float triple into RATIONAL numerator/denominator pairs."""
    flat = []
    for value in values:
        flat.append(int(round(value * _MATRIX_DENOMINATOR)))
        flat.append(_MATRIX_DENOMINATOR)
    return flat


def _ifd_size(entry_count: int) -> int:
    """Bytes an IFD's tag table occupies: count, entries, next-IFD pointer."""
    return 2 + 12 * entry_count + 4


def _build_ifd(entries: list, base_offset: int) -> Tuple[bytes, Dict[int, int]]:
    """Lay out one IFD, and its out-of-line values, at `base_offset`.

    `entries` are (tag, type, values) triples. Returns the assembled bytes and,
    for every tag, the file-absolute position at which its values begin -
    whether that is inline in the 4-byte entry or out in the value block. The
    caller patches through those positions the offsets it could not know when it
    built the entries: where each strip landed, and where the preview SubIFD
    starts. Both are chicken-and-egg, since recording them is what fixes the
    size of the table they are recorded in.

    TIFF requires the entries be sorted by tag, and any value longer than the
    four bytes an entry has room for to live elsewhere in the file and be
    referenced by offset.
    """
    entries = sorted(entries, key=lambda item: item[0])
    values_offset = base_offset + _ifd_size(len(entries))

    table = bytearray()
    values = bytearray()
    positions: Dict[int, int] = {}

    table += struct.pack("<H", len(entries))
    for tag, field_type, raw in entries:
        payload = _pack(field_type, raw)
        count = _count(field_type, raw)
        entry = struct.pack("<HHI", tag, field_type, count)
        if len(payload) <= 4:
            table += entry + payload.ljust(4, b"\x00")
            # The value sits inline in the entry itself; its position is
            # file-absolute, hence measured from the IFD's own base.
            positions[tag] = base_offset + len(table) - 4
        else:
            positions[tag] = values_offset + len(values)
            table += entry + struct.pack("<I", positions[tag])
            values += payload
            if len(values) % 2:
                values += b"\x00"  # keep the next value word-aligned
    table += struct.pack("<I", 0)  # no further IFDs; DNG uses SubIFD trees

    return bytes(table + values), positions


# ----------------------------------------------------------------------
# Source EXIF
# ----------------------------------------------------------------------
# An EXIF block is itself a TIFF stream, which is the whole reason it can be
# carried into a DNG: the tags are already in the right shape, and what has to
# change is only where they sit. A tag's *values* travel unaltered, but every
# offset in the source is measured from the start of that block, so a tag whose
# payload is an offset - or contains one - cannot simply be copied.
#
# Those are the ones left behind. The IFD pointers are re-created by the writer,
# since it is the one that knows where the sub-IFDs landed, and the embedded
# thumbnail is a JPEG the DNG has no use for - it carries its own preview.
_UNRELOCATABLE_TAGS = frozenset({
    _EXIF_IFD, _GPS_IFD, _INTEROP_IFD,
    _JPEG_INTERCHANGE_FORMAT, _JPEG_INTERCHANGE_LENGTH,
})

# MakerNote is the awkward case, and it is decided per note rather than by that
# set. Most vendors measure the offsets inside a note from the start of the file
# it was written into, so a copy anywhere else decodes as noise; some write a
# note that carries its own TIFF header and is measured from *that*, and such a
# note travels intact. `_relocatable_maker_note` is where the two are told apart.
#
# Nikon's is the shape recognised: `Nikon\0`, two version bytes, a 16-bit pad,
# then a TIFF header at offset 10. It is worth recognising because the note is
# where a Nikon records the lens, the shutter count, the focus distance and the
# picture control - none of which has a standard EXIF tag, and all of which a
# reader shows for the source raw and would otherwise not show for the DNG.
_MAKER_NOTE_NIKON_PREFIX = b"Nikon\x00"
_MAKER_NOTE_NIKON_TIFF_AT = 10

# What such a note may not bring with it. A Nikon's holds a preview IFD of its
# own, and inside it a JPEG of the frame the note came from - a few hundred
# kilopixels of *one source frame*, at a fraction of the result's size.
#
# A DNG is a raw file, so a browser looks for an embedded preview before it will
# decode anything, and it does not care which IFD the preview came out of. Left
# in, that JPEG is what XnView shows in place of the fused result whenever the
# fast-load preview is switched off and there is no other preview to prefer. So
# it is removed - the offset that points at it, and the bytes themselves, since
# a reader that hunts for a JPEG start marker would otherwise still find them.
# Everything the note is carried *for* is elsewhere in it and is untouched.
_MAKER_NOTE_PREVIEW_IFD = 0x0011

# Tags of the source's IFD 0 that describe the shot rather than the source file,
# and so belong in the DNG's IFD 0 too.
#
# Make and Model are deliberately not among them, and that is a correction: they
# travelled once, and it is what made a saved stack render wrong.
#
# In a camera DNG those two name the camera whose mosaic the file holds, and a
# converter reads them to decide which camera profile to develop it with. This
# file holds no mosaic - the pixels left camera space during the develop - so
# naming the source camera there does not describe the data, it misdirects the
# reader. Luminar Neo given "NIKON Z 6_2" offers that camera's profile and
# renders a finished sRGB frame through a Nikon sensor matrix, which throws the
# whole image towards magenta.
#
# Which camera shot the frames is still recorded - in the EXIF IFD, and in the
# MakerNote carried with it. What it no longer does is answer "what are these
# pixels", because that is the question Make and Model are read as answering.
#
# The resolution tags travel because they are what a print pipeline reads a
# nominal size out of, and a raw converter writes them into a DNG for the same
# reason; they say nothing about the strips, so nothing here contradicts them.
# DateTimeOriginal is here as well as in the EXIF IFD because TIFF/EP puts it in
# both, and a source that fills both should not come out of a save filling one -
# DateTime stays the writer's own, since it is when *this* file was created.
# Orientation is deliberately absent: a fused result is already the right way up,
# and copying a rotated source's value would turn it.
_CAMERA_IFD0_TAGS = frozenset({
    _IMAGE_DESCRIPTION, _ARTIST, _COPYRIGHT,
    _X_RESOLUTION, _Y_RESOLUTION, _RESOLUTION_UNIT, _DATE_TIME_ORIGINAL,
})

# numpy element type of each TIFF field type, for the byte-order conversion. The
# two byte-stream types are absent: ASCII and UNDEFINED have no element wider
# than a byte, so they read the same either way.
_TIFF_ELEMENT = {
    _BYTE: "u1", _SHORT: "u2", _LONG: "u4", _RATIONAL: "u4",
    6: "i1", 8: "i2", 9: "i4", _SRATIONAL: "i4", 11: "f4", 12: "f8",
}


def _tiff_header(blob: bytes) -> Optional[Tuple[str, int]]:
    """The byte order and first-IFD offset of a TIFF stream, or None.

    Camera EXIF is big-endian about as often as little-endian - it follows the
    camera, not the host - so both are accepted here even though everything this
    module writes is little-endian.
    """
    if len(blob) < 8:
        return None
    if blob[:2] == b"II":
        endian = "<"
    elif blob[:2] == b"MM":
        endian = ">"
    else:
        return None
    magic, offset = struct.unpack_from(endian + "HI", blob, 2)
    return (endian, offset) if magic == 42 else None


def _blob_entries(blob: bytes, endian: str, offset: int) -> List[Tuple[int, int, int, bytes]]:
    """(tag, type, count, payload) of the IFD at `offset`, payloads unconverted.

    Anything that does not fit inside the block - a truncated value, an offset
    past the end - is skipped rather than raising, because the block came from a
    file this module did not write and a bad tag must not cost the save.
    """
    entries: List[Tuple[int, int, int, bytes]] = []
    if offset <= 0 or offset + 2 > len(blob):
        return entries
    (count,) = struct.unpack_from(endian + "H", blob, offset)
    table = offset + 2
    if table + 12 * count > len(blob):
        return entries

    for index in range(table, table + 12 * count, 12):
        tag, field_type, values = struct.unpack_from(endian + "HHI", blob, index)
        size = _READ_TYPE_SIZE.get(field_type)
        if size is None:
            continue
        total = size * values
        if total <= 4:
            payload = blob[index + 8:index + 8 + total]
        else:
            (position,) = struct.unpack_from(endian + "I", blob, index + 8)
            payload = blob[position:position + total]
        if len(payload) < total:
            continue
        entries.append((tag, field_type, values, payload))
    return entries


def _little_endian(payload: bytes, field_type: int, endian: str) -> bytes:
    """A field's payload in the byte order this module writes."""
    element = _TIFF_ELEMENT.get(field_type)
    if endian == "<" or element is None:
        return payload
    return np.frombuffer(payload, dtype=np.dtype(">" + element)).astype(
        np.dtype("<" + element)).tobytes()


def _copied_entry(tag: int, field_type: int, count: int,
                  payload: bytes, endian: str) -> Tuple[int, int, _Raw]:
    """One source tag as an entry the layout can place."""
    return (tag, field_type, _Raw(_little_endian(payload, field_type, endian), count))


def _relocatable_maker_note(payload: bytes) -> bool:
    """Whether a MakerNote can be written at a new offset and still decode.

    True only for a note that opens with a TIFF header of its own, because that
    is what says its internal offsets are measured from the note rather than
    from the file around it. Nikon writes one; the check is deliberately narrow,
    since a note wrongly judged self-contained is worse than one left behind - a
    reader would parse it and report another shot's values as this one's.

    The shape is not taken on trust either. The note's IFD is walked and every
    out-of-line value has to fall inside the note; one that reaches past its end
    is a note whose offsets mean something else, and it is dropped.
    """
    if not payload.startswith(_MAKER_NOTE_NIKON_PREFIX):
        return False

    inner = payload[_MAKER_NOTE_NIKON_TIFF_AT:]
    header = _tiff_header(inner)
    if header is None:
        return False
    endian, offset = header

    if offset <= 0 or offset + 2 > len(inner):
        return False
    (count,) = struct.unpack_from(endian + "H", inner, offset)
    table = offset + 2
    if count == 0 or table + 12 * count + 4 > len(inner):
        return False

    for index in range(table, table + 12 * count, 12):
        _tag, field_type, values = struct.unpack_from(endian + "HHI", inner, index)
        size = _READ_TYPE_SIZE.get(field_type)
        if size is None:
            continue
        total = size * values
        if total > 4:
            (position,) = struct.unpack_from(endian + "I", inner, index + 8)
            if position + total > len(inner):
                return False
    return True


def _without_maker_note_preview(payload: bytes) -> bytes:
    """The same note with the preview image it embeds taken out.

    See `_MAKER_NOTE_PREVIEW_IFD` for why one has to go. The edits are made
    where the values sit rather than by re-laying the note out: every offset
    inside it is measured from its own header, so anything that moved would
    point at the wrong thing. The note keeps its length, and the space the
    preview occupied is left as zeros.

    A note with no preview in it comes back unchanged, as does one whose preview
    IFD does not parse - there is nothing to remove and nothing to risk.
    """
    inner = payload[_MAKER_NOTE_NIKON_TIFF_AT:]
    header = _tiff_header(inner)
    if header is None:
        return payload
    endian, offset = header

    preview_ifd = None
    for tag, field_type, count, value in _blob_entries(inner, endian, offset):
        if tag == _MAKER_NOTE_PREVIEW_IFD and field_type == _LONG and count == 1:
            (preview_ifd,) = struct.unpack(endian + "I", value)
    if not preview_ifd or preview_ifd + 2 > len(inner):
        return payload

    (entries,) = struct.unpack_from(endian + "H", inner, preview_ifd)
    table = preview_ifd + 2
    if table + 12 * entries + 4 > len(inner):
        return payload

    note = bytearray(payload)
    start = length = 0
    for index in range(table, table + 12 * entries, 12):
        tag, _field_type, count = struct.unpack_from(endian + "HHI", inner, index)
        if tag not in (_JPEG_INTERCHANGE_FORMAT, _JPEG_INTERCHANGE_LENGTH) or count != 1:
            continue
        (value,) = struct.unpack_from(endian + "I", inner, index + 8)
        if tag == _JPEG_INTERCHANGE_FORMAT:
            start = value
        else:
            length = value
        struct.pack_into(endian + "I", note, _MAKER_NOTE_NIKON_TIFF_AT + index + 8, 0)

    if start and length and start + length <= len(inner):
        at = _MAKER_NOTE_NIKON_TIFF_AT + start
        note[at:at + length] = b"\x00" * length
    return bytes(note)


def _carried_maker_note(payload: bytes) -> Optional[bytes]:
    """A source MakerNote as the DNG should hold it, or None to leave it behind."""
    if not _relocatable_maker_note(payload):
        return None
    return _without_maker_note_preview(payload)


def exif_camera_entries(exif: Optional[bytes]) -> List[Tuple[int, int, _Raw]]:
    """The source's IFD 0 tags that describe the camera, ready for the DNG's IFD 0."""
    header = _tiff_header(exif) if exif else None
    if header is None:
        return []
    endian, offset = header
    return [_copied_entry(tag, field_type, count, payload, endian)
            for tag, field_type, count, payload in _blob_entries(exif, endian, offset)
            if tag in _CAMERA_IFD0_TAGS]


def exif_sub_ifds(exif: Optional[bytes]) -> List[Tuple[int, List[Tuple[int, int, _Raw]]]]:
    """(pointer tag, entries) of each sub-IFD of the source block worth keeping.

    The EXIF IFD is where the shot itself is recorded - exposure, aperture, ISO,
    lens, the original date - and the GPS IFD where it was taken. Both are
    re-emitted whole apart from the tags that cannot survive being moved; the
    Interoperability IFD is not, since it says only which flavour of EXIF the
    source file claimed to be, which stops being true once it is copied.
    """
    header = _tiff_header(exif) if exif else None
    if header is None:
        return []
    endian, offset = header

    pointers: Dict[int, int] = {}
    for tag, field_type, count, payload in _blob_entries(exif, endian, offset):
        if tag in (_EXIF_IFD, _GPS_IFD) and count == 1 and field_type == _LONG:
            (pointers[tag],) = struct.unpack(endian + "I", payload)

    sub_ifds = []
    for pointer in (_EXIF_IFD, _GPS_IFD):
        if pointer not in pointers:
            continue
        entries = []
        for tag, field_type, count, payload in _blob_entries(exif, endian, pointers[pointer]):
            if tag == _MAKER_NOTE:
                # Kept, dropped or trimmed on its own merits; the length never
                # changes, so the count the source declared still stands.
                carried = _carried_maker_note(payload)
                if carried is None:
                    continue
                payload = carried
            elif tag in _UNRELOCATABLE_TAGS:
                continue
            entries.append(_copied_entry(tag, field_type, count, payload, endian))
        if entries:
            sub_ifds.append((pointer, entries))
    return sub_ifds


# ----------------------------------------------------------------------
# Strip codecs
# ----------------------------------------------------------------------
def _encode_strip_lossless(strip: np.ndarray, bits: int) -> bytes:
    """Lossless Huffman JPEG for one strip of file-order samples.

    The strip's own geometry is what reaches the codestream - one JPEG component
    per DNG sample - because that is what readers assume; see utils.ljpeg for
    why the alternative the spec permits does not survive contact with LibRaw.
    """
    return ljpeg.encode(strip, bits=bits)


def _encode_strip_lossy(strip: np.ndarray, quality: int) -> bytes:
    """Baseline DCT JPEG for one strip of 8-bit file-order samples.

    OpenCV encodes from BGR, so a three-sample strip - which is held here in
    file order, red first - is reversed on the way in. That way the JPEG's own
    first component is the red one, and a reader that simply decodes it gets the
    samples in the order the strip claims rather than mirrored.
    """
    params = [
        cv2.IMWRITE_JPEG_QUALITY, int(quality),
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
    ]
    if strip.ndim == 3 and strip.shape[2] == 3:
        strip = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(strip), params)
    if not ok:
        raise ValueError("OpenCV could not encode a lossy DNG strip.")
    return buffer.tobytes()


def _decode_strip_lossless(payload: bytes) -> Optional[np.ndarray]:
    """Flat sample array from a lossless-JPEG strip, or None if it will not decode.

    The internal geometry is deliberately not checked against the strip's: DNG
    lets them differ, so only the total sample count means anything and the
    caller is the one that knows what it should be.
    """
    if not _LJPEG_AVAILABLE:
        return None
    try:
        return np.asarray(imagecodecs.ljpeg_decode(payload)).ravel()
    except Exception:  # pylint: disable=broad-except
        return None


def _decode_strip_lossy(payload: bytes, samples: int) -> Optional[np.ndarray]:
    """Flat sample array from a lossy-JPEG strip, or None if it will not decode.

    The mirror of `_encode_strip_lossy`: OpenCV hands back BGR, which is
    reversed again to put the samples in the file order the strip describes.
    """
    flags = cv2.IMREAD_COLOR if samples == 3 else cv2.IMREAD_GRAYSCALE
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    if decoded is None:
        return None
    if samples == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return np.asarray(decoded).ravel()


# ----------------------------------------------------------------------
# Fast-load preview
# ----------------------------------------------------------------------
def _preview_jpeg(source: np.ndarray, bits: int) -> Optional[Tuple[bytes, int, int]]:
    """A half-resolution baseline JPEG of the image, with its dimensions.

    `source` is the full-resolution frame as the pipeline holds it - BGR, or a
    single plane - rather than in DNG's file order, so that the preview costs a
    downscale of the frame and not a full-frame channel swap first. Returns None
    when the image is too small for a preview to save a reader any work, or when
    OpenCV declines to encode it.
    """
    height, width = source.shape[:2]
    target_w = max(1, width // _PREVIEW_DIVISOR)
    target_h = max(1, height // _PREVIEW_DIVISOR)
    if max(target_w, target_h) < _PREVIEW_MIN_EDGE:
        return None

    # INTER_AREA is the right filter for a pure downscale: it averages every
    # source pixel that falls in a target one, so the preview does not alias the
    # fine detail a focus-stacked frame is full of.
    small = cv2.resize(source, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if bits == 16:
        # A JPEG preview is 8-bit whatever the raw is. Scaling by 257 rather
        # than shifting keeps full scale at full scale. Done on the downscaled
        # image, so the float temporary is a quarter of the frame.
        small = np.rint(small.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)

    # The preview is a rendering, not raw data, so it is always three-channel:
    # a viewer showing it never has to know the main image was monochrome.
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    ok, buffer = cv2.imencode(
        ".jpg", small,
        [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_QUALITY,
         cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420],
    )
    if not ok:
        return None
    return buffer.tobytes(), target_w, target_h


def _preview_entries(width: int, height: int, byte_count: int,
                     software: str, now: str) -> list:
    """IFD entries for the preview SubIFD, with a placeholder strip offset.

    `NewSubFileType` = 1 is what marks this as the *primary* preview, which is
    the one a reader is meant to display by default; PhotometricInterpretation
    6 with 8/8/8 samples is the combination the spec reserves for a baseline
    JPEG preview, and the `Preview*` tags say who rendered it and in which
    colour space, so no reader has to guess at either.
    """
    return [
        (_NEW_SUBFILE_TYPE, _LONG, [1]),
        (_IMAGE_WIDTH, _LONG, [width]),
        (_IMAGE_LENGTH, _LONG, [height]),
        (_BITS_PER_SAMPLE, _SHORT, [8, 8, 8]),
        (_COMPRESSION, _SHORT, [_COMPRESSION_JPEG]),
        (_PHOTOMETRIC, _SHORT, [_YCBCR]),
        (_STRIP_OFFSETS, _LONG, [0]),  # patched once the layout is fixed
        (_ORIENTATION, _SHORT, [1]),
        (_SAMPLES_PER_PIXEL, _SHORT, [3]),
        (_ROWS_PER_STRIP, _LONG, [height]),  # the whole JPEG is one strip
        (_STRIP_BYTE_COUNTS, _LONG, [byte_count]),
        (_PLANAR_CONFIG, _SHORT, [1]),
        (_YCBCR_SUB_SAMPLING, _SHORT, [2, 2]),  # matches the 4:2:0 encode above
        (_PREVIEW_APPLICATION_NAME, _ASCII, CAMERA_MODEL),
        (_PREVIEW_APPLICATION_VERSION, _ASCII, software),
        (_PREVIEW_COLOR_SPACE, _LONG, [_PREVIEW_COLOR_SPACE_SRGB]),
        (_PREVIEW_DATE_TIME, _ASCII, now),
    ]


def _rows_per_strip(mode: str, width: int, height: int, samples: int, bits: int) -> int:
    """How many rows one strip holds, which the codec decides.

    An uncompressed image is cut into several strips, because one strip spanning
    a 24 MP 16-bit frame is a 144 MB run that some readers handle poorly.

    A compressed image is written as a **single** strip, and that is not a
    preference. In a stripped - as opposed to tiled - DNG, LibRaw reads the
    second and later compressed strips by continuing on from wherever the codec
    left the file pointer rather than by consulting `StripOffsets`, so anything
    the decoder does not consume byte-for-byte desynchronises it: a multi-strip
    lossless file decodes its first strip and renders the rest black. One strip
    per image removes the question, at the cost of holding the frame and its
    encoded form at once - which the uncompressed path does anyway, since the
    whole file is assembled in memory before it is written.
    """
    if mode != COMPRESSION_NONE:
        return height
    row_bytes = width * samples * (bits // 8)
    return max(1, min(height, _STRIP_TARGET_BYTES // max(1, row_bytes)))


def _resolve_options(compression: Optional[str], lossy_quality: Optional[int],
                     fast_load: Optional[bool]) -> Tuple[str, int, bool]:
    """Fill unset encode options from the module settings, and validate them."""
    mode = _compression if compression is None else str(compression)
    if mode not in VALID_COMPRESSIONS:
        raise ValueError(
            f"Unknown DNG compression mode: {mode!r}. Expected one of {VALID_COMPRESSIONS}."
        )
    if mode == COMPRESSION_LOSSLESS and not _LJPEG_AVAILABLE:
        raise RuntimeError(lossless_unavailable_reason())

    quality = _lossy_quality if lossy_quality is None else int(lossy_quality)
    quality = max(MIN_LOSSY_QUALITY, min(MAX_LOSSY_QUALITY, quality))
    return mode, quality, _fast_load if fast_load is None else bool(fast_load)


def _source_view(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """A BGR or single-plane view of the image, and how many samples it has.

    The channel order is left as the pipeline's, and the strips are what get
    turned into DNG's file order - see `_file_order`. Converting the whole frame
    here instead would allocate a second copy of the largest array on the save
    path, for a reordering that a strip needs one strip's worth of at a time.

    An alpha channel is dropped rather than written as an extra sample no
    converter would look at - DNG raw data has no place for one - and that is the
    one case where a full copy is unavoidable.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    if image.ndim == 2:
        return image, 1
    if image.ndim == 3 and image.shape[2] == 3:
        return image, 3
    raise ValueError(f"Unsupported image shape for DNG: {image.shape}")


def _file_order(rows: np.ndarray, samples: int) -> np.ndarray:
    """One strip's rows as contiguous file-order samples.

    DNG, like TIFF, stores a pixel's channels in order, so the pipeline's BGR
    becomes RGB. A strip cut from a contiguous frame is already contiguous, so
    the single-sample case usually costs nothing at all.
    """
    if samples == 3:
        return cv2.cvtColor(rows, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rows)


def _strip_spans(height: int, rows_per_strip: int) -> List[Tuple[int, int]]:
    """The (first, stop) row range of every strip, in file order."""
    return [(first, min(first + rows_per_strip, height))
            for first in range(0, height, rows_per_strip)]


def _prepare(image: np.ndarray,
             compression: Optional[str],
             lossy_quality: Optional[int],
             fast_load: Optional[bool]) -> Tuple[np.ndarray, int, int, str, int, bool]:
    """Validate the image and settle the codec: what is written, and how.

    Returns the frame as it will be stored - which for the lossy mode is a
    narrowed copy - along with its sample count and depth, and the resolved
    encode options.
    """
    if image is None:
        raise ValueError("No image to encode.")
    mode, quality, want_preview = _resolve_options(compression, lossy_quality, fast_load)

    source, samples = _source_view(image)

    if source.dtype == np.uint8:
        bits = 8
    elif source.dtype == np.uint16:
        bits = 16
    else:
        raise ValueError(f"Unsupported image dtype for DNG: {source.dtype}")

    if mode == COMPRESSION_LOSSY and samples == 1:
        # LibRaw's lossy-DNG path reads three components per pixel unconditionally,
        # so a single-sample lossy file is one no mainstream raw reader will open.
        # Falling back keeps the promise that every DNG written here is readable,
        # which matters more than honouring a codec choice for a mono result.
        fallback = COMPRESSION_LOSSLESS if _LJPEG_AVAILABLE else COMPRESSION_NONE
        print(f"[DNG] Lossy compression cannot be read back for a monochrome image; "
              f"writing {fallback} instead.", flush=True)
        mode = fallback

    if mode == COMPRESSION_LOSSY and bits == 16:
        print("[DNG] Lossy compression is 8-bit only in DNG; saving 8-bit. "
              "Use the lossless or uncompressed mode to keep 16 bits.", flush=True)
        source = np.rint(source.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
        bits = 8

    height, width = source.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    return source, samples, bits, mode, quality, want_preview


def _tag_table(source: np.ndarray, samples: int, bits: int, mode: str,
               software: str, counts: Sequence[int], rows_per_strip: int,
               preview: Optional[Tuple[bytes, int, int]],
               exif: Optional[bytes],
               color: _Colorimetry = _SRGB_COLORIMETRY) -> Tuple[bytearray, List[int]]:
    """Everything ahead of the pixel data, and where each block after it starts.

    The header, IFD 0 and the SubIFDs are laid out together because their sizes
    depend only on the tags, not on the pixels: once they are known, so is the
    offset of every strip, which is what the offsets inside them have to record.
    The returned list is the file position of each strip followed by that of the
    preview, so the caller writes the blocks in that order and nothing has to be
    seeked back to.
    """
    height, width = source.shape[:2]
    now = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    backward = (_DNG_BACKWARD_VERSION_LOSSY_BYTES if mode == COMPRESSION_LOSSY
                else _DNG_BACKWARD_VERSION_BYTES)
    entries = [
        (_NEW_SUBFILE_TYPE, _LONG, [0]),  # this is the full-resolution image
        (_IMAGE_WIDTH, _LONG, [width]),
        (_IMAGE_LENGTH, _LONG, [height]),
        (_BITS_PER_SAMPLE, _SHORT, [bits] * samples),
        (_COMPRESSION, _SHORT, [_COMPRESSION_CODES[mode]]),
        (_PHOTOMETRIC, _SHORT, [_LINEAR_RAW]),
        # The source camera's own names, but only in camera space, where the
        # samples really are that camera's. Naming it over sRGB samples is what
        # made a converter offer a sensor profile for data that is not sensor
        # data - see the module docstring.
        (_MAKE, _ASCII, color.make or CAMERA_MODEL),
        (_MODEL, _ASCII, color.model or CAMERA_MODEL),
        (_STRIP_OFFSETS, _LONG, [0] * len(counts)),  # patched once the layout is fixed
        (_ORIENTATION, _SHORT, [1]),
        (_SAMPLES_PER_PIXEL, _SHORT, [samples]),
        (_ROWS_PER_STRIP, _LONG, [rows_per_strip]),
        (_STRIP_BYTE_COUNTS, _LONG, list(counts)),
        (_PLANAR_CONFIG, _SHORT, [1]),  # interleaved
        (_SOFTWARE, _ASCII, software),
        (_DATE_TIME, _ASCII, now),
        (_SAMPLE_FORMAT, _SHORT, [1] * samples),  # unsigned integer
        (_DNG_VERSION, _BYTE, _DNG_VERSION_BYTES),
        (_DNG_BACKWARD_VERSION, _BYTE, backward),
        (_UNIQUE_CAMERA_MODEL, _ASCII, color.camera),
        (_WHITE_LEVEL, _LONG, [_LINEAR_MAX] * samples),
    ]

    if not _stores_scene_linear(mode):
        # The samples are still display-referred, so the curve has to be
        # declared - and the mode that needs it is the one whose readers all
        # support it. Everywhere else the pixels are linear already and there is
        # no encoding left to describe; writing an identity table instead would
        # only give a reader that ignores the tag the same wrong answer.
        entries.append((_LINEARIZATION_TABLE, _SHORT, _linearization_table(bits)))

    # Neither tag is colorimetry, so both are written for a monochrome result
    # too: they say the tones are finished, which is as true of one channel as
    # of three.
    entries.append((_BASELINE_EXPOSURE, _SRATIONAL, [0, _MATRIX_DENOMINATOR]))
    entries.append((_BASELINE_EXPOSURE_OFFSET, _SRATIONAL, [0, _MATRIX_DENOMINATOR]))
    entries.append((_DEFAULT_BLACK_RENDER, _LONG, [_DEFAULT_BLACK_RENDER_NONE]))

    if samples == 3:
        # Colorimetry only means anything for a colour image. A monochrome DNG
        # carries no matrix and no neutral, which is what DNG 1.4 expects.
        entries.append((_COLOR_MATRIX_1, _SRATIONAL, _rational_matrix(color.matrix)))
        if color.forward is not None:
            entries.append((_FORWARD_MATRIX_1, _SRATIONAL, _rational_matrix(color.forward)))
        entries.append((_CALIBRATION_ILLUMINANT_1, _SHORT, [_ILLUMINANT_D65]))
        entries.append((_AS_SHOT_NEUTRAL, _RATIONAL, _rational_vector(color.neutral)))
        # Explicitly nothing between the samples and that matrix - see
        # `_IDENTITY_MATRIX` for why the defaults are not relied on.
        entries.append((_CAMERA_CALIBRATION_1, _SRATIONAL, _rational_matrix(_IDENTITY_MATRIX)))
        entries.append((_ANALOG_BALANCE, _RATIONAL, [1, 1, 1, 1, 1, 1]))
        # The rest of the embedded profile. Without these a converter has no
        # profile to use and falls back to its own default rendering, which is
        # built to develop a camera's scene-referred data and puts a second
        # tone curve on a result that is already finished.
        #
        # A camera-space file carries none of it, and that is the point of the
        # mode rather than an omission: the samples are the camera's, so the
        # converter's own profile for that camera is the correct one to apply and
        # there is nothing better to embed beside it.
        if color.profile is not None:
            entries.append((_PROFILE_NAME, _ASCII, color.profile))
            entries.append((_PROFILE_EMBED_POLICY, _LONG, [_PROFILE_EMBED_ALLOW_COPYING]))
            entries.append((_PROFILE_TONE_CURVE, _FLOAT, _IDENTITY_TONE_CURVE))
            # Naming the embedded profile as the one the file was "shot" with is
            # what makes a converter select it rather than whichever of its own
            # it would otherwise default to.
            entries.append((_AS_SHOT_PROFILE_NAME, _ASCII, color.profile))
        # The matching pair of signatures is what stops a converter substituting
        # a camera profile afterwards. Absent in camera space, where a camera
        # profile is exactly what should be substitutable.
        if color.signature is not None:
            entries.append((_CAMERA_CALIBRATION_SIGNATURE, _ASCII, color.signature))
            entries.append((_PROFILE_CALIBRATION_SIGNATURE, _ASCII, color.signature))
    sub_ifds = exif_sub_ifds(exif)
    for pointer, _ in sub_ifds:
        entries.append((pointer, _LONG, [0]))  # patched once the layout is fixed
    # After the writer's own tags, so that a camera named by the source block
    # replaces the placeholder Make and Model rather than colliding with them.
    entries.extend(exif_camera_entries(exif))
    if preview is not None:
        entries.append((_SUB_IFDS, _LONG, [0]))  # patched once the layout is fixed

    # A tag may only appear once in an IFD, and the later entry is the one meant.
    entries = list({entry[0]: entry for entry in entries}.values())

    # IFD 0 sits straight after the 8-byte TIFF header; the EXIF, GPS and preview
    # IFDs follow it, and the pixel data follows them all, so that every offset
    # any of them records points forward into a block whose size is already known.
    main_ifd, main_positions = _build_ifd(entries, 8)
    header = bytearray(struct.pack("<2sHI", b"II", 42, 8))
    header += main_ifd

    for pointer, sub_entries in sub_ifds:
        struct.pack_into("<I", header, main_positions[pointer], len(header))
        header += _build_ifd(sub_entries, len(header))[0]

    preview_positions: Dict[int, int] = {}
    if preview is not None:
        payload, preview_width, preview_height = preview
        preview_ifd, preview_positions = _build_ifd(
            _preview_entries(preview_width, preview_height, len(payload), software, now),
            len(header),
        )
        struct.pack_into("<I", header, main_positions[_SUB_IFDS], len(header))
        header += preview_ifd

    starts: List[int] = []
    position = len(header)
    for index, count in enumerate(counts):
        struct.pack_into("<I", header, main_positions[_STRIP_OFFSETS] + 4 * index, position)
        starts.append(position)
        position += count

    if preview is not None:
        struct.pack_into("<I", header, preview_positions[_STRIP_OFFSETS], position)
        starts.append(position)

    return header, starts


def _resolve_color(color_space: Optional[str], source_path: Optional[str],
                   samples: int, mode: str) -> _Colorimetry:
    """The colorimetry a write will use, falling back to sRGB where it must.

    Camera space needs three things at once: the mode asked for, a colour result
    to transform, and a source raw whose matrix can be read. Any of them missing
    and the write is sRGB instead - a valid file describing what it holds, rather
    than a refusal or, worse, camera tags over sRGB samples.

    The fallback is announced. A file that quietly came out in a different colour
    space than the one selected would look right in a converter and wrong only
    where it mattered, which is the failure this whole mode exists to fix.
    """
    wanted = _color_space if color_space is None else color_space
    if wanted not in VALID_COLOR_SPACES:
        raise ValueError(
            f"Unknown DNG colour space: {wanted!r}. Expected one of {VALID_COLOR_SPACES}."
        )
    if wanted != COLOR_CAMERA:
        return _SRGB_COLORIMETRY

    if samples != 3:
        print("[DNG] Camera colour space needs a colour image; writing sRGB instead.",
              flush=True)
        return _SRGB_COLORIMETRY
    if mode == COMPRESSION_LOSSY:
        # The lossy mode keeps display-referred samples, and camera space is a
        # transform of linear ones. Undoing the curve to convert and re-applying
        # it to store would put the DCT on values it was never meant for.
        print("[DNG] Camera colour space cannot be written lossily; writing sRGB instead.",
              flush=True)
        return _SRGB_COLORIMETRY

    color = _camera_colorimetry(source_path)
    if color is None:
        name = os.path.basename(source_path) if source_path else "no source"
        print(f"[DNG] No camera colour matrix available ({name}); writing sRGB instead.",
              flush=True)
        return _SRGB_COLORIMETRY
    return color


def _emit(handle, image: np.ndarray, software: str,
          compression: Optional[str], lossy_quality: Optional[int],
          fast_load: Optional[bool], exif: Optional[bytes],
          color_space: Optional[str] = None,
          source_path: Optional[str] = None) -> None:
    """Write a complete linear DNG to an open binary file object.

    Nothing larger than one strip is held on top of the frame itself: the tag
    table is sized before any pixels are touched, so the strips can be converted
    and handed to the file one at a time rather than accumulated into a copy of
    the whole image. An uncompressed 24 MP 16-bit save therefore peaks at a few
    megabytes over the frame, where assembling the file in memory first cost two
    further copies of it.

    The compressed modes still hold their one strip whole, because that strip is
    the entire image - see `_rows_per_strip` for why they cannot be split.
    """
    source, samples, bits, mode, quality, want_preview = _prepare(
        image, compression, lossy_quality, fast_load)

    # Rendered first, and so from the display-referred frame: the preview is a
    # picture for a viewer to show and is tagged sRGB, not raw data. Made after
    # the linearisation below it would be a picture of linear light, which is
    # the same frame with its shadows crushed.
    preview = _preview_jpeg(source, bits) if want_preview else None

    color = _resolve_color(color_space, source_path, samples, mode)

    if _stores_scene_linear(mode):
        # Not in place: `_source_view` may have handed back a view of the
        # caller's own array, and a save must not rewrite the frame it was given.
        source = _scene_linear(source, bits)
        bits = 16
        if color is not _SRGB_COLORIMETRY:
            # On the linear samples, never the display-referred ones: the matrix
            # is a statement about light, and applying it to encoded values would
            # mix channels that have each been bent by the curve first.
            source = _to_camera_space(source, color)

    height, width = source.shape[:2]
    rows_per_strip = _rows_per_strip(mode, width, height, samples, bits)
    spans = _strip_spans(height, rows_per_strip)

    if mode == COMPRESSION_NONE:
        # The size of an uncompressed strip follows from its geometry, so the
        # layout is known without the pixels ever being materialised.
        payloads: Optional[List[bytes]] = None
        counts = [(stop - first) * width * samples * (bits // 8) for first, stop in spans]
    else:
        payloads = []
        for first, stop in spans:
            strip = _file_order(source[first:stop], samples)
            if mode == COMPRESSION_LOSSLESS:
                payloads.append(_encode_strip_lossless(strip, bits))
            else:
                payloads.append(_encode_strip_lossy(strip, quality))
        counts = [len(blob) for blob in payloads]

    header, _ = _tag_table(source, samples, bits, mode, software, counts,
                           rows_per_strip, preview, exif, color)
    handle.write(header)

    if payloads is None:
        for first, stop in spans:
            handle.write(_file_order(source[first:stop], samples).tobytes())
    else:
        for blob in payloads:
            handle.write(blob)

    if preview is not None:
        handle.write(preview[0])


def encode(image: np.ndarray,
           software: str = CAMERA_MODEL,
           compression: Optional[str] = None,
           lossy_quality: Optional[int] = None,
           fast_load: Optional[bool] = None,
           exif: Optional[bytes] = None) -> bytes:
    """Encode a BGR (or greyscale) image as a linear DNG.

    The samples are written at whatever depth the array carries, 8 or 16 bits,
    and it is the tags that describe what they mean; see the module docstring for
    which ones and why. `compression`, `lossy_quality` and `fast_load` default
    to the module settings, so a caller that has no opinion about the codec does
    not have to form one. `exif` is a source file's EXIF block, as a bare TIFF
    stream, and is relocated into an EXIF IFD of the written file.

    The lossy mode is 8-bit only in DNG, so a 16-bit image handed to it is
    narrowed here, with a line on stdout - the same way image_utils announces a
    container that cannot hold the depth it was given.

    `write` is the path a save takes; this exists for callers that want the
    bytes, and pays for a copy of the whole file to hand them over.
    """
    buffer = io.BytesIO()
    _emit(buffer, image, software, compression, lossy_quality, fast_load, exif)
    return buffer.getvalue()


def write(file_path: str,
          image: np.ndarray,
          software: str = CAMERA_MODEL,
          compression: Optional[str] = None,
          lossy_quality: Optional[int] = None,
          fast_load: Optional[bool] = None,
          exif: Optional[bytes] = None,
          color_space: Optional[str] = None,
          source_path: Optional[str] = None) -> bool:
    """Encode and write a DNG file. Returns False instead of raising.

    Mirrors what cv2.imwrite gives the callers in utils.image_utils: a boolean,
    with the reason printed, so a failed DNG save is reported through the same
    "could not write" path as any other format.

    `source_path` is the raw the stack came from, and is needed only by the
    camera colour space, which reads its matrix from it - `exif` carries the
    camera's tags but not its calibration, which lives in LibRaw rather than in
    any file. Without it a camera-space write falls back to sRGB.

    The file is streamed rather than built in memory first, so a partly written
    one can be left behind by a failure mid-way; it is removed, because a
    truncated DNG that parses as far as its tag table is worse than no file.
    """
    try:
        with open(file_path, "wb") as handle:
            _emit(handle, image, software, compression, lossy_quality, fast_load,
                  exif, color_space, source_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[DNG] Could not write {os.path.basename(file_path)}: {exc}", flush=True)
        try:
            os.remove(file_path)
        except OSError:
            pass
        return False
    return True


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
# The scalar field types worth unpacking. RATIONAL (5) and SRATIONAL (10) are
# absent on purpose: nothing needed to find the strips is stored as a fraction,
# so they are left as raw bytes rather than decoded and thrown away.
_UNPACK = {1: "<B", 3: "<H", 4: "<I", 6: "<b", 8: "<h", 9: "<i", 11: "<f", 12: "<d"}
_READ_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _read_ifd0(path: str) -> Optional[Dict[int, object]]:
    """Read the tags of a little-endian TIFF's first IFD, or None if it is not one.

    Only enough of TIFF is parsed to recognise this module's own output and find
    its strips: little-endian files, no rationals, no following IFDs. A camera
    DNG is big-endian as often as not and keeps its raw data in a SubIFD, so it
    simply fails one of the checks in `_is_own_output` and goes to LibRaw, which
    is where it belongs anyway.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) < 8 or head[:2] != b"II":
                return None
            magic, ifd_offset = struct.unpack("<HI", head[2:8])
            if magic != 42:
                return None
            handle.seek(ifd_offset)
            raw_count = handle.read(2)
            if len(raw_count) < 2:
                return None
            (entry_count,) = struct.unpack("<H", raw_count)
            table = handle.read(12 * entry_count)
            if len(table) < 12 * entry_count:
                return None

            tags: Dict[int, object] = {}
            for index in range(entry_count):
                tag, field_type, count = struct.unpack_from("<HHI", table, 12 * index)
                payload = table[12 * index + 8:12 * index + 12]
                size = _READ_TYPE_SIZE.get(field_type)
                if size is None:
                    continue
                total = size * count
                if total > 4:
                    (offset,) = struct.unpack("<I", payload)
                    handle.seek(offset)
                    payload = handle.read(total)
                    if len(payload) < total:
                        return None
                else:
                    payload = payload[:total]

                if field_type in (2, 7):
                    tags[tag] = payload.split(b"\x00")[0].decode("ascii", "replace")
                elif field_type in _UNPACK:
                    fmt = _UNPACK[field_type]
                    tags[tag] = [
                        struct.unpack_from(fmt, payload, size * i)[0] for i in range(count)
                    ]
                else:
                    tags[tag] = payload
            return tags
    except (OSError, struct.error):
        return None


def _own_compression(tags: Dict[int, object]) -> Optional[str]:
    """The compression mode of a DNG this module wrote, or None if it did not.

    Both other conditions matter as much as the codec. A camera DNG names its
    camera in UniqueCameraModel, and puts a small preview - not LinearRaw - in
    IFD 0, so it cannot pass by accident.
    """
    if tags.get(_UNIQUE_CAMERA_MODEL) != CAMERA_MODEL:
        return None
    if tags.get(_PHOTOMETRIC) != [_LINEAR_RAW]:
        return None
    code = tags.get(_COMPRESSION)
    if not isinstance(code, list) or len(code) != 1:
        return None
    return _COMPRESSION_MODES.get(int(code[0]))


def _is_own_output(tags: Dict[int, object]) -> bool:
    """Whether these IFD 0 tags describe a DNG this module wrote."""
    return _own_compression(tags) is not None


def probe(path: str) -> Optional[Tuple[int, int, int]]:
    """(width, height, bits) of an OpenFocus linear DNG, read from its header.

    Lets the loader size a `.dng` stack up front without decoding a frame, and
    without opening it through LibRaw only to be told a depth the file does not
    actually have. Returns None for a camera DNG, whose dimensions the loader
    gets from rawpy instead.
    """
    tags = _read_ifd0(path)
    if tags is None or not _is_own_output(tags):
        return None
    try:
        return (
            int(tags[_IMAGE_WIDTH][0]),
            int(tags[_IMAGE_LENGTH][0]),
            int(tags[_BITS_PER_SAMPLE][0]),
        )
    except (KeyError, IndexError, TypeError):
        return None


def _read_strips_into(path: str, offsets: Sequence[int],
                      counts: Sequence[int], out: np.ndarray) -> bool:
    """Fill `out` from the file's strips; False if any of them is short.

    The strips of an uncompressed DNG *are* the pixels, so they are read straight
    into the array that will be returned rather than into bytes that are then
    joined and copied. That is the difference between one allocation the size of
    the frame and three of them, which at 24 MP and 16 bits is 144 MB against
    over 400.
    """
    view = out.reshape(-1).view(np.uint8)
    position = 0
    try:
        with open(path, "rb") as handle:
            for offset, count in zip(offsets, counts):
                if position + count > view.size:
                    return False
                handle.seek(offset)
                if handle.readinto(view[position:position + count]) != count:
                    return False
                position += count
    except OSError:
        return False
    return position == view.size


def _strip_payloads(path: str, offsets: Sequence[int],
                    counts: Sequence[int]) -> Optional[List[bytes]]:
    """Read each strip's bytes, or None if any of them is short or unreadable."""
    try:
        with open(path, "rb") as handle:
            chunks = []
            for offset, count in zip(offsets, counts):
                handle.seek(offset)
                chunk = handle.read(count)
                if len(chunk) < count:
                    return None
                chunks.append(chunk)
            return chunks
    except OSError:
        return None


def read_linear(path: str) -> Optional[np.ndarray]:
    """Read an OpenFocus linear DNG straight from its strips, as BGR.

    No development happens - no white balance, no colour twist, no tone curve -
    so saving a frame and loading it again is very nearly an identity. Only
    the transfer function is undone, because the samples are stored as scene
    linear light and the pipeline's frames are display-referred; see
    `_display_table` for the few parts in 65535 that costs at the bottom of the
    range, and the lossy mode's DCT for the one term that is larger.

    Returns None for anything that is not this module's own output, which is how
    `read` decides to hand the file to LibRaw instead, and for a file whose own
    mode this build cannot decode.
    """
    tags = _read_ifd0(path)
    if tags is None:
        return None
    mode = _own_compression(tags)
    if mode is None:
        return None

    try:
        width = int(tags[_IMAGE_WIDTH][0])
        height = int(tags[_IMAGE_LENGTH][0])
        samples = int(tags[_SAMPLES_PER_PIXEL][0])
        bits = [int(value) for value in tags[_BITS_PER_SAMPLE]]
        offsets = [int(value) for value in tags[_STRIP_OFFSETS]]
        counts = [int(value) for value in tags[_STRIP_BYTE_COUNTS]]
    except (KeyError, IndexError, TypeError):
        return None

    if tags.get(_PLANAR_CONFIG, [1]) != [1] or len(set(bits)) != 1:
        return None
    dtype = {8: np.uint8, 16: np.uint16}.get(bits[0])
    if dtype is None or samples not in (1, 3) or len(offsets) != len(counts):
        return None
    if width <= 0 or height <= 0:
        return None
    expected = width * height * samples

    if mode == COMPRESSION_NONE:
        if sum(counts) != expected * (bits[0] // 8):
            return None
        pixels = np.empty((height, width, samples), dtype=dtype)
        if not _read_strips_into(path, offsets, counts, pixels):
            return None
        if sys.byteorder != "little" and pixels.itemsize > 1:
            # Samples are little-endian on disk regardless of the host.
            pixels.byteswap(inplace=True)
    else:
        chunks = _strip_payloads(path, offsets, counts)
        if chunks is None:
            return None
        decoded = []
        for chunk in chunks:
            if mode == COMPRESSION_LOSSLESS:
                part = _decode_strip_lossless(chunk)
            else:
                part = _decode_strip_lossy(chunk, samples)
            if part is None:
                return None
            decoded.append(part)
        if not decoded:
            return None
        # A compressed image is a single strip - see `_rows_per_strip` - so the
        # usual case has nothing to join and no copy to pay for.
        flat = decoded[0] if len(decoded) == 1 else np.concatenate(decoded)
        if flat.size != expected:
            return None
        pixels = flat.astype(dtype, copy=False).reshape(height, width, samples)

    if _LINEARIZATION_TABLE not in tags and bits[0] == 16:
        # No table means the samples are scene-linear - the file says so by not
        # describing an encoding - so the curve the save applied is undone here.
        # A file that carries one stored its samples display-referred already and
        # is handed back untouched, which is what keeps every DNG written before
        # the writer stopped declaring the encoding reading as it always did.
        pixels = _display_referred(pixels)

    if samples == 1:
        return np.ascontiguousarray(pixels[:, :, 0])
    # RGB->BGR is a channel swap, done in place so reading a frame does not
    # allocate a second one alongside it.
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR, dst=pixels)


def _develop(raw, output_bps: int) -> np.ndarray:
    """Develop an opened camera raw as BGR, on the GPU where that is possible.

    LibRaw unpacks the mosaic either way - there is no GPU LibRaw - but the
    develop that follows it is the expensive half, and core.gpu_decode
    reimplements it in torch. Deferring to that module is what gives the paths
    reaching a camera DNG through `read` - the fusion methods' folder input above
    all - the same GPU develop the stack loader has, rather than the CPU one they
    used to get.

    The import is deferred because `utils` is loaded long before `core`, and a
    save-only run must not pull in torch to write a file. A build without it
    falls back to LibRaw, which is what a machine without a GPU does anyway.
    """
    try:
        from core import gpu_decode
    except ImportError:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=output_bps)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR, dst=rgb)
    return gpu_decode.develop_raw(raw, output_bps)


def read(path: str, output_bps: int = 16) -> Optional[np.ndarray]:
    """Read a DNG as BGR, or None if it cannot be read.

    An OpenFocus linear DNG is returned verbatim; anything else is a camera raw
    and is developed at `output_bps` bits with the camera's own white balance -
    on the GPU when one is available, and by LibRaw otherwise - which is what
    `.nef` already gets.

    Returns None - rather than raising - for a missing, truncated or unreadable
    file, because the callers report a failed frame themselves and carry on with
    the rest of the stack.
    """
    linear = read_linear(path)
    if linear is not None:
        return linear
    if not _RAWPY_AVAILABLE:
        return None
    try:
        # A file object is passed so paths with non-ASCII characters work, the
        # same way the loader's RAW path does it.
        with open(path, "rb") as handle:
            with rawpy.imread(handle) as raw:
                return _develop(raw, output_bps)
    except Exception:  # pylint: disable=broad-except
        return None
