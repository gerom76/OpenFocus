"""EXIF passthrough and OpenFocus provenance for saved results.

A fused image has no camera of its own, so it inherits the EXIF block of the
first source frame of the stack: camera, lens, exposure and shooting date
survive the fusion instead of being dropped. What OpenFocus itself contributed -
its version, when the render happened and how long it took - has no EXIF
equivalent, so it goes into an XMP packet under a namespace of its own.

That packet also carries a Camera section: every camera tag of the source block
cloned as plain text. It is deliberately redundant with the EXIF block, which
stays the authoritative binary copy - the section exists so the shot's settings
can be read straight out of the packet, without an EXIF parser.

Both are spliced into the already-encoded file rather than written by re-saving
it through an imaging library. Re-saving a JPEG would recompress pixels that
were just written at quality 100, and a 16-bit RGB PNG would not survive the
round trip at all, since Pillow has no matching mode. Inserting a JPEG APP1
segment, a PNG chunk or a JPEG XL box leaves every encoded pixel byte untouched.

JPEG, PNG and JPEG XL are handled here; TIFF and BMP saves are left alone.

Reading the source block is a second asymmetry. A JPEG hands its EXIF over as
the bytes it was stored as, but a TIFF-based source - a NEF, a DNG, a .tif - has
no such segment, and letting an imaging library parse and re-encode the tags
changes them: types collapse, rationals get reduced, unknown tags vanish. Those
sources are therefore read by `_tiff_exif`, which copies the tags byte for byte
and recomputes only the offsets.
"""

import io
import os
import re
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

# Namespaces and prefixes of the two groups inside the XMP packet. The URIs are
# only identifiers - they are never fetched - but they have to stay stable,
# because readers key their fields on them.
OPENFOCUS_NS = "https://github.com/Xinzhe99/OpenFocus/ns/1.0/"
OPENFOCUS_PREFIX = "OpenFocus"
CAMERA_NS = "https://github.com/Xinzhe99/OpenFocus/ns/camera/1.0/"
CAMERA_PREFIX = "Camera"

# Containers this module can splice metadata into. Everything else is written
# as-is - except DNG, which carries EXIF as well but gets it from utils.dng at
# encode time rather than from here; see `carries_exif`.
TAGGABLE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".jxl"}

# Every container a save can put the source's EXIF into, however it gets there.
# Used to say once, before a stack export, that the chosen format will drop it.
EXIF_EXTENSIONS = TAGGABLE_EXTENSIONS | {".dng"}

_JPEG_SOI = b"\xff\xd8"
_JPEG_APP0 = b"\xff\xe0"
_JPEG_APP1 = b"\xff\xe1"
_EXIF_PREFIX = b"Exif\x00\x00"
_XMP_PREFIX = b"http://ns.adobe.com/xap/1.0/\x00"
# A JPEG segment length field counts itself, so the payload has 2 bytes less
# than the 16-bit maximum available to it.
_MAX_JPEG_PAYLOAD = 0xFFFF - 2

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_XMP_KEYWORD = b"XML:com.adobe.xmp"

# A JPEG XL container opens with a signature box; a file that starts with the
# codestream marker instead is a bare codestream and has nowhere to put a box.
_JXL_CONTAINER_SIGNATURE = b"\x00\x00\x00\x0cJXL \r\n\x87\n"
_JXL_EXIF_BOX = b"Exif"
_JXL_XMP_BOX = b"xml "
# Boxes that have a fixed place at the head of the container and cannot be
# pushed back by a metadata box: the file type box, and the level box which
# ISO/IEC 18181-2 requires to come directly after it. A reader that finds the
# level box out of position may stop trusting the header boxes and fall back to
# its own defaults - 8 bits per sample among them - so a 16-bit file written
# with the boxes in the wrong order can be read back as 8-bit.
_JXL_HEADER_BOXES = (b"ftyp", b"jxll")

# Byte order marks of a TIFF header, which is what an EXIF block really is.
_TIFF_HEADERS = (b"II*\x00", b"MM\x00*")

# Field type codes of a TIFF entry mapped to the width of one value. Used to
# walk a source block, so every type a camera may write has to be here even
# though this module never constructs one.
_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
                   8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
_TIFF_SHORT = 3
_TIFF_LONG = 4

# Orientation, and the one value a saved result may carry. See `_upright`.
_ORIENTATION = 274
_ORIENTATION_NORMAL = 1

# Tags of a TIFF-based source's IFD 0 that describe the shot rather than the
# file it was stored in. A whitelist rather than a blacklist, because IFD 0 of a
# raw file is mostly about that file - the geometry and strip offsets of its
# embedded thumbnail, its colour profile, its own XMP - and, if the source is a
# DNG, a linearization table and a colour matrix that would be nonsense in the
# EXIF block of a JPEG. What is left is the camera, the timestamps and the
# people the file credits.
#
# Orientation is absent on purpose, and is the one tag written rather than
# copied - `_upright` says why, and utils.dng leaves it out of its own copied
# set for the same reason.
_SOURCE_IFD0_TAGS = frozenset({
    270,    # ImageDescription
    271,    # Make
    272,    # Model
    282,    # XResolution
    283,    # YResolution
    296,    # ResolutionUnit
    305,    # Software
    306,    # DateTime
    315,    # Artist
    316,    # HostComputer
    33432,  # Copyright
    36867,  # DateTimeOriginal - TIFF/EP puts it here as well as in the EXIF IFD
    37398,  # TIFF/EPStandardID
    40091, 40092, 40093, 40094, 40095,  # XPTitle, XPComment, XPAuthor, XPKeywords, XPSubject
})

# Tags dropped from a copied sub-IFD. The EXIF and GPS IFDs are about the shot
# from end to end, so everything else in them travels; these three are the
# exceptions, and both reasons are that the value is an offset into a file that
# is being left behind - the Interoperability IFD and the thumbnail.
_SOURCE_SUB_IFD_DROPPED = frozenset({40965, 513, 514})

_MAKER_NOTE = 0x927C

# EXIF IFDs cloned into the Camera section, in the order they are written.
# Their tag names come from Pillow, which uses the names of the EXIF standard.
_EXIF_IFD_POINTER = 0x8769
_GPS_IFD_POINTER = 0x8825
# Pointers to the sub-IFDs themselves, and the offsets of the embedded
# thumbnail: plumbing of the EXIF block rather than anything about the camera.
_STRUCTURAL_TAGS = {_EXIF_IFD_POINTER, _GPS_IFD_POINTER, 0xA005, 0x0201, 0x0202}
# An XMP property name has to be a legal XML name.
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
# Long values are binary blobs that lost their type somewhere; the EXIF block
# still carries them intact, so the readable copy skips them.
_MAX_TAG_CHARS = 512

_XMP_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="OpenFocus {version}">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:{prefix}="{namespace}">
   <xmp:CreatorTool>OpenFocus {version}</xmp:CreatorTool>
   <{prefix}:Version>{version}</{prefix}:Version>
   <{prefix}:RenderDate>{render_date}</{prefix}:RenderDate>
   <{prefix}:RenderDuration>{duration}</{prefix}:RenderDuration>
{options}  </rdf:Description>
{camera_section} </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""

_CAMERA_SECTION_TEMPLATE = """  <rdf:Description rdf:about=""
    xmlns:{prefix}="{namespace}">
{properties}  </rdf:Description>
"""


@dataclass
class RenderMetadata:
    """What a saved result records about the render that produced it.

    `source_path` is the first source image of the stack that was rendered; its
    EXIF is read at save time rather than kept here, so the record stays cheap
    to hold alongside every entry in the output list.

    `options` are the settings the render ran with - which stages, which method,
    which parameters - as already-formatted name/value pairs in the order they
    should be read. The producer of the render decides what belongs in there;
    this module only writes them out.
    """

    source_path: Optional[str] = None
    rendered_at: Optional[datetime] = None
    duration_s: float = 0.0
    options: Optional[Dict[str, str]] = None


def app_version() -> str:
    """The running OpenFocus version, or an empty string if unavailable.

    Imported lazily: `utils` is imported long before `core`, and the version is
    only ever needed while writing a file.
    """
    try:
        from core.app import OpenFocusApplication
        return str(OpenFocusApplication.VERSION)
    except Exception:
        return ""


def read_source_exif(path: Optional[str]) -> Optional[bytes]:
    """Return the EXIF block of `path` as a bare TIFF stream, or None.

    Pillow hands back the block with an ``Exif\\0\\0`` prefix in some paths and
    without it in others, so it is normalised away here and re-added per
    container: JPEG wants it, PNG's eXIf chunk does not.

    Sources Pillow cannot open - RAW files above all - simply yield no EXIF; the
    result is still saved, just without the camera block.

    JPEG XL is read from its own Exif box rather than through Pillow, which has
    no JPEG XL reader; that is the same box this module writes, so a `.jxl`
    stack carries its camera tags through a render like any other source.

    A TIFF-based source - a NEF, a DNG, a plain TIFF - is read by `_tiff_exif`
    instead, for the reason given there: Pillow can open those, but only by
    re-encoding what it read, and the block that comes back is no longer the
    block the camera wrote.
    """
    if not path or not os.path.isfile(path):
        return None

    if os.path.splitext(path)[1].lower() == ".jxl":
        blob = _jxl_exif(path)
        return _as_tiff_stream(blob)

    try:
        with open(path, "rb") as handle:
            if handle.read(4) in _TIFF_HEADERS:
                return _tiff_exif(handle)
    except OSError as exc:
        print(f"[Metadata] No EXIF read from {os.path.basename(path)}: {exc}", flush=True)
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(path) as img:
            # PNG only exposes EXIF once the image data has been read.
            img.load()
            blob = img.info.get("exif")
            if not blob:
                exif = img.getexif()
                blob = exif.tobytes() if exif else None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[Metadata] No EXIF read from {os.path.basename(path)}: {exc}", flush=True)
        return None

    return _as_tiff_stream(blob)


def _as_tiff_stream(blob: Optional[bytes]) -> Optional[bytes]:
    """Strip the ``Exif\\0\\0`` prefix, and reject anything that is not TIFF."""
    if not blob:
        return None

    if blob.startswith(_EXIF_PREFIX):
        blob = blob[len(_EXIF_PREFIX):]

    # Anything that is not a TIFF stream would be rejected by readers anyway.
    if not blob.startswith(_TIFF_HEADERS):
        return None

    return _upright(blob)


def _upright(blob: Optional[bytes]) -> Optional[bytes]:
    """Set IFD 0's Orientation to 1, in place, in a bare TIFF stream.

    A fused result is already the right way up: whatever rotation the camera
    recorded was applied when the frames were decoded, so the pixels that reach
    a save are upright and the source's value describes a rotation that has
    already happened. Inherited unchanged it is applied a second time, and the
    result comes out turned - which is what the tag did before this existed.

    Worse, it came out turned only in *some* viewers, because no two of them
    read the tag from the same place: a PNG's `eXIf` chunk is honoured by
    XnView and ignored by Windows Explorer, and a JPEG XL's Exif box is
    honoured by Explorer and ignored by anything decoding through libjxl, which
    reads the codestream's own orientation field instead. So the same render
    saved three ways looked like three different images. Every format now says
    what utils.dng has always written: 1.

    Patched rather than rebuilt, because the block is copied byte for byte and
    only this one value may change. Orientation is a single SHORT, which lives
    inline in its 12-byte entry, so nothing moves and no offset shifts; a source
    that wrote it any other way is malformed and is left alone. A block without
    the tag is left alone too - absent already means 1.
    """
    if not blob or len(blob) < 8:
        return blob

    endian = "<" if blob[:2] == b"II" else ">"
    try:
        magic, first_ifd = struct.unpack(endian + "HI", blob[2:8])
        if magic != 42 or first_ifd <= 0:
            return blob
        (count,) = struct.unpack_from(endian + "H", blob, first_ifd)
        for index in range(count):
            entry = first_ifd + 2 + 12 * index
            tag, field_type, values = struct.unpack_from(endian + "HHI", blob, entry)
            if tag != _ORIENTATION:
                continue
            if field_type != _TIFF_SHORT or values != 1:
                return blob
            patched = bytearray(blob)
            struct.pack_into(endian + "H", patched, entry + 8, _ORIENTATION_NORMAL)
            return bytes(patched)
    except struct.error:
        return blob

    return blob


# ----------------------------------------------------------------------
# EXIF from a TIFF-based source
# ----------------------------------------------------------------------
# A TIFF-based source needs a reader of its own, because Pillow does not have
# one that gives the block back unchanged. For a JPEG it does: the APP1 segment
# is handed over as the bytes it was read as. For a TIFF - which is what a NEF,
# a CR2, a DNG and a .tif all are - there is no such segment, so Pillow parses
# the tags into Python values and `Exif.tobytes()` writes them out again, and
# what comes back is not what went in:
#
# * A rational is reduced. The camera wrote ExposureTime 10/800 and FocalLength
#   1000/10, because a Nikon quotes 1/80 s in tenths of a millisecond and 100 mm
#   in tenths of a millimetre; the round trip returns 1/80 and 100/1. The number
#   is the same and the precision the camera claimed is not.
# * A field type changes. UserComment, FileSource and SceneType are UNDEFINED in
#   the source and come back as BYTE, which is a different tag as far as a
#   strict reader is concerned - UserComment's leading character-set marker only
#   means anything in an UNDEFINED field.
# * Anything Pillow has no decoder for is dropped rather than copied.
#
# So the block is rebuilt here from the source's own bytes instead: the tags
# worth keeping are copied with their type, their count and their payload
# untouched, in the byte order the camera used, and only the offsets - which are
# the one thing that cannot survive a move - are recomputed. The result is a
# small stand-alone TIFF stream, which is exactly what the rest of this module
# and utils.dng already expect.
def _tiff_entries(stream, endian: str,
                  offset: int) -> List[Tuple[int, int, int, bytes]]:
    """(tag, type, count, payload) of the IFD at `offset`, payloads verbatim.

    Reads through the stream rather than from a copy of the file, so extracting
    a few hundred kilobytes of tags out of a 27 MB raw costs a handful of small
    reads. A truncated value, or one whose offset points past the end, is
    skipped: the file was written by something else, and one bad tag must not
    cost the whole block.
    """
    entries: List[Tuple[int, int, int, bytes]] = []
    if offset <= 0:
        return entries

    try:
        stream.seek(offset)
        raw_count = stream.read(2)
        if len(raw_count) < 2:
            return entries
        (count,) = struct.unpack(endian + "H", raw_count)
        table = stream.read(12 * count)
    except (OSError, ValueError, struct.error):
        return entries
    if len(table) < 12 * count:
        return entries

    for index in range(count):
        tag, field_type, values = struct.unpack_from(endian + "HHI", table, 12 * index)
        size = _TIFF_TYPE_SIZE.get(field_type)
        if size is None:
            continue
        total = size * values
        if total <= 4:
            payload = table[12 * index + 8:12 * index + 8 + total]
        else:
            (position,) = struct.unpack_from(endian + "I", table, 12 * index + 8)
            try:
                stream.seek(position)
                payload = stream.read(total)
            except (OSError, ValueError):
                continue
        if len(payload) < total:
            continue
        entries.append((tag, field_type, values, payload))
    return entries


def _pack_ifd(entries: List[Tuple[int, int, int, bytes]], base: int,
              endian: str) -> Tuple[bytes, Dict[int, int]]:
    """Lay out one IFD, and its out-of-line values, at `base`.

    Returns the assembled bytes and, for every tag, the position at which its
    value begins - inline in the 4-byte entry or out in the value block. The
    caller patches the sub-IFD pointers through those positions, since where a
    sub-IFD lands is only known once the table pointing at it has been sized.

    TIFF requires the entries be sorted by tag, and any value longer than the
    four bytes an entry holds to live elsewhere and be referenced by offset.
    """
    entries = sorted(entries, key=lambda entry: entry[0])
    values_offset = base + 2 + 12 * len(entries) + 4

    table = bytearray(struct.pack(endian + "H", len(entries)))
    values = bytearray()
    positions: Dict[int, int] = {}

    for tag, field_type, count, payload in entries:
        head = struct.pack(endian + "HHI", tag, field_type, count)
        if len(payload) <= 4:
            table += head + payload.ljust(4, b"\x00")
            positions[tag] = base + len(table) - 4
        else:
            positions[tag] = values_offset + len(values)
            table += head + struct.pack(endian + "I", positions[tag])
            values += payload
            if len(values) % 2:
                values += b"\x00"  # keep the next value word-aligned
    table += struct.pack(endian + "I", 0)  # no thumbnail IFD follows

    return bytes(table + values), positions


def _tiff_exif(stream, keep_maker_note: bool = True) -> Optional[bytes]:
    """The EXIF of an open TIFF stream, rebuilt as a stand-alone block.

    `stream` is a seekable binary stream positioned anywhere - a source file, or
    a block already in memory. Returns None when it is not a TIFF at all, or
    carries nothing worth keeping.

    IFD 0 is filtered to `_SOURCE_IFD0_TAGS`, and the EXIF and GPS IFDs are
    copied whole apart from the handful of tags that are offsets into the file
    being left behind. The thumbnail IFD is not copied: the saved result has its
    own preview, or none. Orientation is written rather than copied, and only
    once there is a block to write it into - see `_upright`.

    `keep_maker_note` exists for the one container that cannot always take the
    block whole - see `_jpeg_with_metadata`.
    """
    try:
        stream.seek(0)
        head = stream.read(8)
    except (OSError, ValueError):
        return None
    if len(head) < 8 or head[:4] not in _TIFF_HEADERS:
        return None

    endian = "<" if head[:2] == b"II" else ">"
    magic, first_ifd = struct.unpack(endian + "HI", head[2:8])
    if magic != 42:
        return None

    dropped = set(_SOURCE_SUB_IFD_DROPPED)
    if not keep_maker_note:
        dropped.add(_MAKER_NOTE)

    pointers: Dict[int, int] = {}
    kept: List[Tuple[int, int, int, bytes]] = []
    for tag, field_type, count, payload in _tiff_entries(stream, endian, first_ifd):
        if tag in (_EXIF_IFD_POINTER, _GPS_IFD_POINTER):
            # Re-created below, once the sub-IFDs have been placed.
            if field_type == _TIFF_LONG and count == 1:
                (pointers[tag],) = struct.unpack(endian + "I", payload)
        elif tag in _SOURCE_IFD0_TAGS:
            kept.append((tag, field_type, count, payload))

    sub_ifds = []
    for pointer in (_EXIF_IFD_POINTER, _GPS_IFD_POINTER):
        if pointer not in pointers:
            continue
        entries = [entry for entry in _tiff_entries(stream, endian, pointers[pointer])
                   if entry[0] not in dropped]
        if entries:
            sub_ifds.append((pointer, entries))

    if not kept and not sub_ifds:
        return None

    # Stated outright, so that a source that left the tag out and one that wrote
    # a rotation both come out of a save saying the same thing. Appended after
    # the check above, which asks whether the source had anything to say at all.
    kept.append((_ORIENTATION, _TIFF_SHORT, 1,
                 struct.pack(endian + "H", _ORIENTATION_NORMAL)))

    for pointer, _entries in sub_ifds:
        kept.append((pointer, _TIFF_LONG, 1, b"\x00" * 4))  # patched once placed

    table, positions = _pack_ifd(kept, 8, endian)
    blob = bytearray(struct.pack(endian + "2sHI", head[:2], 42, 8)) + table
    for pointer, entries in sub_ifds:
        struct.pack_into(endian + "I", blob, positions[pointer], len(blob))
        blob += _pack_ifd(entries, len(blob), endian)[0]

    return bytes(blob)


def _properties(prefix: str, pairs: Iterable[Tuple[str, str]]) -> str:
    """Render name/value pairs as XMP simple properties under `prefix`.

    A name that cannot be a legal XML element is dropped rather than allowed to
    break the packet; values are escaped.
    """
    lines = []
    for name, value in pairs:
        safe_name = _INVALID_NAME_CHARS.sub("", str(name))
        if not safe_name or safe_name[0].isdigit():
            continue
        lines.append(f"   <{prefix}:{safe_name}>{escape(str(value))}"
                     f"</{prefix}:{safe_name}>\n")
    return "".join(lines)


def _tag_name(tag_id: int, names: dict) -> str:
    """XMP property name for an EXIF tag.

    Tags Pillow has no name for keep their number, so a rare or vendor-specific
    one is still readable rather than silently dropped.
    """
    return str(names.get(tag_id) or f"Tag{tag_id:04X}")


def _tag_value(value) -> Optional[str]:
    """Render one EXIF value as text, or None if it does not belong in XMP.

    Rationals below 1 keep their fraction, because that is how a shutter speed
    is read - 1/200, not 0.005 - while the rest become decimals, because an
    aperture is an f/5.6 and not a 28/5. Byte strings are left to the EXIF
    block, which stores them losslessly and is copied across anyway.
    """
    if isinstance(value, (bytes, bytearray)):
        return None

    if isinstance(value, (tuple, list)):
        parts = []
        for item in value:
            part = _tag_value(item)
            if part is None:
                return None
            parts.append(part)
        text = ", ".join(parts)
    elif hasattr(value, "numerator") and hasattr(value, "denominator"):
        numerator, denominator = value.numerator, value.denominator
        if denominator == 0:
            # Cameras write 0/0 for "not recorded"; there is nothing to say.
            return None
        if denominator == 1:
            text = str(numerator)
        elif abs(numerator) < abs(denominator):
            text = f"{numerator}/{denominator}"
        else:
            text = f"{numerator / denominator:g}"
    else:
        text = str(value)

    # Trailing NULs and stray control characters are common in EXIF strings and
    # are not valid XML content.
    text = "".join(ch for ch in text if ch == "\t" or ch >= " ").strip()
    if not text or len(text) > _MAX_TAG_CHARS:
        return None
    return text


def read_camera_tags(exif_blob: Optional[bytes]) -> List[Tuple[str, str]]:
    """Every camera tag of an EXIF block, as ordered (name, value) pairs.

    Written IFD by IFD - the main image IFD, then the EXIF sub-IFD, then GPS -
    and within each one in the order the source file stored them, so the section
    mirrors the block it was cloned from. Structural tags, meaning the pointers
    between IFDs and the offsets of the embedded thumbnail, are left out: they
    describe the source file's layout rather than the shot.

    A name is only emitted once. Where two IFDs disagree on a tag the main image
    IFD wins, since that is the one a reader of the block sees first.
    """
    if not exif_blob:
        return []

    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS
    except ImportError:
        return []

    try:
        exif = Image.Exif()
        exif.load(exif_blob)
        groups = [
            (exif, TAGS),
            (exif.get_ifd(_EXIF_IFD_POINTER), TAGS),
            (exif.get_ifd(_GPS_IFD_POINTER), GPSTAGS),
        ]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[Metadata] EXIF could not be read into the Camera section: {exc}", flush=True)
        return []

    tags: List[Tuple[str, str]] = []
    seen = set()
    for group, names in groups:
        for tag_id, raw in (group or {}).items():
            if tag_id in _STRUCTURAL_TAGS:
                continue
            name = _tag_name(tag_id, names)
            value = _tag_value(raw)
            if value is None or name in seen:
                continue
            seen.add(name)
            tags.append((name, value))

    return tags


def _camera_section(camera_tags: Optional[List[Tuple[str, str]]]) -> str:
    """The Camera rdf:Description, or an empty string when there is nothing to say."""
    if not camera_tags:
        return ""

    return _CAMERA_SECTION_TEMPLATE.format(
        prefix=CAMERA_PREFIX,
        namespace=escape(CAMERA_NS),
        properties=_properties(CAMERA_PREFIX, camera_tags),
    )


def build_xmp(metadata: RenderMetadata,
              camera_tags: Optional[List[Tuple[str, str]]] = None) -> bytes:
    """Serialise `metadata` and the camera tags as a UTF-8 XMP packet.

    Two groups: OpenFocus' own record of the render - version, timing and every
    option the render ran with - and a Camera section holding the source's EXIF
    tags in readable form. The binary EXIF block travels with the file as well,
    but nothing has to be decoded to read either group.
    """
    rendered_at = metadata.rendered_at or datetime.now()
    if rendered_at.tzinfo is None:
        # XMP dates carry a UTC offset; naive timestamps are local time.
        rendered_at = rendered_at.astimezone()

    packet = _XMP_TEMPLATE.format(
        version=escape(app_version()),
        prefix=OPENFOCUS_PREFIX,
        namespace=escape(OPENFOCUS_NS),
        render_date=escape(rendered_at.isoformat(timespec="seconds")),
        # The unit travels with the number so the value reads on its own.
        duration=f"{max(0.0, float(metadata.duration_s)):.2f} s",
        options=_properties(OPENFOCUS_PREFIX, (metadata.options or {}).items()),
        camera_section=_camera_section(camera_tags),
    )
    return packet.encode("utf-8")


def carries_exif(extension: str) -> bool:
    """Whether a save in this container keeps the source's EXIF block."""
    ext = (extension or "").lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext in EXIF_EXTENSIONS


def _splice(file_path: str, exif: Optional[bytes], xmp: Optional[bytes]) -> bool:
    """Rewrite an encoded file with the metadata blocks it was written without.

    Called after the encoder has already produced the file, so a failure here
    costs the metadata and not the image. Returns True only when the file was
    rewritten.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in TAGGABLE_EXTENSIONS or not (exif or xmp):
        return False

    try:
        with open(file_path, "rb") as handle:
            data = handle.read()

        if ext == ".png":
            tagged = _png_with_metadata(data, exif, xmp)
        elif ext == ".jxl":
            tagged = _jxl_with_metadata(data, exif, xmp)
        else:
            tagged = _jpeg_with_metadata(data, exif, xmp)

        if tagged is None:
            print(f"[Metadata] {os.path.basename(file_path)} is not a container "
                  f"this build can tag; saved without metadata.", flush=True)
            return False

        with open(file_path, "wb") as handle:
            handle.write(tagged)
        return True
    except OSError as exc:
        print(f"[Metadata] Could not tag {os.path.basename(file_path)}: {exc}", flush=True)
        return False


def embed(file_path: str, metadata: Optional[RenderMetadata]) -> bool:
    """Add the source EXIF and the OpenFocus XMP packet to a written file."""
    if metadata is None:
        return False

    # The source file is opened once: the same block is embedded verbatim and
    # cloned into the readable Camera section of the packet.
    exif = read_source_exif(metadata.source_path)
    return _splice(file_path, exif, build_xmp(metadata, read_camera_tags(exif)))


def copy_exif(file_path: str, source_path: Optional[str]) -> bool:
    """Add a source file's EXIF block to a written file, and nothing else.

    What a stack export wants, as against what `embed` does for a fused result.
    A processed input frame is still the frame that came out of the camera - one
    aligned, cropped or relit copy of it - so it keeps that camera's block, and
    each frame keeps its *own*: the exposure of frame 40 belongs to frame 40, not
    to the first of the stack. There is no render to describe on top of it, so no
    XMP packet is written and nothing claims the frame was fused.

    DNG is absent from the containers handled here on purpose. Its EXIF goes in
    while the file is being written, because a TIFF cannot have an IFD spliced
    into it afterwards without every offset behind the insertion moving.
    """
    exif = read_source_exif(source_path)
    return _splice(file_path, exif, None) if exif else False


# ----------------------------------------------------------------------
# JPEG
# ----------------------------------------------------------------------
def _jpeg_app1(payload: bytes) -> Optional[bytes]:
    """Wrap `payload` in an APP1 segment, or None if it exceeds the 64 KB limit."""
    if len(payload) > _MAX_JPEG_PAYLOAD:
        return None
    return _JPEG_APP1 + (len(payload) + 2).to_bytes(2, "big") + payload


def _jpeg_with_metadata(data: bytes, exif: Optional[bytes],
                        xmp: Optional[bytes]) -> Optional[bytes]:
    """Splice EXIF and XMP APP1 segments into an encoded JPEG.

    APP1 has to follow the JFIF APP0 that OpenCV writes, so the insertion point
    is found by walking past any leading APP0 segments. Nothing before or after
    that point is touched, so the compressed scan is bit-identical.
    """
    if not data.startswith(_JPEG_SOI):
        return None

    position = 2
    while data[position:position + 2] == _JPEG_APP0:
        length = int.from_bytes(data[position + 2:position + 4], "big")
        if length < 2:
            return None
        position += 2 + length
        if position > len(data):
            return None

    segments = []
    if exif:
        exif_segment = _jpeg_app1(_EXIF_PREFIX + exif)
        if exif_segment is None:
            # A camera's MakerNote alone routinely exceeds the 64 KB a segment
            # holds - a Nikon one runs to about 180 KB - and it is the part of
            # the block a JPEG reader can do least with. Dropping it and trying
            # again keeps the camera, the lens and the exposure, where before
            # they went over the side along with it.
            trimmed = _tiff_exif(io.BytesIO(exif), keep_maker_note=False)
            exif_segment = _jpeg_app1(_EXIF_PREFIX + trimmed) if trimmed else None
        if exif_segment is None:
            print("[Metadata] Source EXIF is larger than a JPEG segment; "
                  "saved without it.", flush=True)
        else:
            segments.append(exif_segment)

    if xmp:
        xmp_segment = _jpeg_app1(_XMP_PREFIX + xmp)
        if xmp_segment is not None:
            segments.append(xmp_segment)

    if not segments:
        return None

    return data[:position] + b"".join(segments) + data[position:]


# ----------------------------------------------------------------------
# PNG
# ----------------------------------------------------------------------
def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk: length, type, data, CRC over type and data."""
    return (len(payload).to_bytes(4, "big")
            + chunk_type
            + payload
            + zlib.crc32(chunk_type + payload).to_bytes(4, "big"))


def _png_with_metadata(data: bytes, exif: Optional[bytes],
                       xmp: Optional[bytes]) -> Optional[bytes]:
    """Insert eXIf and XMP chunks into an encoded PNG, right after IHDR.

    IHDR is always the first chunk, and both additions are legal anywhere
    between it and IDAT, so that gap is the one insertion point that needs no
    knowledge of what the encoder wrote. The IDAT chunks are copied verbatim,
    which is what keeps a 16-bit PNG at 16 bits.
    """
    if not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return None

    ihdr_length = int.from_bytes(data[8:12], "big")
    insert_at = 20 + ihdr_length  # signature + length + type + data + CRC
    if insert_at > len(data):
        return None

    chunks = []
    if exif:
        chunks.append(_png_chunk(b"eXIf", exif))

    # iTXt payload: keyword, compression flag and method, empty language and
    # translated keyword, then the UTF-8 text. XMP is stored uncompressed so
    # that readers can find the packet by scanning the file.
    if xmp:
        chunks.append(_png_chunk(b"iTXt", _PNG_XMP_KEYWORD + b"\x00" * 5 + xmp))

    if not chunks:
        return None

    return data[:insert_at] + b"".join(chunks) + data[insert_at:]


# ----------------------------------------------------------------------
# JPEG XL
# ----------------------------------------------------------------------
def _jxl_box(box_type: bytes, payload: bytes) -> bytes:
    """Build one JPEG XL container box: 32-bit size (counting itself), type, data."""
    return (len(payload) + 8).to_bytes(4, "big") + box_type + payload


def _jxl_exif(path: str) -> Optional[bytes]:
    """The EXIF block of a JPEG XL file, still prefixed as the box stores it.

    Walks the container box by box, seeking over each payload instead of reading
    it, so a large file costs a handful of small reads rather than loading a
    codestream to find a few hundred bytes of tags. A bare codestream, a
    truncated file or a container without an Exif box all yield None.

    The box payload opens with a 32-bit count of bytes to skip before the TIFF
    header - OpenFocus writes 0, other encoders need not.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(len(_JXL_CONTAINER_SIGNATURE)) != _JXL_CONTAINER_SIGNATURE:
                return None

            while True:
                header = handle.read(8)
                if len(header) < 8:
                    return None
                size = int.from_bytes(header[0:4], "big")
                box_type = header[4:8]

                if box_type == _JXL_EXIF_BOX:
                    # Size 0 means the box runs to the end of the file.
                    payload = handle.read() if size == 0 else handle.read(size - 8)
                    if len(payload) < 4:
                        return None
                    skip = int.from_bytes(payload[0:4], "big")
                    return payload[4 + skip:] or None

                if size == 0:
                    return None  # Ran to the end without an Exif box.
                if size < 8:
                    return None  # Malformed: a box cannot be smaller than its header.
                handle.seek(size - 8, os.SEEK_CUR)
    except OSError:
        return None


def _jxl_with_metadata(data: bytes, exif: Optional[bytes],
                       xmp: Optional[bytes]) -> Optional[bytes]:
    """Insert Exif and XMP boxes into a JPEG XL container, ahead of the codestream.

    utils.jxl always writes the container form precisely so this can be done.
    The boxes go after the header boxes that must come first - the signature
    box, ftyp, and the jxll level box that ISO/IEC 18181-2 pins directly behind
    ftyp - and before whatever follows, which keeps the metadata readable from a
    stream and leaves the codestream boxes byte-identical.

    An Exif box holds a 32-bit offset to the start of the TIFF header before the
    block itself; OpenFocus writes the block at the front, so the offset is 0.
    """
    if not data.startswith(_JXL_CONTAINER_SIGNATURE):
        # A bare codestream: valid JPEG XL, but boxless, so there is nothing to
        # splice into. utils.jxl does not produce these.
        return None

    # Walk past the leading header boxes to find where a metadata box may start.
    insert_at = len(_JXL_CONTAINER_SIGNATURE)
    while True:
        header = data[insert_at:insert_at + 8]
        if len(header) < 8 or header[4:8] not in _JXL_HEADER_BOXES:
            break
        size = int.from_bytes(header[0:4], "big")
        if size < 8 or insert_at + size > len(data):
            return None
        insert_at += size

    boxes = []
    if exif:
        boxes.append(_jxl_box(_JXL_EXIF_BOX, (0).to_bytes(4, "big") + exif))
    if xmp:
        boxes.append(_jxl_box(_JXL_XMP_BOX, xmp))

    if not boxes:
        return None

    return data[:insert_at] + b"".join(boxes) + data[insert_at:]
