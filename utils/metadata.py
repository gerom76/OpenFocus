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
"""

import os
import re
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

# Containers that can carry the metadata. Everything else is written as-is.
TAGGABLE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".jxl"}

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

# Byte order marks of a TIFF header, which is what an EXIF block really is.
_TIFF_HEADERS = (b"II*\x00", b"MM\x00*")

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
    """
    if not path or not os.path.isfile(path):
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

    if not blob:
        return None

    if blob.startswith(_EXIF_PREFIX):
        blob = blob[len(_EXIF_PREFIX):]

    # Anything that is not a TIFF stream would be rejected by readers anyway.
    if not blob.startswith(_TIFF_HEADERS):
        return None

    return blob


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


def embed(file_path: str, metadata: Optional[RenderMetadata]) -> bool:
    """Add the source EXIF and the OpenFocus XMP packet to a written file.

    Called after the encoder has already produced the file, so a failure here
    costs the metadata and not the image. Returns True only when the file was
    rewritten with the metadata in it.
    """
    if metadata is None:
        return False

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in TAGGABLE_EXTENSIONS:
        return False

    # The source file is opened once: the same block is embedded verbatim and
    # cloned into the readable Camera section of the packet.
    exif = read_source_exif(metadata.source_path)
    xmp = build_xmp(metadata, read_camera_tags(exif))

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


# ----------------------------------------------------------------------
# JPEG
# ----------------------------------------------------------------------
def _jpeg_app1(payload: bytes) -> Optional[bytes]:
    """Wrap `payload` in an APP1 segment, or None if it exceeds the 64 KB limit."""
    if len(payload) > _MAX_JPEG_PAYLOAD:
        return None
    return _JPEG_APP1 + (len(payload) + 2).to_bytes(2, "big") + payload


def _jpeg_with_metadata(data: bytes, exif: Optional[bytes], xmp: bytes) -> Optional[bytes]:
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
            # Bulky maker notes and an embedded thumbnail can push a source
            # block past what one segment holds; the XMP group still goes in.
            print("[Metadata] Source EXIF is larger than a JPEG segment; "
                  "saved without it.", flush=True)
        else:
            segments.append(exif_segment)

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


def _png_with_metadata(data: bytes, exif: Optional[bytes], xmp: bytes) -> Optional[bytes]:
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
    chunks.append(_png_chunk(b"iTXt", _PNG_XMP_KEYWORD + b"\x00" * 5 + xmp))

    return data[:insert_at] + b"".join(chunks) + data[insert_at:]


# ----------------------------------------------------------------------
# JPEG XL
# ----------------------------------------------------------------------
def _jxl_box(box_type: bytes, payload: bytes) -> bytes:
    """Build one JPEG XL container box: 32-bit size (counting itself), type, data."""
    return (len(payload) + 8).to_bytes(4, "big") + box_type + payload


def _jxl_with_metadata(data: bytes, exif: Optional[bytes], xmp: bytes) -> Optional[bytes]:
    """Insert Exif and XMP boxes into a JPEG XL container, ahead of the codestream.

    utils.jxl always writes the container form precisely so this can be done.
    The boxes go after the header boxes that must come first - the signature box
    and ftyp - and before whatever follows, which keeps the metadata readable
    from a stream and leaves the codestream boxes byte-identical.

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
        if len(header) < 8 or header[4:8] != b"ftyp":
            break
        size = int.from_bytes(header[0:4], "big")
        if size < 8 or insert_at + size > len(data):
            return None
        insert_at += size

    boxes = []
    if exif:
        boxes.append(_jxl_box(_JXL_EXIF_BOX, (0).to_bytes(4, "big") + exif))
    boxes.append(_jxl_box(_JXL_XMP_BOX, xmp))

    return data[:insert_at] + b"".join(boxes) + data[insert_at:]
