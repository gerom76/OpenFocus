"""EXIF passthrough and OpenFocus provenance for saved results.

A fused image has no camera of its own, so it inherits the EXIF block of the
first source frame of the stack: camera, lens, exposure and shooting date
survive the fusion instead of being dropped. What OpenFocus itself contributed -
its version, when the render happened and how long it took - has no EXIF
equivalent, so it goes into an XMP packet under a namespace of its own.

Both are spliced into the already-encoded file rather than written by re-saving
it through an imaging library. Re-saving a JPEG would recompress pixels that
were just written at quality 100, and a 16-bit RGB PNG would not survive the
round trip at all, since Pillow has no matching mode. Inserting a JPEG APP1
segment or a PNG chunk leaves every encoded pixel byte untouched.

Only JPEG and PNG are handled here; TIFF and BMP saves are left alone.
"""

import os
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from xml.sax.saxutils import escape

# Namespace and prefix of the OpenFocus group inside the XMP packet. The URI is
# only an identifier - it is never fetched - but it has to stay stable, because
# readers key their fields on it.
OPENFOCUS_NS = "https://github.com/Xinzhe99/OpenFocus/ns/1.0/"
OPENFOCUS_PREFIX = "OpenFocus"

# Containers that can carry the metadata. Everything else is written as-is.
TAGGABLE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png"}

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

# Byte order marks of a TIFF header, which is what an EXIF block really is.
_TIFF_HEADERS = (b"II*\x00", b"MM\x00*")

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
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


@dataclass
class RenderMetadata:
    """What a saved result records about the render that produced it.

    `source_path` is the first source image of the stack that was rendered; its
    EXIF is read at save time rather than kept here, so the record stays cheap
    to hold alongside every entry in the output list.
    """

    source_path: Optional[str] = None
    rendered_at: Optional[datetime] = None
    duration_s: float = 0.0


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


def build_xmp(metadata: RenderMetadata) -> bytes:
    """Serialise the OpenFocus group of `metadata` as a UTF-8 XMP packet."""
    rendered_at = metadata.rendered_at or datetime.now()
    if rendered_at.tzinfo is None:
        # XMP dates carry a UTC offset; naive timestamps are local time.
        rendered_at = rendered_at.astimezone()

    packet = _XMP_TEMPLATE.format(
        version=escape(app_version()),
        prefix=OPENFOCUS_PREFIX,
        namespace=escape(OPENFOCUS_NS),
        render_date=escape(rendered_at.isoformat(timespec="seconds")),
        duration=f"{max(0.0, float(metadata.duration_s)):.2f}",
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

    exif = read_source_exif(metadata.source_path)
    xmp = build_xmp(metadata)

    try:
        with open(file_path, "rb") as handle:
            data = handle.read()

        if ext == ".png":
            tagged = _png_with_metadata(data, exif, xmp)
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
