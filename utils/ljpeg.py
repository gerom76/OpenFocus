"""Lossless JPEG (ITU T.81 Annex H) encoding, for DNG's `Compression` = 7.

This exists because nothing else available can write the one codec DNG permits
for lossless 16-bit LinearRaw data. OpenCV writes baseline JPEG only; libjpeg's
lossless mode is not exposed by any of the project's dependencies; and
`imagecodecs`' lossless-JPEG encoder is single-component, which sounds
acceptable - DNG explicitly allows a strip's samples to be encoded with an
internal geometry that does not match the strip - but is not. LibRaw derives its
inner loop from the *declared* component count and advances by
`SamplesPerPixel` per pixel regardless, so a three-sample strip carried as one
component decodes three times too much data per row and renders most of the
image black. A file a raw converter mis-renders is worse than an uncompressed
one, so the component count has to match.

What lossless JPEG actually is, in one paragraph: each sample is predicted from
its already-coded neighbours, and only the prediction error is stored. Predictor
1 - the one used here, and the one every DNG reader implements - predicts a
sample from the one to its left. Left of the first column, the prediction is the
sample directly above; above the first row, it is half of full scale. The error
is then Huffman-coded exactly as a JPEG DC coefficient is: a symbol giving the
magnitude category, followed by that many raw bits. There is no DCT, no
quantisation and no colour transform anywhere in it, which is what makes it
lossless.

Speed
-----
The obvious implementation - a loop over samples - is unusable at these sizes: a
24 MP colour frame is 72 million samples. Everything here is instead expressed as
whole-array numpy work, including the bit packing, which is the part that looks
inherently serial. It is not: the code and its extra bits are known per sample,
so the *length* of every sample's contribution is known, so a prefix sum gives
the bit position each one starts at, and the whole stream can be scattered into
a bit array at once and packed with `np.packbits`.

That bit array costs about one byte per output bit, so it is built a chunk of
rows at a time rather than for the whole frame, with the leftover bits of one
chunk carried into the next. Memory stays bounded by the chunk regardless of how
large the image is, and the output is a single stream - not one per chunk - so
the file still holds one JPEG per strip.
"""

from typing import List, Optional, Tuple

import numpy as np

# Markers used here. SOF3 is the one that says "lossless, Huffman coded"; the
# other lossless variant, SOF11, is arithmetic coded and DNG does not allow it.
_SOI = b"\xff\xd8"
_SOF3 = b"\xff\xc3"
_DHT = b"\xff\xc4"
_SOS = b"\xff\xda"
_EOI = b"\xff\xd9"

# Predictor 1: the sample to the left. See the module docstring for the
# first-row and first-column cases, which the standard defines separately.
_PREDICTOR_LEFT = 1

# Symbols the entropy coder can emit: magnitude categories 0..16.
_CATEGORIES = 17

# Category 16 is the standard's special case - it stands for a difference of
# -32768 exactly and carries no extra bits, so it cannot be confused with the
# 16-bit magnitudes that would otherwise share the category.
_WIDE_CATEGORY = 16

# Thresholds for turning a magnitude into its category: `searchsorted` against
# these gives bit_length(magnitude), which is what the category is.
_MAGNITUDE_BOUNDS = (1 << np.arange(16, dtype=np.int64))

# Samples per chunk of the bit-packing pass. At roughly a dozen bits per sample
# this keeps the intermediate arrays - which are per *bit*, not per sample - to
# something like a hundred megabytes however large the frame is. It also keeps a
# chunk's bit count inside a signed 32-bit range, which is what lets the packing
# index with the narrower dtype.
_CHUNK_SAMPLES = 1 << 20


def _prediction_errors(data: np.ndarray, previous_row: Optional[np.ndarray],
                       bits: int) -> np.ndarray:
    """Prediction errors for a block of rows, as int32.

    `previous_row` is the row above the block, or None when the block starts the
    image - in which case the very first pixel is predicted from half of full
    scale, the standard's substitute for a neighbour that does not exist.

    Each component predicts only from itself, which is the whole reason the
    component count has to reach the file: red predicted from the green beside it
    would code just as correctly and compress noticeably worse.
    """
    wide = data.astype(np.int32, copy=False)
    prediction = np.empty_like(wide)
    prediction[:, 1:, :] = wide[:, :-1, :]  # Ra, the sample to the left
    if previous_row is None:
        prediction[0, 0, :] = 1 << (bits - 1)
        prediction[1:, 0, :] = wide[:-1, 0, :]  # Rb, the sample above
    else:
        prediction[0, 0, :] = previous_row[0, :]
        prediction[1:, 0, :] = wide[:-1, 0, :]

    # The standard takes differences modulo 2^16, so a wrap is not an overflow:
    # the decoder reconstructs modulo 2^16 too and lands back on the sample.
    errors = wide - prediction
    return ((errors + 0x8000) & 0xFFFF) - 0x8000


def _categories(errors: np.ndarray) -> np.ndarray:
    """Magnitude category of every prediction error, as uint8.

    Category 0 is a zero error; category *n* covers the errors needing *n* bits
    of magnitude. Only -32768 reaches category 16, which is what lets that
    category double as the standard's no-extra-bits special case.
    """
    magnitudes = np.abs(errors)
    return np.searchsorted(_MAGNITUDE_BOUNDS, magnitudes, side="right").astype(np.uint8)


def _code_lengths(frequencies: np.ndarray) -> np.ndarray:
    """Huffman code length for each category, from how often each one occurs.

    This is the procedure from the JPEG standard's Annex K.2, and it is used
    rather than a fixed table because the categories' distribution depends
    entirely on the image: a clean gradient concentrates in the low categories, a
    noisy crop spreads across all of them, and a table built for the wrong one
    costs several percent of the file.

    Two details are not optimisation but correctness. A dummy symbol is given a
    frequency of one and dropped at the end, which reserves the all-ones code -
    JPEG forbids assigning it, because it cannot be distinguished from the fill
    bits that pad the final byte. And any code longer than 16 bits is folded back
    to 16, since that is the longest a JPEG Huffman table can express.
    """
    # One slot past the real categories for the reserved dummy symbol.
    freq = np.zeros(_CATEGORIES + 1, dtype=np.int64)
    freq[:_CATEGORIES] = frequencies
    freq[_CATEGORIES] = 1

    codesize = np.zeros(_CATEGORIES + 1, dtype=np.int64)
    others = np.full(_CATEGORIES + 1, -1, dtype=np.int64)

    while True:
        # The two least frequent live symbols, preferring the larger index on a
        # tie so the result matches every other JPEG writer's.
        live = np.flatnonzero(freq > 0)
        if live.size < 2:
            break
        first = live[np.argmin(freq[live] * (_CATEGORIES + 2) - live)]
        rest = live[live != first]
        second = rest[np.argmin(freq[rest] * (_CATEGORIES + 2) - rest)]

        freq[first] += freq[second]
        freq[second] = 0

        # Both branches get one bit longer, which means every symbol already
        # merged into either of them does too - `others` is the chain of those.
        # The new link has to be made from the *end* of the first chain, not from
        # its head, or merging over it again would lose everything behind it.
        tail = first
        codesize[tail] += 1
        while others[tail] >= 0:
            tail = others[tail]
            codesize[tail] += 1

        node = second
        codesize[node] += 1
        while others[node] >= 0:
            node = others[node]
            codesize[node] += 1

        others[tail] = second

    # How many codes there are of each length, then the fold back to 16 bits.
    counts = np.zeros(33, dtype=np.int64)
    for symbol in range(_CATEGORIES + 1):
        if codesize[symbol]:
            counts[codesize[symbol]] += 1

    for length in range(32, 16, -1):
        while counts[length] > 0:
            donor = length - 2
            while counts[donor] == 0:
                donor -= 1
            # Two codes of this length become one longer prefix plus a pair
            # borrowed from a shorter one, which keeps the tree complete.
            counts[length] -= 2
            counts[length - 1] += 1
            counts[donor + 1] += 2
            counts[donor] -= 1

    # Drop the reserved dummy, which by construction holds a longest code.
    longest = int(np.max(np.flatnonzero(counts)))
    counts[longest] -= 1

    lengths = np.zeros(_CATEGORIES, dtype=np.int64)
    # Symbols take the available lengths shortest first, and by category within a
    # length - the canonical order a JPEG decoder rebuilds the table in.
    order = sorted(
        (symbol for symbol in range(_CATEGORIES) if codesize[symbol]),
        key=lambda symbol: (codesize[symbol], symbol),
    )
    position = 0
    for length in range(1, 17):
        for _ in range(int(counts[length])):
            lengths[order[position]] = length
            position += 1
    return lengths


def _canonical_codes(lengths: np.ndarray) -> Tuple[np.ndarray, bytes, bytes]:
    """Codes for each category, plus the BITS and HUFFVAL a DHT segment carries.

    Canonical assignment: codes are handed out in increasing length, and by
    category within a length, so BITS and HUFFVAL are all a decoder needs to
    rebuild exactly this mapping.
    """
    counts = bytes(int(np.count_nonzero(lengths == length)) for length in range(1, 17))
    values = [symbol for length in range(1, 17)
              for symbol in range(_CATEGORIES) if lengths[symbol] == length]

    codes = np.zeros(_CATEGORIES, dtype=np.uint32)
    code = 0
    previous = 0
    for symbol in values:
        length = int(lengths[symbol])
        code <<= length - previous
        codes[symbol] = code
        code += 1
        previous = length
    return codes, counts, bytes(values)


def _pack_bits(values: np.ndarray, lengths: np.ndarray,
               carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Append variable-length codes to a bit stream; return whole bytes and leftovers.

    `values` holds each sample's bits right-aligned and `lengths` how many of
    them count. Because the lengths are known up front, a prefix sum gives the
    bit position every sample starts at - and since those positions are
    consecutive, the *n*th bit of the whole stream can be attributed to its
    sample by repeating each sample's start position `length` times. That turns
    the entire chunk into a handful of whole-array operations with no loop over
    samples or over bit positions.

    Codes are left-aligned in 32 bits first, so extracting bit *k* of the stream
    is one shift by a position counted from the top rather than a shift that
    depends on the code's own length.
    """
    total = int(lengths.sum())
    stream = np.empty(total + carry.size, dtype=np.uint8)
    stream[:carry.size] = carry

    if total:
        starts = np.cumsum(lengths, dtype=np.int32) - lengths
        # Bit position within its own code, rewritten in place into the shift
        # that extracts it, to keep one fewer array of this size alive.
        shifts = np.arange(total, dtype=np.int32)
        shifts -= np.repeat(starts, lengths)
        np.subtract(31, shifts, out=shifts)
        aligned = values << (np.uint32(32) - lengths.astype(np.uint32))
        stream[carry.size:] = (np.repeat(aligned, lengths) >> shifts.view(np.uint32)) & 1

    whole = (stream.size // 8) * 8
    return np.packbits(stream[:whole]), stream[whole:]


def _stuff(payload: np.ndarray) -> bytes:
    """Insert the zero byte JPEG requires after every 0xFF in entropy data.

    Without it a run of set bits would be indistinguishable from a marker.
    """
    marks = np.flatnonzero(payload == 0xFF)
    if marks.size:
        payload = np.insert(payload, marks + 1, 0)
    return payload.tobytes()


def _headers(height: int, width: int, components: int, bits: int,
             counts: bytes, values: bytes) -> bytes:
    """SOI, the Huffman table, SOF3 and SOS - everything ahead of the scan.

    One Huffman table is shared by every component. Per-component tables would
    save a little on an image whose channels differ sharply in noise, and cost a
    table's worth of header each; sharing is what camera DNGs do.
    """
    table = bytes([0x00]) + counts + values  # class 0 (DC), table 0
    dht = _DHT + (len(table) + 2).to_bytes(2, "big") + table

    frame = bytes([bits]) + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    frame += bytes([components])
    for index in range(components):
        # Component id, then 1x1 sampling - lossless JPEG has no subsampling -
        # then quantisation table 0, which is unused and must still be named.
        frame += bytes([index + 1, 0x11, 0x00])
    sof = _SOF3 + (len(frame) + 2).to_bytes(2, "big") + frame

    scan = bytes([components])
    for index in range(components):
        scan += bytes([index + 1, 0x00])  # component id, both tables are 0
    # Predictor selector, then the two fields lossless leaves at zero: the
    # unused spectral-selection end, and the point transform.
    scan += bytes([_PREDICTOR_LEFT, 0x00, 0x00])
    sos = _SOS + (len(scan) + 2).to_bytes(2, "big") + scan

    return _SOI + dht + sof + sos


def encode(image: np.ndarray, bits: Optional[int] = None) -> bytes:
    """Encode a 2D or 3D uint8/uint16 array as a lossless JPEG.

    `image` is (rows, columns) or (rows, columns, components) and is stored
    exactly - the samples come back from any conforming decoder unchanged.
    `bits` is the sample precision to declare, defaulting to the array's own; it
    must be large enough to hold every value present, since a decoder treats a
    sample that overflows the declared precision as a corrupt stream.
    """
    if image is None:
        raise ValueError("No image to encode.")
    data = np.ascontiguousarray(image)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f"Unsupported image shape for lossless JPEG: {image.shape}")

    height, width, components = data.shape
    if not 1 <= components <= 4:
        raise ValueError(
            f"Lossless JPEG allows 1 to 4 components, not {components}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    if data.dtype == np.uint8:
        native = 8
    elif data.dtype == np.uint16:
        native = 16
    else:
        raise ValueError(f"Unsupported dtype for lossless JPEG: {data.dtype}")
    precision = native if bits is None else int(bits)
    if not 2 <= precision <= 16:
        raise ValueError(f"Lossless JPEG precision must be 2 to 16, not {precision}.")
    if precision < native and data.size:
        # Only worth the pass when the caller has narrowed the declared depth
        # below the array's own - 12-bit samples in a uint16 buffer, say. A
        # decoder treats a sample that overflows the declared precision as a
        # corrupt stream, so catching it here beats writing an unreadable file.
        peak = int(data.max())
        if peak >= (1 << precision):
            raise ValueError(
                f"Sample value {peak} does not fit the declared precision of "
                f"{precision} bits."
            )

    rows_per_chunk = max(1, _CHUNK_SAMPLES // (width * components))

    # First pass: how often each magnitude category occurs, which is what the
    # Huffman table is built from. The differences are recomputed in the second
    # pass rather than kept, because keeping them would cost four bytes a sample.
    frequencies = np.zeros(_CATEGORIES, dtype=np.int64)
    for first in range(0, height, rows_per_chunk):
        previous = data[first - 1] if first else None
        errors = _prediction_errors(data[first:first + rows_per_chunk], previous, precision)
        frequencies += np.bincount(_categories(errors).ravel(), minlength=_CATEGORIES)

    lengths_by_category = _code_lengths(frequencies)
    codes, counts, values = _canonical_codes(lengths_by_category)
    code_bits = lengths_by_category.astype(np.uint32)

    chunks: List[bytes] = []
    carry = np.zeros(0, dtype=np.uint8)
    for first in range(0, height, rows_per_chunk):
        previous = data[first - 1] if first else None
        errors = _prediction_errors(data[first:first + rows_per_chunk], previous, precision)
        category = _categories(errors).ravel().astype(np.int64)
        errors = errors.ravel()

        # Category 16 stands alone for -32768 and carries no magnitude bits.
        extra_bits = np.where(category == _WIDE_CATEGORY, 0, category).astype(np.uint32)
        # A negative error is stored as one less than itself, so that its low
        # bits differ from the positive error of the same magnitude.
        magnitude = np.where(errors >= 0, errors, errors - 1).astype(np.uint32)
        magnitude &= (np.uint32(1) << extra_bits) - np.uint32(1)

        packed, carry = _pack_bits(
            (codes[category] << extra_bits) | magnitude,
            (code_bits[category] + extra_bits).astype(np.int32),
            carry,
        )
        chunks.append(_stuff(packed))

    if carry.size:
        # JPEG pads the last byte with set bits, which is why the all-ones code
        # had to be kept out of the table.
        tail = np.ones(8, dtype=np.uint8)
        tail[:carry.size] = carry
        chunks.append(_stuff(np.packbits(tail)))

    header = _headers(height, width, components, precision, counts, values)
    return header + b"".join(chunks) + _EOI
