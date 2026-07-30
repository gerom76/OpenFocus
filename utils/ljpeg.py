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
the bit position each one starts at.

Where those bits then go is the part worth spelling out. A sample's code is at
most 32 bits long, so once its start position is known it falls inside one
64-bit word of the output, or straddles two - never more. Each sample is
therefore shifted into place as a whole word, and the words that several samples
share are combined by summing them: their bit ranges are disjoint by
construction, so a sum *is* an OR. Since the start positions only increase, the
samples sharing a word are consecutive, and `np.add.reduceat` collapses each
such run in one pass. The stream is built one word per handful of samples rather
than one byte per output *bit*, which is what the earlier scatter-and-`packbits`
approach cost.

Parallelism
-----------
The scan is cut into chunks of rows, and both passes over them run on a thread
pool - numpy releases the GIL for the array work that dominates either one.

Independence is what makes that possible, and it comes from the first pass
recording each chunk's category histogram rather than only the total. Once the
Huffman table is built, a histogram gives the exact number of bits its chunk
will occupy, so every chunk's position in the output stream is known before any
of them is encoded. Each one then packs into its own word buffer and the buffers
are stitched together, adjacent chunks sharing at most the single word their
boundary falls in.

Memory stays bounded by the chunk however large the image is, and the output is
a single stream - not one per chunk - so the file still holds one JPEG per strip.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Tuple

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

# Samples per chunk of each pass. Measured on a 24 MP frame, where this is both
# the fastest setting and the lightest: a chunk's intermediates are per sample
# and a few tens of bytes wide, so larger chunks cost a thread's working set more
# than they save in per-chunk overhead, and by 32K samples that overhead has
# taken over completely - four times slower than this.
_CHUNK_SAMPLES = 1 << 17

# Threads the two passes are spread over. Past eight the chunks are the limit
# rather than the cores: on a 32-thread machine, twelve and sixteen were within
# the noise of eight.
_MAX_THREADS = 8

# 64-bit words are what the packer scatters into: a code is at most 32 bits, so a
# sample lands inside one word or across two, never more.
_WORD_BITS = 64
_WORD_BYTES = _WORD_BITS // 8


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


def _or_into_words(words: np.ndarray, indices: np.ndarray,
                   contributions: np.ndarray) -> None:
    """OR each contribution into the word it belongs to.

    `indices` are word numbers and only ever increase, so the contributions
    destined for one word form a consecutive run and `reduceat` collapses every
    run in a single pass. The runs are summed rather than OR-ed because no two
    samples own the same bit: within a word the contributions are disjoint, and a
    sum of disjoint bit patterns is their union.
    """
    if indices.size == 0:
        return
    changes = np.flatnonzero(indices[1:] != indices[:-1])
    boundaries = np.empty(changes.size + 1, dtype=np.intp)
    boundaries[0] = 0
    np.add(changes, 1, out=boundaries[1:])
    words[indices[boundaries]] |= np.add.reduceat(contributions, boundaries)


def _pack_chunk(values: np.ndarray, lengths: np.ndarray,
                base_bit: int, stream: np.ndarray) -> Tuple[int, bytes]:
    """Pack one chunk's codes into `stream`, and hand back the word it starts in.

    `values` holds each sample's bits right-aligned, `lengths` how many of them
    count, and `base_bit` the position the chunk occupies in the stream as a
    whole. Every word after the first belongs to this chunk alone, so it is
    written straight into the shared stream and the chunk's own buffer can go;
    the first word is the one the previous chunk may have written part of, so it
    is returned to be OR-ed in once the pool has finished rather than raced over.

    Positions are kept relative to the word the chunk starts in, so they stay
    inside 32 bits however far into the image the chunk sits.
    """
    starts = np.cumsum(lengths, dtype=np.int32)
    total = int(starts[-1]) if starts.size else 0
    starts -= lengths

    offset = (base_bit >> 6) * _WORD_BYTES
    if total == 0:
        return offset, b""

    starts += base_bit & 63
    word = starts >> 6
    within = starts & 63
    # Which codes run past the end of the word they start in, and so leave a
    # remainder for the next one. Decided in signed arithmetic, before the shift
    # counts become unsigned: numpy resolves a mix of the two through float.
    straddles = np.flatnonzero(within + lengths > _WORD_BITS)
    within = within.astype(np.uint64)

    # Each code is first moved to the top of a 64-bit word, which turns placing
    # it into a single right shift by its position - and its remainder, where
    # there is one, into the matching left shift. Both counts are then in range
    # by construction, so no branch and no clamping.
    aligned = values.astype(np.uint64) << (np.uint64(_WORD_BITS) - lengths.astype(np.uint64))
    words = np.zeros((int(starts[-1]) + int(lengths[-1]) - 1) // _WORD_BITS + 1, dtype=np.uint64)
    _or_into_words(words, word, aligned >> within)

    if straddles.size:
        # The tail of a code that crossed the word boundary, at the top of the
        # next word. The indices stay sorted, so the same reduction applies.
        _or_into_words(words, word[straddles] + 1,
                       aligned[straddles] << (np.uint64(_WORD_BITS) - within[straddles]))

    # A JPEG bit stream is most-significant-bit first, which is what a big-endian
    # view of the words it was assembled in already is.
    blob = words.astype(">u8").view(np.uint8)
    stream[offset + _WORD_BYTES:offset + blob.size] = blob[_WORD_BYTES:]
    return offset, blob[:_WORD_BYTES].tobytes()


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
    bounds = [(first, min(first + rows_per_chunk, height))
              for first in range(0, height, rows_per_chunk)]

    with ThreadPoolExecutor(max_workers=_threads_for(len(bounds))) as pool:
        # First pass: how often each magnitude category occurs, per chunk. The
        # totals build the Huffman table; the per-chunk split then says how many
        # bits each chunk will take, which is what lets the second pass encode
        # them independently. The differences themselves are recomputed there
        # rather than kept, because keeping them would cost four bytes a sample.
        histograms = list(pool.map(
            lambda span: _chunk_histogram(data, span, precision), bounds))

        lengths_by_category = _code_lengths(np.sum(histograms, axis=0))
        codes, counts, values = _canonical_codes(lengths_by_category)
        code_bits = lengths_by_category.astype(np.uint32)

        # Bits a sample of each category occupies: its code, plus the magnitude
        # bits that follow it - none for category 16, which stands alone.
        bits_by_category = code_bits.astype(np.int64) + np.arange(_CATEGORIES, dtype=np.int64)
        bits_by_category[_WIDE_CATEGORY] = code_bits[_WIDE_CATEGORY]
        chunk_bits = [int(histogram @ bits_by_category) for histogram in histograms]
        offsets = np.cumsum([0] + chunk_bits)
        total_bits = int(offsets[-1])

        scan = np.zeros((total_bits + 63) // 64 * _WORD_BYTES, dtype=np.uint8)
        heads = list(pool.map(
            lambda item: _pack_scan_chunk(
                data, item[0], precision, codes, code_bits, item[1], scan),
            zip(bounds, offsets[:-1].tolist())))

    header = _headers(height, width, components, precision, counts, values)
    return header + _stuff(_close_scan(scan, heads, total_bits)) + _EOI


def _threads_for(chunks: int) -> int:
    """How many threads to spread `chunks` over."""
    return max(1, min(_MAX_THREADS, os.cpu_count() or 1, chunks))


def _chunk_histogram(data: np.ndarray, span: Tuple[int, int], precision: int) -> np.ndarray:
    """How often each magnitude category occurs in one chunk of rows."""
    first, stop = span
    errors = _prediction_errors(data[first:stop], data[first - 1] if first else None, precision)
    return np.bincount(_categories(errors).ravel(), minlength=_CATEGORIES)


def _pack_scan_chunk(data: np.ndarray, span: Tuple[int, int], precision: int,
                     codes: np.ndarray, code_bits: np.ndarray,
                     base_bit: int, stream: np.ndarray) -> Tuple[int, bytes]:
    """Entropy-code one chunk of rows at its place in the stream."""
    first, stop = span
    errors = _prediction_errors(data[first:stop], data[first - 1] if first else None, precision)
    category = _categories(errors).ravel()
    errors = errors.ravel()

    # Category 16 stands alone for -32768 and carries no magnitude bits.
    extra_bits = np.where(category == _WIDE_CATEGORY, 0, category).astype(np.uint32)
    # A negative error is stored as one less than itself, so that its low bits
    # differ from the positive error of the same magnitude.
    magnitude = np.where(errors >= 0, errors, errors - 1).astype(np.uint32)
    magnitude &= (np.uint32(1) << extra_bits) - np.uint32(1)

    return _pack_chunk(
        (codes[category] << extra_bits) | magnitude,
        (code_bits[category] + extra_bits).astype(np.int32),
        base_bit,
        stream,
    )


def _close_scan(stream: np.ndarray, heads: Sequence[Tuple[int, bytes]],
                total_bits: int) -> np.ndarray:
    """Fold in the chunks' first words and pad the stream to a whole byte.

    A chunk starts wherever the previous one ended, which need not be a word - or
    even a byte - boundary, so the word that boundary falls in holds bits from
    both. Every other word belongs to one chunk alone and was written where it
    goes; these are the ones that had to wait until nothing was still packing.
    """
    for offset, head in heads:
        if head:
            stream[offset:offset + _WORD_BYTES] |= np.frombuffer(head, dtype=np.uint8)

    # JPEG pads the last byte with set bits, which is why the all-ones code had
    # to be kept out of the table; the whole words past it are simply dropped.
    used = (total_bits + 7) // 8
    stream[used - 1] |= (1 << (used * 8 - total_bits)) - 1
    return stream[:used]
