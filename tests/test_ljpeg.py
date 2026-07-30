"""Lossless JPEG encoding.

utils.ljpeg exists because DNG's only lossless codec for 16-bit linear data is
lossless Huffman JPEG, and nothing in the dependency set can write it with more
than one component. An encoder without a matching decoder is easy to fool - it
only has to agree with itself - so nothing here decodes with our own code. Three
promises are tested:

1. Independent decoders read it back exactly, across depths, component counts
   and the shapes that break off-by-one errors: one pixel, one row, one column
   (`TestRoundTrip`).
2. The container is a real SOF3 stream with a well-formed Huffman table - in
   particular one that never assigns the all-ones code, which JPEG forbids
   because it cannot be told apart from the bits that pad the final byte
   (`TestContainer`).
3. It actually compresses, and refuses what it cannot represent
   (`TestBehaviour`).

`imagecodecs` supplies the independent decoders. Without it the round-trip tests
skip, which is honest: on such a build utils.dng does not offer the lossless mode
either, for exactly the same reason.

Run with:  python -m pytest tests/test_ljpeg.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ljpeg

try:
    import imagecodecs
    _DECODERS = [name for name in ("ljpeg_decode", "jpegsof3_decode")
                 if getattr(imagecodecs, name, None) is not None]
except ImportError:
    imagecodecs = None
    _DECODERS = []

needs_decoder = pytest.mark.skipif(
    not _DECODERS, reason="checking the bitstream needs imagecodecs"
)


def _decode(payload, decoder):
    return np.asarray(getattr(imagecodecs, decoder)(payload))


def _noise(shape, dtype):
    ceiling = 255 if dtype is np.uint8 else 65535
    return (np.random.default_rng(5).random(shape) * ceiling).astype(dtype)


def _gradient(height=64, width=96, components=3):
    """Smooth data with edges - what a fused frame looks like to a predictor."""
    ys, xs = np.mgrid[0:height, 0:width]
    plane = ((xs + ys * 2) % 200).astype(np.uint8)
    return np.dstack([np.roll(plane, i * 5, axis=1) for i in range(components)])


class TestRoundTrip:
    @needs_decoder
    @pytest.mark.parametrize("decoder", _DECODERS)
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16], ids=["8bit", "16bit"])
    @pytest.mark.parametrize("shape", [(1, 1, 1), (1, 1, 3), (1, 37, 3), (37, 1, 3),
                                       (8, 8), (23, 31, 1), (23, 31, 3), (64, 64, 4)],
                             ids=lambda s: "x".join(str(v) for v in s))
    def test_noise_survives_exactly(self, shape, dtype, decoder):
        image = _noise(shape, dtype)
        decoded = _decode(ljpeg.encode(image), decoder)
        # Lossless means lossless: prediction errors are stored, not approximated.
        assert decoded.size == image.size
        assert np.array_equal(decoded.reshape(image.shape).astype(dtype), image)

    @needs_decoder
    @pytest.mark.parametrize("decoder", _DECODERS)
    def test_a_gradient_survives_exactly(self, decoder):
        image = _gradient()
        decoded = _decode(ljpeg.encode(image), decoder)
        assert np.array_equal(decoded.reshape(image.shape), image)

    @needs_decoder
    @pytest.mark.parametrize("decoder", _DECODERS)
    def test_a_flat_field_survives_exactly(self, decoder):
        # Every error is zero, so the table holds a single symbol - the degenerate
        # case a Huffman builder is most likely to get wrong.
        image = np.full((40, 60, 3), 137, np.uint8)
        decoded = _decode(ljpeg.encode(image), decoder)
        assert np.array_equal(decoded.reshape(image.shape), image)

    @needs_decoder
    @pytest.mark.parametrize("decoder", _DECODERS)
    def test_the_extremes_survive_exactly(self, decoder):
        # Full-scale swings drive the differences to the widest category, which
        # the standard codes specially and without any magnitude bits.
        image = np.array([[[0, 65535, 32768], [65535, 0, 65535]],
                          [[65535, 32768, 0], [0, 65535, 0]]], np.uint16)
        decoded = _decode(ljpeg.encode(image), decoder)
        assert np.array_equal(decoded.reshape(image.shape), image)

    @needs_decoder
    @pytest.mark.parametrize("decoder", _DECODERS)
    def test_an_image_larger_than_one_chunk_survives(self, decoder, monkeypatch):
        # The stream is packed a chunk of rows at a time, carrying the leftover
        # bits of one chunk into the next; a bug there shows up only at a seam.
        monkeypatch.setattr(ljpeg, "_CHUNK_SAMPLES", 512)
        image = _noise((64, 48, 3), np.uint8)
        decoded = _decode(ljpeg.encode(image), decoder)
        assert np.array_equal(decoded.reshape(image.shape), image)

    @needs_decoder
    @pytest.mark.parametrize("decoder", _DECODERS)
    def test_a_two_dimensional_array_is_treated_as_one_component(self, decoder):
        image = _noise((23, 31), np.uint8)
        decoded = _decode(ljpeg.encode(image), decoder)
        assert np.array_equal(decoded.reshape(image.shape), image)


class TestContainer:
    def test_it_is_a_lossless_huffman_jpeg(self):
        payload = ljpeg.encode(_gradient())
        assert payload[:2] == b"\xff\xd8"  # SOI
        assert payload[-2:] == b"\xff\xd9"  # EOI
        # SOF3 is the lossless, Huffman-coded frame. SOF0 would be baseline DCT
        # and SOF11 arithmetic coded, neither of which DNG accepts here.
        assert b"\xff\xc3" in payload
        assert b"\xff\xc4" in payload  # DHT
        assert b"\xff\xda" in payload  # SOS

    @pytest.mark.parametrize("components", [1, 3])
    def test_the_frame_header_describes_the_array(self, components):
        image = _gradient(48, 72, components)
        payload = ljpeg.encode(image)
        start = payload.index(b"\xff\xc3") + 4
        precision = payload[start]
        height = int.from_bytes(payload[start + 1:start + 3], "big")
        width = int.from_bytes(payload[start + 3:start + 5], "big")
        declared = payload[start + 5]

        # The component count has to reach the file: LibRaw derives its inner
        # loop from it, so a mismatch decodes the wrong amount of data per row.
        assert (precision, height, width, declared) == (8, 48, 72, components)

    def test_the_scan_selects_the_left_neighbour_predictor(self):
        payload = ljpeg.encode(_gradient())
        start = payload.index(b"\xff\xda") + 2
        length = int.from_bytes(payload[start:start + 2], "big")
        # The three bytes after the component list: predictor, then the two
        # fields lossless mode leaves at zero.
        assert payload[start + length - 3] == 1
        assert payload[start + length - 2:start + length] == b"\x00\x00"

    def test_sixteen_bit_input_declares_sixteen_bit_precision(self):
        payload = ljpeg.encode(_noise((16, 16, 3), np.uint16))
        assert payload[payload.index(b"\xff\xc3") + 4] == 16

    def test_the_huffman_table_is_complete_and_reserves_the_all_ones_code(self):
        # The table is built per image, so it is checked on several: a code set
        # that over-subscribes the tree decodes as garbage rather than failing.
        for image in (_gradient(), _noise((40, 40, 3), np.uint16),
                      np.full((8, 8), 3, np.uint8)):
            payload = ljpeg.encode(image)
            start = payload.index(b"\xff\xc4") + 4
            assert payload[start] == 0x00, "table class 0, id 0"
            counts = list(payload[start + 1:start + 17])

            # Kraft sum: exactly 1 would use every code including all-ones,
            # which JPEG forbids because the byte padding is all ones too.
            kraft = sum(count / (1 << length) for length, count in enumerate(counts, 1))
            assert kraft < 1.0, f"all-ones code was assigned (kraft {kraft})"
            assert kraft > 0.0

            values = payload[start + 17:start + 17 + sum(counts)]
            assert len(set(values)) == len(values), "a category appears twice"
            assert all(value <= 16 for value in values), "category out of range"


class TestBehaviour:
    def test_it_compresses_compressible_data(self):
        image = _gradient(128, 128)
        assert len(ljpeg.encode(image)) < image.nbytes * 0.6

    def test_noise_is_not_inflated_much(self):
        # Incompressible input cannot shrink; what matters is that the entropy
        # coder does not blow it up, which a badly fitted table would.
        image = _noise((128, 128, 3), np.uint8)
        assert len(ljpeg.encode(image)) < image.nbytes * 1.3

    def test_sixteen_bit_beats_storing_it_raw(self):
        image = _gradient(128, 128).astype(np.uint16) * 257
        assert len(ljpeg.encode(image)) < image.nbytes

    @pytest.mark.parametrize("bad", [
        np.zeros((4, 4, 3), np.float32),
        np.zeros((4, 4, 3), np.int16),
    ])
    def test_an_unsupported_dtype_is_refused(self, bad):
        with pytest.raises(ValueError):
            ljpeg.encode(bad)

    def test_too_many_components_are_refused(self):
        with pytest.raises(ValueError):
            ljpeg.encode(np.zeros((4, 4, 5), np.uint8))

    def test_an_impossible_shape_is_refused(self):
        with pytest.raises(ValueError):
            ljpeg.encode(np.zeros((2, 2, 2, 2), np.uint8))
        with pytest.raises(ValueError):
            ljpeg.encode(None)

    def test_a_precision_outside_the_standard_is_refused(self):
        with pytest.raises(ValueError):
            ljpeg.encode(np.zeros((4, 4), np.uint8), bits=17)
        with pytest.raises(ValueError):
            ljpeg.encode(np.zeros((4, 4), np.uint8), bits=1)

    def test_a_precision_that_cannot_hold_the_data_is_refused(self):
        # Writing this would produce a stream every decoder rejects, so it is
        # caught here rather than at the far end of a save.
        image = np.full((4, 4), 5000, np.uint16)
        with pytest.raises(ValueError, match="5000"):
            ljpeg.encode(image, bits=12)
        # 13 bits is enough for 5000, so the same data is fine.
        assert ljpeg.encode(image, bits=13)

    @needs_decoder
    def test_a_declared_precision_below_the_native_one_still_round_trips(self):
        # 12-bit data in a uint16 array is the case this serves; the declared
        # precision is what the decoder reconstructs against.
        image = (np.random.default_rng(3).random((32, 32, 3)) * 4095).astype(np.uint16)
        decoded = _decode(ljpeg.encode(image, bits=12), _DECODERS[0])
        assert np.array_equal(decoded.reshape(image.shape).astype(np.uint16), image)
