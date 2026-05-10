"""Tests for video_stego.message."""
import numpy as np
import pytest

from video_stego.message import MessageFragmenter


class TestMessageFragmenter:
    def setup_method(self):
        self.frag = MessageFragmenter()

    # ------------------------------------------------------------------
    # bytes_to_bits / bits_to_bytes round-trip
    # ------------------------------------------------------------------

    def test_bytes_to_bits_length(self):
        bits = MessageFragmenter.bytes_to_bits(b"A")
        assert len(bits) == 8

    def test_bits_to_bytes_roundtrip(self):
        msg = b"Hello, World!"
        bits = MessageFragmenter.bytes_to_bits(msg)
        recovered = MessageFragmenter.bits_to_bytes(bits)
        assert recovered == msg

    def test_bits_values_binary(self):
        bits = MessageFragmenter.bytes_to_bits(b"\xff")
        assert set(bits.tolist()).issubset({0, 1})
        assert bits.sum() == 8  # all ones

    def test_bits_to_bytes_zero(self):
        bits = np.zeros(8, dtype=np.uint8)
        assert MessageFragmenter.bits_to_bytes(bits) == b"\x00"

    # ------------------------------------------------------------------
    # build_bitstream / parse_bitstream
    # ------------------------------------------------------------------

    def test_build_bitstream_length(self):
        msg = b"test"
        bs = self.frag.build_bitstream(msg)
        expected_len = 32 + 8 * len(msg)
        assert len(bs) == expected_len

    def test_parse_bitstream_roundtrip(self):
        for msg in [b"", b"a", b"hello world", b"\x00\xff\x80"]:
            bs = self.frag.build_bitstream(msg)
            recovered = self.frag.parse_bitstream(bs)
            assert recovered == msg

    def test_parse_too_short_raises(self):
        with pytest.raises(ValueError):
            self.frag.parse_bitstream(np.zeros(10, dtype=np.uint8))

    def test_parse_truncated_payload_raises(self):
        msg = b"hello"
        bs = self.frag.build_bitstream(msg)
        with pytest.raises(ValueError):
            self.frag.parse_bitstream(bs[:33])  # header + 1 bit of payload

    # ------------------------------------------------------------------
    # fragment / assemble
    # ------------------------------------------------------------------

    def test_fragment_chunk_sizes(self):
        chunks = self.frag.fragment(b"hello", chunk_size=16)
        for chunk in chunks[:-1]:
            assert len(chunk) == 16

    def test_fragment_assemble_roundtrip(self):
        for msg in [b"a", b"hello world", b"\x00\x01\x02"]:
            chunks = self.frag.fragment(msg, chunk_size=20)
            recovered = self.frag.assemble(chunks)
            assert recovered == msg

    def test_fragment_zero_chunk_raises(self):
        with pytest.raises(ValueError):
            self.frag.fragment(b"test", chunk_size=0)

    def test_fragment_single_chunk(self):
        msg = b"x"
        chunks = self.frag.fragment(msg, chunk_size=1000)
        assert len(chunks) == 1

    def test_assemble_empty_message(self):
        chunks = self.frag.fragment(b"", chunk_size=10)
        recovered = self.frag.assemble(chunks)
        assert recovered == b""

    # ------------------------------------------------------------------
    # max_message_bytes
    # ------------------------------------------------------------------

    def test_max_message_bytes_zero_capacity(self):
        assert MessageFragmenter.max_message_bytes(0) == 0

    def test_max_message_bytes_header_only(self):
        assert MessageFragmenter.max_message_bytes(32) == 0

    def test_max_message_bytes_with_payload(self):
        # 32 header + 16 payload → 2 bytes
        assert MessageFragmenter.max_message_bytes(48) == 2
