"""Tests for video_stego.embedding (pixel, DCT, inter-frame)."""
import numpy as np
import pytest

from video_stego.embedding import PixelEmbedder, DCTEmbedder, InterFrameEmbedder
from video_stego.message import MessageFragmenter
from tests.conftest import make_frame, make_static_frame_pair


# ---------------------------------------------------------------------------
# PixelEmbedder
# ---------------------------------------------------------------------------

class TestPixelEmbedder:
    def setup_method(self):
        self.emb = PixelEmbedder(lsb_count=1)
        self.frame = make_frame(64, 64, seed=10)

    def test_capacity(self):
        cap = self.emb.capacity_bits(self.frame)
        assert cap == 64 * 64 * 1

    def test_stego_frame_same_shape(self):
        bits = np.zeros(100, dtype=np.uint8)
        stego = self.emb.embed(self.frame, bits)
        assert stego.shape == self.frame.shape

    def test_stego_frame_dtype(self):
        bits = np.zeros(100, dtype=np.uint8)
        stego = self.emb.embed(self.frame, bits)
        assert stego.dtype == np.uint8

    def test_embed_extract_roundtrip(self):
        bits = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        stego = self.emb.embed(self.frame, bits)
        recovered = self.emb.extract(stego, len(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_embed_extract_message_roundtrip(self):
        for msg in [b"hello", b"test message", b"\x00\x01"]:
            stego = self.emb.embed_message(self.frame, msg)
            recovered = self.emb.extract_message(stego)
            assert recovered == msg

    def test_too_many_bits_raises(self):
        cap = self.emb.capacity_bits(self.frame)
        bits = np.zeros(cap + 1, dtype=np.uint8)
        with pytest.raises(ValueError):
            self.emb.embed(self.frame, bits)

    def test_invalid_lsb_count(self):
        with pytest.raises(ValueError):
            PixelEmbedder(lsb_count=0)
        with pytest.raises(ValueError):
            PixelEmbedder(lsb_count=5)

    def test_lsb2_roundtrip(self):
        emb = PixelEmbedder(lsb_count=2)
        bits = np.random.default_rng(0).integers(0, 2, 50, dtype=np.uint8)
        stego = emb.embed(self.frame, bits)
        recovered = emb.extract(stego, len(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_stego_psnr_high(self):
        """LSB(1) should cause very little distortion (PSNR > 40 dB)."""
        from video_stego.metrics import compute_psnr
        bits = np.zeros(100, dtype=np.uint8)
        stego = self.emb.embed(self.frame, bits)
        psnr = compute_psnr(self.frame, stego)
        assert psnr > 40.0


# ---------------------------------------------------------------------------
# DCTEmbedder
# ---------------------------------------------------------------------------

class TestDCTEmbedder:
    def setup_method(self):
        self.emb = DCTEmbedder(quantisation_step=20, coeff_index=5)
        self.frame = make_frame(64, 64, seed=20)

    def test_capacity(self):
        cap = self.emb.capacity_bits(self.frame)
        # 64/8 × 64/8 = 64 blocks
        assert cap == 64

    def test_embed_returns_same_shape(self):
        bits = np.zeros(10, dtype=np.uint8)
        stego = self.emb.embed(self.frame, bits)
        assert stego.shape == self.frame.shape

    def test_embed_extract_roundtrip(self):
        bits = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        stego = self.emb.embed(self.frame, bits)
        recovered = self.emb.extract(stego, len(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_embed_extract_message_roundtrip(self):
        msg = b"dct test"
        stego = self.emb.embed_message(self.frame, msg)
        recovered = self.emb.extract_message(stego)
        assert recovered == msg

    def test_too_many_bits_raises(self):
        cap = self.emb.capacity_bits(self.frame)
        bits = np.zeros(cap + 1, dtype=np.uint8)
        with pytest.raises(ValueError):
            self.emb.embed(self.frame, bits)

    def test_invalid_quantisation_step(self):
        with pytest.raises(ValueError):
            DCTEmbedder(quantisation_step=0)

    def test_invalid_coeff_index(self):
        with pytest.raises(ValueError):
            DCTEmbedder(coeff_index=64)


# ---------------------------------------------------------------------------
# InterFrameEmbedder
# ---------------------------------------------------------------------------

class TestInterFrameEmbedder:
    def setup_method(self):
        self.emb = InterFrameEmbedder(delta=3, block_size=8, motion_threshold=5.0)
        # Use near-identical frames to guarantee static regions exist
        self.prev_frame, self.frame = make_static_frame_pair(64, 64, seed=30)

    def test_capacity_without_prev(self):
        cap = self.emb.capacity_bits(self.frame)
        # 64/8 × 64/8 = 64 total blocks
        assert cap == 64

    def test_capacity_with_prev(self):
        cap = self.emb.capacity_bits(self.frame, self.prev_frame)
        # Static frames → capacity should be > 0
        assert cap > 0

    def test_embed_returns_same_shape(self):
        bits = np.zeros(5, dtype=np.uint8)
        stego = self.emb.embed(self.frame, self.prev_frame, bits)
        assert stego.shape == self.frame.shape

    def test_embed_extract_roundtrip(self):
        bits = np.array([1, 0, 1, 0], dtype=np.uint8)
        stego = self.emb.embed(self.frame, self.prev_frame, bits)
        recovered = self.emb.extract(stego, self.prev_frame, len(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_embed_extract_message_roundtrip(self):
        msg = b"if"  # short message to fit in static blocks of a 64×64 frame
        stego = self.emb.embed_message(self.frame, self.prev_frame, msg)
        recovered = self.emb.extract_message(stego, self.prev_frame)
        assert recovered == msg

    def test_insufficient_capacity_raises(self):
        # Use two completely different (moving) frames → no static blocks
        frame_a = make_frame(64, 64, seed=0)
        frame_b = make_frame(64, 64, seed=99)
        bits = np.ones(100, dtype=np.uint8)
        emb = InterFrameEmbedder(delta=3, block_size=8, motion_threshold=0.0)
        # motion_threshold=0 → every block is "moving" → zero static capacity
        with pytest.raises(ValueError):
            emb.embed(frame_b, frame_a, bits)

    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            InterFrameEmbedder(delta=0)
