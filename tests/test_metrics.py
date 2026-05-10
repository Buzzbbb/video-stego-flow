"""Tests for video_stego.metrics."""
import numpy as np
import pytest

from video_stego.metrics import (
    compute_psnr,
    compute_ssim,
    compute_ber,
    compute_video_psnr,
    compute_video_ssim,
)
from tests.conftest import make_frame


class TestComputePSNR:
    def setup_method(self):
        self.frame = make_frame(64, 64, seed=0)

    def test_identical_frames_inf(self):
        assert compute_psnr(self.frame, self.frame) == float("inf")

    def test_value_positive(self):
        other = make_frame(64, 64, seed=1)
        psnr = compute_psnr(self.frame, other)
        assert psnr > 0

    def test_shape_mismatch_raises(self):
        other = make_frame(32, 32, seed=1)
        with pytest.raises(ValueError):
            compute_psnr(self.frame, other)

    def test_known_mse(self):
        """PSNR = 10*log10(255^2 / MSE)."""
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 1, dtype=np.uint8)
        psnr = compute_psnr(a, b)
        expected = 10.0 * np.log10(255 ** 2 / 1.0)
        assert abs(psnr - expected) < 1e-6


class TestComputeSSIM:
    def setup_method(self):
        self.frame = make_frame(64, 64, seed=5)

    def test_identical_frames_one(self):
        val = compute_ssim(self.frame, self.frame)
        assert abs(val - 1.0) < 1e-6

    def test_value_in_range(self):
        other = make_frame(64, 64, seed=6)
        val = compute_ssim(self.frame, other)
        assert -1.0 <= val <= 1.0

    def test_shape_mismatch_raises(self):
        other = make_frame(32, 32, seed=6)
        with pytest.raises(ValueError):
            compute_ssim(self.frame, other)


class TestComputeBER:
    def test_identical_zero_ber(self):
        bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        assert compute_ber(bits, bits) == 0.0

    def test_all_different_one_ber(self):
        orig = np.array([0, 0, 0, 0], dtype=np.uint8)
        rec = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert compute_ber(orig, rec) == 1.0

    def test_half_different(self):
        orig = np.array([0, 0, 1, 1], dtype=np.uint8)
        rec = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert compute_ber(orig, rec) == 0.5

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_ber(np.zeros(4, dtype=np.uint8), np.zeros(5, dtype=np.uint8))

    def test_empty_returns_zero(self):
        assert compute_ber(np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)) == 0.0


class TestVideoMetrics:
    def setup_method(self):
        self.frames_a = [make_frame(64, 64, seed=i) for i in range(5)]
        self.frames_b = [make_frame(64, 64, seed=i + 10) for i in range(5)]

    def test_video_psnr_identical(self):
        psnr = compute_video_psnr(self.frames_a, self.frames_a)
        assert psnr == float("inf")

    def test_video_psnr_positive(self):
        psnr = compute_video_psnr(self.frames_a, self.frames_b)
        assert psnr > 0

    def test_video_psnr_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_video_psnr(self.frames_a, self.frames_b[:3])

    def test_video_ssim_identical(self):
        val = compute_video_ssim(self.frames_a, self.frames_a)
        assert abs(val - 1.0) < 1e-6

    def test_video_ssim_in_range(self):
        val = compute_video_ssim(self.frames_a, self.frames_b)
        assert -1.0 <= val <= 1.0

    def test_video_ssim_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_video_ssim(self.frames_a, self.frames_b[:2])

    def test_empty_psnr_nan(self):
        import math
        psnr = compute_video_psnr([], [])
        assert math.isnan(psnr)

    def test_empty_ssim_nan(self):
        import math
        ssim = compute_video_ssim([], [])
        assert math.isnan(ssim)
