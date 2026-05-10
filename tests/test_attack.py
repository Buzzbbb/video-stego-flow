"""Tests for video_stego.attack modules."""
import numpy as np
import pytest

from video_stego.attack import CropAttack, FrameDropAttack, CompressionAttack
from tests.conftest import make_frame


def _frames(n: int = 10, h: int = 64, w: int = 64) -> list:
    return [make_frame(h, w, seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# CropAttack
# ---------------------------------------------------------------------------

class TestCropAttack:
    def test_output_length_unchanged(self):
        frames = _frames(5)
        attacked = CropAttack(crop_pixels=8, resize_back=True).apply(frames)
        assert len(attacked) == len(frames)

    def test_output_shape_same_as_input_when_resize_back(self):
        frames = _frames(3)
        attacked = CropAttack(crop_pixels=8, resize_back=True).apply(frames)
        for orig, att in zip(frames, attacked):
            assert att.shape == orig.shape

    def test_output_shape_smaller_without_resize(self):
        frames = _frames(3, h=64, w=64)
        attacked = CropAttack(crop_pixels=8, resize_back=False).apply(frames)
        for att in attacked:
            assert att.shape[0] < 64 or att.shape[1] < 64

    def test_empty_input(self):
        assert CropAttack(crop_pixels=5).apply([]) == []

    def test_asymmetric_crop(self):
        frames = _frames(2, h=64, w=128)
        attacked = CropAttack(crop_pixels=(4, 4, 8, 8), resize_back=False).apply(frames)
        for att in attacked:
            assert att.shape[0] == 56  # 64 - 4 - 4
            assert att.shape[1] == 112  # 128 - 8 - 8


# ---------------------------------------------------------------------------
# FrameDropAttack
# ---------------------------------------------------------------------------

class TestFrameDropAttack:
    def test_output_same_length_with_fill(self):
        frames = _frames(20)
        attacked = FrameDropAttack(drop_rate=0.2, fill_gaps=True, seed=0).apply(frames)
        assert len(attacked) == len(frames)

    def test_output_shorter_without_fill(self):
        frames = _frames(20)
        attacked = FrameDropAttack(drop_rate=0.2, fill_gaps=False, seed=0).apply(frames)
        assert len(attacked) < len(frames)

    def test_empty_input(self):
        assert FrameDropAttack(drop_rate=0.1).apply([]) == []

    def test_invalid_drop_rate_zero(self):
        with pytest.raises(ValueError):
            FrameDropAttack(drop_rate=0.0)

    def test_invalid_drop_rate_one(self):
        with pytest.raises(ValueError):
            FrameDropAttack(drop_rate=1.0)

    def test_reproducible(self):
        frames = _frames(20)
        a1 = FrameDropAttack(drop_rate=0.2, seed=5).apply(frames)
        a2 = FrameDropAttack(drop_rate=0.2, seed=5).apply(frames)
        assert len(a1) == len(a2)


# ---------------------------------------------------------------------------
# CompressionAttack
# ---------------------------------------------------------------------------

class TestCompressionAttack:
    def test_output_length_unchanged(self):
        frames = _frames(5)
        attacked = CompressionAttack(quality=80).apply(frames)
        assert len(attacked) == len(frames)

    def test_output_shape_preserved(self):
        frames = _frames(3, h=64, w=64)
        attacked = CompressionAttack(quality=80).apply(frames)
        for orig, att in zip(frames, attacked):
            assert att.shape == orig.shape

    def test_compression_changes_pixels(self):
        frames = _frames(2, h=64, w=64)
        attacked = CompressionAttack(quality=10).apply(frames)
        # Low quality JPEG should change pixel values
        assert not np.array_equal(frames[0], attacked[0])

    def test_high_quality_close_to_original(self):
        from video_stego.metrics import compute_psnr
        frames = _frames(1, h=64, w=64)
        attacked = CompressionAttack(quality=99).apply(frames)
        psnr = compute_psnr(frames[0], attacked[0])
        assert psnr > 30.0

    def test_empty_input(self):
        assert CompressionAttack(quality=80).apply([]) == []

    def test_invalid_quality(self):
        with pytest.raises(ValueError):
            CompressionAttack(quality=0)
        with pytest.raises(ValueError):
            CompressionAttack(quality=101)
