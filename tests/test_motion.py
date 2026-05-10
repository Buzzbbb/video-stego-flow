"""Tests for video_stego.motion."""
import numpy as np
import pytest

from video_stego.motion import MotionAnalyzer
from tests.conftest import make_frame, make_static_frame_pair


class TestMotionAnalyzer:
    def setup_method(self):
        self.analyzer = MotionAnalyzer(block_size=8, motion_threshold=5.0)
        self.h, self.w = 64, 64
        self.frame_a = make_frame(self.h, self.w, seed=1)
        self.frame_b = make_frame(self.h, self.w, seed=2)

    # ------------------------------------------------------------------
    # frame_difference
    # ------------------------------------------------------------------

    def test_difference_shape(self):
        diff = self.analyzer.frame_difference(self.frame_a, self.frame_b)
        assert diff.shape == (self.h, self.w)

    def test_identical_frames_zero_diff(self):
        diff = self.analyzer.frame_difference(self.frame_a, self.frame_a)
        assert diff.max() == 0

    def test_mismatched_shapes_raises(self):
        frame_c = make_frame(32, 32)
        with pytest.raises(ValueError):
            self.analyzer.frame_difference(self.frame_a, frame_c)

    def test_difference_dtype(self):
        diff = self.analyzer.frame_difference(self.frame_a, self.frame_b)
        assert diff.dtype == np.uint8

    # ------------------------------------------------------------------
    # motion_mask
    # ------------------------------------------------------------------

    def test_mask_shape(self):
        mask = self.analyzer.motion_mask(self.frame_a, self.frame_b)
        assert mask.shape == (self.h, self.w)

    def test_mask_dtype(self):
        mask = self.analyzer.motion_mask(self.frame_a, self.frame_b)
        assert mask.dtype == bool

    def test_static_frames_mostly_false(self):
        static_a, static_b = make_static_frame_pair(self.h, self.w)
        analyzer = MotionAnalyzer(block_size=8, motion_threshold=10.0)
        mask = analyzer.motion_mask(static_a, static_b)
        # Very similar frames → most blocks should be classified as static
        assert mask.mean() < 0.5

    def test_random_frames_have_motion(self):
        mask = self.analyzer.motion_mask(self.frame_a, self.frame_b)
        # Two random frames → most blocks should be classified as moving
        assert mask.mean() > 0.5

    # ------------------------------------------------------------------
    # motion_score_map
    # ------------------------------------------------------------------

    def test_score_map_shape(self):
        score_map = self.analyzer.motion_score_map(self.frame_a, self.frame_b)
        assert score_map.shape == (self.h, self.w)

    def test_score_map_nonnegative(self):
        score_map = self.analyzer.motion_score_map(self.frame_a, self.frame_b)
        assert (score_map >= 0).all()

    def test_identical_frames_zero_score(self):
        score_map = self.analyzer.motion_score_map(self.frame_a, self.frame_a)
        assert score_map.max() == 0.0

    # ------------------------------------------------------------------
    # select_embedding_regions
    # ------------------------------------------------------------------

    def test_select_regions_length(self):
        frames = [self.frame_a, self.frame_b]
        regions = self.analyzer.select_embedding_regions(frames, [1], top_k_blocks=5)
        assert len(regions) == 1
        assert len(regions[0]) <= 5

    def test_select_regions_are_tuples(self):
        frames = [self.frame_a, self.frame_b]
        regions = self.analyzer.select_embedding_regions(frames, [1], top_k_blocks=3)
        for r, c in regions[0]:
            assert isinstance(r, int)
            assert isinstance(c, int)

    def test_block_size_validation(self):
        with pytest.raises(ValueError):
            MotionAnalyzer(block_size=0)
