"""Tests for video_stego.sampling."""
import numpy as np
import pytest

from video_stego.sampling import FrameSampler, SamplingMode
from tests.conftest import make_frame


class TestSamplingMode:
    def test_enum_values(self):
        assert SamplingMode.UNIFORM == "uniform"
        assert SamplingMode.RANDOM == "random"
        assert SamplingMode.SCENE_CHANGE == "scene_change"

    def test_from_string(self):
        assert SamplingMode("uniform") == SamplingMode.UNIFORM


class TestFrameSamplerUniform:
    def setup_method(self):
        self.sampler = FrameSampler(mode="uniform")

    def test_returns_n_indices(self):
        indices = self.sampler.sample(total_frames=100, n=10)
        assert len(indices) == 10

    def test_includes_first_and_last(self):
        indices = self.sampler.sample(total_frames=100, n=10)
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_indices_sorted(self):
        indices = self.sampler.sample(total_frames=50, n=5)
        assert list(indices) == sorted(indices)

    def test_n_clipped_to_total(self):
        indices = self.sampler.sample(total_frames=5, n=100)
        assert len(indices) == 5

    def test_single_frame(self):
        indices = self.sampler.sample(total_frames=1, n=1)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            self.sampler.sample(total_frames=10, n=0)

    def test_invalid_total(self):
        with pytest.raises(ValueError):
            self.sampler.sample(total_frames=0, n=5)


class TestFrameSamplerRandom:
    def setup_method(self):
        self.sampler = FrameSampler(mode="random", seed=42)

    def test_returns_n_indices(self):
        indices = self.sampler.sample(total_frames=100, n=10)
        assert len(indices) == 10

    def test_no_duplicates(self):
        indices = self.sampler.sample(total_frames=100, n=50)
        assert len(set(indices.tolist())) == 50

    def test_within_range(self):
        indices = self.sampler.sample(total_frames=50, n=10)
        assert indices.min() >= 0
        assert indices.max() < 50

    def test_sorted(self):
        indices = self.sampler.sample(total_frames=50, n=10)
        assert list(indices) == sorted(indices)

    def test_reproducible_with_seed(self):
        s1 = FrameSampler(mode="random", seed=7)
        s2 = FrameSampler(mode="random", seed=7)
        idx1 = s1.sample(total_frames=200, n=20)
        idx2 = s2.sample(total_frames=200, n=20)
        np.testing.assert_array_equal(idx1, idx2)


class TestFrameSamplerSceneChange:
    def _make_scene_frames(self):
        """Create a sequence with an obvious scene change at frame 5."""
        frames = [make_frame(32, 32, seed=0)] * 5
        frames += [make_frame(32, 32, seed=99)] * 5
        return frames

    def test_returns_up_to_n_indices(self):
        sampler = FrameSampler(mode="scene_change")
        frames = self._make_scene_frames()
        indices = sampler.sample(total_frames=len(frames), n=3, frames=frames)
        assert len(indices) <= 3
        assert len(indices) >= 1

    def test_includes_scene_boundary(self):
        sampler = FrameSampler(mode="scene_change", seed=0)
        frames = self._make_scene_frames()
        indices = sampler.sample(total_frames=len(frames), n=2, frames=frames)
        # Frame 5 is the scene boundary
        assert 5 in indices

    def test_requires_frames_argument(self):
        sampler = FrameSampler(mode="scene_change")
        with pytest.raises(ValueError):
            sampler.sample(total_frames=10, n=3)

    def test_single_frame_input(self):
        sampler = FrameSampler(mode="scene_change")
        frames = [make_frame(32, 32, seed=0)]
        indices = sampler.sample(total_frames=1, n=1, frames=frames)
        assert len(indices) == 1
