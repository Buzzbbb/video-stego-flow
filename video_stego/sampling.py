"""
sampling.py — 帧序列采样模块
Frame sequence sampling strategies.

Provides :class:`FrameSampler` which selects candidate frames from a decoded
video for steganographic embedding.
"""

from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np


class SamplingMode(str, Enum):
    """Supported frame sampling strategies."""

    UNIFORM = "uniform"          # Pick frames at a fixed interval
    RANDOM = "random"            # Pick frames uniformly at random
    SCENE_CHANGE = "scene_change"  # Pick frames near scene boundaries


class FrameSampler:
    """Select a subset of frames suitable for embedding.

    Parameters
    ----------
    mode:
        Sampling strategy (see :class:`SamplingMode`).
    seed:
        Optional random seed for reproducibility when using
        ``SamplingMode.RANDOM``.

    Examples
    --------
    >>> sampler = FrameSampler(mode="uniform")
    >>> indices = sampler.sample(total_frames=100, n=10)
    >>> print(indices)
    [ 0 10 20 30 40 50 60 70 80 90]
    """

    def __init__(
        self,
        mode: str | SamplingMode = SamplingMode.UNIFORM,
        seed: int | None = None,
    ) -> None:
        self.mode = SamplingMode(mode)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        total_frames: int,
        n: int,
        frames: List[np.ndarray] | None = None,
        scene_threshold: float = 30.0,
    ) -> np.ndarray:
        """Return an array of *n* selected frame indices.

        Parameters
        ----------
        total_frames:
            Total number of frames in the video sequence.
        n:
            Desired number of frames to select.
        frames:
            Optional list of BGR frames required for
            ``SamplingMode.SCENE_CHANGE``.
        scene_threshold:
            Mean absolute pixel difference threshold used to detect scene
            boundaries when ``mode == SamplingMode.SCENE_CHANGE``.

        Returns
        -------
        np.ndarray
            Sorted array of selected frame indices with dtype ``int64``.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if total_frames <= 0:
            raise ValueError("total_frames must be a positive integer")
        n = min(n, total_frames)

        if self.mode == SamplingMode.UNIFORM:
            return self._uniform(total_frames, n)
        if self.mode == SamplingMode.RANDOM:
            return self._random(total_frames, n)
        if self.mode == SamplingMode.SCENE_CHANGE:
            if frames is None:
                raise ValueError(
                    "frames must be supplied for SCENE_CHANGE sampling"
                )
            return self._scene_change(frames, n, scene_threshold)
        raise ValueError(f"Unknown sampling mode: {self.mode}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _uniform(self, total: int, n: int) -> np.ndarray:
        return np.linspace(0, total - 1, n, dtype=np.int64)

    def _random(self, total: int, n: int) -> np.ndarray:
        indices = self._rng.choice(total, size=n, replace=False)
        return np.sort(indices).astype(np.int64)

    def _scene_change(
        self,
        frames: List[np.ndarray],
        n: int,
        threshold: float,
    ) -> np.ndarray:
        """Detect scene boundaries by inter-frame mean absolute difference."""
        total = len(frames)
        if total < 2:
            return np.array([0], dtype=np.int64)[:n]

        scores = np.zeros(total, dtype=np.float64)
        for i in range(1, total):
            diff = np.mean(np.abs(frames[i].astype(np.float64) - frames[i - 1].astype(np.float64)))
            scores[i] = diff

        # Always include frame 0
        boundary_indices = np.where(scores >= threshold)[0]
        candidates = np.concatenate([[0], boundary_indices])
        candidates = np.unique(candidates)

        if len(candidates) >= n:
            # Return the n strongest boundaries
            ranked = candidates[np.argsort(scores[candidates])[::-1]][:n]
        else:
            # Fill remaining slots with uniform spacing
            remaining = n - len(candidates)
            uniform = np.linspace(0, total - 1, remaining + 2, dtype=np.int64)[1:-1]
            ranked = np.unique(np.concatenate([candidates, uniform]))[:n]

        return np.sort(ranked).astype(np.int64)
